import streamlit as st
import pandas as pd
import time
from src.database.db_manager import DBManager
from src.models.roberta_ner import RobertaNER
from src.core.hybrid_predictor import HybridPredictor
from src.core.vector_memory import VectorMemory
from src.core.attention_extractor import AttentionExtractor
from src.core.augmented_embeddings import AugmentedEmbeddingBuilder
from src.core.memory_manager import MemoryManager

# --- CACHED RESOURCES ---
def get_db():
    return DBManager()

@st.cache_resource
def load_prediction_system():
    """Loops once to load heavy models."""
    print("⏳ Loading Smart Review System...")
    
    # 1. Models
    # Use specific local model as requested
    import os
    # Default location: "models/roberta_local" relative to app root
    # Or fallback to base model if not found
    
    # Try to find app root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(current_dir)) # Up from app/pages to root
    local_model_path = os.path.join(app_root, "models", "roberta_local")
    
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        # Fallback to standard huggingface path or inform user
        st.warning(f"⚠️ Local model not found at {local_model_path}. Using 'xlm-roberta-base' (untrained).")
        model_path = "xlm-roberta-base"
        
    roberta = RobertaNER(model_path) 
    
    # 2. Components
    mem_mgr = MemoryManager(db_path="data/annotations.db")
    # Warmup memory
    mem_mgr.warmup()
    
    # 3. Predictor
    # We reconstruct the stack needed for HybridPredictor
    # Note: HybridPredictor constructor signature might need checking
    # __init__(self, roberta_model, vector_memory, attention_extractor, embedding_builder, ...)
    
    # Use MemoryManager's internal vector memory if available, or create new adapter
    # For now, we assume MemoryManager WRAPS the VectorMemory functionality we need
    # But HybridPredictor expects specific objects. 
    
    # Re-instantiating components strictly for the Predictor
    att_extractor = AttentionExtractor()
    emb_builder = AugmentedEmbeddingBuilder()
    
    # We need the LOW LEVEL vector memory for the predictor, 
    # but MemoryManager is the new high level one. 
    # Let's inspect HybridPredictor again to see what it expects.
    # It takes `vector_memory`.
    
    # Hack: We use the MemoryManager as the interface since it has `find_similar`
    predictor = HybridPredictor(
        roberta_model=roberta,
        vector_memory=mem_mgr, # Passing Manager as Memory (Duck Typing: has find_similar)
        attention_extractor=att_extractor,
        embedding_builder=emb_builder
    )
    
    return predictor

# --- PAGE LOGIC ---
st.set_page_config(page_title="Smart Review", layout="wide")

st.title("🧠 Smart Prediction Review")
st.markdown("Review system suggestions based on **Vector Context** and **Fuzzy Identity**.")

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

predictor = load_prediction_system()
db = get_db()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Controls")
    target_suggestions = st.slider("Target Suggestions", 5, 50, 10, help="Stop scanning when we find this many hits.")
    max_scan_limit = st.number_input("Max Search Limit", value=500, step=50, help="Give up if we scan this many sentences without finding enough hits.")
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.6)
    
    if st.button("🚀 Scan for Suggestions", type="primary"):
        with st.spinner(f"Hunting for the next {target_suggestions} suggestions..."):
            
            new_preds = []
            scanned_count = 0
            chunk_size = 50 # Process in small chunks for memory
            
            # Progress Setup
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cursor = db.conn.cursor()
            
            # Keep looping until we find enough suggestions OR hit the hard limit
            while len(new_preds) < target_suggestions and scanned_count < max_scan_limit:
                
                # Fetch next chunk of Pending sentences
                # We use OFFSET to skip what we've already scanned in this session (conceptually), 
                # but actually we just query 'pending' using separate offset tracking if needed?
                # BETTER: Fetch 'pending' limit chunk_size OFFSET scanned_count
                
                cursor.execute(
                    "SELECT id, text FROM sentences WHERE status='pending' LIMIT ? OFFSET ?", 
                    (chunk_size, scanned_count)
                )
                rows = cursor.fetchall()
                
                if not rows:
                    break # No more data in DB
                
                for row in rows:
                    sent_id = row['id']
                    text = row['text']
                    
                    # Rule: Skip very short
                    if len(text) < 10: 
                        continue
                    
                    # Predict
                    try:
                        # READ SETTINGS FROM SESSION SATE
                        config = {
                            'w_vector': st.session_state.get('settings_vector_weight', 0.5),
                            'w_fuzzy': st.session_state.get('settings_fuzzy_weight', 0.5),
                            'thresh_vector': st.session_state.get('settings_vector_threshold', 0.70),
                            'thresh_fuzzy': st.session_state.get('settings_fuzzy_threshold', 80)
                        }
                        candidates = predictor.predict(text, config=config)
                    except Exception as e:
                        print(f"Error predicting {sent_id}: {e}")
                        candidates = []
                    
                    # Filter for High Confidence OR Memory Match
                    valid_cands = []
                    for cand in candidates:
                        # Logic: Must have some memory support OR very high confidence
                        is_memory_backed = (cand.get('source') == 'Hybrid (Memory)')
                        is_high_conf = (cand.get('confidence', 0) > min_conf)
                        
                        if is_memory_backed or is_high_conf:
                            valid_cands.append(cand)
                    
                    if valid_cands:
                        new_preds.append({
                            'sentence_id': sent_id,
                            'text': text,
                            'entities': valid_cands
                        })
                    
                    # Check break condition inside loop slightly redundant but safer
                    if len(new_preds) >= target_suggestions:
                        break
                
                scanned_count += len(rows)
                progress = min(scanned_count / max_scan_limit, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Scanned {scanned_count} sentences... Found {len(new_preds)} matches.")

            st.session_state.predictions = new_preds
            st.session_state.current_index = 0
            
            if not new_preds:
                st.warning(f"Scanned {scanned_count} sentences but found no matches matching criteria.")
            else:
                st.rerun()

# --- MAIN VIEW ---

if not st.session_state.predictions:
    st.info("No predictions loaded. Click **Scan for Suggestions** on the sidebar.")
else:
    # Navigation
    idx = st.session_state.get('current_index', 0)
    if idx >= len(st.session_state.predictions):
        st.success("🎉 All suggestions reviewed!")
        if st.button("Clear and Scan Again"):
            st.session_state.predictions = []
            st.rerun()
        st.stop()
        
    current_item = st.session_state.predictions[idx]
    
    # Progress
    st.progress((idx) / len(st.session_state.predictions))
    st.caption(f"Reviewing {idx + 1} of {len(st.session_state.predictions)}")
    
    # --- CARD UI ---
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Sentence")
            # Highlight entities in text
            annotated_text = current_item['text']
            # Simple highlighter logic (could be improved)
            # Sort entities by start index reversed to replace safely
            sorted_ents = sorted(current_item['entities'], key=lambda x: x['start'], reverse=True)
            
            display_spans = []
            last_idx = len(annotated_text)
            
            final_md = annotated_text
            for ent in sorted_ents:
                lbl = ent['label']
                txt = ent['text']
                conf = ent.get('confidence', 0)
                
                # Color code
                color = "orange" if ent.get('source') == 'Hybrid (Memory)' else "blue"
                
                replacement = f":{color}[**{txt}**] `({lbl})`"
                
                # Replace in string (careful with overlaps, assuming none for now due to NMS)
                s, e = ent['start'], ent['end']
                
                # Simple replacement by index (could break if multiple replaces shift indices)
                # But we iterate reversed, so indices should be stable?
                # Yes: High indices change first.
                final_md = final_md[:s] + replacement + final_md[e:]
                
            st.markdown(f"> {final_md}")
            
        with col2:
            st.subheader("Action")
            
            # Individual Entity Review? Or Whole Sentence?
            # Let's do Whole Sentence for speed
            
            st.write("**Detected Entities:**")
            for ent in current_item['entities']:
                source = ent.get('source', 'Model')
                icon = "🧠" if "Memory" in source else "🤖"
                st.markdown(f"{icon} **{ent['text']}**")
                st.caption(f"{source} • {ent.get('confidence',0):.2f}")
                if 'explanation' in ent:
                     st.caption(f"_{ent['explanation']}_")
            
            st.divider()
            
            c1, c2 = st.columns(2)
            if c1.button("✅ Accept", use_container_width=True):
                # Save to DB
                cursor = db.conn.cursor()
                
                # 1. Update Sentence Status
                cursor.execute("UPDATE sentences SET status='completed' WHERE id=?", (current_item['sentence_id'],))
                
                # 2. Add Annotations
                for ent in current_item['entities']:
                    # Build SQL
                    cursor.execute('''
                        INSERT INTO annotations (sentence_id, text_span, label, start_char, end_char, vector, is_golden)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                    ''', (
                        current_item['sentence_id'],
                        ent['text'],
                        ent['label'],
                        ent['start'],
                        ent['end'],
                        ent.get('vector_blob') # Needs to be passed if available
                    ))
                
                db.conn.commit()
                st.toast(f"Saved {len(current_item['entities'])} entities!")
                
                # Next
                st.session_state.current_index += 1
                st.rerun()
                
            if c2.button("❌ Skip", use_container_width=True):
                 # Mark Ignored? Or just skip in UI?
                 # Let's mark as 'skipped' so we don't see it again immediately
                 cursor = db.conn.cursor()
                 cursor.execute("UPDATE sentences SET status='skipped' WHERE id=?", (current_item['sentence_id'],))
                 db.conn.commit()
                 
                 st.session_state.current_index += 1
                 st.rerun()
