import sqlite3  # Ensure sqlite3 is imported for the helper function
import streamlit as st
import os
import sys
import time

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
# Prioritize local src
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.database.db_manager import DBManager
# LLM Fixer Imports
try:
    from src.judges.llm_client import LLMJudge
    from src.agents.specific_boundary_fixer import TypeSpecificBoundaryFixer
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Model Loader
from src.models.roberta_ner import RobertaNER

st.set_page_config(page_title="Annotation Studio", page_icon="📝", layout="wide")

@st.cache_resource
def load_roberta():
    # Attempt to find a robust model path
    candidates = [
        r"E:\ALGORITHMS\Semi-auto annotation model\to_move\src\agents\roberta_model",
        "src/agents/GreekLegalRoBERTa_New", # Newly trained model (Priority 1)
        "E:/ALGORITHMS/Semi-auto annotation model/src/agents/GreekLegalRoBERTa_New",
        "src/agents/GreekLegalRoBERTa_v3",
        "E:/ALGORITHMS/Semi-auto annotation model/src/agents/GreekLegalRoBERTa_v3",
        "xlm-roberta-base" # Fallback
    ]
    for path in candidates:
        if "xlm" in path or os.path.exists(path):
            try:
                print(f"Loading NER from: {path}")
                return RobertaNER(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    return None

ner_model = load_roberta()

# Initialize LLM Fixer if enabled
if HAS_LLM and "fixer" not in st.session_state:
    try:
        # Check for key in sidebar later or config
        judge = LLMJudge()
        if judge.api_key:
            st.session_state.fixer = TypeSpecificBoundaryFixer(judge)
            st.session_state.judge = judge # Save pure judge for Phase 2
        else:
            st.session_state.fixer = None
            st.session_state.judge = None
    except Exception:
        st.session_state.fixer = None
        st.session_state.judge = None

# --- AI COUNCIL / HYBRID PREDICTOR SETUP ---
from src.core.hybrid_predictor import HybridPredictor
from src.agents.leg_refs_regex_agent import LegRefsRegexAgent
from src.agents.gpe_regex_agent import GpeRegexAgent
from src.agents.org_regex_agent import OrgRegexAgent
from src.agents.date_regex_agent import DateRegexAgent
from src.agents.public_docs_regex_agent import PublicDocsRegexAgent
from src.core.augmented_embeddings import AugmentedEmbeddingBuilder

@st.cache_resource
def load_hybrid_system(_roberta_instance): # Underscore prevents hashing issues
    if not _roberta_instance: return None
    # Agents
    agents = [
        LegRefsRegexAgent(), 
        GpeRegexAgent(), 
        OrgRegexAgent(),
        DateRegexAgent(),
        PublicDocsRegexAgent()
    ]
    # Initialize Hybrid Brain
    return HybridPredictor(
        roberta_model=_roberta_instance,
        vector_memory=None, 
        attention_extractor=None,
        embedding_builder=AugmentedEmbeddingBuilder(),
        rule_agents=agents
    )

hybrid_brain = load_hybrid_system(ner_model)

# Initialize Session State
if "db" not in st.session_state:
    st.warning("Please load or select a database from the Data Loader page first.")
    st.stop()

db = st.session_state.db

# --- INIT MEMORY MANAGER ---
from src.core.memory_manager import MemoryManager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager(db, ner_model)
    st.toast("🧠 Memory Manager Initialized", icon="🧠")

memory_manager = st.session_state.memory_manager

# SELF-HEALING: Check if MemoryManager is stale (missing new methods)
if not hasattr(memory_manager, 'rebuild_all_vectors'):
    st.toast("🔄 Refactoring Memory Intelligence...", icon="🛠️")
    import importlib
    import src.core.memory_manager
    importlib.reload(src.core.memory_manager)
    from src.core.memory_manager import MemoryManager
    st.session_state.memory_manager = MemoryManager(db, ner_model)
    memory_manager = st.session_state.memory_manager

# SELF-HEALING: Check if DB instance is stale (missing new methods)
if not hasattr(db, 'add_annotation'):
    st.warning("🔄 Upgrading Database Connector... (One-time reload)")
    # Re-initialize to pick up new code changes
    try:
        from src.database.db_manager import DBManager
        import importlib
        import src.database.db_manager
        importlib.reload(src.database.db_manager) # Force reload of module
        
        st.session_state.db = src.database.db_manager.DBManager(db.db_path)
        db = st.session_state.db
        st.rerun() # Updated from experimental_rerun
    except Exception as e:
        st.error(f"Failed to reload DB: {e}")

st.title("📝 Annotation Studio")
st.caption(f"Connected to: {os.path.basename(db.db_path)}")

with st.expander("ℹ️ How to use"):
    st.markdown("""
    - **Annotate Pending**: Focuses only on sentences that have not been annotated yet (skips existing ones).
    - **Review/Correct**: Allows browsing ALL sentences (including already annotated ones) to make corrections.
    - Use the sidebar filters to narrow down by dataset split (train/dev/test).
    """)

# Sidebar Controls
st.sidebar.header("Filter Settings")
mode = st.sidebar.radio("Work Mode", ["Annotate Pending", "Review/Correct"], index=1)

target_split = st.sidebar.multiselect("Dataset Split", ["train", "dev", "test", "pending"], default=["train", "dev", "test", "pending"])

st.sidebar.markdown("---")
st.sidebar.header("🧠 Memory & Intelligence")
if st.sidebar.button("🧠 Build/Rebuild Memory Index"):
    with st.spinner("Rebuilding vectors... This clears old vectors to ensure consistency."):
        # Update model ref just in case
        memory_manager.roberta = ner_model 
        pb = st.sidebar.progress(0)
        res = memory_manager.rebuild_all_vectors(progress_callback=lambda x: pb.progress(x))
        st.sidebar.success(res)
        
do_propagate = st.sidebar.checkbox("📡 Auto-Propagate Decisions", value=False, help="If checked, Accepting/Deleting an entity will automatically update identical pending entities.")

st.sidebar.markdown("---")
st.sidebar.header("Visual Filters")
st.sidebar.caption("Toggle visibility by source:")
show_manual = st.sidebar.checkbox("✅ Verified (Human)", value=True)
show_import = st.sidebar.checkbox("📄 Imported (CoNLL)", value=True)
show_auto = st.sidebar.checkbox("🤖 Suggestions (AI)", value=True)

# Helper to get sentences based on mode
@st.cache_data(ttl=60) # Cache for performance, invalidate often
def get_sentence_ids(db_path, mode, splits):
    """Get list of IDs matching criteria."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if mode == "Annotate Pending":
        query = f"SELECT id FROM sentences WHERE status = 'pending' AND dataset_split IN ({','.join(['?']*len(splits))}) ORDER BY id ASC"
        cursor.execute(query, splits)
    else:
        # Review Mode - Get all or annotated
        query = f"SELECT id FROM sentences WHERE dataset_split IN ({','.join(['?']*len(splits))}) ORDER BY id ASC"
        cursor.execute(query, splits)
        
    ids = [r[0] for r in cursor.fetchall()]
    conn.close()
    return ids

# Reload IDs if filters change
ids_pool = get_sentence_ids(db.db_path, mode, target_split)
total_count = len(ids_pool)

# Fetch Data logic updated for navigation
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# --- TABS LAYOUT ---
tab_annotate, tab_auto = st.tabs(["✍️ Workbench (Manual)", "🤖 Auto-Annotator (Council)"])

with tab_auto:
    st.markdown("### 🚀 Mass Annotation Pipeline")
    st.info("This tool scans the filtered sentences and applies AI annotations automatically. It skips sentences that already have annotations.")
    
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    
    with c1:
        st.markdown("**1. Strategy**")
        strategy = st.radio("Phase Selection", 
                           ["Phase 1: Local (Fast)", "Phase 2: Council (Cloud)", "Hybrid"], 
                           index=0, horizontal=True)
    
    with c2:
        st.markdown("**2. Batch**")
        batch_limit = st.number_input("Process max items:", 10, 5000, 50, help="How many NEW items to find and tag.")
    
    with c3:
        st.markdown("**3. Speed**")
        delay = st.number_input("Delay (s)", 0.0, 5.0, 0.1)
    
    with c4:
        st.markdown("**4. Conf**")
        min_conf = st.slider("Threshold", 0.0, 1.0, 0.5)
        
    if st.button("✨ START BATCH", disabled=(total_count==0), type="primary"):
        pbar = st.progress(0)
        logs = st.empty()
        st.toast("Pipeline started...")
        
        # LOGIC: Iterate through the filtered pool
        # But only process if NO annotations exist
        
        processed_count = 0
        skipped_count = 0
        
        # We walk through the pool until we hit the batch_limit of PROCESSED items
        # or we run out of items.
        
        current_batch = []
        
        for pid in ids_pool:
            if processed_count >= batch_limit:
                break
            
            s = db.get_sentence(pid)
            
            # CHECK: Resume Logic
            # If sentence already has annotations (from manual or previous run), skip it.
            if s.get('annotations') and len(s['annotations']) > 0:
                skipped_count += 1
                continue
            
            # Add to processing queue
            current_batch.append(s)
            
            # If queue full, process
            if len(current_batch) >= 10: # Mini-batch for model
                 # RUN MODEL
                 if "Phase 1" in strategy or "Hybrid" in strategy:
                    if hybrid_brain:
                        for item in current_batch:
                            preds = hybrid_brain.predict(item['text'])
                            for p in preds:
                                db.add_annotation(
                                    sentence_id=item['id'],
                                    label=p['label'],
                                    start_offset=p['start'],
                                    end_offset=p['end'],
                                    text_content=p['text'],
                                    source='auto'
                                )
                 
                 # Update counters
                 processed_count += len(current_batch)
                 current_batch = []
                 logs.text(f"Tagged {processed_count} sentences... (Skipped {skipped_count})")
                 pbar.progress(min(processed_count / batch_limit, 1.0))
                 time.sleep(delay)

        # Process remaining
        if current_batch:
             if "Phase 1" in strategy or "Hybrid" in strategy:
                if hybrid_brain:
                    for item in current_batch:
                        preds = hybrid_brain.predict(item['text'])
                        for p in preds:
                            db.add_annotation(
                                sentence_id=item['id'],
                                label=p['label'],
                                start_offset=p['start'],
                                end_offset=p['end'],
                                text_content=p['text'],
                                source='auto'
                            )
             processed_count += len(current_batch)
             pbar.progress(1.0)
        
        st.success(f"Batch Complete! ✅ Tagged: {processed_count}, Skipped (Already Done): {skipped_count}")
        time.sleep(1.5)
        st.rerun()

with tab_annotate:
    # --- MAIN SINGLE ITEM UI ---
    if total_count == 0:
        st.info("No sentences found matching criteria.")
    else:
        # Navigation
        # Ensure index in bounds
        if st.session_state.current_idx >= total_count:
            st.session_state.current_idx = 0
            
        current_id = ids_pool[st.session_state.current_idx]
        
        # Load Sentence
        sentence = db.get_sentence(current_id)



if total_count == 0:
    st.warning("No sentences found matching criteria.")
    st.stop()

# Progress Indicator
curr = st.session_state.current_idx + 1
st.progress(curr / total_count, text=f"Sentence {curr} of {total_count} ({mode})")

# Navigation Controls
col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])

with col_nav1:
    if st.button("⬅️ Previous"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)

with col_nav3:
    if st.button("Next ➡️"):
        st.session_state.current_idx = min(total_count - 1, st.session_state.current_idx + 1)

with col_nav2:
    # Direct Jump
    curr = st.session_state.current_idx + 1
    new_idx = st.number_input(f"Sentence {curr} / {total_count}", min_value=1, max_value=total_count, value=curr, label_visibility="collapsed")
    if new_idx != curr:
        st.session_state.current_idx = new_idx - 1
        st.rerun()

# Get Current ID
current_id = ids_pool[st.session_state.current_idx]
current_sent = db.get_sentence(current_id)

# --- EDITOR UI ---
st.markdown(f"### Sentence #{current_sent['id']}")

# Status Badge
status_color = "green" if current_sent['status'] == 'annotated' else "orange"
st.markdown(f"**Status:** :{status_color}[{current_sent['status'].upper()}]")

# --- RENDERER: Annotated Sentence (HTML Version) ---
def render_inline_annotations(text, annotations):
    """
    Renders text with inline annotations using HTML for rich styling.
    """
    if not annotations:
        return f"><span style='font-size:1.1em; line-height: 1.6;'>{text}</span>"

    # Entities Colors (Hex)
    TYPE_COLORS = {
        "LEG-REFS": "#8A2BE2", # BlueViolet
        "ORG": "#1E90FF",      # DodgerBlue
        "PERSON": "#FF4500",   # OrangeRed
        "GPE": "#DAA520",      # GoldenRod
        "DATE": "#2E8B57",     # SeaGreen
        "FACILITY": "#20B2AA", # LightSeaGreen
        "LOCATION": "#CD853F", # Peru
        "PUBLIC_DOCS": "#708090" # SlateGray
    }

    # Status Backgrounds (Start with rgba for transparency)
    # Verified (Green), Imported (Blue), Auto (Violet)
    BG_COLORS = {
        "verified": "rgba(40, 167, 69, 0.15)",  # Light Green
        "imported": "rgba(0, 123, 255, 0.15)",  # Light Blue
        "auto":     "rgba(111, 66, 193, 0.15)"  # Light Violet
    }
    BORDER_COLORS = {
        "verified": "#28a745",
        "imported": "#007bff",
        "auto":     "#6f42c1"
    }

    # 1. Sort by start index (reverse)
    sorted_anns = sorted(annotations, key=lambda x: x['start_offset'], reverse=True)
    
    # 2. Reconstruct
    processed_text = text
    
    for ann in sorted_anns:
        s, e = ann['start_offset'], ann['end_offset']
        lbl = ann['label']
        src = ann.get('source', 'unknown')
        
        # Determine Status Style
        is_verified = ann.get('is_correct') == 1 or ann.get('is_accepted') == 1 or src == 'manual'
        
        if is_verified:
            status = "verified"
        elif "imported" in src:
            status = "imported"
        else:
            status = "auto"
            
        bg = BG_COLORS[status]
        border = BORDER_COLORS[status]
        
        # Determine Type Color
        type_hex = TYPE_COLORS.get(lbl, "#555") # Default Gray
        
        # Safe slice content
        entity_text = text[s:e]
        
        # HTML Snippet
        # Outer Span: The Entity Highlight (Background)
        # Inner Span: The Label Badge (Solid Color)
        html_snippet = (
            f"<span style='background-color: {bg}; border-bottom: 2px solid {border}; "
            f"padding: 2px 4px; border-radius: 4px; margin: 0 2px; white-space: nowrap;' title='Source: {src}'>"
            f"<strong style='color: #222;'>{entity_text}</strong> "
            f"<span style='background-color: {type_hex}; color: white; padding: 1px 4px; "
            f"border-radius: 3px; font-size: 0.75em; font-weight: bold; vertical-align: middle;'>{lbl}</span>"
            f"</span>"
        )
        
        # Replace
        if e <= len(text):
            processed_text = processed_text[:s] + html_snippet + processed_text[e:]
            
    return f"<div style='font-family: sans-serif; font-size: 1.1em; line-height: 1.8; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #ddd;'>{processed_text}</div>"

# Use get_annotations_for_sentence instead of get_annotations
try:
    annotated_view = render_inline_annotations(current_sent['text'], db.get_annotations_for_sentence(current_sent['id']))
except AttributeError:
    # Fallback in case method name is different in some legacy DBManager version
    try:
        annotated_view = render_inline_annotations(current_sent['text'], db.get_annotations(current_sent['id']))
    except:
        annotated_view = render_inline_annotations(current_sent['text'], [])

st.markdown(annotated_view, unsafe_allow_html=True)
st.caption("Legend: 🟩 Verified (Human) | 🟦 Imported (CoNLL) | 🟪 Suggestion (AI)")

# --- Global Sentence Actions ---
# Find New Entities Button
c_global1, c_global2 = st.columns([0.3, 0.7])
with c_global1:
    if st.button("🕵️ Auto-Detect New Entities", help="scan sentence for missed entities"):
        if not ner_model:
            st.error("NER Model not loaded.")
        else:
            with st.spinner("Scanning..."):
                preds = ner_model.predict(current_sent['text'])
                # Convert to annotation format
                new_found = []
                
                # Get existing ranges to avoid duplicates
                existing_ranges = []
                annotations = db.get_annotations_for_sentence(current_sent['id']) # Refresh
                for a in annotations:
                    existing_ranges.append((a['start_offset'], a['end_offset']))
                
                added_count = 0
                for p in preds:
                    # Check overlap
                    is_overlap = False
                    p_start, p_end = p['start'], p['end']
                    for (e_start, e_end) in existing_ranges:
                        # Simple overlap logic
                        if (p_start < e_end) and (p_end > e_start):
                            is_overlap = True
                            break
                    
                    if not is_overlap:
                        new_ann = {
                            "label": p['label'],
                            "text": p['text'],
                            "start_offset": p['start'],
                            "end_offset": p['end'],
                            "source": "model_scan", # distinct source
                            "is_correct": 0
                        }
                        annotations.append(new_ann)
                        new_found.append(f"{p['text']} ({p['label']})")
                        added_count += 1
                
                if added_count > 0:
                    db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                    st.success(f"Found {added_count} new entities: {', '.join(new_found)}")
                    time.sleep(1.5) # Give user time to see
                    st.rerun()
                else:
                    st.toast("No new entities found.", icon="🕵️")

# Load existing annotations
annotations = db.get_annotations_for_sentence(current_sent['id'])

# --- Manual Add Section --
with st.expander("➕ Add Missing Entity (Manual)"):
    m_col1, m_col2 = st.columns([2, 1])
    
    # 1. Text Input (search)
    manual_text = m_col1.text_input("Entity Text", placeholder="Paste text here...", key="manual_text_input")
    manual_label = m_col2.selectbox("Label", ["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC_DOCS"], key="manual_label")

    if manual_text:
        # 2. Find Occurrences Logic
        import re
        # Escape special chars to treat as literal, but ignore case
        try:
            matches = [m for m in re.finditer(re.escape(manual_text), current_sent['text'], re.IGNORECASE)]
        except:
            matches = []

        if not matches:
            st.warning("⚠️ Text not found in sentence.")
        
        elif len(matches) == 1:
            # Single Match - Simple Add
            m = matches[0]
            st.info(f"Target found at position {m.start()}-{m.end()}")
            if st.button("Add Entity"):
                new_ann = {
                    "label": manual_label,
                    "text": current_sent['text'][m.start():m.end()], # Capture exact case from text
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "source": "manual",
                    "is_correct": 1
                }
                annotations.append(new_ann)
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                st.success("Added!")
                st.rerun()
                
        else:
            # Multiple Matches - Disambiguation UI
            st.info(f"Start found **{len(matches)} occurrences**. Please select which one:")
            
            # Create readable options
            options = {}
            for i, m in enumerate(matches):
                start, end = m.start(), m.end()
                # Create a small context snippet
                snippet_start = max(0, start - 20)
                snippet_end = min(len(current_sent['text']), end + 20)
                
                prefix = ("..." if snippet_start > 0 else "") + current_sent['text'][snippet_start:start]
                target = current_sent['text'][start:end]
                suffix = current_sent['text'][end:snippet_end] + ("..." if snippet_end < len(current_sent['text']) else "")
                
                # HTML formatted label for radio/selectbox wouldn't render HTML, so we use plain text logic or specialized
                options[f"Occurrence {i+1}"] = {
                    "match": m,
                    "display": f"#{i+1}: {prefix}[{target}]{suffix}"
                }
            
            selected_occ = st.radio("Select Occurrence:", list(options.keys()), 
                                    format_func=lambda x: options[x]['display'])
            
            if st.button("Add Selected Occurrence"):
                m = options[selected_occ]['match']
                new_ann = {
                    "label": manual_label,
                    "text": current_sent['text'][m.start():m.end()],
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "source": "manual",
                    "is_correct": 1
                }
                annotations.append(new_ann)
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                st.success("Added!")
                st.rerun()

st.markdown("#### Annotations")

if annotations:
    # Entity Color Map
    ENTITY_COLORS = {
        "LEG-REFS": "#8A2BE2", # BlueViolet
        "ORG": "#1E90FF",      # DodgerBlue
        "PERSON": "#FF4500",   # OrangeRed
        "GPE": "#DAA520",      # GoldenRod
        "DATE": "#2E8B57",     # SeaGreen
        "FACILITY": "#20B2AA", # LightSeaGreen
        "LOCATION": "#DAA520", # Same as GPE usually
        "PUBLIC_DOCS": "#4682B4" # SteelBlue
    }

    for ann in annotations:
        source = ann.get('source', 'unknown')
        
        # Determine source type for filtering
        is_manual = source == 'manual'
        is_import = 'conll' in source or 'import' in source
        is_auto = 'model' in source or 'active' in source
        
        # Apply Sidebar Filters
        if is_manual and not show_manual: continue
        if is_import and not show_import: continue
        if is_auto and not show_auto: continue
        if (not is_manual and not is_import and not is_auto) and not show_import: continue # Fallback for unknown

        label = ann['label']
        text = ann['text']
        color = ENTITY_COLORS.get(label, "#6c757d") # Default Grey

        # Icon based on source
        if source == 'manual': icon = "✅" 
        elif 'conll' in source or 'import' in source: icon = "📄"
        elif 'model' in source or 'active' in source: icon = "🤖"
        else: icon = "❓"

        # HTML Badge Rendering
        html = f"""
        <div style="margin-bottom: 0px;">
            <span style="font-size: 1.2em; margin-right: 6px; vertical-align: middle;">{icon}</span>
            <span style="background-color: {color}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 500; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                {text}
                <span style="background-color: rgba(255,255,255,0.2); margin-left: 8px; padding: 1px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; font-family: sans-serif;">
                    {label}
                </span>
            </span>
            <small style="color: #888; margin-left: 8px;">{source}</small>
        </div>
        """
        
        # Layout: Badge | Actions
        # Adjusted columns for better alignment
        c1, c2 = st.columns([0.7, 0.3])
        
        with c1:
            st.markdown(html, unsafe_allow_html=True)
            
        with c2:
            # Action Buttons: [Edit] [Auto-Fix] [Accept?] [Delete]
            # Using smaller column widths inside c2 for the buttons
            cols = st.columns([1, 1, 1, 1])
            b_edit = cols[0]
            b_fix = cols[1]
            b_accept = cols[2]
            b_del = cols[3]
            
            # 1. Edit (Always available)
            if b_edit.button("✏️", key=f"edit_{ann.get('id', text)}", help="Manual Edit"):
                 st.session_state.editing_annotation = ann
                 st.rerun()

            # 2. Auto-Fix (LLM)
            if HAS_LLM and st.session_state.fixer:
                if b_fix.button("✨", key=f"autofix_{ann.get('id', text)}", help="Auto-Fix Boundaries (LLM)"):
                    with st.spinner("🤖"):
                        corrected = st.session_state.fixer.fix_boundary(current_sent['text'], text, label)
                        if corrected and corrected != text:
                            ann['text'] = corrected
                            ann['source'] = 'manual' # Auto-fixed implies we trust it now, or keep separate? Let's treat as manual fix.
                            db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                            st.toast(f"Fixed: {text} -> {corrected}", icon="✨")
                            st.rerun()
                        else:
                            st.toast("No change suggested by LLM.", icon="🤷")

            # 3. Accept (Only if not already manual)
            if not is_manual:
                if b_accept.button("✅", key=f"accept_{ann.get('id', text)}", help="Verify (Mark as Correct)"):
                    ann['source'] = 'manual'
                    ann['is_correct'] = 1
                    db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                    
                    # Propagation Logic
                    if do_propagate:
                        n_up = memory_manager.propagate(text, label, 'accept')
                        if n_up > 0: st.toast(f"Propagated ACCEPT to {n_up} other items.", icon="📡")
                    
                    st.rerun()
            
            # 4. Delete (Always available)
            if b_del.button("🗑️", key=f"del_{ann.get('id', text)}", help="Delete Annotation"):
                new_anns = [a for a in annotations if a != ann]
                db.save_annotations(current_sent['id'], new_anns, mark_complete=False)
                
                # Propagation Logic
                if do_propagate:
                    n_up = memory_manager.propagate(text, label, 'reject')
                    if n_up > 0: st.toast(f"Propagated REJECT to {n_up} other items.", icon="📡")
                
                st.rerun()

    # --- Editing Modal/Expander ---
    if "editing_annotation" in st.session_state and st.session_state.editing_annotation:
        edit_ann = st.session_state.editing_annotation
        with st.form(key="edit_form"):
            st.markdown(f"**Editing:** `{edit_ann['text']}`")
            new_label = st.selectbox("Label", list(ENTITY_COLORS.keys()), index=list(ENTITY_COLORS.keys()).index(edit_ann['label']) if edit_ann['label'] in ENTITY_COLORS else 0)
            
            # Simple text fix for now (Advanced boundary fixing requires token interaction)
            new_text = st.text_input("Text (Manual Fix)", value=edit_ann['text'])
            
            # --- Auto-Fix Button ---
            if HAS_LLM and st.session_state.fixer:
                if st.form_submit_button("✨ Auto-Fix Boundaries (LLM)"):
                    with st.spinner("🤖 Asking LLM to fix boundaries..."):
                        corrected = st.session_state.fixer.fix_boundary(current_sent['text'], new_text, new_label)
                        if corrected and corrected != new_text:
                            # We can't update the text_input directly without rerun or session state trickery
                            # Simplest: Update the edit_ann and rerun to refresh the modal state
                            st.session_state.editing_annotation['text'] = corrected
                            st.rerun()
                        else:
                            st.warning("LLM proposed no change.")

            c_edit1, c_edit2 = st.columns(2)
            if c_edit1.form_submit_button("💾 Save Changes"):
                # Update the annotation in the list
                # This is tricky without unique IDs if there are duplicates, but we try best match
                for i, a in enumerate(annotations):
                    if a == edit_ann:
                        annotations[i]['label'] = new_label
                        annotations[i]['text'] = new_text
                        annotations[i]['source'] = 'manual' # Edited implies manual
                        annotations[i]['is_correct'] = 1
                        break
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                del st.session_state.editing_annotation
                st.rerun()
                
            if c_edit2.form_submit_button("❌ Cancel"):
                 del st.session_state.editing_annotation
                 st.rerun()

else:
    st.write("_No annotations yet._")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Confirm / Mark as Verified"):
        # Logic: Convert all current annotations to 'manual' source and mark sentence as annotated
        updated_annotations = []
        for ann in annotations:
            ann_copy = ann.copy()
            ann_copy['source'] = 'manual' # Upgrade to manual/verified
            updated_annotations.append(ann_copy)
        
        # Save back to DB
        db.save_annotations(current_sent['id'], updated_annotations, mark_complete=True)
        st.success("✅ Verified & Saved! Moving to next...")
        
        # Move to next
        st.session_state.current_idx = min(total_count - 1, st.session_state.current_idx + 1)
        st.rerun()

with col2:
    if st.button("✏️ Edit"):
        st.info("Editor logic pending implementation.")
