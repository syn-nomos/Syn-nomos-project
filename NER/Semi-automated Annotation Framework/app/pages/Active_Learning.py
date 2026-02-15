import streamlit as st
import sys
import os
import time
import json
import sqlite3

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT of to_move
# app/pages -> app -> to_move
TO_MOVE_ROOT = os.path.dirname(os.path.dirname(APP_DIR))

if TO_MOVE_ROOT not in sys.path:
    sys.path.insert(0, TO_MOVE_ROOT)

# We also need the Workspace Root for data/knowledge_base
# to_move -> Semi-auto annotation model
WORKSPACE_ROOT = os.path.dirname(TO_MOVE_ROOT)

from src.database.db_manager import DBManager

st.set_page_config(page_title="Active Learning Studio", page_icon="🧠", layout="wide")

st.title("🧠 Active Learning & Consistency")
st.markdown("Use this tool to improve **Recall** (find missing entities), enforce **Consistency** (fix contradictions), and **Manage Rules** (Lexicons/Regex).")

# Initialize DB
# Check both possible locations for DB to be safe
db_path_candidate_1 = os.path.join(TO_MOVE_ROOT, "data", "db", "annotations.db")
db_path_candidate_2 = os.path.join(TO_MOVE_ROOT, "data", "annotations.db")

if os.path.exists(db_path_candidate_1):
    db_path = db_path_candidate_1
else:
    db_path = db_path_candidate_2

if "db" not in st.session_state:
    st.session_state.db = DBManager(db_path)
db = st.session_state.db

# TABS
tab_consistency, tab_high_conf, tab_low_conf, tab_mistakes, tab_lexicon = st.tabs([
    "🔍 Find Missing (Memory)", 
    "🚀 High Confidence (Suggestions)",
    "🤔 Low Confidence (Hard Cases)",
    "🗑️ Fix Mistakes", 
    "📚 Rules & Knowledge Base"
])

# --- TAB 1: CONSISTENCY (Batch Add) ---
with tab_consistency:
    st.subheader("1. Find Missing Entities by Memory")
    st.info("Identify phrases you've annotated elsewhere but missed here.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        min_len_filter = st.slider("Minimum Word Length (Avoid short/ambiguous words)", 3, 20, 5)
        if st.button("📊 Scan Frequent Entities"):
            with st.spinner("Analyzing DB statistics..."):
                top_ents = db.get_top_entities(limit=200)
                # Filter out short entities
                filtered_ents = [e for e in top_ents if len(e['text']) >= min_len_filter]
                st.session_state.top_ents = filtered_ents
    
    if "top_ents" in st.session_state:
        st.write(f"Found {len(st.session_state.top_ents)} frequent entities (Length >= {min_len_filter}).")
        
        # Selection
        selected_ent = st.selectbox("Select Entity to Check:", 
                                    st.session_state.top_ents, 
                                    format_func=lambda x: f"{x['text']} ({x['label']}) - Count: {x['count']}")
        
        if selected_ent:
            st.markdown(f"**Scanning for unannotated occurrences of:** `{selected_ent['text']}`")
            
            find_placeholder = st.empty()
            if find_placeholder.button("🔎 Find Missed Occurrences"):
                with st.spinner(f"Searching for '{selected_ent['text']}'..."):
                    matches = db.search_unannotated_matches(selected_ent['text'])
                    st.session_state.consistency_matches = matches
                    st.session_state.consistency_label = selected_ent['label']
                    st.rerun()

if "consistency_matches" in st.session_state:
        matches = st.session_state.consistency_matches
        label_candidate = st.session_state.consistency_label
        
        if matches:
            st.success(f"Found {len(matches)} Missed Occurrences found!")
            
            with st.form("consistency_batch_form"):
                 # Checkboxes
                 selected_indices = []
                 
                 st.markdown(f"**Review Mode:** Accepting as `{label_candidate}`")
                 
                 # Limit view to 50 for performance
                 view_limit = 50
                 for i, m in enumerate(matches[:view_limit]):
                     # Create snippet
                     full_text = m['full_text']
                     start = m['start']
                     end = m['end']
                     # Context window
                     ctx_start = max(0, start - 50)
                     ctx_end = min(len(full_text), end + 50)
                     
                     prefix = full_text[ctx_start:start]
                     match_txt = full_text[start:end]
                     suffix = full_text[end:ctx_end]
                     
                     # Render
                     cols = st.columns([0.5, 4, 1])
                     cols[0].text(f"#{m['sentence_id']}")
                     # Using markdown to bold match
                     # Safe escape for markdown special chars in context if needed, but keeping simple for now
                     display_text = f"...{prefix}**:{label_candidate}[{match_txt}]**{suffix}..."
                     cols[1].markdown(display_text)
                     
                     if cols[2].checkbox("Add", value=True, key=f"c_{i}"):
                         selected_indices.append(i)
                         
                 if len(matches) > view_limit:
                     st.warning(f"⚠️ Showing first {view_limit} of {len(matches)}. Process these first.")
                
                 if st.form_submit_button(f"✅ Add Selected ({len(selected_indices)}) Annotations"):
                     if not selected_indices:
                         st.warning("No annotations selected.")
                     else:
                         bar = st.progress(0)
                         count = 0
                         for idx in selected_indices:
                             m = matches[idx]
                             db.add_annotation(
                                 sentence_id=m['sentence_id'],
                                 label=label_candidate,
                                 start_offset=m['start'],
                                 end_offset=m['end'],
                                 text_content=m['match_text'],
                                 source='consistency_checker'
                             )
                             count += 1
                             bar.progress(count / len(selected_indices))
                         
                         st.success(f"Added {count} annotations!")
                         st.session_state.consistency_matches = [] # Clear results
                         time.sleep(1)
                         st.rerun()

# --- TAB 2: HIGH CONFIDENCE (>90%) ---
with tab_high_conf:
    st.subheader("2. High Confidence > 90% (False Negatives)")
    st.markdown("Entities the model is **sure** about, but you missed.")
    
    if st.button("🚀 Find High Confidence Suggestions"):
        try:
           # We need predictions stored in DB as candidates first
           # For now, let's look for pending annotations if they exist in DB
            candidates = db.get_high_confidence_candidates(min_conf=0.90, limit=50)
            if candidates:
                st.session_state.hi_conf_candidates = candidates
            else:
                st.info("No high confidence candidates found in DB. Run the model prediction pipeline first.")
        except AttributeError:
             st.error("DBManager method missing. Updating...")

    if "hi_conf_candidates" in st.session_state:
        cands = st.session_state.hi_conf_candidates
        st.write(f"Found {len(cands)} candidates.")
        
        with st.form("hi_conf_batch"):
            # Selectable dataframe
            selections = []
            for i, c in enumerate(cands):
                chk = st.checkbox(f"Accept: **{c['text']}** ({c['label']}) in: _{c['context']}_", value=True, key=f"hi_{i}")
                if chk:
                    selections.append(c)
            
            if st.form_submit_button(f"✅ Batch Accept Selected ({len(selections)})"):
                progress = st.progress(0)
                for i, s in enumerate(selections):
                    db.accept_annotation(s['id'])
                    progress.progress((i+1)/len(selections))
                st.success(f"Accepted {len(selections)} suggestions!")
                time.sleep(1)
                st.session_state.hi_conf_candidates = [] # Clear cache
                st.rerun()

# --- TAB 3: LOW CONFIDENCE (40-60%) ---
with tab_low_conf:
    st.subheader("3. Low Confidence (Boundary Cases)")
    st.markdown("These are the most valuable for training. **Accept** correct ones, **Reject** wrong ones.")
    
    if st.button("🤔 Find Low Confidence Samples"):
         try:
             unc = db.get_uncertain_candidates(min_conf=0.4, max_conf=0.6, limit=10)
             if unc:
                 st.session_state.unc_candidates = unc
                 st.success(f"Found {len(unc)} samples.")
             else:
                 st.info("No uncertain candidates found.")
         except AttributeError:
             st.error("DB Feature Missing.")

    if "unc_candidates" in st.session_state and st.session_state.unc_candidates:
        u_list = st.session_state.unc_candidates
        
        for i, u in enumerate(u_list):
            with st.container():
                # highlight clean text
                clean_ctx = u.get('context', '').replace(u['text'], f"**{u['text']}**")
                
                cols = st.columns([4, 1, 1])
                cols[0].markdown(f"Label: `{u['label']}` | Conf: **{u['confidence']:.2f}**\n> ...{clean_ctx}...")
                
                if cols[1].button("✅ Accept", key=f"u_ok_{u['id']}"):
                    db.accept_annotation(u['id'])
                    st.toast(f"Accepted {u['text']}")
                    
                if cols[2].button("❌ Reject", key=f"u_no_{u['id']}"):
                    db.reject_annotation(u['id'])
                    st.toast(f"Rejected {u['text']}")
                st.divider()


# --- TAB 4: MISTAKES & BATCH DELETE ---
with tab_mistakes:
    st.subheader("4. Fix Systematic Mistakes")
    st.warning("⚠️ Corrections applied here affect the entire database.")
    
    col_type = st.radio("Action Type:", ["Delete Wrong Annotation", "Rename Label", "Merge Duplicates"], horizontal=True)

    if col_type == "Delete Wrong Annotation":
        delete_text = st.text_input("Text to delete (Exact Match):", placeholder="e.g. Ένα ρήμα που μπήκε λάθος")
        if delete_text:
             # Quick Stats
             count_bad = db.count_annotations(text=delete_text)
             st.error(f"Found {count_bad} occurrences of '{delete_text}' across all labels.")
             
             if st.button(f"🗑️ Delete ALL {count_bad} occurrences"):
                 deleted = db.delete_annotations_by_text(delete_text)
                 st.success(f"Deleted {deleted} annotations.")
                 
                 # Suggest updating regex/lexicon immediately
                 st.markdown("#### 💡 Prevention")
                 st.info(f"Do you want to add '{delete_text}' to a Negative Regex list to prevent it appearing again?")
                 # Link to regex editor could go here
    
    elif col_type == "Rename Label":
         st.markdown("Batch change label **from X to Y** for a specific text string.")
         
         target_text = st.text_input("Text to rename:", placeholder="e.g. Αθήνα")
         
         c1, c2 = st.columns(2)
         with c1:
             old_lbl = st.selectbox("Old Label:", ["PERSON", "ORG", "GPE", "FACILITY", "DATE", "LEG_REFS", "PUBLIC_DOCS"], index=0)
         with c2:
             new_lbl = st.selectbox("New Label:", ["PERSON", "ORG", "GPE", "FACILITY", "DATE", "LEG_REFS", "PUBLIC_DOCS"], index=2)
             
         if target_text:
             # Check potential impact
             # We reuse count_annotations just to see if text exists
             cnt = db.count_annotations(text=target_text)
             if cnt > 0:
                 st.info(f"Target Text '{target_text}' appears {cnt} times (total across all labels).")
                 
                 if st.button(f"🔄 Rename '{old_lbl}' → '{new_lbl}'"):
                      updated = db.rename_annotations(target_text, old_lbl, new_lbl)
                      if updated > 0:
                          st.success(f"Successfully renamed {updated} annotations!")
                          time.sleep(1)
                          st.rerun()
                      else:
                          st.warning(f"No annotations found for '{target_text}' with label '{old_lbl}'.")
             else:
                 st.warning(f"Text '{target_text}' not found in any annotation.")

# --- TAB 5: LEXICONS & REGEX ---
with tab_lexicon:
    st.subheader("3. Knowledge Base Management")
    st.markdown("Edit the underlying **Lexicons (.json)** and **Regex Patterns (.txt)** that drive the rule-based pre-annotation.")
    
    kb_root = os.path.join(WORKSPACE_ROOT, "data", "knowledge_base")
    
    if os.path.exists(kb_root):
        # 1. Select Entity Type
        entity_types = [d for d in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, d))]
        entity_type = st.selectbox("Select Entity Type:", [""] + entity_types)
        
        if entity_type:
            entity_dir = os.path.join(kb_root, entity_type)
            
            # 2. Select File
            files = [f for f in os.listdir(entity_dir) if f.endswith('.txt') or f.endswith('.json')]
            selected_file = st.selectbox("Select File to Edit:", [""] + files)
            
            if selected_file:
                file_path = os.path.join(entity_dir, selected_file)
                
                # 3. Read & Edit
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        current_content = f.read()
                    
                    st.markdown(f"**Editing:** `{os.path.join(entity_type, selected_file)}`")
                    new_content = st.text_area("File Content:", current_content, height=400)
                    
                    if st.button("💾 Save Changes"):
                        # Validation for JSON
                        valid = True
                        if selected_file.endswith(".json"):
                            try:
                                json.loads(new_content)
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON Format: {e}")
                                valid = False
                        
                        if valid:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(new_content)
                            st.success(f"Successfully saved {selected_file}!")
                            time.sleep(1)
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.error(f"Knowledge Base path not found at: {kb_root}")



