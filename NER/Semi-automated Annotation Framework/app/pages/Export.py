import streamlit as st
import sqlite3
import os
import sys
import pandas as pd
from datetime import datetime

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(ROOT_DIR)

from src.database.db_manager import DBManager

st.set_page_config(page_title="Data Export", page_icon="💾")

# Initialize Session State
if "db" not in st.session_state:
    st.warning("Please load a database first.")
    st.stop()

db = st.session_state.db

st.title("💾 Export Data")
st.markdown("Export your annotated data to CoNLL format for training.")

# --- Export Settings ---
with st.form("export_form"):
    st.subheader("Configuration")
    
    c1, c2 = st.columns(2)
    output_name = c1.text_input("Output Filename", value=f"export_ver_{datetime.now().strftime('%Y%m%d')}.conll")
    output_dir = c2.text_input("Output Directory", value=os.path.join(ROOT_DIR, "dataset"))
    
    st.markdown("---")
    st.subheader("Filters")
    
    # Splits
    all_splits = ["train", "dev", "test", "pending"]
    selected_splits = st.multiselect("Select Splits to Export", all_splits, default=["train"])
    
    # Source Filter
    export_mode = st.radio("Export Strategy", [
        "Verified Only (Strict) - Only manual/verified annotations",
        "Hybrid (Standard) - Manual + Imported + Auto (All active annotations)",
        "Review Mode - Only export sentences marked as 'annotated'"
    ])
    
    submitted = st.form_submit_button("🚀 Generate Export")

if submitted:
    full_path = os.path.join(output_dir, output_name)
    
    # 1. Fetch Sentences
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Filter by split
    placeholders = ','.join(['?'] * len(selected_splits))
    
    # Base Query
    base_query = f"SELECT id, text, dataset_split, status FROM sentences WHERE dataset_split IN ({placeholders})"
    
    # Add status filter if needed
    if "Review Mode" in export_mode:
        base_query += " AND status = 'annotated'"
        
    cursor.execute(base_query, selected_splits)
    rows = cursor.fetchall()
    
    if not rows:
        st.error("No sentences found matching criteria.")
    else:
        st.info(f"Processing {len(rows)} sentences...")
        
        # 2. Process & Write
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                count_exported = 0
                
                # Progress bar
                prog_bar = st.progress(0)
                
                for i, (sid, text, split, status) in enumerate(rows):
                    # Get Annotations
                    cursor.execute("SELECT label, start_offset, end_offset, text, source FROM annotations WHERE sentence_id = ?", (sid,))
                    anns = []
                    for r in cursor.fetchall():
                        anns.append({
                            "label": r[0],
                            "start": r[1],
                            "end": r[2],
                            "text": r[3],
                            "source": r[4]
                        })
                    
                    # Filter Annotations based on Mode
                    valid_anns = []
                    for a in anns:
                        if "Verified Only" in export_mode:
                            if a['source'] == 'manual':
                                valid_anns.append(a)
                        else:
                            # Hybrid: Allow all current annotations in DB
                            valid_anns.append(a)
                            
                    # Tokenize & Tag (Basic whitespace tokenizer to match standard CoNLL)
                    # For robust CoNLL, we need to respect the tokenization.
                    # As a heuristic, we split by space and check alignment.
                    
                    tokens = text.split()
                    
                    # Helper to find tag for token
                    # This is a bit tricky if tokens don't align perfectly with characters.
                    # We track character index.
                    
                    cursor_pos = 0
                    
                    for token in tokens:
                        # Find token start in text (allow skipping spaces)
                        try:
                            start_idx = text.index(token, cursor_pos)
                        except ValueError:
                            # Fallback if text has different whitespace than split
                            # Just increment cursor
                            cursor_pos += len(token) + 1
                            f.write(f"{token} O\n")
                            continue
                            
                        end_idx = start_idx + len(token)
                        
                        # Check overlaps with valid_anns
                        label = "O"
                        for a in valid_anns:
                            # Strict containment: token is inside annotation
                            if start_idx >= a['start'] and end_idx <= a['end']:
                                # Determine B- or I-
                                if start_idx == a['start']:
                                    label = f"B-{a['label']}"
                                else:
                                    label = f"I-{a['label']}"
                                break
                                
                        f.write(f"{token} {label}\n")
                        cursor_pos = end_idx
                    
                    f.write("\n") # Sentence break
                    count_exported += 1
                    
                    if i % 10 == 0:
                        prog_bar.progress((i + 1) / len(rows))
                
                prog_bar.progress(1.0)
                
            st.success(f"✅ Export Complete! Saved to: `{full_path}`")
            st.metric("Sentences Exported", count_exported)
            
        except Exception as e:
            st.error(f"Export failed: {e}")
            
    conn.close()
