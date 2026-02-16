import streamlit as st
import sys
import os
import yaml

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
# Insert at 0 to ensure we load 'src' from 'to_move' not the parent directory
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.database.db_manager import DBManager
from src.core.data_ingestion import DataIngestion

# Page Config
st.set_page_config(page_title="LegalNER Loader", page_icon="📂", layout="wide")

# Initialize Session State
# SELF-HEALING: Check if DB instance is stale or missing methods
if "db" not in st.session_state or not hasattr(st.session_state.db, 'get_stats'):
    db_path = os.path.join(ROOT_DIR, "data", "db", "annotations.db")
    
    # Force reload of the module to ensure we get the latest class definition
    if "src.database.db_manager" in sys.modules:
        import importlib
        import src.database.db_manager
        importlib.reload(src.database.db_manager)
        from src.database.db_manager import DBManager # Re-import after reload

    if "src.core.data_ingestion" in sys.modules:
        import importlib
        import src.core.data_ingestion
        importlib.reload(src.core.data_ingestion)
        from src.core.data_ingestion import DataIngestion 
        
    st.session_state.db = DBManager(db_path)

# Initialize Ingestion Module
# Reload ingestion anyway to be safe on every run during dev
if "src.core.data_ingestion" in sys.modules:
    import importlib
    import src.core.data_ingestion
    importlib.reload(src.core.data_ingestion)
    from src.core.data_ingestion import DataIngestion 

ingester = DataIngestion(st.session_state.db)


st.title("📂 Data Ingestion")
st.markdown("---")
st.info("Start by loading data. You can import an existing CoNLL dataset or create a new one from raw text/PDF.")

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📄 Import Text / PDF (New)", "📊 Import CoNLL (Existing)", "🗄️ Load Existing Database"])

with tab1:
    st.header("Create New Dataset")
    st.write("Upload a raw document to start a new annotation project.")
    
    source_type = st.selectbox("Source Format", ["Raw Text (.txt)", "PDF Document (.pdf)"], key="fmt_new")
    uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf'], key="uploader_new")
    
    st.markdown("### 🎯 Target Database")
    target_mode_new = st.radio("Where should the data go?", ["Create New Database", "Append to Active Database"], horizontal=True, index=0, key="tgt_mode_new")
    
    if target_mode_new == "Create New Database":
        default_name = uploaded_file.name.rsplit('.', 1)[0] + "_db" if uploaded_file else "new_project_db"
        new_db_name_new = st.text_input("Project Name (Database)", value=default_name, key="new_name_new")
    else:
        st.info(f"Active Active DB:_ {os.path.basename(st.session_state.db.db_path)}")

    if uploaded_file and st.button("🚀 Process & Import", key="btn_process"):
        
        # Helper to switch DB if needed
        if target_mode_new == "Create New Database":
             safe_name = "".join(x for x in new_db_name_new if (x.isalnum() or x in "._- "))
             db_file = f"{safe_name}.db"
             new_path = os.path.join(ROOT_DIR, "data", "db", db_file)
             
             st.session_state.db = DBManager(new_path)
             ingester = DataIngestion(st.session_state.db) # Re-init ingester with new DB
             st.toast(f"Switched to new database: {db_file}")

        with st.spinner("Processing document with Spacy..."):
            count = ingester.process_file_upload(uploaded_file, source_type)
            if count > 0:
                st.success(f"Successfully imported {count} sentences from {uploaded_file.name}!")
                st.balloons()
            else:
                st.warning("No valid sentences found or processing failed.")

with tab2:
    st.header("Import Existing Dataset")
    st.write("Load a CoNLL file (BIO format) to review or correct existing annotations.")
    
    uploaded_conll = st.file_uploader("Upload CoNLL File", type=['conll', 'txt'], key="uploader_conll")
    
    st.markdown("### 🎯 Target Database")
    target_mode_conll = st.radio("Where should the data go?", ["Create New Database", "Append to Active Database"], horizontal=True, index=0, key="tgt_mode_conll")
    
    if target_mode_conll == "Create New Database":
        default_name = uploaded_conll.name.rsplit('.', 1)[0] + "_db" if uploaded_conll else "imported_dataset_db"
        new_db_name_conll = st.text_input("Project Name (Database)", value=default_name, key="new_name_conll")
    else:
        st.info(f"Active DB: {os.path.basename(st.session_state.db.db_path)}")
    
    if uploaded_conll and st.button("📥 Parse & Load into DB", key="btn_parse"):
        
        if target_mode_conll == "Create New Database":
             safe_name = "".join(x for x in new_db_name_conll if (x.isalnum() or x in "._- "))
             db_file = f"{safe_name}.db"
             new_path = os.path.join(ROOT_DIR, "data", "db", db_file)
             
             st.session_state.db = DBManager(new_path)
             ingester = DataIngestion(st.session_state.db)
             st.toast(f"Created & Switched to: {db_file}")
         
        with st.spinner("Parsing CoNLL structure... v1.1"): # Changed text to force refresh
            # Read directly as string since helper expects string content for internal call
            content = uploaded_conll.getvalue().decode("utf-8")
            count = ingester.process_conll(content, uploaded_conll.name)
            
            if count > 0:
                st.success(f"Successfully imported {count} sentences into {os.path.basename(st.session_state.db.db_path)}!")
                st.info("Go to the 'Annotator' page to view and check them.")
                st.balloons()
            else:
                st.error("Failed to parse sentences. Check file format.")

with tab3:
    st.header("Switch Database")
    st.write("Select a previously created SQLite (.db) file to resume work.")
    
    # List available DBs in data/db
    db_dir = os.path.join(ROOT_DIR, "data", "db")
    os.makedirs(db_dir, exist_ok=True)
    available_dbs = [f for f in os.listdir(db_dir) if f.endswith(".db")]
    
    selected_db = st.selectbox("Select Database", available_dbs, index=0 if available_dbs else None)
    
    if st.button("🔄 Load Selected Database", key="btn_load_db"):
        if selected_db:
             new_db_path = os.path.join(db_dir, selected_db)
             st.session_state.db = DBManager(new_db_path)
             st.success(f"Switched to database: {selected_db}")
             st.rerun()
        else:
            st.warning("No database selected.")

if "db" in st.session_state:
    try:
        stats = st.session_state.db.get_stats()
        st.caption(f"Current Database Status: {stats.get('total_sentences', 'N/A')} sentences stored.")
    except AttributeError:
        st.caption("Database loaded (Old Version - stats unavailable)")
else:
    st.info("Please select and load a database.")
