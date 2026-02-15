# System Architecture & Logic

## 1. User Journey & Application Workflow

### Step 1: Ingestion (Data Loader)
When you launch the app (`streamlit run app/Data_Loader.py`), you start at the **Data Ingestion** page.

**Supported Formats:**
1.  **Raw Text (`.txt`)**:
    - The system reads utf-8 text.
    - It uses **spaCy (`el_core_news_sm`)** to split the text into sentences.
    - Each sentence is tokenized and stored in the database.

2.  **PDF Documents (`.pdf`)**:
    - The system uses **PyPDF2** to extract the text layer.
    - The extracted text is processed by the same **spaCy pipeline** for robust sentence splitting.
    - *Note:* Scanned PDFs (images) are not supported; they must be text-selectable.

3.  **CoNLL Files (`.conll` / `.txt`)**:
    - Used for importing existing annotated data (BIO format).
    - The system reconstructs sentences and **imports existing annotations**, allowing you to resume work or correct old datasets.

### Step 2: Annotation (The "Annotator" Page)
Navigate to **Annotator** via the sidebar.
- **Selection**: Choose a source document.
- **Interface**: Sentences are presented one by one.
- **Action**: Highlight entities (e.g., "Supreme Court") and assign a label (e.g., "ORG").
- **Smart Suggestions**: If the model is active, suggestions appear in a different color. Click to accept.

### Step 3: Active Learning (Refinement)
After annotating a batch, use the **Active Learning** page.
- **Consistency**: Flags where you annotated "Maria" as a PERSON in sentence 1 but missed it in sentence 50.
- **Mistakes**: Bulk-delete errors (e.g., accidentally labeling a verb as an organism).
- **High Confidence**: Swiftly accept the model's "sure things" (>90% confidence).

### Step 4: Export
Finally, go to **Export** to download your clean dataset in standard CoNLL format, ready for training NLP models.

---

## 2. Core Components

### A. The Hybrid Predictor (Smart Review)

The heart of the system is the **Hybrid Prediction Engine** located in `src/core/hybrid_predictor.py`. It decides whether a text span is truly an entity based on two pillars:

1.  **Vector Context (Understanding "Where")**:
    - Uses the Transformer (RoBERTa) to generate embeddings for the sentence.
    - Compares the semantic context of the candidate phrase against accepted examples in the **Vector Memory**.
    - *Example:* Is "Olympus" used as a mountain or a company here?

2.  **Fuzzy Identity (Understanding "Who")**:
    - Uses Levenshtein distance (via `rapidfuzz`) to compare the spelling against known entities in the database.
    - *Example:* "Papadopoulos" vs "Papadopoulo".

**Decision Logic:**
The system calculates a weighted confidence score:
`Confidence = (W_Vector * Vector_Score) + (W_Fuzzy * Fuzzy_Score)`

- **High Confidence (>0.9)**: Auto-accepted relative to thresholds.
- **Medium Confidence (0.4 - 0.6)**: Flagged for manual review (Active Learning).
- **Low Confidence (<0.4)**: Ignored or marked as negative.

### B. The Memory Mechanism

The system "learns" as you annotate.
- **Vector Memory**: Every time you accept an annotation, its vector embedding is stored. Future similar contexts are recognized instantly.
- **Lexicons & Regex**: Rule-based agents (`src/agents/`) catch obvious patterns (Dates, Laws, Public Documents) before the model even runs.

---

## 3. Data Flow

1.  **Ingestion (`Data_Loader.py`)**:
    - Raw text or CONLL files are parsed into sentences.
    - Stored in SQLite (`data/annotations.db`) with metadata.

2.  **Annotation (`Annotator.py`)**:
    - User selects text spans.
    - Annotations are saved to DB with `status='manual'`.
    - Real-time feedback from the model suggests potential entities.

3.  **Refinement (`Active_Learning.py`)**:
    - **Consistency Check**: Finds where you annotated "X" in one place but missed it in another.
    - **Error Correction**: Batch delete, rename labels, or merge entities.

4.  **Export (`Export.py`)**:
    - Final validated data is exported to CONLL format for training future models.

---

## 4. Key Directories

- `src/agents/`: Logic for specific entity types (Regex/Lexicons).
- `src/core/`: The brain of the system (Predictor, Vector Memory).
- `src/database/`: SQLite management and schema handling.
- `src/models/`: Wrappers for HuggingFace Transformers.
- `src/utils/`: Parsers for CONLL and text processing.
- `app/pages/`: The Streamlit frontend interface.
