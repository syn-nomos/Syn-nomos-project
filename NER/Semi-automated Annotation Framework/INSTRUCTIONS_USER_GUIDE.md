# Application User Guide & Setup

## 1. Quick Start

### Prerequisites
- Python 3.9+ 
- A virtual environment (recommended)

### Installation
1. Navigate to the project folder:
   ```bash
   cd path/to/project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `torch` is installed with CUDA support if you have a GPU).*

### Running the Application
The application is built with Streamlit. To start it, run the entry point:

```bash
streamlit run app/Data_Loader.py
```

This will open the web interface in your browser (usually `http://localhost:8501`).

---

## 2. Setting Up Your Custom Model

Since the fine-tuned Transformer model (RoBERTa) is large, it is **not included** in this repository by default. You must add your own trained model for the "Smart Review" and "Auto-Annotation" features to work effectively.

### Step-by-Step Model Integration:

1. **Locate your model files**:
   You should have a folder containing:
   - `config.json`
   - `pytorch_model.bin` (or `model.safetensors`)
   - `tokenizer.json` / `vocab.json` etc.

2. **Create the directory**:
   Inside this project, create a folder named `roberta_local` inside the `models` directory.
   
   Structure should look like this:
   ```
   project_root/
   ├── app/
   ├── data/
   ├── models/
   │   └── roberta_local/       <-- PUT YOUR FILES HERE
   │       ├── config.json
   │       ├── pytorch_model.bin
   │       └── ...
   ├── src/
   ```

3. **Verification**:
   When you run the app and go to the **Smart Review** page, the system will automatically look for `models/roberta_local`. 
   - If found: It loads your fine-tuned model.
   - If NOT found: It falls back to the standard `xlm-roberta-base` (which won't have your specific entities trained).

---

## 3. Workflow Overview

1.  **Data Loader**: Import your raw text files or initial CONLL datasets.
2.  **Annotator**: Manually label data. The "Smart" features (pre-annotation) use the model if available.
3.  **Active Learning**: Review the model's suggestions, fix systematic mistakes, and check for consistency.
4.  **Smart Review**: A final pass using the Hybrid (Vector + Fuzzy) logic to catch subtle errors before export.
5.  **Export**: Generate the final .conll or .json dataset for training.
