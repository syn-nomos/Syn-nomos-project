# Vocabulary Generator Tool

A unified Python application for generating semantic descriptions of concepts from various vocabularies (EuroLex, MIMIC-III, HellasVoc) using Large Language Models (LLMs) via the OpenRouter API.

## 🚀 Features

- **Multi-Dataset Support**: Works with Legal (EuroLex, HellasVoc) and Clinical (MIMIC-III) datasets.
- **Flexible Models**: Switch easily between OpenAI (GPT-4o), Mistral, and other OpenRouter models.
- **Hierarchy Awareness**: Automatically includes concept hierarchy (breadcrumbs) in the prompt for better context (can be disabled).
- **Resilient**: Includes error handling, retries, and progress tracking.

---

## 🛠️ Setup

### 1. Prerequisites
- Python 3.8+
- An [OpenRouter](https://openrouter.ai/) API Key.

### 2. Environment Variables
Create a `.env` file in the project root directory (one level up from this script or in the same folder):

```env
OPENROUTER_API_KEY=your_key_here_sk-or-v1-...
```

### 3. Data Placement
Ensure your source JSON files are located in the `apps/data/` folder (or adjust `DATASET_PATHS` in the script).

---

## 💻 Usage

Run the script from the terminal:

```bash
python apps/vocabulary_generator.py --dataset [NAME] --model [MODEL] [OPTIONS]
```

### Arguments

| Argument | Options | Description |
| :--- | :--- | :--- |
| `--dataset` | `eurolex`, `mimic`, `hellasvoc` | The specific vocabulary dataset to process. |
| `--model` | `gpt-4o-mini`, `mistral-large`, `ministral-8b`... | The LLM to use for generation. |
| `--overwrite` | (Flag) | If present, re-generates files that already exist. |
| `--no-hierarchy` | (Flag) | If present, **disables** the hierarchy context (it is ON by default). |

### Examples

**Generate MIMIC descriptions using Mistral Large:**
```bash
python apps/vocabulary_generator.py --dataset mimic --model mistral-large
```

**Generate HellasVoc descriptions without hierarchy context:**
```bash
python apps/vocabulary_generator.py --dataset hellasvoc --model ministral-8b --no-hierarchy
```

---

## ⚙️ Customization

### Adding New Models
The script uses **OpenRouter**, so you can use almost any modern LLM.
1. Visit [OpenRouter Models](https://openrouter.ai/models) to find the model ID (e.g., `anthropic/claude-3-opus`).
2. Open `vocabulary_generator.py`.
3. Update the `MODELS` dictionary at the top of the file:

```python
MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "my-new-model": "provider/model-name-id", # <-- Add this line
    # ...
}
```

### Modifying Prompts
To change how the AI describes the concepts (e.g., change the language, tone, or length):
1. Open `vocabulary_generator.py`.
2. Locate the `create_prompt(self, concept: Dict)` method inside the `UnifiedGenerator` class.
3. Edit the f-strings for the relevant `dataset_type`.

**Example:**
```python
if self.dataset_type == "eurolex":
    return f"""Your new custom prompt here for {title}..."""
```

---

## 📂 Data Structure Requirements

If you want to use your own custom data files, ensure they match the expected JSON structure for the loader logic.

### 1. EuroLex (`eurolex`)
Expects a JSON file (list or dict) where items contain:
```json
{
  "id": "12345",
  "title": "Concept Name",
  "eurlex": {
     "hierarchical_paths": [
        { "full_path_string": "Law > Civil Law > Concept Name" }
     ]
  },
  "eurlex_source": true
}
```

### 2. MIMIC-III (`mimic`)
Expects a flat JSON list:
```json
[
  {
    "code": "401.9",
    "title": "Unspecified essential hypertension",
    "path": "Diseases of the circulatory system > Hypertensive disease"
  }
]
```

### 3. HellasVoc (`hellasvoc`)
Expects a **nested/hierarchical** JSON structure (Tree), which the script traverses recursively:
```json
{
  "id": "1",
  "name": "Root Concept",
  "children": [
    {
      "id": "1.1",
      "name": "Child Concept",
      "children": []
    }
  ]
}
```
*Note: The script automatically calculates the hierarchy path for HellasVoc by traversing this tree.*
