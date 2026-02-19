# EUR-Lex Greek Summaries + Legal Texts (Scrape → Clean Dataset)

This folder contains two Python scripts used to build the **EUR-Lex v6** dataset:

1. **Scraper**: downloads Greek EUR-Lex **summaries** and the corresponding **full legal texts** (by CELEX) into a structured folder tree (`eudocs/`).
2. **Cleaner / Dataset builder**: validates each leaf folder, removes noisy parts (TOC, footnotes, notes, annexes), applies length filters, and exports a final dataset (`eudocsCleaned/`) where **each `.txt` file is a single line** (easy to store as JSON and feed to LexRank/LLMs).
---

## 📂 Output Dataset Structure

After the full pipeline, each **valid leaf folder** in `eudocsCleaned/` contains exactly:

- `summary.html`
- `summary.txt`  (cleaned summary, **single line**)
- `<CELEX>.html` (one or more legal texts)
- `<CELEX>.txt`  (cleaned legal text, **single line**)

A leaf folder is kept only if it ends up as a strict **2x2** folder:
- 2 HTML files + 2 TXT files (summary + legal text),

---

# 1) Scraper Script

## What it does
- Starts from: `https://eur-lex.europa.eu/browse/summaries.html`
- Collects the **top-level topics** and their **subtopics**
- For each subtopic:
  - downloads the **Greek summary (HTML + TXT)**
  - navigates to **Document information**
  - extracts the CELEX id(s)
  - downloads each corresponding **Greek legal text** in:
    - HTML
    - TXT
    - PDF 

## 📝 Optional Topic Selection (`topics.txt`)

By default, the scraper processes **all** available topics. 
To restrict the scraper to specific categories, alter the file named `topics.txt` in the root directory and add/remove the exact topic names you want (one per line).

**Available Topics Reference:**

| | | |
| :--- | :--- | :--- |
| Agriculture | Audiovisual_and_media | Budget |
| Competition | Consumers | Culture |
| Customs | Development | Digital_single_market |
| Economic_and_monetary_affairs | Education,_training,_youth,_sport | Employment_and_social_policy |
| Energy | Enlargement | Enterprise |
| Environment_and_climate_change | External_relations | External_trade |
| Food_safety | Foreign_and_security_policy | Fraud_and_corruption |
| Human_rights | Humanitarian_Aid_and_Civil_Protection | Institutional_affairs |
| Internal_market | Justice,_freedom_and_security | Oceans_and_fisheries |
| Public_health | Regional_policy | Research_and_innovation |
| Taxation | Transport | |

**Example `topics.txt` content:**
```text
Energy
Transport
Environment_and_climate_change
```

# 2) Cleaner Script

## What it does

- Reads the raw data from `eudocs/`
- Applies quality filters:
  - Removes broken documents
  - Skips folders where the number of summaries and legal texts do not match
  - Discards invalid or incomplete outputs
- Cleans text for summarization:
  - Removes TOC blocks, footnotes, and note sections
  - Flattens tables into plain text
  - Cuts Annex sections and their content
  - Converts each `.txt` file into a **single-line format**
- Exports the valid, cleaned dataset to `eudocsCleaned/`

## 🛠️ Requirements & Installation

- **Python 3.9+** is recommended.

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
