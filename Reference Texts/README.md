# Greek Legislation Scraper & Dataset Builder

A robust Python scraper that builds a reference dataset of Greek legal documents. It downloads content from **raptarchis.gov.gr**, cleans the HTML/legal formatting, and enriches the metadata by fetching official Government Gazette (FEK) details and thematic topics from the **National Printing House (ET.gr)** APIs.

> **Output:**
> - `scraped_docs/` → Directory containing cleaned plain text files (`.txt`).
> - `dataset_metadata.csv` → A comprehensive index with titles, word counts, FEK links, and topics.

---

## 🚀 Features

- **Automated Scraping:** Iterates through documents by numeric ID range (`START_ID` → `END_ID`).
- **Smart Text Extraction:** Recursively cleans text from complex legal structures:
  - `arrangements`
  - `parts` → `chapters` → `articles` → `paragraphs`
  - Top-level `articles`
- **Metadata Enrichment:** Automatically cross-references documents with the National Printing House to retrieve:
  - **Direct PDF Links:** The official signed PDF of the law.
  - **Thematic Tags:** Official subject classifications (e.g., "Health", "Administration").
- **Polite Scraping:** Implements random delays (`2–5s`) and User-Agent headers to respect server load.

---

## 📊 Dataset Metadata (CSV Columns)

The generated `dataset_metadata.csv` contains the following fields for each document:

| Column | Description |
| :--- | :--- |
| **id** | Unique numeric identifier from the source API. |
| **filename** | Name of the saved file (e.g., `A184_102.txt`). |
| **word_count** | Total word count of the cleaned text. |
| **title** | Short, cleaned title of the Act. |
| **full_title** | Official full title (includes Law Number, Dates, etc.). |
| **article_count** | Number of articles detected in the document. |
| **article_titles** | List of all article headers found (separated by `|`). |
| **fek** | Government Gazette Issue string (e.g., `Α' 184`). |
| **fek_search_url** | Generated search link for the ET.gr portal. |
| **fek_url** | **Direct link** to the official PDF file. |
| **topics** | Thematic categories/subjects (e.g., `ΥΓΕΙΑ | ΔΙΟΙΚΗΣΗ`). |

---

## 🛠️ Requirements & Installation

- **Python 3.9+** is recommended.

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
