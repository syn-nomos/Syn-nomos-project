import requests
import re
import time
import os
import random
import csv
import json
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BASE_URL = "https://api.raptarchis.gov.gr/epk/website/document"
OUTPUT_DIR = "scraped_docs"
METADATA_FILE = "dataset_metadata.csv"
START_ID = 1
END_ID = 3000

# PLATFORM'S API ENDPOINTS
SEARCH_API_URL = "https://searchetv99.azurewebsites.net/api/simplesearch"
TOPICS_API_BASE = "https://searchetv99.azurewebsites.net/api/documententitybyid"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- INITIALIZE CSV ---
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "filename",
            "word_count",
            "title",
            "full_title",
            "article_count",
            "article_titles",
            "fek",
            "fek_search_url",
            "fek_url",
            "topics"
        ])


# --- CLEANING FUNCTIONS ---
def clean_legal_text(raw_text):
    if not raw_text: return None
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    pattern = r'(?:^|\s)(?:\d+|[α-ωΑ-Ωa-zA-Z]{1,3})[\.\)](?=\s)'
    clean_text = re.sub(pattern, " ", text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if not clean_text: return None
    return clean_text


def extract_clean_title(full_title_str):
    match = re.search(r'«(.*)»', full_title_str)
    return match.group(1).strip() if match else full_title_str


def clean_fek_for_filename(fek_str):
    if not fek_str: return "NOFEK"
    clean = fek_str.replace("'", "").replace("’", "").replace(" ", "")
    clean = re.sub(r'[^\w]', '', clean)
    return clean.strip()


# --- GENERATE SEARCH FORM URL ---
def generate_search_form_url(fek_string, date_string):
    try:
        if not fek_string or not date_string: return ""

        # Extract Year, FEK Issue and Number
        year = date_string.split("/")[-1]
        match = re.search(r'([Α-ΩA-Z\.]+).*?(\d+)', fek_string)
        if match:
            issue_raw = match.group(1).replace("΄", "").replace("'", "").strip().upper()
            fek_number = match.group(2)

            issue_map = {
                'Α': '1', 'A': '1',
                'Β': '2', 'B': '2',
                'Γ': '3',
                'Δ': '4',
                'Α.ΕΙ.Δ.': '9',
                'ΑΣΕΠ': '10',
                'ΠΡΑ.Δ.Ι.Τ.': '11',
                'Δ.Δ.Σ': '12',
                'Υ.Ο.Δ.Δ.': '14',
            }

            select_issue = issue_map.get(issue_raw)

            if select_issue:
                return f"https://search.et.gr/el/simple-search/?selectYear={year}&selectIssue={select_issue}&documentNumber={fek_number}"

    except Exception as e:
        print(f"Error generating URL: {e}")
        return ""

    return ""


# --- GET FEK ID  ---
def get_fek_id_from_api(fek_string, date_string):
    try:
        if not fek_string or not date_string: return None
        year = date_string.split("/")[-1]
        match = re.search(r'([Α-ΩA-Z\.]+).*?(\d+)', fek_string)
        if not match: return None

        issue_raw = match.group(1).replace("΄", "").replace("'", "").strip()
        fek_number = match.group(2)
        issue_map = {
                'Α': '1', 'A': '1',
                'Β': '2', 'B': '2',
                'Γ': '3',
                'Δ': '4',
                'Α.ΕΙ.Δ.': '9',
                'ΑΣΕΠ': '10',
                'ΠΡΑ.Δ.Ι.Τ.': '11',
                'Δ.Δ.Σ': '12',
                'Υ.Ο.Δ.Δ.': '14',
            }
        issue_id = issue_map.get(issue_raw, "1")

        payload = {
            "selectYear": [str(year)],
            "selectIssue": [str(issue_id)],
            "documentNumber": str(fek_number),
            "searchText": "",
            "datePublished": "",
            "dateReleased": ""
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Origin": "https://search.et.gr",
            "Referer": "https://search.et.gr/"
        }

        response = requests.post(SEARCH_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            outer_data = response.json()
            inner_data_str = outer_data.get("data", "[]")
            results = json.loads(inner_data_str)
            if results and isinstance(results, list):
                return results[0].get("search_ID")
    except Exception as e:
        print(f"   [Search API Error] {e}")
        return None
    return None


# --- GET TOPICS ---
def get_topics_from_api(fek_id):
    if not fek_id: return ""
    url = f"{TOPICS_API_BASE}/{fek_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            outer_data = response.json()
            inner_data_str = outer_data.get("data", "[]")
            items = json.loads(inner_data_str)
            topics_found = []
            for item in items:
                subject = item.get("documententitybyid_subjects_Value")
                if subject:
                    topics_found.append(subject)
            return " | ".join(topics_found)
    except Exception as e:
        print(f"   [Topics API Error] {e}")
        return ""
    return ""


# --- PROCESS DOCUMENT ---
def process_document(doc_id):
    url = f"{BASE_URL}/{doc_id}"
    print(f"[{doc_id}] Fetching...", end=" ", flush=True)

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed ({response.status_code}) - Skipping.")
            return

        data = response.json()
        if not data or "id" not in data:
            print("Empty Data - Skipping.")
            return

    except Exception as e:
        print(f"Error: {e}")
        return

    # Metadata
    fek_value = ""
    publish_date = ""
    meta_list = data.get("metadata", [])
    if meta_list:
        fek_value = meta_list[0].get("fek", "").strip()
        fek_value = re.sub(r'\s+', ' ', fek_value)
        publish_date = meta_list[0].get("publish_date", "").strip()

    search_form_link = generate_search_form_url(fek_value, publish_date)
    fek_id = get_fek_id_from_api(fek_value, publish_date)

    direct_link = ""
    topics_str = ""
    if fek_id:
        direct_link = f"https://search.et.gr/el/fek/?fekId={fek_id}"
        topics_str = get_topics_from_api(fek_id)

    # Text Extraction
    text_collector = []
    article_headers_list = []
    total_articles = 0

    for item in data.get("arrangements", []):
        txt = clean_legal_text(item.get("text", ""))
        if txt: text_collector.append(txt)

    def process_article_node(article_node):
        nonlocal total_articles
        raw_num = article_node.get("number", "")
        raw_title = article_node.get("title", "")
        art_num = re.sub(r'\s+', ' ', str(raw_num)).strip()
        art_title = re.sub(r'\s+', ' ', str(raw_title)).strip()

        if not art_num:
            clean_node_text = clean_legal_text(article_node.get("text", ""))
            if clean_node_text and len(clean_node_text) < 20:
                art_num = clean_node_text

        full_header = f"Άρθρο {art_num}: {art_title}".strip()
        full_header = full_header.replace(":", " ").replace("  ", " ").strip()
        if full_header.endswith(":"): full_header = full_header[:-1]

        if full_header: article_headers_list.append(full_header)
        total_articles += 1

        for paragraph in article_node.get("paragraphs", []):
            txt = clean_legal_text(paragraph.get("text", ""))
            if txt: text_collector.append(txt)

    for part in data.get("parts", []):
        for chapter in part.get("chapters", []):
            for article in chapter.get("articles", []):
                process_article_node(article)

    for article in data.get("articles", []):
        process_article_node(article)

    if not text_collector:
        print("No readable text found.")
        return

    # Save File
    full_text = " ".join(text_collector)
    full_text = re.sub(r'\s+', ' ', full_text)

    clean_fek = clean_fek_for_filename(fek_value)
    filename_only = f"{clean_fek}_{doc_id}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename_only)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Save CSV
    raw_doc_title = data.get("title", "No Title")
    doc_title = re.sub(r'\s+', ' ', raw_doc_title).strip()
    clean_title = extract_clean_title(doc_title)
    word_count = len(full_text.split())
    articles_joined = " | ".join(article_headers_list)

    with open(METADATA_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            doc_id,
            filename_only,
            word_count,
            clean_title,
            doc_title,
            total_articles,
            articles_joined,
            fek_value,
            search_form_link,
            direct_link,
            topics_str
        ])

    print(f"Saved: {filename_only}")


# --- MAIN ---
print(f"Starting Scrape from ID {START_ID} to {END_ID}...")
for current_id in range(START_ID, END_ID + 1):
    process_document(current_id)
    time.sleep(random.uniform(2, 5))
print("Done.")