import requests
from bs4 import BeautifulSoup
import os
import re
import random
import time
from urllib.parse import urljoin
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://eur-lex.europa.eu"
SUMMARY_HOME = urljoin(BASE_URL, "/browse/summaries.html")
DOWNLOAD_DIR = Path("eudocs")
TOPICS_FILE = "topics.txt"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

def _flatten_bracket_blocks(text: str) -> str:
    # Collapse any newlines/spans inside
    def _flatten(match):
        s = match.group(0)
        open_b, close_b = s[0], s[-1]
        inner = s[1:-1]

        # Replace any newline (or multiple) inside with single space
        inner = re.sub(r'\s*\n+\s*', ' ', inner)

        # Normalize spaces
        inner = re.sub(r'[ \t]{2,}', ' ', inner).strip()

        # Ensure a space around dashes that are glued to words
        inner = re.sub(r'(?<=\S)-(?=\S)', ' - ', inner)
        inner = re.sub(r'(?<! )-(?=\S)', ' - ', inner)

        return f'{open_b}{inner}{close_b}'

    # Non-greedy to keep smallest bracketed chunks
    text = re.sub(r'\(.*?\)', _flatten, text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]', _flatten, text, flags=re.DOTALL)
    return text

def _fix_orphan_brackets_and_punct(text: str) -> str:
    # Pull punctuation that lands on its own next line up to the previous line
    text = re.sub(r'\s*\n\s*([.,;:!?])', r'\1', text)

    # Parentheses: keep "(" and ")" on the same line as their content
    text = re.sub(r'\(\s*\n\s*', '(', text)
    text = re.sub(r'\s*\n\s*\)', ')', text)
    # (and other punctuation)
    text = re.sub(r'\)\s*\n\s*([.,;:!?])', r')\1', text)

    # Same for "[" and "]"
    text = re.sub(r'\[\s*\n\s*', '[', text)
    text = re.sub(r'\s*\n\s*\]', ']', text)
    text = re.sub(r'\]\s*\n\s*([.,;:!?])', r']\1', text)

    return text

def clean_text(raw_text):
    # --- normalize whitespace first ---
    text = re.sub(r'\r\n?', '\n', raw_text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

    merged = []
    for ln in lines:
        if re.fullmatch(r'[.,;:!?]+', ln):
            if merged:
                merged[-1] = (merged[-1] + ln).strip()
            else:
                merged.append(ln)
        else:
            merged.append(ln)
    lines = merged

    def is_all_caps_header(s: str) -> bool:
        if not s: return False
        if not re.search(r'[A-Za-zΑ-Ωάέήίόύώϊϋΐΰ]', s): return False
        if re.fullmatch(r'[\W_]+', s): return False
        return s == s.upper()

    def is_enumeration_or_bullet(s: str) -> bool:
        return bool(re.match(
            r'^\s*(?:[•\-–]\s+|'           
            r'\(?\d+(?:\.\d+)*\)?\s+|'             
            r'[a-z]\)\s+|\([a-z]\)\s+|'           
            r'[IVXLCM]+\.\s+|'                       
            r'[Α-Ω]\)\s+)'                             
            , s))

    def ends_sentence(s: str) -> bool:
        return bool(re.search(r'[.!?…»”)\]]\s*$', s))

    cleaned = []
    prev = ""

    for i, ln in enumerate(lines):
        if ln == prev:
            continue

        # normalize bullet symbol
        ln = re.sub(r'^\s*[\-–]\s+', '• ', ln)

        # headers (numbered or ALL CAPS)
        if is_all_caps_header(ln) or re.match(r"^(\d+(\.\d+)*|[IVXLCMΑ-Ω]{1,4})\.\s.*", ln):
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            cleaned.append(ln)
            cleaned.append("")
            prev = ln
            continue

        # enumeration/bullet lines
        if is_enumeration_or_bullet(ln):
            cleaned.append(ln)
            prev = ln
            continue

        # merge short/wrapped lines
        if cleaned:
            prev_line = cleaned[-1]
            if prev_line == "" or is_all_caps_header(prev_line) or re.search(r':\s*$', prev_line):
                cleaned.append(ln)
            elif not ends_sentence(prev_line):
                # If next line starts lowercase merge
                if re.match(r'^[a-zά-ώα-ω]', ln):
                    cleaned[-1] = (prev_line + " " + ln).strip()
                # Or if it’s relatively short
                elif len(ln) < 120:
                    cleaned[-1] = (prev_line + " " + ln).strip()
                else:
                    cleaned.append(ln)
            else:
                cleaned.append(ln)
        else:
            cleaned.append(ln)

        prev = ln

    out = "\n".join(cleaned)

    # Rejoin punctuation/brackets split across lines
    out = _fix_orphan_brackets_and_punct(out)

    # flatten multi-line content inside () and []
    out = _flatten_bracket_blocks(out)
    out = re.sub(r'[ \t]+', ' ', out)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return out



def safe_filename(name: str, max_length: int = 100) -> str:
    # Make the filename safe for windows, normalize characters, remove illegal characters and truncate name if too long
    safe = name.translate(str.maketrans('', '', '<>:"/\\|?*'))
    safe = re.sub(r"[\u2019']", "", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip(" .")
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._-")

    return safe

def fetch_html(url, timeout=10):
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def parse_topic_links(home_html):
    soup = BeautifulSoup(home_html, 'html.parser')
    links = []
    for a in soup.select("a[href*='/summary/chapter/']"):
        href = a.get("href")
        if href:
            topic_url = urljoin(SUMMARY_HOME, href)
            topic_title = a.get_text(strip=True) or href.split("/")[-1]
            links.append((topic_title, topic_url))
    return links

def parse_subtopic_links(topic_html):
    soup = BeautifulSoup(topic_html, 'html.parser')
    links = []

    for title_link in soup.select("a.summaryTopicTitle"):
        group_title = title_link.get_text(strip=True)
        parent_id = title_link.get("id", "").replace("arrow_", "")
        ul = soup.find("ul", {"data-parent": parent_id})
        if ul:
            for a in ul.select("li.summaryDoc a[href*='/legal-content/summary/']"):
                subtopic_title = a.get_text(strip=True)
                subtopic_url = urljoin(BASE_URL, a["href"])
                links.append((group_title, subtopic_title, subtopic_url))
    return links

def extract_celex(summary_html, summary_url):
    soup = BeautifulSoup(summary_html, 'html.parser')
    info_link = soup.find("a", string=re.compile("Document information", re.IGNORECASE))
    if not info_link or not info_link.get("href"):
        return []

    info_url = urljoin(summary_url, info_link.get("href"))
    try:
        info_html = fetch_html(info_url)
        if not info_html:
            return []
        info_soup = BeautifulSoup(info_html, "html.parser")
        celex_numbers = set()

        for a in info_soup.select("a.title[data-celex]"):
            celex = a.get("data-celex", "").strip()
            if celex:
                celex_numbers.add(celex)

        for a in info_soup.find_all("a", href=True):
            match = re.search(r'CELEX:([0-9A-Z\(\)\-]+)', a['href'])
            if match:
                celex_numbers.add(match.group(1))

        for a in info_soup.find_all("a"):
            text = a.get_text(strip=True)
            if re.fullmatch(r'[A]?\d{6}[A-Z]\d{4}(?:\(\d+\))?(?:-\d{8})?', text):
                celex_numbers.add(text)

        return list(celex_numbers)
    except Exception as e:
        print(f"[ERROR] while extracting CELEX from {info_url}: {e}")
        return []

def fetch_greek_summary(summary_html, summary_url, save_path, subtopic_title, celex=None):
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        soup = BeautifulSoup(summary_html, "html.parser")
        info_link = soup.find("a", string=re.compile("Document information", re.IGNORECASE))
        if not info_link or not info_link.get("href"):
            print(f"[WARN] No Document Information link found in {summary_url}")
            return

        info_url = urljoin(summary_url, info_link.get("href"))
        info_html = fetch_html(info_url)
        if not info_html:
            return

        info_soup = BeautifulSoup(info_html, "html.parser")
        el_link = info_soup.find("a", href=True, id="format_language_table_HTML_EL")
        if not el_link:
            print(f"[WARN] Greek HTML summary not found on Document Info page: {info_url}")
            return

        greek_url = urljoin(info_url, el_link["href"])
        greek_response = requests.get(greek_url, headers=headers)
        greek_response.raise_for_status()

        filename_html = (celex + "_summary.html") if celex else safe_filename(subtopic_title.replace(" ", "_")[:80]) + ".html"
        filepath_html = save_path / filename_html
        with open(filepath_html, "w", encoding="utf-8") as f:
            f.write(greek_response.text)

        filename_txt = filename_html.replace(".html", ".txt")
        filepath_txt = save_path / filename_txt
        with open(filepath_txt, "w", encoding="utf-8") as f:
            raw_txt = html_to_text(greek_response.text)
            f.write(clean_text(raw_txt))
            # f.write(html_to_text(greek_response.text))

        print(f"[OK] Saved Greek summary HTML and TXT for: {subtopic_title}")

    except Exception as e:
        print(f"[ERROR] Failed to fetch Greek summary HTML for {summary_url}: {e}")

def download_document(celex, save_path, source_url=None):
    html_file = save_path / f"{celex}.html"
    pdf_file = save_path / f"{celex}.pdf"
    txt_file = save_path / f"{celex}.txt"

    if html_file.exists() and pdf_file.exists() and txt_file.exists():
        print(f"[SKIP] {celex} already downloaded.")
        return

    html_url = f"{BASE_URL}/legal-content/EL/TXT/HTML/?uri=CELEX:{celex}"
    pdf_url = f"{BASE_URL}/legal-content/EL/TXT/PDF/?uri=CELEX:{celex}"

    try:
        html_response = requests.get(html_url, headers=headers)
        html_response.raise_for_status()
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_response.text)
        with open(txt_file, "w", encoding="utf-8") as f:
            raw_txt = html_to_text(html_response.text)
            f.write(clean_text(raw_txt))
    except Exception as e:
        print(f"[ERROR] Failed to download HTML or TXT for {celex} from {source_url or 'unknown'}: {e}")

    try:
        pdf_response = requests.get(pdf_url, headers=headers)
        pdf_response.raise_for_status()
        with open(pdf_file, "wb") as f:
            f.write(pdf_response.content)
    except Exception as e:
        print(f"[WARN] PDF not available for {celex} from {source_url or 'unknown'}: {e}")

    print(f"[OK] Downloaded {celex}")
    time.sleep(random.uniform(1.0, 2.5))

def read_selected_topics_from_file(filename=TOPICS_FILE):
    if not os.path.exists(filename):
        print(f"[WARN] No {filename} file found. All topics will be processed.")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

def main():
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    home_html = fetch_html(SUMMARY_HOME)
    if not home_html:
        print("[FATAL] Could not fetch summaries home page.")
        return

    topics = parse_topic_links(home_html)
    print(f"Found {len(topics)} topics:")
    for title, url in topics:
        print(f" - {title} ({url})")

    selected_titles = read_selected_topics_from_file()
    if selected_titles:
        topics = [(t, u) for (t, u) in topics if t in selected_titles]

    for topic_title, topic_url in tqdm(topics, desc="Topics"):
        topic_folder = DOWNLOAD_DIR / safe_filename(topic_title.replace(" ", "_"))
        topic_folder.mkdir(exist_ok=True)
        topic_html = fetch_html(topic_url)
        if not topic_html:
            continue
        subtopics = parse_subtopic_links(topic_html)
        print(f"Found {len(subtopics)} subtopics in {topic_title}")
        for g, t, u in subtopics:
            print(f"  - Group: {g} | Subtopic: {t}")

        for group_title, subtopic_title, subtopic_url in tqdm(subtopics, desc=f"{topic_title}", leave=False):
            group_folder = topic_folder / safe_filename(group_title.replace(" ", "_"))
            subtopic_folder = group_folder / safe_filename(subtopic_title.replace(" ", "_"))
            subtopic_folder.mkdir(parents=True, exist_ok=True)
            try:
                summary_html = fetch_html(subtopic_url)
                if not summary_html:
                    continue

                celex_list = extract_celex(summary_html, subtopic_url)

                fetch_greek_summary(summary_html, subtopic_url, subtopic_folder, subtopic_title, celex=celex_list[0] if celex_list else None)

                if celex_list:
                    for celex in celex_list:
                        download_document(celex, subtopic_folder, source_url=subtopic_url)
                else:
                    print(f"[WARN] No CELEX found in {subtopic_url}")
                time.sleep(random.uniform(0.5, 1.2))
            except Exception as e:
                print(f"[ERROR] {subtopic_url}: {e}")

if __name__ == "__main__":
    main()
