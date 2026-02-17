import os
import re
import shutil
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
import trafilatura
import unicodedata

# -------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------
BASE = Path("eudocs")
clean_BASE = Path("eudocsCleaned")
TOPICS_FILE = "topics.txt"
MISMATCH_FOLDERS = []

clean_BASE.mkdir(exist_ok=True)

# --- WHITELIST: KEEP CONTENT UNDER THESE HEADERS ---
SECTIONS_TO_KEEP = [
    "ΤΙ ΠΡΟΒΛΕΠΕΙ", "ΤΙ ΠΡΟΒΛΕΠΟΥΝ", "ΤΙ ΚΑΝΕΙ", "ΤΙ ΘΕΣΠΙΖΕΙ",
    "ΠΟΙΟΣ ΕΙΝΑΙ Ο ΣΚΟΠΟΣ", "ΠΟΙΟΣ ΕΙΝΑΙ Ο ΣΤΟΧΟΣ",
    "ΠΟΙΟΙ ΕΙΝΑΙ ΟΙ ΣΤΟΧΟΙ", "ΠΟΥ ΑΠΟΣΚΟΠΕΙ", "ΠΟΥ ΑΠΟΣΚΟΠΟΥΝ",
    "ΒΑΣΙΚΑ ΣΗΜΕΙΑ", "ΒΑΣΙΚΑ ΣΤΟΙΧΕΙΑ", "ΚΥΡΙΑ ΣΗΜΕΙΑ",
    "ΚΥΡΙΑ ΣΤΟΙΧΕΙΑ", "ΚΥΡΙΕΣ ΠΤΥΧΕΣ", "ΚΥΡΙΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ"
]

SUMMARY_TITLES_FOR_COUNTING = [
    "SUMMARY OF THE FOLLOWING TEXTS", "SUMMARY OF", "ΣΥΝΟΨΗ ΤΩΝ ΑΚΟΛΟΥΘΩΝ ΚΕΙΜΕΝΩΝ",
    "ΣΥΝΟΨΗ ΤΩΝ", "ΣΥΝΟΨΗ ΤΟΥ ΑΚΟΛΟΥΘΟΥ ΕΓΓΡΑΦΟΥ", "ΣΥΝΟΨΗ ΤΟΥ ΑΚΟΛΟΥΘΟΥ",
    "ΣΥΝΟΨΗ ΤΟΥ ΑΚΟΛΟΥΘΟΥ ΚΕΙΜΕΝΟΥ", "ΠΡΑΞΗ"
]

ERROR_SIGNATURES = [
    "cannot be displayed due to its size",
    "δεν μπορεί να προβληθεί λόγω του μεγέθους του",
    "requested document does not exist",
    "το έγγραφο που ζητήσατε δεν υπάρχει",
    "text of this document is not available",
    "κείμενο του εγγράφου αυτού δεν είναι διαθέσιμο",
    "notice: the text of this document is not available",
    "an error has occurred",
    "παρουσιάστηκε σφάλμα"
]

LATIN_TO_GREEK = {
    "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "I": "Ι", "K": "Κ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "Y": "Υ",
    "X": "Χ", "Z": "Ζ"
}


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def cut_annex_fully(soup):
    """
    Finds the Annex header and deletes it + everything following.
    Handles two cases:
    1. 'TexteOnly' (Legacy): Flat list of <p> tags.
    2. Standard (Modern): Nested <div> structures (Bubble Up).
    """

    # 'TEXTEONLY' Scenario (Older Files) ---
    texte_div = soup.find(id="TexteOnly")

    if texte_div:
        # Scan only the direct children (paragraphs/tables) of the main container
        found_annex = False

        # We iterate over a copy (list) so we can modify the DOM safely
        for child in list(texte_div.children):
            if not isinstance(child, Tag): continue

            # If we already found the annex in a previous loop, delete this child
            if found_annex:
                child.decompose()
                continue

            # Check if this child IS the Annex Header
            txt = normalize_greek(child.get_text(" ", strip=True))

            # Strict check for the flat structure:
            # It must start with ΠΑΡΑΡΤΗΜΑ and be short (<60 chars)
            if txt.startswith("ΠΑΡΑΡΤΗΜΑ") and len(txt) < 60:
                found_annex = True
                child.decompose()  # Delete the header itself

        # If  found and cleaned inside TexteOnly continue
        if found_annex:
            return soup

    # --- Rest of cases Scenario (ModernFiles) ---
    candidates = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li"])
    target_tag = None

    for tag in candidates:
        text = normalize_greek(tag.get_text(" ", strip=True))
        if text.startswith("ΠΑΡΑΡΤΗΜΑ") and len(text) < 60:
            target_tag = tag
            break

    if target_tag:
        cursor = target_tag
        while cursor and cursor.name != '[document]':
            siblings = cursor.find_next_siblings()
            for sibling in siblings:
                if hasattr(sibling, "decompose"):
                    sibling.decompose()
                elif hasattr(sibling, "extract"):
                    sibling.extract()
            cursor = cursor.parent
        target_tag.decompose()

    return soup

def normalize_greek(text: str) -> str:
    """Normalizes Greek characters to uppercase for consistent matching."""
    if not text: return ""
    text = unicodedata.normalize("NFKC", text)
    text = ''.join(LATIN_TO_GREEK.get(c, c) for c in text)
    text = ''.join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != 'Mn')
    text = text.upper()
    text = re.sub(r"[\s:·;\*\-\–]+", " ", text).strip()
    return text


def is_heading(tag):
    """
    Determines if a tag is a Structural Header.
    CRITERIA:
    1. Must be h1-h4 OR have a specific 'header' class.
    2. Must be UPPERCASE.
    """
    if not tag or not hasattr(tag, "name"):
        return False

    # 1. Structural Check
    is_structural = False
    if tag.name in ["h1", "h2", "h3", "h4"]:
        is_structural = True
    elif tag.name == "p":
        classes = tag.get("class", [])
        if classes and any(
                cls in classes for cls in ["oj-doc-ti", "oj-ti-grseq1", "ti-chapter", "ti-section", "ti-main"]):
            is_structural = True

    if not is_structural:
        return False

    # 2. Uppercase Check
    text = tag.get_text(strip=True)
    text_letters = re.sub(r'[^a-zA-Zα-ωΑ-Ωά-ώΆ-Ώ]', '', text)

    if len(text_letters) < 2:
        return False

    return text_letters.isupper()


def clean_extracted_text(text):
    """Basic whitespace and list bullet cleanup."""
    # 1. Remove Bullet Points, Dashes, AND Table Pipes (|) at the start of lines
    text = re.sub(r"^\s*[\-\–\—\―\*•\|]+\s*", "", text, flags=re.MULTILINE)

    # 2. Remove " | " that might appear in the middle of sentences due to table merges
    text = text.replace(" | ", " ")

    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def fix_broken_sentences(text):
    text = re.sub(r"\[\d+\]", "", text)
    lines = text.splitlines()
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            if nxt.startswith(". "):
                merged = line + ". " + nxt[2:].lstrip()
                fixed.append(merged)
                i += 2
                continue
            if line and nxt and line[-1].isalnum() and nxt[0].isupper():
                merged = line + ". " + nxt
                fixed.append(merged)
                i += 2
                continue
        fixed.append(line)
        i += 1
    return "\n".join(fixed)


def make_single_line(text: str) -> str:
    if not text: return ""
    return " ".join(text.split())


def clean_annex_from_text(text: str) -> str:
    """
    Truncates text at the 'Annex' keyword.
    Enhanced to catch 'ΠΑΡΑΡΤΗΜΑ' even if stuck to previous text.
    """
    pattern = r"(?:^|\n|\.\s+)(ΠΑΡΑΡΤΗΜΑ\s+[IVX0-9A-Z]+|ΠΑΡΑΡΤΗΜΑ\s*$)"

    # Use Split. We take the first part (everything before the Annex).
    parts = re.split(pattern, text, maxsplit=1)
    return parts[0].strip()


def is_html_broken(html_content: str) -> bool:
    text_lower = html_content.lower()
    for error in ERROR_SIGNATURES:
        if error in text_lower:
            return True
    return False


def clean_text_micro(text: str) -> str:
    if not text: return ""
    text = re.sub(r'([α-ωά-ώa-z0-9])\.([Α-ΩΆ-ΏA-Z])', r'\1. \2', text)
    text = re.sub(r'([α-ωά-ώa-z])\(', r'\1 (', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s\(\d+\)\s', ' ', text)
    text = text.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"')
    text = re.sub(r'\(EE\s+[LcC]\s+\d+.*?\)', '', text)
    return text


# -------------------------------------------------------------------
# HTML EXTRACTION
# -------------------------------------------------------------------

def extract_content_sections(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    whitelist_norm = [normalize_greek(k) for k in SECTIONS_TO_KEEP]

    kept_content = []

    for tag in list(soup.find_all(is_heading)):
        raw_text = tag.get_text(" ", strip=True)
        text_norm = normalize_greek(raw_text)

        match_found = False
        for key in whitelist_norm:
            if text_norm.startswith(key):
                match_found = True
                break

        if match_found:
            node = tag.next_sibling
            while node:
                if isinstance(node, Tag) and is_heading(node):
                    break
                if isinstance(node, Tag):
                    kept_content.append(str(node))
                node = node.next_sibling

    if not kept_content: return ""
    return "\n".join(kept_content)


# -------------------------------------------------------------------
# DOM CLEANING HELPERS
# -------------------------------------------------------------------

def remove_unwanted_sections_legacy(html: str) -> str:
    """Standard cleaning for legal texts (not summaries)."""
    soup = BeautifulSoup(html, "html.parser")
    for p in soup.find_all("p", class_=["lastmod", "lseu-lastmod"]): p.decompose()
    for p in soup.find_all("p"):
        t = normalize_greek(p.get_text(" ", strip=True))
        if t.startswith("ΤΕΛΕΥΤΑΙΑ ΕΝΗΜΕΡΩΣΗ"): p.decompose()
    return str(soup)


def restore_time_tags(soup):
    for t in soup.find_all("time"):
        dt = t.get("datetime")
        if dt:
            try:
                y, m, d = dt.split("-")
                t.string = f"{int(d)}.{int(m)}.{y}"
            except:
                pass
    return soup


def remove_bottom_footnotes(soup):
    footnote_block = soup.find(id="footnotes")
    if footnote_block: footnote_block.decompose()
    for div in soup.find_all("div", class_="footnotes"): div.decompose()
    return soup


def remove_footnotes(soup):
    """Crash-proof footnote remover."""
    pattern = re.compile(r"[\(\[]?\s*\d+\s*[\)\]]?")
    for tag in soup.find_all(["a", "sup", "span"]):
        if not hasattr(tag, 'attrs') or tag.attrs is None:
            pass
        else:
            classes = tag.get("class", [])
            if "footnote" in classes or "num" in classes:
                tag.decompose()
                continue
        if pattern.fullmatch(tag.get_text(strip=True)):
            tag.decompose()
    return soup


def remove_note_blocks(soup):
    for hr in soup.find_all("hr", class_=["note", "oj-note"]):
        nxt = hr.next_sibling
        hr.decompose()
        while nxt:
            if getattr(nxt, "name", None) == "p" and any(c in ["note", "oj-note"] for c in nxt.get("class", [])):
                tmp = nxt;
                nxt = nxt.next_sibling;
                tmp.decompose()
            else:
                break
    for p in soup.find_all("p", class_=["note", "oj-note"]): p.decompose()
    return soup


def remove_toc(soup):
    toc_heading = soup.find("p", class_="TOCheading")
    if not toc_heading: return soup
    node = toc_heading
    next_node = node.next_sibling
    node.decompose()
    while next_node:
        if not hasattr(next_node, "name") or next_node.name != "p": break
        cls = next_node.get("class", [])
        if any("TOC" in c for c in cls):
            tmp = next_node;
            next_node = next_node.next_sibling;
            tmp.decompose()
            continue
        break
    return soup


def flatten_tables_in_texteonly(soup, root):
    for table in root.find_all("table"):
        new_paragraphs = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells: continue
            p = soup.new_tag("p")
            p.string = " ".join(cells)
            new_paragraphs.append(p)
        for p in new_paragraphs: table.insert_before(p)
        table.decompose()


def flatten_all_tables(soup):
    """Converts all tables into simple paragraphs (Handles nesting)."""
    tables = soup.find_all("table")
    for table in tables:
        if table.parent is None: continue

        new_paragraphs = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c.strip() not in ["", "|", "-", "—", "·"]]
            if not cells: continue
            p = soup.new_tag("p")
            p.string = " ".join(cells).strip()
            new_paragraphs.append(p)

        for p in new_paragraphs: table.insert_before(p)
        table.decompose()


# -------------------------------------------------------------------
# PIPELINES
# -------------------------------------------------------------------

def html_to_text_summary(html):
    """Pipeline for Summary files: Whitelist -> Flatten Tables -> Extract."""
    # Whitelist Extraction
    valid_html = extract_content_sections(html)
    if not valid_html: return ""

    # Flatten Tables (Tables had nested content)
    soup = BeautifulSoup(valid_html, "html.parser")
    flatten_all_tables(soup)
    valid_html_flat = str(soup)

    # Extract Text
    extracted = trafilatura.extract(
        valid_html_flat,
        include_tables=False,
        favor_recall=True,
        deduplicate=False,
        target_language="el"
    )

    if not extracted:
        extracted = soup.get_text(" ", strip=True)

    extracted = clean_extracted_text(extracted)
    extracted = clean_text_micro(extracted)
    return make_single_line(extracted)

def html_to_text_celex(html):
    """Pipeline for Legal files."""
    html = remove_unwanted_sections_legacy(html)

    soup = BeautifulSoup(html, "html.parser")
    soup = restore_time_tags(soup)
    soup = remove_footnotes(soup)
    soup = remove_bottom_footnotes(soup)
    soup = remove_note_blocks(soup)
    soup = remove_toc(soup)
    soup = cut_annex_fully(soup)

    texte = soup.find(id="TexteOnly")
    if texte:
        flatten_tables_in_texteonly(soup, texte)
        extracted = texte.get_text("\n", strip=True)
    else:
        flatten_all_tables(soup)
        extracted = trafilatura.html2txt(str(soup))
        if not extracted: extracted = soup.get_text("\n")

    extracted = re.sub(r"\(\d{1,3}\)", "", extracted)
    extracted = clean_extracted_text(extracted)
    extracted = fix_broken_sentences(extracted)
    extracted = clean_text_micro(extracted)

    return make_single_line(extracted)

# -------------------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------------------

def count_docs_in_summary(summary_html: str) -> int:
    soup = BeautifulSoup(summary_html, "html.parser")
    NORMALIZED_TITLES = [normalize_greek(t) for t in SUMMARY_TITLES_FOR_COUNTING]

    def has_summary_title(text: str) -> bool:
        nt = normalize_greek(text)
        return any(nt.startswith(t) or t in nt for t in NORMALIZED_TITLES)

    section = soup.find("section", class_="lseu-section-summary-of")
    if section:
        count = 0
        for p in section.find_all("p"):
            for a in p.find_all("a", href=True):
                href = a["href"].lower()
                if "celex:" in href or "/eli/" in href:
                    count += 1
                    break
        return count

    summary_node = None
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "b", "strong", "span"]):
        text = tag.get_text(" ", strip=True)
        if has_summary_title(text):
            summary_node = tag
            break
    if not summary_node: return 0

    count = 0
    node = summary_node
    while True:
        node = node.next_sibling
        if node is None: break
        if not isinstance(node, Tag): continue
        ntext = normalize_greek(node.get_text(" ", strip=True))
        if is_heading(node) and (ntext.startswith("ΠΟΙΟΣ") or ntext.startswith("ΒΑΣΙΚΑ")): break
        if node.name == "p":
            for a in node.find_all("a", href=True):
                href = a["href"].lower()
                if "celex:" in href or "/eli/" in href:
                    count += 1
                    break
    return count


def count_real_html_files(folder: Path):
    html_files = [f for f in folder.glob("*.html")]
    html_files = [f for f in html_files if not f.name.lower().endswith("_summary.html")]
    return len(html_files)


def read_selected_topics():
    if not os.path.exists(TOPICS_FILE): return None
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def process_folder(topic_path: Path, output_topic_path: Path):
    all_subfolders = list(os.walk(topic_path))

    for root, dirs, files in tqdm(all_subfolders, desc=f"-> {topic_path.name}", leave=False):
        root_path = Path(root)

        # Identify Summary File
        summary_files = [f for f in files if f.lower().endswith("_summary.html")]
        if not summary_files: continue
        summary_file = root_path / summary_files[0]

        # Validation
        with open(summary_file, "r", encoding="utf-8", errors="ignore") as f:
            summary_html = f.read()

        if "<table" in summary_html.lower():
            continue

        num_listed = count_docs_in_summary(summary_html)
        num_actual = count_real_html_files(root_path)

        if num_listed != num_actual:
            MISMATCH_FOLDERS.append({"path": str(root_path), "listed": num_listed, "actual": num_actual})
            continue

        if is_html_broken(summary_html): continue

        # Setup Target Directory
        rel = root_path.relative_to(BASE)
        target_dir = clean_BASE / rel
        target_dir.mkdir(parents=True, exist_ok=True)

        # Process Summary
        shutil.copy(summary_file, target_dir / "summary.html")
        cleaned_summary = html_to_text_summary(summary_html)
        summary_word_count = len(re.findall(r"\w+", cleaned_summary, flags=re.UNICODE))

        if summary_word_count < 30:
            shutil.rmtree(target_dir)
            continue

        with open(target_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(cleaned_summary)

        # Process Legal Texts
        legal_htmls = [f for f in root_path.glob("*.html") if not f.name.lower().endswith("_summary.html")]
        expected_docs = len(legal_htmls)
        successful_docs = 0

        for f_html in legal_htmls:
            with open(f_html, "r", encoding="utf-8", errors="ignore") as f_in:
                html_doc = f_in.read()

            if is_html_broken(html_doc): break

            cleaned_text = html_to_text_celex(html_doc)
            word_count = len(re.findall(r"\w+", cleaned_text, flags=re.UNICODE))

            if word_count > 15000: break
            if word_count < 250: break
            if word_count < summary_word_count: break

            shutil.copy(f_html, target_dir / f_html.name)
            out_txt = target_dir / (f_html.stem + ".txt")
            with open(out_txt, "w", encoding="utf-8") as f_out:
                f_out.write(cleaned_text)

            successful_docs += 1

        if successful_docs != expected_docs:
            shutil.rmtree(target_dir)


# -------------------------------------------------------------------
# FINAL CLEANUP & RUN
# -------------------------------------------------------------------

def is_leaf_folder(folder: Path) -> bool:
    return all(f.is_file() for f in folder.iterdir())


def is_2x2_folder(folder: Path) -> bool:
    files = [f.name.lower() for f in folder.iterdir() if f.is_file()]
    html_count = sum(1 for f in files if f.endswith(".html"))
    txt_count = sum(1 for f in files if f.endswith(".txt"))
    return (html_count == 2 and txt_count == 2 and "summary.html" in files and "summary.txt" in files)


def filter_only_2x2():
    print("\n=== FINAL CLEANUP: REMOVING INCOMPLETE FOLDERS ===")
    removed = 0
    for root, dirs, files in os.walk(clean_BASE, topdown=False):
        folder = Path(root)
        if not is_leaf_folder(folder): continue
        if not is_2x2_folder(folder):
            shutil.rmtree(folder)
            removed += 1
    print(f"[INFO] Removed {removed} incomplete/invalid folders.")


def main():
    selected_topics = read_selected_topics()
    topics = sorted([p for p in BASE.iterdir() if p.is_dir()])
    if selected_topics:
        topics = [t for t in topics if t.name in selected_topics]

    print("\n=== SELECTED TOPICS ===")
    for t in topics: print(" ->", t.name)
    print("\n=== STARTING PROCESSING ===\n")

    for topic in tqdm(topics, desc="TOPICS"):
        topic_out = clean_BASE / topic.name
        topic_out.mkdir(exist_ok=True)
        process_folder(topic, topic_out)

    print("\n=== MISMATCHED FOLDERS (SKIPPED) ===")
    with open("mismatch_report.txt", "w", encoding="utf-8") as f:
        for m in MISMATCH_FOLDERS:
            line = f"{m['path']} - Listed={m['listed']} , Actual={m['actual']}"
            f.write(line + "\n")
            print(line)

    filter_only_2x2()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()