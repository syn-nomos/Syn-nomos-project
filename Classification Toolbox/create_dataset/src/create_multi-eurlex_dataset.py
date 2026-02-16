import json
import logging
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Set, Any

from datasets import load_dataset
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HELLASVOC_PATH = Path("./data/hellasvoc_hierarchy_new.jsonl")
EUROVOC_DESC_PATH = Path("./data/eurovoc_descriptors.json")
OUTPUT_DIR = Path("./multi_eurlex_hellasvoc_out")

LANG = "el"  # γλώσσα Multi-EURLEX
KEEP_DOCS_WITHOUT_HELLAS = False  # αν True κρατάει και όσα δεν έχουν HellasVoc mapping

LABEL_LEVELS = ["level_1", "level_2", "level_3"]

# HF splits -> δικά μας ονόματα
SPLIT_NAME_MAP = {
    "train": "train",
    "dev": "validation",  # HF: "validation"
    "test": "test",
}

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers για HellasVoc
# ----------------------------------------------------------------------
def iter_hellas_nodes(path: Path) -> Iterable[dict]:
    """
    Κάθε γραμμή στο JSONL είναι ένα root node με children tree.
    Κάνουμε DFS και επιστρέφουμε όλα τα nodes.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            root = json.loads(line)
            stack = [root]
            while stack:
                node = stack.pop()
                yield node
                stack.extend(node.get("children", []))


def extract_eurovoc_id_from_link(link: str) -> str:
    """
    Παίρνει ένα external_uris.link και επιστρέφει το EuroVoc ID (π.χ. '100171')
    ή κενό string αν δεν είναι EuroVoc URI.

    Υποστηρίζει:
      - 'http://eurovoc.europa.eu/100171'
      - 'https://eurovoc.europa.eu/100171'
      - 'https://op.europa.eu/...uri=http://eurovoc.europa.eu/100171&lang=el'
    """
    if not isinstance(link, str):
        return ""
    link = link.strip()
    if not link:
        return ""

    # direct EuroVoc link
    if "eurovoc.europa.eu" in link:
        marker_http = "http://eurovoc.europa.eu/"
        marker_https = "https://eurovoc.europa.eu/"
        if marker_http in link:
            idx = link.index(marker_http)
            uri = link[idx:].split("&", 1)[0]
        elif marker_https in link:
            idx = link.index(marker_https)
            uri = link[idx:].split("&", 1)[0]
        else:
            # κάποια άλλη μορφή, αλλά περιέχει eurovoc.europa.eu
            uri = link
        uri = uri.split("?", 1)[0].split("#", 1)[0].strip()
        eurovoc_id = uri.rsplit("/", 1)[-1].strip()
        return eurovoc_id

    # op.europa wrapper με uri=http://eurovoc.europa.eu/XXXX
    if "op.europa.eu" in link and "uri=http://eurovoc.europa.eu/" in link:
        marker = "uri=http://eurovoc.europa.eu/"
        idx = link.index(marker) + len("uri=http://")
        inner = link[idx:].split("&", 1)[0].split("#", 1)[0].strip()
        # inner είναι π.χ. 'eurovoc.europa.eu/100171'
        if "eurovoc.europa.eu" in inner:
            eurovoc_id = inner.rsplit("/", 1)[-1].strip()
            return eurovoc_id

    return ""


def build_hellas_eurovoc_index(path: Path) -> Dict[str, Set[Tuple[str, str]]]:
    """
    eurovoc_id (str) -> set of (hellas_id, hellas_label)

    Δηλαδή, για κάθε EuroVoc concept ID που εμφανίζεται στα external_uris
    του HellasVoc, κρατάμε τα αντίστοιχα HellasVoc nodes (id + name).
    """
    index: Dict[str, Set[Tuple[str, str]]] = {}

    logger.info("Χτίσιμο index EuroVoc ID -> HellasVoc nodes...")
    for node in tqdm(iter_hellas_nodes(path), desc="Indexing HellasVoc → EuroVoc IDs"):
        hellas_id = (str(node.get("id", "")) or "").strip()
        hellas_label = (node.get("name") or "").strip()
        if not hellas_label:
            continue

        ext_uris = node.get("external_uris", []) or []
        if not isinstance(ext_uris, list):
            continue

        for uri_info in ext_uris:
            if not isinstance(uri_info, dict):
                continue
            link = uri_info.get("link", "") or uri_info.get("uri", "") or uri_info.get("url", "")
            eurovoc_id = extract_eurovoc_id_from_link(link)
            if not eurovoc_id:
                continue

            index.setdefault(eurovoc_id, set()).add((hellas_id, hellas_label))

    logger.info("Το HellasVoc περιέχει EuroVoc IDs για %d διαφορετικά concepts.", len(index))
    return index


def collect_hellas_labels_with_eurovoc_uri(path: Path) -> List[Dict[str, Any]]:
    """
    Γυρίζει λίστα με ΟΛΑ τα HellasVoc labels που έχουν EuroVoc URI:
      [
        {
          "hellas_id": ...,
          "hellas_label": ...,
          "eurovoc_id": ...,
          "link": ...
        },
        ...
      ]
    Χρήσιμο για να δούμε ποια δεν χρησιμοποιούνται ποτέ στο Multi-EURLEX.
    """
    records: List[Dict[str, Any]] = []

    logger.info("Συλλογή HellasVoc labels με EuroVoc URI (για στατιστικά)...")
    for node in tqdm(iter_hellas_nodes(path), desc="Scanning HellasVoc for EuroVoc labels"):
        hellas_id = (str(node.get("id", "")) or "").strip()
        hellas_label = (node.get("name") or "").strip()
        if not hellas_label:
            continue

        ext_uris = node.get("external_uris", []) or []
        if not isinstance(ext_uris, list):
            continue

        for uri_info in ext_uris:
            if not isinstance(uri_info, dict):
                continue
            link = uri_info.get("link", "") or uri_info.get("uri", "") or uri_info.get("url", "")
            eurovoc_id = extract_eurovoc_id_from_link(link)
            if not eurovoc_id:
                continue

            records.append(
                {
                    "hellas_id": hellas_id,
                    "hellas_label": hellas_label,
                    "eurovoc_id": eurovoc_id,
                    "link": link,
                }
            )

    logger.info("HellasVoc labels με EuroVoc URI: %d", len(records))
    return records


# ----------------------------------------------------------------------
# Helpers για Multi-EURLEX
# ----------------------------------------------------------------------
def load_eurovoc_descriptors(path: Path) -> Dict[str, Any]:
    """
    Διαβάζει το eurovoc_descriptors.json
    Μπορεί να είναι:
      - { "100160": "industry", ... }
      - ή { "100160": {"en": "...", "el": "...", ...}, ... }
    Το κρατάμε raw και θα τραβάμε ελληνικά αργότερα.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Φορτώθηκαν %d EuroVoc descriptors από %s", len(data), path)
    return data


def get_greek_descriptor(descriptors: Dict[str, Any], eurovoc_id: str) -> str:
    """
    Επιστρέφει μόνο το ελληνικό descriptor για το given EuroVoc ID.
    Αν το entry είναι dict, παίρνει ["el"]. Αν είναι str, το γυρνάει as-is.
    """
    val = descriptors.get(eurovoc_id, "")
    if isinstance(val, dict):
        # πιθανόν τα keys να είναι "el" (lowercase)
        return val.get("el", "") or ""
    if val is None:
        return ""
    return str(val)


# ----------------------------------------------------------------------
# Κύρια συνάρτηση build
# ----------------------------------------------------------------------
def build_multi_eurlex_hellasvoc_dataset(
    hellasvoc_path: Path,
    eurovoc_desc_path: Path,
    output_dir: Path,
    lang: str = "el",
    keep_docs_without_hellas: bool = False,
) -> None:
    logger.info("Ξεκινά build για Multi-EURLEX (lang=%s) → HellasVoc (για όλα τα levels).", lang)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) HellasVoc indices (EuroVoc ID → HellasVoc labels) – κοινό για όλα τα levels
    hellas_eurovoc_index = build_hellas_eurovoc_index(hellasvoc_path)
    hellas_eurovoc_labels = collect_hellas_labels_with_eurovoc_uri(hellasvoc_path)

    # 2) EuroVoc descriptors (multilingual ή όχι)
    eurovoc_desc = load_eurovoc_descriptors(eurovoc_desc_path)

    # Stats struct: per level + split
    stats: Dict[str, Any] = {
        "levels": {},
        "global": {
            "language": lang,
            "label_levels": LABEL_LEVELS,
        },
    }

    # --------------------------------------------------------------
    # Loop ανά level
    # --------------------------------------------------------------
    for label_level in LABEL_LEVELS:
        logger.info("=" * 80)
        logger.info("Επεξεργασία label_level = %s", label_level)

        # Φόρτωση Multi-EURLEX για αυτό το level
        logger.info("Φόρτωση Multi-EURLEX (lang=%s, label_level=%s)...", lang, label_level)
        ds_all = load_dataset("multi_eurlex", lang, label_level=label_level, trust_remote_code=True)

        # HF δίνει splits: "train", "validation", "test"
        # Εμείς θα παράξουμε train/dev/test με dev = validation
        # ClassLabel είναι κοινό για όλα τα splits
        classlabel = ds_all["train"].features["labels"].feature  # ClassLabel

        # Global stats per level
        global_used_hellas_ids: Set[str] = set()
        global_used_eurovoc_ids: Set[str] = set()

        level_stats = {
            "splits": {},
            "eurovoc": {},
            "hellasvoc": {},
        }

        for split_name, hf_split_name in SPLIT_NAME_MAP.items():
            ds = ds_all[hf_split_name]

            out_path = output_dir / f"multi_eurlex_{lang}_{label_level}_hellasvoc_{split_name}.jsonl"
            logger.info("  [level=%s] split '%s' (HF: '%s') → %s (%d docs)",
                        label_level, split_name, hf_split_name, out_path, len(ds))

            num_docs_total = len(ds)
            num_docs_exported = 0
            num_docs_without_hellas = 0

            split_used_hellas_ids: Set[str] = set()
            split_used_eurovoc_ids: Set[str] = set()
            labels_per_doc: List[int] = []

            with out_path.open("w", encoding="utf-8") as fout:
                for sample in tqdm(ds, desc=f"[{label_level}] Processing {split_name}", unit="doc"):
                    celex_id = sample["celex_id"]
                    text = sample["text"]
                    label_ids = sample["labels"]  # list[int]

                    eurovoc_entries: List[Dict[str, Any]] = []
                    hellas_terms_set: Set[Tuple[str, str]] = set()

                    for lid in label_ids:
                        eurovoc_id = classlabel.int2str(lid)  # π.χ. '100160'
                        if not eurovoc_id:
                            continue

                        split_used_eurovoc_ids.add(eurovoc_id)
                        global_used_eurovoc_ids.add(eurovoc_id)

                        desc_el = get_greek_descriptor(eurovoc_desc, eurovoc_id)

                        eurovoc_entries.append(
                            {
                                "id": eurovoc_id,
                                "descriptor": desc_el,  # ΜΟΝΟ ελληνικά
                            }
                        )

                        # HellasVoc matching (ID-based)
                        matches = hellas_eurovoc_index.get(eurovoc_id)
                        if matches:
                            hellas_terms_set.update(matches)

                    if not eurovoc_entries:
                        # δεν θα έπρεπε να συμβεί στο Multi-EURLEX, αλλά ας είμαστε ασφαλείς
                        num_docs_without_hellas += 1
                        if not keep_docs_without_hellas:
                            continue

                    if not hellas_terms_set:
                        num_docs_without_hellas += 1
                        if not keep_docs_without_hellas:
                            # skip όσα δεν έχουν HellasVoc mapping
                            continue

                    hellasvoc_list = [
                        {"id": hid, "label": hlabel}
                        for (hid, hlabel) in sorted(
                            hellas_terms_set, key=lambda x: (x[0] or "", x[1])
                        )
                    ]

                    # stats per split
                    labels_per_doc.append(len(hellasvoc_list))
                    split_used_hellas_ids.update(hid for (hid, _) in hellas_terms_set)
                    global_used_hellas_ids.update(hid for (hid, _) in hellas_terms_set)

                    record = {
                        "celex_id": celex_id,
                        "split": split_name,   # "train"/"dev"/"test"
                        "label_level": label_level,
                        "text": text,
                        "eurovoc": eurovoc_entries,
                        "hellasvoc": hellasvoc_list,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_docs_exported += 1

            if labels_per_doc:
                avg_labels = sum(labels_per_doc) / len(labels_per_doc)
                min_labels = min(labels_per_doc)
                max_labels = max(labels_per_doc)
            else:
                avg_labels = 0.0
                min_labels = 0
                max_labels = 0

            level_stats["splits"][split_name] = {
                "num_docs_total_in_split": num_docs_total,
                "num_docs_exported_with_hellasvoc": num_docs_exported,
                "num_docs_without_hellasvoc_mapping": num_docs_without_hellas,
                "num_unique_hellasvoc_labels": len(split_used_hellas_ids),
                "avg_hellasvoc_labels_per_doc": avg_labels,
                "min_hellasvoc_labels_per_doc": min_labels,
                "max_hellasvoc_labels_per_doc": max_labels,
            }

            logger.info(
                "  [level=%s] [%s] docs total: %d, exported (with HellasVoc): %d, without HellasVoc: %d",
                label_level,
                split_name,
                num_docs_total,
                num_docs_exported,
                num_docs_without_hellas,
            )

        # ----------------------------------------------------------
        # Global stats ανά level: unmatched EuroVoc & unused HellasVoc
        # ----------------------------------------------------------
        hellas_eurovoc_ids: Set[str] = set(hellas_eurovoc_index.keys())

        eurovoc_ids_missing_in_hellas = sorted(list(global_used_eurovoc_ids - hellas_eurovoc_ids))

        missing_eurovoc_records: List[Dict[str, Any]] = []
        for eid in eurovoc_ids_missing_in_hellas:
            missing_eurovoc_records.append(
                {
                    "eurovoc_id": eid,
                    "descriptor": get_greek_descriptor(eurovoc_desc, eid),
                }
            )

        missing_eurovoc_path = output_dir / f"multi_eurlex_{lang}_{label_level}_eurovoc_missing_in_hellasvoc.json"
        with missing_eurovoc_path.open("w", encoding="utf-8") as f:
            json.dump(missing_eurovoc_records, f, ensure_ascii=False, indent=2)
        logger.info(
            "  [level=%s] EuroVoc concepts από Multi-EURLEX που ΔΕΝ καλύπτονται από HellasVoc: %d (export: %s)",
            label_level,
            len(missing_eurovoc_records),
            missing_eurovoc_path,
        )

        # HellasVoc labels με EuroVoc URI που ΔΕΝ χρησιμοποιήθηκαν ποτέ σε αυτό το level
        used_hellas_ids = global_used_hellas_ids
        unused_hellas_labels: List[Dict[str, Any]] = []
        for rec in hellas_eurovoc_labels:
            if rec["hellas_id"] not in used_hellas_ids:
                unused_hellas_labels.append(rec)

        unused_hellas_path = output_dir / f"multi_eurlex_{lang}_{label_level}_hellasvoc_labels_never_used.json"
        with unused_hellas_path.open("w", encoding="utf-8") as f:
            json.dump(unused_hellas_labels, f, ensure_ascii=False, indent=2)
        logger.info(
            "  [level=%s] HellasVoc labels με EuroVoc URI που δεν χρησιμοποιήθηκαν: %d (export: %s)",
            label_level,
            len(unused_hellas_labels),
            unused_hellas_path,
        )

        level_stats["eurovoc"] = {
            "num_eurovoc_ids_in_multi_eurlex": len(global_used_eurovoc_ids),
            "num_eurovoc_ids_in_hellasvoc": len(hellas_eurovoc_ids),
            "num_eurovoc_ids_in_multi_not_in_hellas": len(eurovoc_ids_missing_in_hellas),
        }

        level_stats["hellasvoc"] = {
            "num_hellasvoc_labels_with_eurovoc_uri": len(hellas_eurovoc_labels),
            "num_hellasvoc_labels_used_in_multi_eurlex": len(global_used_hellas_ids),
            "num_hellasvoc_labels_never_used_in_multi_eurlex": len(unused_hellas_labels),
        }

        stats["levels"][label_level] = level_stats

    # Τελικό stats file (όλα τα levels)
    stats_path = output_dir / f"multi_eurlex_{lang}_hellasvoc_stats_all_levels.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Στατιστικά για όλα τα levels γραμμένα στο %s", stats_path)
    logger.info("ΤΕΛΟΣ build Multi-EURLEX → HellasVoc για όλα τα label levels.")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    build_multi_eurlex_hellasvoc_dataset(
        HELLASVOC_PATH,
        EUROVOC_DESC_PATH,
        OUTPUT_DIR,
        lang=LANG,
        keep_docs_without_hellas=KEEP_DOCS_WITHOUT_HELLAS,
    )
