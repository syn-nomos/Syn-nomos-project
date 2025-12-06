import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Any

from tqdm.auto import tqdm


# ---------- Helpers ----------

def normalize_url(url: str) -> str:
    """
    Κανονικοποίηση URL Ραπτάρχη:
    - strip spaces
    - lower()
    - κόβει trailing slash
    """
    if not url:
        return ""
    u = url.strip()
    if u.endswith("/"):
        u = u[:-1]
    return u.lower()


def normalize_label(label: str) -> str:
    """
    Απλό normalization για raw labels του Ραπτάρχη:
    - strip spaces
    (ΔΕΝ αλλάζουμε case/τονισμούς)
    """
    if label is None:
        return ""
    return label.strip()


def normalize_for_match(text: str) -> str:
    """
    Κανονικοποίηση για σύγκριση labels (HellasVoc / Raptarchis vs subject/chapter/volume):

    - strip
    - σε upper
    - συμπίεση πολλαπλών spaces σε ένα
    - αφαίρεση τελικών .;:·
    """
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:·")
    return s


def load_raptarchis_mappings(path: str):
    """
    Διαβάζει το hellasvoc_raptarchis_labels_new.jsonl

    Κάθε γραμμή:
      - 'hellasvoc_id'
      - 'raptarchis_label'
      - 'raptarchis_url'

    Επιστρέφει:
      1) url_to_rows:
           normalized_url -> list of { 'hellasvoc_id', 'raptarchis_label' }
      2) label_to_hv_ids:
           normalized_raptarchis_label (strip only) -> list of hellasvoc_id
      3) hv_id_to_meta:
           hellasvoc_id -> {
               'hellasvoc_id': str,
               'raptarchis_labels': [..],
               'raptarchis_urls': [..]
           }
    """

    url_to_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    label_to_hv_ids_tmp: Dict[str, Set[str]] = defaultdict(set)
    hv_id_info_tmp: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"hellasvoc_id": None, "raptarchis_labels": set(), "raptarchis_urls": set()}
    )

    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Loading hellasvoc_raptarchis"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            hv_id = str(obj.get("hellasvoc_id"))
            r_label = obj.get("raptarchis_label") or ""
            url = obj.get("raptarchis_url") or ""

            if not hv_id:
                continue

            nurl = normalize_url(url)
            nlabel = normalize_label(r_label)

            # URL -> rows
            if nurl:
                url_to_rows[nurl].append(
                    {
                        "hellasvoc_id": hv_id,
                        "raptarchis_label": r_label,
                    }
                )

            # label -> hv_ids (raw, strip-only)
            if nlabel:
                label_to_hv_ids_tmp[nlabel].add(hv_id)

            # hv_id -> meta (συγκεντρωτικά)
            info = hv_id_info_tmp[hv_id]
            info["hellasvoc_id"] = hv_id
            if nlabel:
                info["raptarchis_labels"].add(r_label)
            if nurl:
                info["raptarchis_urls"].add(url)

    label_to_hv_ids: Dict[str, List[str]] = {
        lbl: sorted(list(hv_ids)) for lbl, hv_ids in label_to_hv_ids_tmp.items()
    }

    hv_id_to_meta: Dict[str, Dict[str, Any]] = {}
    for hv_id, info in hv_id_info_tmp.items():
        hv_id_to_meta[hv_id] = {
            "hellasvoc_id": hv_id,
            "raptarchis_labels": sorted(list(info["raptarchis_labels"])),
            "raptarchis_urls": sorted(list(info["raptarchis_urls"])),
        }

    return url_to_rows, label_to_hv_ids, hv_id_to_meta


def flatten_hellasvoc_hierarchy(path: str) -> Dict[str, str]:
    """
    Διαβάζει το hellasvoc_hierarchy_new.jsonl και χτίζει
    ένα index: hellasvoc_id -> name (label name).
    """

    id_to_name: Dict[str, str] = {}

    def visit_node(node: Dict[str, Any], depth: int = 0):
        node_id = str(node.get("id"))
        node_name = node.get("name", "")
        if node_id:
            id_to_name[node_id] = node_name
        for child in node.get("children", []):
            visit_node(child, depth + 1)

    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Loading HellasVoc hierarchy"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            visit_node(obj)

    return id_to_name


def build_match_label_to_hv_ids(
    hv_id_to_meta: Dict[str, Dict[str, Any]],
    hellasvoc_id_to_name: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Φτιάχνει mapping για matching με βάση subject/chapter/volume:

      match_label_to_hv_ids:
        normalize_for_match(label_text) -> set(hellasvoc_id)

    label_text ∈:
      - hellasvoc_name
      - όλα τα raptarchis_labels
    """

    match_map: Dict[str, Set[str]] = defaultdict(set)

    for hv_id, meta in hv_id_to_meta.items():
        hv_name = hellasvoc_id_to_name.get(hv_id)
        if isinstance(hv_name, str) and hv_name.strip():
            n = normalize_for_match(hv_name)
            if n:
                match_map[n].add(hv_id)

        for lbl in meta.get("raptarchis_labels", []):
            if isinstance(lbl, str) and lbl.strip():
                n = normalize_for_match(lbl)
                if n:
                    match_map[n].add(hv_id)

    return match_map


def extract_raptarchis_urls_from_doc(doc: Dict[str, Any]) -> List[str]:
    urls: List[str] = []

    if isinstance(doc.get("raptarchis_url"), str) and doc["raptarchis_url"]:
        urls.append(doc["raptarchis_url"])

    if isinstance(doc.get("raptarchis_urls"), list):
        urls.extend(
            [u for u in doc["raptarchis_urls"] if isinstance(u, str) and u]
        )

    if isinstance(doc.get("leg_uri"), str) and doc["leg_uri"].startswith("http"):
        urls.append(doc["leg_uri"])

    if isinstance(doc.get("law_id"), str) and doc["law_id"].startswith("http"):
        urls.append(doc["law_id"])

    urls_norm = []
    seen = set()
    for u in urls:
        nu = normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu)
            urls_norm.append(nu)

    return urls_norm


def extract_raptarchis_labels_from_doc(
    doc: Dict[str, Any],
    label_to_hv_ids: Dict[str, List[str]],
) -> Set[str]:
    """
    Εξάγει hv_ids με βάση Raptarchis label text από explicit fields + metadata.
    """

    found_ids: Set[str] = set()

    possible_keys = ["raptarchis_label", "raptarchis_labels"]
    for key in possible_keys:
        if key in doc:
            val = doc[key]
            if isinstance(val, str):
                cand = normalize_label(val)
                if cand in label_to_hv_ids:
                    found_ids.update(label_to_hv_ids[cand])
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        cand = normalize_label(item)
                        if cand in label_to_hv_ids:
                            found_ids.update(label_to_hv_ids[cand])

    if found_ids:
        return found_ids

    skip_keys = {"title", "header", "articles", "text", "body", "content"}

    def rec(obj: Any, parent_key: str = None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in skip_keys:
                    continue
                rec(v, k)
        elif isinstance(obj, list):
            for v in obj:
                rec(v, parent_key)
        elif isinstance(obj, str):
            cand = normalize_label(obj)
            if cand in label_to_hv_ids:
                found_ids.update(label_to_hv_ids[cand])

    rec(doc)
    return found_ids


def build_text_from_doc(doc: Dict[str, Any]) -> str:
    title = doc.get("title", "") or ""
    header = doc.get("header", "") or ""
    articles = doc.get("articles", []) or []
    if not isinstance(articles, list):
        articles = []

    parts = [title.strip(), header.strip()] + [
        a.strip() for a in articles if isinstance(a, str)
    ]
    parts = [p for p in parts if p]
    return "\n\n".join(parts)


# ---------- Main processing per split ----------

def process_split(
    split_name: str,
    split_dir: str,
    raptarchis_url_to_rows: Dict[str, List[Dict[str, str]]],
    raptarchis_label_to_hv_ids: Dict[str, List[str]],
    match_label_to_hv_ids: Dict[str, Set[str]],
    hellasvoc_id_to_name: Dict[str, str],
    output_jsonl_path: str,
    unlabeled_jsonl_path: str,
    stats_accumulator: Dict[str, Any],
):
    num_docs = 0
    labels_in_split: Set[str] = set()
    labels_per_doc_counts: List[int] = []
    docs_with_no_labels = 0

    file_paths: List[str] = []
    for root, _, files in os.walk(split_dir):
        for fname in files:
            if fname.endswith(".json"):
                file_paths.append(os.path.join(root, fname))

    file_paths = sorted(file_paths)
    print(f"[INFO] {split_name}: found {len(file_paths)} JSON files.")

    with open(output_jsonl_path, "w", encoding="utf-8") as out_f, \
         open(unlabeled_jsonl_path, "w", encoding="utf-8") as unl_f:

        for fpath in tqdm(file_paths, desc=f"Processing {split_name}", unit="doc"):
            with open(fpath, "r", encoding="utf-8") as in_f:
                try:
                    doc = json.load(in_f)
                except json.JSONDecodeError:
                    print(f"[WARN] Could not decode JSON: {fpath}")
                    continue

            num_docs += 1
            rel_path = os.path.relpath(fpath, split_dir)
            text = build_text_from_doc(doc)

            doc_labels_ids: Set[str] = set()

            # 1) URLs
            doc_urls = extract_raptarchis_urls_from_doc(doc)
            for nurl in doc_urls:
                if nurl in raptarchis_url_to_rows:
                    for row in raptarchis_url_to_rows[nurl]:
                        hv_id = str(row["hellasvoc_id"])
                        if hv_id:
                            doc_labels_ids.add(hv_id)

            # 2) Raptarchis label text
            label_based_ids = extract_raptarchis_labels_from_doc(
                doc, raptarchis_label_to_hv_ids
            )
            doc_labels_ids.update(label_based_ids)

            # 3) subject / chapter / volume
            for level in ("subject", "chapter", "volume"):
                val = doc.get(level)
                if isinstance(val, str) and val.strip():
                    # 3.a) normalized_for_match -> match_label_to_hv_ids (HellasVoc names + Raptarchis labels)
                    n_match = normalize_for_match(val)
                    if n_match in match_label_to_hv_ids:
                        doc_labels_ids.update(match_label_to_hv_ids[n_match])

                    # 3.b) strip-only normalize_label -> direct match με raptarchis_label_to_hv_ids
                    cand_raw = normalize_label(val)
                    if cand_raw in raptarchis_label_to_hv_ids:
                        doc_labels_ids.update(raptarchis_label_to_hv_ids[cand_raw])

            # Names
            doc_labels_names: Set[str] = set()
            for hv_id in doc_labels_ids:
                name = hellasvoc_id_to_name.get(hv_id)
                if name:
                    doc_labels_names.add(name)

            labels_count = len(doc_labels_ids)
            labels_per_doc_counts.append(labels_count)
            if labels_count == 0:
                docs_with_no_labels += 1

            labels_in_split.update(doc_labels_ids)

            meta = dict(doc)

            output_obj = {
                "doc_id": rel_path,
                "split": split_name,
                "text": text,
                "hellasvoc_labels": sorted(list(doc_labels_ids)),
                "hellasvoc_labels_names": sorted(list(doc_labels_names)),
                "raptarchis_urls": doc_urls,
                "meta": meta,
            }

            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

            if labels_count == 0:
                unl_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

    if labels_per_doc_counts:
        avg_labels = sum(labels_per_doc_counts) / len(labels_per_doc_counts)
        min_labels = min(labels_per_doc_counts)
        max_labels = max(labels_per_doc_counts)
    else:
        avg_labels = 0.0
        min_labels = 0
        max_labels = 0

    stats_accumulator["splits"][split_name] = {
        "num_docs": num_docs,
        "num_unique_labels": len(labels_in_split),
        "avg_labels_per_doc": avg_labels,
        "min_labels_per_doc": min_labels,
        "max_labels_per_doc": max_labels,
        "num_docs_with_no_labels": docs_with_no_labels,
        "labels_in_split": sorted(list(labels_in_split)),
    }

    return labels_in_split


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Δημιουργία annotated Greek Legal Code dataset με labels HellasVoc "
                    "μέσω Ραπτάρχη (URI + label text) + subject/chapter/volume matching."
    )
    parser.add_argument("--glc_root", required=True)
    parser.add_argument("--hellasvoc_raptarchis", required=True)
    parser.add_argument("--hellasvoc_hierarchy", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_name", default="train")
    parser.add_argument("--dev_name", default="dev")
    parser.add_argument("--test_name", default="test")

    args = parser.parse_args()

    glc_root = args.glc_root
    hv_raptarchis_path = args.hellasvoc_raptarchis
    hv_hierarchy_path = args.hellasvoc_hierarchy
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading Raptarchis -> HellasVoc mappings (URL + label)...")
    (
        raptarchis_url_to_rows,
        raptarchis_label_to_hv_ids,
        hv_id_to_meta,
    ) = load_raptarchis_mappings(hv_raptarchis_path)
    print(f"[INFO] Loaded {len(raptarchis_url_to_rows)} distinct Raptarchis URLs.")
    print(f"[INFO] Loaded {len(raptarchis_label_to_hv_ids)} distinct Raptarchis labels.")
    print(f"[INFO] Loaded {len(hv_id_to_meta)} HellasVoc IDs from mapping.")

    print("[INFO] Loading HellasVoc hierarchy (id -> name)...")
    hellasvoc_id_to_name = flatten_hellasvoc_hierarchy(hv_hierarchy_path)
    print(f"[INFO] Loaded {len(hellasvoc_id_to_name)} HellasVoc ids with names.")

    print("[INFO] Building match_label_to_hv_ids (HellasVoc names + Raptarchis labels)...")
    match_label_to_hv_ids = build_match_label_to_hv_ids(
        hv_id_to_meta, hellasvoc_id_to_name
    )
    print(f"[INFO] Built {len(match_label_to_hv_ids)} normalized label keys for matching.")

    stats: Dict[str, Any] = {
        "splits": {},
        "label_overlap": {},
        "global": {},
    }

    train_dir = os.path.join(glc_root, args.train_name)
    dev_dir = os.path.join(glc_root, args.dev_name)
    test_dir = os.path.join(glc_root, args.test_name)

    train_out = os.path.join(output_dir, "hellasvoc_glc_train.jsonl")
    dev_out = os.path.join(output_dir, "hellasvoc_glc_dev.jsonl")
    test_out = os.path.join(output_dir, "hellasvoc_glc_test.jsonl")

    train_unlabeled_out = os.path.join(output_dir, "hellasvoc_glc_train_unlabeled.jsonl")
    dev_unlabeled_out = os.path.join(output_dir, "hellasvoc_glc_dev_unlabeled.jsonl")
    test_unlabeled_out = os.path.join(output_dir, "hellasvoc_glc_test_unlabeled.jsonl")

    print("[INFO] Processing train split...")
    train_labels = process_split(
        "train",
        train_dir,
        raptarchis_url_to_rows,
        raptarchis_label_to_hv_ids,
        match_label_to_hv_ids,
        hellasvoc_id_to_name,
        train_out,
        train_unlabeled_out,
        stats,
    )

    print("[INFO] Processing dev split...")
    dev_labels = process_split(
        "dev",
        dev_dir,
        raptarchis_url_to_rows,
        raptarchis_label_to_hv_ids,
        match_label_to_hv_ids,
        hellasvoc_id_to_name,
        dev_out,
        dev_unlabeled_out,
        stats,
    )

    print("[INFO] Processing test split...")
    test_labels = process_split(
        "test",
        test_dir,
        raptarchis_url_to_rows,
        raptarchis_label_to_hv_ids,
        match_label_to_hv_ids,
        hellasvoc_id_to_name,
        test_out,
        test_unlabeled_out,
        stats,
    )

    labels_all = train_labels | dev_labels | test_labels

    only_train = train_labels - dev_labels - test_labels
    only_dev = dev_labels - train_labels - test_labels
    only_test = test_labels - train_labels - dev_labels

    dev_not_train = dev_labels - train_labels
    test_not_train = test_labels - train_labels
    zero_shot_labels = (dev_labels | test_labels) - train_labels

    stats["label_overlap"] = {
        "labels_only_in_train": sorted(list(only_train)),
        "labels_only_in_dev": sorted(list(only_dev)),
        "labels_only_in_test": sorted(list(only_test)),
        "num_labels_only_in_train": len(only_train),
        "num_labels_only_in_dev": len(only_dev),
        "num_labels_only_in_test": len(only_test),
        "labels_in_dev_not_in_train": sorted(list(dev_not_train)),
        "labels_in_test_not_in_train": sorted(list(test_not_train)),
        "num_labels_in_dev_not_in_train": len(dev_not_train),
        "num_labels_in_test_not_in_train": len(test_not_train),
        "zero_shot_labels_wrt_train": sorted(list(zero_shot_labels)),
        "num_zero_shot_labels_wrt_train": len(zero_shot_labels),
    }

    stats["global"] = {
        "num_all_labels": len(labels_all),
        "all_labels": sorted(list(labels_all)),
    }

    stats_path = os.path.join(output_dir, "hellasvoc_glc_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Stats written to: {stats_path}")

    all_hv_ids_from_mapping = set(hv_id_to_meta.keys())
    unmatched_ids = sorted(list(all_hv_ids_from_mapping - labels_all))

    unmatched_records: List[Dict[str, Any]] = []
    for hv_id in unmatched_ids:
        meta = hv_id_to_meta.get(
            hv_id,
            {"hellasvoc_id": hv_id, "raptarchis_labels": [], "raptarchis_urls": []},
        )
        rec = {
            "hellasvoc_id": hv_id,
            "hellasvoc_name": hellasvoc_id_to_name.get(hv_id),
            "raptarchis_labels": meta.get("raptarchis_labels", []),
            "raptarchis_urls": meta.get("raptarchis_urls", []),
        }
        unmatched_records.append(rec)

    unmatched_path = os.path.join(output_dir, "hellasvoc_unmatched_labels.json")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        json.dump(unmatched_records, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Unmatched HellasVoc labels written to: {unmatched_path}")

    print(f"[INFO] Train JSONL: {train_out}")
    print(f"[INFO] Dev   JSONL: {dev_out}")
    print(f"[INFO] Test  JSONL: {test_out}")
    print(f"[INFO] Train unlabeled JSONL: {train_unlabeled_out}")
    print(f"[INFO] Dev   unlabeled JSONL: {dev_unlabeled_out}")
    print(f"[INFO] Test  unlabeled JSONL: {test_unlabeled_out}")


if __name__ == "__main__":
    main()
