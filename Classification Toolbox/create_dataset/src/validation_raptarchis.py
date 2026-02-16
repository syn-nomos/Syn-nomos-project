import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

from tqdm.auto import tqdm


# ---------------- Normalization helpers ----------------

def normalize_for_match(text: str) -> str:
    """
    Κανονικοποίηση για σύγκριση labels (HellasVoc vs volume/chapter/subject):

    - strip
    - σε upper
    - συμπίεση πολλαπλών spaces σε ένα
    - αφαίρεση τελικών .;:·
    (Κρατάμε τόνους/diacritics όπως είναι, απλώς ελπίζουμε ότι ταιριάζουν.)
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


# ---------------- Index Greek Legal Code labels ----------------

def index_glc_labels(
    glc_root: str,
    train_name: str = "train",
    dev_name: str = "dev",
    test_name: str = "test",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Διαβάζει train/dev/test από glc_root και χτίζει index:

      (level, normalized_label) ->
        {
          "level": level,  # 'volume' / 'chapter' / 'subject'
          "normalized": normalized_label,
          "original_strings": set([...]),
          "doc_ids_by_split": {
              "train": set([...]),
              "dev": set([...]),
              "test": set([...]),
          }
        }

    level ∈ {'volume', 'chapter', 'subject'}
    """

    label_index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    splits = {
        "train": os.path.join(glc_root, train_name),
        "dev": os.path.join(glc_root, dev_name),
        "test": os.path.join(glc_root, test_name),
    }

    for split_name, split_dir in splits.items():
        file_paths: List[str] = []
        for root, _, files in os.walk(split_dir):
            for fname in files:
                if fname.endswith(".json"):
                    file_paths.append(os.path.join(root, fname))
        file_paths = sorted(file_paths)
        print(f"[INFO] {split_name}: found {len(file_paths)} JSON files.")

        for fpath in tqdm(file_paths, desc=f"Indexing {split_name}", unit="doc"):
            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARN] Could not decode JSON: {fpath}")
                    continue

            rel_id = os.path.join(split_name, os.path.relpath(fpath, split_dir))

            for level in ("volume", "chapter", "subject"):
                val = doc.get(level)
                if not isinstance(val, str) or not val.strip():
                    continue
                norm = normalize_for_match(val)
                if not norm:
                    continue

                key = (level, norm)
                if key not in label_index:
                    label_index[key] = {
                        "level": level,
                        "normalized": norm,
                        "original_strings": set(),
                        "doc_ids_by_split": {"train": set(), "dev": set(), "test": set()},
                    }

                label_index[key]["original_strings"].add(val)
                label_index[key]["doc_ids_by_split"][split_name].add(rel_id)

    # Μετατροπή sets -> sorted lists για πιο εύκολο serialization αργότερα.
    for key, info in label_index.items():
        info["original_strings"] = sorted(info["original_strings"])
        for split_name in ("train", "dev", "test"):
            info["doc_ids_by_split"][split_name] = sorted(info["doc_ids_by_split"][split_name])

    print(f"[INFO] Built index for {len(label_index)} (level, label) combinations.")
    return label_index


# ---------------- Load unmatched HellasVoc labels ----------------

def load_unmatched_hellasvoc(unmatched_path: str) -> List[Dict[str, Any]]:
    """
    Διαβάζει το hellasvoc_unmatched_labels.json:

    Αναμένεται κάτι της μορφής:
      [
        {
          "hellasvoc_id": "...",
          "hellasvoc_name": "...",
          "raptarchis_labels": [...],
          "raptarchis_urls": [...]
        },
        ...
      ]
    """
    with open(unmatched_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} unmatched HellasVoc records from {unmatched_path}.")
    return data


# ---------------- Validation logic ----------------

def validate_unmatched_against_glc(
    unmatched_records: List[Dict[str, Any]],
    glc_label_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Για κάθε unmatched HellasVoc label:

    - Χτίζει ένα set από candidate strings:
        - hellasvoc_name
        - όλα τα raptarchis_labels
    - Τα κανονικοποιεί (normalize_for_match)
    - Για κάθε επίπεδο (subject/chapter/volume)
      προσπαθεί να βρει match στο glc_label_index.

    Επιστρέφει ένα dict με:
      {
        "summary": {...},
        "hellasvoc_matches": [
          {
            "hellasvoc_id": ...,
            "hellasvoc_name": ...,
            "raptarchis_labels": [...],
            "raptarchis_urls": [...],
            "matches": [
              {
                "level": "subject" | "chapter" | "volume",
                "normalized_key": "...",
                "glc_labels": [ ... distinct original strings ... ],
                "occurrences": {
                  "train": int,
                  "dev": int,
                  "test": int,
                  "total": int
                }
              },
              ...
            ]
          },
          ...
        ]
      }
    """

    results: List[Dict[str, Any]] = []
    num_with_any_match = 0

    for rec in tqdm(unmatched_records, desc="Validating unmatched HellasVoc"):
        hv_id = rec.get("hellasvoc_id")
        hv_name = rec.get("hellasvoc_name")
        r_labels = rec.get("raptarchis_labels", []) or []
        r_urls = rec.get("raptarchis_urls", []) or []

        # Συλλογή candidate strings
        candidates: Set[str] = set()
        if isinstance(hv_name, str) and hv_name.strip():
            candidates.add(hv_name)
        for lbl in r_labels:
            if isinstance(lbl, str) and lbl.strip():
                candidates.add(lbl)

        # Κανονικοποίηση
        norm_candidates: Set[str] = set()
        for s in candidates:
            n = normalize_for_match(s)
            if n:
                norm_candidates.add(n)

        matches_for_rec: List[Dict[str, Any]] = []

        for norm_s in norm_candidates:
            for level in ("volume", "chapter", "subject"):
                key = (level, norm_s)
                if key not in glc_label_index:
                    continue
                info = glc_label_index[key]

                # Υπολογίζουμε counts ανά split
                occ_train = len(info["doc_ids_by_split"]["train"])
                occ_dev = len(info["doc_ids_by_split"]["dev"])
                occ_test = len(info["doc_ids_by_split"]["test"])
                occ_total = occ_train + occ_dev + occ_test

                matches_for_rec.append(
                    {
                        "level": level,
                        "normalized_key": norm_s,
                        "glc_labels": info["original_strings"],
                        "occurrences": {
                            "train": occ_train,
                            "dev": occ_dev,
                            "test": occ_test,
                            "total": occ_total,
                        },
                    }
                )

        if matches_for_rec:
            num_with_any_match += 1

        results.append(
            {
                "hellasvoc_id": hv_id,
                "hellasvoc_name": hv_name,
                "raptarchis_labels": r_labels,
                "raptarchis_urls": r_urls,
                "matches": matches_for_rec,
            }
        )

    summary = {
        "num_unmatched_input": len(unmatched_records),
        "num_with_any_match_in_glc_hierarchy": num_with_any_match,
        "num_without_match_in_glc_hierarchy": len(unmatched_records) - num_with_any_match,
    }

    return {
        "summary": summary,
        "hellasvoc_matches": results,
    }


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Validation script: ελέγχει τα unmatched HellasVoc labels "
                    "έναντι των volume/chapter/subject του Greek Legal Code."
    )
    parser.add_argument(
        "--glc_root",
        required=True,
        help="Root folder του Greek Legal Code που περιέχει τα subfolders train/dev/test."
    )
    parser.add_argument(
        "--unmatched_hellasvoc",
        required=True,
        help="Path στο hellasvoc_unmatched_labels.json (output από το προηγούμενο βήμα)."
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path για export του validation JSON."
    )
    parser.add_argument(
        "--train_name",
        default="train",
        help="Όνομα υποφακέλου για το train split (default: train)"
    )
    parser.add_argument(
        "--dev_name",
        default="dev",
        help="Όνομα υποφακέλου για το dev split (default: dev)"
    )
    parser.add_argument(
        "--test_name",
        default="test",
        help="Όνομα υποφακέλου για το test split (default: test)"
    )

    args = parser.parse_args()

    # 1. Χτίζουμε index από volume/chapter/subject των splits
    label_index = index_glc_labels(
        glc_root=args.glc_root,
        train_name=args.train_name,
        dev_name=args.dev_name,
        test_name=args.test_name,
    )

    # 2. Φορτώνουμε unmatched HellasVoc labels
    unmatched_records = load_unmatched_hellasvoc(args.unmatched_hellasvoc)

    # 3. Validation
    validation_result = validate_unmatched_against_glc(
        unmatched_records, label_index
    )

    # 4. Export
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(validation_result, f, ensure_ascii=False, indent=2)

    print("[INFO] Validation summary:")
    print(json.dumps(validation_result["summary"], ensure_ascii=False, indent=2))
    print(f"[INFO] Full validation written to: {args.output_json}")


if __name__ == "__main__":
    main()
