import json
import logging
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple, Set, Any

import requests
from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HELLASVOC_PATH = Path("./data/hellasvoc_hierarchy_new.jsonl")
CELEX_METADATA_PATH = Path("./data/celex_metadata.jsonl")

OUTPUT_PATH = Path("./celex_out/celex_hellasvoc_dataset.jsonl")
UNRESOLVED_EU_CONCEPTS_PATH = Path("./celex_out/hellasvoc_eu_concepts_missing_from_sparql.json")
HELLASVOC_UNUSED_LABELS_PATH = Path("./celex_out/hellasvoc_labels_with_eurovoc_uri_not_indexed.json")
# ΝΕΟ: concepts που μοιράζονται 2+ HellasVoc labels
HELLASVOC_SHARED_CONCEPTS_PATH = Path("./celex_out/hellasvoc_shared_concepts.json")

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"


# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------
def parse_eurovoc_string(eurovoc_str: str) -> List[str]:
    if not eurovoc_str:
        return []
    return [t.strip() for t in eurovoc_str.split("|") if t.strip()]


def iter_hellas_nodes(path: Path) -> Iterable[dict]:
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


def extract_eu_concept_id_and_uri(link: str) -> Tuple[str, str]:
    if not isinstance(link, str):
        return "", ""
    link = link.strip()
    if not link:
        return "", ""

    # 1) EuroVoc
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
            uri = link
        uri = uri.split("?", 1)[0].split("#", 1)[0].strip()
        eurovoc_id = uri.rsplit("/", 1)[-1].strip()
        if not eurovoc_id:
            return "", ""
        concept_uri = f"http://eurovoc.europa.eu/{eurovoc_id}"
        return eurovoc_id, concept_uri

    # 2) op.europa wrapper με uri=
    if "op.europa.eu" in link and "uri=" in link:
        parts = link.split("uri=", 1)
        inner = parts[1].split("&", 1)[0].split("#", 1)[0].strip()
        if inner:
            if "eurovoc.europa.eu" in inner:
                eurovoc_id = inner.rsplit("/", 1)[-1].strip()
                concept_uri = f"http://eurovoc.europa.eu/{eurovoc_id}"
                return eurovoc_id, concept_uri
            else:
                concept_uri = inner
                concept_id = concept_uri
                return concept_id, concept_uri

    # 3) data.europa.eu concepts
    if "data.europa.eu" in link:
        concept_uri = link.split("?", 1)[0].split("#", 1)[0].strip()
        concept_id = concept_uri
        return concept_id, concept_uri

    return "", ""


def collect_hellas_eu_uris(path: Path) -> Set[str]:
    uris: Set[str] = set()
    logger.info("Σκανάρισμα HellasVoc για European concept URIs...")

    for node in tqdm(iter_hellas_nodes(path), desc="Scanning HellasVoc nodes"):
        ext_uris = node.get("external_uris", []) or []
        if not isinstance(ext_uris, list):
            continue
        for uri_info in ext_uris:
            if not isinstance(uri_info, dict):
                continue
            link = uri_info.get("link", "") or uri_info.get("uri", "") or uri_info.get("url", "")
            cid, curi = extract_eu_concept_id_and_uri(link)
            if curi:
                uris.add(curi)

    logger.info("Βρέθηκαν %d μοναδικά European concept URIs στο HellasVoc.", len(uris))
    return uris


def build_hellas_eu_index(path: Path) -> Dict[str, Set[Tuple[str, str]]]:
    """
    concept_id -> set( (hellas_id, hellas_label) )
    concept_id:
      - EuroVoc: numeric ID (e.g. "100171")
      - other EU concepts: full URI
    """
    index: Dict[str, Set[Tuple[str, str]]] = {}
    logger.info("Χτίσιμο index concept_id -> HellasVoc nodes...")

    for node in tqdm(iter_hellas_nodes(path), desc="Indexing HellasVoc → EU concept IDs"):
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
            concept_id, concept_uri = extract_eu_concept_id_and_uri(link)
            if not concept_id or not concept_uri:
                continue

            index.setdefault(concept_id, set()).add((hellas_id, hellas_label))

    logger.info("Το HellasVoc περιέχει EU concept IDs για %d concepts.", len(index))
    return index


# ----------------------------------------------------------------------
# SPARQL
# ----------------------------------------------------------------------
def query_eu_concept_by_uri(uri: str, lang: str = "el") -> Optional[Tuple[str, str]]:
    query = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?label ?def WHERE {{
      <{uri}> a skos:Concept ;
              skos:prefLabel ?label .
      FILTER (lang(?label) = "{lang}")
      OPTIONAL {{
        <{uri}> skos:definition ?def .
        FILTER (lang(?def) = "{lang}")
      }}
    }}
    LIMIT 10
    """
    resp = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "application/sparql-results+json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    b = bindings[0]
    label_val = b.get("label", {}).get("value")
    def_val = b.get("def", {}).get("value") if "def" in b else ""
    if not label_val:
        return None
    return label_val, def_val or ""


def build_concept_info_from_hellas_uris(
    uris: Set[str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Set[str]]:
    """
    id_to_info[concept_id] = {id, label, description, uri}
    label_to_id[label_el] = concept_id
    resolved_ids = concept_ids που βρέθηκαν via SPARQL
    """
    id_to_info: Dict[str, Dict[str, str]] = {}
    label_to_id: Dict[str, str] = {}
    resolved_ids: Set[str] = set()

    logger.info("SPARQL queries για EU concept URIs που υπάρχουν στο HellasVoc...")

    for uri in tqdm(sorted(uris), desc="Querying EU concepts (SPARQL)", unit="uri"):
        try:
            res = query_eu_concept_by_uri(uri, lang="el")
        except Exception as e:
            logger.warning("SPARQL error για URI '%s': %s", uri, e)
            continue

        if res is None:
            continue

        label_el, def_el = res

        if "eurovoc.europa.eu" in uri:
            eurovoc_id = uri.rsplit("/", 1)[-1].strip()
            concept_id = eurovoc_id
        else:
            concept_id = uri

        info = {
            "id": concept_id,
            "label": label_el,
            "description": def_el,
            "uri": uri,
        }
        id_to_info[concept_id] = info
        resolved_ids.add(concept_id)

        prev = label_to_id.get(label_el)
        if prev and prev != concept_id:
            logger.debug(
                "Προειδοποίηση: label '%s' αντιστοιχεί σε πολλαπλά IDs (%s, %s)",
                label_el,
                prev,
                concept_id,
            )
        else:
            label_to_id[label_el] = concept_id

    logger.info("Έχουμε %d EU concepts (IDs) με SPARQL info από HellasVoc.", len(id_to_info))
    return id_to_info, label_to_id, resolved_ids


# ----------------------------------------------------------------------
# CELEX helpers
# ----------------------------------------------------------------------
def extract_celex_id(obj: dict) -> str:
    for key in ("celex_id", "celex", "CELEX_ID", "CELEX"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def normalize_concept_id(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def extract_eurovoc_entries_from_celex(
    obj: dict,
    label_to_id: Dict[str, str],
    id_to_info: Dict[str, Dict[str, str]],
) -> List[dict]:
    """
    Επιστρέφει entries:
      [{"id": <concept_id>, "label": <label>, "description": <desc>, "uri": <uri>}, ...]
    """

    entries: List[dict] = []

    def norm_lbl(text: str) -> str:
        return " ".join(text.split()).casefold()

    normalized_label_to_id: Dict[str, str] = {
        norm_lbl(lbl): cid for lbl, cid in label_to_id.items()
    }

    # 1) structured eurovoc list
    if isinstance(obj.get("eurovoc"), list):
        for ev in obj["eurovoc"]:
            if not isinstance(ev, dict):
                continue
            ev_id_raw = ev.get("id") or ev.get("eurovoc_id")
            ev_label = ev.get("label") or ev.get("pref_label")
            info = None

            if ev_id_raw:
                cid = normalize_concept_id(ev_id_raw)
                info = id_to_info.get(cid)

            if info is None and ev_label:
                cid = label_to_id.get(ev_label)
                if not cid:
                    cid = normalized_label_to_id.get(norm_lbl(ev_label))
                if cid:
                    info = id_to_info.get(cid)

            if info:
                entries.append(info)

    # 2) eurovoc_descriptors
    elif isinstance(obj.get("eurovoc_descriptors"), list):
        for ev in obj["eurovoc_descriptors"]:
            if not isinstance(ev, dict):
                continue
            ev_id_raw = ev.get("id") or ev.get("eurovoc_id")
            ev_label = ev.get("label") or ev.get("pref_label")
            info = None

            if ev_id_raw:
                cid = normalize_concept_id(ev_id_raw)
                info = id_to_info.get(cid)

            if info is None and ev_label:
                cid = label_to_id.get(ev_label)
                if not cid:
                    cid = normalized_label_to_id.get(norm_lbl(ev_label))
                if cid:
                    info = id_to_info.get(cid)

            if info:
                entries.append(info)

    # 3) eurovoc_ids μόνο
    elif isinstance(obj.get("eurovoc_ids"), list):
        for ev_id_raw in obj["eurovoc_ids"]:
            cid = normalize_concept_id(ev_id_raw)
            if not cid:
                continue
            info = id_to_info.get(cid)
            if info:
                entries.append(info)

    # 4) fallback: string eurovoc με labels "A | B | C"
    else:
        eurovoc_str = obj.get("eurovoc", "")
        if isinstance(eurovoc_str, str) and eurovoc_str.strip():
            terms = parse_eurovoc_string(eurovoc_str)
            for term in terms:
                cid = label_to_id.get(term)
                if not cid:
                    cid = normalized_label_to_id.get(norm_lbl(term))
                if cid:
                    info = id_to_info.get(cid)
                    if info:
                        entries.append(info)

    unique: Dict[str, dict] = {}
    for ev in entries:
        cid = ev["id"]
        unique[cid] = ev

    return list(unique.values())


# ----------------------------------------------------------------------
# Fallback: συμπλήρωση concepts που ΔΕΝ βρέθηκαν από SPARQL, με labels από HellasVoc
# ----------------------------------------------------------------------
def apply_hellasvoc_fallback_for_missing_concepts(
    eu_index: Dict[str, Set[Tuple[str, str]]],
    id_to_info: Dict[str, Dict[str, str]],
    label_to_id: Dict[str, str],
) -> None:
    existing_ids = set(id_to_info.keys())
    all_ids = set(eu_index.keys())
    missing_ids = sorted(all_ids - existing_ids)

    logger.info(
        "Fallback από HellasVoc για %d EU concepts χωρίς SPARQL info...",
        len(missing_ids),
    )

    for cid in missing_ids:
        hellas_concepts = eu_index.get(cid, set())
        if not hellas_concepts:
            continue

        hellas_concepts_sorted = sorted(hellas_concepts, key=lambda x: (x[0] or "", x[1]))
        primary_label = hellas_concepts_sorted[0][1]

        if cid.isdigit():
            uri = f"http://eurovoc.europa.eu/{cid}"
        else:
            uri = cid

        if cid not in id_to_info:
            id_to_info[cid] = {
                "id": cid,
                "label": primary_label,
                "description": "",
                "uri": uri,
            }

        for _, hlabel in hellas_concepts_sorted:
            if hlabel not in label_to_id:
                label_to_id[hlabel] = cid


# ----------------------------------------------------------------------
# ΝΕΟ: HellasVoc labels με EuroVoc URI που ΔΕΝ εμφανίζονται στο eu_index
# ----------------------------------------------------------------------
def collect_hellasvoc_labels_with_eurovoc_uri(path: Path) -> List[dict]:
    records: List[dict] = []

    for node in iter_hellas_nodes(path):
        hv_id = (str(node.get("id", "")) or "").strip()
        hv_label = (node.get("name") or "").strip()
        if not hv_label:
            continue

        ext_uris = node.get("external_uris", []) or []
        if not isinstance(ext_uris, list):
            continue

        eurovoc_links: List[str] = []
        for uri_info in ext_uris:
            if not isinstance(uri_info, dict):
                continue
            link = uri_info.get("link", "") or uri_info.get("uri", "") or uri_info.get("url", "")
            if not isinstance(link, str):
                continue
            link_s = link.strip()
            if not link_s:
                continue

            if "eurovoc.europa.eu" in link_s:
                eurovoc_links.append(link_s)
            elif ("op.europa.eu" in link_s) and ("uri=http://eurovoc.europa.eu" in link_s):
                eurovoc_links.append(link_s)

        if eurovoc_links:
            records.append(
                {
                    "hellasvoc_id": hv_id,
                    "hellasvoc_label": hv_label,
                    "eurovoc_links": eurovoc_links,
                }
            )

    return records


def export_hellasvoc_labels_not_indexed(
    hellas_path: Path,
    eu_index: Dict[str, Set[Tuple[str, str]]],
    out_path: Path,
) -> None:
    logger.info("Εξαγωγή HellasVoc labels με EuroVoc URI που δεν χρησιμοποιούνται από το eu_index...")

    eurovoc_labels = collect_hellasvoc_labels_with_eurovoc_uri(hellas_path)
    total_labels_with_eurovoc = len(eurovoc_labels)

    used_ids: Set[str] = set()
    for concept_id, items in eu_index.items():
        for hid, _ in items:
            used_ids.add(hid)

    unused_records: List[dict] = []
    for rec in eurovoc_labels:
        hv_id = rec["hellasvoc_id"]
        if hv_id not in used_ids:
            unused_records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(unused_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "HellasVoc labels με EuroVoc URI (σύνολο): %d",
        total_labels_with_eurovoc,
    )
    logger.info(
        "HellasVoc labels με EuroVoc URI που χρησιμοποιούνται από το eu_index: %d",
        total_labels_with_eurovoc - len(unused_records),
    )
    logger.info(
        "HellasVoc labels με EuroVoc URI που ΔΕΝ χρησιμοποιούνται από το eu_index: %d",
        len(unused_records),
    )
    logger.info("Γράφτηκαν στο %s", out_path)


# ----------------------------------------------------------------------
# ΝΕΟ: export concepts που έχουν 2+ HellasVoc labels
# ----------------------------------------------------------------------
def export_shared_concepts_with_multiple_hellas_labels(
    eu_index: Dict[str, Set[Tuple[str, str]]],
    id_to_info: Dict[str, Dict[str, str]],
    out_path: Path,
) -> None:
    """
    Βγάζει JSON με όλα τα concepts (concept_id) που έχουν 2+ HellasVoc labels.
    """
    logger.info("Εξαγωγή concepts που αντιστοιχούν σε 2+ HellasVoc labels...")

    shared_records: List[dict] = []

    for cid, hellas_set in eu_index.items():
        if len(hellas_set) < 2:
            continue

        hellas_list = [
            {"id": hid, "label": hlabel}
            for (hid, hlabel) in sorted(hellas_set, key=lambda x: (x[0] or "", x[1]))
        ]

        info = id_to_info.get(cid, {})
        uri = info.get("uri")
        label = info.get("label")
        description = info.get("description")

        if not uri:
            if cid.isdigit():
                uri = f"http://eurovoc.europa.eu/{cid}"
            else:
                uri = cid

        if cid.isdigit():
            eurovoc_id = cid
        else:
            eurovoc_id = None

        shared_records.append(
            {
                "concept_id": cid,
                "eurovoc_id": eurovoc_id,
                "concept_uri": uri,
                "label_from_id_to_info": label,
                "description_from_id_to_info": description,
                "hellasvoc_concepts": hellas_list,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(shared_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "Βρέθηκαν %d concepts που μοιράζονται 2+ HellasVoc labels. Γράφτηκαν στο %s",
        len(shared_records),
        out_path,
    )


# ----------------------------------------------------------------------
# Main dataset builder
# ----------------------------------------------------------------------
def build_celex_hellasvoc_dataset(
    hellas_path: Path,
    celex_path: Path,
    output_path: Path,
    unresolved_output_path: Path,
    keep_docs_without_hellas: bool = False,
) -> None:
    logger.info("Ξεκινά ενιαίο χτίσιμο CELEX → EU concepts → HellasVoc dataset.")

    # 1) HellasVoc: EU concept URIs + index concept_id → HellasVoc labels
    eu_uris = collect_hellas_eu_uris(hellas_path)
    eu_index = build_hellas_eu_index(hellas_path)

    # Export labels με EuroVoc URI που δεν μπήκαν στο eu_index (αναμένεται 0)
    export_hellasvoc_labels_not_indexed(hellas_path, eu_index, HELLASVOC_UNUSED_LABELS_PATH)

    # 2) SPARQL
    id_to_info, label_to_id, resolved_ids = build_concept_info_from_hellas_uris(eu_uris)
    logger.info(
        "HELLASVOC EU concepts με SPARQL info: %d (αναμένουμε ~509 EuroVoc concepts).",
        len(id_to_info),
    )

    # 3) Fallback: συμπλήρωση concepts που δεν ήρθαν από SPARQL με labels από HellasVoc
    apply_hellasvoc_fallback_for_missing_concepts(eu_index, id_to_info, label_to_id)

    # ΝΕΟ: export concepts που μοιράζονται 2+ HellasVoc labels
    export_shared_concepts_with_multiple_hellas_labels(
        eu_index, id_to_info, HELLASVOC_SHARED_CONCEPTS_PATH
    )

    # Μετά το fallback, ό,τι ΔΕΝ υπάρχει στο id_to_info είναι πραγματικά “ορφανό”
    all_concept_ids_from_hellas: Set[str] = set(eu_index.keys())
    missing_after_fallback: List[str] = sorted(all_concept_ids_from_hellas - set(id_to_info.keys()))

    logger.info(
        "Μετά το fallback, concepts στο HellasVoc: %d, concepts με info: %d, unmatched: %d",
        len(all_concept_ids_from_hellas),
        len(id_to_info),
        len(missing_after_fallback),
    )

    # 4) Export unresolved concepts (μετά από SPARQL + fallback)
    unresolved_records: List[dict] = []
    for cid in missing_after_fallback:
        hellas_concepts = [
            {"id": hid, "label": hlabel}
            for (hid, hlabel) in sorted(
                eu_index.get(cid, set()), key=lambda x: (x[0] or "", x[1])
            )
        ]

        if cid.isdigit():
            uri = f"http://eurovoc.europa.eu/{cid}"
            eurovoc_id = cid
        else:
            uri = cid
            eurovoc_id = None

        unresolved_records.append(
            {
                "concept_id": cid,
                "eurovoc_id": eurovoc_id,
                "concept_uri": uri,
                "hellasvoc_concepts": hellas_concepts,
            }
        )

    unresolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with unresolved_output_path.open("w", encoding="utf-8") as f:
        json.dump(unresolved_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "Γράφτηκαν %d 'unmatched' EU concepts (μετά από SPARQL + fallback) στο %s",
        len(unresolved_records),
        unresolved_output_path,
    )

    # 5) CELEX πλευρά
    n_in = 0
    n_out = 0
    n_no_eurovoc = 0
    n_no_hellas = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with celex_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        logger.info("Επεξεργασία CELEX metadata και δημιουργία τελικού dataset...")

        for line in tqdm(fin, desc="Processing CELEX metadata", unit="lines"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            celex_id = extract_celex_id(obj)
            if not celex_id:
                continue

            eurovoc_entries = extract_eurovoc_entries_from_celex(obj, label_to_id, id_to_info)
            if not eurovoc_entries:
                n_no_eurovoc += 1
                continue

            n_in += 1

            hellas_terms_set: Set[Tuple[str, str]] = set()
            for ev in eurovoc_entries:
                cid = normalize_concept_id(ev.get("id"))
                if not cid:
                    continue
                matches = eu_index.get(cid)
                if matches:
                    hellas_terms_set.update(matches)

            if not hellas_terms_set:
                n_no_hellas += 1
                if not keep_docs_without_hellas:
                    continue

            hellasvoc_list = [
                {"id": hid, "label": hlabel}
                for (hid, hlabel) in sorted(
                    hellas_terms_set, key=lambda x: (x[0] or "", x[1])
                )
            ]

            record = {
                "celex_id": celex_id,
                "eurovoc": eurovoc_entries,
                "hellasvoc": hellasvoc_list,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    logger.info("[SUMMARY] CELEX records με EuroVoc/EU concept entries: %d", n_in)
    logger.info("[SUMMARY] CELEX records χωρίς EuroVoc info: %d", n_no_eurovoc)
    logger.info(
        "[SUMMARY] CELEX records με EuroVoc/EU concepts αλλά ΧΩΡΙΣ HellasVoc mapping: %d",
        n_no_hellas,
    )
    logger.info(
        "[SUMMARY] Γράφτηκαν %d CELEX records με HellasVoc mappings στο %s",
        n_out,
        output_path,
    )


if __name__ == "__main__":
    build_celex_hellasvoc_dataset(
        HELLASVOC_PATH,
        CELEX_METADATA_PATH,
        OUTPUT_PATH,
        UNRESOLVED_EU_CONCEPTS_PATH,
        keep_docs_without_hellas=False,
    )
