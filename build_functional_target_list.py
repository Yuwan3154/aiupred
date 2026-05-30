#!/usr/bin/env python3
"""
Build a functional-relevance target list from v6 AFDB datasets.

Pipeline:
  Stage 1: AFDB-based pre-filter (low masked_mean_plddt AND long consecutive
           ordered region).
  Stage 2: UniProt batch enrichment (protein_existence, annotation_score,
           xref_pdb, xref_pfam, xref_interpro, go_*, cc_disease, keyword,
           lit_pubmed_id, cc_subcellular_location, ec).
  Stage 3: DisProt enrichment (disorder_content).
  Stage 4: Hard filter on PDB cross-reference absence; write outputs.

masked_mean_plddt in the v6 CSVs was computed with disorder_threshold=0.5
(process_afdb.py: order_mask = np.array(disorder_score) <= 0.5).
Using --disorder_threshold 0.5 (the default) reuses that precomputed column
directly (faster: no plddt array parsing). Any other value recomputes it
from the per-residue arrays and will be somewhat slower.

Usage:
    python build_functional_target_list.py \\
        --data_dir /home/jupyter-chenxi/data/afdb \\
        --output_dir /home/jupyter-chenxi/data/afdb \\
        --cache_dir /home/jupyter-chenxi/data/afdb/.uniprot_cache \\
        [--masked_plddt_cutoff 0.7] [--ordered_run_min 50] \\
        [--disorder_threshold 0.5] [--max_len -1]

For faster reruns build precomputed stats Parquets first:
    python precompute_afdb_stats.py --data_dir ... [--thresholds 0.3 0.5 0.7]
Then pass --stats_dir to skip per-residue array parsing entirely in Stage 1.
"""

import argparse
import ast
import json
import os
import re
import time
from io import StringIO

import numpy as np
import requests

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd

# ---------- Dataset registry (mirrors build_filtered_target_list.py) ----------

MODEL_ORGANISM_DATASETS = [
    ("UP000005640_9606_HUMAN", "Human"),
    ("UP000000625_83333_ECOLI", "E. coli"),
    ("UP000001940_6239_CAEEL", "C. elegans"),
    ("UP000006548_3702_ARATH", "A. thaliana"),
    ("UP000000559_237561_CANAL", "C. albicans"),
    ("UP000000437_7955_DANRE", "D. rerio"),
    ("UP000002195_44689_DICDI", "D. discoideum"),
    ("UP000000803_7227_DROME", "D. melanogaster"),
    ("UP000008827_3847_SOYBN", "G. max"),
    ("UP000000805_243232_METJA", "M. jannaschii"),
    ("UP000000589_10090_MOUSE", "M. musculus"),
    ("UP000059680_39947_ORYSJ", "O. sativa"),
    ("UP000002494_10116_RAT", "R. norvegicus"),
    ("UP000002311_559292_YEAST", "S. cerevisiae"),
    ("UP000002485_284812_SCHPO", "S. pombe"),
    ("UP000007305_4577_MAIZE", "Z. mays"),
    ("UP000008854_6183_SCHMA", "*S. mansoni"),
    ("UP000001450_36329_PLAF7", "*P. falciparum"),
    ("UP000001631_447093_AJECG", "*A. capsulatus"),
    ("UP000000806_272631_MYCLE", "*M. leprae"),
    ("swissprot_pdb", "Swiss-Prot"),
]

TSV_DATASETS = [
    ("uniprotkb_Nematocida_parisii_2025_04_25", "*N. parisii"),
    ("uniprotkb_Toxoplasma_gondii_RH88_2026_03_20", "*T. gondii"),
]

# Threshold at which masked_mean_plddt was originally precomputed (process_afdb.py).
PRECOMPUTED_MASK_THRESHOLD = 0.5


# ---------- UniProt config ----------

UNIPROT_BATCH_URL = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_BATCH_SIZE = 100
UNIPROT_FIELDS = [
    "accession",
    "protein_existence",
    "annotation_score",
    "xref_pdb",
    "xref_pfam",
    "xref_interpro",
    "go_p",
    "go_f",
    "go_c",
    "cc_disease",
    "keyword",
    "lit_pubmed_id",
    "cc_subcellular_location",
    "ec",
]

# Map UniProt TSV column names back to our internal short names.
UNIPROT_TSV_RENAME = {
    "Entry": "accession",
    "Protein existence": "protein_existence",
    "Annotation": "annotation_score",
    "PDB": "xref_pdb",
    "Pfam": "xref_pfam",
    "InterPro": "xref_interpro",
    "Gene Ontology (biological process)": "go_p",
    "Gene Ontology (molecular function)": "go_f",
    "Gene Ontology (cellular component)": "go_c",
    "Involvement in disease": "cc_disease",
    "Keywords": "keyword",
    "PubMed ID": "lit_pubmed_id",
    "Subcellular location [CC]": "cc_subcellular_location",
    "EC number": "ec",
}

# AFDB accessions include fragment suffix like '-F1', '-F2'. Strip to get UniProt accession.
AFDB_FRAGMENT_RE = re.compile(r"-F\d+$")


def to_uniprot_acc(accession_id):
    return AFDB_FRAGMENT_RE.sub("", str(accession_id))


PE_TEXT_TO_NUM = {
    "Evidence at protein level": 1,
    "Evidence at transcript level": 2,
    "Inferred from homology": 3,
    "Predicted": 4,
    "Uncertain": 5,
}


# ---------- DisProt config ----------

DISPROT_URL = "https://disprot.org/api/search?release=current&get_consensus=true&format=json"


# ---------- Helpers (shared with build_filtered_target_list.py) ----------

def cutoff_to_suffix(val):
    """0.5 -> 05, 0.05 -> 005, 50 -> 50."""
    s = str(val).rstrip("0").rstrip(".")
    return s.replace(".", "")


def merge_sources(series):
    return ";".join(dict.fromkeys(series))


def merge_duplicate_accessions(df):
    cols = [c for c in df.columns if c != "accession_id"]
    agg = {c: "first" for c in cols if c != "source"}
    agg["source"] = merge_sources
    return df.groupby("accession_id", as_index=False).agg(agg)


# ---------- Stage 1: AFDB-based pre-filter ----------

def ordered_run_info(disorder_score, threshold=0.5):
    """Compute (longest_run_length, start_idx, end_idx_exclusive) over residues
    with disorder_score <= threshold (matching process_afdb.py convention).
    Accepts either a stringified list (from CSV) or a pre-parsed list/array
    (from Parquet). Returns (0, -1, -1) on parse failure or no ordered residues."""
    if disorder_score is None or (isinstance(disorder_score, float) and np.isnan(disorder_score)):
        return (0, -1, -1)
    if isinstance(disorder_score, str):
        try:
            scores = ast.literal_eval(disorder_score)
        except (ValueError, SyntaxError):
            return (0, -1, -1)
    else:
        scores = disorder_score
    if not hasattr(scores, "__len__") or len(scores) == 0:
        return (0, -1, -1)
    arr = np.asarray(scores, dtype=float)
    ordered = arr <= threshold
    if not ordered.any():
        return (0, -1, -1)
    diff = np.diff(ordered.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    idx = int(lengths.argmax())
    return (int(lengths[idx]), int(starts[idx]), int(ends[idx]))


def ordered_segments_info(disorder_score, threshold=0.5, min_run=0):
    """Return ALL maximal runs of ordered residues (disorder_score <= threshold)
    with length >= min_run, as [start, end_exclusive] pairs in chain order.
    Mirrors ordered_run_info boundary logic but keeps every passing run.
    Returns [] on parse failure or no ordered residues. Accepts a stringified
    list (CSV) or a pre-parsed list/array (Parquet)."""
    if disorder_score is None or (isinstance(disorder_score, float) and np.isnan(disorder_score)):
        return []
    if isinstance(disorder_score, str):
        try:
            scores = ast.literal_eval(disorder_score)
        except (ValueError, SyntaxError):
            return []
    else:
        scores = disorder_score
    if not hasattr(scores, "__len__") or len(scores) == 0:
        return []
    arr = np.asarray(scores, dtype=float)
    ordered = arr <= threshold
    if not ordered.any():
        return []
    diff = np.diff(ordered.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [[int(st), int(en)] for st, en in zip(starts, ends) if (en - st) >= min_run]


def _parse_array(value):
    """Return a numpy float array from either a string (CSV) or a pre-parsed list/array (Parquet)."""
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    return np.asarray(value, dtype=float)


def _recompute_masked_plddt(disorder_score, plddt, threshold):
    """Compute masked_mean_plddt for a row using a custom disorder threshold.
    Accepts strings (CSV) or pre-parsed lists/arrays (Parquet)."""
    try:
        disorder = _parse_array(disorder_score)
        plddt_arr = _parse_array(plddt)
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    mask = disorder <= threshold
    return float(np.mean(plddt_arr[mask])) if mask.any() else np.nan


def load_and_prefilter_dataset(data_dir, base_name, source_label,
                               masked_plddt_cutoff, ordered_run_min,
                               max_len, disorder_threshold, version="v6",
                               parquet_dir=None, stats_dir=None):
    """Load a v6 dataset, compute longest_ordered_run, and apply AFDB pre-filter.

    Fast-path priority (first match wins):
      1. Full Parquet (--parquet_dir): arrays pre-parsed, no literal_eval.
      2. Stats Parquet (--stats_dir): scalars only, no array parsing at all.
      3. v6 CSV (--data_dir): slowest; literal_eval per row.
    """
    # --- 1. Full Parquet fast path ---
    if parquet_dir is not None:
        parquet_path = os.path.join(parquet_dir, f"{base_name}_{version}.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            df["source"] = source_label
            if "length" not in df.columns:
                df["length"] = df["sequence"].str.len()
            # Arrays are already Python lists/arrays — ordered_run_info handles both.
            run_info = df["disorder_score"].apply(lambda s: ordered_run_info(s, disorder_threshold))
            df["longest_ordered_run"] = run_info.apply(lambda t: t[0])
            df["ordered_run_start"] = run_info.apply(lambda t: t[1])
            df["ordered_run_end"] = run_info.apply(lambda t: t[2])
            seg_info = df["disorder_score"].apply(
                lambda x: ordered_segments_info(x, disorder_threshold, ordered_run_min))
            df["ordered_segments"] = seg_info.apply(json.dumps)
            df["n_ordered_segments"] = seg_info.apply(len)
            if abs(disorder_threshold - PRECOMPUTED_MASK_THRESHOLD) > 1e-9:
                df["masked_mean_plddt"] = [
                    _recompute_masked_plddt(ds, pl, disorder_threshold)
                    for ds, pl in zip(df["disorder_score"], df["plddt"])
                ]
            cond = (
                (df["masked_mean_plddt"] < masked_plddt_cutoff)
                & (df["longest_ordered_run"] >= ordered_run_min)
            )
            if max_len >= 0:
                cond = cond & (df["length"] <= max_len)
            return df[cond]

    # --- 2. Stats Parquet fast path (scalars only, no arrays) ---
    if stats_dir is not None:
        t_key = cutoff_to_suffix(disorder_threshold)
        sparquet_path = os.path.join(stats_dir, f"{base_name}_{version}_stats_t{t_key}.parquet")
        if os.path.exists(sparquet_path):
            df = pd.read_parquet(sparquet_path)
            df["source"] = source_label
            cond = (
                (df["masked_mean_plddt"] < masked_plddt_cutoff)
                & (df["longest_ordered_run"] >= ordered_run_min)
            )
            if max_len >= 0:
                cond = cond & (df["length"] <= max_len)
            return df[cond]

    # --- 3. CSV path (slowest) ---
    if base_name.startswith("uniprotkb_"):
        csv_candidates = [
            os.path.join(data_dir, f"{base_name}_{version}.csv"),
            os.path.join(data_dir, f"{base_name}.csv"),
        ]
    else:
        csv_candidates = [os.path.join(data_dir, f"{base_name}_{version}.csv")]

    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    df["source"] = source_label
    df["length"] = df["sequence"].str.len()

    run_info = df["disorder_score"].apply(lambda s: ordered_run_info(s, disorder_threshold))
    df["longest_ordered_run"] = run_info.apply(lambda t: t[0])
    df["ordered_run_start"] = run_info.apply(lambda t: t[1])
    df["ordered_run_end"] = run_info.apply(lambda t: t[2])
    seg_info = df["disorder_score"].apply(
        lambda x: ordered_segments_info(x, disorder_threshold, ordered_run_min))
    df["ordered_segments"] = seg_info.apply(json.dumps)
    df["n_ordered_segments"] = seg_info.apply(len)

    if abs(disorder_threshold - PRECOMPUTED_MASK_THRESHOLD) > 1e-9:
        df["masked_mean_plddt"] = [
            _recompute_masked_plddt(ds, pl, disorder_threshold)
            for ds, pl in zip(df["disorder_score"], df["plddt"])
        ]

    cond = (
        (df["masked_mean_plddt"] < masked_plddt_cutoff)
        & (df["longest_ordered_run"] >= ordered_run_min)
    )
    if max_len >= 0:
        cond = cond & (df["length"] <= max_len)
    return df[cond]


# ---------- Stage 2: UniProt enrichment ----------

def _request_with_retry(url, params, max_retries=3, timeout=60):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def fetch_uniprot_batch(accessions, fields=UNIPROT_FIELDS):
    """Fetch a batch of UniProt accessions in TSV format with the given fields."""
    params = {
        "accessions": ",".join(accessions),
        "fields": ",".join(fields),
        "format": "tsv",
    }
    r = _request_with_retry(UNIPROT_BATCH_URL, params=params)
    text = r.text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    df = df.rename(columns=UNIPROT_TSV_RENAME)
    return df


def enrich_with_uniprot(accessions, cache_dir, fields=UNIPROT_FIELDS,
                        rate_limit_sleep=0.5):
    """Look up UniProt metadata for the given accessions with persistent cache.

    Cache layout: cache_dir/uniprot_metadata.tsv (append-only, keyed on accession).
    Only previously-uncached accessions are queried.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "uniprot_metadata.tsv")

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, sep="\t", dtype=str, keep_default_na=False)
    else:
        cached = pd.DataFrame()

    cached_set = set(cached["accession"].tolist()) if "accession" in cached.columns else set()
    missing = [a for a in accessions if a not in cached_set]
    print(f"  UniProt cache: {len(cached_set)} cached, {len(missing)} to query "
          f"({len(accessions)} total requested)")

    if missing:
        batches = []
        n_batches = (len(missing) + UNIPROT_BATCH_SIZE - 1) // UNIPROT_BATCH_SIZE
        for i in range(0, len(missing), UNIPROT_BATCH_SIZE):
            batch = missing[i:i + UNIPROT_BATCH_SIZE]
            batch_num = i // UNIPROT_BATCH_SIZE + 1
            try:
                df_batch = fetch_uniprot_batch(batch, fields)
            except requests.exceptions.RequestException as e:
                print(f"  Batch {batch_num}/{n_batches}: FAILED ({e})")
                continue
            print(f"  Batch {batch_num}/{n_batches}: {len(df_batch)} entries returned")
            if not df_batch.empty:
                batches.append(df_batch)
            time.sleep(rate_limit_sleep)

        if batches:
            new_df = pd.concat(batches, ignore_index=True)
            cached = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
            # Drop accidental duplicates (e.g., reruns after interruption).
            cached = cached.drop_duplicates(subset=["accession"], keep="last")
            cached.to_csv(cache_path, sep="\t", index=False)
            print(f"  UniProt cache updated: {cache_path} ({len(cached)} total)")

    # Return only rows for the requested accessions.
    return cached[cached["accession"].isin(accessions)].reset_index(drop=True)


def _count_semi(value):
    if not value:
        return 0
    return sum(1 for x in str(value).split(";") if x.strip())


def _count_go(value):
    if not value:
        return 0
    return str(value).count("[GO:")


def _parse_pe(value):
    if not value:
        return np.nan
    s = str(value).strip()
    if s in PE_TEXT_TO_NUM:
        return PE_TEXT_TO_NUM[s]
    try:
        return int(s)
    except ValueError:
        return np.nan


def derive_uniprot_columns(uniprot_df):
    """Add summary columns derived from raw UniProt fields."""
    out = uniprot_df.copy()
    for col in UNIPROT_TSV_RENAME.values():
        if col not in out.columns:
            out[col] = ""

    out["protein_existence_num"] = out["protein_existence"].apply(_parse_pe)
    out["annotation_score_num"] = pd.to_numeric(out["annotation_score"], errors="coerce")

    out["n_pdb"] = out["xref_pdb"].apply(_count_semi)
    out["pdb_ids"] = out["xref_pdb"].fillna("")

    out["n_pfam"] = out["xref_pfam"].apply(_count_semi)
    out["pfam_ids"] = out["xref_pfam"].fillna("")
    out["n_interpro"] = out["xref_interpro"].apply(_count_semi)

    out["n_go_p"] = out["go_p"].apply(_count_go)
    out["n_go_f"] = out["go_f"].apply(_count_go)
    out["n_go_c"] = out["go_c"].apply(_count_go)

    out["has_disease"] = out["cc_disease"].apply(lambda v: bool(v) and str(v).strip() != "")
    out["n_keywords"] = out["keyword"].apply(_count_semi)
    out["n_pubmed"] = out["lit_pubmed_id"].apply(_count_semi)
    out["subcell_loc"] = out["cc_subcellular_location"].fillna("")
    out["ec_number"] = out["ec"].fillna("")
    return out


# ---------- Stage 3: DisProt enrichment ----------

def fetch_disprot(cache_dir):
    """Download (or load cached) DisProt dump and return {acc: disorder_content}."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "disprot_dump.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"  DisProt cache loaded: {cache_path}")
    else:
        print(f"  Fetching DisProt: {DISPROT_URL}")
        r = _request_with_retry(DISPROT_URL, params=None, timeout=120)
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"  DisProt cached to: {cache_path}")

    entries = data.get("data") if isinstance(data, dict) else data
    lookup = {}
    for e in entries or []:
        acc = e.get("acc") or e.get("uniprot_acc") or e.get("uniprot")
        dc = e.get("disorder_content")
        if acc and dc is not None:
            lookup[acc] = dc
    print(f"  DisProt entries with disorder_content: {len(lookup)}")
    return lookup


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Build functional-relevance AFDB target list with UniProt + DisProt enrichment"
    )
    parser.add_argument("--data_dir", required=True, help="Directory with v6 CSV files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: data_dir)")
    parser.add_argument("--version", default="v6", help="Version suffix (default: v6)")
    parser.add_argument("--masked_plddt_cutoff", type=float, default=0.7,
                        help="Filter masked_mean_plddt < this value (default: 0.7)")
    parser.add_argument("--ordered_run_min", type=int, default=50,
                        help="Require longest_ordered_run >= this value (default: 50)")
    parser.add_argument("--disorder_threshold", type=float, default=0.5,
                        help="Per-residue disorder cutoff defining 'ordered' residues "
                             "(default: 0.5, matches precomputed masked_mean_plddt). "
                             "Using 0.5 avoids recomputing masked_mean_plddt from arrays.")
    parser.add_argument("--max_len", type=int, default=-1,
                        help="Maximum sequence length; -1 disables (default: -1)")
    parser.add_argument("--parquet_dir", default=None,
                        help="Directory with full v6 Parquets (from convert_afdb_to_parquet.py). "
                             "Arrays are stored as native lists — no literal_eval overhead. "
                             "Takes priority over --stats_dir and CSV.")
    parser.add_argument("--stats_dir", default=None,
                        help="Directory with stats-only Parquets (from precompute_afdb_stats.py). "
                             "Fastest option but locked to precomputed thresholds. "
                             "Used only if --parquet_dir not found.")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache dir for UniProt/DisProt responses "
                             "(default: <output_dir>/.uniprot_cache)")
    parser.add_argument("--skip_uniprot", action="store_true",
                        help="Skip UniProt enrichment (testing only)")
    parser.add_argument("--skip_disprot", action="store_true",
                        help="Skip DisProt enrichment")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(output_dir, ".uniprot_cache")

    if abs(args.disorder_threshold - PRECOMPUTED_MASK_THRESHOLD) > 1e-9:
        print(f"NOTE: disorder_threshold={args.disorder_threshold} differs from "
              f"precomputed value ({PRECOMPUTED_MASK_THRESHOLD}); "
              f"masked_mean_plddt will be recomputed from per-residue arrays (slower).")

    p_suffix = f"pmasked-{cutoff_to_suffix(args.masked_plddt_cutoff)}"
    r_suffix = f"run-{args.ordered_run_min}"
    base_name_out = f"afdb_functional_{p_suffix}_{r_suffix}"
    if abs(args.disorder_threshold - PRECOMPUTED_MASK_THRESHOLD) > 1e-9:
        base_name_out += f"_dt-{cutoff_to_suffix(args.disorder_threshold)}"
    if args.max_len >= 0:
        base_name_out += f"_max{args.max_len}"

    # Stage 1: pre-filter each dataset
    print("\n=== Stage 1: AFDB pre-filter ===")
    all_datasets = MODEL_ORGANISM_DATASETS + TSV_DATASETS
    filtered_dfs = []
    for base_name, source_label in all_datasets:
        df = load_and_prefilter_dataset(
            args.data_dir, base_name, source_label,
            args.masked_plddt_cutoff, args.ordered_run_min,
            args.max_len, args.disorder_threshold, args.version,
            parquet_dir=args.parquet_dir,
            stats_dir=args.stats_dir,
        )
        if df is None:
            print(f"  WARNING: no v6 CSV for {source_label} ({base_name})")
            continue
        print(f"  {source_label}: {len(df)} entries (from {base_name}_{args.version}.csv)")
        filtered_dfs.append(df)

    if not filtered_dfs:
        print("No datasets loaded. Exiting.")
        return

    combined = pd.concat(filtered_dfs, ignore_index=True)
    print(f"\nStage 1 survivors (pre-dedup): {len(combined)}")
    print(f"  Filters: masked_mean_plddt<{args.masked_plddt_cutoff}, "
          f"longest_ordered_run>={args.ordered_run_min}"
          + (f", length<={args.max_len}" if args.max_len >= 0 else ""))

    n_before = len(combined)
    combined = merge_duplicate_accessions(combined)
    n_merged = n_before - len(combined)
    if n_merged > 0:
        print(f"Merged {n_merged} duplicate accession rows; unique accessions: {len(combined)}")
    else:
        print(f"Unique accessions (no duplicates): {len(combined)}")

    # Strip AFDB fragment suffix (-F1, -F2, ...) to obtain plain UniProt accession.
    combined["uniprot_acc"] = combined["accession_id"].map(to_uniprot_acc)

    # Stage 2: UniProt enrichment
    if args.skip_uniprot:
        print("\n=== Stage 2: UniProt enrichment SKIPPED ===")
    else:
        print("\n=== Stage 2: UniProt enrichment ===")
        unique_uniprot_accs = sorted(set(combined["uniprot_acc"].tolist()))
        uniprot_raw = enrich_with_uniprot(unique_uniprot_accs, cache_dir, UNIPROT_FIELDS)
        if uniprot_raw.empty:
            print("  WARNING: UniProt returned no rows; continuing without enrichment.")
        else:
            uniprot_derived = derive_uniprot_columns(uniprot_raw)
            combined = combined.merge(
                uniprot_derived, how="left",
                left_on="uniprot_acc", right_on="accession",
            )
            if "accession" in combined.columns:
                combined = combined.drop(columns=["accession"])

    # Stage 3: DisProt enrichment
    if args.skip_disprot:
        print("\n=== Stage 3: DisProt enrichment SKIPPED ===")
    else:
        print("\n=== Stage 3: DisProt enrichment ===")
        try:
            disprot_map = fetch_disprot(cache_dir)
            combined["disprot_disorder_content"] = combined["uniprot_acc"].map(disprot_map)
            n_disprot = combined["disprot_disorder_content"].notna().sum()
            print(f"  Entries with DisProt disorder_content: {n_disprot}")
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: DisProt fetch failed ({e}); skipping disprot column.")

    # Stage 4: hard filter (no PDB) and write outputs
    print("\n=== Stage 4: Filter and write ===")
    if "n_pdb" in combined.columns:
        before = len(combined)
        combined = combined[combined["n_pdb"].fillna(0).astype(int) == 0]
        print(f"  PDB filter: {before} -> {len(combined)} (removed {before - len(combined)} with PDB xref)")
    else:
        print("  No 'n_pdb' column (UniProt enrichment skipped?); not applying PDB filter.")

    full_path = os.path.join(output_dir, f"{base_name_out}_full.csv")
    combined.to_csv(full_path, index=False)
    print(f"Saved full: {full_path}  ({len(combined)} rows)")

    drop_cols = [c for c in ("disorder_score", "plddt") if c in combined.columns]
    reduced = combined.drop(columns=drop_cols) if drop_cols else combined
    reduced_path = os.path.join(output_dir, f"{base_name_out}.csv")
    reduced.to_csv(reduced_path, index=False)
    print(f"Saved reduced (per-residue arrays dropped): {reduced_path}")


if __name__ == "__main__":
    main()
