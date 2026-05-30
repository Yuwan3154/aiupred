#!/usr/bin/env python3
"""
Precompute per-protein summary statistics from v6 AFDB CSV files and save
as compact Parquet files.  Running this once lets build_functional_target_list.py
skip per-residue array parsing on every run, reducing Stage 1 from ~5-10 min
to ~30 s.

For each v6 CSV and each requested disorder threshold T, this script:
  - Parses disorder_score and plddt per-residue arrays
  - Computes masked_mean_plddt  (mean pLDDT over residues with disorder <= T)
  - Computes longest_ordered_run, ordered_run_start, ordered_run_end
  - Saves a stats Parquet: {out_dir}/{base_name}_{version}_stats_t{T_suffix}.parquet

The Parquet contains no per-residue arrays, so it is ~100x smaller than the
source CSV.  The source CSV is not modified.

Usage:
    python precompute_afdb_stats.py \\
        --data_dir /home/jupyter-chenxi/data/afdb \\
        [--out_dir /home/jupyter-chenxi/data/afdb/stats]  # default: data_dir
        [--thresholds 0.3 0.5 0.7]                        # default: 0.5
        [--version v6]
        [--overwrite]   # recompute even if Parquet already exists
"""

import argparse
import ast
import os
import sys
import time

try:
    import pyarrow  # noqa: F401
except ImportError:
    print("ERROR: pyarrow is required for Parquet support.  Install with:")
    print("    conda install pyarrow   or   pip install pyarrow")
    sys.exit(1)

import numpy as np

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd

# ---------- Dataset registry (must mirror build_functional_target_list.py) ----------

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


def cutoff_to_suffix(val):
    s = str(val).rstrip("0").rstrip(".")
    return s.replace(".", "")


def compute_stats_for_row(disorder_score_str, plddt_str, threshold):
    """Return (masked_mean_plddt, longest_run, run_start, run_end) for one protein."""
    try:
        disorder = np.asarray(ast.literal_eval(disorder_score_str), dtype=float)
        plddt = np.asarray(ast.literal_eval(plddt_str), dtype=float)
    except (ValueError, SyntaxError, TypeError):
        return (np.nan, 0, -1, -1)

    ordered = disorder <= threshold

    # masked_mean_plddt
    if ordered.any():
        masked_plddt = float(np.mean(plddt[ordered]))
    else:
        masked_plddt = np.nan

    # longest ordered run
    if not ordered.any():
        return (masked_plddt, 0, -1, -1)
    diff = np.diff(ordered.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    idx = int(lengths.argmax())
    return (masked_plddt, int(lengths[idx]), int(starts[idx]), int(ends[idx]))


def process_dataset(data_dir, base_name, version, threshold, out_dir, overwrite):
    t_key = cutoff_to_suffix(threshold)
    out_path = os.path.join(out_dir, f"{base_name}_{version}_stats_t{t_key}.parquet")

    if os.path.exists(out_path) and not overwrite:
        print(f"  SKIP (exists): {os.path.basename(out_path)}")
        return True

    if base_name.startswith("uniprotkb_"):
        csv_candidates = [
            os.path.join(data_dir, f"{base_name}_{version}.csv"),
            os.path.join(data_dir, f"{base_name}.csv"),
        ]
    else:
        csv_candidates = [os.path.join(data_dir, f"{base_name}_{version}.csv")]

    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print(f"  SKIP (no CSV): {base_name}_{version}.csv")
        return False

    t0 = time.time()
    df = pd.read_csv(csv_path)
    n = len(df)

    # Compute per-row stats
    results = [
        compute_stats_for_row(ds, pl, threshold)
        for ds, pl in zip(df["disorder_score"], df["plddt"])
    ]

    # Build stats DataFrame (no per-residue arrays)
    stats = pd.DataFrame({
        "accession_id": df["accession_id"],
        "sequence": df["sequence"],
        "length": df["sequence"].str.len(),
        "mean_plddt": df["mean_plddt"],
        "mean_disorder_score": df["mean_disorder_score"],
        "masked_mean_plddt": [r[0] for r in results],
        "longest_ordered_run": [r[1] for r in results],
        "ordered_run_start": [r[2] for r in results],
        "ordered_run_end": [r[3] for r in results],
    })

    stats.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {os.path.basename(out_path)}: {n} rows -> {size_mb:.1f} MB  ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-protein AFDB stats Parquets for fast filtering"
    )
    parser.add_argument("--data_dir", required=True, help="Directory with v6 CSV files")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for Parquets (default: data_dir)")
    parser.add_argument("--version", default="v6", help="Version suffix (default: v6)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5],
                        help="Disorder thresholds to precompute (default: 0.5)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if Parquet already exists")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    all_datasets = MODEL_ORGANISM_DATASETS + TSV_DATASETS
    total = len(all_datasets) * len(args.thresholds)
    done = 0

    for threshold in args.thresholds:
        print(f"\n=== Threshold {threshold} ===")
        for base_name, source_label in all_datasets:
            print(f"  {source_label} ...", end=" ", flush=True)
            process_dataset(args.data_dir, base_name, args.version,
                            threshold, out_dir, args.overwrite)
            done += 1

    print(f"\nDone: {done}/{total} datasets processed.")
    print(f"Pass --stats_dir {out_dir} to build_functional_target_list.py to use these Parquets.")


if __name__ == "__main__":
    main()
