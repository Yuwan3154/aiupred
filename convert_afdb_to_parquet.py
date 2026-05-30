#!/usr/bin/env python3
"""
One-time conversion of v6 AFDB CSV files to Parquet format.

The v6 CSVs store per-residue disorder_score and plddt as stringified Python
lists (e.g. "[0.9672, 0.9781, ...]").  Reading them requires ast.literal_eval
per row, which is the main bottleneck in Stage 1 of the filtering pipeline.

This script converts each v6 CSV to a Parquet file where those columns are
stored as native list<float64> columns.  On subsequent runs, build_functional_
target_list.py can read the Parquets with --parquet_dir and skip literal_eval
entirely, reducing Stage 1 from ~5-10 min to ~1-2 min.

Properties:
  - All original columns preserved (disorder_score and plddt become list columns)
  - Source CSV is never modified
  - Already-converted datasets are skipped unless --overwrite is passed
  - The converted Parquet is ~5-10x smaller than the source CSV

Usage:
    python convert_afdb_to_parquet.py \\
        --data_dir /home/jupyter-chenxi/data/afdb \\
        --out_dir  /home/jupyter-chenxi/data/afdb/parquet \\
        [--overwrite]
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

LIST_COLUMNS = ["disorder_score", "plddt"]


def convert_dataset(data_dir, base_name, version, out_dir, overwrite):
    out_path = os.path.join(out_dir, f"{base_name}_{version}.parquet")

    if os.path.exists(out_path) and not overwrite:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  SKIP (exists, {size_mb:.1f} MB): {os.path.basename(out_path)}")
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

    csv_mb = os.path.getsize(csv_path) / 1e6
    t0 = time.time()
    df = pd.read_csv(csv_path)
    n = len(df)

    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    parquet_mb = os.path.getsize(out_path) / 1e6
    ratio = csv_mb / parquet_mb if parquet_mb > 0 else float("inf")
    print(f"  {os.path.basename(out_path)}: {n} rows  "
          f"{csv_mb:.0f} MB CSV -> {parquet_mb:.0f} MB Parquet  "
          f"({ratio:.1f}x smaller)  {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert v6 AFDB CSV files to Parquet (one-time operation)"
    )
    parser.add_argument("--data_dir", required=True, help="Directory with v6 CSV files")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for Parquets (default: <data_dir>/parquet)")
    parser.add_argument("--version", default="v6", help="Version suffix (default: v6)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert even if Parquet already exists")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "parquet")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    all_datasets = MODEL_ORGANISM_DATASETS + TSV_DATASETS
    ok = sum(
        convert_dataset(args.data_dir, base_name, args.version, out_dir, args.overwrite)
        for base_name, _ in all_datasets
    )
    print(f"\nConverted {ok}/{len(all_datasets)} datasets.")
    print(f"Pass --parquet_dir {out_dir} to build_functional_target_list.py to use these files.")


if __name__ == "__main__":
    main()
