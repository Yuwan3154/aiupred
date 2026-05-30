#!/usr/bin/env python3
"""
Build a concatenated target list from v6 AFDB datasets by filtering for
mean_plddt < plddt_cutoff and mean_disorder_score < disorder_cutoff.

Usage:
    python build_filtered_target_list.py \
        --data_dir /home/jupyter-chenxi/data/afdb \
        --output_dir /home/jupyter-chenxi/data/afdb \
        [--plddt_cutoff 0.5] [--disorder_cutoff 0.2] [--max_len -1]
"""

import os
import argparse

import fireducks.pandas as pd

# Dataset registry matching analyze_afdb.py (v6 only)
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
    """Convert cutoff to filename suffix by removing the decimal point. 0.5 -> 05, 0.05 -> 005."""
    s = str(val).rstrip("0").rstrip(".")
    return s.replace(".", "")


def merge_sources(series):
    """Join distinct source labels in first-seen order, semicolon-separated."""
    return ";".join(dict.fromkeys(series))


def merge_duplicate_accessions(df):
    """One row per accession_id; source values merged with semicolons."""
    cols = [c for c in df.columns if c != "accession_id"]
    agg = {c: "first" for c in cols if c != "source"}
    agg["source"] = merge_sources
    return df.groupby("accession_id", as_index=False).agg(agg)


def load_and_filter_dataset(data_dir, base_name, source_label, plddt_cutoff, disorder_cutoff, version="v6"):
    """
    Load a v6 CSV, add source column, filter by mean_plddt < plddt_cutoff and
    mean_disorder_score < disorder_cutoff, add length column.
    Returns None if file not found.
    """
    if base_name.startswith("uniprotkb_"):
        csv_candidates = [
            os.path.join(data_dir, f"{base_name}_{version}.csv"),
            os.path.join(data_dir, f"{base_name}.csv"),
        ]
    else:
        csv_candidates = [
            os.path.join(data_dir, f"{base_name}_{version}.csv"),
        ]

    csv_path = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    df["source"] = source_label
    df["length"] = df["sequence"].str.len()

    filtered = df[(df["mean_plddt"] < plddt_cutoff) & (df["mean_disorder_score"] < disorder_cutoff)]
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Build filtered AFDB target list by mean_plddt and mean_disorder_score cutoffs"
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing v6 CSV files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: data_dir)")
    parser.add_argument("--version", default="v6", help="Version suffix (default: v6)")
    parser.add_argument("--plddt_cutoff", type=float, default=0.5,
                        help="Filter mean_plddt < this value (default: 0.5)")
    parser.add_argument("--disorder_cutoff", type=float, default=0.2,
                        help="Filter mean_disorder_score < this value (default: 0.2)")
    parser.add_argument("--max_len", type=int, default=-1,
                        help="Filter length <= this value; -1 means no filter (default: -1)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    p_suffix = f"plddt-{cutoff_to_suffix(args.plddt_cutoff)}"
    d_suffix = f"aiupred-{cutoff_to_suffix(args.disorder_cutoff)}"
    base_name_out = f"afdb_model_org_{p_suffix}_{d_suffix}"
    if args.max_len >= 0:
        base_name_out = f"{base_name_out}_max{args.max_len}"

    all_datasets = MODEL_ORGANISM_DATASETS + TSV_DATASETS
    filtered_dfs = []

    for base_name, source_label in all_datasets:
        df = load_and_filter_dataset(
            args.data_dir, base_name, source_label,
            args.plddt_cutoff, args.disorder_cutoff, args.version
        )
        if df is not None:
            n = len(df)
            print(f"  {source_label}: {n} entries (from {base_name}_{args.version}.csv)")
            filtered_dfs.append(df)
        else:
            print(f"  WARNING: No v6 CSV found for {source_label} ({base_name})")

    if not filtered_dfs:
        print("No datasets loaded. Exiting.")
        return

    combined = pd.concat(filtered_dfs, ignore_index=True)
    if args.max_len >= 0:
        combined = combined[combined["length"] <= args.max_len]
        print(f"\nTotal filtered entries (before accession dedup): {len(combined)} (mean_plddt<{args.plddt_cutoff}, mean_disorder_score<{args.disorder_cutoff}, length<={args.max_len})")
    else:
        print(f"\nTotal filtered entries (before accession dedup): {len(combined)} (mean_plddt<{args.plddt_cutoff}, mean_disorder_score<{args.disorder_cutoff})")

    n_before_dedup = len(combined)
    combined = merge_duplicate_accessions(combined)
    n_merged = n_before_dedup - len(combined)
    if n_merged > 0:
        print(f"Merged {n_merged} duplicate accession_id row(s); unique accessions: {len(combined)}")
    else:
        print(f"Unique accessions (no duplicates): {len(combined)}")

    full_path = os.path.join(output_dir, f"{base_name_out}_full.csv")
    combined.to_csv(full_path, index=False)
    print(f"Saved full: {full_path}")

    reduced = combined.drop(columns=["disorder_score", "plddt"])
    reduced_path = os.path.join(output_dir, f"{base_name_out}.csv")
    reduced.to_csv(reduced_path, index=False)
    print(f"Saved reduced (disorder_score, plddt dropped): {reduced_path}")


if __name__ == "__main__":
    main()
