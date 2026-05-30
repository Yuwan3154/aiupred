#!/usr/bin/env python3
"""
Safe, idempotent cleanup of v4 AFDB data after v6 results are confirmed.

Phases:
  1. csvs  — delete *_v4.csv files (only if *_v6.csv has ≥99% as many rows)
  2. pngs  — delete the 6 old analysis PNG files (only if they were recently regenerated)
  3. dirs  — delete v4 PDB/CIF structure directories (~330 GB; irreversible)

Usage:
    # Preview what would be deleted (safe, no changes)
    python cleanup_v4.py --data_dir /home/jupyter-chenxi/data/afdb --dry-run

    # Delete CSVs and PNGs only (no interactive prompt needed for these)
    python cleanup_v4.py --data_dir /home/jupyter-chenxi/data/afdb --phase csvs
    python cleanup_v4.py --data_dir /home/jupyter-chenxi/data/afdb --phase pngs

    # Delete everything including structure dirs (will ask for confirmation)
    python cleanup_v4.py --data_dir /home/jupyter-chenxi/data/afdb --phase all

    # Non-interactive (for scripting, after --dry-run review)
    python cleanup_v4.py --data_dir /home/jupyter-chenxi/data/afdb --confirm
"""

import os
import sys
import shutil
import argparse
import time
from glob import glob
from pathlib import Path

import pandas as pd

ANALYSIS_PNGS = [
    "mean_masked_plddt_distribution.png",
    "mean_plddt_vs_mean_disorder_score_colored.png",
    "mean_plddt_vs_mean_disorder_score_hex.png",
    "mean_plddt_vs_mean_disorder_score_hex_no_title.png",
    "proportion_disordered_vs_masked_mean_plddt.png",
    "top_50_ordered_mean_plddt_vs_mean_plddt.png",
]

# Nematocida directory is named without a _v4 version suffix
NEMATOCIDA_DIR = "uniprotkb_Nematocida_parisii_2025_04_25"


def dir_size_gb(path):
    """Return total size of a directory in GB."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


def phase_csvs(data_dir, dry_run):
    """Delete *_v4.csv files where a valid *_v6.csv counterpart exists."""
    print("\n--- Phase 1: v4 CSV files ---")
    v4_csvs = sorted(glob(os.path.join(data_dir, "*_v4.csv")))
    if not v4_csvs:
        print("  No *_v4.csv files found.")
        return 0

    deleted = 0
    skipped = 0
    for v4_csv in v4_csvs:
        base = Path(v4_csv).stem.replace('_v4', '')
        v6_csv = os.path.join(data_dir, f"{base}_v6.csv")

        if not os.path.exists(v6_csv):
            print(f"  SKIP {os.path.basename(v4_csv)}: no v6 CSV found ({os.path.basename(v6_csv)})")
            skipped += 1
            continue

        try:
            n_v4 = len(pd.read_csv(v4_csv))
            n_v6 = len(pd.read_csv(v6_csv))
        except Exception as e:
            print(f"  SKIP {os.path.basename(v4_csv)}: could not read CSVs ({e})")
            skipped += 1
            continue

        if n_v6 < n_v4 * 0.99:
            print(f"  SKIP {os.path.basename(v4_csv)}: v6 has {n_v6} rows vs v4 {n_v4} "
                  f"(less than 99% — v6 may be incomplete)")
            skipped += 1
            continue

        size_mb = os.path.getsize(v4_csv) / 1024 / 1024
        print(f"  DELETE {os.path.basename(v4_csv)} ({size_mb:.1f} MB, "
              f"v4={n_v4} rows, v6={n_v6} rows)")
        if not dry_run:
            os.remove(v4_csv)
        deleted += 1

    print(f"  Summary: {deleted} to delete, {skipped} skipped"
          + (" [DRY RUN]" if dry_run else ""))
    return deleted


def phase_pngs(data_dir, dry_run):
    """Delete the 6 analysis PNGs if they appear to have been recently regenerated."""
    print("\n--- Phase 2: Analysis PNG files ---")
    one_day_ago = time.time() - 86400

    deleted = 0
    skipped = 0
    for png_name in ANALYSIS_PNGS:
        png_path = os.path.join(data_dir, png_name)
        if not os.path.exists(png_path):
            print(f"  NOT FOUND: {png_name}")
            skipped += 1
            continue

        mtime = os.path.getmtime(png_path)
        if mtime < one_day_ago:
            print(f"  SKIP {png_name}: last modified more than 1 day ago "
                  f"(run analyze_afdb.py first to regenerate)")
            skipped += 1
            continue

        size_kb = os.path.getsize(png_path) / 1024
        print(f"  DELETE {png_name} ({size_kb:.0f} KB)")
        if not dry_run:
            os.remove(png_path)
        deleted += 1

    print(f"  Summary: {deleted} to delete, {skipped} skipped"
          + (" [DRY RUN]" if dry_run else ""))
    return deleted


def phase_dirs(data_dir, dry_run, confirmed):
    """
    Delete v4 structure directories. This is irreversible and requires explicit confirmation.
    """
    print("\n--- Phase 3: v4 Structure Directories (IRREVERSIBLE) ---")

    v4_dirs = sorted([
        d for d in glob(os.path.join(data_dir, "*_v4"))
        if os.path.isdir(d)
    ])

    # Also check the Nematocida directory for v4-only PDB files
    nema_dir = os.path.join(data_dir, NEMATOCIDA_DIR)
    has_nema_v4 = False
    has_nema_v6 = False
    if os.path.isdir(nema_dir):
        v4_pdbs = glob(os.path.join(nema_dir, "*_v4.pdb"))
        v6_pdbs = glob(os.path.join(nema_dir, "*_v6.pdb"))
        has_nema_v4 = len(v4_pdbs) > 0
        has_nema_v6 = len(v6_pdbs) > 0

    if not v4_dirs and not has_nema_v4:
        print("  No v4 structure directories found.")
        return 0

    # Safety checks and size calculation
    to_delete = []
    skipped = []
    total_gb = 0.0

    for v4_dir in v4_dirs:
        base_name = os.path.basename(v4_dir)
        v6_dir = v4_dir.replace('_v4', '_v6')
        v6_csv = os.path.join(data_dir, base_name.replace('_v4', '_v6') + '.csv')

        if not os.path.exists(v6_dir) or not os.listdir(v6_dir):
            skipped.append((v4_dir, "v6 dir missing or empty"))
            continue
        if not os.path.exists(v6_csv):
            skipped.append((v4_dir, "v6 CSV missing"))
            continue

        gb = dir_size_gb(v4_dir)
        total_gb += gb
        to_delete.append((v4_dir, gb))

    # Nematocida: v4 PDB files inside the shared directory
    nema_v4_files = []
    nema_v6_csv = os.path.join(data_dir, f"{NEMATOCIDA_DIR}_v6.csv")
    if has_nema_v4:
        if has_nema_v6 and os.path.exists(nema_v6_csv):
            nema_v4_files = glob(os.path.join(nema_dir, "*_v4.pdb"))
            nema_gb = sum(os.path.getsize(f) for f in nema_v4_files) / 1024 ** 3
            total_gb += nema_gb
            print(f"  Nematocida v4 PDB files: {len(nema_v4_files)} files ({nema_gb:.2f} GB)")
        else:
            skipped.append((nema_dir + "/*_v4.pdb",
                            "v6 PDB files or v6 CSV missing for Nematocida"))

    for v4_dir, gb in to_delete:
        print(f"  DELETE {os.path.basename(v4_dir)} ({gb:.2f} GB)")
    for path, reason in skipped:
        print(f"  SKIP   {os.path.basename(path)}: {reason}")

    print(f"\n  Total to free: {total_gb:.2f} GB")

    if dry_run:
        print("  [DRY RUN] No files deleted.")
        return 0

    if not confirmed:
        print("\n  WARNING: This will permanently delete the above directories.")
        print("  Type 'DELETE v4 DIRS' to confirm, or anything else to cancel:")
        try:
            answer = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "DELETE v4 DIRS":
            print("  Cancelled — no files deleted.")
            return 0

    deleted = 0
    for v4_dir, gb in to_delete:
        print(f"  Deleting {os.path.basename(v4_dir)}...")
        shutil.rmtree(v4_dir)
        deleted += 1

    if nema_v4_files:
        print(f"  Deleting {len(nema_v4_files)} Nematocida v4 PDB files...")
        for f in nema_v4_files:
            os.remove(f)
        deleted += len(nema_v4_files)

    print(f"  Deleted {deleted} items, freed ~{total_gb:.2f} GB")
    return deleted


def main():
    parser = argparse.ArgumentParser(
        description='Safe cleanup of v4 AFDB data after v6 results are confirmed'
    )
    parser.add_argument('--data_dir', required=True,
                        help='AFDB data directory containing CSV files and proteome dirs')
    parser.add_argument('--phase', choices=['all', 'csvs', 'pngs', 'dirs'], default='all',
                        help='Which phase(s) to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be deleted without deleting anything')
    parser.add_argument('--confirm', action='store_true',
                        help='Skip interactive confirmation prompt for directory deletion')
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN MODE — no files will be deleted ===")

    run_csvs = args.phase in ('all', 'csvs')
    run_pngs = args.phase in ('all', 'pngs')
    run_dirs = args.phase in ('all', 'dirs')

    if run_csvs:
        phase_csvs(args.data_dir, args.dry_run)
    if run_pngs:
        phase_pngs(args.data_dir, args.dry_run)
    if run_dirs:
        phase_dirs(args.data_dir, args.dry_run, args.confirm)

    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
