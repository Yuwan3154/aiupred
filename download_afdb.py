#!/usr/bin/env python3
"""
Unified download script for AFDB v6 datasets.

Handles:
  - Proteome tarball downloads (20 model organisms + Swiss-Prot)
  - TSV-based individual PDB downloads (per-accession from AFDB)
  - CIF file cleanup (post-extraction and retroactive)

Usage:
    # Download all proteome tarballs
    python download_afdb.py --datasets proteomes

    # Download Swiss-Prot only
    python download_afdb.py --datasets swissprot

    # Download TSV-based datasets (individual PDBs)
    python download_afdb.py --datasets tsv

    # Clean up .cif files from existing directories
    python download_afdb.py --cleanup-cif

    # Dry run (show what would be done)
    python download_afdb.py --datasets all --dry-run
"""

import os
import sys
import argparse
import tarfile
import logging
import urllib.request
from glob import glob

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset registries
# ---------------------------------------------------------------------------

PROTEOME_DATASETS = [
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar",   "name": "Human"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI_v6.tar",  "name": "E. coli"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001940_6239_CAEEL_v6.tar",   "name": "C. elegans"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006548_3702_ARATH_v6.tar",   "name": "A. thaliana"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000559_237561_CANAL_v6.tar", "name": "C. albicans"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000437_7955_DANRE_v6.tar",   "name": "D. rerio"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002195_44689_DICDI_v6.tar",  "name": "D. discoideum"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000803_7227_DROME_v6.tar",   "name": "D. melanogaster"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008827_3847_SOYBN_v6.tar",   "name": "G. max"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v6.tar", "name": "M. jannaschii"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v6.tar",  "name": "M. musculus"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000059680_39947_ORYSJ_v6.tar",  "name": "O. sativa"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002494_10116_RAT_v6.tar",    "name": "R. norvegicus"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002311_559292_YEAST_v6.tar", "name": "S. cerevisiae"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v6.tar", "name": "S. pombe"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000007305_4577_MAIZE_v6.tar",   "name": "Z. mays"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008854_6183_SCHMA_v6.tar",   "name": "S. mansoni"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001450_36329_PLAF7_v6.tar",  "name": "P. falciparum"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001631_447093_AJECG_v6.tar", "name": "A. capsulatus"},
    {"url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000806_272631_MYCLE_v6.tar", "name": "M. leprae"},
]

SWISSPROT_DATASET = {
    "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar",
    "name": "Swiss-Prot",
}

TSV_DATASETS = [
    {
        "tsv": "uniprotkb_Nematocida_parisii_2025_04_25.tsv",
        "name": "N. parisii",
        "pdb_dir": "uniprotkb_Nematocida_parisii_2025_04_25",
    },
    {
        "tsv": "uniprotkb_Toxoplasma_gondii_RH88_2026_03_20.tsv",
        "name": "T. gondii",
        "pdb_dir": "uniprotkb_Toxoplasma_gondii_RH88_2026_03_20",
    },
]

# AFDB returns a 127-byte HTML "not found" page for missing models
AFDB_NOT_FOUND_SIZE = 127

DEFAULT_DATA_DIR = "/home/jupyter-chenxi/data/afdb"


# ---------------------------------------------------------------------------
# Tarball download / extract
# ---------------------------------------------------------------------------

def download_and_extract(dataset, data_dir, delete_cif=True, dry_run=False):
    """Download and extract a v6 proteome tarball. Returns extract dir name or None."""
    url = dataset["url"]
    name = dataset["name"]
    filename = url.split('/')[-1]
    pdb_dir = filename.split('.')[0]  # e.g. UP000000625_83333_ECOLI_v6

    extract_dir = os.path.join(data_dir, pdb_dir)

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"  {name} ({pdb_dir}): already extracted, skipping")
        return pdb_dir

    if dry_run:
        print(f"  {name}: would download {filename} and extract to {pdb_dir}/")
        return pdb_dir

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {name} ({filename})...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  Downloaded {filename} ({size_mb:.0f} MB)")
        except Exception as e:
            print(f"  FAILED to download {name}: {e}")
            return None

    print(f"  Extracting {name}...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(path=extract_dir)
        os.remove(filepath)
        print(f"  Extracted {name}")
    except Exception as e:
        print(f"  FAILED to extract {name}: {e}")
        return None

    # Gunzip any remaining .gz files
    gz_files = glob(os.path.join(extract_dir, "*.gz"))
    if gz_files:
        print(f"  Decompressing {len(gz_files)} .gz files...")
        for gz_file in gz_files:
            os.system(f"gunzip -q '{gz_file}'")

    if delete_cif:
        _delete_cif_in_dir(extract_dir)

    return pdb_dir


# ---------------------------------------------------------------------------
# TSV-based individual PDB downloads
# ---------------------------------------------------------------------------

def download_tsv_pdbs(tsv_file, pdb_dir, version="v6", dry_run=False):
    """
    Download individual PDB files for each accession in the TSV.
    Skips files already present. Deletes 127-byte "not found" responses.
    Returns the list of accession IDs that were successfully downloaded or already present.
    """
    df = pd.read_csv(tsv_file, sep='\t')
    entries = df['Entry'].tolist()

    if dry_run:
        existing = sum(1 for e in entries
                       if os.path.exists(os.path.join(pdb_dir, f"AF-{e}-F1-model_{version}.pdb")))
        print(f"  {os.path.basename(tsv_file)}: {len(entries)} entries, "
              f"{existing} already downloaded, {len(entries) - existing} to download")
        return entries

    os.makedirs(pdb_dir, exist_ok=True)
    print(f"  TSV contains {len(entries)} entries. Downloading {version} PDB files...")

    available = []
    for entry in tqdm(entries, desc="Downloading PDB files"):
        pdb_path = os.path.join(pdb_dir, f"AF-{entry}-F1-model_{version}.pdb")
        if os.path.exists(pdb_path):
            if os.path.getsize(pdb_path) == AFDB_NOT_FOUND_SIZE:
                os.remove(pdb_path)
                continue
            available.append(entry)
            continue

        url = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_{version}.pdb"
        try:
            urllib.request.urlretrieve(url, pdb_path)
            if os.path.getsize(pdb_path) == AFDB_NOT_FOUND_SIZE:
                os.remove(pdb_path)
                logging.warning(f"Not available in AFDB {version}: {entry}")
            else:
                available.append(entry)
        except Exception as e:
            logging.warning(f"Failed to download {entry}: {e}")

    print(f"  {len(available)}/{len(entries)} entries available in AFDB {version}")
    return available


# ---------------------------------------------------------------------------
# CIF cleanup
# ---------------------------------------------------------------------------

def _delete_cif_in_dir(directory):
    """Delete all .cif files in a directory, return (count, bytes_freed)."""
    cif_files = glob(os.path.join(directory, "*.cif"))
    if not cif_files:
        return 0, 0
    total_bytes = sum(os.path.getsize(f) for f in cif_files)
    for f in cif_files:
        os.remove(f)
    print(f"  Deleted {len(cif_files)} .cif files ({total_bytes / 1024**3:.1f} GB)")
    return len(cif_files), total_bytes


def cleanup_existing_cif(data_dir, dry_run=False):
    """Walk all *_v6 subdirectories and delete .cif files."""
    print("\nCIF Cleanup")
    print("=" * 50)

    v6_dirs = sorted(glob(os.path.join(data_dir, "*_v6")))
    if not v6_dirs:
        print("  No *_v6 directories found")
        return

    total_files = 0
    total_bytes = 0
    for d in v6_dirs:
        dirname = os.path.basename(d)
        cif_files = glob(os.path.join(d, "*.cif"))
        if not cif_files:
            print(f"  {dirname}: no .cif files")
            continue

        nbytes = sum(os.path.getsize(f) for f in cif_files)
        if dry_run:
            print(f"  {dirname}: would delete {len(cif_files)} .cif files "
                  f"({nbytes / 1024**3:.1f} GB)")
        else:
            print(f"  {dirname}: deleting {len(cif_files)} .cif files "
                  f"({nbytes / 1024**3:.1f} GB)...")
            for f in cif_files:
                os.remove(f)

        total_files += len(cif_files)
        total_bytes += nbytes

    action = "Would free" if dry_run else "Freed"
    print(f"\n  Total: {total_files} .cif files, {action} {total_bytes / 1024**3:.1f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download and manage AFDB v6 datasets'
    )
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                        help=f'Data directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--datasets', default='all',
                        help='Which datasets to download: all, proteomes, swissprot, tsv, '
                             'or a specific dataset name')
    parser.add_argument('--cleanup-cif', action='store_true',
                        help='Clean up .cif files from existing directories (no download)')
    parser.add_argument('--keep-cif', action='store_true',
                        help='Keep .cif files after extraction (default: delete them)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')

    args = parser.parse_args()
    data_dir = args.data_dir

    os.makedirs(data_dir, exist_ok=True)

    # CIF cleanup mode
    if args.cleanup_cif:
        cleanup_existing_cif(data_dir, dry_run=args.dry_run)
        return

    choice = args.datasets.lower()
    delete_cif = not args.keep_cif

    # Determine which tarballs to download
    if choice in ('all', 'proteomes'):
        print("\nProteome Downloads")
        print("=" * 50)
        for ds in PROTEOME_DATASETS:
            download_and_extract(ds, data_dir, delete_cif=delete_cif, dry_run=args.dry_run)

    if choice in ('all', 'swissprot'):
        print("\nSwiss-Prot Download")
        print("=" * 50)
        download_and_extract(SWISSPROT_DATASET, data_dir, delete_cif=delete_cif,
                             dry_run=args.dry_run)

    if choice in ('all', 'tsv'):
        print("\nTSV-based Downloads")
        print("=" * 50)
        for tsv_ds in TSV_DATASETS:
            tsv_path = os.path.join(data_dir, tsv_ds["tsv"])
            if not os.path.exists(tsv_path):
                print(f"  WARNING: TSV not found: {tsv_path}, skipping {tsv_ds['name']}")
                continue
            pdb_dir = os.path.join(data_dir, tsv_ds["pdb_dir"])
            print(f"  {tsv_ds['name']}:")
            download_tsv_pdbs(tsv_path, pdb_dir, version="v6", dry_run=args.dry_run)

    # Handle specific dataset name
    if choice not in ('all', 'proteomes', 'swissprot', 'tsv'):
        found = False
        for ds in PROTEOME_DATASETS + [SWISSPROT_DATASET]:
            if choice in ds["name"].lower() or choice in ds["url"].lower():
                download_and_extract(ds, data_dir, delete_cif=delete_cif, dry_run=args.dry_run)
                found = True
                break
        for tsv_ds in TSV_DATASETS:
            if choice in tsv_ds["name"].lower():
                tsv_path = os.path.join(data_dir, tsv_ds["tsv"])
                pdb_dir = os.path.join(data_dir, tsv_ds["pdb_dir"])
                download_tsv_pdbs(tsv_path, pdb_dir, version="v6", dry_run=args.dry_run)
                found = True
                break
        if not found:
            print(f"Unknown dataset: {choice}")
            print("Available: all, proteomes, swissprot, tsv, or a dataset name")
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
