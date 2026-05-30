#!/usr/bin/env python3
"""
Pack AlphaFold mmCIF files for entries in a build_filtered_target_list.py CSV into a tarball.

Filenames match AFDB HTTPS layout (same as aiupred.ipynb / process_afdb PDB names, .cif):
    AF-{accession_id}-model_{version}.cif

Example URL:
    https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v6.cif

Usage:
    conda activate aiupred
    python pack_filtered_cifs.py \\
        --csv /path/to/afdb_model_org_plddt-05_aiupred-02.csv \\
        --output_tar filtered_structures.tar.gz \\
        --search_dirs /home/jupyter-chenxi/data/afdb/UP000005640_9606_HUMAN_v6 \\
        --download_missing \\
        --compress gzip
"""

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request

from tqdm import tqdm

import fireducks.pandas as pd

AFDB_NOT_FOUND_SIZE = 127
AFDB_FILES_BASE = "https://alphafold.ebi.ac.uk/files"


def cif_filename(accession_id: str, version: str) -> str:
    return f"AF-{accession_id}-model_{version}.cif"


def cif_url(accession_id: str, version: str) -> str:
    return f"{AFDB_FILES_BASE}/{cif_filename(accession_id, version)}"


def find_local_cif(accession_id: str, version: str, search_dirs):
    basename = cif_filename(accession_id, version)
    for d in search_dirs:
        path = os.path.join(d, basename)
        if os.path.isfile(path) and os.path.getsize(path) > AFDB_NOT_FOUND_SIZE:
            return path
    return None


def fetch_cif_to_path(url: str, dest_path: str):
    urllib.request.urlretrieve(url, dest_path)
    size = os.path.getsize(dest_path)
    if size <= AFDB_NOT_FOUND_SIZE:
        os.remove(dest_path)
        sys.stderr.write(f"ERROR: not found or AFDB stub ({size} B): {url}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Collect AFDB mmCIF files for a filtered accession CSV and write a tar archive."
    )
    parser.add_argument("--csv", required=True, help="Filtered list CSV from build_filtered_target_list.py")
    parser.add_argument("--output_tar", required=True, help="Output .tar or .tar.gz path")
    parser.add_argument(
        "--version",
        default="v6",
        help="AlphaFold model version suffix in filenames (default: v6)",
    )
    parser.add_argument(
        "--search_dirs",
        action="append",
        default=[],
        help="Directory(ies) to search for AF-*-model_*.cif before download (repeatable)",
    )
    parser.add_argument(
        "--download_missing",
        action="store_true",
        help="Download from alphafold.ebi.ac.uk when file is not found under search_dirs",
    )
    parser.add_argument(
        "--compress",
        choices=("none", "gzip"),
        default="gzip",
        help="gzip compress the tar (default: gzip); use 'none' for uncompressed .tar",
    )
    args = parser.parse_args()

    if not args.search_dirs and not args.download_missing:
        print("Error: provide --search_dirs and/or --download_missing.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if "accession_id" not in df.columns:
        print("Error: CSV must contain an 'accession_id' column.", file=sys.stderr)
        sys.exit(1)

    ids = df["accession_id"].astype(str).drop_duplicates().tolist()
    open_mode = "w:gz" if args.compress == "gzip" else "w"

    with tempfile.TemporaryDirectory(prefix="pack_cifs_") as tmp_root:
        paths_added = []

        for accession_id in tqdm(ids, desc="Resolving mmCIF files"):
            local = find_local_cif(accession_id, args.version, args.search_dirs) if args.search_dirs else None

            if local is not None:
                paths_added.append((local, cif_filename(accession_id, args.version)))
                continue

            if not args.download_missing:
                sys.stderr.write(
                    f"Error: missing {cif_filename(accession_id, args.version)} (no download)\n"
                )
                sys.exit(1)

            staged = os.path.join(tmp_root, cif_filename(accession_id, args.version))
            url = cif_url(accession_id, args.version)
            fetch_cif_to_path(url, staged)
            paths_added.append((staged, cif_filename(accession_id, args.version)))

        out_path = args.output_tar
        if args.compress == "gzip" and not out_path.endswith(".gz"):
            out_path = out_path + ".gz"

        with tarfile.open(out_path, open_mode) as tar:
            for fs_path, arcname in tqdm(paths_added, desc="Writing tar"):
                tar.add(fs_path, arcname=arcname, recursive=False)

        print(f"Wrote {len(paths_added)} file(s) to {out_path}")


if __name__ == "__main__":
    main()
