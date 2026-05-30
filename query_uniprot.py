#!/usr/bin/env python3
"""
Query UniProtKB REST API and generate a TSV file for a given organism.

Produces TSV files compatible with the AFDB processing pipeline, matching the
format of uniprotkb_Nematocida_parisii_2025_04_25.tsv.

Usage:
    # By organism taxonomy ID
    python query_uniprot.py --organism_id 508771 --organism_name Nematocida_parisii_ERTm3

    # By custom query (e.g. proteome ID for T. gondii RH-88)
    python query_uniprot.py --query "proteome:UP000557509" --organism_name Toxoplasma_gondii_RH88
"""

import os
import sys
import argparse
import time
import re
import urllib.request
import urllib.error
from datetime import date

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,reviewed,id,protein_name,gene_names,organism_name,length,sequence"
PAGE_SIZE = 500
MAX_RETRIES = 3
DEFAULT_OUTPUT_DIR = "/home/jupyter-chenxi/data/afdb"


def fetch_page(url, retry=0):
    """Fetch a single page from the UniProt REST API. Returns (content, next_url)."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "text/tab-separated-values")

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read().decode("utf-8")
            # Parse Link header for pagination cursor
            link_header = response.headers.get("Link", "")
            next_url = None
            if link_header:
                match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
                if match:
                    next_url = match.group(1)
            return content, next_url
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        if retry < MAX_RETRIES:
            wait = 2 ** (retry + 1)
            print(f"  Retry {retry + 1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
            return fetch_page(url, retry + 1)
        raise


def query_uniprot(query_str, output_file):
    """
    Query UniProtKB and write results to a TSV file.
    query_str can be e.g. "organism_id:508771" or "proteome:UP000557509".
    Returns the total number of entries written.
    """
    url = (f"{BASE_URL}?query={query_str}"
           f"&format=tsv&fields={FIELDS}&size={PAGE_SIZE}")

    total_entries = 0
    page = 0

    with open(output_file, "w") as f:
        while url:
            page += 1
            content, next_url = fetch_page(url)
            lines = content.rstrip("\n").split("\n")

            if page == 1:
                # Write header + data
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
                total_entries += len(lines) - 1  # subtract header
                print(f"  Page {page}: {len(lines) - 1} entries (header: {lines[0][:80]}...)")
            else:
                # Skip header, write data only
                data_lines = lines[1:]
                for line in data_lines:
                    f.write(line + "\n")
                total_entries += len(data_lines)
                print(f"  Page {page}: {len(data_lines)} entries ({total_entries} total)")

            url = next_url

    return total_entries


def main():
    parser = argparse.ArgumentParser(
        description='Query UniProtKB and generate a TSV file for an organism'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--organism_id', type=int,
                       help='UniProt taxonomy ID (e.g. 508771 for N. parisii ERTm3)')
    group.add_argument('--query',
                       help='Custom UniProt query (e.g. "proteome:UP000557509")')
    parser.add_argument('--organism_name', required=True,
                        help='Organism name for filename (e.g. Toxoplasma_gondii_RH88)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--date', default=None,
                        help='Date string for filename (default: today, YYYY_MM_DD)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output file')

    args = parser.parse_args()

    date_str = args.date or date.today().strftime("%Y_%m_%d")
    filename = f"uniprotkb_{args.organism_name}_{date_str}.tsv"
    output_file = os.path.join(args.output_dir, filename)

    if os.path.exists(output_file) and not args.force:
        print(f"Output file already exists: {output_file}")
        print("Use --force to overwrite")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.organism_id:
        query_str = f"organism_id:{args.organism_id}"
    else:
        query_str = args.query

    print(f"Querying UniProtKB: {query_str} ({args.organism_name})")
    print(f"Output: {output_file}")

    total = query_uniprot(query_str, output_file)

    size_kb = os.path.getsize(output_file) / 1024
    print(f"\nDone. Wrote {total} entries to {filename} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
