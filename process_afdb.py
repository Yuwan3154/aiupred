#!/usr/bin/env python3
"""
Process a single AFDB proteome directory with AIUPred.

PDB files must already be downloaded (use download_afdb.py for downloads).

Usage:
    python process_afdb.py \
        --pdb_dir /home/jupyter-chenxi/data/afdb/UP000000625_83333_ECOLI_v6 \
        --result_file /home/jupyter-chenxi/data/afdb/UP000000625_83333_ECOLI_v6.csv \
        --gpu 0 \
        --reuse_csv /home/jupyter-chenxi/data/afdb/UP000000625_83333_ECOLI_v4.csv
"""

import os
import sys
import argparse
import warnings
import logging

warnings.filterwarnings("ignore")

# Add aiupred directory to path so aiupred_lib is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
from Bio.PDB import PDBParser

import torch
from aiupred_lib import init_models, batch_predict_disorder, WINDOW

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# AFDB returns a 127-byte HTML "not found" page for missing models
AFDB_NOT_FOUND_SIZE = 127


def parse_pdb(pdb_file):
    """
    Parse a PDB file and return (sequence, plddt_array).
    plddt values are in 0-100 range (raw B-factors from AlphaFold PDB).
    Returns (None, None) on error.
    """
    try:
        parser = PDBParser(PERMISSIVE=1)
        accession_id = "-".join(Path(pdb_file).name.split(".")[0].split("-")[1:3])
        structure = parser.get_structure(accession_id, pdb_file)
        model = structure[0]
        chain = model["A"]

        sequence = []
        plddt = []
        for residue in chain:
            aa = THREE_TO_ONE.get(residue.resname, 'X')
            sequence.append(aa)
            plddt.append(residue['CA'].get_bfactor())

        return ''.join(sequence), np.array(plddt)
    except Exception as e:
        logging.warning(f"Error parsing {pdb_file}: {e}")
        return None, None


def build_reuse_cache(reuse_csv):
    """
    Build a {sequence_str -> np.ndarray} lookup from an existing result CSV.
    Only entries with a non-null disorder_score are included.
    """
    if not reuse_csv or not os.path.exists(reuse_csv):
        return {}
    print(f"Loading reuse cache from {reuse_csv}...")
    df = pd.read_csv(reuse_csv)
    cache = {}
    for _, row in df.iterrows():
        try:
            seq = row['sequence']
            disorder = np.array(eval(row['disorder_score']))
            cache[seq] = disorder
        except Exception:
            pass
    print(f"  Loaded {len(cache)} sequences into reuse cache")
    return cache



def emit_result(results_list, accession_id, sequence, plddt_raw, disorder_score):
    """
    Compute summary stats and append a result dict to results_list.
    plddt_raw is 0-100 scale (from B-factors); stored as 0-1.
    """
    plddt = plddt_raw / 100.0
    order_mask = np.array(disorder_score) <= 0.5
    mean_plddt = float(np.mean(plddt))
    if order_mask.any():
        masked_mean_plddt = float(np.mean(plddt[order_mask]))
    else:
        masked_mean_plddt = float('nan')
    mean_disorder_score = float(np.mean(disorder_score))

    results_list.append({
        'accession_id': accession_id,
        'sequence': sequence,
        'disorder_score': str(np.around(disorder_score, 4).tolist()),
        'plddt': str(np.around(plddt, 4).tolist()),
        'mean_plddt': mean_plddt,
        'masked_mean_plddt': masked_mean_plddt,
        'mean_disorder_score': mean_disorder_score,
    })


def save_results(results_list, result_file, existing_df):
    """Concat existing_df with results_list and save to CSV."""
    new_df = pd.DataFrame(results_list)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(result_file, index=False)
    return combined


def process_pdb_dir(pdb_dir, result_file, embedding_model, reg_model, device,
                    reuse_cache=None, inference_batch=256,
                    window_batch_size=4096, save_interval=100,
                    version="v6"):
    """
    Process all v{version} PDB files in pdb_dir, saving results to result_file.

    Uses batch_predict_disorder for efficient GPU inference and optionally
    skips AIUPred for sequences already present in reuse_cache.
    """
    if reuse_cache is None:
        reuse_cache = {}

    pdb_files = sorted(glob(os.path.join(pdb_dir, f"*.pdb")))
    # Filter out any non-v6 files and 127-byte stubs
    pdb_files = [f for f in pdb_files
                 if f"model_{version}.pdb" in f
                 and os.path.getsize(f) > AFDB_NOT_FOUND_SIZE]
    print(f"Number of {version} PDB files: {len(pdb_files)}")

    # Load existing results to support incremental resumption
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
        processed_ids = set(existing_df['accession_id'].values)
        print(f"  Already processed: {len(processed_ids)} entries")
    else:
        existing_df = pd.DataFrame(columns=[
            'accession_id', 'sequence', 'disorder_score',
            'plddt', 'mean_plddt', 'masked_mean_plddt', 'mean_disorder_score'
        ])
        processed_ids = set()

    pdb_files = [f for f in pdb_files
                 if "-".join(Path(f).name.split(".")[0].split("-")[1:3]) not in processed_ids]
    print(f"  Files to process: {len(pdb_files)}")

    if not pdb_files:
        print("Nothing to process.")
        return

    new_results = []
    pending = []  # list of (accession_id, sequence, plddt_raw)
    reused = 0
    inferred = 0

    def flush_pending():
        nonlocal inferred
        if not pending:
            return
        seqs = [s for _, s, _ in pending]
        total_residues = sum(len(s) for s in seqs)
        if device.type == 'cuda':
            mem_before = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  [batch] Running AIUPred on {len(seqs)} seqs "
              f"({total_residues} residues, {total_residues * (WINDOW + 1)} windows)")
        disorders = batch_predict_disorder(
            seqs, embedding_model, reg_model, device,
            no_smoothing=True, window_batch_size=window_batch_size
        )
        if device.type == 'cuda':
            mem_after = torch.cuda.memory_allocated(device) / 1024**2
            print(f"  [batch] GPU mem: {mem_before:.0f}MB -> {mem_after:.0f}MB")
        for (acc, seq, plddt_raw), disorder in zip(pending, disorders):
            emit_result(new_results, acc, seq, plddt_raw, disorder)
            inferred += 1
        pending.clear()

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        accession_id = "-".join(Path(pdb_file).name.split(".")[0].split("-")[1:3])
        sequence, plddt_raw = parse_pdb(pdb_file)
        if sequence is None:
            continue

        if sequence in reuse_cache:
            emit_result(new_results, accession_id, sequence, plddt_raw, reuse_cache[sequence])
            reused += 1
        else:
            pending.append((accession_id, sequence, plddt_raw))

        if len(pending) >= inference_batch:
            flush_pending()

        # Incremental save
        total_new = len(new_results)
        if total_new > 0 and total_new % save_interval == 0:
            existing_df = save_results(new_results, result_file, existing_df)
            new_results.clear()

    flush_pending()

    # Final save
    if new_results:
        existing_df = save_results(new_results, result_file, existing_df)

    print(f"Done. Reused from cache: {reused}, New AIUPred inferences: {inferred}")
    print(f"Results saved to {result_file} ({len(existing_df)} total rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Process a single AFDB proteome with AIUPred (v6)'
    )
    parser.add_argument('--pdb_dir', required=True,
                        help='Directory containing v6 PDB files')
    parser.add_argument('--result_file', required=True,
                        help='Output CSV file path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (default: 0)')
    parser.add_argument('--window_batch_size', type=int, default=4096,
                        help='Number of 101-token windows per GPU mini-batch (default: 4096)')
    parser.add_argument('--inference_batch', type=int, default=256,
                        help='Number of proteins per batch_predict_disorder call (default: 256)')
    parser.add_argument('--reuse_csv', default=None,
                        help='Existing result CSV to reuse AIUPred disorder scores from')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save CSV every N newly processed proteins (default: 100)')
    parser.add_argument('--dataset_name', default='',
                        help='Label for log messages')

    args = parser.parse_args()

    label = args.dataset_name or os.path.basename(args.pdb_dir)
    print(f"Starting processing of: {label}")
    print(f"  PDB dir:     {args.pdb_dir}")
    print(f"  Result file: {args.result_file}")
    if args.reuse_csv:
        print(f"  Reuse CSV:   {args.reuse_csv}")

    # Step 1: Build reuse cache
    reuse_cache = build_reuse_cache(args.reuse_csv)

    # Step 2: Initialize models
    print(f"Initializing models on GPU {args.gpu}...")
    embedding_model, reg_model, device = init_models(force_cpu=False, gpu_num=args.gpu)

    # Step 3: Process
    process_pdb_dir(
        pdb_dir=args.pdb_dir,
        result_file=args.result_file,
        embedding_model=embedding_model,
        reg_model=reg_model,
        device=device,
        reuse_cache=reuse_cache,
        inference_batch=args.inference_batch,
        window_batch_size=args.window_batch_size,
        save_interval=args.save_interval,
        version="v6",
    )

    print(f"Completed: {label}")


if __name__ == "__main__":
    main()
