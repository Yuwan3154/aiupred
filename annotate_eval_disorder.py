"""Annotate evaluation-set structures with AIUPred disorder + ordered segments.

Eval CSVs have no sequence column; sequences are extracted from the GT structures
(.cif, sharded as pdb/<pdbid[1:3]>/<pdbid>.cif). Per row: extract the chain sequence,
run AIUPred predict_disorder, and compute ordered_segments via the SAME shared function
the AFDB filtering uses (build_functional_target_list.ordered_segments_info). The raw
disorder_score is stored so a downstream comparison can recompute segments at any cutoff.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aiupred_lib
from build_functional_target_list import ordered_segments_info, ordered_run_info

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def resolve_structure(struct_dir, idval):
    """id '6UF2_A'/'1a2y_C' -> ({struct_dir}/pdb/<pdbid[1:3]>/<pdbid>.cif, chain)."""
    pdbid, _, chain = str(idval).partition("_")
    return os.path.join(struct_dir, "pdb", pdbid[1:3], pdbid + ".cif"), (chain or "A")


def extract_sequence(cif_path, chain):
    """Modeled-residue sequence of `chain` (fallback: first chain) via BioPython."""
    structure = MMCIFParser(QUIET=True).get_structure("x", cif_path)
    target = None
    for model in structure:
        for c in model:
            if c.id == chain:
                target = c
                break
        if target is not None:
            break
    if target is None:
        for model in structure:
            for c in model:
                target = c
                break
            break
    seq = []
    for res in target:
        rn = res.get_resname().strip()
        if rn in THREE_TO_ONE:
            seq.append(THREE_TO_ONE[rn])
        elif res.id[0] == " ":
            seq.append("X")
    return "".join(seq)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id_col", required=True, help="e.g. 'pdb' (bad_afdb) or 'natives_rcsb' (af2rank)")
    ap.add_argument("--struct_dir", required=True, help="dir containing pdb/<shard>/<pdbid>.cif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="disorder cutoff; ordered = score <= threshold")
    ap.add_argument("--min_run", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="annotate only the first N rows (0 = all)")
    ap.add_argument("--no_smoothing", action="store_true", help="disable AIUPred savgol smoothing (default: smoothing ON)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit).copy()
    em, rm, dev = aiupred_lib.init_models(gpu_num=args.gpu)

    seqs, disorders, segs, nsegs, longest, seqlens = [], [], [], [], [], []
    n_missing = 0
    for idv in df[args.id_col]:
        cif, chain = resolve_structure(args.struct_dir, idv)
        if not os.path.exists(cif):
            print("WARNING: missing structure for %s: %s" % (idv, cif))
            n_missing += 1
            seqs.append(""); disorders.append(""); segs.append("[]"); nsegs.append(0); longest.append(0); seqlens.append(0)
            continue
        seq = extract_sequence(cif, chain)
        ds = np.asarray(aiupred_lib.predict_disorder(seq, em, rm, dev, no_smoothing=args.no_smoothing), dtype=float)
        seg = ordered_segments_info(ds, args.threshold, args.min_run)
        seqs.append(seq)
        disorders.append(json.dumps([round(float(x), 4) for x in ds]))
        segs.append(json.dumps(seg))
        nsegs.append(len(seg))
        longest.append(int(ordered_run_info(ds, args.threshold)[0]))
        seqlens.append(len(seq))

    df["sequence"] = seqs
    df["disorder_score"] = disorders
    df["ordered_segments"] = segs
    df["n_ordered_segments"] = nsegs
    df["longest_ordered_run"] = longest
    df["struct_seqlen"] = seqlens
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote %d rows to %s (threshold=%.2f, min_run=%d, missing=%d)" % (len(df), args.out, args.threshold, args.min_run, n_missing))


if __name__ == "__main__":
    main()
