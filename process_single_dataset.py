#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add aiupred directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fireducks.pandas as pd
import numpy as np
from pathlib import Path
from aiupred_lib import init_models, predict_disorder
import torch
from Bio.PDB import PDBParser
from glob import glob
from tqdm import tqdm

def aiupred_masked_mean_plddt(pdb_dir, result_file, embedding_model, reg_model, device):
    """
    Process the PDB files in the given directory and save the results to a CSV file.
    
    Args:
        pdb_dir (str): The directory containing the PDB files.
        result_file (str): The path to the CSV file to save the results.
    """
    # Convert three-letter codes to one-letter codes
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    pdb_files = glob(os.path.join(pdb_dir, "*.pdb"))
    print(f"Number of PDB files: {len(pdb_files)}")

    if os.path.exists(result_file):
        results_df = pd.read_csv(result_file)
        # Get list of already processed accession IDs
        processed_ids = set(results_df['accession_id'].values)
    else:
        results_df = pd.DataFrame(columns=['accession_id', 'sequence', 'disorder_score', 
                                        'plddt', 'mean_plddt', 'masked_mean_plddt'])
        processed_ids = set()
    
    for pdb_file in tqdm(pdb_files):
        accession_id = "-".join(Path(pdb_file).name.split(".")[0].split("-")[1:3])
        if accession_id in processed_ids:
            continue

        try:
            parser = PDBParser(PERMISSIVE=1)
            structure = parser.get_structure(accession_id, pdb_file)

            # Get the first model and chain (AlphaFold PDBs typically have only one model and chain)
            model = structure[0]
            chain = model["A"]

            # Extract sequence and B-factors
            sequence = []
            plddt = []

            for residue in chain:
                # Get the residue's one-letter code
                sequence.append(three_to_one[residue.resname])
                
                # Get CA atom's B-factor (using CA as it's present in all amino acids)
                ca_atom = residue['CA']
                plddt.append(ca_atom.get_bfactor())

            sequence = ''.join(sequence)
            disorder_score = predict_disorder(sequence, embedding_model, reg_model, device, no_smoothing=True)
            order_mask = np.array(disorder_score) <= 0.5
            plddt = np.array(plddt) / 100
            mean_plddt = np.mean(plddt)
            masked_mean_plddt = np.mean(plddt[order_mask])
            mean_disorder_score = np.mean(disorder_score)
            
            # Add to DataFrame
            new_row_df = pd.DataFrame({
                'accession_id': [accession_id],
                'sequence': [sequence],
                'disorder_score': [str(disorder_score.tolist())],
                'plddt': [str(plddt.tolist())],
                'mean_plddt': [mean_plddt],
                'masked_mean_plddt': [masked_mean_plddt],
                'mean_disorder_score': [mean_disorder_score],
            })
            
            results_df = pd.concat([results_df, new_row_df], ignore_index=True, axis=0, verify_integrity=True)
            
            # Save after every 100 proteins (you can adjust this number)
            if len(results_df) % 100 == 0:
                results_df.to_csv(result_file, index=False)

        except Exception as e:
            print(f"Error processing {accession_id}: {e}")
    
    results_df["disorder_score"] = results_df["disorder_score"].apply(lambda x: str(np.around(np.array(eval(x)), 4).tolist()))
    results_df["plddt"] = results_df["plddt"].apply(lambda x: str(np.around(np.array(eval(x)), 4).tolist()))

    results_df.to_csv(result_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process a single dataset with AIUPred')
    parser.add_argument('--pdb_dir', required=True, help='PDB directory to process')
    parser.add_argument('--result_file', required=True, help='Output CSV file')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--dataset_name', required=True, help='Dataset name for logging')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting processing of {args.dataset_name} on GPU {args.gpu_id}")
    print(f"   PDB dir: {args.pdb_dir}")
    print(f"   Result file: {args.result_file}")
    
    try:
        # Initialize models
        print(f"🔧 Initializing models on GPU {args.gpu_id}...")
        embedding_model, reg_model, device = init_models(force_cpu=False, gpu_num=0)
        
        # Process the dataset
        print(f"⚙️ Processing {args.dataset_name}...")
        aiupred_masked_mean_plddt(args.pdb_dir, args.result_file, embedding_model, reg_model, device)
        
        print(f"✅ Completed processing of {args.dataset_name}")
        
    except Exception as e:
        print(f"❌ Error processing {args.dataset_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
