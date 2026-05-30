#!/usr/bin/env python3
"""
Example script demonstrating the use of aiupred_masked_mean_plddt with multiprocessing support.
"""

import torch
from aiupred_lib import init_models, aiupred_masked_mean_plddt

def main():
    # Initialize models and device
    embedding_model, reg_model, device = init_models(force_cpu=True, gpu_num=0)
    
    # Define parameters
    pdb_dir = "/home/jupyter-chenxi/data/bfvd/pdb"
    result_file = "/home/jupyter-chenxi/data/bfvd/bfvd.csv"
    
    # # Example 1: Use all available CPUs (default for CPU device)
    # print("=== Example 1: Using all available CPUs ===")
    # aiupred_masked_mean_plddt(pdb_dir, result_file, embedding_model, reg_model, device, extract_accession_id=False)
    
    # Example 2: Use specific number of CPUs
    print("\n=== Example 2: Using 16 CPUs ===")
    aiupred_masked_mean_plddt(pdb_dir, result_file, embedding_model, reg_model, device, num_cpus=16, extract_accession_id=False)
    
    # # Example 3: Force sequential processing (even on CPU)
    # print("\n=== Example 3: Sequential processing (GPU-like behavior) ===")
    # # This will use sequential processing even on CPU
    # aiupred_masked_mean_plddt(pdb_dir, result_file, embedding_model, reg_model, device, num_cpus=1, extract_accession_id=False)

if __name__ == "__main__":
    main() 