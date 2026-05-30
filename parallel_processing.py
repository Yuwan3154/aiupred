#!/usr/bin/env python3
"""
Parallel processing script for AIUPred analysis across multiple datasets.
Uses 4 GPUs to process datasets in parallel, with automatic GPU management.
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from glob import glob
from queue import Queue
import threading
import signal
from pathlib import Path
import urllib.request
import tarfile
from tqdm import tqdm

# Dataset configuration
DATASETS = [
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar",
        "name": "Human"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI_v4.tar",
        "name": "E. coli"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001940_6239_CAEEL_v4.tar",
        "name": "C. elegans"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006548_3702_ARATH_v4.tar",
        "name": "A. thaliana"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000559_237561_CANAL_v4.tar",
        "name": "C. albicans"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000437_7955_DANRE_v4.tar",
        "name": "D. rerio"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002195_44689_DICDI_v4.tar",
        "name": "D. discoideum"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000803_7227_DROME_v4.tar",
        "name": "D. melanogaster"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008827_3847_SOYBN_v4.tar",
        "name": "G. max"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v4.tar",
        "name": "M. jannaschii"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v4.tar",
        "name": "M. musculus"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000059680_39947_ORYSJ_v4.tar",
        "name": "O. sativa"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002494_10116_RAT_v4.tar",
        "name": "R. norvegicus"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002311_559292_YEAST_v4.tar",
        "name": "S. cerevisiae"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v4.tar",
        "name": "S. pombe"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000007305_4577_MAIZE_v4.tar",
        "name": "Z. mays"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008854_6183_SCHMA_v4.tar",
        "name": "S. mansoni"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001450_36329_PLAF7_v4.tar",
        "name": "P. falciparum"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001631_447093_AJECG_v4.tar",
        "name": "A. capsulatus"
    },
    {
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000806_272631_MYCLE_v4.tar",
        "name": "M. leprae"
    }
]

# Configuration
NUM_GPUS = 4
DATA_DIR = "/home/jupyter-chenxi/data/afdb"
AIUPRED_DIR = "/home/jupyter-chenxi/aiupred"

class GPUManager:
    """Manages GPU availability and assignment"""
    
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.available_gpus = Queue()
        self.running_processes = {}
        self.lock = threading.Lock()
        
        # Initialize available GPUs
        for i in range(num_gpus):
            self.available_gpus.put(i)
    
    def get_gpu(self):
        """Get an available GPU (blocks if none available)"""
        return self.available_gpus.get()
    
    def release_gpu(self, gpu_id):
        """Release a GPU back to the available pool"""
        with self.lock:
            self.available_gpus.put(gpu_id)
    
    def add_process(self, gpu_id, process, dataset_info):
        """Track a running process"""
        with self.lock:
            self.running_processes[gpu_id] = {
                'process': process,
                'dataset': dataset_info
            }
    
    def remove_process(self, gpu_id):
        """Remove a finished process"""
        with self.lock:
            if gpu_id in self.running_processes:
                del self.running_processes[gpu_id]
    
    def get_running_processes(self):
        """Get list of currently running processes"""
        with self.lock:
            return list(self.running_processes.items())

def download_and_extract(dataset, data_dir):
    """Download and extract a dataset if not already present"""
    url = dataset["url"]
    name = dataset["name"]
    filename = url.split('/')[-1]
    pdb_dir = filename.split('.')[0]
    
    filepath = os.path.join(data_dir, filename)
    extract_dir = os.path.join(data_dir, pdb_dir)
    
    # Check if already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"✓ {name} ({pdb_dir}) already extracted")
        return pdb_dir
    
    # Download if not present
    if not os.path.exists(filepath):
        print(f"📥 Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            return None
    else:
        print(f"✓ {name} already downloaded")
    
    # Extract
    print(f"📦 Extracting {name}...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(path=extract_dir)
        print(f"✓ Extracted {name}")
        return pdb_dir
    except Exception as e:
        print(f"❌ Failed to extract {name}: {e}")
        return None

def create_processing_script():
    """Create the individual processing script that will be run for each dataset"""
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = os.path.join(AIUPRED_DIR, "process_single_dataset.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    return script_path

def run_dataset_processing(gpu_manager, pdb_dir, dataset_name, data_dir):
    """Run processing for a single dataset on an available GPU"""
    
    # Get an available GPU
    gpu_id = gpu_manager.get_gpu()
    
    try:
        print(f"🎯 Assigned {dataset_name} to GPU {gpu_id}")
        
        # Create processing script if it doesn't exist
        script_path = create_processing_script()
        
        # Set up paths
        pdb_path = os.path.join(data_dir, pdb_dir)
        result_file = os.path.join(data_dir, f"{pdb_dir}.csv")
        
        # Run the processing script
        cmd = [
            sys.executable, script_path,
            '--pdb_dir', pdb_path,
            '--result_file', result_file,
            '--gpu_id', str(gpu_id),
            '--dataset_name', dataset_name
        ]
        
        print(f"🔧 Starting subprocess for {dataset_name} on GPU {gpu_id}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        )
        
        # Track the process
        gpu_manager.add_process(gpu_id, process, {'name': dataset_name, 'pdb_dir': pdb_dir})
        
        # Wait for completion and capture output
        output, _ = process.communicate()
        
        if process.returncode == 0:
            print(f"✅ Successfully completed {dataset_name} on GPU {gpu_id}")
        else:
            print(f"❌ Failed processing {dataset_name} on GPU {gpu_id}")
            print(f"Output: {output}")
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {dataset_name} on GPU {gpu_id}: {e}")
        return False
    
    finally:
        # Always release the GPU and remove process tracking
        gpu_manager.remove_process(gpu_id)
        gpu_manager.release_gpu(gpu_id)

def monitor_progress(gpu_manager):
    """Monitor and display progress of running processes"""
    while True:
        running = gpu_manager.get_running_processes()
        if running:
            print(f"\n📊 Currently running ({len(running)}/{NUM_GPUS} GPUs busy):")
            for gpu_id, info in running:
                print(f"   GPU {gpu_id}: {info['dataset']['name']}")
        time.sleep(60)

def main():
    print("🚀 AIUPred Parallel Processing Pipeline")
    print(f"📊 Configuration: {NUM_GPUS} GPUs, {len(DATASETS)} datasets")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🔬 AIUPred directory: {AIUPRED_DIR}")
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)
    
    # Initialize GPU manager
    gpu_manager = GPUManager(NUM_GPUS)
    
    # Start progress monitor in background
    monitor_thread = threading.Thread(target=monitor_progress, args=(gpu_manager,), daemon=True)
    monitor_thread.start()
    
    # Phase 1: Download and extract all datasets
    print("\n📥 Phase 1: Download and Extract")
    print("=" * 50)
    
    valid_datasets = []
    for dataset in DATASETS:
        pdb_dir = download_and_extract(dataset, DATA_DIR)
        if pdb_dir:
            valid_datasets.append((pdb_dir, dataset["name"]))
    
    print(f"\n✅ Successfully prepared {len(valid_datasets)} datasets")
    
    # Phase 2: Process datasets in parallel
    print("\n⚙️ Phase 2: Parallel Processing")
    print("=" * 50)
    
    # Use ThreadPoolExecutor to manage parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_dataset_wrapper(dataset_info):
        pdb_dir, dataset_name = dataset_info
        return run_dataset_processing(gpu_manager, pdb_dir, dataset_name, DATA_DIR)
    
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(process_dataset_wrapper, dataset_info): dataset_info 
            for dataset_info in valid_datasets
        }
        
        # Process completions
        for future in as_completed(future_to_dataset):
            dataset_info = future_to_dataset[future]
            pdb_dir, dataset_name = dataset_info
            
            try:
                success = future.result()
                if success:
                    completed += 1
                    print(f"✅ Completed {dataset_name} ({completed}/{len(valid_datasets)})")
                else:
                    failed += 1
                    print(f"❌ Failed {dataset_name} ({failed} failures so far)")
            except Exception as e:
                failed += 1
                print(f"❌ Exception processing {dataset_name}: {e}")
    
    # Final summary
    print("\n🏁 Processing Complete!")
    print("=" * 50)
    print(f"✅ Successfully processed: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {completed/(completed+failed)*100:.1f}%")
    
    # List output files
    print(f"\n📄 Output files in {DATA_DIR}:")
    csv_files = glob(os.path.join(DATA_DIR, "*.csv"))
    for csv_file in sorted(csv_files):
        size = os.path.getsize(csv_file) / (1024*1024)  # MB
        print(f"   {os.path.basename(csv_file)} ({size:.1f} MB)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)