#!/usr/bin/env python3
"""
Parallel processing pipeline for AIUPred analysis across all AFDB v6 datasets.
Uses multiple GPUs to process proteome datasets in parallel.

Usage:
    python parallel_processing.py
"""

import os
import sys
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from queue import Queue
from tqdm import tqdm

from download_afdb import (
    PROTEOME_DATASETS, SWISSPROT_DATASET,
    download_and_extract, download_tsv_pdbs,
)

# Combined dataset list for tarball downloads (proteomes + Swiss-Prot)
DATASETS = PROTEOME_DATASETS + [SWISSPROT_DATASET]

# TSV-based datasets downloaded individually from AFDB
TSV_DATASETS = [
    {
        "tsv": "uniprotkb_Nematocida_parisii_2025_04_25.tsv",
        "name": "N. parisii",
        "pdb_dir": "uniprotkb_Nematocida_parisii_2025_04_25",
        "result_file": "uniprotkb_Nematocida_parisii_2025_04_25_v6.csv",
        "reuse_csv": "uniprotkb_Nematocida_parisii_2025_04_25.csv",
    },
    {
        "tsv": "uniprotkb_Toxoplasma_gondii_RH88_2026_03_20.tsv",
        "name": "T. gondii",
        "pdb_dir": "uniprotkb_Toxoplasma_gondii_RH88_2026_03_20",
        "result_file": "uniprotkb_Toxoplasma_gondii_RH88_2026_03_20_v6.csv",
        "reuse_csv": "",
    },
]

NUM_GPUS = 4
DATA_DIR = "/home/jupyter-chenxi/data/afdb"
AIUPRED_DIR = "/home/jupyter-chenxi/aiupred"


class GPUManager:
    """Manages GPU availability and assignment."""

    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.available_gpus = Queue()
        self.running_processes = {}
        self.lock = threading.Lock()
        for i in range(num_gpus):
            self.available_gpus.put(i)

    def get_gpu(self):
        return self.available_gpus.get()

    def release_gpu(self, gpu_id):
        with self.lock:
            self.available_gpus.put(gpu_id)

    def add_process(self, gpu_id, process, dataset_info):
        with self.lock:
            self.running_processes[gpu_id] = {'process': process, 'dataset': dataset_info}

    def remove_process(self, gpu_id):
        with self.lock:
            if gpu_id in self.running_processes:
                del self.running_processes[gpu_id]

    def get_running_processes(self):
        with self.lock:
            return list(self.running_processes.items())



def run_dataset_processing(gpu_manager, pdb_dir, dataset_name, data_dir,
                            tsv=None, extra_args=None):
    """Run process_afdb.py for a single dataset on an available GPU."""
    gpu_id = gpu_manager.get_gpu()
    try:
        print(f"  GPU {gpu_id} -> {dataset_name}")

        pdb_path = os.path.join(data_dir, pdb_dir)
        result_file = os.path.join(data_dir, f"{pdb_dir}.csv")

        # Check for old v4 CSV to reuse AIUPred disorder scores
        old_csv = os.path.join(data_dir, pdb_dir.replace('_v6', '_v4') + '.csv')

        script_path = os.path.join(AIUPRED_DIR, "process_afdb.py")
        aiupred_python = os.path.join(
            os.path.expanduser("~"), ".conda", "envs", "aiupred", "bin", "python"
        )
        python_exe = aiupred_python if os.path.exists(aiupred_python) else sys.executable
        cmd = [
            python_exe, script_path,
            '--pdb_dir', pdb_path,
            '--result_file', result_file,
            '--gpu', '0',  # CUDA_VISIBLE_DEVICES restricts to one GPU, always indexed as 0
            '--dataset_name', dataset_name,
        ]
        if tsv:
            cmd += ['--tsv', os.path.join(data_dir, tsv)]
        if os.path.exists(old_csv):
            cmd += ['--reuse_csv', old_csv]
        if extra_args:
            cmd += extra_args

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        )
        gpu_manager.add_process(gpu_id, process, {'name': dataset_name, 'pdb_dir': pdb_dir})

        output, _ = process.communicate()
        if process.returncode == 0:
            print(f"  Completed: {dataset_name} (GPU {gpu_id})")
        else:
            print(f"  FAILED: {dataset_name} (GPU {gpu_id})\n{output}")
        return process.returncode == 0

    except Exception as e:
        print(f"  Error processing {dataset_name} on GPU {gpu_id}: {e}")
        return False
    finally:
        gpu_manager.remove_process(gpu_id)
        gpu_manager.release_gpu(gpu_id)


def monitor_progress(gpu_manager, stop_event):
    """Periodically log running dataset progress."""
    while not stop_event.is_set():
        running = gpu_manager.get_running_processes()
        if running:
            print(f"\n[Progress] {len(running)}/{NUM_GPUS} GPUs busy: "
                  + ", ".join(f"GPU{g}: {i['dataset']['name']}" for g, i in running))
        stop_event.wait(60)


def main():
    print("AIUPred Parallel Processing Pipeline (v6)")
    print(f"  GPUs: {NUM_GPUS} | Datasets: {len(DATASETS)} proteomes + {len(TSV_DATASETS)} TSV")
    print(f"  Data dir: {DATA_DIR}")

    os.makedirs(DATA_DIR, exist_ok=True)

    gpu_manager = GPUManager(NUM_GPUS)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress, args=(gpu_manager, stop_event), daemon=True
    )
    monitor_thread.start()

    # Phase 0: TSV-based datasets (individual PDB downloads + processing)
    if TSV_DATASETS:
        print("\nPhase 0: TSV-based datasets")
        print("=" * 50)
        for tsv_ds in TSV_DATASETS:
            result_file = os.path.join(DATA_DIR, tsv_ds["result_file"])
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                print(f"  {tsv_ds['name']}: result CSV already exists, skipping")
                continue
            tsv_path = os.path.join(DATA_DIR, tsv_ds["tsv"])
            if not os.path.exists(tsv_path):
                print(f"  WARNING: TSV not found: {tsv_path}, skipping {tsv_ds['name']}")
                continue
            pdb_dir = tsv_ds["pdb_dir"]
            reuse_csv_name = tsv_ds.get("reuse_csv", "")
            reuse_csv = os.path.join(DATA_DIR, reuse_csv_name) if reuse_csv_name else ""

            # Download individual PDBs from AFDB
            pdb_path = os.path.join(DATA_DIR, pdb_dir)
            print(f"  Downloading PDBs for {tsv_ds['name']}...")
            download_tsv_pdbs(tsv_path, pdb_path, version="v6")

            # Process with AIUPred
            script_path = os.path.join(AIUPRED_DIR, "process_afdb.py")
            aiupred_python = os.path.join(
                os.path.expanduser("~"), ".conda", "envs", "aiupred", "bin", "python"
            )
            python_exe = aiupred_python if os.path.exists(aiupred_python) else sys.executable
            cmd = [
                python_exe, script_path,
                '--pdb_dir', pdb_path,
                '--result_file', result_file,
                '--gpu', '0',
                '--dataset_name', tsv_ds["name"],
            ]
            if reuse_csv and os.path.exists(reuse_csv):
                cmd += ['--reuse_csv', reuse_csv]

            print(f"  Processing {tsv_ds['name']}...")
            result = subprocess.run(
                cmd,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
            )
            if result.returncode == 0:
                print(f"  Completed: {tsv_ds['name']}")
            else:
                print(f"  FAILED: {tsv_ds['name']}")

    # Phase 1: Download and extract proteome tarballs
    print("\nPhase 1: Download and Extract")
    print("=" * 50)
    valid_datasets = []
    for dataset in DATASETS:
        pdb_dir = download_and_extract(dataset, DATA_DIR, delete_cif=True)
        if pdb_dir:
            valid_datasets.append((pdb_dir, dataset["name"]))

    print(f"\nPrepared {len(valid_datasets)} proteome datasets")

    # Phase 2: Process proteome datasets in parallel
    print("\nPhase 2: Parallel Processing")
    print("=" * 50)
    completed = 0
    failed = 0

    def process_wrapper(dataset_info):
        pdb_dir, dataset_name = dataset_info
        return run_dataset_processing(
            gpu_manager, pdb_dir, dataset_name, DATA_DIR
        )

    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        future_to_dataset = {
            executor.submit(process_wrapper, info): info
            for info in valid_datasets
        }
        for future in as_completed(future_to_dataset):
            pdb_dir, dataset_name = future_to_dataset[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                    print(f"  [{completed}/{len(valid_datasets)}] Done: {dataset_name}")
                else:
                    failed += 1
                    print(f"  FAILED: {dataset_name}")
            except Exception as e:
                failed += 1
                print(f"  Exception processing {dataset_name}: {e}")

    stop_event.set()

    print("\nProcessing Complete!")
    print("=" * 50)
    print(f"  Succeeded: {completed}")
    print(f"  Failed:    {failed}")
    if completed + failed > 0:
        print(f"  Success rate: {completed/(completed+failed)*100:.1f}%")

    csv_files = sorted(glob(os.path.join(DATA_DIR, "*_v6.csv")))
    print(f"\nOutput CSVs in {DATA_DIR}:")
    for f in csv_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
