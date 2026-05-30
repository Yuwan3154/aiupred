# AIUPred Parallel Processing Pipeline

This pipeline automatically downloads, extracts, and processes multiple AlphaFold datasets in parallel using multiple GPUs for efficient AIUPred disorder prediction analysis.

## 🚀 Quick Start

```bash
cd /home/jupyter-chenxi/aiupred
./run_parallel_processing.sh
```

## 📋 Overview

The pipeline processes 20 different organisms from the AlphaFold database:

- **Human** (UP000005640_9606_HUMAN_v4)
- **E. coli** (UP000000625_83333_ECOLI_v4)
- **C. elegans** (UP000001940_6239_CAEEL_v4)
- **A. thaliana** (UP000006548_3702_ARATH_v4)
- **C. albicans** (UP000000559_237561_CANAL_v4)
- **D. rerio** (UP000000437_7955_DANRE_v4)
- **D. discoideum** (UP000002195_44689_DICDI_v4)
- **D. melanogaster** (UP000000803_7227_DROME_v4)
- **G. max** (UP000008827_3847_SOYBN_v4)
- **M. jannaschii** (UP000000805_243232_METJA_v4)
- **M. musculus** (UP000000589_10090_MOUSE_v4)
- **O. sativa** (UP000059680_39947_ORYSJ_v4)
- **R. norvegicus** (UP000002494_10116_RAT_v4)
- **S. cerevisiae** (UP000002311_559292_YEAST_v4)
- **S. pombe** (UP000002485_284812_SCHPO_v4)
- **Z. mays** (UP000007305_4577_MAIZE_v4)
- **S. mansoni** (UP000008854_6183_SCHMA_v4)
- **P. falciparum** (UP000001450_36329_PLAF7_v4)
- **A. capsulatus** (UP000001631_447093_AJECG_v4)
- **M. leprae** (UP000000806_272631_MYCLE_v4)

## 🎯 Features

- **Parallel Processing**: Uses all 4 GPUs simultaneously
- **Smart GPU Management**: Automatically assigns datasets to available GPUs
- **Resume Capability**: Skips already processed proteins
- **Progress Monitoring**: Real-time status updates
- **Error Handling**: Robust error handling and recovery
- **Automatic Downloads**: Downloads and extracts datasets if needed

## 📁 File Structure

```
aiupred/
├── parallel_processing.py      # Main pipeline script
├── run_parallel_processing.sh  # Convenience runner script
├── monitor_progress.py         # Progress monitoring script
├── process_single_dataset.py   # Individual dataset processor (auto-generated)
├── aiupred_lib.py             # AIUPred library
└── README_parallel_processing.md

data/afdb/                      # Data directory
├── UP000005640_9606_HUMAN_v4/  # Extracted PDB files
├── UP000005640_9606_HUMAN_v4.csv  # Results
└── ... (other datasets)
```

## 🔧 How It Works

### Phase 1: Download & Extract
1. Downloads AlphaFold tar files if not present
2. Extracts PDB files to individual directories
3. Validates successful extraction

### Phase 2: Parallel Processing
1. Creates a GPU manager with 4 available GPUs (0, 1, 2, 3)
2. For each dataset:
   - Waits for an available GPU
   - Spawns a separate Python process with that GPU
   - Processes all PDB files in the dataset
   - Calculates AIUPred disorder scores and masked mean pLDDT
   - Saves results to CSV file
   - Releases GPU for next dataset

### GPU Management
- **Smart Assignment**: Automatically assigns datasets to free GPUs
- **Process Isolation**: Each dataset runs in its own process
- **Memory Safety**: GPU memory is properly isolated between processes
- **Fault Tolerance**: Failed processes don't affect others

## 📊 Output Format

Each dataset produces a CSV file with columns:
- `accession_id`: Protein identifier
- `sequence`: Amino acid sequence
- `disorder_score`: Per-residue disorder scores (JSON array)
- `plddt`: Per-residue confidence scores (JSON array)
- `mean_plddt`: Average confidence score
- `masked_mean_plddt`: Average confidence for ordered regions only
- `mean_disorder_score`: Average disorder score

## 🖥️ Monitoring Progress

### Real-time Monitoring
```bash
# In a separate terminal
cd /home/jupyter-chenxi/aiupred
python3 monitor_progress.py
```

This shows:
- GPU utilization and memory usage
- Active processing jobs
- Progress for each dataset
- File sizes and protein counts

### Manual Checks
```bash
# Check GPU usage
nvidia-smi

# Check running processes
ps aux | grep process_single_dataset

# Check output files
ls -lh /home/jupyter-chenxi/data/afdb/*.csv
```

## ⚙️ Configuration

Edit `parallel_processing.py` to modify:

```python
NUM_GPUS = 4                    # Number of GPUs to use
DATA_DIR = "/path/to/data"      # Data directory
AIUPRED_DIR = "/path/to/aiupred" # AIUPred directory
```

## 🚨 Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Check GPU memory
nvidia-smi

# Kill stuck processes
pkill -f process_single_dataset.py
```

**Download Failures**
- Check internet connection
- Verify AlphaFold URLs are still valid
- Check disk space in `/home/jupyter-chenxi/data/afdb`

**Processing Stuck**
- Use monitor script to check progress
- Look for error messages in terminal output
- Check individual CSV files for partial results

### Recovery

The pipeline is designed to resume automatically:
- Already downloaded files are skipped
- Already processed proteins are skipped
- Partial CSV files are extended, not overwritten

To force reprocessing:
```bash
# Remove specific CSV file to reprocess dataset
rm /home/jupyter-chenxi/data/afdb/UP000005640_9606_HUMAN_v4.csv

# Remove all CSV files to reprocess everything
rm /home/jupyter-chenxi/data/afdb/*.csv
```

## 🔍 Performance

**Expected Performance** (varies by dataset size):
- **Small datasets** (< 1,000 proteins): ~10-30 minutes
- **Medium datasets** (1,000-5,000 proteins): ~1-3 hours  
- **Large datasets** (> 5,000 proteins): ~3-8 hours

**Total Pipeline Time**: ~6-12 hours for all 20 datasets

## 📈 Resource Requirements

- **GPUs**: 4x NVIDIA GPUs with CUDA support
- **RAM**: ~64GB recommended (16GB per GPU process)
- **Storage**: ~500GB for downloaded files + extracted PDBs + results
- **Network**: Stable internet for ~100GB of downloads

## 🛠️ Advanced Usage

### Custom Dataset Lists
Modify the `DATASETS` list in `parallel_processing.py` to process different organisms.

### Different GPU Configurations
```python
# Use 2 GPUs instead of 4
NUM_GPUS = 2

# Use specific GPU IDs
# Modify the GPU initialization in GPUManager.__init__()
```

### Processing Single Datasets
```bash
# Process just one dataset manually
python3 process_single_dataset.py \
    --pdb_dir /path/to/pdb/files \
    --result_file output.csv \
    --gpu_id 0 \
    --dataset_name "Test Dataset"
```

## 📚 Dependencies

- Python 3.8+
- PyTorch with CUDA support
- BioPython
- pandas (fireducks.pandas)
- numpy
- tqdm
- scipy

All dependencies should already be available in your environment.