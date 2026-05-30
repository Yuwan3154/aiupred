#!/bin/bash
"""
Convenience script to run the parallel processing pipeline
"""

echo "🚀 Starting AIUPred Parallel Processing Pipeline"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "parallel_processing.py" ]; then
    echo "❌ Error: Please run this script from the aiupred directory"
    echo "   cd /home/jupyter-chenxi/aiupred && ./run_parallel_processing.sh"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# Set environment variables for better performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Run the pipeline
echo ""
echo "🚀 Launching parallel processing pipeline..."
python parallel_processing.py

echo ""
echo "✅ Pipeline execution complete!"