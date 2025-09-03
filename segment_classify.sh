#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=milk10k_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8               # Increased for better performance
#SBATCH --mem=64G                       # Increased for ConceptCLIP and image processing
#SBATCH --gres=gpu:1                    # GPU required for SAM2 and ConceptCLIP
#SBATCH --time=04:00:00                 # Increased time for full dataset processing
#SBATCH --mail-user=amminhajjari@gmail.com
#SBATCH --mail-type=ALL

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# Change to your project directory
cd /project/def-arashmoh/shahab33/XAI

# Clear all loaded modules
module purge

# Load required modules
module load python/3.10        # Specify Python version
module load cuda/11.8          # Specify CUDA version
module load gcc/9.3.0          # Sometimes needed for compilation

# Print loaded modules
echo "Loaded modules:"
module list

# Set Hugging Face token for ConceptCLIP access
# REPLACE 'your_hf_token_here' WITH YOUR ACTUAL TOKEN FROM https://huggingface.co/settings/tokens
export HF_TOKEN="your_hf_token_here"

# Set cache directories to use scratch space (avoids quota issues)
export HF_HOME="/project/def-arashmoh/shahab33/XAI/hf_cache"
export HF_CACHE_DIR="/project/def-arashmoh/shahab33/XAI/hf_cache"
export TRANSFORMERS_CACHE="/project/def-arashmoh/shahab33/XAI/hf_cache"
export TORCH_HOME="/project/def-arashmoh/shahab33/XAI/torch_cache"

# Create cache directories
mkdir -p $HF_CACHE_DIR
mkdir -p $TORCH_HOME

# Activate your Python virtual environment
# Update this path to match your actual virtual environment location
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegmnetationConceptCLIP/venv/bin/activate

# Print environment information
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set additional environment variables for better performance
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print dataset and output paths
echo "Dataset path: $DATASET_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Cache directory: $HF_CACHE_DIR"

# Run your MILK10k pipeline
echo "=========================================="
echo "Starting MILK10k Pipeline..."
echo "=========================================="

python path.py

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MILK10k Pipeline completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "MILK10k Pipeline failed with exit code: $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

# Optional: Print output directory contents
echo "Output directory contents:"
ls -la /project/def-arashmoh/shahab33/XAI/outputs/

echo "Segmented outputs for ConceptCLIP:"
ls -la /project/def-arashmoh/shahab33/XAI/outputs/segmented_for_conceptclip/ | head -10