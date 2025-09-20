#!/bin/bash
#SBATCH --job-name=SegCon_MILK10k
#SBATCH --account=def-arashmoh
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.out
#SBATCH --error=/home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.err

# Load modules
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load opencv/4.12.0
module load python/3.11.5

# Environment variables
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE=/home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache
export HF_HOME=$TRANSFORMERS_CACHE

# Threading optimizations
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create output directories
mkdir -p /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs
mkdir -p /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/reports
mkdir -p /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/segmented

# Activate virtual environment
source /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Verify environment
echo "======================================"
echo "Environment Verification"
echo "======================================"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "======================================"

# Check critical imports
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo "======================================"
echo "Starting pipeline for 50 folders"
echo "======================================"

# Run the pipeline script
# IMPORTANT: Update 'pipeline.py' to your actual script name
# New (correct) line
python -u path.py \
    --max-folders 50 \
    2>&1 | tee /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/pipeline_output_${SLURM_JOB_ID}.log
echo "======================================"
echo "Pipeline completed"
echo "Job ID: ${SLURM_JOB_ID}"
echo "======================================"
