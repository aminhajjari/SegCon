#!/bin/bash
#SBATCH --job-name=SegCon_Rerun
#SBATCH --account=def-arashmoh
#SBATCH --time=04:00:00  # Increased to 4 hours
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.out
#SBATCH --error=/home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.err

# Load modules in the exact sequence
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Create output directory
mkdir -p /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs

# Activate virtual environment
source /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current modules loaded:"
module list
python -c "import cv2; print('OpenCV version:', cv2.__version__)" || echo "OpenCV import failed"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || echo "PyTorch import failed"

# Run the script with unbuffered output
python -u /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/path.py > /home/shahab33/projects/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/debug_output_%j.log 2>&1
