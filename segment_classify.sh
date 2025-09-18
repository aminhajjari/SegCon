#!/bin/bash
#SBATCH --job-name=SegCon
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.err

# Load modules - MATCH your Python version
module load python/3.11.5
module load cuda/11.7
module load opencv/4.12.0  # Latest OpenCV version

# Environment variables
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache
export HF_HOME=$TRANSFORMERS_CACHE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Create output directory
mkdir -p /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs

# Activate your existing virtual environment
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Verify OpenCV is available
echo "Testing OpenCV import..."
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Run your script
python /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/path.py --test
