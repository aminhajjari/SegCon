#!/bin/bash
#SBATCH --job-name=SegCon
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# === Environment Variables (preserved from your code) ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Project Paths ===
export PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
export OUTPUT_PATH="$PROJECT_DIR/SegCon/outputs"

# === Load Modules ===
module load python/3.10
module load cuda
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate
  t

# === Run Code ===
python path.py
