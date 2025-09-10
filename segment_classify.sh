#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=milk10k-pipeline

# CRITICAL: Set threading environment variables to prevent conflicts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Alternative approach - match allocated cores:
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load required modules
module load python/3.9 cuda/11.7

# Activate your virtual environment
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate


# Run the pipeline
python path.py
