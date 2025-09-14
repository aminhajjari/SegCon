#!/bin/bash
#SBATCH --job-name=SegCon
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/%x-%j.err

# === Environment Variables (preserved from your code) ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Project Paths ===
export PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
export OUTPUT_PATH="$PROJECT_DIR/SegConOutputs"

# Create output folder if missing
mkdir -p "$OUTPUT_PATH"

# === Load Modules ===
module load python/3.10
module load cuda
source "$PROJECT_DIR/venv/bin/activate"

# === Ensure we use the GPU allocated by Slurm ===
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# === Print GPU Info for Debugging ===
echo "------------------------------------------------------------"
echo "Running on Node: $SLURMD_NODENAME"
echo "Allocated GPU(s): $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Info:"
nvidia-smi
echo "------------------------------------------------------------"

# === Run Code ===
python path.py

# === Post-Run Check ===
echo "------------------------------------------------------------"
echo "Job finished. Output directory content:"
ls -lh "$OUTPUT_PATH"
echo "------------------------------------------------------------"
