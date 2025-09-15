#!/bin/bash
#SBATCH --job-name=milk10k_pipeline
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs/milk10k_%j.err

module load python/3.10
module load cuda/11.7
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

source /project/def-arashmoh/shahab33/venv/bin/activate
mkdir -p /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs/logs
python /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/path.py --test
