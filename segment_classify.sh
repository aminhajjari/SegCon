#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=milk10k_local_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "=========================================="
echo "Job Started on Node: $SLURMD_NODENAME" 
echo "Start Time: $(date)"
echo "=========================================="

cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon

# Load modules BEFORE activating venv (critical for OpenCV)
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
module load opencv/4.12.0

echo "Loaded modules:"
module list

# Activate virtual environment AFTER loading modules
echo "Activating virtual environment..."
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Install missing packages (not opencv-python)
echo "Installing packages..."
pip install pandas matplotlib --quiet

# Set environment variables
export PYTHONPATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input:$PYTHONPATH"
export DATASET_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
export OUTPUT_PATH="/project/def-arashmoh/shahab33/XAI/outputs"

# Performance optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE

# Test critical imports before running
echo "Testing imports..."
python -c "
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')
import cv2, pandas, torch, numpy
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
from ConceptModel.modeling_conceptclip import ConceptCLIP
from sam2.sam2_image_predictor import SAM2ImagePredictor
print('All imports successful')
"

if [ $? -ne 0 ]; then
    echo "Import test failed. Exiting."
    exit 1
fi

# Run script
echo "Starting pipeline..."
python path.py

# Check completion status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Pipeline failed. Check error log."
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
