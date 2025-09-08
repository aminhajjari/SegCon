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

# Navigate to project directory
cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon

# Clear environment and load modules
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
# Remove opencv module - conflicts with pip version

# Activate virtual environment
echo "Activating virtual environment..."
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Install missing packages
echo "Installing missing packages..."
pip install opencv-python pandas matplotlib --quiet

# Set environment variables
export PYTHONPATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input:$PYTHONPATH"
export DATASET_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
export OUTPUT_PATH="/project/def-arashmoh/shahab33/XAI/outputs"

# Test imports before running
echo "Testing critical imports..."
python -c "
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')
import cv2, pandas, torch, numpy
from ConceptModel.modeling_conceptclip import ConceptCLIP
print('All imports successful')
"

# Run Python script
echo "Starting Python script..."
python path.py

# Check completion
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
