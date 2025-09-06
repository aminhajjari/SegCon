#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=milk10k_local_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8               # Increased for better performance
#SBATCH --mem=64G                       # Increased for ConceptCLIP and image processing
#SBATCH --gres=gpu:1                    # GPU required for SAM2 and ConceptCLIP
#SBATCH --time=06:00:00                 # Increased time for full dataset processing
#SBATCH --mail-user=aminhjjr@gmail.com
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

# Load required modules for local models
module load StdEnv/2020           # Standard environment
module load python/3.10          # Python version
module load cuda/11.8            # CUDA version for GPU support
module load gcc/9.3.0            # Compiler for local builds
module load cmake/3.21.4         # If needed for building dependencies
module load git/2.36.1           # For any git operations

# Print loaded modules
echo "Loaded modules:"
module list

# Set environment variables for local models (NO HF_TOKEN needed)
echo "Setting up environment for local models..."

# Set cache directories to use project space (avoids quota issues)
export TORCH_HOME="/project/def-arashmoh/shahab33/XAI/torch_cache"
export CUDA_CACHE_PATH="/project/def-arashmoh/shahab33/XAI/cuda_cache"

# Create cache directories
mkdir -p $TORCH_HOME
mkdir -p $CUDA_CACHE_PATH

# Set paths for your local models (UPDATE THESE PATHS)
export SAM2_MODEL_PATH="/project/def-arashmoh/shahab33/XAI/models/sam2"
export CONCEPTCLIP_MODEL_PATH="/project/def-arashmoh/shahab33/XAI/models/conceptclip"

# Dataset and output paths
export DATASET_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
export OUTPUT_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/outputs"

echo "Model paths:"
echo "SAM2 Model: $SAM2_MODEL_PATH"
echo "ConceptCLIP Model: $CONCEPTCLIP_MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"

# Activate your Python virtual environment
echo "Activating virtual environment..."
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# Verify local model installations and paths
echo "=========================================="
echo "Verifying local model installations..."
echo "=========================================="

# Check if local SAM2 model exists
if [ -d "$SAM2_MODEL_PATH" ]; then
    echo "SAM2 model directory: ✓ Found at $SAM2_MODEL_PATH"
    ls -la $SAM2_MODEL_PATH | head -5
else
    echo "SAM2 model directory: ✗ NOT FOUND at $SAM2_MODEL_PATH"
    echo "Please update SAM2_MODEL_PATH in the script"
fi

# Check if local ConceptCLIP model exists
if [ -d "$CONCEPTCLIP_MODEL_PATH" ]; then
    echo "ConceptCLIP model directory: ✓ Found at $CONCEPTCLIP_MODEL_PATH"
    ls -la $CONCEPTCLIP_MODEL_PATH | head -5
else
    echo "ConceptCLIP model directory: ✗ NOT FOUND at $CONCEPTCLIP_MODEL_PATH"
    echo "Please update CONCEPTCLIP_MODEL_PATH in the script"
fi

# Check if ConceptModel modules exist
if [ -d "/project/def-arashmoh/shahab33/XAI/ConceptModel" ]; then
    echo "ConceptModel modules: ✓ Found"
    ls -la /project/def-arashmoh/shahab33/XAI/ConceptModel/
else
    echo "ConceptModel modules: ✗ NOT FOUND"
    echo "Please ensure ConceptModel directory with modeling_conceptclip.py and preprocessor_conceptclip.py exists"
fi

# Verify Python installations
echo "=========================================="
echo "Verifying Python environment..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

# Check SAM2 installation
echo "Checking SAM2..."
python -c "
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print('SAM2: ✓ Import successful')
except ImportError as e:
    print(f'SAM2: ✗ Import failed - {e}')
"

# Check ConceptCLIP modules
echo "Checking ConceptCLIP modules..."
python -c "
try:
    from ConceptModel import modeling_conceptclip
    from ConceptModel import preprocessor_conceptclip
    print('ConceptCLIP modules: ✓ Import successful')
except ImportError as e:
    print(f'ConceptCLIP modules: ✗ Import failed - {e}')
"

# Check other required packages
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"

# GPU information
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set additional environment variables for better performance
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better memory management

# Add ConceptModel directory to Python path for local module import
export PYTHONPATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input:$PYTHONPATH"

echo "Environment variables:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "PYTHONPATH: $PYTHONPATH"

# Create output directories if they don't exist
mkdir -p $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/segmented
mkdir -p $OUTPUT_PATH/segmented_for_conceptclip
mkdir -p $OUTPUT_PATH/classifications
mkdir -p $OUTPUT_PATH/visualizations
mkdir -p $OUTPUT_PATH/reports

# Check if dataset exists
echo "=========================================="
echo "Dataset verification..."
echo "=========================================="
if [ -d "$DATASET_PATH" ]; then
    echo "Dataset directory: ✓ Found"
    echo "Dataset contents (first 10 files):"
    find $DATASET_PATH -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tiff" -o -name "*.dcm" \) | head -10
    echo "Total images found: $(find $DATASET_PATH -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tiff" -o -name "*.dcm" \) | wc -l)"
else
    echo "Dataset directory: ✗ NOT FOUND at $DATASET_PATH"
fi

if [ -f "$GROUNDTRUTH_PATH" ]; then
    echo "Ground truth file: ✓ Found"
    echo "Ground truth preview:"
    head -3 $GROUNDTRUTH_PATH
else
    echo "Ground truth file: ✗ NOT FOUND at $GROUNDTRUTH_PATH"
fi

# Run your MILK10k pipeline with local models
echo "=========================================="
echo "Starting MILK10k Pipeline with Local Models..."
echo "=========================================="

# Setup cleanup trap for better resource management
cleanup() {
    echo "Job interrupted or finished, cleaning up..."
    # Clean temporary files
    if [ -d "$SLURM_TMPDIR" ]; then
        rm -rf $SLURM_TMPDIR/*
    fi
    echo "Cleanup completed."
}
trap cleanup EXIT SIGTERM SIGINT

# Update the Python script name to match your actual file
python milk10k_pipeline.py  # Change this to your actual Python file name

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MILK10k Pipeline completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
    
    # Print output summary
    echo "Output directory contents:"
    ls -la $OUTPUT_PATH/
    
    echo "Segmented outputs for ConceptCLIP (first 5 directories):"
    ls -la $OUTPUT_PATH/segmented_for_conceptclip/ | head -10
    
    echo "Reports generated:"
    ls -la $OUTPUT_PATH/reports/
    
    echo "Visualizations generated:"
    ls -la $OUTPUT_PATH/visualizations/
    
else
    echo "=========================================="
    echo "MILK10k Pipeline failed with exit code: $?"
    echo "End Time: $(date)"
    echo "Check the logs above for error details"
    echo "=========================================="
    exit 1
fi

# Final resource usage summary
echo "=========================================="
echo "Job completion summary:"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo "=========================================="
