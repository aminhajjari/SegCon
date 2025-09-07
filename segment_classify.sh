#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=milk10k_local_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8               # Increased for better performance
#SBATCH --mem=64G                       # Increased for ConceptCLIP and image processing
#SBATCH --gres=gpu:v100:1               # Specify GPU type for better allocation
#SBATCH --time=06:00:00                 # Increased time for full dataset processing
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out              # Better output file naming
#SBATCH --error=%x-%j.err               # Separate error file

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# Change to your project directory
cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon

# Clear all loaded modules
module purge

# Load required modules (preserve your venv Python)
module load StdEnv/2020           # Standard environment
module load cuda/11.8            # CUDA version for GPU support
module load gcc/9.3.0            # Compiler for local builds
module load cmake/3.21.4         # If needed for building dependencies

# Print loaded modules
echo "Loaded modules:"
module list

# Use SLURM_TMPDIR for temporary files and better I/O performance
export TMPDIR=$SLURM_TMPDIR
export TEMP=$SLURM_TMPDIR
export TMP=$SLURM_TMPDIR

# Set cache directories to use project space (avoids quota issues)
export TORCH_HOME="/project/def-arashmoh/shahab33/XAI/torch_cache"
export CUDA_CACHE_PATH="/project/def-arashmoh/shahab33/XAI/cuda_cache"

# Create cache directories
mkdir -p $TORCH_HOME
mkdir -p $CUDA_CACHE_PATH

# Set paths for your local models
export SAM2_MODEL_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
export CONCEPTCLIP_MODEL_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"

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

# Install missing packages if needed
echo "Checking and installing missing packages..."
pip install pydicom nibabel SimpleITK --quiet

# Set Python path for ConceptModel imports (CRITICAL)
export PYTHONPATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input:$PYTHONPATH"

# Verify local model installations and paths
echo "=========================================="
echo "Verifying local model installations..."
echo "=========================================="

# Check if local SAM2 model exists
if [ -d "$SAM2_MODEL_PATH" ]; then
    echo "SAM2 model directory: ✓ Found at $SAM2_MODEL_PATH"
else
    echo "SAM2 model directory: ✗ NOT FOUND at $SAM2_MODEL_PATH"
fi

# Check if local ConceptCLIP model exists with required files
if [ -d "$CONCEPTCLIP_MODEL_PATH" ]; then
    echo "ConceptModel directory: ✓ Found at $CONCEPTCLIP_MODEL_PATH"
    echo "Checking for required ConceptCLIP files:"
    [ -f "$CONCEPTCLIP_MODEL_PATH/__init__.py" ] && echo "  __init__.py: ✓" || echo "  __init__.py: ✗"
    [ -f "$CONCEPTCLIP_MODEL_PATH/modeling_conceptclip.py" ] && echo "  modeling_conceptclip.py: ✓" || echo "  modeling_conceptclip.py: ✗"
    [ -f "$CONCEPTCLIP_MODEL_PATH/preprocessor_conceptclip.py" ] && echo "  preprocessor_conceptclip.py: ✓" || echo "  preprocessor_conceptclip.py: ✗"
else
    echo "ConceptModel directory: ✗ NOT FOUND at $CONCEPTCLIP_MODEL_PATH"
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

# Check ConceptCLIP modules with proper path
echo "Checking ConceptCLIP local modules..."
python -c "
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')
try:
    from ConceptModel.modeling_conceptclip import ConceptCLIP
    from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor
    print('ConceptCLIP modules: ✓ Import successful')
except ImportError as e:
    print(f'ConceptCLIP modules: ✗ Import failed - {e}')
"

# Check other required packages
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import pydicom; print(f'pydicom: {pydicom.__version__}')"

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
export CUDA_LAUNCH_BLOCKING=0  # Async GPU operations for better performance

# Memory and performance optimizations
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

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

# Print working directory contents
echo "Working directory contents:"
echo "Python script: path.py"
echo "Requirements file: req.txt"
ls -la

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

# Run your Python script (path.py in SegCon directory)
python path.py

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
