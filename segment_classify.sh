```bash
#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.err

echo "=========================================="
echo "MILK10k Segmentation and Classification Pipeline Started (Narval)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "=========================================="

# ==================== PROJECT SETUP ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR/SegCon"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="path.py"

# Navigate to script directory
cd "$SCRIPT_DIR" || {
    echo "❌ ERROR: Failed to change directory to $SCRIPT_DIR. Exiting."
    exit 1
}

# ==================== ENVIRONMENT VARIABLES ====================
echo "⚙️ Setting environment variables..."

# Prevent BLIS/OpenMP conflicts (matching path.py)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Additional stability settings
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Pipeline-specific paths (matching path.py)
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs"
export SAM2_MODEL_PATH="$PROJECT_DIR/segment-anything-2"
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"

echo "✅ Environment variables set."

# ==================== NARVAL-SPECIFIC MODULE SETUP ====================
echo "📦 Loading modules for Narval..."

module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/11.8
module load cudnn/8.9.7
module load opencv/4.12.0
# SimpleITK is in venv, not a Narval module

echo "✅ Modules loaded successfully."
echo "📋 Loaded modules:"
module list

# ==================== NARVAL GPU DIAGNOSTICS ====================
echo ""
echo "🔍 Narval GPU Diagnostics (BEFORE srun):"
echo "========================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES (before srun): $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️ Available GPUs on node:"
    nvidia-smi --list-gpus
else
    echo "⚠️ nvidia-smi not available"
fi

# ==================== VIRTUAL ENVIRONMENT ====================
echo ""
echo "💻 Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    echo "❌ ERROR: Failed to activate virtual environment at $VENV_DIR. Exiting."
    exit 1
}
echo "✅ Virtual environment activated."

# Verify Python version
echo "Python version:"
python --version

# Install dependencies (redundant since confirmed installed)
pip install --no-index torch torchvision torchaudio
pip install --no-index transformers pillow pandas numpy opencv-python-headless pydicom nibabel matplotlib seaborn tqdm simpleitk
pip install --no-index -e "$SAM2_MODEL_PATH"  # Ensure sam2 is installed

echo "Python environment:"
which python
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ==================== PRE-EXECUTION CHECKS ====================
echo ""
echo "🔍 Pre-execution checks:"
echo "======================="

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$SAM2_MODEL_PATH" ]; then
    echo "❌ ERROR: SAM2 model path not found: $SAM2_MODEL_PATH"
    exit 1
fi

if [ ! -f "$SAM2_MODEL_PATH/checkpoints/sam2_hiera_large.pt" ]; then
    echo "❌ ERROR: SAM2 checkpoint not found: $SAM2_MODEL_PATH/checkpoints/sam2_hiera_large.pt"
    echo "Download it from https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt and upload to $SAM2_MODEL_PATH/checkpoints/"
    exit 1
fi

if [ ! -d "$CONCEPTCLIP_MODEL_PATH" ]; then
    echo "❌ ERROR: ConceptCLIP model path not found: $CONCEPTCLIP_MODEL_PATH"
    exit 1
fi

if [ ! -d "$HUGGINGFACE_CACHE_PATH" ]; then
    echo "⚠️ WARNING: Hugging Face cache path not found, creating: $HUGGINGFACE_CACHE_PATH"
    mkdir -p "$HUGGINGFACE_CACHE_PATH"
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "⚠️ WARNING: Ground truth file not found: $GROUNDTRUTH_PATH"
fi

echo "✅ All required paths verified."
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Create output directory
mkdir -p "$OUTPUT_PATH"

# ==================== CRITICAL: USE SRUN FOR GPU ALLOCATION ====================
echo ""
echo "🚀 Starting MILK10k pipeline with srun..."
echo "=================================================================="

# Create a wrapper script to check GPU availability inside srun
cat > gpu_check_and_run.py << 'EOF'
import os
import torch
import sys

print("=" * 60)
print("GPU CHECK INSIDE SRUN")
print("=" * 60)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"SLURM_STEP_GPUS: {os.environ.get('SLURM_STEP_GPUS', 'NOT SET')}")
print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'NOT SET')}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
print("=" * 60)

# Run the actual script
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon')
exec(open('path.py').read())
EOF

# Launch the Python script with srun
srun --gres=gpu:1 python gpu_check_and_run.py 2>&1 | tee "${OUTPUT_PATH}/pipeline_log.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "📊 Post-execution analysis:"
echo "=========================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    
    if [ -f "${OUTPUT_PATH}/reports/detailed_results.csv" ]; then
        PROCESSED_COUNT=$(wc -l < "${OUTPUT_PATH}/reports/detailed_results.csv")
        echo "📁 Detailed results created: $((PROCESSED_COUNT-1)) images processed"
    else
        echo "⚠️ No detailed results found at ${OUTPUT_PATH}/reports/detailed_results.csv"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/processing_report.json" ]; then
        echo "📄 Processing report generated"
    else
        echo "⚠️ No processing report found at ${OUTPUT_PATH}/reports/processing_report.json"
    fi
    
    if [ -d "${OUTPUT_PATH}/segmented_for_conceptclip" ]; then
        SEGMENTED_COUNT=$(find "${OUTPUT_PATH}/segmented_for_conceptclip" -name "*.png" 2>/dev/null | wc -l)
        echo "📁 Segmented outputs created: $SEGMENTED_COUNT files"
    else
        echo "⚠️ No segmented outputs found at ${OUTPUT_PATH}/segmented_for_conceptclip"
    fi
    
    if [ -f "${OUTPUT_PATH}/visualizations/summary_plots.png" ]; then
        echo "📊 Summary plots generated"
    else
        echo "⚠️ No summary plots found at ${OUTPUT_PATH}/visualizations/summary_plots.png"
    fi
    
else
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log and pipeline_log.txt for details"
fi

echo ""
echo "💡 KEY LESSON: On Narval, always use 'srun' to launch GPU programs!"
echo "   - 'python script.py' → No GPU access"
echo "   - 'srun python script.py' → GPU access ✅"

# Show output directory contents
echo "Output directory contents:"
ls -la "$OUTPUT_PATH/" || echo "outputs directory not found"

# Show final job statistics
echo "=========================================="
echo "Job statistics:"
sstat --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize --jobs=$SLURM_JOB_ID || echo "sstat not available"

echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
```
