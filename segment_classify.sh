#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-pipeline-narval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "=========================================="
echo "MILK10K Pipeline Job Started (Narval Fixed)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
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

# ==================== NARVAL-SPECIFIC MODULE SETUP ====================
echo "📦 Loading modules for Narval..."

# Purge and load modules in correct order for Narval
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
module load opencv/4.12.0

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

# Check GPU status
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
    echo "❌ ERROR: Failed to activate virtual environment. Exiting."
    exit 1
}
echo "✅ Virtual environment activated."

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "⚙️ Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/groundtruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs"

# Narval-specific optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CRITICAL: Do NOT manually set CUDA_VISIBLE_DEVICES on Narval
# Let SLURM handle it through srun

echo "✅ Environment variables set."

# ==================== PRE-EXECUTION CHECKS ====================
echo ""
echo "🔍 Pre-execution checks:"
echo "======================="

# Check if required paths exist
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "✅ All required paths verified."

# Create output directory
mkdir -p "$OUTPUT_PATH"

# ==================== CRITICAL: USE SRUN FOR GPU ALLOCATION ====================
echo ""
echo "🚀 Starting MILK10k pipeline with srun (CRITICAL for GPU access)..."
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

# Now run the actual script
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon')
exec(open('path.py').read())
EOF

# CRITICAL: Use srun to launch the Python script
# This is what actually allocates the GPU to your process on Narval
srun --gres=gpu:1 python gpu_check_and_run.py 2>&1 | tee "${OUTPUT_PATH}/pipeline_log.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "📊 Post-execution analysis:"
echo "=========================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    
    # Check output files
    if [ -d "${OUTPUT_PATH}/segmented_for_conceptclip" ]; then
        SEGMENTED_COUNT=$(find "${OUTPUT_PATH}/segmented_for_conceptclip" -name "*.png" 2>/dev/null | wc -l)
        echo "📁 Segmented outputs created: $SEGMENTED_COUNT files"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/processing_report.json" ]; then
        echo "📄 Processing report generated"
    fi
    
else
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log and pipeline_log.txt for details"
fi

echo ""
echo "💡 KEY LESSON: On Narval, always use 'srun' to launch GPU programs!"
echo "   - 'python script.py' → No GPU access"
echo "   - 'srun python script.py' → GPU access ✅"

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
