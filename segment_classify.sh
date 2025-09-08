#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-pipeline-fixed
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
echo "MILK10K Pipeline Job Started"
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

# ==================== ENVIRONMENT SETUP ====================
echo "📦 Setting up environment..."

# Purge existing modules and load required ones
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
module load opencv/4.12.0

echo "✅ Modules loaded successfully."
echo "📋 Loaded modules:"
module list

# ==================== GPU DIAGNOSTICS ====================
echo ""
echo "🔍 GPU Diagnostics:"
echo "===================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check if GPU is allocated
if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    echo "⚠️ WARNING: No GPU allocation detected in SLURM_GPUS_ON_NODE"
else
    echo "✅ GPU allocation detected: $SLURM_GPUS_ON_NODE"
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️ Available GPUs:"
    nvidia-smi --list-gpus
    echo ""
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
else
    echo "⚠️ nvidia-smi not available"
fi

# ==================== VIRTUAL ENVIRONMENT ====================
echo ""
echo "💻 Activating virtual environment from $VENV_DIR..."
source "$VENV_DIR/bin/activate" || {
    echo "❌ ERROR: Failed to activate virtual environment. Exiting."
    exit 1
}
echo "✅ Virtual environment activated."

# Verify Python and PyTorch installation
echo ""
echo "🐍 Python environment verification:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not found')"
echo "CUDA available in PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Cannot check')"
echo "GPU count in PyTorch: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Cannot check')"

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "⚙️ Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/groundtruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs"

# Performance optimizations for Slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

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

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "⚠️ WARNING: Ground truth file not found: $GROUNDTRUTH_PATH"
    echo "Pipeline will run without ground truth evaluation."
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "✅ All required paths verified."

# Create output directory
mkdir -p "$OUTPUT_PATH"
echo "✅ Output directory ready: $OUTPUT_PATH"

# ==================== PIPELINE EXECUTION ====================
echo ""
echo "🚀 Starting the MILK10k pipeline..."
echo "===================================="

# Run with timeout and error handling
timeout 5h python "$PYTHON_SCRIPT" 2>&1 | tee "${OUTPUT_PATH}/pipeline_log.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "📊 Post-execution analysis:"
echo "=========================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    
    # Check output files
    if [ -d "${OUTPUT_PATH}/segmented_for_conceptclip" ]; then
        SEGMENTED_COUNT=$(find "${OUTPUT_PATH}/segmented_for_conceptclip" -name "*.png" | wc -l)
        echo "📁 Segmented outputs created: $SEGMENTED_COUNT files"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/processing_report.json" ]; then
        echo "📄 Processing report generated"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/detailed_results.csv" ]; then
        echo "📊 Detailed results saved"
    fi
    
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏰ Pipeline timed out (5 hour limit reached)"
    echo "Partial results may be available in: $OUTPUT_PATH"
else
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log and pipeline_log.txt for details"
fi

# Display resource usage
echo ""
echo "💻 Resource Usage Summary:"
echo "========================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64G"
echo "GPU allocated: $SLURM_GPUS_ON_NODE"

# If seff is available, show efficiency
if command -v seff &> /dev/null; then
    echo ""
    echo "📈 Job Efficiency:"
    seff $SLURM_JOB_ID
fi

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
