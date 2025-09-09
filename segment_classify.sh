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
PYTHON_SCRIPT="path.py"  # UPDATE THIS TO YOUR ACTUAL SCRIPT NAME
CACHE_DIR="$PROJECT_DIR/huggingface_cache"

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
module load gcc/12.3       # Using a modern GCC for StdEnv/2023
module load cuda/12.6
module load opencv/4.12.0

echo "✅ Modules loaded successfully."
echo "📋 Loaded modules:"
module list

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

# NEW: Offline mode and cache configuration
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_OFFLINE=1

# Narval-specific optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "✅ Environment variables set."
echo "🔒 Offline mode enabled with cache: $CACHE_DIR"

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

# NEW: Check cache directory and cached models
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ ERROR: Cache directory does not exist: $CACHE_DIR"
    echo "Please ensure the Hugging Face models are properly cached."
    exit 1
fi

# Check for the specific cached model
if [ ! -d "$CACHE_DIR/models--google--siglip-so400m-patch14-384" ]; then
    echo "❌ WARNING: Expected cached model not found: models--google--siglip-so400m-patch14-384"
    echo "The pipeline may fail during ConceptCLIP loading."
fi

echo "✅ All required paths verified."
echo "🏠 Cache directory: $CACHE_DIR"
CACHED_MODELS=$(ls -1 "$CACHE_DIR" | grep "models--" | wc -l)
echo "📦 Found $CACHED_MODELS cached model(s)"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# ==================== GPU ENVIRONMENT CHECK ====================
echo ""
echo "🔧 GPU Environment Check:"
echo "========================"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-Not set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"

# Test CUDA availability
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}')
else:
    print('⚠️ CUDA not detected - pipeline will run on CPU')
"

# ==================== CRITICAL: USE SRUN FOR GPU ALLOCATION ====================
echo ""
echo "🚀 Starting MILK10k pipeline with srun (CRITICAL for GPU access)..."
echo "=================================================================="

# CRITICAL: Use srun to launch the Python script
# This is what actually allocates the GPU to your process on Narval
srun python "$PYTHON_SCRIPT"
EXIT_CODE=$?

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
        
        # Count directories (one per processed image)
        DIR_COUNT=$(find "${OUTPUT_PATH}/segmented_for_conceptclip" -mindepth 1 -type d 2>/dev/null | wc -l)
        echo "📁 Images processed: $DIR_COUNT"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/processing_report.json" ]; then
        echo "📄 Processing report generated"
        
        # Extract key metrics from report if possible
        if command -v jq &> /dev/null; then
            echo "📈 Quick Report Summary:"
            echo "  - Device used: $(jq -r '.system_info.device_used' "${OUTPUT_PATH}/reports/processing_report.json" 2>/dev/null || echo 'N/A')"
            echo "  - Offline mode: $(jq -r '.system_info.offline_mode' "${OUTPUT_PATH}/reports/processing_report.json" 2>/dev/null || echo 'N/A')"
            echo "  - Images processed: $(jq -r '.dataset_info.total_images_found' "${OUTPUT_PATH}/reports/processing_report.json" 2>/dev/null || echo 'N/A')"
            echo "  - Overall accuracy: $(jq -r '.accuracy_metrics.overall_accuracy' "${OUTPUT_PATH}/reports/processing_report.json" 2>/dev/null || echo 'N/A')"
        fi
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/detailed_results.csv" ]; then
        echo "📄 Detailed results CSV generated"
    fi
    
    if [ -f "${OUTPUT_PATH}/visualizations/summary_plots.png" ]; then
        echo "📊 Summary visualizations created"
    fi
    
else
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log and your script's output for details"
    
    # Check for common error indicators
    if [ -f "${SLURM_JOB_ID}.err" ]; then
        echo "🔍 Checking error log for common issues..."
        if grep -q "CUDA out of memory" "${SLURM_JOB_ID}.err"; then
            echo "  - GPU memory issue detected"
        fi
        if grep -q "No module named" "${SLURM_JOB_ID}.err"; then
            echo "  - Missing Python module detected"
        fi
        if grep -q "FileNotFoundError" "${SLURM_JOB_ID}.err"; then
            echo "  - File/directory access issue detected"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "Output files location: $OUTPUT_PATH"
echo "Cache directory used: $CACHE_DIR"
echo "=========================================="

# Deactivate venv (this is handled by the script's exit, but good practice to show)
deactivate

exit $EXIT_CODE
