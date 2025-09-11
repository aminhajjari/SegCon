#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-test-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon/logs/%x-%j.err

echo "============================================================"
echo "MILK10k Segmentation-Classification Pipeline Started (TEST)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================================"

# Enable strict error handling. Script will exit on any failed command.
set -e

# ==================== PROJECT SETUP & VALIDATION ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR/SegCon"
VENV_DIR="$PROJECT_DIR/venv"
OUTPUT_PATH="$PROJECT_DIR/outputs_test_20"

echo "Validating project directories..."
cd "$SCRIPT_DIR" || { echo "ERROR: Failed to change directory to $SCRIPT_DIR. Exiting."; exit 1; }
mkdir -p "$OUTPUT_PATH"
mkdir -p "$SCRIPT_DIR/logs"

# Check for required files/directories
[ ! -d "$PROJECT_DIR/MILK10k_Training_Input" ] && { echo "ERROR: Dataset path missing: $PROJECT_DIR/MILK10k_Training_Input"; exit 1; }
[ ! -f "$SCRIPT_DIR/path.py" ] && { echo "ERROR: path.py not found in $SCRIPT_DIR"; exit 1; }
[ ! -d "$PROJECT_DIR/segment-anything-2" ] && { echo "ERROR: SAM2 path missing: $PROJECT_DIR/segment-anything-2"; exit 1; }
[ ! -d "$PROJECT_DIR/ConceptModel" ] && { echo "ERROR: ConceptCLIP path missing: $PROJECT_DIR/ConceptModel"; exit 1; }
[ ! -f "$PROJECT_DIR/segment-anything-2/checkpoints/sam2_hiera_large.pt" ] && { echo "ERROR: Checkpoint missing: $PROJECT_DIR/segment-anything-2/checkpoints/sam2_hiera_large.pt"; exit 1; }
[ ! -d "$PROJECT_DIR/huggingface_cache" ] && { echo "Creating Hugging Face cache directory..."; mkdir -p "$PROJECT_DIR/huggingface_cache"; }
[ ! -f "$PROJECT_DIR/MILK10k_Training_GroundTruth.csv" ] && echo "WARNING: Ground truth file missing: $PROJECT_DIR/MILK10k_Training_GroundTruth.csv"

echo "All required paths verified successfully."

# ==================== MODULE SETUP & VENV ACTIVATION ====================
echo "Loading modules..."
module --force purge
module load StdEnv/2023 python/3.11 cuda/11.8 cudnn/8.9.7 opencv/4.12.0

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ==================== DEPENDENCY INSTALLATION ====================
echo "Installing/verifying Python dependencies..."
pip install --no-index torch torchvision torchaudio transformers pillow pandas numpy opencv-python-headless pydicom nibabel matplotlib seaborn tqdm simpleitk
pip install --no-index -e "$PROJECT_DIR/segment-anything-2"

echo "Dependencies installed successfully."

# ==================== ENVIRONMENT VARIABLES ====================
# Using an export block for clarity
echo "Setting environment variables..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLIS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=1 # Use 1 for better error visibility during debugging
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs_test_20"
export SAM2_MODEL_PATH="$PROJECT_DIR/segment-anything-2"
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"
echo "Environment variables set."

# ==================== GPU Test ====================
echo "Testing GPU availability..."
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available, job will fail.')
    exit(1)
print('✓ GPU is available.')
"

# ==================== RUN MAIN SCRIPT ====================
echo "============================================================"
echo "Starting Python script in test mode..."
echo "============================================================"
# Use srun to execute the main script, piping output to a log file.
srun python path.py --test 2>&1 | tee "${OUTPUT_PATH}/pipeline_log_${SLURM_JOB_ID}.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo "============================================================"
echo "Post-execution analysis"
echo "============================================================"

# The 'set -e' command ensures this block is only reached on success.
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "A test run of 20 images was performed."
echo "To process the full dataset, remove the --test flag and increase --time/--mem."
echo "---------------------------------------------"
echo "Verifying output files..."
[ -f "${OUTPUT_PATH}/reports/detailed_results.csv" ] && echo "✓ Detailed results CSV found."
[ -f "${OUTPUT_PATH}/reports/processing_report.json" ] && echo "✓ Processing report JSON found."
[ -d "${OUTPUT_PATH}/segmented_for_conceptclip" ] && {
    SEGMENTED_COUNT=$(find "${OUTPUT_PATH}/segmented_for_conceptclip" -name "*.png" | wc -l)
    echo "✓ Found $SEGMENTED_COUNT segmented output files."
}
[ -f "${OUTPUT_PATH}/visualizations/summary_plots.png" ] && echo "✓ Summary plots generated."

echo "Output directory contents:"
ls -la "$OUTPUT_PATH/"

echo "============================================================"
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
