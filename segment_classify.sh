#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-pipeline
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

# Set up project directory for clarity and robustness
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR/SegCon"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="path.py"

# Navigate to script directory
cd "$SCRIPT_DIR" || {
    echo "❌ ERROR: Failed to change directory to $SCRIPT_DIR. Exiting."
    exit 1
}

# Purge existing modules and load required ones
echo "📦 Loading required modules..."
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
module load opencv/4.12.0

echo "✅ Modules loaded successfully."
module list

# Activate virtual environment
echo "💻 Activating virtual environment from $VENV_DIR..."
source "$VENV_DIR/bin/activate" || {
    echo "❌ ERROR: Failed to activate virtual environment. Exiting."
    exit 1
}
echo "✅ Virtual environment activated."

# Set environment variables for local model paths and data
echo "⚙️ Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/groundtruth.csv"
export OUTPUT_PATH="$PROJECT_DIR/outputs"

# Performance optimizations for Slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE

# Run the main Python pipeline script
echo "🚀 Starting the MILK10k pipeline..."
python "$PYTHON_SCRIPT" || {
    echo "❌ ERROR: Pipeline execution failed. Please check the error log."
    exit 1
}

# Check final exit status and report
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    echo "=========================================="
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "❌ Pipeline failed. Check error log for details."
    echo "=========================================="
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
