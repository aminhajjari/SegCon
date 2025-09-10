#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=milk10k-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=MILK10k-pipeline-narval-%j.out
#SBATCH --error=MILK10k-pipeline-narval-%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "=========================================="

# CRITICAL: Set threading environment variables to prevent BLIS/OpenMP conflicts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Set additional environment variables for stability
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# Load required modules
echo "Loading modules..."
module load python/3.9
module load cuda/11.7
module load gcc/9.3.0

# Show loaded modules
echo "Loaded modules:"
module list

# Show CUDA info
echo "CUDA environment:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Activate virtual environment (adjust path as needed)
echo "Activating virtual environment..."
# Uncomment and adjust the path to your virtual environment:
# source /path/to/your/venv/bin/activate

# Alternative: if using conda/mamba environment:
# module load miniconda3
# source activate your_env_name

# Show Python environment info
echo "Python environment:"
which python
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Set working directory (adjust as needed)
cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input

# Show current directory and files
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Verify key files exist
echo "Checking for required files and directories:"
echo "Dataset directory: $(ls -ld MILK10k_Training_Input 2>/dev/null || echo 'NOT FOUND')"
echo "Ground truth file: $(ls -l MILK10k_Training_GroundTruth.csv 2>/dev/null || echo 'NOT FOUND')"
echo "SAM2 directory: $(ls -ld segment-anything-2 2>/dev/null || echo 'NOT FOUND')"
echo "ConceptModel directory: $(ls -ld ConceptModel 2>/dev/null || echo 'NOT FOUND')"

# Run the pipeline
echo "=========================================="
echo "Starting MILK10k pipeline..."
echo "Start time: $(date)"

# Run your Python script
python path.py

# Capture exit code
exit_code=$?

echo "Pipeline completed with exit code: $exit_code"
echo "End time: $(date)"

# Show final job statistics
echo "=========================================="
echo "Job statistics:"
sstat --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize --jobs=$SLURM_JOB_ID || echo "sstat not available"

# Show output directory contents
echo "Output directory contents:"
ls -la outputs/ || echo "outputs directory not found"

echo "Job completed at: $(date)"

# Exit with the same code as the Python script
exit $exit_code
