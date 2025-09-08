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

# 1. Navigate to your project directory (where the script is located)
cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegCon

# 2. Clear environment and load the correct modules
module --force purge
module load StdEnv/2023
module load python/3.11.5       # Your venv was created with this version
module load gcc/12.3
module load opencv/4.12.0
module load cuda/12.6

# 3. Activate your virtual environment using its absolute path
echo "Activating virtual environment..."
source /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/venv/bin/activate

# 4. Set environment variables (for paths and performance)
echo "Setting environment variables..."
export DATASET_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
export OUTPUT_PATH="/project/def-arashmoh/shahab33/XAI/outputs"

# 5. Run your Python script
echo "Starting Python script..."
python path.py

# 6. Check for successful completion
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MILK10k Pipeline completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "MILK10k Pipeline failed. Check the error log for details."
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

# 7. Final cleanup
echo "Job finished. Deactivating environment."
deactivate
