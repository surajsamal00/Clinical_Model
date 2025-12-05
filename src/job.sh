#!/bin/bash
#SBATCH --job-name=medsum_train       # Job name
#SBATCH --output=logs/train_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/train_%j.err     # Standard error log  
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --mem=32G                      # RAM
#SBATCH --time=12:00:00                # Max runtime hh:mm:ss
#SBATCH --partition=gpu                # Partition name (adjust as needed)

# ------------------------------
# Load modules or activate environment
# ------------------------------


# Activate your virtual environment
source ~/.venv/bin/activate   # Adjust path to your .venv

# Create logs directory if not exist
mkdir -p logs

# ------------------------------
# Run training
# ------------------------------
# Example: batch_size=4, 5 epochs, AMP enabled
python src/train.py --batch_size 4 --num_epochs 5 --use_amp
