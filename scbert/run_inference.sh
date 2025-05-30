#!/bin/bash

#SBATCH --job-name="conv1d_representations"          # Job Name

#SBATCH --partition=gpu                     # Partition name

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=24:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/conv1d_representations_%A_%a.out    # STDOUT file

#SBATCH --error=results/conv1d_representations_%A_%a.err     # STDERR file

CUDA_LAUNCH_BLOCKING=1 poetry run python inference.py

echo "All Done!"
wait
