#!/bin/bash

#SBATCH --job-name="inference"          # Job Name

#SBATCH --partition=besteffort                     # Partition name

#SBATCH --gres=gpu:4

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=24:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/inference_%A_%a.out    # STDOUT file

#SBATCH --error=results/inference_%A_%a.err     # STDERR file

CUDA_LAUNCH_BLOCKING=1 poetry run torchrun --nproc_per_node=4 inference.py

echo "All Done!"
wait
