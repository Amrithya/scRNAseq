#!/bin/bash

#SBATCH --job-name="inference_predict"          # Job Name

#SBATCH --partition=besteffort                     # Partition name

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=08:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/inference_predict_%A_%a.out    # STDOUT file

#SBATCH --error=results/inference_predict_%A_%a.err     # STDERR file

poetry run python -u train_inference.py

echo "All Done!"
wait
