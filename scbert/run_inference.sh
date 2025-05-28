#!/bin/bash

#SBATCH --job-name="inference"          # Job Name

#SBATCH --partition=besteffort                     # Partition name

#SBATCH --gres=gpu:4

#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=08:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/inference_%A_%a.out    # STDOUT file

#SBATCH --error=results/inference_%A_%a.err     # STDERR file

poetry run python -u inference.py

echo "All Done!"
wait
