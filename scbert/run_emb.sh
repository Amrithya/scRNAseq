#!/bin/bash

#SBATCH --job-name="embeddings"          # Job Name

#SBATCH --partition=gpu                     # Partition name

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=08:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/embeddings_%A_%a.out    # STDOUT file

#SBATCH --error=results/embeddings_%A_%a.err     # STDERR file

poetry run python -u embeddings.py

echo "All Done!"
wait
