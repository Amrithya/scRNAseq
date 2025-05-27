#!/bin/bash

model_name="lr"
export SLURM_JOB_NAME="$model_name"

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/$model_name_%A_%a.out
#SBATCH --error=results/$model_name_%A_%a.err
#SBATCH --array=1

# run your job
poetry run python -m gene_final.py \
    -m "$model_name" \
    -c

echo "All Done at $(date)!"
wait
