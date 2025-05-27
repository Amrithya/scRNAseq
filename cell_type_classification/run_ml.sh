#!/bin/bash

model_name="lr"
output_dir="results"
mkdir -p "$output_dir"

exec > "${output_dir}/${model_name}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

#SBATCH --job-name=lr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=1

poetry run python -u gene_final.py \
    -m "$model_name" \
    -c

echo "All Done at $(date)!"
wait
