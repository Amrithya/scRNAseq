#!/bin/bash

#SBATCH --job-name="besteffort_finetune"          # Job Name

#SBATCH --partition=besteffort                     # Use besteffort partition

#SBATCH --gres=gpu:a40-48:4                              # Request 4 GPUs

#SBATCH --ntasks=4                                 # One task per GPU

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=48:00:00                            # Max runtime (HH:MM:SS)

#SBATCH --output=results/besteffort_finetune_%A_%a.out    # STDOUT file

#SBATCH --error=results/besteffort_finetune_%A_%a.err     # STDERR file

#SBATCH --requeue                                 

CUDA_LAUNCH_BLOCKING=1 poetry run python -m torch.distributed.launch --nproc_per_node=4 besteffort_finetuning.py \
          --data_path "/data1/data/corpus/Zheng68K.h5ad" \
          --model_path "/data1/data/corpus/panglao_pretrain.pth"

echo "All Done!"
wait

