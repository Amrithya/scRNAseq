#!/bin/bash

#SBATCH --job-name="besteffort_finetune"
#SBATCH --partition=besteffort
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=results/besteffort_finetune_%A_%a.out
#SBATCH --error=results/besteffort_finetune_%A_%a.err
#SBATCH --requeue

echo $CUDA_VISIBLE_DEVICES

NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_LAUNCH_BLOCKING=1 poetry run torchrun --nproc_per_node=4 besteffort_finetuning.py \
    --data_path "/data1/data/corpus/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/panglao_pretrain.pth"

echo "All Done!"
wait
