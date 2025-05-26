#!/bin/bash

#SBATCH --job-name="besteffort_finetune"
#SBATCH --partition=besteffort
#SBATCH --nodelist=lisnode3
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=results/besteffort_finetune_%A_%a.out
#SBATCH --error=results/besteffort_finetune_%A_%a.err
#SBATCH --requeue

CUDA_LAUNCH_BLOCKING=1 poetry run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 besteffort_finetuning.py \
    --data_path "/data1/data/corpus/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/panglao_pretrain.pth"

echo "All Done!"
wait
