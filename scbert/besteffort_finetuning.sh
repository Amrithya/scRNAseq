#!/bin/bash

#SBATCH --job-name="besteffort_finetune"
#SBATCH --partition=besteffort
#SBATCH --nodelist=lisnode4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=results/besteffort_finetune_%A_%a.out
#SBATCH --error=results/besteffort_finetune_%A_%a.err
#SBATCH --requeue

echo $CUDA_VISIBLE_DEVICES

RESUME_FLAG=""
if [ -f ckpts/finetune_latest.pth ]; then
    echo "Checkpoint found. Resuming training..."
    RESUME_FLAG="--resume"
fi

CUDA_LAUNCH_BLOCKING=1 poetry run torchrun --nproc_per_node=4 besteffort_finetuning.py \
    --data_path "/data1/data/corpus/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/panglao_pretrain.pth" \
    --resume $RESUME_FLAG

echo "All Done!"
wait
