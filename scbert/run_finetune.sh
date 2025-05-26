#!/bin/bash

#SBATCH --job-name="scBERT_finetune"          # Job Name

#SBATCH --partition=gpu                     # Partition name

#SBATCH --gres=gpu:2

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=3

#SBATCH --time=24:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --hint=nomultithread

#SBATCH --output=results/scBERT_finetune_%A_%a.out    # STDOUT file

#SBATCH --error=results/scBERT_finetune_%A_%a.err     # STDERR file

#SBATCH --array=1                       # Array job indices

echo "gooo"
poetry run python -u -m torch.distributed.launch finetune.py \
	--data_path "/data1/data/corpus/Zheng68K.h5ad" \
	--model_path "/data1/data/corpus/panglao_pretrain.pth"

echo "All Done!"
wait
