#!/bin/bash

#SBATCH --job-name=mrh

#SBATCH --partition=gpu_p2

#SBATCH --qos=qos_gpu-t4

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=3

#SBATCH --hint=nomultithread

#SBATCH --time=100:00:00

#SBATCH --output=/gpfswork/rech/qcr/usi11oa/jobs_scLight/%j/out.txt

#SBATCH --error=/gpfswork/rech/qcr/usi11oa/jobs_scLight/%j/out.txt

poetry run python -u -m torch.distributed.launch finetune.py --data_path "/data1/data/corpus/Zheng68K.h5ad" --model_path "/data1/data/corpus/panglao_pretrain.pth"

file_path="/gpfswork/rech/qcr/usi11oa/jobs_scLight/${SLURM_JOB_ID}/out.txt"
folder=$(tail -n 1 "$file_path")
echo $folder

#mv "./folder""folder""{WORK}/jobs_scLight/${SLURM_JOB_ID}"
