#!/bin/bash

#SBATCH --job-name="scBERT_nobp_finetune"          # Job Name

#SBATCH --partition=gpu                     # Partition name

#SBATCH --gres=gpu:1

#SBATCH --time=05:00:00                     # Max runtime (HH:MM:SS)

#SBATCH --output=results/scBERT_nobp_finetune_%A_%a.out    # STDOUT file

#SBATCH --error=scBERT_nobp_finetune_%A_%a.err     # STDERR file

#SBATCH --mail-type=ALL                     # Mail notification (BEGIN, END, FAIL, ALL)

#SBATCH --mail-user=amrithyarao1999@gmail.com

#SBATCH --array=1                       # Array job indices
Uncomment below lines if you want to fetch parameters for each array task from a file
LINE=(sed−n"(sed−n"{SLURM_ARRAY_TASK_ID}p" parameters_combinations/chairs_expes.txt)
echo "Running with parameters: $LINE"
Run your distributed training

poetry run python -m torch.distributed.launch  finetune_nobp.py \
          --data_path "/data1/data/corpus/Zheng68K.h5ad" \
            --model_path "/data1/data/corpus/panglao_pretrain.pth"

echo "All Done!"
wait
