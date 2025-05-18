#!/bin/bash
#SBATCH --job-name="VM-dl2-Model2_training"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx3090:1


module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dl_a3
python train.py --device-id=0 --config-file-path=./configs/config_model_2.yaml --experiment-name=model_2 --num-epochs=45 --no-log --resume --num-workers=2
