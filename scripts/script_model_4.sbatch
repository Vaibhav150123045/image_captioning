#!/bin/bash
#SBATCH --job-name="VM-dl3-Model4_training"
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1


module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dl_a3
python train.py --device-id=0 --config-file-path=./configs/config_model_4.yaml --experiment-name=model_4 --num-epochs=50 --no-log --num-workers=2
