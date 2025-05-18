#!/bin/bash
#SBATCH --job-name="VM-dl2-Model6_training"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=h100:1


module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dl_a3
export WANDB_API_KEY=274d2d1e695f61443c54649beb174a1aba998a64
python train.py --device-id=0 --config-file-path=./configs/config_model_6.yaml --experiment-name=model_6 --num-epochs=50 --num-workers=2
