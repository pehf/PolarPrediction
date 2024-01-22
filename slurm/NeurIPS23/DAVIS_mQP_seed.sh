#!/bin/bash
#SBATCH --job-name=compexp
#SBATCH --time=0-2:00:00
#SBATCH --mem=80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --error=/mnt/home/pfiquet/PolarPrediction/logs/%A_%a.err
#SBATCH --output=/mnt/home/pfiquet/PolarPrediction/logs/%A_%a.out

module purge
module load modules/2.1.1
module load ffmpeg

python /mnt/home/pfiquet/PolarPrediction/ppm/predict.py \
    --model mQP \
    --filter-size 11 \
    --num-channels 16 \
    --crop 17 \
    --num-scales 4 \
    --pad-mode same \
    --group-size 4 \
    --num-quadratics 12 \
    --epsilon 1e-10 \
    --dataset DAVIS \
    --image-size 128 \
    --num-downs 1 \
    --num-crops 1 \
    --fold 2017 \
    --normalize 1 \
    --output-dir DAVIS/mQP/mQP_seed${SLURM_ARRAY_TASK_ID} \
    --seed $SLURM_ARRAY_TASK_ID
