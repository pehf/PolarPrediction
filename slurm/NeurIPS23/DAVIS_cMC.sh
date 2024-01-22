#!/bin/bash
#SBATCH --job-name=compexp
#SBATCH --time=0-0:20:00
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

# NOTE: MC is slow (one big loop)
python /mnt/home/pfiquet/PolarPrediction/ppm/predict.py \
    --model cMC \
    --filter-size 8 \
    --crop 17 \
    --branch causal \
    --dataset DAVIS \
    --image-size 128 \
    --num-downs 1 \
    --num-crops 1 \
    --fold 2017 \
    --normalize 1 \
    --gray 1 \
    --train 0 \
    --output-dir DAVIS/baselines/cMC
