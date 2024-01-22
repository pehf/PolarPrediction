#!/bin/bash
#SBATCH --job-name=compexp
#SBATCH --time=0-1:00:00
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
    --model PP \
    --filter-size 16 \
    --num-channels 32 \
    --crop 0 \
    --pad-mode valid \
    --branch phase \
    --epsilon 1e-10 \
    --dataset PLANTED \
    --transform rotate \
    --image-size 128 \
    --num-downs 1 \
    --num-crops 1 \
    --fold 2017 \
    --normalize 1 \
    --scheduler step \
    --learning-rate 1e-3 \
    --num-epochs 1000 \
    --output-dir PLANTED/rotate
