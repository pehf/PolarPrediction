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
    --model mPP \
    --filter-size 11 \
    --num-channels 16 \
    --crop 17 \
    --num-scales 4 \
    --pad-mode same \
    --branch phase \
    --epsilon 1e-10 \
    --tied $SLURM_ARRAY_TASK_ID \
    --dataset DAVIS \
    --image-size 128 \
    --num-downs 1 \
    --num-crops 1 \
    --fold 2017 \
    --normalize 1 \
    --dataset DAVIS \
    --image-size 128 \
    --num-downs 1 \
    --num-crops 1 \
    --fold 2017 \
    --normalize 1 \
    --output-dir DAVIS/experiments/tied/mPP_tied${SLURM_ARRAY_TASK_ID}
