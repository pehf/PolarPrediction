#!/bin/bash

DIR=slurm/NeurIPS23

# synthetic
sbatch $DIR/PLANTED_translate.sh
sbatch $DIR/PLANTED_rotate.sh
sbatch $DIR/PLANTED_translate_open.sh
sbatch $DIR/PLANTED_translate_rotate.sh

# baselines
sbatch $DIR/DAVIS_C.sh
sbatch $DIR/DAVIS_cMC.sh
sbatch $DIR/DAVIS_Spyr.sh

# models
sbatch --array=0-9 $DIR/DAVIS_mPP_seed.sh
sbatch --array=0-9 $DIR/DAVIS_mQP_seed.sh
sbatch --array=0-9 $DIR/DAVIS_CNN_seed.sh
sbatch --array=0-9 $DIR/DAVIS_Unet_seed.sh

# experiment
sbatch --array=0,1 $DIR/DAVIS_mPP_tied.sh