#!/bin/bash
#OAR -n diffusion_model
#OAR -l /nodes=1/gpu=1,walltime=00:05:00
#OAR --stdout output.out
#OAR --stderr error.err
#OAR -p gpumodel='V100'
#OAR --project pr-material-acceleration
source /applis/environments/conda.sh
source /applis/environments/cuda_env.sh bigfoot  12.6
conda activate proj
cd proj/3A-SIC-CADOT
python train_RCNN.py

