#!/usr/bin/env bash
#OAR -n cadot_detr_eval
#OAR -l /nodes=1/gpu=1,walltime=00:30:00
#OAR --project pr-material-acceleration
#OAR -O logs_detr/%jobid%.out
#OAR -E logs_detr/%jobid%.err

#OAR -p gpumodel='A100'

echo "Evaluation job started on $(hostname)"
echo "Date: $(date)"

source /applis/environments/conda.sh
conda activate cadot-detr


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

python DETR/evaluate.py

echo "Evaluation job finished"
echo "Date: $(date)"
