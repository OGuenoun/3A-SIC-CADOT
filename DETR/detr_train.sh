#!/usr/bin/env bash
#OAR -n detr_train_augmented
#OAR -l /nodes=1/gpu=2,walltime=7:00:00
#OAR --project pr-material-acceleration
#OAR -O logs_detr/logs_detr_augm/%jobid%.out
#OAR -E logs_detr/logs_detr_augm/%jobid%.err

#OAR -p gpumodel='A100'
set -euo pipefail



echo "Job started on $(hostname)"
echo "Date: $(date)"

source /applis/environments/conda.sh
conda activate cadot-detr


export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4


torchrun \
  --nproc_per_node=2 \
  DETR/train.py

echo "Job finished on $(hostname)"
echo "Date: $(date)"