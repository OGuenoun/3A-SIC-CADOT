#!/usr/bin/env bash
#OAR -n yolo
#OAR -l /nodes=1/gpu=1,walltime=03:00:00
#OAR --project pr-material-acceleration
#OAR -O logs_yolo/%jobid%.out
#OAR -E logs_yolo/%jobid%.err
#OAR -p gpumodel='A100'


set -euo pipefail

echo "Job started on $(hostname)"
echo "Date: $(date)"

source /applis/environments/conda.sh
conda activate yolo_env


python - << 'EOF'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
print("Using GPU:", torch.cuda.get_device_name(0))
EOF

# Train YOLO
yolo detect train \
  data=cadot.yaml \
  model=yolov8s.pt \
  imgsz=800 \
  epochs=100 \
  batch=16 \
  device=0 \
  workers=8 \
  project=runs_cadot \
  name=yolov8s_500 \
  patience=20 \
  multi_scale=True \
  degrees=10 \
  scale=0.5 \
  translate=0.1 \
  fliplr=0.5

