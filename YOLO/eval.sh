#!/usr/bin/env bash
#OAR -n dYolo
#OAR -l /nodes=1/gpu=1,walltime=00:30:00
#OAR --project pr-material-acceleration
#OAR -O logs_yolo/%jobid%.out
#OAR -E logs_yolo/%jobid%.err

#OAR -p gpumodel='A100'

source /applis/environments/conda.sh
conda activate yolo_env

# Safety check
python - << 'EOF'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
print("Using GPU:", torch.cuda.get_device_name(0))
EOF


python eval_script.py

