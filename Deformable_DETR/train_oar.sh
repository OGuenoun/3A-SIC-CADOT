#!/usr/bin/env bash
#OAR -n detr_deformable
#OAR -l /nodes=1/gpu=2,walltime=10:00:00
#OAR --project pr-material-acceleration
#OAR -O logs/%jobid%.out
#OAR -E logs/%jobid%.err

#OAR -p gpumodel='V100'

set -euo pipefail


source /applis/environments/conda.sh
conda activate mmdet-cadot

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


# Sanity prints

python - << 'EOF'
import torch, numpy as np
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", np.__version__)
EOF

# Paths

CONFIG="configs/cadot_deformable_detr.py"
WORKDIR="work_dirs/cadot_deformable_detr_augmented"

mkdir -p "${WORKDIR}"

# MMDetection train tool
TRAIN_TOOL=$(python - << 'EOF'
import mmdet, os
print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))
EOF
)

echo "TRAIN COMMAND:"
echo torchrun --nproc_per_node=2 ${TRAIN_TOOL} ${CONFIG} --work-dir ${WORKDIR}

torchrun --nproc_per_node=2 ${TRAIN_TOOL} \
  ${CONFIG} \
  --launcher pytorch \
  --work-dir ${WORKDIR} \
  

