#!/usr/bin/env bash
#OAR -n detr_deformable
#OAR -l /nodes=1/gpu=1,walltime=00:30:00
#OAR --project pr-material-acceleration
#OAR -O logs/%jobid%.out
#OAR -E logs/%jobid%.err

#OAR -p gpumodel='A100'

source /applis/environments/conda.sh
conda activate mmdet-cadot

TRAIN_TOOL=$(python - << 'EOF'
import mmdet, os
print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/tools/train.py"))
EOF
)


python ${TRAIN_TOOL/\/train.py/\/test.py} \
  configs/cadot_deformable_detr.py \
  work_dirs/cadot_deformable_detr_augmented/epoch_50.pth \
  --out work_dirs/cadot_deformable_detr_augmented/preds.pkl


