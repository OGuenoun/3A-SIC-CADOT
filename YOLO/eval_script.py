import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

# CONFIGURATION

WEIGHTS = "runs_cadot/yolov8s_800_baseline/weights/best.pt"
DATA = "cadot.yaml"      
OUT_DIR = "runs_eval/yolov8s_test_api"

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(WEIGHTS)

# RUN EVALUATION

metrics = model.val(
    data=DATA,
    split="test",
    imgsz=800,
    device="cpu",        
    workers=8,
    plots=True,
    save_json=True,
    project="runs_eval,
    name="yolov8s_test_api",
)

# 1. GLOBAL METRICS (SCALARS)

overall_metrics = {
    "mAP50": float(metrics.box.map50),
    "mAP75": float(metrics.box.map75),
    "mAP50_95": float(metrics.box.map),
    "mean_precision": float(np.mean(metrics.box.p)),
    "mean_recall": float(np.mean(metrics.box.r)),
}

print("\nOverall metrics:")
for k, v in overall_metrics.items():
    print(f"{k}: {v:.4f}")

pd.DataFrame([overall_metrics]).to_csv(
    os.path.join(OUT_DIR, "overall_metrics.csv"),
    index=False
)

# 2. PER-CLASS METRICS
names = model.names  # dict: class_id -> class_name

per_class_rows = []
for cid, ap in enumerate(metrics.box.maps):
    per_class_rows.append({
        "class_id": cid,
        "class_name": names[cid],
        "AP50_95": float(ap),
        "precision": float(metrics.box.p[cid]),
        "recall": float(metrics.box.r[cid]),
        "F1": float(metrics.box.f1[cid]),
    })

per_class_df = pd.DataFrame(per_class_rows).sort_values(
    "AP50_95", ascending=False
)

per_class_df.to_csv(
    os.path.join(OUT_DIR, "per_class_metrics.csv"),
    index=False
)

print("\nSaved per-class metrics to per_class_metrics.csv")

# ==========================================================
# 3. PER-CLASS AP@0.5 and AP@0.75 (FROM all_ap)
# ==========================================================
# all_ap shape: (nc, 10) for IoU thresholds 0.50:0.95
ious = np.arange(0.50, 1.00, 0.05)

ap05_idx = np.where(np.isclose(ious, 0.50))[0][0]
ap075_idx = np.where(np.isclose(ious, 0.75))[0][0]

ap50 = metrics.box.all_ap[:, ap05_idx]
ap75 = metrics.box.all_ap[:, ap075_idx]

ap_iou_rows = []
for cid in range(len(names)):
    ap_iou_rows.append({
        "class_id": cid,
        "class_name": names[cid],
        "AP50": float(ap50[cid]),
        "AP75": float(ap75[cid]),
    })

pd.DataFrame(ap_iou_rows).to_csv(
    os.path.join(OUT_DIR, "per_class_ap50_ap75.csv"),
    index=False
)

print("Saved per-class AP50 and AP75.")

##
print("\nEvaluation completed successfully.")
print(f"All metrics saved under: {OUT_DIR}")
