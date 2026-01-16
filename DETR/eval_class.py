from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

ANN_FILE = "data/data_coco/valid_as_test.json"
PRED_FILE = "DETR/predictions_DETR.json"

coco_gt = COCO(ANN_FILE)
coco_dt = coco_gt.loadRes(PRED_FILE)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()

cat_ids = coco_gt.getCatIds()
cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

# COCOeval precision dims:
# [IoU, Recall, Class, Area, MaxDets]
# IoU index: 0=.5, 5=.75, : = .5:.95
# Area index: 0=all, 1=small, 2=medium, 3=large

def mean_ap(prec):
    prec = prec[prec > -1]
    return np.mean(prec) if prec.size else float("nan")

print("+------------------+-------+--------+--------+-------+-------+-------+")
print("| category         | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |")
print("+------------------+-------+--------+--------+-------+-------+-------+")

for i, cat_id in enumerate(cat_ids):
    name = cat_names[cat_id]

    p_all = coco_eval.eval["precision"][:, :, i, 0, 2]
    p_50  = coco_eval.eval["precision"][0, :, i, 0, 2]
    p_75  = coco_eval.eval["precision"][5, :, i, 0, 2]

    p_s = coco_eval.eval["precision"][:, :, i, 1, 2]
    p_m = coco_eval.eval["precision"][:, :, i, 2, 2]
    p_l = coco_eval.eval["precision"][:, :, i, 3, 2]

    print(
        f"| {name:<16} | "
        f"{mean_ap(p_all):.3f} | "
        f"{mean_ap(p_50):.3f} | "
        f"{mean_ap(p_75):.3f} | "
        f"{mean_ap(p_s):.3f} | "
        f"{mean_ap(p_m):.3f} | "
        f"{mean_ap(p_l):.3f} |"
    )

print("+------------------+-------+--------+--------+-------+-------+-------+")
coco_eval.summarize()

stats = coco_eval.stats
print("\nGlobal metrics:")
print(f"bbox_mAP:     {stats[0]:.3f}")
print(f"bbox_mAP_50:  {stats[1]:.3f}")
print(f"bbox_mAP_75:  {stats[2]:.3f}")
print(f"bbox_mAP_s:   {stats[3]:.3f}")
print(f"bbox_mAP_m:   {stats[4]:.3f}")
print(f"bbox_mAP_l:   {stats[5]:.3f}")
