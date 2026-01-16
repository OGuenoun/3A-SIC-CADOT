from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

ann_file = "data/valid_as_test.json"
pred_file = "Deformable_DETR/preds.pkl"

coco_gt = COCO(ann_file)

# MMDet saves pkl → convert to COCO json first
import mmengine
preds = mmengine.load(pred_file)
json.dump(preds, open("preds.json", "w"))

coco_dt = coco_gt.loadRes("preds.json")

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
