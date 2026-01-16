import os
import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image


# ---------------- CONFIG ----------------
MODEL_DIR = "DETR/DETR_chekpoints"
IMG_DIR   = "data/data_coco/valid/"
ANN_FILE  = "data/data_coco/valid_as_test.json"
OUT_JSON  = "DETR/predictions_DETR.json"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
#


def main():
    print("Loading model...")
    model = DetrForObjectDetection.from_pretrained(MODEL_DIR).to(DEVICE)
    processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()

    coco_gt = COCO(ANN_FILE)
    img_ids = coco_gt.getImgIds()

    results = []

    for img_id in tqdm(img_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]]).to(DEVICE)
        detections = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0
        )[0]

        for score, label, box in zip(
            detections["scores"],
            detections["labels"],
            detections["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score)
            })

    print("Saving predictions...")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    print("Running COCO evaluation...")
    coco_dt = coco_gt.loadRes(OUT_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save detailed stats
    stats = coco_eval.stats
    print("\nMain metrics:")
    print(f"mAP@[.5:.95]: {stats[0]:.4f}")
    print(f"AP@0.5:       {stats[1]:.4f}")
    print(f"AP@0.75:      {stats[2]:.4f}")
    print(f"AP_small:     {stats[3]:.4f}")
    print(f"AP_medium:    {stats[4]:.4f}")
    print(f"AP_large:     {stats[5]:.4f}")


if __name__ == "__main__":
    main()
