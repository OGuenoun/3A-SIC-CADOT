import os
import json
import shutil
from collections import defaultdict, Counter
from PIL import Image


# Configuration
#
SRC_ROOT = "data/train"                 # contains images/ and json files
SRC_IMAGES = os.path.join(SRC_ROOT, "images")

TRAIN_JSON = os.path.join(SRC_ROOT, "train_split_fixed.json")
VAL_JSON   = os.path.join(SRC_ROOT, "val_split_fixed.json")

OUT_ROOT = "Yolo/data"               # output YOLO dataset root


# Utils

def coco_to_yolo_bbox(bbox, W, H):
    """
    COCO bbox: [x_min, y_min, width, height] in pixels
    YOLO bbox: x_center, y_center, width, height (normalized)
    """
    x, y, w, h = bbox
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    return xc, yc, w / W, h / H


def build_used_category_map(coco):
    """
    Build COCO id -> YOLO id mapping
    ONLY for categories that actually appear in annotations.
    Ensures:
      - no empty classes
      - YOLO ids start at 0
      - YOLO class 0 = basketball field
    """
    used_cat_ids = Counter()
    for ann in coco["annotations"]:
        used_cat_ids[ann["category_id"]] += 1

    used_categories = [
        c for c in coco["categories"]
        if c["id"] in used_cat_ids
    ]

    # Sort by original COCO id for stable ordering
    used_categories = sorted(used_categories, key=lambda x: x["id"])

    cat_map = {c["id"]: i for i, c in enumerate(used_categories)}

    print("\nFINAL COCO → YOLO CLASS MAPPING (USED ONLY):")
    for c in used_categories:
        print(f"  COCO id {c['id']} → YOLO id {cat_map[c['id']]} ({c['name']})")

    print(f"\nTotal YOLO classes: {len(cat_map)}")
    return cat_map


def convert_split(json_path, split_name, cat_map):
    with open(json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    img_out_dir = os.path.join(OUT_ROOT, "images", split_name)
    lbl_out_dir = os.path.join(OUT_ROOT, "labels", split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for img_id, img in images.items():
        img_name = img["file_name"]
        src_img_path = os.path.join(SRC_IMAGES, img_name)

        if not os.path.exists(src_img_path):
            print(f"WARNING: missing image {src_img_path}")
            continue

        # Copy image
        shutil.copy2(src_img_path, os.path.join(img_out_dir, img_name))

        W, H = Image.open(src_img_path).size
        label_path = os.path.join(
            lbl_out_dir, os.path.splitext(img_name)[0] + ".txt"
        )

        with open(label_path, "w") as f:
            for ann in anns_by_image.get(img_id, []):
                coco_cat = ann["category_id"]

                # Skip unused categories (e.g. small-object)
                if coco_cat not in cat_map:
                    continue

                yolo_cat = cat_map[coco_cat]
                xc, yc, w, h = coco_to_yolo_bbox(ann["bbox"], W, H)

                # Safety check
                if w <= 0 or h <= 0:
                    continue

                f.write(
                    f"{yolo_cat} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                )


def main():
    # Build mapping from TRAIN annotations only
    with open(TRAIN_JSON) as f:
        coco_train = json.load(f)

    cat_map = build_used_category_map(coco_train)

    # Convert splits
    convert_split(TRAIN_JSON, "train", cat_map)
    convert_split(VAL_JSON,   "val",   cat_map)

    print("\nCOCO → YOLO conversion completed successfully.")
    print(f"YOLO dataset written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
