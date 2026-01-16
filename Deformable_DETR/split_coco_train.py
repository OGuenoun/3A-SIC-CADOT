import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)

def split_coco(coco):
    random.seed(SEED)

    images = coco["images"]
    annotations = coco["annotations"]

    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)

    n_val = int(len(img_ids) * VAL_RATIO)
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    def filter_coco(img_id_set):
        imgs = [img for img in images if img["id"] in img_id_set]
        anns = [ann for ann in annotations if ann["image_id"] in img_id_set]
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": imgs,
            "annotations": anns,
        }

    return filter_coco(train_ids), filter_coco(val_ids)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original train COCO json")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    coco = load_json(args.input)
    train_coco, val_coco = split_coco(coco)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_split.json"
    val_path = out_dir / "val_split.json"

    save_json(train_coco, train_path)
    save_json(val_coco, val_path)

    print(f"Train split: {len(train_coco['images'])} images")
    print(f"Val split:   {len(val_coco['images'])} images")

if __name__ == "__main__":
    main()
