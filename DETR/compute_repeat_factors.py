import json
import math
from collections import defaultdict
import pickle

ANN_FILE = "data/train/train_split.json"
REPEAT_THRESHOLD = 0.1   # standard LVIS value

with open(ANN_FILE, "r") as f:
    coco = json.load(f)

# Collect image ids per category
img_ids_per_cat = defaultdict(set)
for ann in coco["annotations"]:
    img_ids_per_cat[ann["category_id"]].add(ann["image_id"])

num_images = len(coco["images"])

# Category frequency
cat_freq = {
    c: len(img_ids) / num_images
    for c, img_ids in img_ids_per_cat.items()
}

# Category repeat factor
cat_repeat = {
    c: max(1.0, math.sqrt(REPEAT_THRESHOLD / f))
    for c, f in cat_freq.items()
}

# Image repeat factor = max over its categories
img_repeat = defaultdict(lambda: 1.0)

img_to_cats = defaultdict(set)
for ann in coco["annotations"]:
    img_to_cats[ann["image_id"]].add(ann["category_id"])

for img in coco["images"]:
    img_id = img["id"]
    img_repeat[img_id] = max(cat_repeat[c] for c in img_to_cats[img_id])

with open("repeat_factors_augm.pkl", "wb") as f:
    pickle.dump(dict(img_repeat), f)

print("Saved repeat_factors.pkl")
