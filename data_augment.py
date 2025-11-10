import json
from collections import Counter
import os
import cv2
import numpy as np
curr_dir=os.getcwd()
train_dir=os.path.join(curr_dir,"train\\")
json_path=os.path.join(train_dir,"_annotations.coco.json")
with open(json_path, "r") as f:
        coco = json.load(f)
def get_underrepresented_categories(coco, min_fraction=0.3):

    cat_counts = Counter(a["category_id"] for a in coco["annotations"])
    max_count = max(cat_counts.values())
    threshold = max_count * min_fraction

    under = {cid for cid, c in cat_counts.items() if c < threshold}

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    for cid in under:
        print(f"  {cid} ({id_to_name.get(cid, 'unknown')}): {cat_counts[cid]}")

    return under

def collect_patches(coco, images_dir, under_cats, max_patches_per_cat=15):

    images = {img["id"]: img for img in coco["images"]}

    patches_by_cat = {cid: [] for cid in under_cats}
    counts = {cid: 0 for cid in under_cats}

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in under_cats:
            continue

        if counts[cid] >= max_patches_per_cat:
            continue

        img_info = images[ann["image_id"]]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        H, W = img.shape[:2]
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        patch = img[y:y2, x:x2].copy()
        patches_by_cat[cid].append(patch)
        counts[cid] += 1
    return patches_by_cat
def bbox_iou_coco(box1, box2):
    if isinstance(box2[0], (int, float)):
        box2 = [box2]
    x1, y1, w1, h1 = box1
    x1b, y1b = x1 + w1, y1 + h1

    ious = []
    for b in box2:

        x2, y2, w2, h2 = b
        x2b, y2b = x2 + w2, y2 + h2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1b, x2b)
        inter_y2 = min(y1b, y2b)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter_area + 1e-6  # avoid /0

        iou = inter_area / union
        ious.append(iou)
    return max(ious) if ious else 0.0
def place_patch_on_image(img, patch, existing_bboxes, max_tries=50, max_iou=0.3):
    H, W = img.shape[:2]
    ph, pw = patch.shape[:2]
    if ph >= H or pw >= W:
        return None, None
    for _ in range(max_tries):
        x = np.random.randint(0, W - pw)
        y = np.random.randint(0, H - ph)
        candidate_bbox = [x, y, pw, ph]

        if existing_bboxes:
            iou = bbox_iou_coco(candidate_bbox, existing_bboxes)
            if iou > max_iou:
                continue
        new_img = img.copy()
        new_img[y:y+ph, x:x+pw] = patch
        return new_img, candidate_bbox
    return None, None
from collections import defaultdict, Counter

def augment_by_patching(
    coco,
    images_dir,
    out_images_dir,
    max_bboxes_per_image=3,
    min_fraction_underrep=0.5,
    patches_per_image=2,
    max_patches_per_cat=200,
    out_json_path="annotations_patched.json"
):

    os.makedirs(out_images_dir, exist_ok=True)
    images = {img["id"]: img for img in coco["images"]}

    under_cats=get_underrepresented_categories(coco,min_fraction_underrep)

    patches_by_cat = collect_patches(
        coco, images_dir, under_cats, max_patches_per_cat=max_patches_per_cat
    )

    all_patches = []
    for cid, patches in patches_by_cat.items():
        for p in patches:
            all_patches.append((cid, p))

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    candidate_image_ids = [
        img_id for img_id, anns in anns_by_image.items()
        if len(anns) < max_bboxes_per_image
    ]
    print(f"Found {len(candidate_image_ids)} candidate background images (< {max_bboxes_per_image} bboxes).")

    next_image_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = max(a["id"] for a in coco["annotations"]) + 1

    new_images = []
    new_annotations = []

    for img_id in candidate_image_ids:
        img_info = images[img_id]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path)
            continue

        H, W = img.shape[:2]
        existing_bboxes = [ann["bbox"] for ann in anns_by_image[img_id]]

        # start with original as base
        aug_img = img.copy()
        bboxes = existing_bboxes.copy()
        cat_ids = [ann["category_id"] for ann in anns_by_image[img_id]]

        # 5) Paste patches
        for _ in range(patches_per_image):
            cid, patch = all_patches[np.random.randint(0, len(all_patches))]

            placed_img, new_bbox = place_patch_on_image(
                aug_img, patch, bboxes, max_tries=50, max_iou=0.2
            )
            if placed_img is None:
                continue

            aug_img = placed_img
            bboxes.append(new_bbox)
            cat_ids.append(cid)

        # If nothing was added, skip
        if len(bboxes) == len(existing_bboxes):
            continue

        # Save new image
        base, ext = os.path.splitext(os.path.basename(img_info["file_name"]))
        new_fname = f"{base}_patched{ext}"
        out_path = os.path.join(out_images_dir, new_fname)
        cv2.imwrite(out_path, aug_img)

        # New image entry
        new_img = {
            "id": next_image_id,
            "file_name": os.path.relpath(out_path, images_dir),  # or just new_fname
            "width": W,
            "height": H,
        }
        new_images.append(new_img)

        # New annotations
        for bbox, cid in zip(bboxes[len(existing_bboxes):], cat_ids[len(existing_bboxes):]):
            new_ann = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": cid,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
            new_annotations.append(new_ann)
            next_ann_id += 1

        next_image_id += 1

    # 6) Save updated COCO
    coco["images"].extend(new_images)
    coco["annotations"].extend(new_annotations)

    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"Created {len(new_images)} patched images with {len(new_annotations)} new annotations.")
    print("Saved updated JSON to:", out_json_path)
under_rep=get_underrepresented_categories(coco)
out_imgs_dir=os.path.join(curr_dir,"images_patched")
augment_by_patching(coco,train_dir,out_imgs_dir,max_bboxes_per_image=10,min_fraction_underrep=0.3,patches_per_image=6,max_patches_per_cat=200)
