import json
from collections import Counter
import os
import cv2
import numpy as np
from collections import defaultdict, Counter
curr_dir=os.getcwd()
train_dir=os.path.join(curr_dir,"train\\")
json_path=os.path.join(train_dir,"_annotations.coco.json")
with open(json_path, "r") as f:
        coco = json.load(f)
# Number of underrepresented categories
def get_underrepresented_categories(coco, min_fraction=0.3):

    cat_counts = Counter(a["category_id"] for a in coco["annotations"])
    max_count = max(cat_counts.values())
    threshold = max_count * min_fraction

    under = {cid for cid, c in cat_counts.items() if c < threshold}

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    for cid in under:
        print(f"  {cid} ({id_to_name.get(cid, 'unknown')}): {cat_counts[cid]}")

    return under
# Get Patches
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
# Create image to be added
def augment_patch_rect(patch, min_scale=0.5, max_scale=1.5):
    """
    Augment a rectangular patch:
      - random scaling
      - random flips
      - random 90° rotations
      - random brightness/contrast
    """
    img = patch.copy()
    h, w = img.shape[:2]

    # scale
    scale = np.random.uniform(min_scale, max_scale)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h))

    # flips
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
    if np.random.rand() < 0.2:
        img = cv2.flip(img, 0)

    # 90° rotations
    r = np.random.rand()
    if r < 0.25:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif r < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif r < 0.75:
        img = cv2.rotate(img, cv2.ROTATE_180)

    # brightness/contrast
    if np.random.rand() < 0.7:
        alpha = 1.0 + np.random.uniform(-0.2, 0.2)
        beta  = np.random.uniform(-0.2, 0.2) * 255
        img_f = img.astype(np.float32) * alpha + beta
        img   = np.clip(img_f, 0, 255).astype(np.uint8)

    return img
# Assure that there is not a lot of intersection of the patches
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
        union = area1 + area2 - inter_area + 1e-6 

        iou = inter_area / union
        ious.append(iou)
    return max(ious) if ious else 0.0

def make_soft_rect_mask(h, w, edge_frac=0.10):

    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]

    dx = np.abs(x)
    dy = np.abs(y)

    t = edge_frac

    def smooth_step(d):
        m = np.ones_like(d)
        inner = d > t
        u = (d[inner] - t) / (1.0 - t)
        m[inner] = 0.5 * (1 + np.cos(np.clip(u, 0, 1) * np.pi))  
        return np.clip(m, 0.0, 1.0)

    mx = smooth_step(dx)
    my = smooth_step(dy)
    mask = mx * my 
    return mask.astype(np.float32)

def paste_rect_smooth(bg, patch, x, y, edge_frac=0.10):

    ph, pw = patch.shape[:2]
    H, W   = bg.shape[:2]

    x = max(0, min(x, W - pw))
    y = max(0, min(y, H - ph))

    roi = bg[y:y+ph, x:x+pw]

    patch_cm = match_color_LAB_rect(patch, roi)

    alpha = make_soft_rect_mask(ph, pw, edge_frac=edge_frac)[:, :, None]  # HxWx1

    comp = (alpha * patch_cm.astype(np.float32) +
            (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)

    out = bg.copy()
    out[y:y+ph, x:x+pw] = comp
    return out

#Place the patch on the image
def place_patch_on_image(img, patch, existing_bboxes, max_tries=50, max_iou=0.2):
    H, W = img.shape[:2]
    ph, pw = patch.shape[:2]

    # handle patches bigger than image
    if ph >= H or pw >= W:
        scale = min((H - 1) / ph, (W - 1) / pw)
        if scale <= 0:
            return None, None
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        patch = cv2.resize(patch, (new_w, new_h))
        ph, pw = patch.shape[:2]

    max_x = W - pw
    max_y = H - ph
    if max_x <= 0 or max_y <= 0:
        return None, None

    for _ in range(max_tries):
        x = int(np.random.randint(0, max_x + 1))
        y = int(np.random.randint(0, max_y + 1))
        candidate_bbox = [x, y, pw, ph]

        if existing_bboxes:
            iou = bbox_iou_coco(candidate_bbox, existing_bboxes)
            if iou > max_iou:
                continue
        new_img = paste_rect_smooth(img, patch, x, y, edge_frac=0.10)

        return new_img, candidate_bbox

    return None, None
# Match means and stds
def match_color_LAB_rect(patch, roi, eps=1e-6):

    sp = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    dr = cv2.cvtColor(roi,   cv2.COLOR_BGR2LAB).astype(np.float32)

    smu, sstd = sp.mean((0,1)), sp.std((0,1)) + eps
    dmu, dstd = dr.mean((0,1)), dr.std((0,1)) + eps

    sp_adj = (sp - smu) / sstd * dstd + dmu
    sp_adj = np.clip(sp_adj, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sp_adj, cv2.COLOR_LAB2BGR)

# Full pipeline 
def rotate_image_90(img, k):
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    if k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_bbox_coco(bbox, img_w, img_h, k):
    x, y, w, h = bbox
    if k == 0:
        return [x, y, w, h]
    elif k == 2:  # 180°
        new_x = img_w - (x + w)
        new_y = img_h - (y + h)
        return [new_x, new_y, w, h]
    else:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2],
        ], dtype=np.float32)

        if k == 1:  # 90° CW
            rot = np.stack([img_h - corners[:,1], corners[:,0]], axis=1)
            new_w, new_h = img_h, img_w
        else:       # k == 3: 270° CW
            rot = np.stack([corners[:,1], img_w - corners[:,0]], axis=1)
            new_w, new_h = img_h, img_w

        x_min, y_min = rot.min(0)
        x_max, y_max = rot.max(0)
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
def transform_background(img, anns_for_img):
    """
    Apply a random 90° rotation and global brightness/contrast
    to a background image and its boxes.
    """
    H, W = img.shape[:2]
    k = np.random.choice([0, 1, 2, 3])  # 0/90/180/270

    img_rot = rotate_image_90(img, k)
    if k in (0, 2):
        new_W, new_H = W, H
    else:
        new_W, new_H = H, W

    new_anns = []
    for ann in anns_for_img:
        bbox_rot = rotate_bbox_coco(ann["bbox"], W, H, k)
        a2 = ann.copy()
        a2["bbox"] = bbox_rot
        a2["area"] = float(bbox_rot[2] * bbox_rot[3])
        new_anns.append(a2)

    # global brightness/contrast
    if np.random.rand() < 0.7:
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)
        beta  = np.random.uniform(-0.1, 0.1) * 255
        img_f = img_rot.astype(np.float32) * alpha + beta
        img_rot = np.clip(img_f, 0, 255).astype(np.uint8)

    return img_rot, new_anns, new_W, new_H
def compute_targets(coco, min_per_class=None, balance_factor=0.2):
    """
    Compute current counts and target count per class.
    - If min_per_class is given, target[cid] = max(current, min_per_class).
    - Else target[cid] = balance_factor * max_count.
    """
    counts = Counter(a["category_id"] for a in coco["annotations"])
    max_count = max(counts.values())

    if min_per_class is not None:
        base = min_per_class
    else:
        base = int(balance_factor * max_count)

    target = {cid: max(base, counts[cid]) for cid in counts}
    return counts, target

def augment_by_patching_targeted(
    coco,
    images_dir,
    out_images_dir,
    min_per_class=300,
    max_bboxes_per_image=10,
    patches_per_image=6,
    max_patches_per_cat=200,
    iou_threshold=0.2,
    min_scale=0.5,
    max_scale=1.5,
    max_rounds=3,
    out_json_path=None,
):
    os.makedirs(out_images_dir, exist_ok=True)

    # 1) counts & fixed target
    cat_counts, target = compute_targets(coco, min_per_class=min_per_class)
    print("Initial class counts:", dict(cat_counts))
    print("Target per class (first few):", {k: target[k] for k in list(target)[:5]})

    # classes that currently need more examples
    under_cats = {cid for cid, c in cat_counts.items() if c < target[cid]}
    if not under_cats:
        print("No under-represented classes under the chosen target.")
        return coco

    # 2) collect patches for those classes
    patches_by_cat = collect_patches(
        coco, images_dir, under_cats, max_patches_per_cat
    )
    # keep only classes for which we actually have patches
    under_cats = {cid for cid in under_cats if len(patches_by_cat.get(cid, [])) > 0}
    if not under_cats:
        print("No patches available for under-represented classes.")
        return coco

    # 3) build mapping image_id -> annotations, and find backgrounds
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images = {img["id"]: img for img in coco["images"]}

    bg_image_ids = [
        img_id for img_id, anns in anns_by_image.items()
        if len(anns) < max_bboxes_per_image
    ]
    print(f"Found {len(bg_image_ids)} background images (< {max_bboxes_per_image} boxes).")

    next_image_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id   = max(a["id"] for a in coco["annotations"]) + 1

    all_new_images = []
    all_new_anns   = []

    def all_reached_target():
        return all(cat_counts[cid] >= target[cid] for cid in under_cats)

    # 4) multiple rounds over backgrounds, until targets or rounds exhausted
    for round_idx in range(max_rounds):
        print(f"\n=== Round {round_idx+1}/{max_rounds} ===")
        if all_reached_target():
            print("All under-represented classes reached target.")
            break

        for img_id in bg_image_ids:
            if all_reached_target():
                break

            img_info = images[img_id]
            img_path = os.path.join(images_dir, img_info["file_name"])
            img = cv2.imread(img_path)
            if img is None:
                continue

            anns = anns_by_image[img_id]

            # global transform (rotation + mild global brightness/contrast)
            aug_img, aug_anns, W_aug, H_aug = transform_background(img, anns)

            # start from transformed boxes/labels
            bboxes = [a["bbox"] for a in aug_anns]
            cats   = [a["category_id"] for a in aug_anns]

            added = 0

            for _ in range(patches_per_image):
                # choose among classes that still need examples
                needy_cats = [cid for cid in under_cats if cat_counts[cid] < target[cid]]
                if not needy_cats:
                    break

                cid = int(np.random.choice(needy_cats))
                src_patches = patches_by_cat.get(cid, [])
                if not src_patches:
                    continue

                patch = src_patches[np.random.randint(len(src_patches))]
                patch_aug = augment_patch_rect(patch, min_scale=min_scale, max_scale=max_scale)

                placed_img, new_bbox = place_patch_on_image(
                    aug_img, patch_aug, bboxes, max_tries=50, max_iou=iou_threshold
                )
                if placed_img is None:
                    continue

                aug_img = placed_img
                bboxes.append(new_bbox)
                cats.append(cid)
                cat_counts[cid] += 1
                added += 1

            # also count the (possibly rotated) original objects in this new image
            if added > 0:
                for cid in cats:
                    cat_counts[cid] += 0  # already counted originals? you can +1 if you want to count them too

                base_name, ext = os.path.splitext(os.path.basename(img_info["file_name"]))
                new_fname = f"{base_name}_round{round_idx}_bg{img_id}_patched{ext}"
                out_path  = os.path.join(out_images_dir, new_fname)
                cv2.imwrite(out_path, aug_img)

                new_img = {
                    "id": next_image_id,
                    "file_name": os.path.relpath(out_path, images_dir),
                    "width":  W_aug,
                    "height": H_aug,
                }
                all_new_images.append(new_img)

                # create annotations for all boxes (existing + patches) for this new image
                for bbox, cid in zip(bboxes, cats):
                    new_ann = {
                        "id": int(next_ann_id),
                        "image_id": int(next_image_id),
                        "category_id": int(cid),
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "iscrowd": 0,
                    }
                    all_new_anns.append(new_ann)
                    next_ann_id += 1

                next_image_id += 1

        print("Class counts after round:", {cid: cat_counts[cid] for cid in under_cats})

    # 5) update COCO and optionally save
    coco["images"].extend(all_new_images)
    coco["annotations"].extend(all_new_anns)

    print(f"\nCreated {len(all_new_images)} new images and {len(all_new_anns)} new annotations.")
    print("Final counts for under-represented:", {cid: cat_counts[cid] for cid in under_cats})

    if out_json_path is not None:
        with open(out_json_path, "w") as f:
            json.dump(coco, f)
        print("Saved augmented COCO to:", out_json_path)

    return coco


images_dir     = train_dir
out_images_dir = "./images_patched"

coco_aug = augment_by_patching_targeted(
    coco,
    images_dir=images_dir,
    out_images_dir=out_images_dir,
    min_per_class=1000,          # target examples per class
    max_bboxes_per_image=10,    # 'few objects' threshold for backgrounds
    patches_per_image=7,        # how many patches to try per background
    max_patches_per_cat=200,
    iou_threshold=0.1,
    min_scale=0.5,
    max_scale=1.5,
    max_rounds=10,
    out_json_path="annotations_train_augmented.json",
)
