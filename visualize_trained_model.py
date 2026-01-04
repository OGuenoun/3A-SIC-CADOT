import json
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -----------------------------
# COCO helpers
# -----------------------------
def load_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    # contiguous labels: background=0, categories -> 1..N sorted by category_id
    cat_ids_sorted = [c["id"] for c in cats]
    cat_id_to_label = {cat_id: (i + 1) for i, cat_id in enumerate(cat_ids_sorted)}
    label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}

    file_to_img = {img["file_name"]: img for img in coco["images"]}

    ann_by_image_id = {}
    for ann in coco.get("annotations", []):
        ann_by_image_id.setdefault(ann["image_id"], []).append(ann)

    class_names = ["__bg__"] + [cat_id_to_name[label_to_cat_id[i]] for i in range(1, len(label_to_cat_id) + 1)]
    return coco, file_to_img, ann_by_image_id, cat_id_to_label, class_names


def get_gt_for_image(image_path, coco_json_path):
    _, file_to_img, ann_by_image_id, cat_id_to_label, class_names = load_coco(coco_json_path)

    fname = Path(image_path).name
    if fname not in file_to_img:
        raise ValueError(f"Image '{fname}' not found in COCO json 'images' list.")

    img_info = file_to_img[fname]
    anns = ann_by_image_id.get(img_info["id"], [])

    gt_boxes, gt_labels = [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(cat_id_to_label[ann["category_id"]])

    return gt_boxes, gt_labels, class_names


# -----------------------------
# Model helpers
# -----------------------------
def build_fasterrcnn(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = F.to_tensor(img).to(device)
    out = model([x])[0]
    return img, out


# -----------------------------
# Visualization (same figure: GT | Pred)
# -----------------------------
def draw_on_ax(ax, title, img_pil, boxes, labels, class_names, scores=None, score_thr=0.0, linewidth=2):
    ax.imshow(img_pil)
    ax.axis("off")
    ax.set_title(title)

    for i, box in enumerate(boxes):
        if scores is not None and float(scores[i]) < score_thr:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=linewidth)
        ax.add_patch(rect)

        lab = int(labels[i])
        name = class_names[lab] if lab < len(class_names) else f"class_{lab}"
        txt = name
        if scores is not None:
            txt += f" {float(scores[i]):.2f}"

        ax.text(
            x1, max(0, y1 - 5),
            txt,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )


def visualize_gt_and_pred_same_figure(image_path, coco_json_path, ckpt_path, pred_score_thr=0.5, linewidth=2):
    # Load GT
    gt_boxes, gt_labels, class_names = get_gt_for_image(image_path, coco_json_path)

    # Load model & predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    model = build_fasterrcnn(num_classes=num_classes)
    model = load_checkpoint(model, ckpt_path, device)

    img_pil, out = predict(model, image_path, device)
    pred_boxes = out["boxes"].detach().cpu()
    pred_labels = out["labels"].detach().cpu()
    pred_scores = out["scores"].detach().cpu()

    # Same figure: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    draw_on_ax(
        axes[0],
        "Ground Truth",
        img_pil,
        gt_boxes,
        gt_labels,
        class_names,
        scores=None,
        linewidth=linewidth,
    )

    draw_on_ax(
        axes[1],
        f"Prediction (score ≥ {pred_score_thr})",
        img_pil,
        pred_boxes,
        pred_labels,
        class_names,
        scores=pred_scores,
        score_thr=pred_score_thr,
        linewidth=linewidth,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMAGE_PATH = "./valid/75-2021-0655-6860-LA93-0M20-E080-510_jpeg_jpg.rf.fdfc9c3819c46f0edbfc12282a38e668.jpg"
    COCO_JSON  = "./valid/_annotations.coco.json"
    CKPT_PATH  = "./fasterrcnn_cadot.pth"

    visualize_gt_and_pred_same_figure(
        image_path=IMAGE_PATH,
        coco_json_path=COCO_JSON,
        ckpt_path=CKPT_PATH,
        pred_score_thr=0.5,
    )
