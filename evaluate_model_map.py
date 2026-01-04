import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image


# -----------------------------
# Dataset (COCO JSON)
# IMPORTANT: labels are 1..N (0 is background)
# -----------------------------
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco.get("annotations", [])
        self.categories = coco["categories"]

        # image_id -> list of anns
        self.anns_by_img = {}
        for ann in self.annotations:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # category_id -> contiguous label in 1..N (0 reserved for background)
        cat_ids_sorted = sorted([c["id"] for c in self.categories])
        self.catid2label = {cid: (i + 1) for i, cid in enumerate(cat_ids_sorted)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}
        self.catid2name = {c["id"]: c["name"] for c in self.categories}

        # class names in contiguous label order
        self.class_names = ["__bg__"] + [self.catid2name[self.label2catid[i]] for i in range(1, len(cat_ids_sorted) + 1)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.anns_by_img.get(img_id, [])

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2label[ann["category_id"]])  # 1..N
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(img_id, dtype=torch.int64),  # scalar
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def get_transform():
    return F.to_tensor


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Model loading
# -----------------------------
def build_model(num_classes: int):
    # Use weights=None for backbone if you trained from scratch;
    # If you trained starting from COCO weights, you can keep COCO_V1.
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device,weights_only=True)

    # support common checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip DDP prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


# -----------------------------
# mAP evaluation (TorchMetrics)
# -----------------------------
@torch.no_grad()
def evaluate_map(model, dataloader, device, class_metrics=True):
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
    )

    model.eval()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds_cpu, targets_cpu = [], []
        for out, tgt in zip(outputs, targets):
            preds_cpu.append({
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            })
            targets_cpu.append({
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu(),
            })

        metric.update(preds_cpu, targets_cpu)

    return metric.compute()


# -----------------------------
# Visualization (2 separate figures): GT and Pred
# -----------------------------
def show_boxes(title, img_pil, boxes, labels, class_names, scores=None, score_thr=0.0, linewidth=2):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
        text = name
        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        ax.text(
            x1, max(0, y1 - 5),
            text,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )

    plt.show()



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- paths (EDIT THESE) ----
    val_root = "./valid"  # set to your images root used by file_name in JSON
    val_ann = os.path.join(val_root, "_annotations.coco.json")
    checkpoint_path = "./fasterrcnn_cadot.pth"

    # ---- data ----
    val_dataset = COCODataset(
        img_dir=val_root,
        ann_file=val_ann,
        transforms=get_transform(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- model ----
    num_classes = len(val_dataset.class_names)  # includes background
    print("num_classes (incl bg):", num_classes)

    model = build_model(num_classes=num_classes)
    model = load_checkpoint(model, checkpoint_path, device)


    # ---- evaluate mAP ----
    metrics = evaluate_map(model, val_loader, device, class_metrics=True)
    print("\n===== mAP results on validation set =====")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
            else:
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

