# train_fasterrcnn_rare_aug.py
# Full Faster R-CNN training script (COCO JSON) with:
# - torchvision.transforms.v2 (boxes updated correctly)
# - stronger augmentation only for images containing rare classes
# - optional Repeat-Factor Sampler for imbalance
# - correct val loss computation (train mode + no_grad)

import os
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    train_images_dir: str = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/train"
    train_ann_json: str = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/train/_annotations.coco.json"

    val_images_dir: str = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/valid"
    val_ann_json: str = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/valid/_annotations.coco.json"

    num_workers: int = 4
    batch_size: int = 2
    epochs: int = 10

    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    step_size: int = 5
    gamma: float = 0.1

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # rare classes definition
    rare_min_instances: int = 200  # boxes count threshold; adjust to your dataset
    rare_aug_prob: float = 1.0     # apply rare transforms with this prob when rare class exists

    # imbalance sampler
    use_repeat_factor_sampler: bool = False
    repeat_factor_threshold: float = 0.01  # smaller => more oversampling of rare classes

    # model
    weights: str = "DEFAULT"  # torchvision>=0.13 ; if it errors, set None and use weights_backbone=None


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Detection-safe transforms (v2)
# -----------------------------
def make_base_train_tf():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomShortestSize(min_size=[512, 544, 576, 608, 640], max_size=1024),
    ])


def make_rare_train_tf():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.7),
        T.RandomShortestSize(min_size=[480, 512, 544, 576, 608, 640, 704], max_size=1024),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.15),
    ])


def make_val_tf():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomShortestSize(min_size=640, max_size=1024),
    ])


# -----------------------------
# COCO dataset -> Faster R-CNN target dict
# + rare-class conditional augmentation
# -----------------------------
class CocoDetectionRareAug(torch.utils.data.Dataset):
    """
    Returns:
      img: FloatTensor[C,H,W] in [0,1]
      target: dict with boxes (xyxy), labels in [1..K], image_id, area, iscrowd
    """
    def __init__(self, img_dir, ann_json, base_tf=None, rare_tf=None, rare_classes=None, rare_aug_prob=1.0):
        self.coco_ds = CocoDetection(img_dir, ann_json)
        self.coco = self.coco_ds.coco

        # Build mapping from the JSON categories (same as your second code)
        import json as _json
        with open(ann_json, "r") as f:
            coco_json = _json.load(f)

        cat_ids = sorted([c["id"] for c in coco_json["categories"]])

        # OPTIONAL but often needed: if your JSON contains category_id=0 as "background", drop it
        cat_ids = [cid for cid in cat_ids if cid != 0]

        self.catid2label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}  # <-- key for evaluation

        self.base_tf = base_tf
        self.rare_tf = rare_tf
        self.rare_classes = set(rare_classes or [])
        self.rare_aug_prob = float(rare_aug_prob)

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img, anns = self.coco_ds[idx]
        image_id = self.coco_ds.ids[idx]

        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO xywh
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.catid2label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        has_rare = any(int(c) in self.rare_classes for c in labels.tolist())

        if has_rare and self.rare_tf is not None and random.random() < self.rare_aug_prob:
            img, target = self.rare_tf(img, target)
        elif self.base_tf is not None:
            img, target = self.base_tf(img, target)

        return img, target


# -----------------------------
# Rare classes from counts
# -----------------------------
def compute_class_counts(dataset):
    counts = Counter()
    for i in range(len(dataset)):
        _, t = dataset[i]
        counts.update(t["labels"].tolist())
    counts.pop(0, None)
    return counts


def get_rare_classes(counts, min_instances):
    return {c for c, n in counts.items() if n < min_instances}



class RepeatFactorSampler(Sampler[int]):
    """
    Oversamples images containing rare classes.
    r(c)=max(1, sqrt(t/f(c))) where f(c)=fraction of images containing class c
    r(i)=max_c in image r(c)
    """
    def __init__(self, dataset: CocoDetectionRareAug, threshold: float = 0.01, shuffle: bool = True):
        self.dataset = dataset
        self.threshold = float(threshold)
        self.shuffle = shuffle

        img_to_classes = []
        class_img_count = defaultdict(int)

        for i in range(len(dataset)):
            _, t = dataset[i]
            classes = set(t["labels"].tolist())
            classes.discard(0)
            img_to_classes.append(classes)
            for c in classes:
                class_img_count[c] += 1

        nimg = len(dataset)
        class_freq = {c: class_img_count[c] / max(1, nimg) for c in class_img_count.keys()}
        class_repeat = {c: max(1.0, math.sqrt(self.threshold / max(f, 1e-12))) for c, f in class_freq.items()}

        # per-image repeat factor
        img_repeat = []
        for classes in img_to_classes:
            if not classes:
                img_repeat.append(1.0)
            else:
                img_repeat.append(max(class_repeat.get(c, 1.0) for c in classes))

        # stochastic rounding
        indices = []
        for idx, r in enumerate(img_repeat):
            k = int(math.floor(r))
            prob = r - k
            indices.extend([idx] * k)
            if random.random() < prob:
                indices.append(idx)

        self.indices = indices if indices else list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_model(num_classes_including_background: int, weights: str = "DEFAULT"):
    # num_classes_including_background = K + 1
    # we set predictor to output that many classes
    if weights is None:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_including_background)
    return model



def train_one_epoch(model, loader, optimizer, scaler, device, epoch, print_every=50):
    model.train()
    running = 0.0
    seen = 0

    for it, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip fully-empty batches (common in some datasets)
        if any(t["boxes"].numel() == 0 for t in targets):
            continue

        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            loss_dict = model(images, targets)  # dict in train mode
            loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        seen += 1

        if print_every and it % print_every == 0 and seen > 0:
            print(f"Epoch {epoch} | iter {it}/{len(loader)} | loss {running/seen:.4f}")

    return running / max(seen, 1)


@torch.no_grad()
def compute_val_loss(model, loader, device):
    # IMPORTANT: losses are only returned in train() mode for torchvision detection
    model.train()
    total = 0.0
    n = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if any(t["boxes"].numel() == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        total += loss
        n += 1

    model.eval()
    return total / max(n, 1)


def main():
    cfg = CFG()
    set_seed(cfg.seed)

    print("Device:", cfg.device)

    # Temporary dataset without transforms to count instances
    tmp_ds = CocoDetectionRareAug(
        cfg.train_images_dir, cfg.train_ann_json,
        base_tf=None, rare_tf=None
    )
    counts = compute_class_counts(tmp_ds)
    rare_classes = get_rare_classes(counts, cfg.rare_min_instances)

    print(f"Found {len(counts)} classes (labels 1..K).")
    print(f"Rare classes (label ids) (<{cfg.rare_min_instances} instances): {sorted(rare_classes)}")

    # Actual datasets with conditional augmentation
    train_ds = CocoDetectionRareAug(
        cfg.train_images_dir, cfg.train_ann_json,
        base_tf=make_base_train_tf(),
        rare_tf=make_rare_train_tf(),
        rare_classes=rare_classes,
        rare_aug_prob=cfg.rare_aug_prob
    )
    val_ds = CocoDetectionRareAug(
        cfg.val_images_dir, cfg.val_ann_json,
        base_tf=make_val_tf(),
        rare_tf=None
    )

    # num classes including background
    # Because labels start at 1..K, K = number of categories in COCO file
    K = len(train_ds.catid2label)
    num_classes_including_background = K + 1

    # Dataloaders
    if cfg.use_repeat_factor_sampler:
        sampler = RepeatFactorSampler(train_ds, threshold=cfg.repeat_factor_threshold, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True
        )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = build_model(num_classes_including_background, weights=cfg.weights).to(cfg.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.device.startswith("cuda"))


    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, cfg.device, epoch, print_every=50)
        scheduler.step()

        val_loss = compute_val_loss(model, val_loader, cfg.device)

        print(f"Epoch [{epoch}/{cfg.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__},
                "checkpoints/fasterrcnn_best.pt"
            )
            print("Saved best checkpoint: checkpoints/fasterrcnn_best.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "checkpoints/fasterrcnn_last.pt")
    print("Training done. Saved: checkpoints/fasterrcnn_last.pt")


if __name__ == "__main__":
    main()
