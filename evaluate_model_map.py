import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


import json
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        # Map image_id -> list of anns
        self.anns_by_img = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.anns_by_img.setdefault(img_id, []).append(ann)

        # map category ids to 0..N-1
        cat_ids = [c["id"] for c in self.categories]
        cat_ids_sorted = sorted(cat_ids)
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids_sorted)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.anns_by_img.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2idx[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def get_transform(train=True):
    transforms = []
    transforms.append(F.to_tensor)

    def apply(img):
        t = img
        for tf in transforms:
            t = tf(t)
        return t

    return apply


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint_path, num_classes, device):
    # same backbone as during training
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def evaluate_map(model, dataloader, device):
    metric = MeanAveragePrecision()  # mAP@0.5:0.95 + others

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            # move images to device
            images = [img.to(device) for img in images]

            # forward pass
            outputs = model(images)

            # prepare inputs for TorchMetrics (CPU tensors, only boxes/labels/scores)
            preds_cpu = []
            targets_cpu = []
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ⚠️ paths must match your training script
    val_root = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/valid"
    val_ann = os.path.join(val_root, "_annotations.coco.json")

    val_dataset = COCODataset(
        img_dir=val_root,
        ann_file=val_ann,
        transforms=get_transform(train=False),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # same num_classes logic as in training
    num_classes = len(val_dataset.categories) + 1

    checkpoint_path = "./fasterrcnn_cadot.pth"  # your saved model
    model = load_model(checkpoint_path, num_classes, device)

    metrics = evaluate_map(model, val_loader, device)

    print("===== mAP results on validation set =====")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
            # scalar tensor → safe to convert to float
                print(f"{k}: {v.item():.4f}")
            else:
            # vector tensor (per-class metrics, etc.)
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")
