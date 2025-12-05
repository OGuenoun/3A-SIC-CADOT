import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as F


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
            boxes.append([x, y,x+w,x+h])
            labels.append(self.catid2idx[ann["category_id"]])  # +1, 0 reserved for background
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
    # Very simple transforms to start; you can add more later
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = "/bettik/PROJECTS/pr-material-acceleration/guenouno/data/data_augmented/train/"  # change if needed

    train_dataset = COCODataset(
        img_dir=data_root,
        ann_file=os.path.join(data_root, "annotations_train_augmented.json"),
        transforms=get_transform(train=True),
    )
    val_root="/bettik/PROJECTS/pr-material-acceleration/guenouno/data/valid"
    val_dataset = COCODataset(
        img_dir=val_root,
        ann_file=os.path.join(val_root, "_annotations.coco.json"),
        transforms=get_transform(train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    num_classes = len(train_dataset.categories) + 1  # + background

    # Load Faster R-CNN pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    # Replace head for CADOT num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

	batches = 0
        
	for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value
	    batches+=1
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
	avg_train_loss=train_loss/max(batches,1)
        lr_scheduler.step()
	val_loss = 0.0
        val_batches = 0
        # Simple validation loop: just forward & maybe later compute metrics
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
		val_loss += losses.item()
                val_batches += 1
                # TODO: compute mAP
	avg_val_loss = val_loss / max(val_batches, 1)
    # Save the trained model weights
	print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
    torch.save(model.state_dict(), "fasterrcnn_cadot.pth")
    print("Training finished, model saved as fasterrcnn_cadot.pth")


if __name__ == "__main__":
    main()
