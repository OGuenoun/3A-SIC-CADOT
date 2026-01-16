import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    get_cosine_schedule_with_warmup
)
from dataset import CocoDetrDataset
from collate import collate_fn
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import WeightedRandomSampler
import pickle



# ---------------- CONFIG ----------------
TRAIN_JSON = "data/train/train_split.json"
VAL_JSON   = "data/train/val_split.json"
IMG_DIR    = "data/train/images/"

NUM_CLASSES = 15          # IMPORTANT (includes unused class 0)
BATCH_SIZE  = 2
EPOCHS      = 100
LR          = 1e-4
NUM_WORKERS = 4
OUT_DIR     = "DETR_chekpoints"
# ----------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()

###########################

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        pixel_values = torch.stack(batch["pixel_values"]).to(device)
        pixel_mask   = torch.stack(batch["pixel_mask"]).to(device)

        labels = [
            {k: v.to(device) for k, v in target.items()}
            for target in batch["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        pixel_values = torch.stack(batch["pixel_values"]).to(device)
        pixel_mask   = torch.stack(batch["pixel_mask"]).to(device)

        labels = [
            {k: v.to(device) for k, v in target.items()}
            for target in batch["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        total_loss += outputs.loss.item()

    return total_loss / len(loader)


def main():
    distributed = "RANK" in os.environ

    if distributed:
        local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    if rank == 0:
        print("Using device:", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    model.config.class_cost = 2.0


    if distributed:
        model = DDP(model, device_ids=[device.index])

    train_ds = CocoDetrDataset(IMG_DIR, TRAIN_JSON, processor)
    with open("repeat_factors_augm.pkl", "rb") as f:
        repeat_factors = pickle.load(f)

    weights = [
        repeat_factors[img_id]
        for img_id in train_ds.image_ids
    ]

    val_ds   = CocoDetrDataset(IMG_DIR, VAL_JSON, processor)

    #train_sampler = DistributedSampler(train_ds) if distributed else None
    ## RFS activated:
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    val_sampler   = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},  # backbone
            {"params": other_params, "lr": 1e-4},     # transformer + heads
        ],
        weight_decay=1e-4
    )


    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * EPOCHS
    )

    best_val = float("inf")

    for epoch in range(EPOCHS):
        print(f"Rank {rank} on device {device}")

        #if distributed:
            #train_sampler.set_epoch(epoch)
        if distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)


        if rank == 0:
            print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)

        lr_scheduler.step()

        if rank == 0:
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss:   {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                model.module.save_pretrained(OUT_DIR) if distributed else model.save_pretrained(OUT_DIR)
                processor.save_pretrained(OUT_DIR)
                print("✓ Saved best model")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()



