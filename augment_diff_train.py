import json, os, math
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
import torch.nn as nn

curr_dir=os.getcwd()
train_dir=os.path.join(curr_dir,"data_coco/train/")
COCO_JSON=os.path.join(train_dir,"_annotations.coco.json")
IMAGES_DIR = train_dir
NUM_CLASSES = 14
IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
MODEL_OUT = os.path.join(curr_dir,"bbox_guided_ddpm.pth")
DEVICE = "cuda"



#LOAD COCO + BUILD BOX CONDITION MAPS

class BBoxGuidedDataset(Dataset):
    def __init__(self, coco_json, images_dir):
        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images = {i["id"]: i for i in coco["images"]}
        self.anns = coco["annotations"]
        self.images_dir = images_dir

        # group bboxes by image
        self.bboxes_by_img = {}
        for ann in self.anns:
            iid = ann["image_id"]
            if iid not in self.bboxes_by_img:
                self.bboxes_by_img[iid] = []
            self.bboxes_by_img[iid].append(ann)

        self.ids = list(self.images.keys())

        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        info = self.images[iid]
        path = os.path.join(self.images_dir, info["file_name"])

        img = Image.open(path).convert("RGB")
        img = self.tf(img)                        

        # build  mask: NUM_CLASSES channels
        mask= torch.zeros(NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

        for ann in self.bboxes_by_img.get(iid, []):
            cid = ann["category_id"]-1
            x, y, w, h = ann["bbox"]
            # scale coords to 256×256
            sx = x / info["width"]  * IMAGE_SIZE
            sy = y / info["height"] * IMAGE_SIZE
            sw = w / info["width"]  * IMAGE_SIZE
            sh = h / info["height"] * IMAGE_SIZE

            x1, y1 = int(sx), int(sy)
            x2, y2 = int(sx+sw), int(sy+sh)
            mask[cid, y1:y2, x1:x2] = 1.0

        # blur box edges so diffusion learns soft shapes
        mask = torch.nn.functional.avg_pool2d(mask.unsqueeze(0), 25, stride=1, padding=12)[0]

        return img, mask

# DIFFUSION MODEL

def build_unet():
    return UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3 + NUM_CLASSES,       # RGB + box mask channels
        out_channels=3,
        layers_per_block=2,
        block_out_channels=[32, 64, 64, 128],
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    )



#TRAINING LOOP

def train():
    ds = BBoxGuidedDataset(COCO_JSON, IMAGES_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    unet = build_unet().to(DEVICE)
    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    sched = DDPMScheduler(num_train_timesteps=1000)

    for epoch in range(EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for img, cond in pbar:
            img, cond = img.to(DEVICE), cond.to(DEVICE)
            B = img.shape[0]

            # random diffusion timestep
            t = torch.randint(0, sched.num_train_timesteps, (B,), device=DEVICE)

            noise = torch.randn_like(img)
            noisy = sched.add_noise(img, noise, t)

            # concatenate condition channels
            noisy_in = torch.cat([noisy, cond], dim=1)

            pred = unet(noisy_in, t).sample
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss))

    torch.save(unet.state_dict(), MODEL_OUT)


if __name__ == "__main__":
    train()
