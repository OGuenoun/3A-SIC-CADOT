import os, json, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import Counter
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler

from augment_diff_train import build_unet, IMAGE_SIZE, NUM_CLASSES

MODEL_PATH = "bbox_guided_ddpm.pth"
curr_dir=os.getcwd()
train_dir=os.path.join(curr_dir,"train\\")
COCO_IN=os.path.join(train_dir,"_annotations.coco.json")
OUT_JSON  = "annotations_augmented.json"
OUT_DIR   = "generated_images"

NUM_SAMPLES_PER_BBOX = 10    
DEVICE="cuda"
unet = build_unet()
unet.load_state_dict(torch.load(MODEL_PATH))
unet.to(DEVICE).eval()

sched = DDIMScheduler(num_train_timesteps=1000)
sched.set_timesteps(50) 

with open(COCO_IN,"r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

anns_by_img = {}
for ann in coco["annotations"]:
    anns_by_img.setdefault(ann["image_id"], []).append(ann)

os.makedirs(OUT_DIR, exist_ok=True)

next_img_id = max(images.keys())+1
next_ann_id = max(a["id"] for a in coco["annotations"])+1

cat_counts = Counter(a["category_id"] for a in coco["annotations"])
max_count = max(cat_counts.values())
target = {cid: max_count*0.2 for cid in cat_counts}
missing = {cid: max(0, target[cid] - cat_counts[cid]) for cid in cat_counts}
under_cats = {cid for cid, m in missing.items() if m > 0}


# GENERATE  IMAGES

for img_id, img_info in tqdm(images.items(),desc="Generating for underrepresented classes"):

    for ann in anns_by_img.get(img_id, []):
        cid = ann["category_id"]
        if cid not in under_cats:
                continue
        if missing.get(cid, 0) <= 0:
                continue
        x,y,w,h = ann["bbox"]
        
        cond = torch.zeros(NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
        sx = int(x / img_info["width"]  * IMAGE_SIZE)
        sy = int(y / img_info["height"] * IMAGE_SIZE)
        sw = int(w / img_info["width"]  * IMAGE_SIZE)
        sh = int(h / img_info["height"] * IMAGE_SIZE)
        cond[cid, sy:sy+sh, sx:sx+sw] = 1.0
        cond = torch.nn.functional.avg_pool2d(cond.unsqueeze(0),25,stride=1,padding=12)[0]
        cond = cond.unsqueeze(0).to(DEVICE)

        # generate N samples for this bbox
        for _ in range(NUM_SAMPLES_PER_BBOX):
            x_t = torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)

            for t in reversed(range(1000)):
                tt = torch.tensor([t], device=DEVICE)
                x_in = torch.cat([x_t, cond], dim=1)
                eps = unet(x_in, tt).sample
                x_t = sched.step(eps, t, x_t).prev_sample

            img = (x_t.clamp(-1,1)+1)/2
            img = (img[0].mul(255).byte().cpu().permute(1,2,0).numpy())

            fname = f"gen_{next_img_id}.jpg"
            Image.fromarray(img).save(os.path.join(OUT_DIR,fname))

            # build new COCO entries
            new_img = {
                "id": next_img_id,
                "file_name": os.path.join(OUT_DIR,fname),
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE
            }
            new_ann = {
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": cid,
                "bbox": [sx,sy,sw,sh],
                "area": int(sw*sh),
                "iscrowd": 0
            }

            coco["images"].append(new_img)
            coco["annotations"].append(new_ann)

            next_img_id += 1
            next_ann_id += 1


with open(OUT_JSON,"w") as f:
    json.dump(coco,f)

print("Saved:", OUT_JSON)
print("Generated images:", OUT_DIR)