import json
from pathlib import Path

# CONFIG

DATA_ROOT = Path("data/")
ANN_FILE = DATA_ROOT / "train/val_split.json"
IMG_DIR = DATA_ROOT / "train/images"

# LOAD ANNOTATIONS

with open(ANN_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]

print(f"Total images in annotation file: {len(images)}")
print(f"Image directory: {IMG_DIR.resolve()}")

# CHECK EXISTENCE

missing = []
present = 0

for img in images:
    img_path = IMG_DIR / img["file_name"]
    if img_path.exists():
        present += 1
    else:
        missing.append(img["file_name"])

# REPORT

print(f"Images present: {present}")
print(f"Images missing: {len(missing)}")

if missing:
    print("Missing images (first 20):")
    for m in missing[:20]:
        print("  -", m)
else:
    print("All annotated images exist on disk!")

# OPTIONAL: ORPHAN IMAGES

annotated_files = set(img["file_name"] for img in images)
disk_files = set(p.name for p in IMG_DIR.glob("*"))

orphans = disk_files - annotated_files

print("\nOrphan images on disk (not in annotations):", len(orphans))
if orphans:
    print("First 20 orphans:")
    for o in list(orphans)[:20]:
        print("  -", o)
