import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDetrDataset(Dataset):
    """
    COCO-format dataset for DETR (HuggingFace).

    """

    def __init__(self, image_dir, annotation_file, processor):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        target = {
            "image_id": image_id,
            "annotations": anns
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        encoding["pixel_values"] = encoding["pixel_values"].squeeze(0)
        encoding["pixel_mask"]   = encoding["pixel_mask"].squeeze(0)

        encoding["labels"] = encoding["labels"][0]

        return encoding



