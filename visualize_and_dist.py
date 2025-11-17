import json
import os
import cv2
import matplotlib.pyplot as plt



def draw_coco_annotations(json, image_filename, images_dir,):
    image_info = next((img for img in json["images"] if img["file_name"] == image_filename), None)
    if image_info is None:
        raise ValueError(f"Image '{image_filename}' not found in JSON annotations.")

    image_id = image_info["id"]
    image_path = os.path.join(images_dir, image_filename)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    anns = [a for a in json["annotations"] if a["image_id"] == image_id]
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cat_id = ann["category_id"]
        cat_name = next((c["name"] for c in json["categories"] if c["id"] == cat_id), "unknown")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cat_name, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img



def show_images_grid(images, cols=5):
    n = len(images)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, img in enumerate(images[:10]):  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()

from collections import Counter
import matplotlib.pyplot as plt

def annotation_distribution(json,plot=False):

    id_to_name = {c["id"]: c["name"] for c in json["categories"]}
    category_counts = Counter([ann["category_id"] for ann in json["annotations"]])

    category_names = [id_to_name[cid] for cid in category_counts.keys()]
    counts = [category_counts[cid] for cid in category_counts.keys()]
    for name, count in zip(category_names, counts):
        print(f"{name:20s}: {count}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.bar(category_names, counts)
        plt.xlabel("Category")
        plt.ylabel("Number of Annotations")
        plt.title("Distribution of Annotations per Category")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return dict(zip(category_names, counts))


current_wd = os.getcwd()
train_dir=os.path.join(current_wd,"train\\")
annotations_dir = os.path.join(current_wd,"annotations_train_augmented.json")
with open(annotations_dir, "r") as f:
    annotations=json.load(f)
images=[]
n_images=0
for file_name in os.listdir(train_dir):
    if n_images>9:
        break
    if file_name.endswith(".json"):
        continue
    n_images+=1
    img_path = os.path.join(train_dir,file_name)
    image=draw_coco_annotations(annotations,file_name,train_dir)
    images.append(image)
#show_images_grid(images)
annotation_distribution(annotations,plot=True)
print(images[0].max())