import os, glob, random, shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def images_in(dir):
    imgs = []
    for e in IMG_EXTS:
        imgs += glob.glob(os.path.join(dir, f"*{e}"))
    return sorted(imgs)

def copy_pair(img, src_lbl, dst_img, dst_lbl):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    base = os.path.splitext(os.path.basename(img))[0]
    shutil.copy2(img, os.path.join(dst_img, os.path.basename(img)))

    src = os.path.join(src_lbl, base + ".txt")
    dst = os.path.join(dst_lbl, base + ".txt")
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        open(dst, "w").close()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    train_imgs = images_in(os.path.join(args.src, "images/train"))
    random.shuffle(train_imgs)

    n_val = int(len(train_imgs) * args.val_ratio)
    val_imgs = train_imgs[:n_val]
    trn_imgs = train_imgs[n_val:]

    for p in trn_imgs:
        copy_pair(p,
                  os.path.join(args.src, "labels/train"),
                  os.path.join(args.dst, "images/train"),
                  os.path.join(args.dst, "labels/train"))

    for p in val_imgs:
        copy_pair(p,
                  os.path.join(args.src, "labels/train"),
                  os.path.join(args.dst, "images/val"),
                  os.path.join(args.dst, "labels/val"))

    for p in images_in(os.path.join(args.src, "images/valid")):
        copy_pair(p,
                  os.path.join(args.src, "labels/valid"),
                  os.path.join(args.dst, "images/test"),
                  os.path.join(args.dst, "labels/test"))

    print("Split completed.")

if __name__ == "__main__":
    main()
