import os
import glob
from collections import Counter

def inspect(root):
    lbl_dir = os.path.join(root, "labels")
    counter = Counter()

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(lbl_dir, split)
        if not os.path.exists(split_dir):
            continue
        for f in glob.glob(os.path.join(split_dir, "*.txt")):
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    cls = int(float(line.split()[0]))
                    counter[cls] += 1

    print("Class ID distribution:")
    for k in sorted(counter):
        print(f"  class {k}: {counter[k]}")

    print("\nTotal classes used:", len(counter))
    print("Max class id:", max(counter))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    inspect(args.root)
