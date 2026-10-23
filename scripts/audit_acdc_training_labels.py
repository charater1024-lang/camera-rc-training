"""Count ACDC TRAIN labels only, before any sampling or model selection.

Counts are pixels and frames containing labels, never object-instance recall.
No predictions, validation or test labels are used to select training examples.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_acdc import CONDITIONS, CITYSCAPES_VALID, remap_labels

NAMES = {7: "road", 8: "sidewalk", 11: "building", 12: "wall", 13: "fence",
         17: "pole", 19: "traffic_light", 20: "traffic_sign", 21: "vegetation",
         22: "terrain", 23: "sky", 24: "person", 25: "rider", 26: "car",
         27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}


def count_one(path):
    raw_bytes = path.read_bytes()
    with Image.open(path) as image:
        values = np.asarray(image).copy()
    inv_path = path.with_name(path.name.replace("_labelIds", "_invIds"))
    inv_bytes = inv_path.read_bytes()
    with Image.open(inv_path) as image:
        inv = np.asarray(image).copy()
    remap_labels(values, inv)  # strict encoding/shape validation
    ids, counts = np.unique(values, return_counts=True)
    return {"name": path.name, "label_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "uncertainty_sha256": hashlib.sha256(inv_bytes).hexdigest()}, dict(zip(map(int, ids), map(int, counts)))


def audit(gt_root):
    gt_root = Path(gt_root).resolve(strict=True)
    result = {"dataset": "ACDC", "scope": "official training annotations only",
              "rgb_available_verified": False, "new_training_performed": False,
              "instance_metrics": False, "conditions": {}}
    for condition in CONDITIONS:
        paths = sorted((gt_root / condition / "train").glob("*/*_gt_labelIds.png"))
        if len(paths) != 400:
            raise ValueError(f"Expected 400 train frames for {condition}")
        pixels, frames, proofs = Counter(), Counter(), []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for proof, counts in pool.map(count_one, paths):
                pixels.update(counts)
                frames.update(counts.keys())
                proofs.append(proof)
        total = sum(pixels.values())
        valid = sum(pixels[k] for k in CITYSCAPES_VALID)
        result["conditions"][condition] = {"frames": len(paths), "total_pixels": total,
            "ignored_pixel_fraction": (total - valid) / total,
            "classes": {NAMES[k]: {"label_id": k, "frames": frames[k], "pixels": pixels[k]} for k in sorted(CITYSCAPES_VALID)},
            "annotation_hashes": proofs}
        print(json.dumps({"condition": condition, "frames": len(paths), "ignored_fraction": (total-valid)/total}), flush=True)
    result["complete"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous audit")
    result = audit(args.gt_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
