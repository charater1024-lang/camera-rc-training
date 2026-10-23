"""Prepare A2D2 road masks with date-level holdouts and optional resized RGB.

This creates independently trained camera images, not synchronized multi-view
samples. Original publisher files are preserved. Targets mean road surface,
not guaranteed free space, legal permission, or safe vehicle trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from download_a2d2 import BASE, PREFIX, check_inventory, emit, save_json, target_path

LABEL_SPACE = "road_surface_v1"
VAL_DATES = frozenset({"20181008"})
TEST_DATES = frozenset({"20181107"})
EXPECTED_SPLITS = {"train": 33330, "val": 4873, "test": 3074}
ROAD_NAMES = frozenset({"RD normal street", "Drivable cobblestone", "Slow drive area",
                        "Painted driv. instr.", "Solid line", "Dashed line", "Zebra crossing"})
# Parking/bump annotations do not specify whether this vehicle should traverse
# them; avoid teaching those ambiguous classes as either confidently free or blocked.
IGNORE_NAMES = frozenset({"Blurred area", "Rain dirt", "Ego car", "Parking area", "Speed bumper"})


def split_for_sequence(sequence):
    date = sequence.split("_", 1)[0]
    return "val" if date in VAL_DATES else "test" if date in TEST_DATES else "train"


def label_tables(classes, strict=True):
    if not isinstance(classes, dict) or strict and (len(classes) != 55 or not ROAD_NAMES | IGNORE_NAMES <= set(classes.values())):
        raise ValueError("Unexpected publisher semantic class table")
    lookup = np.full(1 << 24, 255, dtype=np.uint8)
    known = np.zeros(1 << 24, dtype=np.bool_)
    mapping = {}
    for color, name in classes.items():
        if len(color) != 7 or color[0] != "#":
            raise ValueError("Expected RGB hex semantic colors")
        try:
            packed = int(color[1:], 16)
        except ValueError as error:
            raise ValueError("Invalid semantic color") from error
        target = 0 if name in ROAD_NAMES else 255 if name in IGNORE_NAMES else 1
        lookup[packed], known[packed] = target, True
        mapping[color] = {"name": name, "target": target}
    # No class in the publisher's class table uses black. Treat black/unmapped
    # pixels as void; never infer background/nonroad from an unknown color.
    known[0] = True
    return lookup, known, mapping


def convert_mask(raw, lookup, known):
    values = np.asarray(raw)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("A2D2 labels must be RGB uint8 semantic colors")
    packed = values[:, :, 0].astype(np.uint32)
    packed <<= 16
    packed |= values[:, :, 1].astype(np.uint32) << 8
    packed |= values[:, :, 2].astype(np.uint32)
    target = lookup[packed]
    unknown = ~known[packed]
    unknown_colors, unknown_counts = np.unique(packed[unknown], return_counts=True)
    return target, {f"#{int(color):06x}": int(count) for color, count in zip(unknown_colors, unknown_counts)}


def checked_bytes(root, record):
    path = target_path(root, record["key"])
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing or wrong-size source: {record['key']}")
    content = path.read_bytes()
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != record["md5"]:
        raise ValueError(f"Publisher MD5 mismatch during preparation: {record['key']}")
    return content


def collect_pairs(inventory):
    objects = {obj["key"]: obj for obj in inventory["objects"]}
    pairs = []
    for key in sorted(objects):
        if "/camera/" not in key or not key.endswith(".png"):
            continue
        label_key = key.replace("/camera/", "/label/").replace("_camera_", "_label_")
        if label_key not in objects:
            raise ValueError("Unpaired source image")
        sequence, camera = key.split("/")[1], key.split("/")[3]
        pairs.append({"image": objects[key], "label": objects[label_key], "sequence": sequence,
                      "camera": camera, "split": split_for_sequence(sequence)})
    return pairs


def prepare(root, output, image_size=(384, 640), workers=4):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a fresh output folder separate from original sources")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if image_size is not None and (len(image_size) != 2 or min(image_size) < 32 or max(image_size) > 4096):
        raise ValueError("Invalid requested image size")
    inventory_path = root / "source_inventory.json"
    raw_inventory = inventory_path.read_bytes()
    inventory = json.loads(raw_inventory)
    check_inventory(inventory)
    complete = json.loads((root / "DOWNLOAD_COMPLETE.json").read_text(encoding="utf-8"))
    inventory_sha = hashlib.sha256(raw_inventory).hexdigest()
    if not complete.get("complete") or complete.get("inventory_sha256") != inventory_sha:
        raise ValueError("Download is incomplete or inventory identity changed")
    objects = {obj["key"]: obj for obj in inventory["objects"]}
    classes = json.loads(checked_bytes(root, objects[PREFIX + "class_list.json"]))
    lookup, known, mapping = label_tables(classes)
    pairs = collect_pairs(inventory)
    counts = Counter(pair["split"] for pair in pairs)
    if dict(counts) != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected date split counts: {dict(counts)}")
    output.mkdir(parents=True)
    (output / ".incomplete").write_text("Preparation incomplete. Do not train.\n", encoding="utf-8")
    for split in EXPECTED_SPLITS:
        (output / split / "targets").mkdir(parents=True)
        if image_size:
            (output / split / "images").mkdir()
    save_json(output / "label_mapping.json", {"label_space_id": LABEL_SPACE, "mapping": mapping,
              "unknown_colors": 255, "black_void": 255, "source": BASE + PREFIX + "class_list.json"})
    report = {"complete": False, "source_root": str(root), "source_inventory_sha256": inventory_sha,
              "dataset_id": "a2d2", "label_space_id": LABEL_SPACE, "license": "CC BY-ND 4.0",
              "original_files_modified": False, "test_used_for_training_or_validation": False,
              "split_policy": "Custom recording-date split, not an official benchmark split.",
              "validation_dates": sorted(VAL_DATES), "test_dates": sorted(TEST_DATES),
              "test_camera_coverage": "Front center only; the test date contains no labelled side/rear images.",
              "synchronized_multiview_samples": False,
              "prepared_image_size_hw": list(image_size) if image_size else None,
              "preprocessing": "RGB bilinear resize and JPEG quality 95; semantic target nearest-neighbor resize." if image_size else "Original RGB references and original-size target PNGs.",
              "small_obstacle_limit": "Resizing can erase small obstacles; independent native-resolution obstacle evaluation remains required.",
              "scope": "Road surface only. Markings do not confer permission or guarantee free space. No camera calibration or undistortion applied.",
              "label_mapping": mapping}
    save_json(output / "report.json", report)

    def one(pair):
        image_bytes = checked_bytes(root, pair["image"])
        label_bytes = checked_bytes(root, pair["label"])
        with Image.open(io.BytesIO(image_bytes)) as src:
            src.load()
            image = src.convert("RGB")
        with Image.open(io.BytesIO(label_bytes)) as src:
            src.load()
            if src.mode != "RGB" or src.size != image.size:
                raise ValueError("RGB/semantic image mode or dimensions mismatch")
            target, unknown = convert_mask(src, lookup, known)
        if unknown:
            raise ValueError(f"Unrecognized publisher label colors in {pair['label']['key']}: {unknown}")
        native_counts = np.bincount(target.ravel(), minlength=256)
        filename = Path(pair["image"]["key"]).stem
        split = pair["split"]
        target_path_local = output / split / "targets" / (filename + ".png")
        target_image = Image.fromarray(target)
        original_size = image.size
        if image_size:
            height, width = image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            target_image = target_image.resize((width, height), Image.Resampling.NEAREST)
            image_path = output / split / "images" / (filename + ".jpg")
            image.save(image_path, format="JPEG", quality=95, subsampling=0)
        else:
            image_path = target_path(root, pair["image"]["key"])
        target_image.save(target_path_local)
        prepared_counts = np.bincount(np.asarray(target_image).ravel(), minlength=256)
        date = pair["sequence"].split("_", 1)[0]
        row = {"task": "segmentation", "image": str(image_path), "target": str(target_path_local),
               "dataset_id": "a2d2", "label_space_id": LABEL_SPACE,
               "group_id": "a2d2:" + date, "origin_id": "a2d2:" + pair["sequence"],
               "source_recording": pair["sequence"], "camera_id": pair["camera"],
               "source_image": str(target_path(root, pair["image"]["key"])),
               "source_target": str(target_path(root, pair["label"]["key"])),
               "source_image_key": pair["image"]["key"], "source_target_key": pair["label"]["key"],
               "source_image_md5": pair["image"]["md5"], "source_target_md5": pair["label"]["md5"],
               "original_size_hw": [original_size[1], original_size[0]],
               "synchronized_multiview": False}
        return split, row, native_counts[[0, 1, 255]], prepared_counts[[0, 1, 255]], hashlib.sha256(image_bytes).hexdigest()

    rows = {split: [] for split in EXPECTED_SPLITS}
    native_totals, prepared_totals = Counter(), Counter()
    hashes, duplicates, camera_counts, started, last_log = {}, [], Counter(), time.monotonic(), time.monotonic()
    # map preserves the immutable inventory order; only bounded batches are
    # submitted so decoded images and futures cannot grow with dataset size.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pairs), workers * 8):
            for split, row, native, prepared, digest in pool.map(one, pairs[start:start + workers * 8]):
                prior = hashes.get(digest)
                if prior is not None:
                    if prior[0] != split:
                        raise ValueError(f"Identical RGB crosses held-out splits: {prior[1]} and {row['source_image']}")
                    duplicates.append({"kept": prior[1], "duplicate": row["source_image"], "split": split})
                    continue
                hashes[digest] = (split, row["source_image"])
                rows[split].append(row)
                camera_counts[(split, row["camera_id"])] += 1
                for name, n, p in zip(("road", "nonroad", "ignore"), native, prepared):
                    native_totals[name] += int(n)
                    prepared_totals[name] += int(p)
            if time.monotonic() - last_log >= 30:
                emit("prepare_progress", examined=min(start + workers * 8, len(pairs)), total=len(pairs), elapsed_seconds=round(time.monotonic() - started, 1))
                last_log = time.monotonic()
    manifests = {}
    for split, samples in rows.items():
        path = output / (split + ".jsonl")
        with path.open("x", encoding="utf-8", newline="\n") as target:
            for row in samples:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifests[split] = {"manifest": str(path), "samples": len(samples), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report.update(complete=True, source_pairs=len(pairs), manifests=manifests, native_pixel_counts=dict(native_totals),
                  prepared_pixel_counts=dict(prepared_totals), within_split_duplicates_excluded=duplicates,
                  camera_counts={split + "/" + camera: count for (split, camera), count in camera_counts.items()},
                  elapsed_seconds=time.monotonic() - started)
    save_json(output / "report.json", report)
    (output / ".incomplete").unlink()
    emit("prepare_complete", manifests=manifests, duplicates_excluded=len(duplicates))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, default=(384, 640), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--original-resolution", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.root, args.output, None if args.original_resolution else tuple(args.image_size), args.workers)


if __name__ == "__main__":
    main()
