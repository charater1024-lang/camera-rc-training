"""Prepare original-resolution ACDC road masks using its official geographic split.

Only adverse-condition train/val frames are supervised. Normal-condition
references and the hidden-label test split are not put in any training manifest.
Cityscapes labelIds (not labelTrainIds) are mapped to road_surface_v1. ACDC's
invIds records uncertainty in the adverse image, not invalid final annotation:
standard semantic segmentation retains a valid final semantic label there.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


CONDITIONS = ("fog", "night", "rain", "snow")
COUNTS = {"train": {condition: 400 for condition in CONDITIONS},
          "val": {"fog": 100, "night": 106, "rain": 100, "snow": 100}}
CITYSCAPES_VALID = frozenset((7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33))
LABEL_SPACE = "road_surface_v1"
PAPER = ("https://acdc.vision.ee.ethz.ch/other/ACDC_the_Adverse_Conditions_Dataset_with_"
         "Correspondences_for_semantic_driving_scene_understanding-Sakaridis%2BDai%2BVan_Gool-ICCV_21.pdf")
LABEL_DEFINITION = "https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py"
NAME = re.compile(r"^(?P<sequence>[A-Za-z0-9_-]+)_frame_(?P<frame>[0-9]{6})_gt_labelIds\.png$")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def remap_labels(raw, invalid):
    """Return final semantic target; uncertainty is not a semantic void mask."""
    raw, invalid = np.asarray(raw), np.asarray(invalid)
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("ACDC labelIds must be a single-channel uint8 mask")
    if not np.isin(raw, list(range(34)) + [255]).all():
        raise ValueError("Unexpected ACDC/Cityscapes labelIds; do not supply trainIds or RGB colors")
    if invalid.shape != raw.shape or invalid.dtype != np.uint8 or not np.isin(invalid, [0, 1]).all():
        raise ValueError("ACDC invIds must be a same-size uint8 0/1 uncertainty mask")
    target = np.full(raw.shape, 255, dtype=np.uint8)
    target[np.isin(raw, list(CITYSCAPES_VALID))] = 1
    target[raw == 7] = 0
    if not (target != 255).any():
        raise ValueError("No valid final semantic labels in ACDC target")
    return target


def _source_file(path, root):
    if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
        raise ValueError(f"Missing or escaping ACDC source file: {path}")


def discover_pairs(image_root, gt_root):
    image_root, gt_root = Path(image_root).resolve(), Path(gt_root).resolve()
    for path, name in ((image_root, "rgb_anon"), (gt_root, "gt")):
        if path.name != name or not path.is_dir():
            raise ValueError(f"Expected extracted {name} directory, got {path}")
    pairs, seen, image_counts = [], set(), Counter()
    for split, condition_counts in COUNTS.items():
        for condition, expected in condition_counts.items():
            folder = gt_root / condition / split
            targets = sorted(folder.glob("*/*_gt_labelIds.png"))
            if len(targets) != expected:
                raise ValueError(f"Expected {expected} ACDC {condition}/{split} semantic masks, found {len(targets)}")
            matched_images = set()
            for target in targets:
                match = NAME.fullmatch(target.name)
                if not match or target.parent.name != match["sequence"]:
                    raise ValueError(f"ACDC sequence/filename mismatch: {target}")
                sequence, frame = match["sequence"], match["frame"]
                key = (condition, sequence, frame)
                if key in seen:
                    raise ValueError(f"Duplicate ACDC frame identity across official splits: {key}")
                seen.add(key)
                image = image_root / condition / split / sequence / f"{sequence}_frame_{frame}_rgb_anon.png"
                invalid = target.with_name(f"{sequence}_frame_{frame}_gt_invIds.png")
                for path, root in ((image, image_root), (target, gt_root), (invalid, gt_root)):
                    _source_file(path, root)
                matched_images.add(image.resolve())
                pairs.append({"image": image, "target_original": target, "uncertainty_mask": invalid,
                              "split": split, "condition": condition, "sequence": sequence, "frame": frame})
            available = set((image_root / condition / split).glob("*/*_rgb_anon.png"))
            if {path.resolve() for path in available} != matched_images:
                raise ValueError(f"ACDC paired image inventory mismatch: {condition}/{split}")
            image_counts[split] += len(matched_images)
    # This adapter never trains on reference or test images, even if available.
    excluded = Counter()
    for condition in CONDITIONS:
        for split in ("train_ref", "val_ref", "test", "test_ref"):
            excluded[split] += len(list((image_root / condition / split).glob("*/*.png")))
    return pairs, dict(excluded)


def prepare_acdc(image_root, gt_root, output):
    image_root, gt_root, output = (Path(path).expanduser().resolve() for path in (image_root, gt_root, output))
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    for source in (image_root, gt_root):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from ACDC source directories")
    pairs, excluded = discover_pairs(image_root, gt_root)
    output.mkdir(parents=True)
    pending = output / ".incomplete"
    pending.write_text("Preparation incomplete. Do not train from this directory.\n", encoding="utf-8")
    report = {"complete": False, "training_started": False, "dataset_id": "acdc",
              "label_space_id": LABEL_SPACE, "image_root": str(image_root), "gt_root": str(gt_root),
              "source": "https://acdc.vision.ee.ethz.ch/", "protocol": PAPER, "label_definition": LABEL_DEFINITION,
              "split_policy": "Official global geographic train/val split across all conditions (paper section 3.3); no random resplit.",
              "group_policy": "Official geographic partition is group/origin identity; recording IDs are metadata, not asserted geographic boundaries.",
              "uncertainty_policy": "Keep valid final semantic labels where invIds=1 (paper sections 3.4 and 4); invIds retained separately, never used as ignore by itself.",
              "mapping": {"7": 0, "other_18_Cityscapes_evaluation_labels": 1, "void_and_255": 255},
              "original_resolution_preserved": True, "original_files_modified": False,
              "excluded_images": excluded, "test_or_reference_used_for_supervised_learning": False,
              "source_authenticity": "Verify official archive MD5 with download_acdc.py before preparation.",
              "duplicate_policy": "Reject identical decoded RGB across train/val; preserve official temporal/geographic grouping."}
    try:
        rows = {"train": [], "val": []}
        counts = {split: Counter() for split in rows}
        uncertain = {split: Counter() for split in rows}
        rgb_hashes = {}
        for number, pair in enumerate(pairs, 1):
            image_path, raw_path, inv_path = pair["image"], pair["target_original"], pair["uncertainty_mask"]
            with Image.open(image_path) as image:
                image.load()
                if image.format != "PNG":
                    raise ValueError(f"ACDC RGB must be PNG: {image_path}")
                rgb = np.array(image.convert("RGB"))
                size = image.size
            with Image.open(raw_path) as semantic, Image.open(inv_path) as inv:
                semantic.load()
                inv.load()
                if semantic.format != "PNG" or inv.format != "PNG" or semantic.size != size or inv.size != size:
                    raise ValueError(f"Image/annotation size or format mismatch: {raw_path}")
                raw, invalid = np.array(semantic), np.array(inv)
            target = remap_labels(raw, invalid)
            pixel_hash = hashlib.sha256(str(rgb.shape).encode() + b"\0" + rgb.tobytes()).hexdigest()
            if pixel_hash in rgb_hashes:
                previous = rgb_hashes[pixel_hash]
                if previous["split"] != pair["split"]:
                    raise ValueError(f"Decoded RGB train/val leakage: {image_path} and {previous['image']}")
                raise ValueError(f"Duplicate decoded RGB in ACDC split: {image_path}")
            rgb_hashes[pixel_hash] = pair
            split, condition = pair["split"], pair["condition"]
            relative = Path("masks") / condition / split / pair["sequence"] / image_path.name
            target_path = output / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target).save(target_path)
            with Image.open(target_path) as checked:
                if checked.size != size or not np.array_equal(np.asarray(checked), target):
                    raise ValueError(f"Converted ACDC mask verification failed: {target_path}")
            ids, frequencies = np.unique(target, return_counts=True)
            counts[split].update({str(int(label)): int(count) for label, count in zip(ids, frequencies)})
            uncertain[split].update({"uncertain_pixels": int((invalid == 1).sum()),
                                     "uncertain_semantically_labelled_pixels": int(((invalid == 1) & (target != 255)).sum())})
            group = f"acdc:geographic_partition:{split}"
            rows[split].append({"image": str(image_path), "source_image": str(image_path),
                "target": relative.as_posix(), "target_original": str(raw_path), "uncertainty_mask": str(inv_path),
                "task": "segmentation", "label_space_id": LABEL_SPACE, "dataset_id": "acdc",
                "group_id": group, "origin_id": group, "split": split, "source_split": split,
                "condition": condition, "source_sequence": pair["sequence"], "source_frame": pair["frame"],
                "image_sha256": _sha256(image_path), "image_rgb_sha256": pixel_hash,
                "target_sha256": _sha256(target_path),
                "target_original_sha256": _sha256(raw_path), "uncertainty_mask_sha256": _sha256(inv_path),
                "width": size[0], "height": size[1]})
            if number % 100 == 0 or number == len(pairs):
                print(json.dumps({"event": "acdc_remapped", "images": number, "total": len(pairs)}), flush=True)
        report["manifests"] = {}
        for split, values in rows.items():
            manifest = output / f"{split}.jsonl"
            temporary = manifest.with_suffix(".jsonl.pending")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
            report["manifests"][split] = {"manifest": str(manifest), "samples": len(values),
                "sha256": _sha256(temporary), "conditions": dict(Counter(row["condition"] for row in values)),
                "target_pixel_counts": dict(counts[split]), **dict(uncertain[split]),
                "source_sequences": sorted({row["source_sequence"] for row in values})}
        report["source_sequence_names_present_in_both_geographic_splits"] = sorted(
            {row["source_sequence"] for row in rows["train"]} & {row["source_sequence"] for row in rows["val"]})
        for split in rows:
            (output / f"{split}.jsonl.pending").replace(output / f"{split}.jsonl")
        report.update(complete=True, total_samples=len(pairs), exact_rgb_duplicates=0, all_samples_decoded=True)
        _json(output / "report.json", report)
        pending.unlink()
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        _json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path, help="Extracted rgb_anon directory")
    parser.add_argument("--gt-root", required=True, type=Path, help="Extracted gt directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_acdc(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
