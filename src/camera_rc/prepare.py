"""Explicit dataset conversion; never infer absent annotations as background.

BDD schema and IDs: https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst
Drivable PNG IDs are 0=direct, 1=alternative, 2=background (NOT visualization RGB).
Semantic segmentation uses a different image subset; pair files explicitly.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid

BDD_CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle", "traffic light", "traffic sign"]
BDD_ALIASES = {"person": "pedestrian", "motor": "motorcycle", "bike": "bicycle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _json(path: Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_bdd(labels: Path, images: Path):
    from PIL import Image

    frames = _json(labels)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Expected a nonempty BDD frame list: {labels}")
    items, seen, groups = [], set(), set()
    counters = Counter()
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("name"), str):
            raise ValueError("Each BDD frame needs a string name")
        rel = Path(frame["name"])
        source = (images / rel).resolve()
        if rel.is_absolute() or not _inside(source, images) or rel.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsafe or unsupported image path: {rel}")
        # Windows paths are case insensitive. Also reject duplicate label stems.
        key = rel.with_suffix("").as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate image/label stem: {rel}")
        seen.add(key)
        if not source.is_file():
            raise FileNotFoundError(f"Missing annotated image: {source}")
        with Image.open(source) as image:
            width, height = image.size
            image.verify()
        labels_list = frame.get("labels")
        if not isinstance(labels_list, list):
            raise ValueError(f"Missing labels in {rel}; only explicit [] is a negative image")
        lines = []
        for label in labels_list:
            if not isinstance(label, dict):
                raise ValueError(f"Malformed label in {rel}")
            category = label.get("category")
            # Legacy multi-task files contain polygons unrelated to box detection.
            if category in {"lane", "drivable area"} and "box2d" not in label:
                counters[f"skipped_non_box:{category}"] += 1
                continue
            category = BDD_ALIASES.get(category, category)
            if category not in BDD_CLASSES:
                raise ValueError(f"Unknown detection category {category!r} in {rel}; declare a mapping before training")
            attrs = label.get("attributes") or {}
            if any(attrs.get(k) for k in ("ignore", "crowd", "Ignore", "Crowd")):
                raise ValueError(f"Ignored/crowd box in {rel}; plain YOLO labels cannot preserve ignore regions")
            box = label.get("box2d")
            try:
                coords = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing/malformed box2d in {rel}") from error
            if not all(math.isfinite(v) for v in coords):
                raise ValueError(f"Non-finite box in {rel}")
            x1, y1, x2, y2 = coords
            if x2 < x1 or y2 < y1:
                raise ValueError(f"Reversed box in {rel}")
            # Official BDD/Scalabel format describes inclusive max coordinates.
            # Convert to continuous edge coordinates before clipping/normalizing.
            original = (x1, y1, x2 + 1.0, y2 + 1.0)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2 + 1.0), min(float(height), y2 + 1.0)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box entirely outside image in {rel}")
            counters["clipped_boxes"] += int(original != (x1, y1, x2, y2))
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height)
            lines.append(str(BDD_CLASSES.index(category)) + " " + " ".join(f"{v:.8f}" for v in values))
            counters[f"class:{category}"] += 1
        if not lines:
            counters["negative_images"] += 1
        group = frame.get("videoName")
        if group is not None:
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"Invalid videoName in {rel}")
            groups.add(group)
        items.append((rel, source, lines, group))
    return items, seen, groups, dict(counters)


def prepare_bdd_detection(train_labels: Path, val_labels: Path, train_images: Path,
                          val_images: Path, output: Path, overwrite: bool = False) -> dict:
    """Convert explicit official splits; hard-link images or copy across volumes.

    No downloads occur. Unknown categories/absent labels/missing images fail.
    --overwrite preserves the previous output as a sibling backup; it never
    recursively deletes source files. Native official train/val separation is kept.
    """
    output = Path(output).resolve()
    inputs = [Path(p).resolve() for p in (train_labels, val_labels, train_images, val_images)]
    if any(_inside(p, output) for p in inputs):
        raise ValueError("Output cannot contain any input path")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use a new folder or --overwrite")
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory")
    if inputs[0] == inputs[1]:
        raise ValueError("Train and validation require separate label files")
    train = _read_bdd(inputs[0], inputs[2])
    val = _read_bdd(inputs[1], inputs[3])
    # Basename collisions are rejected even if files live in different subfolders.
    train_names = {item[0].stem.casefold() for item in train[0]}
    val_names = {item[0].stem.casefold() for item in val[0]}
    if train_names & val_names or train[2] & val[2]:
        raise ValueError("Train/validation image stems or videoName groups overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    report = {"format": "camera_rc_bdd_detection_v1", "classes": BDD_CLASSES,
              "box_coordinates": "BDD inclusive maxima converted with x2+1,y2+1 then clipped",
              "hardlinked_images": 0, "copied_images": 0, "output": str(output)}
    try:
        for split, data in (("train", train), ("val", val)):
            records = []
            for rel, source, lines, group in data[0]:
                dest = staging / "images" / split / rel
                label_path = staging / "labels" / split / rel.with_suffix(".txt")
                dest.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, dest)
                    report["hardlinked_images"] += 1
                except OSError:
                    shutil.copy2(source, dest)
                    report["copied_images"] += 1
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                records.append({"image": str(output / "images" / split / rel),
                                "target": str(output / "labels" / split / rel.with_suffix(".txt")),
                                "source_image": str(source), "sample_id": rel.as_posix(),
                                "group_id": group or rel.stem, "dataset_id": "bdd100k_detection",
                                "split": split})
            _write_jsonl(staging / f"{split}.jsonl", records)
            report[split] = {"images": len(records), **data[3]}
        # JSON is a YAML subset, with portable Unicode and path escaping.
        _write_json(staging / "dataset.yaml", {"path": output.as_posix(),
                    "train": "images/train", "val": "images/val", "names": BDD_CLASSES})
        if output.exists():
            backup = output.with_name(output.name + ".backup-" + uuid.uuid4().hex[:8])
            output.rename(backup)
            report["backup"] = str(backup)
        _write_json(staging / "conversion_report.json", report)
        staging.rename(output)
    except BaseException:
        # Only the tool-created staging directory, verified within the output parent.
        if staging.exists() and _inside(staging, output.parent) and staging.name.startswith("." + output.name + "-"):
            shutil.rmtree(staging)
        backup = report.get("backup")
        if backup and not output.exists():
            Path(backup).rename(output)
        raise
    return report


def split_manifest(input: Path, output: Path, val_fraction: float = 0.2, seed: int = 42) -> dict:
    """Split complete user-declared groups, never individual adjacent frames.

    Path fields image/target (if string)/mask/depth are resolved relative to the
    source JSONL. Existing output directories are refused; original order is kept.
    The requested fraction refers to groups, not necessarily the frame count.
    """
    input, output = Path(input).resolve(), Path(output).resolve()
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between zero and one")
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    rows, groups, seen_images = [], set(), set()
    with input.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("group_id"), str) or not row["group_id"].strip():
                raise ValueError(f"Line {line_number} requires a nonempty string group_id")
            if not isinstance(row.get("image"), str) or not row["image"].strip():
                raise ValueError(f"Line {line_number} requires image")
            for key in ("image", "target", "mask", "depth"):
                if isinstance(row.get(key), str):
                    path = Path(row[key])
                    path = (path if path.is_absolute() else input.parent / path).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"Line {line_number}: {path}")
                    row[key] = str(path)
            identity = os.path.normcase(row["image"])
            if identity in seen_images:
                raise ValueError(f"Duplicate image: {row['image']}")
            seen_images.add(identity)
            groups.add(row["group_id"])
            rows.append(row)
    if len(groups) < 2:
        raise ValueError("At least two independent group_id values are required")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    count = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
    val_groups = set(ordered[:count])
    splits = {"train": [], "val": []}
    for row in rows:
        splits["val" if row["group_id"] in val_groups else "train"].append(row)
    output.mkdir(parents=True)
    for split, split_rows in splits.items():
        _write_jsonl(output / f"{split}.jsonl", split_rows)
    report = {"seed": seed, "val_fraction_groups": val_fraction,
              "train_groups": sorted(groups - val_groups), "val_groups": sorted(val_groups),
              "train_samples": len(splits["train"]), "val_samples": len(splits["val"])}
    _write_json(output / "split_report.json", report)
    return report


def make_paired_manifest(images: Path, targets: Path, output: Path, task: str,
                         group_pattern: str | None = None) -> dict:
    """Pair by identical relative stems; requires a user-defined sequence regex.

    Regex applies to the relative image path, with named group 'group' or its
    first capture. Use source split-specific directories; do not pair semantic
    masks against the 100K detection images. KITTI raw/depth trees often need an
    explicit custom manifest because their directory structures differ.
    """
    if task not in {"segmentation", "depth"}:
        raise ValueError("Paired manifest task must be segmentation or depth")
    if not group_pattern:
        raise ValueError("Explicit group_pattern required; sequence IDs are never guessed")
    pattern = re.compile(group_pattern)
    if pattern.groups < 1:
        raise ValueError("group_pattern needs a capturing group")
    images, targets, output = Path(images).resolve(), Path(targets).resolve(), Path(output).resolve()
    if not images.is_dir() or not targets.is_dir():
        raise FileNotFoundError("Both image and target directories must exist")
    if output.exists():
        raise FileExistsError(output)
    target_index = {}
    for path in targets.rglob("*.png"):
        key = path.relative_to(targets).with_suffix("").as_posix()
        if key in target_index:
            raise ValueError(f"Duplicate target stem: {key}")
        target_index[key] = path
    rows, seen = [], set()
    for image in sorted(images.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image.relative_to(images).as_posix()
        key = image.relative_to(images).with_suffix("").as_posix()
        if key in seen:
            raise ValueError(f"Duplicate image stem: {key}")
        seen.add(key)
        if key not in target_index:
            raise FileNotFoundError(f"No target with matching relative stem: {relative}")
        match = pattern.search(relative)
        group = (match.groupdict().get("group") or match.group(1)) if match else None
        if not group:
            raise ValueError(f"group_pattern did not identify a group: {relative}")
        rows.append({"image": str(image), "target": str(target_index[key]), "group_id": group,
                     "sample_id": relative, "dataset_id": "unknown", "task": task})
    if not rows:
        raise ValueError("No image/target pairs found")
    orphan_targets = set(target_index) - seen
    if orphan_targets:
        raise ValueError(f"{len(orphan_targets)} target files have no matching images")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    return {"samples": len(rows), "groups": len({row['group_id'] for row in rows}), "output": str(output)}


def _commit_paired_splits(output: Path, splits: dict, dataset_id: str) -> dict:
    """Publish complete manifests atomically, refusing existing destinations."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in ("train", "val"):
        if not splits[name]:
            raise ValueError(f"No samples found for {name}")
    group_sets = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    if group_sets["train"] & group_sets["val"]:
        raise ValueError("Official train/validation group IDs overlap; do not split stereo views or sequences")
    images = {name: {os.path.normcase(row["image"]) for row in rows} for name, rows in splits.items()}
    if images["train"] & images["val"]:
        raise ValueError("Train/validation image paths overlap")
    report = {"dataset_id": dataset_id, "output": str(output)}
    for name, rows in splits.items():
        report[name] = {"samples": len(rows), "groups": len(group_sets[name])}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, rows in splits.items():
            _write_jsonl(staging / f"{name}.jsonl", rows)
        _write_json(staging / "preparation_report.json", report)
        staging.rename(output)
    except BaseException:
        if staging.exists() and _inside(staging, output.parent) and staging.name.startswith("." + output.name + "-"):
            shutil.rmtree(staging)
        raise
    return report


def prepare_bdd_drivable(train_images: Path, val_images: Path, train_masks: Path,
                         val_masks: Path, output: Path) -> dict:
    """Pair official BDD100K drivable splits without assuming the semantic subset.

    Masks may be FRAME.png or FRAME_drivable_id.png. Raw IDs must be 0/1/2;
    visualization RGB masks are rejected. Each official BDD100K frame stems
    from its own clip, so the known source clip stem is the group ID.
    """
    from PIL import Image

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    splits = {}
    for split, image_dir, mask_dir in (("train", train_images, train_masks),
                                        ("val", val_images, val_masks)):
        image_dir, mask_dir = Path(image_dir).resolve(), Path(mask_dir).resolve()
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} image/mask directory")
        index = {}
        for image in image_dir.rglob("*"):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image.stem.casefold()
            if stem in index:
                raise ValueError(f"Ambiguous BDD image stem: {image.stem}")
            index[stem] = image
        rows, seen = [], set()
        for mask in sorted(mask_dir.rglob("*")):
            if not mask.is_file() or mask.suffix.lower() != ".png":
                continue
            stem = mask.stem.removesuffix("_drivable_id")
            key = stem.casefold()
            if key in seen:
                raise ValueError(f"Duplicate BDD mask stem: {stem}")
            seen.add(key)
            image = index.get(key)
            if image is None:
                raise FileNotFoundError(f"Missing BDD image for mask: {mask}")
            with Image.open(mask) as target, Image.open(image) as rgb:
                if target.mode not in {"L", "P"}:
                    raise ValueError(f"Expected raw BDD 0/1/2 mask, not color visualization: {mask}")
                colors = target.getcolors(maxcolors=256)
                if colors is None or any(value not in (0, 1, 2) for _, value in colors):
                    raise ValueError(f"Invalid BDD drivable mask IDs: {mask}")
                if target.size != rgb.size:
                    raise ValueError(f"Image/mask dimensions differ: {mask}")
            rows.append({"image": str(image), "target": str(mask), "group_id": key,
                         "dataset_id": "bdd100k_drivable", "sample_id": image.name,
                         "task": "segmentation"})
        # Missing target is never silently interpreted as background.
        if set(index) - seen:
            raise ValueError(f"{len(set(index) - seen)} {split} images have no drivable mask; use matching official split directories")
        splits[split] = rows
    return _commit_paired_splits(output, splits, "bdd100k_drivable")


def prepare_kitti_depth(raw_root: Path, depth_root: Path, output: Path) -> dict:
    """Pair KITTI depth ground truth to synchronized raw RGB, preserving drives.

    depth_root/{train,val}/DRIVE/proj_depth/groundtruth/image_0{2,3}/FRAME.png
    pairs raw_root/[DATE/]DRIVE/image_0{2,3}/data/FRAME.png. Exactly one of the
    two raw layouts must exist. Both stereo cameras share group_id=DRIVE.
    KITTI ground truth is uint16 PNG, 256 units/m; zero is invalid. Decoding and
    dimension checks are performed by the depth data loader / check-data.
    """
    raw_root, depth_root, output = Path(raw_root).resolve(), Path(depth_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not raw_root.is_dir() or not depth_root.is_dir():
        raise FileNotFoundError("KITTI raw_root and depth_root must be directories")
    splits = {}
    for split in ("train", "val"):
        split_root = depth_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing official KITTI depth split: {split_root}")
        rows, seen = [], set()
        for target in sorted(split_root.glob("**/proj_depth/groundtruth/image_0[23]/*.png")):
            drive = target.parents[3].name
            if not re.fullmatch(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync", drive):
                raise ValueError(f"Unrecognized KITTI synchronized drive directory: {drive}")
            camera, frame = target.parent.name, target.name
            date = drive[:10]
            candidates = [raw_root / date / drive / camera / "data" / frame,
                          raw_root / drive / camera / "data" / frame]
            existing = [p.resolve() for p in candidates if p.is_file()]
            if not existing:
                raise FileNotFoundError(f"Missing raw RGB for {drive}/{camera}/{frame}; tried {candidates}")
            if len(existing) != 1:
                raise ValueError(f"Ambiguous raw RGB layouts for {drive}/{camera}/{frame}")
            sample_id = f"{drive}/{camera}/{frame}"
            if sample_id in seen:
                raise ValueError(f"Duplicate KITTI ground-truth sample: {sample_id}")
            seen.add(sample_id)
            rows.append({"image": str(existing[0]), "target": str(target.resolve()), "group_id": drive,
                         "dataset_id": "kitti_depth", "sample_id": sample_id, "task": "depth"})
        splits[split] = rows
    return _commit_paired_splits(output, splits, "kitti_depth")
