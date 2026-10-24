"""Prepare explicitly annotated ego-lane masks without importing a trainer.

The label space is independent of road-surface segmentation. Preparation never
starts training, downloads weights, or changes a runtime/deployment model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LABEL_SPACE_ID = "ego_lane_v1"
LABELS = {"0": "annotated_outside_ego_lane", "1": "ego_lane", "255": "unobserved_or_unannotated"}
_EPSILON = 1e-8


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(point, a, b):
    return (abs(_cross(a, b, point)) <= _EPSILON
            and min(a[0], b[0]) - _EPSILON <= point[0] <= max(a[0], b[0]) + _EPSILON
            and min(a[1], b[1]) - _EPSILON <= point[1] <= max(a[1], b[1]) + _EPSILON)


def _segments_intersect(a, b, c, d):
    first, second, third, fourth = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if (((first > _EPSILON and second < -_EPSILON) or (first < -_EPSILON and second > _EPSILON))
            and ((third > _EPSILON and fourth < -_EPSILON) or (third < -_EPSILON and fourth > _EPSILON))):
        return True
    return any((_on_segment(c, a, b), _on_segment(d, a, b), _on_segment(a, c, d), _on_segment(b, c, d)))


def _inside(point, polygon):
    """Boundary-inclusive even/odd containment for a validated simple polygon."""
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(point, a, b):
            return True
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def _polygon(value, name, width, height):
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} must contain at least three pixel [x,y] points")
    points = []
    for point in value:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError(f"{name} requires finite numeric pixel [x,y] points")
        x, y = map(float, point)
        if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
            raise ValueError(f"{name} point is outside image pixel bounds")
        points.append((x, y))
    if points[-1] == points[0]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError(f"{name} requires at least three distinct vertices without repeated points")
    twice_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if abs(twice_area) <= _EPSILON:
        raise ValueError(f"{name} must have nonzero area")
    edges = list(zip(points, points[1:] + points[:1]))
    for index, (a, b) in enumerate(edges):
        for other in range(index + 1, len(edges)):
            if other == index + 1 or (index == 0 and other == len(edges) - 1):
                continue
            if _segments_intersect(a, b, *edges[other]):
                raise ValueError(f"{name} must be a simple polygon without self-intersections")
    return points


def _polygon_within(inner, outer):
    """Check whole edges, including a crossing of a concave valid boundary."""
    if not all(_inside(point, outer) for point in inner):
        return False
    for a, b in zip(inner, inner[1:] + inner[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        cuts = [0., 1.]
        for c, d in zip(outer, outer[1:] + outer[:1]):
            ex, ey = d[0] - c[0], d[1] - c[1]
            denominator = dx * ey - dy * ex
            if abs(denominator) > _EPSILON:
                t = ((c[0] - a[0]) * ey - (c[1] - a[1]) * ex) / denominator
                u = ((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / denominator
                if -_EPSILON <= t <= 1 + _EPSILON and -_EPSILON <= u <= 1 + _EPSILON:
                    cuts.append(min(1., max(0., t)))
            elif abs(_cross(a, b, c)) <= _EPSILON:
                squared = dx * dx + dy * dy
                for point in (c, d):
                    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / squared
                    if 0 <= t <= 1:
                        cuts.append(t)
        cuts = sorted(set(cuts))
        for left, right in zip(cuts, cuts[1:]):
            middle = (left + right) / 2
            if not _inside((a[0] + middle * dx, a[1] + middle * dy), outer):
                return False
    return True


def _mask(width, height, valid, ego):
    layers = []
    for polygon in (valid, ego):
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon(polygon, fill=1)
        layers.append(np.asarray(layer, dtype=np.uint8).astype(bool))
    valid_pixels, ego_pixels = layers
    if not ego_pixels.any() or not valid_pixels.any():
        raise ValueError("Polygons must cover nonempty raster pixels")
    if np.any(ego_pixels & ~valid_pixels):
        raise ValueError("ego_lane_polygon raster must be entirely inside valid_polygon")
    result = np.full((height, width), 255, dtype=np.uint8)
    result[valid_pixels] = 0
    result[ego_pixels] = 1
    return result


def _training_config(image_size, diagnostic_only):
    return {
        "task": "segmentation", "diagnostic_only": diagnostic_only,
        "model": {"num_classes": 2, "width": 16},
        "data": {"label_space_id": LABEL_SPACE_ID, "train_manifest": "train.jsonl",
                 "val_manifest": "val.jsonl", "image_size": list(image_size)},
        "training": {"epochs": 3, "batch_size": 2, "num_workers": 0, "lr": .001,
                     "seed": 42, "device": "cpu", "amp": False, "grad_accum_steps": 1,
                     "segmentation_class_weights": [1., 2.], "output_dir": "training_run"},
    }


def prepare_lane_dataset(annotation_jsonl, output, image_size=(180, 320)):
    """Create new masks/manifests/config only, preserving all input files.

    Rows require image, group_id, split (train/val), ego_lane_polygon and
    valid_polygon. Coordinates are original-image pixel centres, inclusive
    0..width-1 and 0..height-1. Ego polygons must be contained in valid polygons.
    A diagnostic_only=true row marks the entire generated training run diagnostic.
    """
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(value) is not int or value < 16 for value in image_size)):
        raise ValueError("image_size must be [height,width], both integers >=16")
    annotation_jsonl, output = Path(annotation_jsonl).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Lane preparation requires a new output directory: {output}")
    source_bytes = annotation_jsonl.read_bytes()
    annotation_sha = _sha256(source_bytes)
    rows = []
    for line_number, line in enumerate(source_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Annotation line {line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"Annotation line {line_number}: expected a JSON object")
        for name in ("image", "group_id"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Annotation line {line_number}: nonempty {name} is required")
        if row.get("split") not in ("train", "val"):
            raise ValueError(f"Annotation line {line_number}: split must be train or val")
        dataset = row.get("dataset_id", "ego_lane_annotations")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"Annotation line {line_number}: dataset_id must be nonempty text")
        if type(row.get("diagnostic_only", False)) is not bool:
            raise ValueError(f"Annotation line {line_number}: diagnostic_only must be boolean")
        if row.get("label_space_id", LABEL_SPACE_ID) != LABEL_SPACE_ID:
            raise ValueError("Existing road-surface label spaces cannot become ego-lane annotations")
        if "ego_lane_polygon" not in row or "valid_polygon" not in row:
            raise ValueError(f"Annotation line {line_number}: explicit ego_lane_polygon and valid_polygon are required")
        image = Path(row["image"]).expanduser()
        image = (image if image.is_absolute() else annotation_jsonl.parent / image).resolve()
        rows.append((line_number, row, dataset, image))
    if not rows or {row[1]["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Annotations require nonempty train and val splits")
    datasets = {row[2] for row in rows}
    if len(datasets) != 1:
        raise ValueError("Mixed dataset_id values require a separate, explicitly reviewed preparation")
    groups = {split: {row[1]["group_id"] for row in rows if row[1]["split"] == split} for split in ("train", "val")}
    if groups["train"] & groups["val"]:
        raise ValueError("Train/val group_id leakage: one complete drive/group must stay in one split")
    output.mkdir(parents=True, exist_ok=False)
    pending = output / ".incomplete"
    pending.write_text("Incomplete: require .complete, matching report SHA256 and no .incomplete.\n", encoding="utf-8")
    masks = output / "masks"
    masks.mkdir()
    manifests = {"train": [], "val": []}
    counts = {split: Counter() for split in manifests}
    resized_counts = {split: Counter() for split in manifests}
    sources, path_seen, byte_seen, rgb_seen = [], set(), set(), set()
    diagnostic_only = any(row[1].get("diagnostic_only", False) for row in rows)
    for index, (line_number, row, dataset, image) in enumerate(rows):
        normalized = str(image).casefold()
        if normalized in path_seen:
            raise ValueError(f"Duplicate image path at annotation line {line_number}")
        raw_bytes = image.read_bytes()
        image_sha = _sha256(raw_bytes)
        if image_sha in byte_seen:
            raise ValueError(f"Duplicate image byte SHA256 at annotation line {line_number}")
        try:
            with Image.open(io.BytesIO(raw_bytes)) as raw:
                raw.load()
                rgb = raw.convert("RGB")
                width, height = rgb.size
                rgb_sha = _sha256(f"RGB:{width}x{height}\0".encode("ascii") + rgb.tobytes())
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot fully decode image at annotation line {line_number}: {image}") from error
        if rgb_sha in rgb_seen:
            raise ValueError(f"Duplicate decoded RGB SHA256 at annotation line {line_number}")
        valid = _polygon(row["valid_polygon"], "valid_polygon", width, height)
        ego = _polygon(row["ego_lane_polygon"], "ego_lane_polygon", width, height)
        if not _polygon_within(ego, valid):
            raise ValueError(f"ego_lane_polygon must be entirely inside valid_polygon at line {line_number}")
        mask = _mask(width, height, valid, ego)
        resized = np.asarray(Image.fromarray(mask).resize((image_size[1], image_size[0]), Image.Resampling.NEAREST))
        if not np.any(resized != 255):
            raise ValueError(f"No supervised pixels after configured image_size resizing at line {line_number}")
        target = masks / f"{index:09d}.png"
        temporary = target.with_suffix(".tmp")
        Image.fromarray(mask).save(temporary, format="PNG")
        temporary.replace(target)
        mask_sha = _sha256(target.read_bytes())
        split = row["split"]
        provenance = {"annotation_line": line_number, "image": str(image), "image_sha256": image_sha,
                      "rgb_sha256": rgb_sha, "mask": str(target), "mask_sha256": mask_sha,
                      "original_image_size_wh": [width, height], "group_id": row["group_id"], "split": split,
                      "diagnostic_only": row.get("diagnostic_only", False)}
        manifest_row = {"image": str(image), "target": str(target), "source_image": str(image),
                        "group_id": row["group_id"], "dataset_id": dataset, "task": "segmentation",
                        "label_space_id": LABEL_SPACE_ID, "diagnostic_only": row.get("diagnostic_only", False),
                        "source_provenance": {**provenance, "annotation_jsonl": str(annotation_jsonl),
                                              "annotation_sha256": annotation_sha}}
        manifests[split].append(manifest_row)
        sources.append(provenance)
        values, frequencies = np.unique(mask, return_counts=True)
        counts[split].update({int(value): int(frequency) for value, frequency in zip(values, frequencies)})
        values, frequencies = np.unique(resized, return_counts=True)
        resized_counts[split].update({int(value): int(frequency) for value, frequency in zip(values, frequencies)})
        path_seen.add(normalized)
        byte_seen.add(image_sha)
        rgb_seen.add(rgb_sha)
    file_hashes = {}
    for split, manifest in manifests.items():
        path = output / f"{split}.jsonl"
        _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in manifest))
        file_hashes[path.name] = _sha256(path.read_bytes())
    config_path = output / "train_config.json"
    _atomic_json(config_path, _training_config(image_size, diagnostic_only))
    file_hashes[config_path.name] = _sha256(config_path.read_bytes())
    report = {
        "schema_version": 1, "preparation_complete": True, "label_space_id": LABEL_SPACE_ID, "labels": LABELS,
        "output": str(output), "annotation_jsonl": str(annotation_jsonl), "annotation_sha256": annotation_sha,
        "dataset_id": next(iter(datasets)), "diagnostic_only": diagnostic_only, "image_size_hw": list(image_size),
        "samples": {split: len(manifest) for split, manifest in manifests.items()},
        "groups": {split: len(values) for split, values in groups.items()},
        "pixel_counts": {split: {str(label): values[label] for label in (0, 1, 255)} for split, values in counts.items()},
        "resized_pixel_counts": {split: {str(label): values[label] for label in (0, 1, 255)} for split, values in resized_counts.items()},
        "rgb_hash_contract": "sha256(ASCII RGB:<width>x<height> followed by NUL, then decoded RGB bytes)",
        "duplicate_policy": "reject repeated resolved paths, byte SHA256 or decoded RGB SHA256 across all rows",
        "train_val_group_leakage": False, "file_sha256": file_hashes, "sources": sources,
        "train_config": str(config_path), "training_executed": False, "model_adopted": False,
        "automatic_model_replacement": False, "accuracy_measured": False,
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)
    _atomic_json(output / ".complete", {"preparation_complete": True, "report_sha256": _sha256(report_path.read_bytes())})
    pending.unlink()
    return report
