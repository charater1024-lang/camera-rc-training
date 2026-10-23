"""Prepare AIHub collections 13/14/15 polygon labels for road_surface_v1.

Consumes verified extractions; does not download, extract, train or use a GPU.
--source-spec JSON: {"schema_version": 1, "sources": [{"dataset_number": 13,
"dataset_id": "aihub_day_city", "train_root": ".../TS", "val_root": ".../VS",
"train_extraction_report": ".../TS/EXTRACTION_COMPLETE.json",
"val_extraction_report": ".../VS/EXTRACTION_COMPLETE.json"}]}.
Paths are relative to the specification. Training/Validation remain unchanged.

Only explicit freespace is road. Explicit obstacle/nonroad polygons are class
1. Ambiguous classes (including lane paint) and unknown classes are ignore255;
background supplies no new supervision. All nonroad objects override road and
ignore polygons regardless of annotation order. This is partial supervision,
not a complete occupancy or drivable-policy target. Collection 12 is unsupported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from camera_rc.data import validate_splits

LABEL_SPACE = "road_surface_v1"
ROAD_CLASSES = frozenset({"freespace"})
# Exact labels observed in source TS/VS polygon annotations. Do not infer labels
# from the TL/VL relation/QA archives: those are sparse and have a different task.
NONROAD_CLASSES = frozenset({
    "sideWalk", "curb", "fence", "trafficSign", "trafficLight",
    "pedestrian", "rider", "bicycle", "motorcycle", "twoWheeler",
    "vehicle", "bus", "truck", "otherCar", "ambulance", "policeCar", "schoolBus",
    "rubberCone", "constructionGuide",
})
BACKGROUND_CLASSES = frozenset({"background"})
MAPPING = {
    "road0": sorted(ROAD_CLASSES), "nonroad1": sorted(NONROAD_CLASSES),
    "background_no_new_evidence": sorted(BACKGROUND_CLASSES),
    "all_other_classes": 255, "priority": ["road0", "ambiguous_or_unknown255", "nonroad1"],
    "canvas": 255,
    "boundary_convention": "Finite coordinates in [0,width] x [0,height]; width/height mark image edges.",
    "rasterizer": "Pillow ImageDraw.polygon; native dimensions; no resizing",
}
CLIP = re.compile(r"[0-9]+_[0-9]{6}_[0-9]{6}\Z")
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ANNOTATION_VALIDATOR = "aihub_polygon_geometry_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _path(value, parent):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty source path is required")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def annotation_errors(annotation, size, image_name):
    """Inspect all polygons without decoding an image or allocating a mask."""
    width, height = size
    def general(message):
        return [{"polygon_index": None, "class": None, "error": message}]
    if not isinstance(annotation, dict) or not isinstance(annotation.get("information"), dict):
        return general("Expected source annotations/information schema")
    information = annotation["information"]
    resolution = information.get("resolution")
    if (not isinstance(resolution, list) or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
            or resolution != [width, height]):
        return general("Annotation resolution does not match the source image")
    if information.get("filename") != image_name:
        return general("Annotation filename does not match its paired source image")
    annotations = annotation.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return general("A nonempty annotations list is required")
    errors = []
    for index, item in enumerate(annotations):
        error = None
        if not isinstance(item, dict) or not isinstance(item.get("class"), str) or not item["class"]:
            errors.append({"polygon_index": index, "class": None, "error": "Each polygon requires an exact nonempty class name"})
            continue
        polygon = item.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            error = "Polygon must contain at least three flat coordinate pairs"
        elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in polygon):
            error = "Polygon coordinates must be finite numbers"
        else:
            points = list(zip(polygon[::2], polygon[1::2]))
            if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
                error = "Polygon coordinates fall outside declared image boundaries"
            elif len(set(points)) < 3:
                error = "Degenerate polygon with fewer than three distinct points"
            elif abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) < 1e-9:
                error = "Degenerate zero-area polygon"
        if error:
            # Coordinates remain in the untouched source JSON; include finite
            # geometry for review but never emit nonstandard NaN/Infinity JSON.
            record = {"polygon_index": index, "class": item["class"], "error": error}
            if isinstance(polygon, list) and all(type(v) in (int, float) and math.isfinite(v) for v in polygon):
                record["polygon"] = polygon
            errors.append(record)
    return errors


def rasterize(annotation, size, image_name):
    """Return native uint8 mask plus actual polygon class counts."""
    errors = annotation_errors(annotation, size, image_name)
    if errors:
        first = errors[0]
        raise ValueError(f"{first['error']} (image={image_name}, polygon={first['polygon_index']}, class={first['class']})")
    width, height = size
    layers = {key: Image.new("1", size, 0) for key in (0, 1, 255)}
    draws = {key: ImageDraw.Draw(value) for key, value in layers.items()}
    classes = Counter()
    for item in annotation["annotations"]:
        name, polygon = item["class"], item["polygon"]
        points = list(zip(polygon[::2], polygon[1::2]))
        classes[name] += 1
        if name in BACKGROUND_CLASSES:
            continue  # full-image background polygons are annotation canvases, not negative evidence
        value = 0 if name in ROAD_CLASSES else 1 if name in NONROAD_CLASSES else 255
        draws[value].polygon(points, fill=1)
    mask = np.full((height, width), 255, dtype=np.uint8)
    for value in (0, 255, 1):
        mask[np.asarray(layers[value], dtype=bool)] = value
    if not (mask != 255).any():
        raise ValueError("Image has no supervised pixels after conservative mapping")
    return mask, classes


def _inventory(root):
    """Require the observed weather/images|labels/clip/frame archive layout."""
    images, labels = {}, {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "EXTRACTION_COMPLETE.json":
            continue
        relative = path.relative_to(root)
        if "images" not in relative.parts and "labels" not in relative.parts:
            raise ValueError(f"Unexpected source file outside images/labels: {relative}")
        if len(relative.parts) != 4:
            raise ValueError(f"Expected weather/images|labels/clip/frame layout: {relative}")
        weather, kind, clip, filename = relative.parts
        if not CLIP.fullmatch(clip) or not path.stem.startswith(clip + "_"):
            raise ValueError(f"Source frame does not match its explicit clip directory: {relative}")
        key = (weather, clip, path.stem)
        if kind == "images" and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target = images
        elif kind == "labels" and path.suffix.lower() == ".json":
            target = labels
        else:
            raise ValueError(f"Unexpected source extension or directory: {relative}")
        if key in target:
            raise ValueError(f"Duplicate source pairing identity: {relative}")
        target[key] = path.resolve(strict=True)
    if not images or set(images) != set(labels):
        raise ValueError("Missing or orphan source image/polygon annotation")
    return [(key, images[key], labels[key]) for key in sorted(images)]


def prepare(source_spec, output, quarantine_ledger=None):
    source_spec, output = Path(source_spec).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new prepared output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.write_text("Preparation incomplete; do not train this folder.\n", encoding="utf-8")
    report = {
        "complete": False, "diagnostic": False, "format": "camera_rc_aihub_road_v1",
        "label_space_id": LABEL_SPACE, "mapping": MAPPING, "dataset_numbers_supported": [13, 14, 15],
        "source_spec": {"path": str(source_spec), "sha256": sha256(source_spec)},
        "mapping_sha256": hashlib.sha256(json.dumps(MAPPING, sort_keys=True).encode()).hexdigest(),
        "supervision": "Partial: freespace positive, explicit known nonroad negative, all ambiguous pixels ignored.",
        "split_policy": "Publisher Training/Validation retained; explicit clip folders never split or reassigned.",
        "drive_or_day_independence_claimed": False, "clip_namespace": "aihub_commercial:<original clip directory>",
        "archive_identity_basis": "Upstream extraction marker. Archives are not rehashed here; marker is not publisher authentication.",
        "training_started": False, "gpu_used": False, "source_files_modified": False,
        "sources": [], "manifests": {}, "error": None,
    }
    write_json(output / "report.json", report)
    snapshots, digests = {}, {}

    def fingerprint(path):
        path = Path(path)
        key = str(path)
        if key not in digests:
            before = path.stat()
            value = sha256(path)
            after = path.stat()
            signature = (before.st_size, before.st_mtime_ns)
            if signature != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while hashing: {path}")
            digests[key], snapshots[key] = value, signature
        return digests[key]

    try:
        fingerprint(source_spec)
        excluded, consumed = {}, set()
        if quarantine_ledger is not None:
            quarantine_ledger = Path(quarantine_ledger).resolve(strict=True)
            ledger_hash = fingerprint(quarantine_ledger)
            ledger = json.loads(quarantine_ledger.read_text(encoding="utf-8-sig"))
            if (ledger.get("schema_version") != 1 or ledger.get("complete") is not True
                    or ledger.get("format") != "camera_rc_aihub_annotation_quarantine_v1"
                    or ledger.get("validator") != ANNOTATION_VALIDATOR
                    or ledger.get("policy") != "exclude_entire_frame_no_label_repair"
                    or ledger.get("source_spec", {}).get("sha256") != report["source_spec"]["sha256"]):
                raise ValueError("Quarantine ledger is incomplete or bound to a different source/validator")
            for entry in ledger.get("excluded_frames", []):
                key = (entry["dataset_id"], entry["split"], entry["sample_id"])
                if key in excluded or not entry.get("errors") or not SHA256.fullmatch(entry.get("annotation_sha256", "")):
                    raise ValueError("Invalid duplicate/empty quarantine entry")
                excluded[key] = entry
            report["quarantine"] = {"ledger": str(quarantine_ledger), "sha256": ledger_hash,
                                     "policy": ledger["policy"], "excluded_frames": [], "annotation_audit_only": True}
        spec = json.loads(source_spec.read_text(encoding="utf-8-sig"))
        if (type(spec.get("schema_version")) is not int or spec["schema_version"] != 1
                or not isinstance(spec.get("sources"), list) or not spec["sources"]):
            raise ValueError("source-spec requires schema_version=1 and nonempty sources")
        pairs, rows = {"train": [], "val": []}, {"train": [], "val": []}
        dataset_ids, roots, input_clips = set(), set(), {"train": set(), "val": set()}
        for source in sorted(spec["sources"], key=lambda value: value.get("dataset_id", "")):
            dataset_id, number = source.get("dataset_id"), source.get("dataset_number")
            if (not isinstance(dataset_id, str) or not IDENTIFIER.fullmatch(dataset_id)
                    or dataset_id in dataset_ids or type(number) is not int or number not in (13, 14, 15)):
                raise ValueError("Unique safe dataset_id and supported dataset_number 13/14/15 required")
            dataset_ids.add(dataset_id)
            for split in ("train", "val"):
                root = _path(source.get(split + "_root"), source_spec.parent)
                marker = _path(source.get(split + "_extraction_report"), source_spec.parent)
                if root in roots or marker != root / "EXTRACTION_COMPLETE.json":
                    raise ValueError("Each split requires a distinct root and its own EXTRACTION_COMPLETE.json")
                if (root / "INCOMPLETE").exists() or (root / ".incomplete").exists():
                    raise ValueError("Source extraction is incomplete")
                roots.add(root)
                extraction = json.loads(marker.read_text(encoding="utf-8-sig"))
                if (extraction.get("zip_crc_checked") is not True
                        or not isinstance(extraction.get("archive_sha256"), str)
                        or not SHA256.fullmatch(extraction["archive_sha256"])
                        or type(extraction.get("files")) is not int or extraction["files"] < 1
                        or type(extraction.get("expanded_bytes")) is not int or extraction["expanded_bytes"] < 1):
                    raise ValueError("Source extraction marker lacks CRC/completion identity")
                inventory = _inventory(root)
                actual_files = 2 * len(inventory)
                actual_bytes = sum(image.stat().st_size + annotation.stat().st_size
                                   for _, image, annotation in inventory)
                if actual_files != extraction["files"] or actual_bytes != extraction["expanded_bytes"]:
                    raise ValueError("Extracted image/annotation inventory differs from completion marker")
                report["sources"].append({
                    "dataset_id": dataset_id, "dataset_number": number, "split": split, "root": str(root),
                    "samples": len(inventory), "extraction_report": str(marker),
                    "extraction_report_sha256": fingerprint(marker), "archive_sha256": extraction["archive_sha256"],
                    "source_archive": extraction.get("source_archive"),
                    "extracted_files": actual_files, "extracted_bytes": actual_bytes,
                })
                if quarantine_ledger is not None:
                    if ledger.get("audited_counts", {}).get(dataset_id + "/" + split) != len(inventory):
                        raise ValueError("Quarantine audit inventory counts differ from extracted source")
                for key, image, annotation in inventory:
                    input_clips[split].add(key[1])
                    pairs[split].append((dataset_id, number, key, image, annotation))
        if input_clips["train"] & input_clips["val"]:
            raise ValueError("Cross-source official train/validation clip overlap")
        seen_images, seen_rgb, seen_frames = {}, {}, set()
        class_totals, pixel_totals, unknown_totals = {}, {}, {}
        for split in ("train", "val"):
            for index, (dataset_id, number, key, image_path, annotation_path) in enumerate(pairs[split]):
                weather, clip, stem = key
                # The same publisher frame in another category/weather is not a new independent sample.
                frame_key = stem.casefold()
                if frame_key in seen_frames:
                    raise ValueError(f"Duplicate publisher frame identity: {stem}")
                seen_frames.add(frame_key)
                image_hash, annotation_hash = fingerprint(image_path), fingerprint(annotation_path)
                if annotation_path.stat().st_size > 4 * 1024**2:
                    raise ValueError("Annotation exceeds supported 4MiB metadata budget")
                annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
                quarantine_key = (dataset_id, split, f"{weather}/{clip}/{image_path.name}")
                if quarantine_key in excluded:
                    entry = excluded[quarantine_key]
                    information = annotation.get("information", {}) if isinstance(annotation, dict) else {}
                    resolution = information.get("resolution") if isinstance(information, dict) else None
                    size = tuple(resolution) if isinstance(resolution, list) and len(resolution) == 2 else (0, 0)
                    errors = annotation_errors(annotation, size, image_path.name)
                    if (entry["annotation_sha256"] != annotation_hash or not errors or errors != entry["errors"]
                            or Path(entry["annotation"]).resolve() != annotation_path
                            or Path(entry["image"]).resolve() != image_path):
                        raise ValueError(f"Quarantine proof changed or excludes a valid frame: {annotation_path}")
                    consumed.add(quarantine_key)
                    report["quarantine"]["excluded_frames"].append(entry)
                    continue
                with Image.open(image_path) as image:
                    if image.getexif().get(274, 1) not in (None, 1):
                        raise ValueError("EXIF image orientation requires an explicitly aligned preprocessing step")
                    width, height = image.size
                    if min(width, height) < 10:
                        raise ValueError("Source image is smaller than the training loader minimum")
                    rgb = np.asarray(image.convert("RGB"))
                rgb_digest = hashlib.sha256(f"RGB:{width}:{height}:".encode("ascii"))
                rgb_digest.update(memoryview(rgb))
                rgb_hash = rgb_digest.hexdigest()
                for seen, digest in ((seen_images, image_hash), (seen_rgb, rgb_hash)):
                    if digest in seen:
                        raise ValueError(f"Duplicate image bytes/pixels in {seen[digest]} and {split}: {image_path}")
                    seen[digest] = split
                try:
                    mask, classes = rasterize(annotation, (width, height), image_path.name)
                except ValueError as error:
                    raise ValueError(f"{annotation_path}: {error}") from error
                identity = hashlib.sha256(f"{dataset_id}/{clip}/{stem}".encode()).hexdigest()[:32]
                mask_path = output / "masks" / dataset_id / split / (identity + ".png")
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(mask_path, format="PNG")
                counts = np.bincount(mask.reshape(-1), minlength=256)
                pixels = {str(value): int(counts[value]) for value in (0, 1, 255) if counts[value]}
                aggregate = dataset_id + "/" + split
                class_totals.setdefault(aggregate, Counter()).update(classes)
                pixel_totals.setdefault(aggregate, Counter()).update(pixels)
                unknown = {name: count for name, count in classes.items()
                           if name not in ROAD_CLASSES | NONROAD_CLASSES | BACKGROUND_CLASSES}
                unknown_totals.setdefault(aggregate, Counter()).update(unknown)
                rows[split].append({
                    "image": str(image_path), "source_image": str(image_path), "target": str(mask_path),
                    "target_original": str(annotation_path), "task": "segmentation", "label_space_id": LABEL_SPACE,
                    "dataset_id": dataset_id, "source_dataset_number": number, "group_id": clip,
                    "origin_id": "aihub_commercial:" + clip, "source_clip": clip,
                    "sample_id": f"{weather}/{clip}/{image_path.name}", "split": split,
                    "source_split": "Training" if split == "train" else "Validation",
                    "source_weather": weather, "image_sha256": image_hash, "image_rgb_sha256": rgb_hash,
                    "target_original_sha256": annotation_hash, "target_sha256": sha256(mask_path),
                    "width": width, "height": height, "source_class_counts": dict(classes),
                    "ignored_class_counts": unknown, "target_pixel_counts": pixels,
                    "mapping_sha256": report["mapping_sha256"],
                })
                if (index + 1) % 100 == 0 or index + 1 == len(pairs[split]):
                    print(json.dumps({"event": "aihub_prepare", "split": split, "completed": index + 1,
                                      "total": len(pairs[split])}), flush=True)
        if set(excluded) != consumed:
            raise ValueError("Quarantine ledger contains unknown/unconsumed source frames")
        if not rows["train"] or not rows["val"]:
            raise ValueError("Quarantine leaves an empty training or validation split")
        if quarantine_ledger is not None and sha256(quarantine_ledger) != ledger_hash:
            raise ValueError("Quarantine ledger changed during preparation")
        if quarantine_ledger is not None:
            report["quarantine"].update(
                excluded_frame_count=len(consumed),
                excluded_by_source=dict(Counter(key[0] + "/" + key[1] for key in consumed)),
                retained_by_split={split: len(values) for split, values in rows.items()},
                decoded_image_and_mask_coverage="Retained frames only; quarantined images are not decoded or rasterized.")
        validate_splits(rows["train"], rows["val"])
        for path, signature in snapshots.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != signature:
                raise RuntimeError(f"Source changed during preparation: {path}")
        if sha256(source_spec) != report["source_spec"]["sha256"]:
            raise RuntimeError("Source specification changed")
        for source in report["sources"]:
            if sha256(source["extraction_report"]) != source["extraction_report_sha256"]:
                raise RuntimeError("Extraction completion marker changed")
        for split in ("train", "val"):
            pending, manifest = output / f"{split}.jsonl.pending", output / f"{split}.jsonl"
            pending.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                       for row in rows[split]), encoding="utf-8")
            report["manifests"][split] = {
                "manifest": str(manifest), "sha256": sha256(pending), "samples": len(rows[split]),
                "groups": sorted({row["group_id"] for row in rows[split]}), "all_samples_decoded": True,
            }
        for split in ("train", "val"):
            (output / f"{split}.jsonl.pending").replace(output / f"{split}.jsonl")
        report.update(complete=True, total_samples=sum(map(len, rows.values())), all_samples_decoded=True,
                      class_inventory={key: dict(value) for key, value in class_totals.items()},
                      ignored_class_inventory={key: dict(value) for key, value in unknown_totals.items()},
                      target_pixel_counts={key: dict(value) for key, value in pixel_totals.items()},
                      cross_source_clip_leakage_checked=True, exact_and_decoded_rgb_duplicates_checked=True)
        write_json(output / "report.json", report)
        incomplete.unlink()
        (output / ".complete").write_text(json.dumps({"report_sha256": sha256(output / "report.json")}) + "\n",
                                         encoding="utf-8")
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        incomplete.write_text("Preparation failed/incomplete; use a new output directory.\n", encoding="utf-8")
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quarantine-ledger", type=Path,
                        help="Explicit SHA-bound full annotation audit; exclude only proven malformed entire frames")
    args = parser.parse_args()
    report = prepare(args.source_spec, args.output, args.quarantine_ledger)
    print(json.dumps({"complete": report["complete"], "total_samples": report["total_samples"],
                      "manifests": report["manifests"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
