"""CPU-only, append-only correction of a pinned training snapshot.

Reuses byte-verified images and masks. Only explicitly annotated trafficDrum
and warningTriangle pixels change from ignore255 to nonroad1. Official
reporting targets remain untouched. No training, inference or CUDA operations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aihub_preprocess_split import partition_rows

ADDED_NONROAD = frozenset({"trafficDrum", "warningTriangle"})


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    path = Path(path)
    before = path.stat()
    actual = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {path}")
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": after.st_size}


def write_json(path, value):
    # New output only: never overwrite an existing report or failed artifact.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def counts(array):
    values = np.bincount(array.ravel(), minlength=256)
    if values.sum() != sum(int(values[i]) for i in (0, 1, 255)):
        raise ValueError("Mask contains unsupported class IDs")
    return {str(i): int(values[i]) for i in (0, 1, 255) if values[i]}


def correct_mask(old, annotation, adapter, image_name):
    height, width = old.shape
    errors = adapter.annotation_errors(annotation, (width, height), image_name)
    if errors:
        raise ValueError(f"Invalid original annotation: {errors[:3]}")
    reconstructed, inventory = adapter.rasterize(annotation, (width, height), image_name)
    if not np.array_equal(reconstructed, old):
        raise ValueError("Previous rasterizer does not reproduce the pinned old mask")
    union = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(union)
    objects = []
    for index, item in enumerate(annotation["annotations"]):
        if item["class"] in ADDED_NONROAD:
            draw.polygon(list(zip(item["polygon"][::2], item["polygon"][1::2])), fill=1)
            objects.append({"polygon_index": index, "class": item["class"], "polygon": item["polygon"]})
    covered = np.asarray(union, dtype=bool)
    corrected = old.copy()
    corrected[covered] = 1
    changed = corrected != old
    if np.any(changed & ((old != 255) | (corrected != 1))):
        raise ValueError("Correction must only change ignore255 to nonroad1")
    return corrected, objects, int(changed.sum()), dict(inventory)


def check_protected_overlap(training, reporting):
    for field in ("image_sha256", "image_rgb_sha256"):
        a = {r[field] for r in training}
        b = {r[field] for r in reporting}
        if a & b:
            raise ValueError(f"Training/reporting overlap: {field}")
    for field in ("group_id", "origin_id"):
        a = {(r["dataset_id"], r[field]) if field == "group_id" else r[field] for r in training if r.get(field)}
        b = {(r["dataset_id"], r[field]) if field == "group_id" else r[field] for r in reporting if r.get(field)}
        if a & b:
            raise ValueError(f"Training/reporting overlap: {field}")


def check_quarantine(rows, excluded):
    for row in rows:
        if row["dataset_id"].startswith("aihub_") and not row.get("sample_id"):
            raise ValueError("AIHub sample identity required for quarantine check")
        # Replay manifests can identify frames by group/frame fields instead.
        if (row["dataset_id"], row.get("sample_id")) in excluded:
            raise ValueError("Quarantined frame reintroduced")


def process_row(row, output, adapter, mapping_sha):
    if row.get("split") != "train" or row.get("export_split") != "train":
        raise ValueError("Only the frozen training pool may be preprocessed")
    verify(row["image"], row["image_sha256"])
    verify(row["target"], row["target_sha256"])
    with Image.open(row["target"]) as image:
        old = np.array(image)
    if old.ndim != 2 or old.dtype != np.uint8 or old.shape != (row["height"], row["width"]):
        raise ValueError("Mask shape or type mismatch")
    if counts(old) != {str(k): v for k, v in row["target_pixel_counts"].items() if v}:
        raise ValueError("Saved pixel statistics differ from the mask")
    result = dict(row)
    result["preprocessing_provenance"] = {
        "previous_target": row["target"], "previous_target_sha256": row["target_sha256"],
        "previous_mapping_sha256": row.get("mapping_sha256"),
        "previous_export_split": row["export_split"],
        "input_target_fields_role": "Historical input to the previous export; the current mask is target/target_sha256.",
        "image_byte_verified_this_run": True,
        "decoded_rgb_hash_reused_from_byte_identical_verified_export": True,
    }
    change = None
    if row["dataset_id"].startswith("aihub_"):
        verify(row["target_original"], row["target_original_sha256"])
        annotation = json.loads(Path(row["target_original"]).read_text(encoding="utf-8"))
        errors = adapter.annotation_errors(annotation, (row["width"], row["height"]), Path(row["source_image"]).name)
        if errors:
            raise ValueError(f"Unexpected malformed annotation: {row['sample_id']}")
        inventory = dict(Counter(item["class"] for item in annotation["annotations"]))
        if inventory != row["source_class_counts"]:
            raise ValueError("Original annotation class inventory changed")
        result["mapping_sha256"] = mapping_sha
        result["ignored_class_counts"] = {k:v for k,v in row.get("ignored_class_counts", {}).items() if k not in ADDED_NONROAD}
        if any(inventory.get(k, 0) for k in ADDED_NONROAD):
            new, polygons, changed, _ = correct_mask(old, annotation, adapter, Path(row["source_image"]).name)
            target = output / "masks" / row["dataset_id"] / (row["export_id"] + ".png")
            Image.fromarray(new).save(target)
            with Image.open(target) as check:
                if not np.array_equal(np.array(check), new):
                    raise ValueError("New mask round-trip mismatch")
            result["target"] = str(target)
            result["target_sha256"] = sha(target)
            result["target_pixel_counts"] = counts(new)
            change = {"dataset_id": row["dataset_id"], "sample_id": row["sample_id"],
                      "image": row["image"], "source_annotation": row["target_original"],
                      "source_annotation_sha256": row["target_original_sha256"],
                      "old_target": row["target"], "old_sha256": row["target_sha256"],
                      "new_target": str(target), "new_sha256": result["target_sha256"],
                      "ignore_to_nonroad_pixels": changed, "polygons": polygons}
    return result, change


def summarize(rows):
    result = {}
    for row in rows:
        key = row["dataset_id"]
        if key not in result:
            result[key] = {"frames": 0, "pixels": Counter(), "classes_polygons": Counter(),
                           "classes_frames": Counter(), "weather_frames": Counter(), "mean_image_fractions": Counter()}
        stat = result[key]
        stat["frames"] += 1
        stat["pixels"].update(row["target_pixel_counts"])
        stat["classes_polygons"].update(row.get("source_class_counts", {}))
        stat["classes_frames"].update(row.get("source_class_counts", {}).keys())
        stat["weather_frames"].update([row.get("source_weather", "not_recorded")])
        for label, count in row["target_pixel_counts"].items():
            stat["mean_image_fractions"][label] += count / (row["width"] * row["height"])
    for stat in result.values():
        stat["mean_image_fractions"] = {k: v / stat["frames"] for k,v in stat["mean_image_fractions"].items()}
    return result


def run(policy_path, output, workers=4):
    policy_path = Path(policy_path).resolve(strict=True)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists; preserve existing success/failure artifacts")
    # Verify all immutable inputs before creating any output.
    inputs = {name: verify(value["path"], value["sha256"]) for name,value in policy["inputs"].items()}
    report = json.loads(Path(inputs["export_report"]["path"]).read_text(encoding="utf-8"))
    if report.get("complete") is not True or report.get("diagnostic_only") is not False:
        raise ValueError("A completed full export is required")
    marker = json.loads(Path(inputs["export_marker"]["path"]).read_text(encoding="utf-8"))
    if marker["report_sha256"] != inputs["export_report"]["sha256"]:
        raise ValueError("Completion marker/report mismatch")
    for split, name in (("train", "train_manifest"), ("val", "reporting_manifest")):
        if report["manifest_outputs"][split]["sha256"] != inputs[name]["sha256"]:
            raise ValueError("Manifest/report mismatch")
    adapter_spec = importlib.util.spec_from_file_location("pinned_old_aihub_adapter", inputs["old_adapter"]["path"])
    adapter = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter)
    rows = read_rows(inputs["train_manifest"]["path"])
    official = read_rows(inputs["reporting_manifest"]["path"])
    if len(rows) != policy["expected_training_pool_count"]:
        raise ValueError("Unexpected training pool count")
    quarantine = json.loads(Path(inputs["quarantine"]["path"]).read_text(encoding="utf-8"))
    excluded = {(r["dataset_id"], r["sample_id"]) for r in quarantine["excluded_frames"]}
    check_quarantine(rows, excluded)
    check_protected_overlap(rows, official)
    # Partition uses identity, grouping and hashes, never mask performance/labels.
    train_rows, dev_rows, split_report = partition_rows(rows, seed=policy["seed"], fraction=policy["development_fraction_target"])
    membership = {r["export_id"]: ("train", r["development_group_id"]) for r in train_rows}
    membership.update({r["export_id"]: ("dev", r["development_group_id"]) for r in dev_rows})
    if len(membership) != len(rows):
        raise ValueError("Duplicate export IDs")
    output.mkdir(parents=True)
    write_json(output / ".incomplete", {"started_utc": datetime.now(timezone.utc).isoformat(), "policy_sha256": sha(policy_path)})
    write_json(output / "policy.json", policy)
    mapping = dict(adapter.MAPPING)
    mapping["nonroad1"] = sorted(set(mapping["nonroad1"]) | ADDED_NONROAD)
    mapping["revision"] = "training_only_explicit_obstacles_v1"
    mapping["official_reporting_mapping_changed"] = False
    write_json(output / "mapping.json", mapping)
    mapping_sha = sha(output / "mapping.json")
    for dataset_id in {r["dataset_id"] for r in rows}:
        (output / "masks" / dataset_id).mkdir(parents=True)
    prepared = {"train": [], "dev": []}
    changed = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves input order; there are only lightweight row tasks.
            tasks = pool.map(lambda r: process_row(r, output, adapter, mapping_sha), rows)
            for index, (row, change) in enumerate(tasks, 1):
                split, group = membership[row["export_id"]]
                row["split"] = split
                row["export_split"] = split
                row["development_group_id"] = group
                row["preprocessing_provenance"]["development_role"] = "parent_exposed_selection_only" if split == "dev" else "training"
                prepared[split].append(row)
                if change:
                    change["new_split"] = split
                    change["development_group_id"] = group
                    changed.append(change)
                if index % 1000 == 0 or index == len(rows):
                    print(json.dumps({"stage": "verify_and_preprocess", "completed": index, "total": len(rows), "changed_frames": len(changed)}), flush=True)
        manifests = {}
        for split, selected in prepared.items():
            path = output / (split + ".jsonl")
            with path.open("x", encoding="utf-8") as stream:
                for row in selected:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            manifests[split] = {"path": str(path), "sha256": sha(path), "samples": len(selected)}
        # Prove pinned manifests/reports/official contract were not modified.
        for name, item in inputs.items():
            verify(item["path"], item["sha256"])
        write_json(output / "changes.json", changed)
        write_json(output / "split_report.json", split_report)
        report = {"complete": True, "format": "aihub_training_preprocessing_revision_v1",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                  "policy_sha256": sha(output / "policy.json"), "mapping_sha256": mapping_sha,
                  "implementation": {"main_sha256": sha(__file__), "split_sha256": sha(Path(__file__).with_name("aihub_preprocess_split.py"))},
                  "manifests": manifests, "pool_count": len(rows), "quarantine_preserved_frames": len(excluded),
                  "changed_frames": len(changed), "changed_polygons": sum(len(c["polygons"]) for c in changed),
                  "ignore_to_nonroad_pixels": sum(c["ignore_to_nonroad_pixels"] for c in changed),
                  "changes_sha256": sha(output / "changes.json"), "split_report_sha256": sha(output / "split_report.json"),
                  "statistics": {s: summarize(v) for s,v in prepared.items()}, "original_pool_statistics": summarize(rows),
                  "validation": {"all_referenced_training_images_byte_verified": True,
                     "all_referenced_training_masks_byte_verified_and_decoded": True,
                     "all_aihub_training_original_annotations_sha_and_geometry_verified": True,
                     "changed_masks_reproduce_old_rasterizer_before_correction": True,
                     "only_ignore255_to_nonroad1_changed": True,
                     "new_masks_decoded_and_exact_array_checked": True,
                     "image_rgb_hashes_inherited_from_byte_identical_fully_decoded_export": True,
                     "reporting_cross_split_byte_rgb_group_overlap": 0,
                     "official_reporting_manifests_contract_reports_unchanged": True,
                     "reporting_image_mask_bytes_not_reverified_this_run": True},
                  "limitations": policy["limitations"], "training_started": False, "gpu_used": False,
                  "speed_measurement_started": False, "automation_resumed": False,
                  "source_files_modified": False, "storage": "Versioned manifests reference verified existing images/masks; only corrected masks are new files."}
        write_json(output / "report.json", report)
        write_json(output / ".complete", {"report_sha256": sha(output / "report.json")})
        # Keep the start journal for provenance; .complete alone signals success.
        print(json.dumps({"complete": True, "output": str(output), "changed_frames": len(changed), "counts": {k:len(v) for k,v in prepared.items()}}, ensure_ascii=False), flush=True)
        return report
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "training_started": False, "gpu_used": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    run(args.policy, args.output, args.workers)
