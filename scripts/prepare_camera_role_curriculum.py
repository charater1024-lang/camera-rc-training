"""Partition real source views/conditions for shared-model training and evaluation.

The source-spec JSON lists sources with name, kind, manifests (train/val and
optional test), and optional sampling weight. Paths are relative to that JSON.
Supported kinds: bdd, acdc, vkitti2, a2d2, vehicle. Public source camera IDs are
preserved; they never become fabricated examples from the proposed vehicle rig.
Vehicle rows require camera_role, condition, origin_id and explicit split.
The generated stage can use the existing train/evaluate-sources commands.
Test manifests are reserved and never included in the generated stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from camera_rc.data import read_manifest, validate_splits


LABEL_SPACE = "road_surface_v1"
SPLITS = ("train", "val", "test")
ROLES = ("front_main", "front_wide", "front_near", "left_forward", "right_forward",
         "left_rearward", "right_rearward", "rear")
DATASETS = {"bdd": "bdd100k_drivable", "acdc": "acdc", "vkitti2": "vkitti2_0_3", "a2d2": "a2d2"}
A2D2_VIEWS = {"cam_front_center", "cam_front_left", "cam_front_right",
              "cam_rear_center", "cam_side_left", "cam_side_right"}
VK_VARIANTS = {"15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone",
               "fog", "morning", "overcast", "rain", "sunset"}
TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")


def _text(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty text without surrounding whitespace")
    return value


def _token(value, field):
    value = _text(value, field)
    if not TOKEN.fullmatch(value) or re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", value):
        raise ValueError(f"{field} must be a safe lowercase identifier")
    return value


def _path(value, root):
    result = Path(_text(value, "manifest path")).expanduser()
    return (result if result.is_absolute() else root / result).resolve()


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(row, kind, split):
    if row.get("task") != "segmentation" or row.get("label_space_id") != LABEL_SPACE:
        raise ValueError("Only explicitly remapped segmentation/road_surface_v1 rows are accepted")
    if row.get("split", split) != split:
        raise ValueError("Row split disagrees with its source manifest")
    if kind in DATASETS and row.get("dataset_id") != DATASETS[kind]:
        raise ValueError(f"Unexpected dataset_id for source kind {kind}")
    if kind != "vehicle" and (row.get("camera_role") is not None or row.get("rig_id") is not None):
        raise ValueError("Public data cannot claim a vehicle camera_role or rig_id")
    if kind == "bdd" and not row.get("origin_id"):
        row["origin_id"] = "bdd100k:" + row["group_id"]
    _text(row.get("origin_id"), "canonical origin_id")
    if kind == "vehicle":
        role = _text(row.get("camera_role"), "camera_role")
        if role not in ROLES:
            raise ValueError(f"Unknown vehicle camera_role: {role}")
        _text(row.get("rig_id"), "rig_id")
        dataset = _text(row.get("dataset_id"), "dataset_id")
        clip = _text(row.get("clip_id"), "original clip_id")
        if dataset == "default" or row.get("group_id") != clip or row["origin_id"] != f"{dataset}:{clip}":
            raise ValueError("Vehicle group_id must equal original clip_id and origin_id must be dataset_id:clip_id")
        if row.get("split") != split:
            raise ValueError("Vehicle rows require an explicit original-clip split")
        view, condition = role, _token(row.get("condition"), "condition")
    elif kind == "a2d2":
        view = row.get("camera_id")
        if view not in A2D2_VIEWS:
            raise ValueError(f"Unrecognized A2D2 camera_id: {view}")
        condition = "unlabelled_condition"
    elif kind == "vkitti2":
        camera = row.get("camera_id")
        if isinstance(camera, bool) or not isinstance(camera, int) or camera not in (0, 1):
            raise ValueError("VKITTI requires actual stereo camera_id 0 or 1")
        view, condition = f"front_stereo_{camera}", row.get("variant")
        if condition not in VK_VARIANTS:
            raise ValueError("Unknown VKITTI weather/view variant")
    elif kind == "acdc":
        view, condition = "front", row.get("condition")
        if condition not in {"fog", "night", "rain", "snow"}:
            raise ValueError("Unknown ACDC condition")
    else:
        view, condition = "front", "unlabelled_condition"
    # No cross-camera frame matching, geometry, exposure timestamp or rig role
    # is inferred here, even when the source dataset contains stereo metadata.
    row.update(source_view=view, source_condition=condition, role_training_kind=kind,
               role_training_sample_type="independent_image")
    return view, condition


def _check_pixels(row):
    if not isinstance(row.get("target"), str):
        raise ValueError("Segmentation target must reference an existing ID mask")
    with Image.open(row["image"]) as image, Image.open(row["target"]) as mask:
        image.convert("RGB").load()
        if image.size != mask.size:
            raise ValueError("Image and mask dimensions differ")
        labels = np.asarray(mask)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 255]).all():
            raise ValueError("Expected single-channel 0=road, 1=nonroad, 255=ignore masks")
        reduced = np.asarray(mask.resize((320, 192), Image.Resampling.NEAREST))
        if not (reduced != 255).any():
            raise ValueError("No supervised pixels remain at the training resolution")


def _write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_curriculum(spec_path, output, *, validate_images=True, require_vehicle_coverage=False):
    """Create disjoint evaluation slices without treating public views as rig data."""
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or not isinstance(spec.get("sources"), list) or not spec["sources"]:
        raise ValueError("Source spec requires a nonempty sources list")
    rig_id = _text(spec.get("rig_id", "proposed_eight_exterior_cameras_v1"), "rig_id")
    epochs = spec.get("epochs", 1)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    slices, all_rows, source_reports = {}, {split: [] for split in SPLITS}, []
    names, role_counts, role_conditions = set(), Counter(), defaultdict(set)
    for source in spec["sources"]:
        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")
        name, kind = _token(source.get("name"), "source name"), source.get("kind")
        if name in names or kind not in {*DATASETS, "vehicle"}:
            raise ValueError("Source names must be unique and kind must be supported")
        names.add(name)
        manifests = source.get("manifests")
        if not isinstance(manifests, dict) or not {"train", "val"} <= manifests.keys() or not manifests.keys() <= set(SPLITS):
            raise ValueError("Source manifests require train/val and optionally test")
        weight = source.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Source weight must be finite and positive")
        source_report = {"name": name, "kind": kind, "weight": weight, "manifests": {}}
        for split in SPLITS:
            if split not in manifests:
                continue
            path = _path(manifests[split], spec_path.parent)
            if any((parent / ".incomplete").exists() for parent in path.parents):
                raise ValueError(f"Manifest lies inside incomplete preparation: {path}")
            rows = read_manifest(path)
            for row in rows:
                view, condition = _metadata(row, kind, split)
                if kind == "vehicle":
                    if row["rig_id"] != rig_id:
                        raise ValueError("Vehicle rig_id must match the requested rig")
                    role_counts[(split, view)] += 1
                    role_conditions[(split, view)].add(condition)
                if validate_images:
                    _check_pixels(row)
                slices.setdefault((name, kind, split, view, condition), []).append(row)
            all_rows[split].extend(rows)
            source_report["manifests"][split] = {"path": str(path), "sha256": _fingerprint(path), "samples": len(rows)}
        source_reports.append(source_report)
    # Checking all three pairs also catches duplicated images within each split.
    # Canonical source origins survive view/condition partitioning and derivatives.
    validate_splits(all_rows["train"], all_rows["val"])
    validate_splits(all_rows["train"], all_rows["test"])
    validate_splits(all_rows["val"], all_rows["test"])
    coverage = {role: {split: {"samples": role_counts[(split, role)],
                              "conditions": sorted(role_conditions[(split, role)])}
                       for split in SPLITS} for role in ROLES}
    missing = [f"{split}/{role}" for split in ("train", "val") for role in ROLES if not role_counts[(split, role)]]
    if require_vehicle_coverage and missing:
        raise ValueError("Real-rig coverage is incomplete: " + ", ".join(missing))
    sample_budget = spec.get("samples_per_epoch", min(32000, len(all_rows["train"])))
    if (isinstance(sample_budget, bool) or not isinstance(sample_budget, int)
            or not 1 <= sample_budget <= len(all_rows["train"])):
        raise ValueError("samples_per_epoch must be between 1 and the number of unique training rows")
    # Each source receives its declared probability mass, split equally between
    # actual views and then actual conditions. We never copy front data into gaps.
    train_keys = [key for key in slices if key[2] == "train"]
    source_weights = {row["name"]: row["weight"] for row in source_reports}
    views = {name: {key[3] for key in train_keys if key[0] == name} for name in names}
    conditions = {(name, view): {key[4] for key in train_keys if key[0] == name and key[3] == view}
                  for name in names for view in views[name]}
    output.mkdir(parents=True)
    pending = output / ".incomplete"
    pending.write_text("Do not use until report.json declares preparation complete.\n", encoding="utf-8")
    train_specs, val_specs, manifests_report = [], [], []
    for index, (key, rows) in enumerate(sorted(slices.items())):
        name, kind, split, view, condition = key
        # Numeric filenames avoid pathological path lengths and delimiter collisions.
        manifest = output / f"slice_{index:04d}_{split}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        slice_name = f"{index:04d}__{name}__{view}__{condition}"
        entry = {"name": slice_name, "manifest": str(manifest)}
        if split == "train":
            entry["weight"] = source_weights[name] / len(views[name]) / len(conditions[(name, view)])
            train_specs.append(entry)
        elif split == "val":
            val_specs.append(entry)
        manifests_report.append({**entry, "split": split, "kind": kind, "source": name,
                                 "view": view, "condition": condition, "samples": len(rows),
                                 "origins": len({row["origin_id"] for row in rows}), "sha256": _fingerprint(manifest)})
    config = {"task": "segmentation", "model": {"num_classes": 2, "width": 16},
              "data": {"label_space_id": LABEL_SPACE, "image_size": [192, 320],
                       "train_sources": train_specs, "val_manifests": val_specs},
              "training": {"output_dir": str(output / "run"), "epochs": epochs, "batch_size": 16,
                           "grad_accum_steps": 1, "num_workers": 2, "lr": 0.0002, "seed": 42,
                           "device": "auto", "amp": True, "samples_per_epoch": sample_budget,
                           "segmentation_class_weights": [1.0, 2.0]}}
    _write_json(output / "stage.json", config)
    report = {"complete": True, "spec": str(spec_path), "spec_sha256": _fingerprint(spec_path),
              "rig_id": rig_id, "label_space_id": LABEL_SPACE, "sources": source_reports,
              "manifests": manifests_report, "samples": {split: len(rows) for split, rows in all_rows.items()},
              "image_masks_decoded_and_validated": bool(validate_images),
              "samples_per_epoch": sample_budget,
              "rig_role_coverage": coverage, "missing_real_rig_train_val": missing,
              "all_roles_have_real_rig_train_and_val_samples": not missing,
              "test_used_for_training_or_validation": False,
              "synchronized_multiview_samples_created": False,
              "models_validated_for_control": False,
              "stage": str(output / "stage.json"),
              "model_policy": "One shared per-image road segmentation model; no role-specific geometry or multi-camera fusion is learned by this stage.",
              "sampling_policy": "With replacement; equal view and condition mass inside each weighted source. An epoch is a sampling budget, not a guarantee that every image was seen.",
              "evaluation_policy": "Existing evaluate_sources returns independent metrics for every disjoint validation source/view/condition slice. Test remains reserved.",
              "limitations": ["No object detection, distance, speed or collision-avoidance guarantee follows from road masks.",
                              "Coverage counts do not prove annotation quality, synchronization, calibration or safety.",
                              "Original source group/origin IDs must consistently identify drives across all cameras and conditions; this tool cannot infer missing provenance.",
                              "No copied-image content deduplication; preserve canonical origins for renamed derivatives.",
                              "Unknown source conditions remain explicitly unlabelled; no weather labels are guessed."]}
    _write_json(output / "report.json", report)
    pending.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="Skip image decoding; report this reduced validation explicitly")
    parser.add_argument("--require-vehicle-coverage", action="store_true", help="Require real-rig labelled train and val examples for all eight roles")
    args = parser.parse_args()
    report = prepare_curriculum(args.spec, args.output, validate_images=not args.metadata_only,
                                require_vehicle_coverage=args.require_vehicle_coverage)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
