"""Derive joint temporal stereo trajectory examples from verified Virtual KITTI 2.

The frame period is an explicit experimental assumption, not measured exposure
timing. Two synthetic front stereo cameras do not represent eight surround views.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from camera_rc.joint_policy.data import DEFAULT_FUTURE_DT, validate_intrinsic, validate_rigid, validate_split_manifests
from download_vkitti2 import SPECS
from prepare_vkitti2 import SCENE_COUNTS, SCENE_SPLITS, SOURCE_DRIVES, MAPPING_URL, MAPPING_SHA256


VARIANTS = ("clone", "fog", "morning", "overcast", "rain", "sunset")
OPTICAL_TO_ANCHOR = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
EXPECTED_HW = (375, 1242)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verified_sources(root):
    results = {}
    for key, expected_files in (("rgb", 42520), ("metadata", 300)):
        record_path, extraction_path = root / f"{key}_complete.json", root / key / "EXTRACTION_COMPLETE.json"
        if (root / key / "INCOMPLETE").exists():
            raise ValueError(f"Incomplete {key} extraction")
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
        extraction = json.loads(extraction_path.read_text(encoding="utf-8-sig"))
        download = record.get("download", {})
        spec = SPECS[key]
        archive = root / "archives" / spec["filename"]
        digest = download.get("sha256")
        if (download.get("verified") is not True or download.get("bytes") != spec["size"]
                or download.get("md5") != spec["md5"] or not isinstance(digest, str) or len(digest) != 64
                or Path(download.get("path", "")).resolve() != archive.resolve()
                or not archive.is_file() or archive.stat().st_size != spec["size"]
                or record.get("extraction") != extraction or extraction.get("asset") != key
                or extraction.get("archive_sha256") != digest or extraction.get("files") != expected_files):
            raise ValueError(f"Missing or inconsistent verified publisher {key} download/extraction records")
        results[key] = {"download_record": str(record_path), "download_record_sha256": _sha256(record_path),
                        "extraction_record_sha256": _sha256(extraction_path), "archive": str(archive),
                        "archive_sha256_recorded_by_downloader": digest, "publisher_md5": spec["md5"],
                        "archive_bytes": spec["size"], "archive_rehashed_in_this_step": False}
    return results


def _read_tables(folder, count):
    result, records = {}, {}
    for filename, columns in (("intrinsic.txt", 6), ("extrinsic.txt", 18)):
        path = folder / filename
        text = path.read_text(encoding="utf-8-sig").splitlines()
        header = ("frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]" if filename == "intrinsic.txt" else
                  "frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1")
        if not text or text[0].split() != header.split():
            raise ValueError(f"Unexpected calibration table header: {path}")
        rows = {}
        for line in text[1:]:
            parts = line.split()
            if len(parts) != columns or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Malformed calibration table row: {path}")
            key = (int(parts[0]), int(parts[1]))
            if key in rows:
                raise ValueError(f"Duplicate calibration frame/camera: {path}")
            values = np.asarray([float(item) for item in parts[2:]], dtype=np.float64)
            if filename == "intrinsic.txt":
                fx, fy, cx, cy = values
                rows[key] = validate_intrinsic([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            else:
                rows[key] = validate_rigid(values.reshape(4, 4), "world_to_camera")
        if set(rows) != {(frame, camera) for frame in range(count) for camera in (0, 1)}:
            raise ValueError(f"Incomplete or extra calibration frame/camera records: {path}")
        result[filename] = rows
        records[filename] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "records": len(rows)}
    return result["intrinsic.txt"], result["extrinsic.txt"], records


def _future_frames(frame_period_s):
    if isinstance(frame_period_s, bool) or not isinstance(frame_period_s, (int, float)) or not math.isfinite(frame_period_s) or frame_period_s <= 0:
        raise ValueError("An explicit finite positive nominal frame_period_s is required")
    frames = [round(dt / frame_period_s) for dt in DEFAULT_FUTURE_DT]
    if any(index < 1 or not math.isclose(index * frame_period_s, dt, abs_tol=1e-7, rel_tol=0) for index, dt in zip(frames, DEFAULT_FUTURE_DT)):
        raise ValueError("Nominal frame period must represent every requested future offset with an integer frame")
    return frames


def _build_rows(scene, variant, count, intrinsics, extrinsics, rgb_root, frame_period_s, stride=5):
    """Private geometry constructor; tests may pass analytic calibration fixtures."""
    offsets = _future_frames(frame_period_s)
    inverse = {key: np.linalg.inv(value) for key, value in extrinsics.items()}
    rows = []
    for anchor in range(2, count - max(offsets), stride):
        anchor_from_world = OPTICAL_TO_ANCHOR @ extrinsics[(anchor, 0)]
        frames = []
        for frame in range(anchor - 2, anchor + 1):
            views = []
            for camera in (0, 1):
                image = rgb_root / scene / variant / "frames" / "rgb" / f"Camera_{camera}" / f"rgb_{frame:05d}.jpg"
                transform = anchor_from_world @ inverse[(frame, camera)]
                validate_rigid(transform)
                views.append({"camera_id": camera, "image": str(image.resolve()),
                              "intrinsic": intrinsics[(frame, camera)].tolist(),
                              "camera_to_anchor": transform.tolist(), "image_size": list(EXPECTED_HW)})
            frames.append({"timestamp_s": frame * frame_period_s, "views": views})
        # Current inputs use ONLY the current and preceding camera poses.
        center = inverse[(anchor, 0)][:3, 3]
        previous_center = inverse[(anchor - 1, 0)][:3, 3]
        current_speed = float(np.linalg.norm(center - previous_center) / frame_period_s)
        previous_heading_world = inverse[(anchor - 1, 0)][:3, 2]
        previous_heading_anchor = anchor_from_world[:3, :3] @ previous_heading_world
        if np.linalg.norm(previous_heading_anchor[:2]) < 0.1:
            raise ValueError("Cannot derive ground-plane yaw from a nearly vertical viewing direction")
        yaw_change = -math.atan2(previous_heading_anchor[1], previous_heading_anchor[0])
        yaw_rate = math.atan2(math.sin(yaw_change), math.cos(yaw_change)) / frame_period_s
        targets = []
        for delta in offsets:
            future = anchor + delta
            future_center = inverse[(future, 0)][:3, 3]
            future_previous = inverse[(future - 1, 0)][:3, 3]
            position = anchor_from_world @ np.r_[future_center, 1]
            speed = np.linalg.norm(future_center - future_previous) / frame_period_s
            targets.append([float(position[0]), float(position[1]), float(speed)])
        origin = "kitti:" + SOURCE_DRIVES[scene]
        rows.append({"schema_version": 1, "dataset_id": "vkitti2_joint_policy_v1",
                     "origin_id": origin, "group_id": origin, "split": SCENE_SPLITS[scene],
                     "source_scene": scene, "condition": variant, "variant": variant, "anchor_frame": anchor,
                     "anchor_timestamp_s": anchor * frame_period_s, "frames": frames,
                     "state": [current_speed, yaw_rate], "future_dt_s": list(DEFAULT_FUTURE_DT), "targets": targets,
                     "diagnostic": False, "partial_dataset": False, "synthetic_simulator_data": True,
                     "nominal_timing_assumed": True, "configured_frame_period_s": frame_period_s,
                     "state_provenance": "Past/current simulator camera0 pose differences only; no future pose or image used as input.",
                     "synchronized_source": "Simulator stereo frame index; not measured real exposure synchronization."})
    return rows


def prepare(root, output, frame_period_s, stride=5, max_train_samples=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    _future_frames(frame_period_s)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_train_samples is not None and (isinstance(max_train_samples, bool) or not isinstance(max_train_samples, int) or max_train_samples < 1):
        raise ValueError("max_train_samples must be a positive integer")
    if output.exists() or root == output or root in output.parents or output in root.parents:
        raise ValueError("Use a fresh output folder separate from downloaded sources")
    verified = _verified_sources(root)
    output.mkdir(parents=True)
    (output / ".incomplete").write_text("Joint sequence preparation incomplete; do not train.\n", encoding="utf-8")
    rows = {split: [] for split in ("train", "val", "test")}
    metadata = {}
    for scene, count in SCENE_COUNTS.items():
        for variant in VARIANTS:
            intrinsics, extrinsics, evidence = _read_tables(root / "metadata" / scene / variant, count)
            metadata[f"{scene}/{variant}"] = evidence
            rows[SCENE_SPLITS[scene]].extend(_build_rows(scene, variant, count, intrinsics, extrinsics, root / "rgb", frame_period_s, stride))
    available_counts = {split: len(values) for split, values in rows.items()}
    partial = max_train_samples is not None and max_train_samples < len(rows["train"])
    if partial:
        def rank(row):
            return hashlib.sha256(f"42:{row['source_scene']}:{row['variant']}:{row['anchor_frame']}".encode()).hexdigest()
        rows["train"] = sorted(rows["train"], key=rank)[:max_train_samples]
        for row in rows["train"]:
            row["partial_dataset"] = True
    image_files = {}
    for split, values in rows.items():
        for row in values:
            for frame in row["frames"]:
                for view in frame["views"]:
                    path = Path(view["image"])
                    path.relative_to(root / "rgb")
                    if str(path) not in image_files:
                        with Image.open(path) as image:
                            rgb = np.asarray(image.convert("RGB"))
                            if (image.height, image.width) != EXPECTED_HW or not np.any(rgb):
                                raise ValueError(f"Invalid original VKITTI image dimensions or empty pixels: {path}")
                        image_files[str(path)] = {"path": str(path), "bytes": path.stat().st_size,
                                                  "image_size": list(EXPECTED_HW), "sha256": _sha256(path)}
                    view["image_sha256"] = image_files[str(path)]["sha256"]
        print(json.dumps({"event": "joint_sequences_checked", "split": split, "sequences": len(values), "unique_input_images": len(image_files)}), flush=True)
    manifests = {}
    for split, values in rows.items():
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as target:
            for row in values:
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        manifests[split] = {"manifest": str(path), "sha256": _sha256(path), "samples": len(values),
                            "conditions": dict(Counter(row["condition"] for row in values)),
                            "origins": sorted({row["origin_id"] for row in values})}
    inventory_path = output / "image_files.jsonl"
    inventory_path.write_text("".join(json.dumps(image_files[key], ensure_ascii=False) + "\n" for key in sorted(image_files)), encoding="utf-8")
    split_validation = validate_split_manifests(*(output / f"{split}.jsonl" for split in ("train", "val", "test")))
    report = {"complete": True, "schema_version": 1, "dataset_id": "vkitti2_joint_policy_v1",
              "manifests": manifests, "metadata_files": metadata, "rgb_and_metadata_source_verification": verified,
              "input_image_inventory": {"path": str(inventory_path), "sha256": _sha256(inventory_path), "files": len(image_files)},
              "split_validation": split_validation, "scene_splits": SCENE_SPLITS, "source_drives": SOURCE_DRIVES,
              "source_mapping_evidence": {"url": MAPPING_URL, "sha256": MAPPING_SHA256},
              "variants": list(VARIANTS), "rotated_camera_variants_excluded": True,
              "history": 3, "views": 2, "camera_ids": [0, 1], "future_dt_s": list(DEFAULT_FUTURE_DT),
              "coordinate_system": "Anchor at current Camera0; x forward, y left, z up. Camera optical x right, y down, z forward.",
              "extrinsics_convention": "Publisher world_to_camera E. camera_to_anchor=C @ E_anchor @ inverse(E_camera_time).",
              "state_derivation": "Backward difference of current/previous camera0 world positions and ground-plane headings; future poses are target labels only.",
              "target_speed_derivation": "Backward position difference at each target frame; positive 3D speed magnitude in m/s under the configured nominal frame period.",
              "configured_frame_period_s": frame_period_s, "nominal_timing_assumed": True,
              "timing_limit": "Source image timestamps/exposure periods were not measured or independently established here. Seconds, speed and yaw rate depend on the user-configured nominal frame period.",
              "stride": stride, "available_samples_before_training_cap": available_counts,
              "max_train_samples": max_train_samples, "partial_dataset": partial, "diagnostic": False,
              "synthetic_simulator_data": True, "models_validated_for_control": False,
              "physical_output": False, "test_used_for_training_or_validation": False,
              "scope": "Two synthetic front stereo cameras only; no eight-camera surround coverage or real-world collision avoidance validation.",
              "license": "Virtual KITTI 2: CC BY-NC-SA 3.0; personal noncommercial research.",
              "source_url": "https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/"}
    _json(output / "report.json", report)
    (output / ".incomplete").unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-period-s", type=float, required=True, help="Explicit assumed simulator frame period; not measured sensor timestamps")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args()
    report = prepare(args.root, args.output, args.frame_period_s, args.stride, args.max_train_samples)
    print(json.dumps({key: report[key] for key in ("complete", "manifests", "diagnostic", "partial_dataset", "nominal_timing_assumed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
