"""Validated temporal multi-camera samples for a joint image-to-trajectory model.

Coordinates: optical camera x right, y down, z forward; anchor x forward,
y left, z up. Only frames at or before the anchor enter model inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_FUTURE_DT = (0.5, 1.0, 1.5, 2.0, 2.5)


def _text(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty text without surrounding whitespace")
    return value


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric data")
    return float(value)


def _array(value, shape, field):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Expected numeric array without booleans or strings")
        result = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite with shape {shape}")
    return result


def validate_intrinsic(value):
    matrix = _array(value, (3, 3), "intrinsic")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[1, 0]) > 1e-7
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-7, rtol=0)):
        raise ValueError("intrinsic must have positive focal lengths and pinhole bottom row")
    return matrix


def validate_rigid(value, field="camera_to_anchor"):
    matrix = _array(value, (4, 4), field)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0)):
        raise ValueError(f"{field} must be a proper rigid transform with rotation determinant +1")
    return matrix


def resize_intrinsic(intrinsic, original_hw, resized_hw):
    """Pixel-center convention matching PIL bilinear resize, including skew."""
    original_h, original_w = original_hw
    resized_h, resized_w = resized_hw
    sx, sy = resized_w / original_w, resized_h / original_h
    pixel_map = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return pixel_map @ validate_intrinsic(intrinsic)


def _image_size(value, field, minimum=1):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < minimum for x in value)):
        raise ValueError(f"{field} must be integer [height,width] >= {minimum}")
    return tuple(value)


class JointSequenceDataset(Dataset):
    def __init__(self, manifest, image_size=(128, 224), expected_history=3, expected_views=2,
                 expected_future_dt_s=DEFAULT_FUTURE_DT):
        self.manifest = Path(manifest).resolve()
        self.image_size = _image_size(image_size, "image_size", 16)
        if isinstance(expected_history, bool) or not isinstance(expected_history, int) or not 2 <= expected_history <= 4:
            raise ValueError("expected_history must be 2..4")
        if isinstance(expected_views, bool) or not isinstance(expected_views, int) or not 1 <= expected_views <= 8:
            raise ValueError("expected_views must be 1..8")
        expected_dt = np.asarray(expected_future_dt_s, dtype=np.float64)
        if expected_dt.ndim != 1 or not len(expected_dt) or not np.isfinite(expected_dt).all() or np.any(expected_dt <= 0) or np.any(np.diff(expected_dt) <= 0):
            raise ValueError("expected_future_dt_s must be finite positive increasing offsets")
        self.future_dt_s = expected_dt
        payload = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        self.rows, self.file_inventory = [], {}
        split_names, seen_rows, manifest_camera_ids = set(), set(), None
        for lineno, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                raise ValueError(f"Expected joint schema_version=1 at {self.manifest}:{lineno}")
            for key in ("dataset_id", "origin_id", "group_id"):
                _text(row.get(key), key)
            if row.get("split") not in {"train", "val", "test"}:
                raise ValueError("split must be train, val or test")
            split_names.add(row["split"])
            if not isinstance(row.get("diagnostic"), bool) or not isinstance(row.get("partial_dataset", False), bool):
                raise ValueError("Explicit diagnostic boolean and optional partial_dataset boolean are required")
            anchor = _number(row.get("anchor_timestamp_s"), "anchor_timestamp_s")
            if anchor < 0:
                raise ValueError("anchor_timestamp_s must be nonnegative")
            frames = row.get("frames")
            if not isinstance(frames, list) or len(frames) != expected_history:
                raise ValueError("Temporal history length differs from expected_history")
            reference_ids, times, paths = None, [], []
            for frame in frames:
                if not isinstance(frame, dict):
                    raise ValueError("Each frame must be an object")
                timestamp = _number(frame.get("timestamp_s"), "frame timestamp_s")
                if timestamp < 0 or timestamp > anchor + 1e-7:
                    raise ValueError("Future images or negative timestamps are forbidden in inputs")
                times.append(timestamp)
                views = frame.get("views")
                if not isinstance(views, list) or len(views) != expected_views:
                    raise ValueError("View count differs from expected_views")
                identifiers = []
                for view in views:
                    if not isinstance(view, dict):
                        raise ValueError("Each view must be an object")
                    camera = view.get("camera_id")
                    if isinstance(camera, bool) or not isinstance(camera, (str, int)) or isinstance(camera, str) and (not camera or camera != camera.strip()):
                        raise ValueError("camera_id must be nonempty text or an integer")
                    identifiers.append((type(camera).__name__, camera))
                    size = _image_size(view.get("image_size"), "original image_size")
                    validate_intrinsic(view.get("intrinsic"))
                    validate_rigid(view.get("camera_to_anchor"))
                    path = Path(_text(view.get("image"), "image")).expanduser()
                    path = (path if path.is_absolute() else self.manifest.parent / path).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    identity = str(path).casefold()
                    if identity not in self.file_inventory:
                        with Image.open(path) as image:
                            actual_size = (image.height, image.width)
                        self.file_inventory[identity] = {"path": str(path), "bytes": path.stat().st_size, "image_size": list(actual_size)}
                    if tuple(self.file_inventory[identity]["image_size"]) != size:
                        raise ValueError("Declared original image dimensions differ from the file")
                    expected_hash = view.get("image_sha256")
                    if expected_hash is not None:
                        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
                            raise ValueError("image_sha256 must be a lowercase SHA256 hex string")
                        record = self.file_inventory[identity]
                        if "sha256" not in record:
                            digest = hashlib.sha256()
                            with path.open("rb") as source:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    digest.update(block)
                            record["sha256"] = digest.hexdigest()
                        if record["sha256"] != expected_hash:
                            raise ValueError("Image SHA256 differs from the manifest")
                    view["image"] = str(path)
                    paths.append(identity)
                if len(set(identifiers)) != len(identifiers):
                    raise ValueError("Duplicate camera_id within a frame")
                if reference_ids is not None and identifiers != reference_ids:
                    raise ValueError("Camera IDs and order must stay consistent across history frames")
                reference_ids = identifiers
            if any(right <= left for left, right in zip(times, times[1:])) or not math.isclose(times[-1], anchor, abs_tol=1e-7, rel_tol=0):
                raise ValueError("History must strictly increase and finish at the anchor timestamp")
            if len(paths) != len(set(paths)):
                raise ValueError("One image cannot masquerade as multiple times or camera views in a sample")
            if manifest_camera_ids is not None and reference_ids != manifest_camera_ids:
                raise ValueError("Camera IDs and order must stay consistent across all manifest samples")
            manifest_camera_ids = reference_ids
            dt = _array(row.get("future_dt_s"), expected_dt.shape, "future_dt_s")
            if not np.allclose(dt, expected_dt, atol=1e-7, rtol=0):
                raise ValueError("future_dt_s differs from expected_future_dt_s")
            state = _array(row.get("state"), (2,), "state")
            targets = _array(row.get("targets"), (len(dt), 3), "targets")
            if state[0] < 0 or np.any(targets[:, 2] < 0):
                raise ValueError("State and target speed must be nonnegative")
            identity = (row["origin_id"], anchor, tuple(paths))
            if identity in seen_rows:
                raise ValueError("Duplicate joint sequence sample")
            seen_rows.add(identity)
            self.rows.append(row)
        if not self.rows or len(split_names) != 1:
            raise ValueError("Manifest must contain nonempty samples from exactly one split")
        self.split = next(iter(split_names))
        self.diagnostic = any(row["diagnostic"] for row in self.rows)
        self.partial_dataset = any(row.get("partial_dataset", False) for row in self.rows)
        self.provenance = {"manifest": str(self.manifest), "sha256": self.manifest_sha256,
                           "samples": len(self.rows), "split": self.split,
                           "origins": sorted({row["origin_id"] for row in self.rows}),
                           "unique_image_files": len(self.file_inventory),
                           "image_files_sha256_verified": sum("sha256" in record for record in self.file_inventory.values()),
                           "source_image_hashes_verified": all("sha256" in record for record in self.file_inventory.values()),
                           "camera_ids": [camera for _, camera in manifest_camera_ids],
                           "image_hash_limit": "Undeclared image hashes cannot be checked against preparation-time bytes; file headers and dimensions are still validated.",
                           "diagnostic": self.diagnostic, "partial_dataset": self.partial_dataset}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images, intrinsics, transforms = [], [], []
        for frame in row["frames"]:
            frame_images, frame_intrinsics, frame_transforms = [], [], []
            for view in frame["views"]:
                with Image.open(view["image"]) as raw:
                    if (raw.height, raw.width) != tuple(view["image_size"]):
                        raise ValueError("Source image dimensions changed after manifest validation")
                    image = raw.convert("RGB")
                    values = np.asarray(image)
                    if not np.any(values):
                        raise ValueError("All-zero images cannot be used as valid camera observations")
                    image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
                    if not np.any(np.asarray(image)):
                        raise ValueError("Image became all-zero at model input resolution")
                    frame_images.append(torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0)
                frame_intrinsics.append(torch.tensor(resize_intrinsic(view["intrinsic"], view["image_size"], self.image_size), dtype=torch.float32))
                frame_transforms.append(torch.tensor(view["camera_to_anchor"], dtype=torch.float32))
            images.append(torch.stack(frame_images))
            intrinsics.append(torch.stack(frame_intrinsics))
            transforms.append(torch.stack(frame_transforms))
        return {"images": torch.stack(images), "intrinsics": torch.stack(intrinsics),
                "camera_to_anchor": torch.stack(transforms),
                "time_offsets": torch.tensor([frame["timestamp_s"] - row["anchor_timestamp_s"] for frame in row["frames"]], dtype=torch.float32),
                "state": torch.tensor(row["state"], dtype=torch.float32),
                "targets": torch.tensor(row["targets"], dtype=torch.float32),
                "future_dt_s": torch.tensor(row["future_dt_s"], dtype=torch.float32),
                "diagnostic": torch.tensor(row["diagnostic"], dtype=torch.bool)}


def validate_split_manifests(train, val, test=None, **dataset_options):
    datasets = {"train": JointSequenceDataset(train, **dataset_options), "val": JointSequenceDataset(val, **dataset_options)}
    if test is not None:
        datasets["test"] = JointSequenceDataset(test, **dataset_options)
    for split, dataset in datasets.items():
        if dataset.split != split:
            raise ValueError(f"Manifest rows do not declare their requested {split} split")
    keys = list(datasets)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            a, b = datasets[left], datasets[right]
            if {row["origin_id"] for row in a.rows} & {row["origin_id"] for row in b.rows}:
                raise ValueError(f"Original-drive leakage across {left}/{right}, including weather or camera variants")
            if {(row["dataset_id"], row["group_id"]) for row in a.rows} & {(row["dataset_id"], row["group_id"]) for row in b.rows}:
                raise ValueError(f"Scene group leakage across {left}/{right}")
            if a.file_inventory.keys() & b.file_inventory.keys():
                raise ValueError(f"Image leakage across {left}/{right}")
    return {"disjoint": True, "original_drives_and_image_paths_checked": True,
            "manifests": {split: dataset.provenance for split, dataset in datasets.items()},
            "diagnostic": any(dataset.diagnostic for dataset in datasets.values()),
            "partial_dataset": any(dataset.partial_dataset for dataset in datasets.values()),
            "limits": "Canonical origin IDs must be preserved across renamed/cropped derivatives. This check does not infer source provenance or hash image pixels."}
