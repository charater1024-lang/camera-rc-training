"""Validate and prepare Virtual KITTI 2.0.3 RGB/class labels for road training.

Pass the three extracted archive roots containing SceneXX directories. Original
images and camera metadata stay untouched. This is a custom whole-source-drive
split, not an official benchmark split. All weather, view and stereo variants
of one scene stay together. Scene01 shares a source with the existing KITTI
Road validation set and therefore remains validation here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


SOURCE_URL = "https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/"
LABEL_SPACE = "road_surface_v1"
VARIANTS = (
    "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone",
    "fog", "morning", "overcast", "rain", "sunset",
)
CAMERAS = ("Camera_0", "Camera_1")
SCENE_COUNTS = {"Scene01": 447, "Scene02": 233, "Scene06": 270, "Scene18": 339, "Scene20": 837}
SCENE_SPLITS = {"Scene01": "val", "Scene02": "test", "Scene06": "train", "Scene18": "train", "Scene20": "train"}
SOURCE_DRIVES = {
    "Scene01": "2011_09_26_drive_0009", "Scene02": "2011_09_26_drive_0011",
    "Scene06": "2011_09_26_drive_0018", "Scene18": "2011_09_29_drive_0004",
    "Scene20": "2011_10_03_drive_0047",
}
# Mapping research uses file hashes to match original KITTI Tracking to Raw.
# This is an independently published mapping, not a NAVER or KITTI assertion.
MAPPING_REVISION = "0292997c33783c2dce583273b4ff4dd03d5e0c39"
MAPPING_URL = (
    "https://raw.githubusercontent.com/pwais/psegs-kitti-ext/" + MAPPING_REVISION
    + "/assets/bench_to_raw_df/part-00000-636d2fca-c4d7-4597-af62-25537c9d2a4f-c000.gz.parquet"
)
MAPPING_SHA256 = "d0a5a7ebd2b31ee6e03ae79e31c530d50fa0902eecbb761a8dbdb19803196b5e"
EXPECTED_SIZE = (1242, 375)
PALETTE = {
    "Terrain": (210, 0, 200), "Sky": (90, 200, 255), "Tree": (0, 199, 0),
    "Vegetation": (90, 240, 0), "Building": (140, 140, 140), "Road": (100, 60, 100),
    "GuardRail": (250, 100, 255), "TrafficSign": (255, 255, 0),
    "TrafficLight": (200, 200, 0), "Pole": (255, 130, 0), "Misc": (80, 80, 80),
    "Truck": (160, 60, 60), "Car": (255, 127, 80), "Van": (0, 139, 139), "Undefined": (0, 0, 0),
}
METADATA_NAMES = ("colors.txt", "intrinsic.txt", "extrinsic.txt", "info.txt", "bbox.txt", "pose.txt")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _contained(path, root):
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError(f"Source path escapes root: {path}")
    return resolved


def _palette(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or lines[0].split() != ["Category", "r", "g", "b"]:
        raise ValueError(f"Unexpected palette header: {path}")
    actual = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] in actual or any(not re.fullmatch(r"\d{1,3}", value) for value in parts[1:]):
            raise ValueError(f"Invalid or duplicate palette entry: {path}")
        actual[parts[0]] = tuple(int(value) for value in parts[1:])
    if actual != PALETTE:
        raise ValueError(f"Palette differs from the official VKITTI 2 palette: {path}")
    return actual


def _lookup_table():
    lookup = np.full(1 << 24, 254, dtype=np.uint8)
    for name, (red, green, blue) in PALETTE.items():
        lookup[(red << 16) | (green << 8) | blue] = 0 if name == "Road" else 255 if name == "Undefined" else 1
    return lookup


def _labels(raw, lookup):
    values = np.asarray(raw)
    if raw.mode != "RGB" or values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("VKITTI class segmentation must be an RGB uint8 mask")
    packed = values[:, :, 0].astype(np.uint32) << 16
    packed |= values[:, :, 1].astype(np.uint32) << 8
    packed |= values[:, :, 2]
    target = lookup[packed]
    if np.any(target == 254):
        unknown = np.unique(packed[target == 254])[:10]
        colors = [(int(x >> 16), int((x >> 8) & 255), int(x & 255)) for x in unknown]
        raise ValueError(f"Unknown semantic colors: {colors}")
    return target


def _frame_files(folder, root, prefix, suffix, count):
    _contained(folder, root)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {folder}")
    actual = {}
    pattern = re.compile(re.escape(prefix) + r"_([0-9]{5})" + re.escape(suffix))
    for path in sorted(folder.iterdir()):
        match = pattern.fullmatch(path.name)
        if not path.is_file() or match is None:
            raise ValueError(f"Unexpected frame entry: {path}")
        index = int(match.group(1))
        if index in actual:
            raise ValueError(f"Duplicate frame identity: {path}")
        actual[index] = _contained(path, root)
    if set(actual) != set(range(count)):
        raise ValueError(f"Incomplete or extra frame IDs: {folder}; expected {count}, found {len(actual)}")
    return actual


def _inventory(rgb_root, segmentation_root, metadata_root):
    pairs, metadata = [], {}
    for root in (rgb_root, segmentation_root, metadata_root):
        scenes = {path.name for path in root.iterdir() if path.name.startswith("Scene")}
        if scenes != set(SCENE_COUNTS):
            raise ValueError(f"Unexpected scene directories: {root}: {sorted(scenes)}")
        for scene in SCENE_COUNTS:
            scene_path = root / scene
            _contained(scene_path, root)
            if not scene_path.is_dir() or {x.name for x in scene_path.iterdir()} != set(VARIANTS):
                raise ValueError(f"Unexpected or missing scene variations: {scene_path}")
    for scene, count in SCENE_COUNTS.items():
        for variant in VARIANTS:
            info = metadata_root / scene / variant
            metadata_key = f"{scene}/{variant}"
            metadata[metadata_key] = {}
            for name in METADATA_NAMES:
                path = _contained(info / name, metadata_root)
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f"Required camera/source metadata missing: {path}")
                metadata[metadata_key][name] = {"path": str(path), "sha256": _sha256(path)}
            _palette(info / "colors.txt")
            rgb_parent = rgb_root / scene / variant / "frames/rgb"
            # Actual 2.0.3 archive uses camelCase; the website's format example
            # spells classsegmentation lowercase. Accept either, not both.
            frame_root = segmentation_root / scene / variant / "frames"
            candidates = [frame_root / name for name in ("classSegmentation", "classsegmentation") if (frame_root / name).is_dir()]
            # On Windows these names resolve to the same directory.
            if len(candidates) == 2 and candidates[0].samefile(candidates[1]):
                candidates = candidates[:1]
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous class segmentation directory: {frame_root}")
            label_parent = candidates[0]
            for parent in (rgb_parent, label_parent):
                if not parent.is_dir() or {x.name for x in parent.iterdir()} != set(CAMERAS):
                    raise ValueError(f"Unexpected camera directories: {parent}")
            for camera in CAMERAS:
                images = _frame_files(rgb_parent / camera, rgb_root, "rgb", ".jpg", count)
                labels = _frame_files(label_parent / camera, segmentation_root, "classgt", ".png", count)
                for frame in range(count):
                    pairs.append((scene, variant, camera, frame, images[frame], labels[frame]))
    return pairs, metadata


def _check_related(manifests, candidate_splits):
    """Also check earlier stages: pixels can differ while their source is shared."""
    details = []
    for split, paths in manifests.items():
        for source in paths:
            path = Path(source).expanduser().resolve()
            origins = set()
            for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("origin_id"), str) or not row["origin_id"]:
                    raise ValueError(f"Related manifest requires explicit origin_id: {path}:{index}")
                origins.add(row["origin_id"])
            if not origins:
                raise ValueError(f"Related manifest is empty: {path}")
            for other, candidates in candidate_splits.items():
                if other != split and origins & candidates:
                    raise ValueError(f"Cross-dataset source leakage: related {split} versus VKITTI {other}: {sorted(origins & candidates)}")
            details.append({"split": split, "path": str(path), "sha256": _sha256(path), "origin_count": len(origins)})
    return details


def prepare_vkitti2(rgb_root, segmentation_root, metadata_root, output, related_train_manifest=(), related_val_manifest=(), related_test_manifest=()):
    roots = [Path(x).expanduser().resolve() for x in (rgb_root, segmentation_root, metadata_root)]
    rgb_root, segmentation_root, metadata_root = roots
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {output}")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Source root missing: {root}")
        if output == root or output in root.parents or root in output.parents:
            raise ValueError("Output must be separate from all source roots")
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("Preparation incomplete. Do not train from this directory.\n", encoding="utf-8")
    groups = {split: {"kitti:" + SOURCE_DRIVES[scene] for scene in SCENE_SPLITS if SCENE_SPLITS[scene] == split} for split in ("train", "val", "test")}
    report = {
        "complete": False, "dataset_id": "vkitti2_0_3", "label_space_id": LABEL_SPACE,
        "source_url": SOURCE_URL, "license": "CC BY-NC-SA 3.0; Naver Corporation",
        "source_roots": {"rgb": str(rgb_root), "segmentation": str(segmentation_root), "metadata": str(metadata_root)},
        "source_archive_verification": "Separate prerequisite: validate the downloaded 2.0.3 archives with publisher MD5 checksums.",
        "scene_splits": SCENE_SPLITS, "source_drives": SOURCE_DRIVES,
        "source_drive_mapping_evidence": {"url": MAPPING_URL, "sha256": MAPPING_SHA256, "provenance": "Independent KITTI hash-matching study; not a publisher-provided map."},
        "split_policy": "Custom whole-original-drive split; all variants/cameras remain together. Scene01 preserves existing KITTI Road validation holdout.",
        "mapping": {"Road": 0, "other_known_classes": 1, "Undefined": 255},
        "palette": PALETTE, "original_files_modified": False,
        "training_started": False, "deployment_validated": False, "test_used_for_training_or_validation": False,
        "image_resize": "None during preparation; native trainer selects input resolution.",
        "duplicate_policy": "Reject reused paths and exact RGB duplicates across different source drives; count within-drive duplicates without treating them as independent scenes.",
    }
    try:
        report["related_manifests"] = _check_related({"train": related_train_manifest, "val": related_val_manifest, "test": related_test_manifest}, groups)
        pairs, report["metadata_files"] = _inventory(*roots)
        lookup = _lookup_table()
        rows = {split: [] for split in groups}
        counts = {split: Counter() for split in groups}
        file_hashes, pixel_hashes, paths = {}, {}, set()
        duplicates = Counter()
        for number, (scene, variant, camera, frame, image_path, label_path) in enumerate(pairs, 1):
            split = SCENE_SPLITS[scene]
            group = "kitti:" + SOURCE_DRIVES[scene]
            image_hash = _sha256(image_path)
            label_hash = _sha256(label_path)
            identity = str(image_path).casefold()
            if identity in paths:
                raise ValueError(f"Reused source image path: {image_path}")
            paths.add(identity)
            with Image.open(image_path) as image:
                image.load()
                if image.format != "JPEG" or image.mode != "RGB" or image.size != EXPECTED_SIZE:
                    raise ValueError(f"Unexpected RGB image format or dimensions: {image_path}")
                rgb = np.asarray(image)
                pixel_hash = hashlib.sha256(str(rgb.shape).encode("ascii") + b"\0" + rgb.tobytes()).hexdigest()
            for kind, table, digest in (("file", file_hashes, image_hash), ("decoded_rgb", pixel_hashes, pixel_hash)):
                if digest in table:
                    previous_group = table[digest]
                    if previous_group != group:
                        raise ValueError(f"Duplicate image across source drives ({kind}): {image_path}")
                    duplicates[kind] += 1
                table[digest] = group
            with Image.open(label_path) as raw:
                raw.load()
                if raw.format != "PNG" or raw.size != EXPECTED_SIZE:
                    raise ValueError(f"Label format or dimensions mismatch: {label_path}")
                target = _labels(raw, lookup)
            if not np.any(target != 255):
                raise ValueError(f"All pixels ignored: {label_path}")
            target_path = output / "masks" / scene / variant / camera / f"road_{frame:05d}.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target).save(target_path)
            with Image.open(target_path) as checked:
                checked.load()
                if checked.mode != "L" or not np.array_equal(np.asarray(checked), target):
                    raise ValueError(f"Converted mask verification failed: {target_path}")
            frequencies = np.bincount(target.ravel(), minlength=256)
            counts[split].update({str(k): int(frequencies[k]) for k in (0, 1, 255)})
            rows[split].append({
                "image": str(image_path), "source_image": str(image_path),
                "target": target_path.relative_to(output).as_posix(), "target_original": str(label_path),
                "task": "segmentation", "label_space_id": LABEL_SPACE, "dataset_id": "vkitti2_0_3",
                "group_id": group, "origin_id": group, "split": split, "source_split": "unsplit_release",
                "source_scene": scene, "source_video": SOURCE_DRIVES[scene], "variant": variant,
                "camera_id": int(camera.rsplit("_", 1)[1]), "frame_id": frame, "synthetic": True,
                "image_sha256": image_hash, "image_rgb_sha256": pixel_hash,
                "target_original_sha256": label_hash, "width": EXPECTED_SIZE[0], "height": EXPECTED_SIZE[1],
                "camera_metadata_key": f"{scene}/{variant}",
            })
            if number % 500 == 0 or number == len(pairs):
                print(f"decoded/remapped {number}/{len(pairs)}", flush=True)
        report["manifests"] = {}
        for split, samples in rows.items():
            manifest = output / (split + ".jsonl")
            temporary = output / (split + ".jsonl.pending")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
            report["manifests"][split] = {"manifest": str(manifest), "samples": len(samples), "groups": sorted(groups[split]), "sha256": _sha256(temporary), "target_pixel_counts": dict(counts[split])}
        for split in rows:
            (output / (split + ".jsonl.pending")).replace(output / (split + ".jsonl"))
        report.update(complete=True, total_samples=len(pairs), all_samples_decoded=True,
                      source_drive_holdouts_checked=True, within_drive_duplicates=dict(duplicates),
                      cross_drive_exact_duplicates=0, independent_original_scenes=len(SCENE_COUNTS))
        _json(output / "report.json", report)
        marker.unlink()
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        _json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("rgb-root", "segmentation-root", "metadata-root", "output"):
        parser.add_argument("--" + argument, required=True, type=Path)
    for split in ("train", "val", "test"):
        parser.add_argument("--related-" + split + "-manifest", action="append", default=[], type=Path,
                            help="Manifest from another dataset or previous training stage; source origins must not cross splits.")
    args = parser.parse_args()
    report = prepare_vkitti2(**vars(args))
    print(json.dumps({"complete": report["complete"], "total_samples": report["total_samples"], "manifests": report["manifests"]}, indent=2))


if __name__ == "__main__":
    main()
