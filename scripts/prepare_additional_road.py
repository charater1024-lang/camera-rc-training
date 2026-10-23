"""Prepare checked local road masks without downloading, copying RGB, or training.

CamVid source: alexgkendall/SegNet-Tutorial at CAMVID_COMMIT. Its standard
367/101/233 split shares source videos across splits. This adapter deliberately
uses the fixed, nonstandard whole-video split below: 406 train, 124 validation,
171 test. Test is written separately and never added to training or validation.
Results on this split are not directly comparable with the standard benchmark.

KITTI Road uses its 289 labelled training frames and official raw-drive mapping:
231 training / 58 validation frames, with whole drives held out (constants below).
Its official 290 test frames have no public targets and are never used here.
Use --mapping-root for the separately extracted devkit_road_mapping.zip root;
it must contain training/{um,umm,uu}_mapping.txt. No raw files need to be copied.

Example: python scripts/prepare_additional_road.py --sourcekind camvid
         --root /data/SegNetTutorial/CamVid --output /data/camvid_road_v1
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


LABEL_SPACE = "road_surface_v1"
CAMVID_COMMIT = "bb68b64e72cb004b3ed927fee0f5624dd8097498"
CAMVID_TREE_URL = (
    "https://api.github.com/repos/alexgkendall/SegNet-Tutorial/git/trees/"
    + CAMVID_COMMIT + "?recursive=1"
)
CAMVID_SOURCE_COUNTS = {"train": 367, "val": 101, "test": 233}
CAMVID_VIDEO_COUNTS = {"0016E5": 305, "0006R0": 101, "0001TP": 124, "Seq05VD": 171}
CAMVID_VIDEO_SPLITS = {
    "0016E5": "train", "0006R0": "train", "0001TP": "val", "Seq05VD": "test",
}
CAMVID_SPLIT_COUNTS = {"train": 406, "val": 124, "test": 171}
CAMVID_NAME = re.compile(r"^(0001TP|0006R0|0016E5|Seq05VD)_(?:f)?[0-9]+\.png$")
KITTI_CATEGORY_COUNTS = {"um": 95, "umm": 96, "uu": 98}
KITTI_DRIVE_COUNT = 37
KITTI_VAL_DRIVES = frozenset({
    "2011_09_26_drive_0009", "2011_09_26_drive_0015", "2011_09_26_drive_0056",
    "2011_09_30_drive_0016", "2011_10_03_drive_0042",
})
KITTI_SPLIT_COUNTS = {"train": 231, "val": 58}
KITTI_MAPPING_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road_mapping.zip"


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(folder, root):
    if not folder.is_dir():
        raise FileNotFoundError(f"Required source folder missing: {folder}")
    result = {}
    for path in sorted(folder.iterdir()):
        # The pinned folders contain PNGs only; reject ambiguous additions.
        if not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected source entry: {path}")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Source entry escapes root: {path}")
        if path.name.casefold() in result:
            raise ValueError(f"Duplicate case-insensitive filename: {path}")
        result[path.name.casefold()] = path
    return result


def _camvid_pairs(root):
    pairs, seen, videos = [], set(), Counter()
    for source_split, expected in CAMVID_SOURCE_COUNTS.items():
        images = _files(root / source_split, root)
        labels = _files(root / (source_split + "annot"), root)
        if images.keys() != labels.keys():
            raise ValueError(f"Image/label pairing mismatch in {source_split}")
        if len(images) != expected:
            raise ValueError(f"Expected {expected} source {source_split} pairs, found {len(images)}")
        for name, image in images.items():
            match = CAMVID_NAME.fullmatch(image.name)
            if not match:
                raise ValueError(f"Unknown CamVid source-video filename: {image.name}")
            if name in seen:
                raise ValueError(f"Duplicate image identity across source folders: {image.name}")
            seen.add(name)
            video = match.group(1)
            videos[video] += 1
            pairs.append((image, labels[name], source_split, video, CAMVID_VIDEO_SPLITS[video]))
    if dict(videos) != CAMVID_VIDEO_COUNTS:
        raise ValueError(f"Unexpected source-video counts: {dict(videos)}")
    return pairs


def _camvid_labels(raw):
    values = np.asarray(raw)
    # This release uses grayscale class IDs, including RGB grayscale PNGs.
    # Accept indexed masks as IDs, never convert semantic colors to luminance.
    if values.ndim == 3 and values.shape[2] == 3:
        if not np.array_equal(values[:, :, 0], values[:, :, 1]) or not np.array_equal(values[:, :, 0], values[:, :, 2]):
            raise ValueError("CamVid label must contain class IDs, not semantic RGB colors")
        values = values[:, :, 0]
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("CamVid label must be an 8-bit class-ID mask")
    unique = np.unique(values)
    if np.any(unique > 11):
        raise ValueError(f"Unexpected CamVid class IDs: {unique.tolist()}")
    target = np.ones(values.shape, dtype=np.uint8)
    target[values == 3] = 0  # road
    target[values == 11] = 255  # unlabelled; excluded by original evaluation
    return values, target


def _kitti_pairs(root, mapping_root):
    images = _files(root / "training/image_2", root)
    labels = _files(root / "training/gt_image_2", root)
    pairs, expected_images, expected_road_labels, drives, origins = [], set(), set(), set(), set()
    for category, expected in KITTI_CATEGORY_COUNTS.items():
        mapping = mapping_root / "training" / (category + "_mapping.txt")
        if not mapping.is_file() or mapping_root not in mapping.resolve().parents:
            raise FileNotFoundError(f"Official raw-drive mapping missing: {mapping}")
        lines = mapping.read_text(encoding="utf-8-sig").splitlines()
        if len(lines) != expected:
            raise ValueError(f"Expected {expected} mapping rows for {category}, found {len(lines)}")
        for index, line in enumerate(lines):
            match = re.fullmatch(r"(2011_[0-9]{2}_[0-9]{2}_drive_[0-9]{4})\s+([0-9]{10})", line.strip())
            if not match:
                raise ValueError(f"Invalid KITTI drive mapping: {mapping}:{index + 1}")
            drive, frame = match.groups()
            if (drive, frame) in origins:
                raise ValueError(f"Duplicate KITTI source frame: {drive}/{frame}")
            origins.add((drive, frame))
            drives.add(drive)
            image_name = f"{category}_{index:06d}.png"
            label_name = f"{category}_road_{index:06d}.png"
            expected_images.add(image_name)
            expected_road_labels.add(label_name)
            if image_name not in images or label_name not in labels:
                raise ValueError(f"KITTI image/road-label pairing mismatch: {image_name}")
            split = "val" if drive in KITTI_VAL_DRIVES else "train"
            pairs.append((images[image_name], labels[label_name], "training", drive, split))
    if images.keys() != expected_images:
        raise ValueError("Unexpected KITTI training image names or counts")
    # Lane labels cover only the current lane, unlike the requested road surface.
    # They may be present in the archive, but must never enter these manifests.
    allowed_lane_labels = {f"um_lane_{index:06d}.png" for index in range(KITTI_CATEGORY_COUNTS["um"])}
    if not expected_road_labels <= labels.keys() or labels.keys() - expected_road_labels - allowed_lane_labels:
        raise ValueError("Unexpected KITTI ground-truth files")
    if len(drives) != KITTI_DRIVE_COUNT or not KITTI_VAL_DRIVES <= drives:
        raise ValueError(f"Unexpected KITTI raw-drive groups: {len(drives)}")
    return pairs


def _kitti_labels(raw):
    values = np.asarray(raw)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("KITTI road target must be an RGB uint8 PNG")
    if np.any(values[:, :, 1] != 0) or not np.isin(values[:, :, (0, 2)], [0, 255]).all():
        raise ValueError("Unexpected KITTI RGB ground-truth encoding")
    # Official development-kit README: R is valid area, B is road ground truth.
    valid = values[:, :, 0] == 255
    road = values[:, :, 2] == 255
    target = np.full(values.shape[:2], 255, dtype=np.uint8)
    target[valid & ~road] = 1
    target[valid & road] = 0
    # Counts describe meaningful raw states without conflating RGB channels.
    states = np.where(valid, np.where(road, 2, 1), 0).astype(np.uint8)
    return states, target


def prepare_additional_road(sourcekind, root, output, mapping_root=None):
    """Create only a fresh output; validate every selected input and split.

    File and decoded-RGB hashes catch exact duplicates, including differently
    encoded copies. They cannot detect near duplicates or prove provenance;
    upstream download validation must check the pinned Git blob manifest.
    """
    if sourcekind not in {"camvid", "kitti_road"}:
        raise ValueError("Supported sourcekind: camvid, kitti_road")
    camvid = sourcekind == "camvid"
    if camvid and mapping_root is not None:
        raise ValueError("--mapping-root is only supported for kitti_road")
    dataset_id = "camvid_segnet" if camvid else "kitti_road"
    origin_prefix = "camvid" if camvid else "kitti"
    root = Path(root).expanduser().resolve()
    mapping_root = Path(mapping_root).expanduser().resolve() if mapping_root is not None else root
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from the source directory")
    if not camvid and (mapping_root == output or mapping_root in output.parents or output in mapping_root.parents):
        raise ValueError("Output must be separate from the mapping source directory")
    if not root.is_dir():
        raise FileNotFoundError(f"Source root missing: {root}")
    output.mkdir(parents=True, exist_ok=False)
    pending = output / ".incomplete"
    pending.write_text("Preparation incomplete. Do not train from this directory.\n", encoding="utf-8")
    report = {
        "complete": False, "training_started": False, "deployment_validated": False,
        "sourcekind": sourcekind, "source_root": str(root), "dataset_id": dataset_id,
        "label_space_id": LABEL_SPACE, "original_files_modified": False,
        "expected_source_commit": CAMVID_COMMIT, "expected_source_tree_url": CAMVID_TREE_URL,
        "source_authenticity": "Not certified here; verify downloaded blobs against the pinned Git tree.",
        "split_policy": "Custom whole-video split; not the standard SegNet benchmark split.",
        "video_splits": CAMVID_VIDEO_SPLITS, "test_used_for_training_or_validation": False,
        "mapping": {"3": 0, "0,1,2,4,5,6,7,8,9,10": 1, "11": 255},
        "duplicate_policy": "Reject duplicate image IDs, resolved image paths, file SHA256, or decoded RGB SHA256 across all splits.",
        "image_resize": "None during preparation; native trainer uses bilinear RGB and nearest-neighbor target resize.",
        "evaluation_image_size": [192, 320],
    }
    if not camvid:
        for field in ("expected_source_commit", "expected_source_tree_url", "video_splits"):
            report.pop(field)
        report.update(
            source_authenticity="Not certified here; verify the official archive and mapping during download.",
            source_url="https://www.cvlibs.net/datasets/kitti/eval_road.php",
            mapping_source_url=KITTI_MAPPING_URL,
            mapping_root=str(mapping_root),
            split_policy="Custom whole-drive split of the official labelled training set; no official validation split.",
            validation_drives=sorted(KITTI_VAL_DRIVES),
            mapping={"R=255,B=255": 0, "R=255,B=0": 1, "R=0": 255},
            source_pixel_count_encoding={"0": "ignored (R=0)", "1": "nonroad (R=255,B=0)", "2": "road (R=255,B=255)"},
            official_test="290 unlabelled benchmark frames excluded; no native test manifest is generated.",
            ego_lane_labels_used=False,
        )
    try:
        pairs = _camvid_pairs(root) if camvid else _kitti_pairs(root, mapping_root)
        if not camvid:
            report["mapping_sha256"] = {category: _sha256(mapping_root / "training" / (category + "_mapping.txt")) for category in KITTI_CATEGORY_COUNTS}
        rows = {split: [] for split in (("train", "val", "test") if camvid else ("train", "val"))}
        raw_counts = {split: Counter() for split in rows}
        target_counts = {split: Counter() for split in rows}
        hashes, pixels, paths = {}, {}, {}
        for split in rows:
            (output / "masks" / split).mkdir(parents=True)
        for number, (image_path, label_path, source_split, video, split) in enumerate(pairs, 1):
            identity = str(image_path.resolve()).casefold()
            image_hash = _sha256(image_path)
            label_hash = _sha256(label_path)
            with Image.open(image_path) as image:
                image.load()
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG image: {image_path}")
                rgb = np.asarray(image.convert("RGB"))
                size = image.size
            pixel_hash = hashlib.sha256(str(rgb.shape).encode("ascii") + b"\0" + rgb.tobytes()).hexdigest()
            for table, key, kind in ((paths, identity, "resolved path"), (hashes, image_hash, "file hash"), (pixels, pixel_hash, "decoded RGB hash")):
                if key in table:
                    raise ValueError(f"Duplicate image {kind}: {image_path} and {table[key]}")
                table[key] = str(image_path)
            with Image.open(label_path) as label:
                label.load()
                if label.format != "PNG" or label.size != size:
                    raise ValueError(f"Image/label size or format mismatch: {label_path}")
                original, target = _camvid_labels(label) if camvid else _kitti_labels(label)
                target_image = Image.fromarray(target)
            # Match the native trainer's ignore handling after its 320x192 resize.
            resized = np.asarray(target_image.resize((320, 192), Image.Resampling.NEAREST))
            if not np.any(resized != 255):
                raise ValueError(f"No supervised pixels after resizing: {label_path}")
            converted = output / "masks" / split / image_path.name
            target_image.save(converted)
            with Image.open(converted) as checked:
                checked.load()
                if checked.size != size or not np.array_equal(np.asarray(checked), target):
                    raise ValueError(f"Converted label verification failed: {converted}")
            for counts, values in ((raw_counts[split], original), (target_counts[split], target)):
                ids, frequencies = np.unique(values, return_counts=True)
                counts.update({str(int(index)): int(count) for index, count in zip(ids, frequencies)})
            rows[split].append({
                "image": str(image_path.resolve()), "source_image": str(image_path.resolve()),
                "target": converted.relative_to(output).as_posix(), "target_original": str(label_path.resolve()),
                "task": "segmentation", "label_space_id": LABEL_SPACE,
                "dataset_id": dataset_id, "group_id": f"{origin_prefix}:{video}", "origin_id": f"{origin_prefix}:{video}",
                "source_split": source_split, "split": split, "source_video": video,
                "image_sha256": image_hash, "image_rgb_sha256": pixel_hash, "target_original_sha256": label_hash,
                "width": size[0], "height": size[1],
            })
            if number % 100 == 0 or number == len(pairs):
                print(f"decoded/remapped {number}/{len(pairs)}", flush=True)
        groups = {split: {row["group_id"] for row in values} for split, values in rows.items()}
        for index, first in enumerate(rows):
            for second in list(rows)[index + 1:]:
                if groups[first] & groups[second]:
                    raise ValueError(f"Source-video leakage: {first}/{second}")
        expected_split_counts = CAMVID_SPLIT_COUNTS if camvid else KITTI_SPLIT_COUNTS
        if {split: len(values) for split, values in rows.items()} != expected_split_counts:
            raise ValueError("Unexpected custom split counts")
        report["source_split_counts"] = dict(Counter(row["source_split"] for values in rows.values() for row in values))
        report["manifests"] = {}
        for split, values in rows.items():
            manifest = output / (split + ".jsonl")
            temporary = output / (split + ".jsonl.pending")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(values, key=lambda row: row["image"])), encoding="utf-8")
            report["manifests"][split] = {
                "manifest": str(manifest), "samples": len(values), "groups": sorted(groups[split]),
                "sha256": _sha256(temporary), "all_samples_decoded": True,
                "source_pixel_counts": dict(raw_counts[split]), "target_pixel_counts": dict(target_counts[split]),
            }
        # Publish only after complete-data validation, while keeping the marker
        # present until the final report and all manifests are committed.
        for split in rows:
            (output / (split + ".jsonl.pending")).replace(output / (split + ".jsonl"))
        report.update(complete=True, total_samples=len(pairs), global_holdouts_checked=True,
                      all_samples_decoded=True, exact_image_duplicates=0)
        _write_json(output / "report.json", report)
        pending.unlink()
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        _write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourcekind", required=True, choices=("camvid", "kitti_road"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping-root", type=Path, help="KITTI mapping archive root containing training/*_mapping.txt; defaults to --root")
    args = parser.parse_args()
    print(json.dumps(prepare_additional_road(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
