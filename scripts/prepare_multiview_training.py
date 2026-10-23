"""Validate reviewed four-camera masks and create leak-resistant manifests.

Images remain on disk and are decoded one at a time. No labels, train/val split,
camera geometry, or synchronized correspondences are inferred by this tool.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ("front", "rear", "left", "right")
SPLITS = ("train", "val")
LABEL_SPACE = "road_surface_v1"
TRAINING_SIZE = (320, 192)  # PIL uses width, height.


def _text(row, key, location):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{location}: {key} must be nonempty text without surrounding whitespace")
    return value


def _source_path(row, key, root, location):
    path = Path(_text(row, key, location)).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{location}: {key}: {path}")
    return path


def prepare_multiview(manifest, output, *, dataset_id="vehicle_multiview"):
    """Create train/val manifests for every direction from an explicit split.

    clip_id must identify the entire original drive/scene, consistently across
    cameras, derivatives, and future dataset revisions. Dataset names must not
    be changed to evade existing curriculum scene-leakage checks.
    """
    source, output = Path(manifest).resolve(), Path(output).resolve()
    if not isinstance(dataset_id, str) or not dataset_id.strip() or dataset_id != dataset_id.strip():
        raise ValueError("dataset_id must be nonempty text without surrounding whitespace")
    if output.exists():
        raise FileExistsError(f"Multiview output exists: {output}")
    raw_bytes = source.read_bytes()
    source_text = raw_bytes.decode("utf-8-sig")
    groups = {(split, camera): [] for split in SPLITS for camera in CAMERAS}
    pixels = {key: Counter() for key in groups}
    clip_splits, seen_images = {}, set()
    for lineno, line in enumerate(source_text.splitlines(), 1):
        if not line.strip():
            continue
        location = f"{source}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a JSON object")
        camera, split = _text(row, "camera", location), _text(row, "split", location)
        clip = _text(row, "clip_id", location)
        if camera not in CAMERAS:
            raise ValueError(f"{location}: camera must be one of {CAMERAS}")
        if split not in SPLITS:
            raise ValueError(f"{location}: split must be train or val; no automatic frame split")
        if clip in clip_splits and clip_splits[clip] != split:
            raise ValueError(f"{location}: clip split leakage for {clip!r} across cameras/frames")
        clip_splits[clip] = split
        image = _source_path(row, "image", source.parent, location)
        mask = _source_path(row, "mask", source.parent, location)
        identity = str(image).casefold()
        if identity in seen_images:
            raise ValueError(f"{location}: duplicate image path: {image}")
        seen_images.add(identity)
        # Reject conflicting semantics/provenance instead of silently relabeling.
        if row.get("label_space_id", LABEL_SPACE) != LABEL_SPACE:
            raise ValueError(f"{location}: label_space_id must be {LABEL_SPACE}; remap masks first")
        if row.get("dataset_id", dataset_id) != dataset_id:
            raise ValueError(f"{location}: dataset_id differs from --dataset-id")
        if row.get("group_id", clip) != clip:
            raise ValueError(f"{location}: group_id must equal the original clip_id")
        origin = f"{dataset_id}:{clip}"
        if row.get("origin_id", origin) != origin:
            raise ValueError(f"{location}: origin_id must be {origin!r}")
        with Image.open(image) as rgb, Image.open(mask) as labels:
            rgb.convert("RGB").load()
            if labels.size != rgb.size:
                raise ValueError(f"{location}: image/mask alignment mismatch")
            ids = np.asarray(labels)
            if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError(f"{location}: use single-channel integer-ID masks, not color masks")
            values, counts = np.unique(ids, return_counts=True)
            if not np.isin(values, [0, 1, 255]).all():
                raise ValueError(f"{location}: mask IDs must be 0=road, 1=other, 255=ignore")
            resized = np.asarray(labels.resize(TRAINING_SIZE, Image.Resampling.NEAREST))
            if not (resized != 255).any():
                raise ValueError(f"{location}: no supervised pixels at training resolution 192x320")
            pixels[(split, camera)].update({int(v): int(n) for v, n in zip(values, counts)})
        canonical = {
            "image": str(image), "target": str(mask), "source_image": str(image),
            "task": "segmentation", "label_space_id": LABEL_SPACE,
            "dataset_id": dataset_id, "group_id": clip, "origin_id": origin,
            "camera": camera, "clip_id": clip, "split": split,
        }
        groups[(split, camera)].append(canonical)
    empty = [f"{split}/{camera}" for (split, camera), rows in groups.items() if not rows]
    if empty:
        raise ValueError(f"Every camera needs labeled train and val samples; missing: {', '.join(empty)}")

    # Perform all data validation before creating the output. Failed writes leave
    # an explicit incomplete marker; an existing output is never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    pending = output / ".incomplete"
    pending.write_text("Do not train from this output until report.json says complete.\n", encoding="utf-8")
    reports = {}
    for (split, camera), rows in groups.items():
        path = output / f"{split}_{camera}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)
        reports[f"{split}_{camera}"] = {
            "manifest": str(path), "samples": len(rows),
            "clips": len({row["group_id"] for row in rows}),
            "pixel_counts": dict(sorted(pixels[(split, camera)].items())),
        }
    report = {
        "complete": True, "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "dataset_id": dataset_id, "label_space_id": LABEL_SPACE,
        "cameras": list(CAMERAS), "samples": sum(len(rows) for rows in groups.values()),
        "manifests": reports, "split_policy": "explicit_original_clip_id_across_all_cameras",
        "annotation_policy": "User-supplied reviewed ID masks; quality is not inferred by this tool.",
        "duplicate_check": "Resolved image paths only; renamed copies require preserved clip_id.",
        "model_policy": "Shared per-image segmentation, not joint multi-camera BEV training.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed raw JSONL with explicit clip-level split")
    parser.add_argument("--output", type=Path, required=True, help="New output folder; existing folders are refused")
    parser.add_argument("--dataset-id", default="vehicle_multiview", help="Stable original dataset namespace across all cameras")
    args = parser.parse_args()
    print(json.dumps(prepare_multiview(args.manifest, args.output, dataset_id=args.dataset_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
