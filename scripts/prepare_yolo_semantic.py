"""Export existing road_surface_v1 manifests to Ultralytics semantic PNG datasets.

No downloads, GPU work, model loading or training. Sources are never written.
Images are hardlinked where possible (never edit them), otherwise copied.
Masks are fresh lossless PNGs at native resolution. Completion is explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

from camera_rc.config import load_config
from camera_rc.data import ManifestDataset, load_data_sources

LABEL_SPACE = "road_surface_v1"
CLASS_NAMES = {0: "road_surface", 1: "not_road_surface"}
CLASS_MAPPING = {0: 0, 1: 1, 255: 255}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_path(value, manifest):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _check_mapping(row):
    # Native ManifestDataset reads prepared targets; it does not apply label_map.
    mapping = row.get("label_map")
    if mapping is None:
        return
    if not isinstance(mapping, dict):
        raise ValueError("Prepare raw labels with remap_masks before exporting")
    try:
        parsed = {int(k): int(v) for k, v in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid row.label_map") from exc
    if any(k not in CLASS_MAPPING or v != k for k, v in parsed.items()):
        raise ValueError("Native ManifestDataset does not apply row.label_map; remap raw labels first")


def _image_info(path):
    with Image.open(path) as image:
        if image.getexif().get(274, 1) not in (None, 1):
            raise ValueError(f"Nontrivial EXIF rotation requires aligned preprocessing: {path}")
        width, height = image.size
        if min(width, height) < 10:
            raise ValueError(f"Image smaller than Ultralytics minimum: {path}")
        rgb = np.asarray(image.convert("RGB"))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        with path.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            if handle.read(2) != b"\xff\xd9":
                # Avoid downstream Ultralytics repairs modifying a hardlink.
                raise ValueError(f"JPEG missing final EOI; repair a separate copy first: {path}")
    digest = hashlib.sha256(f"RGB:{width}:{height}:".encode("ascii"))
    digest.update(memoryview(rgb))
    return width, height, digest.hexdigest()


def _mask(path, size):
    with Image.open(path) as image:
        values = np.asarray(image)
        if image.size != size:
            raise ValueError(f"Image/target dimensions differ: {path}")
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Use integer class-ID masks, not RGB visualizations: {path}")
    # Do not sort a full-resolution mask just to count three canonical IDs.
    # Check before the uint8 cast so 16/32-bit unknown IDs cannot wrap to road.
    if int(values.min()) < 0 or int(values.max()) > 255:
        unknown = [int(v) for v in np.unique(values) if int(v) not in CLASS_MAPPING]
        raise ValueError(f"Unexpected canonical class IDs {unknown}: {path}")
    values = values.astype(np.uint8, copy=False)
    counts = np.bincount(values.reshape(-1), minlength=256)
    unknown = np.flatnonzero(counts[2:255]) + 2
    if len(unknown):
        raise ValueError(f"Unexpected canonical class IDs {unknown.tolist()}: {path}")
    if counts[0] + counts[1] == 0:
        raise ValueError(f"Target has no supervised pixels: {path}")
    return values, {str(v): int(counts[v]) for v in CLASS_MAPPING if counts[v]}


def _decode_image_and_mask(image_path, target_path):
    """Pure bounded worker work; hashing, source checks and writes stay ordered."""
    width, height, rgb_hash = _image_info(image_path)
    values, pixel_counts = _mask(target_path, (width, height))
    return width, height, rgb_hash, values, pixel_counts


@contextmanager
def _ordered_preprocessed(items, workers, prepare_input):
    """Yield ordered results with at most 2*workers submitted tasks in memory.

    prepare_input runs on the consuming thread before decoding starts, retaining
    the original fingerprint-before-decode order. Closing this context cancels
    pending work and waits for running readers even if the consumer raises.
    """
    if workers == 1:
        def serial():
            for item in items:
                prepared = prepare_input(item)
                row = item[2]
                yield item, prepared, _decode_image_and_mask(Path(row["image"]), Path(row["target"]))
        yield serial()
        return
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="semantic-export")
    pending, iterator = deque(), iter(items)

    def submit_next():
        try:
            item = next(iterator)
        except StopIteration:
            return False
        prepared = prepare_input(item)
        row = item[2]
        future = executor.submit(_decode_image_and_mask, Path(row["image"]), Path(row["target"]))
        pending.append((item, prepared, future))
        return True

    def results():
        for _ in range(workers * 2):
            if not submit_next():
                break
        while pending:
            item, prepared, future = pending.popleft()
            yield item, prepared, future.result()
            submit_next()

    try:
        yield results()
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _copy_or_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def prepare(config_path, output, *, limit_train_per_source=None, limit_val_per_source=None, workers=4):
    """Export all rows. Optional first-N-per-dataset_id limits are diagnostics only.

    Full input split identity checks precede limiting. Source sampling weights
    are recorded but an image-list export visits each selected frame once.
    """
    config_path, output = Path(config_path).expanduser().resolve(), Path(output).expanduser().resolve()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    for name, value in (("limit_train_per_source", limit_train_per_source),
                        ("limit_val_per_source", limit_val_per_source)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer or omitted")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("Incomplete export. Do not train from this folder.\n", encoding="utf-8")
    report = {
        "complete": False, "format": "camera_rc_ultralytics_semantic_v1",
        "output": str(output), "config": str(config_path), "label_space_id": LABEL_SPACE,
        "names": CLASS_NAMES, "source_class_id_to_export_class_id": CLASS_MAPPING,
        "diagnostic_only": limit_train_per_source is not None or limit_val_per_source is not None,
        "limits_per_dataset_id": {"train": limit_train_per_source, "val": limit_val_per_source},
        "source_files_modified": False, "training_started": False, "gpu_used": False,
        "mask_resolution": "native source image dimensions; no resize",
        "image_storage": "hardlink where possible, otherwise identical copy; never edit exported images",
        "input_target_encoding": "prepared road_surface_v1, not target_original raw labels",
        "subset_policy": "Limits select first N rows per dataset_id in source order, for integration diagnostics only.",
        "error": None,
        "preprocessing_workers": workers,
        "preprocessing_max_pending": workers * 2 if workers > 1 else 1,
    }
    _write(output / "report.json", report)
    hashes, original_stats = {}, {}
    storage = Counter()

    def fingerprint(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = str(path)
        if key not in hashes:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            signature = (before.st_size, before.st_mtime_ns)
            if signature != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input changed while hashing: {path}")
            hashes[key], original_stats[key] = digest, signature
        return hashes[key]

    def prepare_input(item):
        row = item[2]
        image_path, target_path = Path(row["image"]), Path(row["target"])
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {image_path}")
        image_hash, target_hash = fingerprint(image_path), fingerprint(target_path)
        if row.get("image_sha256") and row["image_sha256"] != image_hash:
            raise ValueError(f"Image SHA256 differs from source manifest: {image_path}")
        return image_hash, target_hash

    try:
        report["config_sha256"] = fingerprint(config_path)
        report["source_config"] = {"path": str(config_path), "sha256": report["config_sha256"]}
        config = load_config(config_path)
        if config.get("task") != "segmentation" or config.get("model", {}).get("num_classes") != 2:
            raise ValueError("Expected a two-class native segmentation configuration")
        if config.get("data", {}).get("label_space_id") != LABEL_SPACE:
            raise ValueError("Configuration must declare label_space_id=road_surface_v1")
        train_sources, val_sources = load_data_sources(config)
        for source in train_sources + val_sources:
            ManifestDataset(source["rows"], config)  # native metadata validation, no resize or GPU
        splits = {"train": train_sources, "val": val_sources}
        report["native_train_sampling"] = "weighted_replacement" if "train_sources" in config["data"] else "shuffle_each_frame_once"
        report["export_sampling"] = "each selected frame once; source weights are not applied"
        report["sources"] = []
        full_counts, selected_counts, selected = {}, {}, {}
        for split, sources in splits.items():
            limit = limit_train_per_source if split == "train" else limit_val_per_source
            full_counts[split], selected_counts[split], selected[split] = Counter(), Counter(), []
            for source in sources:
                manifest = Path(source["manifest"])
                report["sources"].append({
                    "split": split, "name": source["name"], "manifest": str(manifest),
                    "manifest_sha256": fingerprint(manifest), "samples": len(source["rows"]),
                    "native_weight": source["weight"],
                })
                for index, row in enumerate(source["rows"]):
                    _check_mapping(row)
                    dataset_id = row["dataset_id"]
                    full_counts[split][dataset_id] += 1
                    if limit is None or selected_counts[split][dataset_id] < limit:
                        selected_counts[split][dataset_id] += 1
                        selected[split].append((source, index, row))
        report["full_input_counts"] = {s: dict(v) for s, v in full_counts.items()}
        report["export_counts"] = {s: dict(v) for s, v in selected_counts.items()}
        report["input_split_identity_leakage_checked"] = True
        report["selected_image_byte_and_rgb_hash_leakage_checked"] = False
        _write(output / "report.json", report)
        seen_hashes, seen_ids = {"file": {}, "rgb": {}}, set()
        pixel_totals = {"train": Counter(), "val": Counter()}
        manifests, lists = {}, {}
        for split, items in selected.items():
            with (output / f"{split}.jsonl.incomplete").open("x", encoding="utf-8", newline="\n") as records:
                with ((output / f"{split}.txt.incomplete").open("x", encoding="utf-8", newline="\n") as images,
                      _ordered_preprocessed(items, workers, prepare_input) as preprocessed):
                    for position, (item, prepared, decoded) in enumerate(preprocessed, 1):
                        source, source_index, row = item
                        manifest_path = Path(source["manifest"])
                        image_path, target_path = Path(row["image"]), Path(row["target"])
                        image_hash, target_hash = prepared
                        width, height, rgb_hash, values, pixel_counts = decoded
                        for kind, digest in (("file", image_hash), ("rgb", rgb_hash)):
                            previous = seen_hashes[kind].get(digest)
                            if previous is not None and previous != split:
                                raise ValueError(f"Train/validation image {kind} SHA256 leakage: {image_path}")
                            seen_hashes[kind][digest] = split
                        source_image = _source_path(row.get("source_image", image_path), manifest_path)
                        source_image_hash = fingerprint(source_image)
                        original_target = _source_path(row.get("target_original", target_path), manifest_path)
                        original_target_hash = fingerprint(original_target)
                        if row.get("target_original_sha256") and row["target_original_sha256"] != original_target_hash:
                            raise ValueError(f"Original target SHA256 differs from source manifest: {original_target}")
                        identity = [row["dataset_id"], row["group_id"], row.get("origin_id"),
                                    os.path.normcase(str(image_path.resolve()))]
                        export_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]
                        if export_id in seen_ids:
                            raise ValueError(f"Duplicate export identity: {image_path}")
                        seen_ids.add(export_id)
                        dataset_slug = re.sub(r"[^A-Za-z0-9_-]", "_", row["dataset_id"])[:60] or "dataset"
                        image_out = output / "images" / split / dataset_slug / (export_id + image_path.suffix.lower())
                        mask_out = output / "masks" / split / dataset_slug / (export_id + ".png")
                        method = _copy_or_link(image_path, image_out)
                        storage[method] += 1
                        if method == "copy" and sha256(image_out) != image_hash:
                            raise RuntimeError(f"Image copy differs from source: {image_out}")
                        mask_out.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(values).save(mask_out, format="PNG")
                        exported_row = dict(row)
                        exported_row.update(
                            image=str(image_out), target=str(mask_out), source_image=str(source_image),
                            input_image=str(image_path), input_target=str(target_path), target_original=str(original_target),
                            export_id=export_id, export_split=split, split=split,
                            source_manifest=str(manifest_path), source_manifest_sha256=hashes[str(manifest_path)],
                            source_manifest_index=source_index, source_name=source["name"],
                            image_sha256=image_hash, image_rgb_sha256=rgb_hash,
                            image_rgb_sha256_encoding="RGB:width:height: header plus decoded uint8 RGB bytes",
                            source_image_sha256=source_image_hash, input_target_sha256=target_hash,
                            target_original_sha256=original_target_hash, target_sha256=sha256(mask_out),
                            source_class_ids=sorted(int(v) for v in pixel_counts), target_pixel_counts=pixel_counts,
                            source_target_encoding=LABEL_SPACE, width=width, height=height,
                            label_space_id=LABEL_SPACE, task="segmentation", image_storage=method,
                        )
                        exported_row.pop("label_map", None)
                        records.write(json.dumps(exported_row, ensure_ascii=False, allow_nan=False) + "\n")
                        images.write(image_out.as_posix() + "\n")
                        pixel_totals[split].update(pixel_counts)
                        if position % 500 == 0 or position == len(items):
                            print(json.dumps({"event": "semantic_export", "split": split,
                                              "completed": position, "total": len(items)}), flush=True)
            manifests[split], lists[split] = output / f"{split}.jsonl", output / f"{split}.txt"
        for path, signature in original_stats.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != signature:
                raise RuntimeError(f"Input changed during export: {path}")
        for source in report["sources"]:
            if sha256(source["manifest"]) != source["manifest_sha256"]:
                raise RuntimeError(f"Source manifest changed: {source['manifest']}")
        if sha256(config_path) != report["config_sha256"]:
            raise RuntimeError("Configuration changed during export")
        for split in splits:
            os.replace(output / f"{split}.jsonl.incomplete", manifests[split])
            os.replace(output / f"{split}.txt.incomplete", lists[split])
        yaml_path = output / "dataset.yaml"
        # JSON is a YAML subset with safe Unicode and Windows path quoting.
        _write(output / "dataset.pending.yaml", {
            "path": output.as_posix(), "train": lists["train"].as_posix(), "val": lists["val"].as_posix(),
            "names": CLASS_NAMES, "nc": 2, "masks_dir": "masks",
        })
        report.update(
            complete=True, selected_image_byte_and_rgb_hash_leakage_checked=True,
            image_storage_counts=dict(storage), target_pixel_counts={s: dict(v) for s, v in pixel_totals.items()},
            manifest_outputs={s: {"path": str(p), "sha256": sha256(p)} for s, p in manifests.items()},
            image_lists={s: {"path": str(p), "sha256": sha256(p)} for s, p in lists.items()},
            dataset_yaml=str(yaml_path), dataset_yaml_sha256=sha256(output / "dataset.pending.yaml"),
            source_inputs_fingerprinted=len(hashes),
        )
        _write(output / "report.json", report)
        os.replace(output / "dataset.pending.yaml", yaml_path)
        marker.unlink()
        # Last operation: this binds the completed report and YAML.
        (output / ".complete").write_text(json.dumps({
            "report_sha256": sha256(output / "report.json"), "dataset_yaml_sha256": sha256(yaml_path),
            "diagnostic_only": report["diagnostic_only"],
        }) + "\n", encoding="utf-8")
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}", image_storage_counts=dict(storage))
        marker.write_text("Export failed/interrupted. Do not train from this folder; use a new output.\n", encoding="utf-8")
        _write(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-train-per-source", type=int, help="Diagnostic only: first N per dataset_id")
    parser.add_argument("--limit-val-per-source", type=int, help="Diagnostic only: first N per dataset_id")
    parser.add_argument("--workers", type=int, default=4,
                        help="Image/mask decode threads (default 4); at most 2*workers prefetched rows")
    args = parser.parse_args()
    report = prepare(args.config, args.output, limit_train_per_source=args.limit_train_per_source,
                     limit_val_per_source=args.limit_val_per_source, workers=args.workers)
    print(json.dumps({"complete": report["complete"], "diagnostic_only": report["diagnostic_only"],
                      "dataset_yaml": report["dataset_yaml"], "export_counts": report["export_counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
