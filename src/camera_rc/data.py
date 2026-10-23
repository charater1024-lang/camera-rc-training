"""Validated JSONL datasets; scene-level separation is mandatory."""
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"{path}:{lineno}: nonempty group_id is required")
        if not isinstance(row.get("image"), str):
            raise ValueError(f"{path}:{lineno}: image path required")
        for field in ("image", "target"):
            if isinstance(row.get(field), str):
                p = Path(row[field]).expanduser()
                row[field] = str((p if p.is_absolute() else path.parent / p).resolve())
                if not Path(row[field]).is_file():
                    raise FileNotFoundError(f"{path}:{lineno}: {field}: {row[field]}")
        row.setdefault("dataset_id", "default")
        if not isinstance(row["dataset_id"], str) or not row["dataset_id"].strip():
            raise ValueError(f"{path}:{lineno}: dataset_id must be nonempty text")
        if "origin_id" in row and (not isinstance(row["origin_id"], str) or not row["origin_id"].strip()):
            raise ValueError(f"{path}:{lineno}: origin_id must be a canonical source scene ID")
        if isinstance(row.get("source_image"), str):
            p = Path(row["source_image"]).expanduser()
            row["source_image"] = str((p if p.is_absolute() else path.parent / p).resolve())
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def split_identities(rows):
    """Identities shared by raw and derived datasets; no pixel-content hashing."""
    identities = set()
    for row in rows:
        identities.add(("scene", row.get("dataset_id", "default"), row["group_id"]))
        for key in ("image", "source_image"):
            if row.get(key):
                identities.add(("image", str(Path(row[key]).resolve()).casefold()))
        if row.get("origin_id"):
            identities.add(("origin", row["origin_id"]))
    return identities


def validate_splits(train, val):
    def images(rows):
        paths = [str(Path(r["image"]).resolve()).casefold() for r in rows]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate image in a split; deduplicate before training")
        return set(paths)
    if images(train) & images(val):
        raise ValueError("Train/validation image leakage")
    groups = lambda rows: {(r.get("dataset_id", "default"), r["group_id"]) for r in rows}
    overlap = groups(train) & groups(val)
    if overlap:
        raise ValueError(f"Train/validation scene leakage: {sorted(overlap)[:5]}")
    origins = split_identities(train) & split_identities(val)
    if origins:
        raise ValueError(f"Train/validation source or origin leakage: {sorted(origins)[:5]}")


def load_data_sources(config):
    """Load metadata only, keeping image decoding lazy for 16 GB host RAM.

    Weights are *per-source probability mass*, not per-image weights. A source
    with 100 frames and weight 1 has the same expected sampling share as a
    source with 10,000 frames and weight 1. Sampling is with replacement.
    """
    data = config["data"]
    if "train_sources" in data and "train_manifest" in data:
        raise ValueError("Choose data.train_sources or train_manifest, not both")
    if "val_manifests" in data and "val_manifest" in data:
        raise ValueError("Choose data.val_manifests or val_manifest, not both")
    if "train_sources" in data:
        train_specs = data["train_sources"]
        if not isinstance(data.get("label_space_id"), str) or not data["label_space_id"].strip():
            raise ValueError("Multiple-source training requires explicit data.label_space_id")
    else:
        train_specs = [{"manifest": data["train_manifest"], "weight": 1.0}]
    val_specs = data.get("val_manifests", [data.get("val_manifest")])
    if not isinstance(train_specs, list) or not train_specs or not isinstance(val_specs, list) or not val_specs:
        raise ValueError("Training and validation sources must be nonempty lists")

    def load(specs, training):
        result, names = [], set()
        for index, spec in enumerate(specs):
            spec = {"manifest": spec} if isinstance(spec, str) else spec
            if not isinstance(spec, dict) or not spec.get("manifest"):
                raise ValueError("Each source requires a manifest path")
            path = str(Path(spec["manifest"]).resolve())
            name = spec.get("name", f"{'train' if training else 'val'}_{index}_{Path(path).stem}")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Source names must be nonempty and unique within each split")
            names.add(name)
            weight = spec.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Source weight must be a finite positive number")
            if spec.get("label_space_id", data.get("label_space_id")) != data.get("label_space_id"):
                raise ValueError("Source label_space_id differs; explicitly remap labels before mixing")
            rows = read_manifest(path)
            result.append({"name": name, "manifest": path, "weight": float(weight), "rows": rows})
        return result

    train_sources, val_sources = load(train_specs, True), load(val_specs, False)
    train = [row for source in train_sources for row in source["rows"]]
    val = [row for source in val_sources for row in source["rows"]]
    validate_splits(train, val)
    return train_sources, val_sources


class ManifestDataset(Dataset):
    def __init__(self, rows, config):
        self.rows = rows
        self.task = config["task"]
        self.height, self.width = config["data"].get("image_size", [256, 512])
        if min(self.height, self.width) < 16:
            raise ValueError("image_size must be [height,width], both at least 16")
        self.classes = int(config.get("model", {}).get("num_classes", 3))
        self.horizon = int(config.get("model", {}).get("horizon", 5))
        self.depth_scale = float(config["data"].get("depth_scale", 256.0))
        self.max_depth = float(config["data"].get("max_depth_m", 80.0))
        self.offsets = config["data"].get("future_offsets_s", [0.5 * (k + 1) for k in range(self.horizon)])
        if self.task == "trajectory" and (len(self.offsets) != self.horizon or
                not all(math.isfinite(v) and v > 0 for v in self.offsets) or self.offsets != sorted(set(self.offsets))):
            raise ValueError("future_offsets_s must be increasing positive seconds, matching model.horizon")
        if self.depth_scale <= 0 or self.max_depth <= 0:
            raise ValueError("depth_scale and max_depth_m must be positive")
        if self.task not in {"segmentation", "depth", "trajectory"}:
            raise ValueError(f"Unsupported task: {self.task}")
        for row in rows:
            if "target" not in row:
                raise ValueError(f"target required for {row['image']}")
            if row.get("task", self.task) != self.task:
                raise ValueError("Manifest task does not match config")
            if row.get("label_space_id", config["data"].get("label_space_id")) != config["data"].get("label_space_id"):
                raise ValueError("Manifest label_space_id differs from configuration")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as raw:
            original_size = raw.size
            rgb = raw.convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(rgb, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        result = {"image": image}
        if self.task == "trajectory":
            offsets = row.get("future_offsets_s")
            if offsets is None or len(offsets) != self.horizon or not np.allclose(offsets, self.offsets):
                raise ValueError("Trajectory future_offsets_s missing or inconsistent with config")
            state = np.asarray(row.get("state"), dtype=np.float32)
            target = np.asarray(row["target"], dtype=np.float32)
            if state.shape != (2,) or target.shape != (self.horizon, 3):
                raise ValueError(f"Expected state[2] and target[{self.horizon},3]: {row['image']}")
            if not np.isfinite(state).all() or not np.isfinite(target).all():
                raise ValueError("Trajectory/state contains NaN or infinity")
            if state[0] < 0 or (target[:, 2] < 0).any():
                raise ValueError("This baseline supports nonnegative forward speed only")
            result.update(state=torch.from_numpy(state.copy()), target=torch.from_numpy(target.copy()))
        else:
            if not isinstance(row["target"], str):
                raise ValueError("Segmentation/depth target must be an image file path")
            with Image.open(row["target"]) as raw:
                if raw.size != original_size:
                    raise ValueError(f"Image/target alignment mismatch: {row['image']}")
                original = np.array(raw)
                if original.ndim != 2:
                    raise ValueError("Target must be a single-channel ID mask or depth PNG")
                if self.task == "depth" and original.dtype == np.uint8:
                    raise ValueError("Depth requires 16-bit (or integer) PNG, not an 8-bit visualization")
                resized = np.array(raw.resize((self.width, self.height), Image.Resampling.NEAREST))
            if self.task == "segmentation":
                # Validate before resizing, so small invalid regions are not silently dropped.
                bad = (original != 255) & ((original < 0) | (original >= self.classes))
                if bad.any():
                    raise ValueError(f"Unexpected class IDs in {row['target']}; use a declared label mapping")
                if not (resized != 255).any():
                    raise ValueError(f"No supervised pixels after resizing: {row['target']}")
                result["target"] = torch.from_numpy(resized.astype(np.int64, copy=True))
            else:
                depth = resized.astype(np.float32) / self.depth_scale
                depth[(depth <= 0) | (depth > self.max_depth) | ~np.isfinite(depth)] = 0
                if not (depth > 0).any():
                    raise ValueError(f"No valid depth after resizing: {row['target']}")
                result["target"] = torch.from_numpy(depth.copy()).unsqueeze(0)
        return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(config):
    train_sources, val_sources = load_data_sources(config)
    train = [row for source in train_sources for row in source["rows"]]
    val = [row for source in val_sources for row in source["rows"]]
    train_ds, val_ds = ManifestDataset(train, config), ManifestDataset(val, config)
    opts = config.get("training", {})
    batch_size = int(opts.get("batch_size", 2))
    workers = int(opts.get("num_workers", 0))
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size>=1 and num_workers>=0 required")
    use_cuda = str(opts.get("device", "auto")) != "cpu" and torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=use_cuda,
                  worker_init_fn=seed_worker, persistent_workers=workers > 0)
    generator = torch.Generator().manual_seed(int(opts.get("seed", 42)))
    sampler = None
    if "train_sources" in config["data"]:
        sample_count = opts.get("samples_per_epoch", len(train))
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("training.samples_per_epoch must be a positive integer")
        weights = [source["weight"] / len(source["rows"]) for source in train_sources for _ in source["rows"]]
        sampler = WeightedRandomSampler(weights, sample_count, replacement=True, generator=generator)
    return (DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, generator=generator, **common),
            DataLoader(val_ds, shuffle=False, **common))


def build_validation_loaders(config):
    """Separate heldout-source metrics so a large dataset cannot hide regression."""
    _, sources = load_data_sources(config)
    opts = config.get("training", {})
    use_cuda = str(opts.get("device", "auto")) != "cpu" and torch.cuda.is_available()
    return {source["name"]: DataLoader(ManifestDataset(source["rows"], config),
            batch_size=opts.get("batch_size", 2), num_workers=0,
            shuffle=False, pin_memory=use_cuda) for source in sources}


def remap_masks(manifest, mapping_config, output, *, workers=0, progress=None):
    """Explicit segmentation-ID conversion; every undeclared ID becomes 255.

    mapping_config is JSON with label_space_id, num_classes, and mapping. Source
    images/scenes/origins are retained exactly; only new masks are written.
    workers=0 preserves sequential processing; 1..8 uses bounded threads.
    progress(done, total), when supplied, runs on the calling thread in input
    order. At most 2*workers conversion futures are retained at any time.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 8:
        raise ValueError("workers must be an integer from 0 to 8")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    manifest, mapping_config, output = Path(manifest).resolve(), Path(mapping_config).resolve(), Path(output).resolve()
    spec = json.loads(mapping_config.read_text(encoding="utf-8-sig"))
    space, classes, raw_mapping = spec.get("label_space_id"), spec.get("num_classes"), spec.get("mapping")
    if not isinstance(space, str) or not space.strip():
        raise ValueError("Mapping requires explicit label_space_id")
    if isinstance(classes, bool) or not isinstance(classes, int) or not 1 <= classes <= 255:
        raise ValueError("Mapping num_classes must be 1..255 (255 remains ignore)")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("Mapping requires a nonempty source-ID to target-ID mapping")
    mapping = {}
    for source, target in raw_mapping.items():
        if not isinstance(source, str) or not source.isdecimal() or not 0 <= int(source) <= 65535:
            raise ValueError("Source class IDs must be decimal strings in 0..65535")
        if isinstance(target, bool) or not isinstance(target, int) or target != 255 and not 0 <= target < classes:
            raise ValueError("Mapped target class must be 0..num_classes-1 or 255 (ignore)")
        if int(source) in mapping:
            raise ValueError("Duplicate numeric source class ID")
        mapping[int(source)] = target
    if 255 in mapping and mapping[255] != 255:
        raise ValueError("Source ignore ID 255 cannot become a supervised class")
    rows = read_manifest(manifest)
    if any(row.get("task", "segmentation") != "segmentation" for row in rows):
        raise ValueError("Mask remapping accepts segmentation manifests only")
    if output.exists():
        raise FileExistsError(f"Remapping output exists: {output}")
    output.mkdir(parents=True)
    pending = output / ".incomplete"
    pending.write_text("Conversion is incomplete until report.json and manifest.jsonl are committed.\n", encoding="utf-8")
    masks = output / "masks"
    masks.mkdir()
    remapped_rows, source_counts, target_counts, unknown_counts = [], Counter(), Counter(), Counter()
    def convert(index, original_row):
        row = dict(original_row)
        local_source, local_target, local_unknown = Counter(), Counter(), Counter()
        if not isinstance(row.get("target"), str):
            raise ValueError("Mask remapping requires target image files")
        with Image.open(row["target"]) as mask, Image.open(row["image"]) as image:
            pixels = np.asarray(mask)
            if pixels.ndim != 2 or not np.issubdtype(pixels.dtype, np.integer):
                raise ValueError("Use single-channel integer-ID masks, not color visualizations")
            if mask.size != image.size:
                raise ValueError(f"Image/target alignment mismatch: {row['image']}")
            if (pixels < 0).any() or (pixels > 65535).any():
                raise ValueError("Source mask IDs must be in 0..65535")
            converted = np.full(pixels.shape, 255, np.uint8)
            values, counts = np.unique(pixels, return_counts=True)
            for value, count in zip(values, counts):
                value, count = int(value), int(count)
                local_source[value] += count
                target = mapping.get(value, 255)
                local_target[target] += count
                if value not in mapping:
                    local_unknown[value] += count
                converted[pixels == value] = target
        target_path = masks / f"{index:09d}.png"
        Image.fromarray(converted).save(target_path)
        row.update(target=str(target_path), target_original=row["target"],
                   label_space_id=space, task="segmentation")
        return row, local_source, local_target, local_unknown

    def collect(result):
        row, local_source, local_target, local_unknown = result
        remapped_rows.append(row)
        source_counts.update(local_source)
        target_counts.update(local_target)
        unknown_counts.update(local_unknown)
        if progress is not None:
            progress(len(remapped_rows), len(rows))

    if workers == 0:
        for index, row in enumerate(rows):
            collect(convert(index, row))
    else:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=workers)
        queued = deque()
        next_index = 0
        try:
            while next_index < min(2 * workers, len(rows)):
                queued.append(executor.submit(convert, next_index, rows[next_index]))
                next_index += 1
            while queued:
                # Waiting in input order preserves stable mask names, manifest
                # order and callback order even when later jobs finish first.
                collect(queued.popleft().result())
                if next_index < len(rows):
                    queued.append(executor.submit(convert, next_index, rows[next_index]))
                    next_index += 1
        except BaseException:
            for future in queued:
                future.cancel()
            raise
        finally:
            # Running jobs finish before returning; failed output stays marked
            # incomplete and no ready manifest is published.
            executor.shutdown(wait=True, cancel_futures=True)
    result_manifest = output / "manifest.jsonl"
    temporary = output / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remapped_rows), encoding="utf-8")
    temporary.replace(result_manifest)
    report = {"manifest": str(result_manifest), "samples": len(remapped_rows),
              "source_manifest": str(manifest), "mapping_config": str(mapping_config),
              "mapping": spec, "label_space_id": space, "num_classes": classes,
              "source_pixel_counts": dict(sorted(source_counts.items())),
              "target_pixel_counts": dict(sorted(target_counts.items())),
              "undeclared_ids_mapped_to_ignore": dict(sorted(unknown_counts.items())),
              "unknown_policy": "ignore_255", "complete": True}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.unlink()
    return report


def prepare_cityscapes(left_image_root, gt_fine_root, output):
    """Pair official fine train/val RGB and raw labelIds without filename guesses.

    Roots contain train/CITY and val/CITY. Test data and train_extra are not
    consumed. Consecutive frames/derived labels retain one canonical sequence.
    """
    from .prepare import _commit_paired_splits

    left_image_root, gt_fine_root, output = Path(left_image_root).resolve(), Path(gt_fine_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    splits = {}
    for split in ("train", "val"):
        image_root, label_root = left_image_root / split, gt_fine_root / split
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"Missing official Cityscapes {split} directories")
        rows, used = [], set()
        for image in sorted(image_root.glob("*/*_leftImg8bit.png")):
            stem = image.name.removesuffix("_leftImg8bit.png")
            match = re.fullmatch(r"([A-Za-z]+)_([0-9]{6})_([0-9]{6})", stem)
            if not match or match.group(1) != image.parent.name:
                raise ValueError(f"Unrecognized Cityscapes city/sequence/frame: {image}")
            target = label_root / image.parent.name / f"{stem}_gtFine_labelIds.png"
            if not target.is_file():
                raise FileNotFoundError(f"Missing Cityscapes raw labelIds (trainIds are different): {target}")
            with Image.open(image) as rgb, Image.open(target) as mask:
                if rgb.size != mask.size:
                    raise ValueError(f"Cityscapes image/target alignment mismatch: {image}")
                values = np.asarray(mask)
                if values.ndim != 2 or not np.isin(values, [*range(34), 255]).all():
                    raise ValueError(f"Invalid Cityscapes raw labelIds: {target}")
            group = f"{match.group(1)}_{match.group(2)}"
            rows.append({"image": str(image), "target": str(target), "group_id": group,
                         "dataset_id": "cityscapes_fine", "origin_id": f"cityscapes:{group}",
                         "sample_id": stem, "task": "segmentation"})
            used.add(target.resolve())
        extras = set(label_root.glob("*/*_gtFine_labelIds.png")) - used
        if extras:
            raise ValueError(f"{len(extras)} Cityscapes labelIds masks have no matching RGB image")
        splits[split] = rows
    return _commit_paired_splits(output, splits, "cityscapes_fine")
