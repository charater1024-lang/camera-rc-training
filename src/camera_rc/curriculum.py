"""Audited staged training with global holdouts and explicit weight transfer.

Planning is read-only. Dataset downloads, task conversion and label guesses are
deliberately outside this runner. All paths in the curriculum are relative to
its JSON; paths inside each stage config are relative to that stage config.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .config import load_config
from .data import ManifestDataset, load_data_sources, read_manifest, split_identities, validate_splits
from .training import (_check_compatible, _read_checkpoint, checkpoint_identity,
                       evaluate_sources, run_training, validate_config)


def _resolve(base, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_detection_data(path):
    path = Path(path)
    content = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Non-JSON dataset YAML requires PyYAML (included in the detection extra)") from exc
        data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError("Detection dataset.yaml must be a mapping")
    names = data.get("names")
    if isinstance(names, dict):
        if set(str(k) for k in names) != {str(k) for k in range(len(names))}:
            raise ValueError("Detection names must have consecutive class IDs starting at zero")
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("Detection dataset requires explicit, unique ordered class names")
    root = _resolve(path.parent, data.get("path", "."))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def collect(value):
        if isinstance(value, list):
            return [p for item in value for p in collect(item)]
        if not isinstance(value, str) or "://" in value:
            raise ValueError("Detection splits must refer to local image directories or text lists")
        entry = _resolve(root, value)
        if entry.is_dir():
            return [str(p.resolve()) for p in sorted(entry.rglob("*")) if p.is_file() and p.suffix.lower() in extensions]
        if entry.is_file() and entry.suffix.lower() == ".txt":
            result = []
            for line in entry.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                image = _resolve(entry.parent, line.strip())
                if not image.is_file() or image.suffix.lower() not in extensions:
                    raise FileNotFoundError(f"Detection image list entry: {image}")
                result.append(str(image))
            return result
        raise FileNotFoundError(f"Unsupported/missing detection image directory or text list: {entry}")

    splits = {split: collect(data.get(split)) for split in ("train", "val")}
    for split, images in splits.items():
        if not images or len(images) != len(set(p.casefold() for p in images)):
            raise ValueError(f"Detection {split} is empty or contains duplicate image paths")
    return {"names": names, **splits}


def _detection_stage_data(spec, config, base):
    dataset = _read_detection_data(config["data"])
    manifests, rows = {}, {}
    for split in ("train", "val"):
        values = spec.get(f"{split}_manifests")
        if values is None:
            values = [str(Path(config["data"]).parent / f"{split}.jsonl")]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Detection stage {split}_manifests must be a nonempty list")
        manifests[split] = [str(_resolve(base, value)) for value in values]
        rows[split] = [row for manifest in manifests[split] for row in read_manifest(manifest)]
        actual = {row["image"].casefold() for row in rows[split]}
        expected = {p.casefold() for p in dataset[split]}
        if actual != expected or len(rows[split]) != len(expected):
            raise ValueError(f"Detection {split} scene manifests must describe exactly the YAML split images")
        for row in rows[split]:
            # Ultralytics derives label paths from /images/ -> /labels/. Verify
            # its actual labels, not an unrelated target named in a manifest.
            image = Path(row["image"])
            parts = list(image.parts)
            positions = [i for i, part in enumerate(parts) if part == "images"]
            if positions:
                parts[positions[-1]] = "labels"
            expected_label = Path(*parts).with_suffix(".txt").resolve()
            target = row.get("target")
            if not isinstance(target, str) or Path(target).resolve() != expected_label or not expected_label.is_file():
                raise ValueError(f"Detection manifest target must be the explicit YOLO label file (empty file for a verified negative): {expected_label}")
            for line in expected_label.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError as exc:
                    raise ValueError(f"Malformed YOLO label: {expected_label}") from exc
                if len(values) != 5 or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"YOLO boxes require five finite values: {expected_label}")
                class_id, x, y, w, h = values
                if class_id != int(class_id) or not 0 <= class_id < len(dataset["names"]) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"YOLO class ID or normalized box is invalid: {expected_label}")
    validate_splits(rows["train"], rows["val"])
    return dataset, manifests, rows


def _compatible(current, previous):
    if current["task"] != previous["task"]:
        raise ValueError("Stage initialization cannot cross tasks; use an independent task chain")
    if current["label_space_id"] != previous["label_space_id"]:
        raise ValueError("Stage label_space_id differs; remap target labels before transfer")
    if current["task"] == "detection":
        if current["detection_data"]["names"] != previous["detection_data"]["names"]:
            raise ValueError("Detection class order differs between stages")
        if current["config"].get("model", "yolo11n.pt") != previous["config"].get("model", "yolo11n.pt"):
            raise ValueError("Detection stage model architectures must match; set the same model name in each config")
    else:
        _check_compatible(current["config"], previous["config"])


def _prepare(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document.get("stages"), list) or not document["stages"]:
        raise ValueError("Curriculum requires a nonempty stages list")
    if not isinstance(document.get("output_dir"), str) or not document["output_dir"]:
        raise ValueError("Curriculum output_dir is required")
    output = _resolve(path.parent, document["output_dir"])
    stages, by_name = [], {}
    all_train, all_val = set(), set()
    files = {str(path): _hash(path)}
    for spec in document["stages"]:
        name = spec.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in by_name:
            raise ValueError("Stage names must be unique letters/digits/underscore/hyphen, maximum 64 characters")
        config_path = _resolve(path.parent, spec["config"])
        files[str(config_path)] = _hash(config_path)
        config = load_config(config_path)
        task = config.get("task")
        stage_output = output / "stages" / name
        stage = {"name": name, "task": task, "config_path": str(config_path),
                 "output_dir": str(stage_output), "config": config,
                 "label_space_id": spec.get("label_space_id") if task == "detection" else config.get("data", {}).get("label_space_id")}
        if not isinstance(stage["label_space_id"], str) or not stage["label_space_id"].strip():
            raise ValueError(f"Stage {name} requires explicit {'stage' if task == 'detection' else 'data'}.label_space_id")
        if task == "detection":
            if config.get("resume"):
                raise ValueError("Curriculum detection configs must not set resume; completed-stage continuation uses --resume-run")
            stage["detection_data"], manifests, rows = _detection_stage_data(spec, config, path.parent)
            stage["manifests"] = manifests
            files[config["data"]] = _hash(config["data"])
            config.update(project=str(output / "stages"), name=name, exist_ok=False)
        else:
            validate_config(config)
            sources, heldout = load_data_sources(config)
            rows = {"train": [row for source in sources for row in source["rows"]],
                    "val": [row for source in heldout for row in source["rows"]]}
            # Metadata/semantic validation only; use check-data for a full decode scan.
            ManifestDataset(rows["train"], config)
            ManifestDataset(rows["val"], config)
            stage["manifests"] = {"train": [s["manifest"] for s in sources], "val": [s["manifest"] for s in heldout]}
            stage["sources"] = [{k: v for k, v in source.items() if k != "rows"} | {"samples": len(source["rows"])} for source in sources]
            config.setdefault("training", {})["output_dir"] = str(stage_output)
        for manifest in stage["manifests"]["train"] + stage["manifests"]["val"]:
            files[manifest] = _hash(manifest)
        all_train.update(split_identities(rows["train"]))
        all_val.update(split_identities(rows["val"]))
        stage["samples"] = {split: len(values) for split, values in rows.items()}
        configured_init = config.get("training", {}).pop("init_from", None) if task != "detection" else None
        if configured_init and spec.get("init_from"):
            raise ValueError("Put init_from in the curriculum stage or stage config, not both")
        init = spec.get("init_from", configured_init)
        stage["init_from"] = None
        if init is not None:
            if not isinstance(init, str):
                raise ValueError("init_from must be previous, stage:NAME, or a native checkpoint path")
            if init == "previous" or init.startswith("stage:"):
                previous_name = stages[-1]["name"] if init == "previous" and stages else init.removeprefix("stage:")
                if previous_name not in by_name:
                    raise ValueError(f"Stage {name}: init_from must reference an earlier stage")
                previous = by_name[previous_name]
                _compatible(stage, previous)
                stage["init_from"] = {"stage": previous_name, "checkpoint": str(Path(previous["output_dir"]) / ("weights/best.pt" if task == "detection" else "best.pt"))}
            else:
                checkpoint = _resolve(path.parent, init)
                if task == "detection":
                    raise ValueError("Detection checkpoint initialization requires an earlier audited stage; named pretrained weights belong in config.model")
                stored = _read_checkpoint(checkpoint)
                if not stored["config"].get("data", {}).get("label_space_id"):
                    raise ValueError("External checkpoint lacks explicit label_space_id")
                _check_compatible(config, stored["config"])
                stage["init_from"] = {"checkpoint": str(checkpoint)}
                files[str(checkpoint)] = _hash(checkpoint)
        stages.append(stage)
        by_name[name] = stage
    overlap = all_train & all_val
    if overlap:
        raise ValueError(f"Global cross-stage train/heldout leakage: {sorted(overlap)[:5]}")
    plan = {"format_version": 1, "curriculum": str(path), "output_dir": str(output),
            "file_hashes": files, "stages": stages,
            "validation": {"global_holdouts_checked": True, "image_files_decoded": False,
                           "identity_keys": ["image", "source_image", "dataset_id + group_id", "origin_id (when supplied)"]}}
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return plan


def _native_holdouts(stage, previous_stages, checkpoint):
    """Re-evaluate compatible old validation sources to expose forgetting."""
    report, seen = {}, set()
    for previous in [*previous_stages, stage]:
        try:
            _compatible(stage, previous)
        except ValueError:
            continue
        for manifest in previous["manifests"]["val"]:
            if manifest in seen:
                continue
            seen.add(manifest)
            config = deepcopy(stage["config"])
            config["data"].pop("val_manifest", None)
            config["data"]["val_manifests"] = [{"manifest": manifest, "name": manifest}]
            report.update(evaluate_sources(config, checkpoint))
    return report


def _normalize_detection(stage, output):
    folder = output / "inputs" / stage["name"]
    folder.mkdir(parents=True, exist_ok=True)
    data = {"path": str(folder), "names": stage["detection_data"]["names"]}
    for split in ("train", "val"):
        listing = folder / f"{split}.txt"
        listing.write_text("\n".join(stage["detection_data"][split]) + "\n", encoding="utf-8")
        data[split] = str(listing)
    dataset = folder / "dataset.yaml"
    _write(dataset, data)
    config = deepcopy(stage["config"])
    config["data"] = str(dataset)
    return config


def run_curriculum(path, execute=False, resume_run=False):
    """Plan by default; execute opt-in. Resume skips verified completed stages.

    An interrupted native stage may be resumed from a complete epoch. Partial
    diagnostic or interrupted YOLO stages are left untouched for inspection.
    """
    plan = _prepare(path)
    if not execute:
        return {"mode": "plan", "training_complete": False, **plan}
    output = Path(plan["output_dir"])
    state_path = output / "state.json"
    if output.exists():
        if not resume_run:
            raise FileExistsError(f"Curriculum output exists: {output}; use a new output_dir or --resume-run")
        if not state_path.is_file():
            raise FileExistsError("Existing output has no curriculum state; choose a new output_dir")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Curriculum, stage configs, or manifest files changed; use a new output_dir")
    else:
        if resume_run:
            raise FileNotFoundError("--resume-run requires an existing curriculum state")
        output.mkdir(parents=True)
        state = {"format_version": 1, "plan_sha256": plan["plan_sha256"], "training_complete": False, "stages": []}
        _write(output / "plan.json", plan)
        _write(state_path, state)
    completed = {item["name"]: item for item in state["stages"] if item.get("status") == "complete"}
    for index, stage in enumerate(plan["stages"]):
        name = stage["name"]
        if name in completed:
            artifact = completed[name]["checkpoint"]
            if not Path(artifact["path"]).is_file() or _hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Completed checkpoint changed or is missing: {name}")
            continue
        stage_output = Path(stage["output_dir"])
        record = next((item for item in state["stages"] if item["name"] == name), None)
        if record is None:
            if stage_output.exists():
                raise FileExistsError(f"Untracked stage output: {stage_output}")
            record = {"name": name, "task": stage["task"], "status": "running"}
            state["stages"].append(record)
        elif record.get("status") == "incomplete":
            raise ValueError(f"Stage {name} was a partial/diagnostic run; inspect it and use a new curriculum output")
        record["status"] = "running"
        _write(state_path, state)
        init = stage["init_from"]["checkpoint"] if stage["init_from"] else None
        try:
            if stage["task"] == "detection":
                from .detection import run_detection
                if stage_output.exists():
                    raise ValueError("Interrupted detection stage is preserved; resume manually or use a new curriculum output")
                config = _normalize_detection(stage, output)
                result = run_detection(config, mode="train", checkpoint=init)
                if Path(result["save_dir"]).resolve() != stage_output:
                    raise ValueError("Detection trainer changed output directory unexpectedly")
                best = stage_output / "weights" / "best.pt"
                if not best.is_file():
                    raise FileNotFoundError("Detection training did not produce best.pt")
                result["epochs_requested"] = config.get("epochs", 100)
                csv = stage_output / "results.csv"
                result["epochs_recorded"] = max(0, len(csv.read_text(encoding="utf-8").splitlines()) - 1) if csv.is_file() else None
                # A checkpoint alone does not prove every requested epoch ran.
                # Early stopping/time limits remain explicitly incomplete here.
                result["training_complete"] = result["epochs_recorded"] is not None and result["epochs_recorded"] >= result["epochs_requested"]
                holdouts = {}
                for previous in plan["stages"][:index + 1]:
                    try:
                        _compatible(stage, previous)
                    except ValueError:
                        continue
                    previous_config = _normalize_detection(previous, output)
                    previous_config.update(project=str(output / "evaluation"), name=f"{name}-on-{previous['name']}")
                    holdouts[previous["name"]] = run_detection(previous_config, mode="val", checkpoint=best)
            else:
                config = deepcopy(stage["config"])
                resume = stage_output / "last.pt" if (stage_output / "last.pt").is_file() else None
                if stage_output.exists() and resume is None:
                    # Empty/interrupted directories are not silently reused.
                    raise ValueError(f"Incomplete stage lacks last.pt: {stage_output}; preserve it and use a new output_dir")
                # A crash may happen after summary.json was committed but before
                # state.json. Verify the summary and retain completed training.
                summary = stage_output / "summary.json"
                recovered = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else None
                if recovered and recovered.get("training_complete"):
                    result = recovered
                    stored = _read_checkpoint(stage_output / "last.pt")
                    _check_compatible(config, stored["config"])
                    if stored["epoch"] + 1 != config["training"].get("epochs", 30):
                        raise ValueError("Recovered checkpoint did not reach requested epochs")
                else:
                    result = run_training(config, resume=resume, init_from=None if resume else init)
                best = Path(result["best_checkpoint"])
                holdouts = _native_holdouts(stage, plan["stages"][:index], best)
            record.update(result=result, checkpoint=checkpoint_identity(best), holdout_metrics=holdouts,
                          status="complete" if result.get("training_complete") else "incomplete")
            if stage["init_from"]:
                record["initialization"] = checkpoint_identity(init)
            _write(state_path, state)
            if record["status"] != "complete":
                return {"mode": "execute", "output_dir": str(output), **state}
            completed[name] = record
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write(state_path, state)
            raise
    state["training_complete"] = len(completed) == len(plan["stages"])
    state["diagnostic"] = any(item.get("result", {}).get("diagnostic", False) for item in state["stages"])
    _write(state_path, state)
    _write(output / "report.json", state)
    return {"mode": "execute", "output_dir": str(output), **state}
