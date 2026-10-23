"""Offline, supervised joint image-to-waypoint experiments; never actuates a vehicle."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import JointSequenceDataset, validate_split_manifests
from .model import build_joint_policy, init_encoder


INPUT_KEYS = ("images", "intrinsics", "camera_to_anchor", "time_offsets", "state")
MODEL_TYPE = "joint_camera_temporal_waypoints"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_joint_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for section, keys in (("data", ("train_manifest", "val_manifest", "test_manifest")),
                          ("training", ("output_dir", "init_encoder"))):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                item = Path(value).expanduser()
                cfg[section][key] = str((item if item.is_absolute() else path.parent / item).resolve())
    return cfg


def _integer(value, name, minimum=1, maximum=100000):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def validate_config(config):
    if config.get("task") != "joint_policy":
        raise ValueError("Expected task=joint_policy")
    if config.get("models_validated_for_control", False) is not False:
        raise ValueError("Joint policy is offline research only")
    training, data = config["training"], config["data"]
    for key in ("train_manifest", "val_manifest"):
        if not data.get(key):
            raise ValueError(f"data.{key} is required")
    if not training.get("output_dir"):
        raise ValueError("training.output_dir is required")
    for key, default, upper in (("epochs", 1, 1000), ("batch_size", 1, 64), ("grad_accum_steps", 1, 64)):
        _integer(training.get(key, default), key, maximum=upper)
    _integer(training.get("num_workers", 0), "num_workers", minimum=0, maximum=8)
    _integer(training.get("seed", 42), "seed", minimum=0, maximum=2**32-1)
    for key, default in (("lr", .0001), ("clip_grad_norm", 1.), ("target_scale", 10.), ("amp_init_scale", 1024.)):
        _positive(training.get(key, default), key)
    if training.get("max_steps") is not None:
        _integer(training["max_steps"], "max_steps")
    if training.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    for key in ("amp",):
        if key in training and not isinstance(training[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(config.get("diagnostic_only", False), bool):
        raise ValueError("diagnostic_only must be boolean")


def make_dataset(config, split):
    data = config["data"]
    return JointSequenceDataset(data[f"{split}_manifest"], image_size=data["image_size"],
        expected_history=data["expected_history"], expected_views=data["expected_views"],
        expected_future_dt_s=data["future_dt_s"])


def _device(config):
    requested = config["training"].get("device", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly for diagnostics")
    return torch.device("cuda" if torch.cuda.is_available() and requested != "cpu" else "cpu")


def _forward(model, batch, device):
    return model(**{key: batch[key].to(device, non_blocking=True) for key in INPUT_KEYS})


def _metric_values(prediction, target):
    distances = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    return {"ade_m": distances.mean(-1), "fde_m": distances[:, -1],
            "speed_mae_mps_nominal": (prediction[..., 2] - target[..., 2]).abs().mean(-1)}


@torch.inference_mode()
def evaluate_joint_policy(model, dataset, *, device="cpu", batch_size=1, num_workers=0):
    """Full supplied validation split; test split is never used here by training."""
    device = torch.device(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_model, all_cv, groups = [], [], {}
    offset = 0
    for batch in loader:
        prediction = _forward(model, batch, device).float().cpu()
        target = batch["targets"].float()
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError("Invalid policy prediction")
        # An honest state-only comparison: keep current speed and heading.
        baseline = torch.zeros_like(target)
        baseline[..., 0] = batch["state"][:, 0, None] * batch["future_dt_s"]
        baseline[..., 2] = batch["state"][:, 0, None]
        current, cv = _metric_values(prediction, target), _metric_values(baseline, target)
        for index in range(len(target)):
            a = {name: float(value[index]) for name, value in current.items()}
            b = {name: float(value[index]) for name, value in cv.items()}
            row = dataset.rows[offset + index]
            group = str(row.get("origin_id", "unknown")) + "/" + str(row.get("condition", row.get("variant", "unspecified")))
            groups.setdefault(group, [[], []])[0].append(a)
            groups[group][1].append(b)
            all_model.append(a)
            all_cv.append(b)
        offset += len(target)
    def mean(rows):
        if not rows:
            raise ValueError("Validation split is empty")
        return {"samples": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}}
    return {"model": mean(all_model), "constant_velocity_baseline": mean(all_cv),
            "by_origin_condition": {key: {"model": mean(value[0]), "constant_velocity_baseline": mean(value[1])}
                                    for key, value in sorted(groups.items())},
            "partial_validation": False, "open_loop_only": True, "collision_rate_measured": False,
            "timing_notice": "Speed/seconds inherit the manifest timing. VKITTI nominal 0.1s is an assumption, not measured exposure timing.",
            "limits": "ADE/FDE compare recorded paths; they do not measure collision avoidance, alternate safe paths, traffic-rule compliance or closed-loop stability."}


def train_joint_policy(config):
    config = copy.deepcopy(config)
    validate_config(config)
    training, data = config["training"], config["data"]
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        if data.get(key):
            manifest = Path(data[key]).resolve()
            if any((parent / marker).exists() for parent in manifest.parents for marker in (".incomplete", "INCOMPLETE")):
                raise ValueError("A manifest belongs to an incomplete preparation")
    split_check = validate_split_manifests(data["train_manifest"], data["val_manifest"], data.get("test_manifest"),
        image_size=data["image_size"], expected_history=data["expected_history"],
        expected_views=data["expected_views"], expected_future_dt_s=data["future_dt_s"])
    train_data, val_data = make_dataset(config, "train"), make_dataset(config, "val")
    loaded_datasets = {"train": train_data, "val": val_data}
    provenance = {}
    for split in ("train", "val", "test"):
        if not data.get(f"{split}_manifest"):
            continue
        expected_hash = split_check["manifests"][split]["sha256"]
        if (split in loaded_datasets and loaded_datasets[split].manifest_sha256 != expected_hash
                or digest(data[f"{split}_manifest"]) != expected_hash):
            raise ValueError("Manifest changed between split validation and dataset loading")
        provenance[split] = {"path": str(Path(data[f"{split}_manifest"]).resolve()), "sha256": expected_hash}
    diagnostic = bool(config.get("diagnostic_only", False) or train_data.diagnostic or val_data.diagnostic
                      or training.get("max_steps") is not None)
    config["diagnostic_only"] = diagnostic
    seed = training.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(config)
    output = Path(training["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Training is incomplete. This policy is never approved for vehicle control.\n", encoding="utf-8")
    write_json(output / "config.json", config)
    write_json(output / "input_provenance.json", {"manifests": provenance, "split_check": split_check,
                "source_image_hashes_verified_at_load": all(ds.provenance.get("source_image_hashes_verified", False) for ds in loaded_datasets.values()),
                "notice": "Declared image hashes are verified at dataset construction. Keep source image bytes immutable during training; pixels are not rehashed on every epoch."})
    model = build_joint_policy(config)
    initialization = init_encoder(model, training["init_encoder"], allow_diagnostic=diagnostic) if training.get("init_encoder") else None
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_parameters = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
    batch_size, workers = training.get("batch_size", 1), training.get("num_workers", 0)
    before = evaluate_joint_policy(model, val_data, device=device, batch_size=batch_size, num_workers=workers)
    write_json(output / "validation_before.json", before)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.get("lr", .0001), weight_decay=.0001)
    amp = device.type == "cuda" and training.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=training.get("amp_init_scale", 1024.))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                        generator=generator, pin_memory=device.type == "cuda")
    accumulated = training.get("grad_accum_steps", 1)
    requested_epochs = training.get("epochs", 1)
    steps = 0
    best = math.inf
    epoch_logs = []
    started = time.perf_counter()
    stop = False
    for epoch in range(1, requested_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen, epoch_steps = 0., 0, 0
        for batch_index, batch in enumerate(loader):
            target = batch["targets"].to(device)
            # Divide the final incomplete accumulation group by its real size.
            group_start = (batch_index // accumulated) * accumulated
            group_size = min(accumulated, len(loader) - group_start)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = _forward(model, batch, device)
                scale = training.get("target_scale", 10.)
                loss = F.smooth_l1_loss(prediction.float() / scale, target.float() / scale)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy loss")
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
            if (batch_index + 1) % accumulated == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("clip_grad_norm", 1.), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                epoch_steps += 1
                if training.get("max_steps") is not None and steps >= training["max_steps"]:
                    stop = True
                    break
        if epoch_steps == 0:
            raise RuntimeError("No optimizer updates were applied")
        validation = evaluate_joint_policy(model, val_data, device=device, batch_size=batch_size, num_workers=workers)
        score = validation["model"]["ade_m"]
        payload = {"format_version": 1, "model_type": MODEL_TYPE, "config": config,
                   "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                   "diagnostic": diagnostic, "optimizer_steps": steps, "epoch": epoch,
                   "initialization": initialization, "input_manifests": provenance,
                   "models_validated_for_control": False, "training_complete": False}
        torch.save(payload, output / "last.pt")
        if score < best:
            best = score
            torch.save(payload, output / "best.pt")
            write_json(output / "best_validation.json", validation)
        record = {"epoch": epoch, "samples": seen, "loss": total_loss / seen, "optimizer_steps": steps,
                  "validation_ade_m": score, "baseline_ade_m": validation["constant_velocity_baseline"]["ade_m"],
                  "diagnostic": diagnostic, "seconds": time.perf_counter() - started}
        epoch_logs.append(record)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if stop:
            break
    # Re-read manifest hashes before publishing successful completion.
    for identity in provenance.values():
        if digest(identity["path"]) != identity["sha256"]:
            raise ValueError("Manifest changed during training")
    changed = sum(not torch.equal(initial_parameters[key], value.detach().cpu()) for key, value in model.named_parameters())
    if changed == 0:
        raise RuntimeError("No parameter tensors changed")
    for filename in ("best.pt", "last.pt"):
        payload = torch.load(output / filename, map_location="cpu", weights_only=True)
        payload["training_complete"] = True
        torch.save(payload, output / filename)
    after = json.loads((output / "best_validation.json").read_text(encoding="utf-8"))
    summary = {"training_complete": True, "full_requested_epochs_completed": not stop,
               "diagnostic": diagnostic, "best_checkpoint": str(output / "best.pt"),
               "checkpoint_sha256": digest(output / "best.pt"), "initialization": initialization,
               "optimizer_steps": steps, "changed_parameter_tensors": changed,
               "requested_epochs": requested_epochs, "completed_epochs": len(epoch_logs),
               "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
               "parameter_count": sum(value.numel() for value in model.parameters()),
               "train_samples": len(train_data), "validation_samples": len(val_data),
               "dataset_provenance": {"train": train_data.provenance, "val": val_data.provenance},
               "partial_dataset": train_data.partial_dataset or val_data.partial_dataset,
               "trained_views": data["expected_views"], "history_frames": data["expected_history"],
               "validation_before": before, "validation_after": after,
               "better_than_constant_velocity_on_validation": after["model"]["ade_m"] < after["constant_velocity_baseline"]["ade_m"],
               "input_manifests": provenance, "models_validated_for_control": False, "deployed": False,
               "collision_rate_measured": False, "test_split_evaluated": False,
               "limits": "Open-loop image-to-waypoint imitation only. No direct steering/throttle training, real surround driving, collision guarantee, measured MSI latency or deployment approval."}
    write_json(output / "summary.json", summary)
    incomplete.unlink()
    return summary
