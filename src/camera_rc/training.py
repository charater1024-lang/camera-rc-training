"""Single-GPU/CPU training with masked objectives and resumable checkpoints.

``max_steps`` caps training minibatches for diagnostics, not optimizer updates.
Partial-epoch checkpoints can be evaluated but cannot be resumed. A completed
epoch checkpoint resumes at the next epoch. No weights are downloaded here.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import random
import time

import torch

from .data import build_dataloaders, build_validation_loaders
from .metrics import MetricAccumulator, task_loss
from .models import build_model


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_config(config: dict) -> None:
    if config.get("task") not in {"segmentation", "depth", "trajectory"}:
        raise ValueError("task must be segmentation, depth, or trajectory")
    training = config.get("training", {})
    for key, default in (("epochs", 30), ("batch_size", 2), ("grad_accum_steps", 1), ("log_every_steps", 50)):
        _positive_integer(training.get(key, default), f"training.{key}")
    lr = training.get("lr", 1e-3)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("training.lr must be finite and positive")
    workers = training.get("num_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("training.num_workers must be a nonnegative integer")
    seed = training.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    for key in ("max_train_steps", "max_val_steps"):
        if training.get(key) is not None:
            _positive_integer(training[key], f"training.{key}")
    class_weights = training.get("segmentation_class_weights")
    if class_weights is not None:
        if config["task"] != "segmentation":
            raise ValueError("training.segmentation_class_weights require task=segmentation")
        if (not isinstance(class_weights, (list, tuple))
                or len(class_weights) != config.get("model", {}).get("num_classes", 3)):
            raise ValueError("training.segmentation_class_weights must contain model.num_classes weights")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in class_weights):
            raise ValueError("training.segmentation_class_weights must be finite positive numbers")
    data = config.get("data", {})
    if "label_space_id" in data and (not isinstance(data["label_space_id"], str) or not data["label_space_id"].strip()):
        raise ValueError("data.label_space_id must be nonempty text defining the target semantics")
    size = data.get("image_size", [256, 512])
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 16 for v in size):
        raise ValueError("data.image_size must be [height, width], each >= 16")
    if not isinstance(training.get("amp", True), bool):
        raise ValueError("training.amp must be true or false")
    amp_scale = training.get("amp_init_scale", 65536.0)
    if isinstance(amp_scale, bool) or not isinstance(amp_scale, (int, float)) or not math.isfinite(amp_scale) or amp_scale <= 0:
        raise ValueError("training.amp_init_scale must be finite and positive")
    _positive_integer(training.get("max_consecutive_amp_skips", 8), "training.max_consecutive_amp_skips")
    for key, default in (("depth_scale", 256.0), ("max_depth_m", 80.0)):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"data.{key} must be finite and positive")
    offsets = data.get("future_offsets_s")
    if offsets is not None:
        if not isinstance(offsets, (list, tuple)) or len(offsets) != config.get("model", {}).get("horizon", 5):
            raise ValueError("data.future_offsets_s must contain model.horizon time offsets")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in offsets):
            raise ValueError("data.future_offsets_s must be finite positive seconds")
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("data.future_offsets_s must be strictly increasing")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("This baseline supports training.device=auto, cpu, cuda or cuda:N")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; install CUDA-enabled PyTorch or use device=cpu")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict, device: torch.device, task: str):
    image = batch["image"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Batch images must have shape [B,3,H,W]")
    state = batch.get("state")
    if task == "trajectory" and state is None:
        raise ValueError("Trajectory batches require current state [speed_mps, yaw_rate_rps]")
    return image, target, state.to(device, non_blocking=True) if state is not None else None


@torch.no_grad()
def _evaluate(model, loader, config, device, max_batches=None) -> dict:
    if len(loader) == 0:
        raise ValueError("Validation loader is empty")
    model.eval()
    metrics = MetricAccumulator(config["task"], config.get("model", {}).get("num_classes", 3))
    loss_sum, samples, batches = 0.0, 0, 0
    enabled = device.type == "cuda" and config.get("training", {}).get("amp", True)
    for batch in loader:
        image, target, state = _move_batch(batch, device, config["task"])
        with torch.autocast(device_type=device.type, enabled=enabled):
            prediction = model(image, state)
            loss = task_loss(config["task"], prediction.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss; inspect images, labels, and numerical settings")
        metrics.update(prediction, target)
        loss_sum += float(loss.item()) * image.shape[0]
        samples += image.shape[0]
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return {"loss": loss_sum / samples, **metrics.compute(), "batches": batches, "partial_validation": batches < len(loader)}


def _generator(loader):
    generator = getattr(loader, "generator", None)
    return generator if generator is not None else getattr(getattr(loader, "sampler", None), "generator", None)


def _check_compatible(config: dict, stored: dict) -> None:
    if config.get("task") != stored.get("task"):
        raise ValueError("Checkpoint task differs from configuration")
    for key, default in (("width", 16), ("num_classes", 3), ("horizon", 5)):
        if config.get("model", {}).get(key, default) != stored.get("model", {}).get(key, default):
            raise ValueError(f"Checkpoint model.{key} differs from configuration")
    for key, default in (("image_size", [256, 512]), ("depth_scale", 256.0), ("max_depth_m", 80.0), ("future_offsets_s", None), ("label_space_id", None)):
        current = config.get("data", {}).get(key, default)
        previous = stored.get("data", {}).get(key, default)
        current = list(current) if isinstance(current, tuple) else current
        previous = list(previous) if isinstance(previous, tuple) else previous
        if current != previous:
            raise ValueError(f"Checkpoint data.{key} differs from configuration; preserve preprocessing and target semantics")


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_checkpoint(path: Path) -> dict:
    # Only ordinary dict/list/number/string/tensor values are written by us.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def checkpoint_identity(path: Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _apply_optimizer_step(model, optimizer, scaler) -> bool:
    """Return whether weights were updated, allowing AMP scale backoff.

    GradScaler records nonfinite gradients during unscale_. Clipping these
    gradients first would raise before its normal skipped-step/backoff path.
    A finite loss is checked by the caller; persistent skips are bounded there.
    """
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("No parameter gradients exist for this optimizer update")
    gradients_finite = bool(torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).all())
    if not gradients_finite:
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite training gradients without AMP; inspect the model and objective")
        previous_scale = scaler.get_scale()
        # step() observes the inf/nan found by unscale_ and does not call
        # optimizer.step(). update() then lowers the dynamic loss scale.
        scaler.step(optimizer)
        scaler.update()
        next_scale = scaler.get_scale()
        if not math.isfinite(next_scale) or not 0 < next_scale < previous_scale:
            raise FloatingPointError("AMP non-finite gradients did not recover to a smaller positive loss scale")
        return False
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def run_training(config: dict, resume: Path | None = None, max_steps: int | None = None,
                 init_from: Path | None = None) -> dict:
    validate_config(config)
    # Convert Path configuration values to strings for portable checkpoints.
    config = json.loads(json.dumps(config, default=str))
    training = config.get("training", {})
    configured_init = training.get("init_from")
    if init_from is not None and configured_init and Path(init_from).resolve() != Path(configured_init).resolve():
        raise ValueError("Conflicting init_from argument and training.init_from")
    init_from = init_from or configured_init
    if resume is not None and init_from is not None:
        raise ValueError("resume and init_from cannot be combined: resume restores optimizer; init_from starts a new stage")
    initialization = None
    initial_checkpoint = None
    diagnostic_provenance = bool(config.get("diagnostic_only", False))
    if init_from is not None:
        initial_checkpoint = _read_checkpoint(init_from)
        if not config.get("data", {}).get("label_space_id") or not initial_checkpoint["config"].get("data", {}).get("label_space_id"):
            raise ValueError("Weights-only init_from requires explicit data.label_space_id in both checkpoints and config")
        _check_compatible(config, initial_checkpoint["config"])
        initialization = checkpoint_identity(init_from)
        diagnostic_provenance = diagnostic_provenance or bool(initial_checkpoint.get("diagnostic", False))
    limit = max_steps if max_steps is not None else training.get("max_train_steps")
    if limit is not None:
        _positive_integer(limit, "max_steps")
    output = Path(training.get("output_dir", "runs/baseline"))
    if resume is not None and Path(resume).resolve().parent != output.resolve():
        raise ValueError("Resume requires training.output_dir to be the checkpoint directory so existing best.pt and metrics remain together")
    if resume is None and ((output / "last.pt").exists() or (output / "metrics.jsonl").exists()):
        raise FileExistsError(f"Existing run at {output}; use resume or a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    _seed(training.get("seed", 42))
    device = _device(training.get("device", "auto"))
    model = build_model(config).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.get("lr", 1e-3))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training.get("amp", True),
                                  init_scale=training.get("amp_init_scale", 65536.0))
    train_loader, val_loader = build_dataloaders(config)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Both train and validation loaders must contain data")
    start_epoch, global_step, optimizer_steps, best_score = 0, 0, 0, -math.inf
    amp_skipped_steps, consecutive_amp_skips = 0, 0
    if resume is not None:
        checkpoint = _read_checkpoint(resume)
        _check_compatible(config, checkpoint["config"])
        if training.get("segmentation_class_weights") != checkpoint["config"].get("training", {}).get("segmentation_class_weights"):
            raise ValueError("Resume cannot change training.segmentation_class_weights; use init_from in a new stage")
        initialization = checkpoint.get("initialization")
        diagnostic_provenance = diagnostic_provenance or bool(checkpoint.get("diagnostic", False))
        if not checkpoint["epoch_complete"]:
            raise ValueError("Diagnostic checkpoint stopped mid-epoch; evaluate it or start a new full training run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # CPU checkpoints carry an empty scaler; switching to CUDA starts a
        # fresh scaler while preserving model/optimizer state.
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        optimizer_steps = checkpoint["optimizer_steps"]
        amp_skipped_steps = checkpoint.get("amp_skipped_steps", 0)
        consecutive_amp_skips = checkpoint.get("consecutive_amp_skips", 0)
        best_score = checkpoint["best_score"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator = _generator(train_loader)
        if generator is not None and checkpoint.get("loader_rng") is not None:
            generator.set_state(checkpoint["loader_rng"])
    epochs = training.get("epochs", 30)
    if start_epoch >= epochs:
        raise ValueError("Checkpoint already reached training.epochs; increase epochs or run evaluation")
    accumulation = training.get("grad_accum_steps", 1)
    enabled = device.type == "cuda" and training.get("amp", True)
    class_weights = training.get("segmentation_class_weights")
    class_weights = (torch.tensor(class_weights, device=device, dtype=torch.float32)
                     if class_weights is not None else None)
    run_steps = 0
    selection_key = {"segmentation": "miou", "depth": "rmse_m", "trajectory": "ade_m"}[config["task"]]
    config_path = output / ("resume_config.json" if resume else "config.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    last_record = {}
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        epoch_optimizer_steps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches_to_run = len(train_loader) if limit is None else min(len(train_loader), limit - run_steps)
        if batches_to_run <= 0:
            break
        loss_sum, samples = 0.0, 0
        for index, batch in enumerate(train_loader):
            if index >= batches_to_run:
                break
            if index % accumulation == 0:
                # Scale by the *actual* final group length, including diagnostics.
                group_size = min(accumulation, batches_to_run - index)
            image, target, state = _move_batch(batch, device, config["task"])
            with torch.autocast(device_type=device.type, enabled=enabled):
                prediction = model(image, state)
                loss = task_loss(config["task"], prediction.float(), target, class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; inspect inputs, learning rate and AMP settings")
            scaler.scale(loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == batches_to_run:
                updated = _apply_optimizer_step(model, optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                if updated:
                    optimizer_steps += 1
                    epoch_optimizer_steps += 1
                    consecutive_amp_skips = 0
                else:
                    amp_skipped_steps += 1
                    consecutive_amp_skips += 1
                    print(json.dumps({"event": "amp_step_skipped", "epoch": epoch + 1,
                                      "batch": index + 1, "amp_scale": scaler.get_scale(),
                                      "amp_skipped_steps": amp_skipped_steps}), flush=True)
                    if consecutive_amp_skips >= training.get("max_consecutive_amp_skips", 8):
                        raise FloatingPointError("Repeated non-finite AMP gradients despite scale backoff; inspect inputs/model or disable AMP")
            loss_sum += float(loss.detach().item()) * image.shape[0]
            samples += image.shape[0]
            global_step += 1
            run_steps += 1
            if (index + 1) % training.get("log_every_steps", 50) == 0:
                print(json.dumps({"event": "batch_progress", "epoch": epoch + 1,
                                  "batch": index + 1, "batches_this_epoch": batches_to_run,
                                  "train_loss": loss_sum / samples, "seconds": time.perf_counter() - started}), flush=True)
        epoch_complete = batches_to_run == len(train_loader)
        if epoch_optimizer_steps == 0:
            raise FloatingPointError("No optimizer updates were applied in this epoch; AMP skipped every update. Lower training.amp_init_scale or disable AMP and retry in a new run")
        val_limit = training.get("max_val_steps")
        if limit is not None:
            val_limit = min(val_limit, limit) if val_limit is not None else limit
        validation = _evaluate(model, val_loader, config, device, val_limit)
        score = validation[selection_key] * (1 if config["task"] == "segmentation" else -1)
        improved = score > best_score
        if improved:
            best_score = score
        generator = _generator(train_loader)
        checkpoint = {
            "format_version": 1, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "epoch_complete": epoch_complete, "global_step": global_step,
            "optimizer_steps": optimizer_steps, "best_score": best_score,
            "amp_skipped_steps": amp_skipped_steps, "consecutive_amp_skips": consecutive_amp_skips,
            "selection_metric": selection_key, "validation": validation,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_rng": generator.get_state() if generator is not None else None,
            "initialization": initialization,
            "diagnostic": diagnostic_provenance or limit is not None or validation["partial_validation"],
        }
        _save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            _save_checkpoint(output / "best.pt", checkpoint)
        last_record = {
            "epoch": epoch + 1, "epoch_complete": epoch_complete, "train_loss": loss_sum / samples,
            "validation": validation, "global_step": global_step, "optimizer_steps": optimizer_steps,
            "epoch_optimizer_steps": epoch_optimizer_steps, "amp_skipped_steps": amp_skipped_steps,
            "amp_scale": scaler.get_scale(),
            "seconds": time.perf_counter() - started, "device": str(device),
            "diagnostic": diagnostic_provenance or limit is not None or validation["partial_validation"],
            "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else None,
        }
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(last_record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(last_record, ensure_ascii=False, allow_nan=False), flush=True)
        if limit is not None and run_steps >= limit:
            break
    result = {
        "output_dir": str(output.resolve()), "last_checkpoint": str((output / "last.pt").resolve()),
        "best_checkpoint": str((output / "best.pt").resolve()) if (output / "best.pt").exists() else None,
        "completed_epochs": last_record["epoch"] if last_record["epoch_complete"] else last_record["epoch"] - 1,
        "requested_epochs": epochs, "run_minibatches": run_steps, "final": last_record,
        "initialization": initialization,
        "diagnostic": last_record["diagnostic"],
        "training_complete": limit is None and not last_record["validation"]["partial_validation"] and last_record["epoch_complete"] and last_record["epoch"] == epochs,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def evaluate_checkpoint(config: dict, checkpoint: Path) -> dict:
    validate_config(config)
    stored = _read_checkpoint(checkpoint)
    _check_compatible(config, stored["config"])
    device = _device(config.get("training", {}).get("device", "auto"))
    model = build_model(config).to(device)
    model.load_state_dict(stored["model"])
    _, loader = build_dataloaders(config)
    return _evaluate(model, loader, config, device)


def evaluate_sources(config: dict, checkpoint: Path) -> dict:
    """Evaluate all declared validation sources individually, without step caps."""
    validate_config(config)
    stored = _read_checkpoint(checkpoint)
    _check_compatible(config, stored["config"])
    device = _device(config.get("training", {}).get("device", "auto"))
    model = build_model(config).to(device)
    model.load_state_dict(stored["model"])
    return {name: _evaluate(model, loader, config, device)
            for name, loader in build_validation_loaders(config).items()}
