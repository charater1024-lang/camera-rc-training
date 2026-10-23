"""Compact native deployment checkpoints without optimizer or training data paths.

This preserves model arithmetic and image preprocessing. It does not quantize,
convert to ONNX, validate driving quality, or remove diagnostic provenance.
The result supports inference/evaluation and weight initialization, not resume.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ..models import build_model
from ..training import validate_config


def export_runtime_checkpoint(checkpoint, output) -> dict:
    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        stream.seek(0)
        stored = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(stored, dict) or stored.get("format_version") != 1:
        raise ValueError("Unsupported native checkpoint format")
    config = stored.get("config")
    if not isinstance(config, dict) or config.get("task") not in {"segmentation", "depth", "trajectory"}:
        raise ValueError("Export supports native segmentation, depth, or trajectory checkpoints")
    validate_config(config)
    model = build_model(config)
    model.load_state_dict(stored["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model weights")
    # Deliberately exclude dataset, output and weight initialization file paths.
    compact_config = {
        "task": config["task"],
        "model": {key: value for key, value in config.get("model", {}).items()
                  if key in {"width", "num_classes", "horizon"}},
        "data": {key: value for key, value in config.get("data", {}).items()
                 if key in {"image_size", "depth_scale", "max_depth_m", "future_offsets_s"}},
    }
    # A standard JSON round-trip guarantees portable basic config types and
    # rejects NaN/Infinity instead of putting them in a deployment artifact.
    compact_config = json.loads(json.dumps(compact_config, allow_nan=False))
    options = config.get("training", {})
    diagnostic = bool(stored.get("diagnostic") or stored.get("diagnostic_only")
                      or config.get("diagnostic") or config.get("diagnostic_only")
                      or options.get("max_train_steps") or options.get("max_val_steps")
                      or stored.get("epoch_complete") is False)
    payload = {
        "format_version": 1, "config": compact_config,
        "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "diagnostic": diagnostic, "export_kind": "native_runtime", "resumable": False,
        "source_sha256": hasher.hexdigest(),
    }
    if "epoch_complete" in stored:
        payload["epoch_complete"] = stored["epoch_complete"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 'xb' also prevents an overwrite if another process creates the path
    # between the earlier check and this write.
    with destination.open("xb") as stream:
        try:
            torch.save(payload, stream)
        except Exception:
            stream.close()
            destination.unlink()
            raise
    return {"output_checkpoint": str(destination), "source_sha256": hasher.hexdigest(),
            "task": config["task"], "diagnostic": diagnostic,
            "source_bytes": source.stat().st_size, "output_bytes": destination.stat().st_size,
            "note": "Native model and preprocessing unchanged; no optimizer/RNG state; not resumable."}
