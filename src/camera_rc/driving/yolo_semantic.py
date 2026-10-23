"""Verified local YOLO26x semantic adapter for offline/replay research.

The pinned report is provenance, not a signature or permission to load an
untrusted pickle. Only use checkpoints and report hashes obtained from a
trusted local training run. Dataset/ancestor audits belong to the evaluator;
runtime verifies the pinned completion evidence and the selected weight bytes.
No dataset or weight is downloaded. Physical actuation is blocked in config.py.

Numerics deliberately match scripts/evaluate_yolo_semantic.py: RGB PIL bilinear
direct resize, float /255, FP32 raw-logit resize to input size, softmax, then
bilinear road probability resize to the original camera resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


BACKEND = "yolo26x_semantic"
CLASS_NAMES = ["road_surface", "not_road_surface"]


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _local_file(value, name):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} requires an explicit local file")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must be a local file; no downloads are supported")
    return path


def file_identity(path):
    path = _local_file(path, "identity")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
            count += len(data)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": count}


def _recorded_identity(value):
    return (isinstance(value, dict) and isinstance(value.get("path"), str) and bool(value["path"])
            and _digest(value.get("sha256")) and type(value.get("bytes")) is int and value["bytes"] > 0)


def verify_completion(config, checkpoint):
    """Bind a relocatable checkpoint to a pinned, fully completed training report.

Original report paths remain unchanged: copied deployment files may reside on
another machine. Content SHA256 + byte count + explicit role bind the copy.
No source data or original training-machine paths are opened here.
"""
    path = _local_file(config.get("segmentation_training_report"), "segmentation_training_report")
    expected = config.get("segmentation_training_report_sha256")
    if not _digest(expected):
        raise ValueError("segmentation_training_report_sha256 requires pinned lowercase SHA256")
    payload = path.read_bytes()
    report_identity = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if report_identity["sha256"] != expected:
        raise ValueError("Training report SHA256 differs from the pinned configuration")
    report = json.loads(payload.decode("utf-8-sig"))
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or report.get("model_family") != "yolo26x-sem"
            or report.get("task") != "semantic" or report.get("class_names") != CLASS_NAMES
            or type(report.get("road_class_id")) is not int or report["road_class_id"] != 0
            or report.get("training_complete") is not True or report.get("diagnostic") is not False):
        raise ValueError("Require completed non-diagnostic YOLO26x-sem with canonical road=0/nonroad=1")
    actual, history, progress = (report.get(key) for key in ("actual_args", "history", "progress"))
    if not isinstance(actual, dict) or not isinstance(history, list) or not isinstance(progress, dict):
        raise ValueError("Training completion requires actual_args, epoch history and update counters")
    epochs, batch = actual.get("epochs"), actual.get("batch")
    if (type(epochs) is not int or epochs < 1 or type(batch) is not int or batch < 1
            or actual.get("resume") is not False or actual.get("fraction") != 1.0
            or len(history) != epochs
            or any(not isinstance(row, dict) or type(row.get("epoch")) not in (int, float)
                   or row["epoch"] != index
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in row.values())
                   for index, row in enumerate(history, 1))):
        raise ValueError("Incomplete or nonfinite epoch history")
    dataset = report.get("dataset")
    counts = dataset.get("counts") if isinstance(dataset, dict) else None
    if (not isinstance(counts, dict) or type(counts.get("train")) is not int or counts["train"] < 1
            or not isinstance(dataset.get("provenance_path"), str) or not dataset["provenance_path"]
            or not _digest(dataset.get("provenance_sha256"))
            or not _recorded_identity(report.get("source_config"))):
        raise ValueError("Training report lacks dataset/source provenance")
    batches = math.ceil(counts["train"] / batch)
    keys = ("train_batches", "optimizer_attempts", "optimizer_updates", "amp_skipped_updates")
    if (type(report.get("batches_per_epoch")) is not int or report["batches_per_epoch"] != batches
            or any(type(progress.get(k)) is not int or progress[k] < 0 for k in keys)
            or progress["train_batches"] != epochs * batches or progress["optimizer_updates"] < 1
            or progress["optimizer_attempts"] > progress["train_batches"]
            or progress["optimizer_attempts"] != progress["optimizer_updates"] + progress["amp_skipped_updates"]):
        raise ValueError("Training report does not prove completed batches and optimizer updates")
    checkpoints = report.get("checkpoints")
    role = config.get("segmentation_checkpoint_role", "best")
    if (role not in ("best", "last") or not isinstance(checkpoints, dict)
            or any(not _recorded_identity(checkpoints.get(k)) for k in ("best", "last"))):
        raise ValueError("Require recorded best/last identities and a best or last checkpoint role")
    selected = file_identity(checkpoint)
    recorded = checkpoints[role]
    if any(selected[key] != recorded[key] for key in ("sha256", "bytes")):
        raise ValueError("Selected checkpoint SHA256/size differs from the completed training report role")
    return {"training_report": report_identity, "checkpoint": selected, "checkpoint_role": role,
            "recorded_checkpoint": dict(recorded), "class_names": list(CLASS_NAMES), "road_class_id": 0,
            "completed_epochs": epochs, "optimizer_updates": progress["optimizer_updates"],
            "physical_actuation_supported": False,
            "verification_scope": "Pinned report completion evidence and selected weight bytes; no dataset or ancestor re-audit"}


def validate_input_size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(d) is not int or d < 32 or d > 4096 or d % 32 for d in value)):
        raise ValueError("segmentation_input_size requires [height,width] multiples of 32 in 32..4096")
    return tuple(value)


def validate_semantic_model(wrapper, *, model_class, head_class):
    model = wrapper.model
    if wrapper.task != "semantic" or not isinstance(model, model_class):
        raise ValueError("Expected native PyTorch semantic segmentation model")
    head = model.model[-1]
    if not isinstance(head, head_class) or head.nc != 2 or head.export or head.bake_argmax:
        raise ValueError("Require two-class non-export raw semantic logits head")
    if model.names != dict(enumerate(CLASS_NAMES)) and model.names != CLASS_NAMES:
        raise ValueError("Checkpoint class mapping differs from canonical road=0/nonroad=1")
    if (model.yaml.get("scale") != "x"
            or Path(model.yaml.get("yaml_file", "")).stem not in {"yolo26-sem", "yolo26x-sem"}):
        raise ValueError("Require YOLO26x-sem architecture, scale x")
    return model


def _load_semantic_model(checkpoint, settings_dir):
    # Lazy dependency; TinyUNet-only installs do not import Ultralytics. The
    # local .pt file must be trusted because Ultralytics loads pickle objects.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    try:
        from ultralytics import YOLO, settings
        from ultralytics.nn.tasks import SemanticSegmentationModel
        from ultralytics.nn.modules.head import SemanticSegment
    except ImportError as exc:
        raise RuntimeError("Install a tested Ultralytics version with YOLO26 semantic support (project [detection] extra)") from exc
    settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "neptune", "comet") if key in settings})
    wrapper = YOLO(str(checkpoint), task="semantic")
    return validate_semantic_model(wrapper, model_class=SemanticSegmentationModel, head_class=SemanticSegment)


def image_tensor(rgb, input_size):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2):
        raise ValueError("Semantic adapter expects HxWx3 uint8 RGB")
    image = Image.fromarray(rgb).resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0


def restore_road_probability(logits, input_size, original_size):
    if (not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits)
            or logits.ndim != 4 or logits.shape[:2] != (1, 2)
            or min(logits.shape[2:]) < 1 or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite raw floating [1,2,H,W] semantic logits")
    # Resizing logits BEFORE softmax is significant; do not resize a low-
    # resolution probability map directly as the native TinyUNet branch does.
    logits = F.interpolate(logits.float(), size=input_size, mode="bilinear", align_corners=False)
    road = logits.softmax(dim=1)[:, 0:1]
    return F.interpolate(road, size=original_size, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()


class VerifiedYoloSemantic:
    def __init__(self, config, checkpoint, *, device, half):
        self.input_size = validate_input_size(config.get("segmentation_input_size"))
        if config.get("drivable_class_ids") != [0] or type(config["drivable_class_ids"][0]) is not int:
            raise ValueError("YOLO semantic requires drivable_class_ids=[0]")
        if config.get("allow_diagnostic", False):
            raise ValueError("YOLO semantic backend does not accept diagnostic checkpoints")
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"
        checkpoint = _local_file(checkpoint, "segmentation_checkpoint")
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("YOLO semantic requires a local PyTorch .pt checkpoint")
        self.provenance = verify_completion(config, checkpoint)
        settings_dir = Path(config.get("yolo_settings_dir", checkpoint.parent / ".ultralytics")).expanduser().resolve()
        self.model = _load_semantic_model(checkpoint, settings_dir).float().eval()
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        # Detect ordinary replacement during loading before any model transfer.
        if (file_identity(checkpoint) != self.provenance["checkpoint"]
                or file_identity(self.provenance["training_report"]["path"]) != self.provenance["training_report"]):
            raise ValueError("Checkpoint/report changed while loading semantic model")
        self.model = self.model.to(self.device)

    @torch.inference_mode()
    def __call__(self, rgb):
        tensor = image_tensor(rgb, self.input_size).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):
            logits = self.model(tensor)
        return restore_road_probability(logits, self.input_size, tuple(rgb.shape[:2]))
