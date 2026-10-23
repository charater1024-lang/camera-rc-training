"""Batch-one perception using local checkpoints, with no weight downloads.

RGB input must already be lens-rectified in the calibration's original pixel
coordinates. Native segmentation preprocessing matches ManifestDataset: PIL
bilinear resize, RGB float /255; normalization is inside the native model.
The explicitly selected YOLO26x semantic backend matches the offline evaluator
and is restricted to dry-run/replay; it does not imply a validated controller.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ..models import build_model
from .types import Detection, Perception


def _local_file(value, name):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must name an existing local checkpoint")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}; runtime never downloads weights")
    return path


class CameraPerception:
    def __init__(self, config: dict, device: str = "auto"):
        self.config = dict(config)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Perception device must be auto, cpu, cuda or cuda:N")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.half = bool(config.get("half", True)) and self.device.type == "cuda"
        self.backend = config.get("segmentation_backend", "native")
        if self.backend not in {"native", "yolo26x_semantic"}:
            raise ValueError("segmentation_backend must be native or yolo26x_semantic")
        checkpoint_path = _local_file(config.get("segmentation_checkpoint"), "segmentation_checkpoint")
        self.semantic_adapter = None
        self.provenance = None
        if self.backend == "yolo26x_semantic":
            from .yolo_semantic import VerifiedYoloSemantic
            self.semantic_adapter = VerifiedYoloSemantic(config, checkpoint_path, device=self.device, half=self.half)
            self.height, self.width = self.semantic_adapter.input_size
            self.classes, self.diagnostic = 2, False
            self.model = self.semantic_adapter.model
            self.provenance = self.semantic_adapter.provenance
        else:
            if any(key in config for key in ("segmentation_training_report", "segmentation_training_report_sha256",
                                              "segmentation_checkpoint_role", "segmentation_input_size")):
                raise ValueError("YOLO semantic settings require explicit segmentation_backend=yolo26x_semantic")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
                raise ValueError("Unsupported native checkpoint format")
            stored = checkpoint.get("config", {})
            if stored.get("task") != "segmentation":
                raise ValueError("Runtime requires a segmentation checkpoint")
            train_options = stored.get("training", {})
            self.diagnostic = bool(checkpoint.get("diagnostic") or checkpoint.get("diagnostic_only")
                                   or stored.get("diagnostic") or stored.get("diagnostic_only")
                                   or train_options.get("max_train_steps") or train_options.get("max_val_steps")
                                   or checkpoint.get("epoch_complete") is False)
            if self.diagnostic and not config.get("allow_diagnostic", False):
                raise ValueError("Diagnostic checkpoint cannot drive; allow_diagnostic is only for replay/benchmark")
            size = stored.get("data", {}).get("image_size", [256, 512])
            if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 16 for v in size):
                raise ValueError("Checkpoint image_size must contain height,width integers >=16")
            self.height, self.width = size
            self.classes = int(stored.get("model", {}).get("num_classes", 3))
            self.model = build_model(stored)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
                raise ValueError("Checkpoint contains nonfinite model weights")
            self.model.to(self.device).eval()
        ids = config.get("drivable_class_ids")
        if not isinstance(ids, (list, tuple)) or not ids or any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < self.classes for v in ids
        ) or len(ids) != len(set(ids)):
            raise ValueError("Explicit unique drivable_class_ids must match checkpoint class IDs")
        self.drivable_ids = list(ids)
        self.min_probability = float(config.get("min_probability", 0.6))
        if not np.isfinite(self.min_probability) or not 0 < self.min_probability <= 1:
            raise ValueError("min_probability must be in (0,1]")
        self.detector = None
        self.detection_precision = {"half": self.half}
        self.obstacle_labels = config.get("obstacle_labels")
        if self.obstacle_labels is not None:
            if not isinstance(self.obstacle_labels, list) or not all(isinstance(label, str) for label in self.obstacle_labels):
                raise ValueError("obstacle_labels must be a list of strings or omitted")
            self.obstacle_labels = set(self.obstacle_labels)
        self.detection_size = config.get("detection_image_size", 320)
        self.max_detections = config.get("detection_max_detections", 30)
        self.detection_confidence = float(config.get("detection_confidence", 0.35))
        if isinstance(self.detection_size, bool) or not isinstance(self.detection_size, int) or self.detection_size < 32:
            raise ValueError("detection_image_size must be an integer >=32")
        if isinstance(self.max_detections, bool) or not isinstance(self.max_detections, int) or self.max_detections < 1:
            raise ValueError("detection_max_detections must be a positive integer")
        if not np.isfinite(self.detection_confidence) or not 0 < self.detection_confidence <= 1:
            raise ValueError("detection_confidence must be in (0,1]")
        if config.get("detection_checkpoint"):
            detection_path = _local_file(config["detection_checkpoint"], "detection_checkpoint")
            if detection_path.suffix.lower() != ".pt":
                raise ValueError("Detection adapter currently accepts local Ultralytics .pt checkpoints")
            # Ultralytics loads its own checkpoint format; use only trusted local files.
            settings_dir = Path(config.get("yolo_settings_dir", checkpoint_path.parent / ".ultralytics"))
            os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir.resolve()))
            os.environ.setdefault("YOLO_OFFLINE", "true")
            try:
                from ultralytics import YOLO, settings
                from ultralytics.cfg import DEFAULT_CFG_DICT
            except ImportError as exc:
                raise RuntimeError("Install the project's [detection] extra for optional object detection") from exc
            settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "neptune", "comet") if key in settings})
            self.detector = YOLO(str(detection_path), task="detect")
            # Newer Ultralytics uses quantize; feature-detect to support older 8.x.
            if "quantize" in DEFAULT_CFG_DICT:
                self.detection_precision = {"quantize": 16 if self.half else 32}
            if self.detector.task != "detect":
                raise ValueError("Detection checkpoint must be an object detection model")

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray) -> Perception:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("infer expects an HxWx3 uint8 RGB image")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if self.semantic_adapter is not None:
            mask = self.semantic_adapter(rgb)
        else:
            resized = Image.fromarray(rgb).resize((self.width, self.height), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy()).unsqueeze(0).div_(255).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):
                logits = self.model(tensor)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite segmentation logits; no driving output produced")
            probability = logits.float().softmax(dim=1)[:, self.drivable_ids].sum(dim=1, keepdim=True)
            probability = F.interpolate(probability, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            mask = probability.cpu().numpy().astype(np.float32, copy=False)
        # Low-confidence pixels become unknown/blocked; the planner can impose
        # an equal or higher confidence threshold, never recover removed pixels.
        mask[mask < self.min_probability] = 0.0
        detections = []
        if self.detector is not None:
            # Ultralytics numpy image source expects BGR, unlike the native network.
            results = self.detector.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=self.detection_size,
                                            device=self.device, conf=self.detection_confidence,
                                            max_det=self.max_detections, batch=1, verbose=False, save=False, stream=False,
                                            **self.detection_precision)
            if len(results) != 1:
                raise ValueError("Expected one detection result per camera frame")
            boxes = results[0].boxes
            if boxes is not None:
                for xyxy, confidence, category in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    if not np.isfinite([*xyxy, confidence, category]).all():
                        raise FloatingPointError("Nonfinite object detection")
                    label = str(results[0].names[int(category)])
                    if self.obstacle_labels is None or label in self.obstacle_labels:
                        x1, y1, x2, y2 = map(float, xyxy)
                        x1, x2 = np.clip([x1, x2], 0, rgb.shape[1] - 1)
                        y1, y2 = np.clip([y1, y2], 0, rgb.shape[0] - 1)
                        if x2 > x1 and y2 > y1:
                            detections.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(confidence), label))
        if not np.isfinite(mask).all():
            raise FloatingPointError("Nonfinite drivable probability")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return Perception(mask, detections, (time.perf_counter() - started) * 1000)

    __call__ = infer
