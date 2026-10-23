"""Optional Ultralytics detection adapter, lazily imported.

Ultralytics has AGPL-3.0 / commercial license options; see its license before
redistribution/deployment: https://www.ultralytics.com/license . Dataset terms
are separate. Official train API: https://docs.ultralytics.com/modes/train/ .
Pretrained named .pt models may download weights on first explicit training run.
"""
from __future__ import annotations

from pathlib import Path
import os


def run_detection(config: dict, mode: str = "train", checkpoint=None) -> dict:
    if mode not in {"train", "val", "export"}:
        raise ValueError("Detection mode must be train, val, or export")
    settings = dict(config)
    settings.pop("task", None)
    model_name = settings.pop("model", "yolo11n.pt")
    resume = settings.pop("resume", False)
    export_settings = settings.pop("export", {"format": "onnx", "dynamic": True, "simplify": False})
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model_name = str(checkpoint)
    if resume and checkpoint is None:
        raise ValueError("Resume requires an explicit existing --checkpoint last.pt")
    if mode != "train" and checkpoint is None:
        raise ValueError("Validation/export require an explicit trained --checkpoint")
    if mode != "export":
        data = settings.get("data")
        if not data or not Path(data).is_file():
            raise FileNotFoundError("Prepare the local dataset.yaml and set config.data before training")
        settings["data"] = str(Path(data).resolve())
    try:
        # Keep project-specific preferences outside the user's global settings.
        local_settings = Path(settings.get("project", "runs/detection")).resolve().parent / ".ultralytics"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(local_settings))
        from ultralytics import YOLO, settings as library_settings
        disabled = {key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "neptune", "comet") if key in library_settings}
        library_settings.update(disabled)
    except ImportError as error:
        raise RuntimeError("Detection requires the optional dependency: pip install -e \".[detection]\"") from error
    model = YOLO(model_name, task="detect")
    if mode == "train":
        results = model.train(**settings, resume=bool(resume))
    elif mode == "val":
        supported = {"data", "imgsz", "batch", "device", "workers", "project", "name", "plots", "save_json", "conf", "iou", "split"}
        results = model.val(**{k: v for k, v in settings.items() if k in supported})
    else:
        result_path = model.export(**export_settings)
        return {"mode": mode, "artifact": str(result_path)}
    metrics = getattr(results, "results_dict", {})
    metrics = {str(k): float(v) for k, v in metrics.items()}
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and getattr(model, "trainer", None) is not None:
        save_dir = model.trainer.save_dir
    return {"mode": mode, "metrics": metrics, "save_dir": str(save_dir) if save_dir else None}
