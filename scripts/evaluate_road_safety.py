"""Evaluate a native road-surface checkpoint; never train or deploy a model.

Ordinary road datasets use evaluate_sources and their original validation
manifests. RoadObstacle21/RoadAnomaly21 use only locally available public GT.
Their GT=0 means usual, NOT a general road label. Only GT=1 is used to measure
the fraction of labelled obstacle/anomaly pixels incorrectly called road.
These are pixel diagnostics, not the official anomaly AP or a driving score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from camera_rc.config import load_config
from camera_rc.data import load_data_sources
from camera_rc.models import build_model
from camera_rc.training import _check_compatible, _device, _read_checkpoint, evaluate_sources


LABEL_SPACE = "road_surface_v1"
INPUT_SIZE = (192, 320)
TRACKS_SOURCE = "https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/datasets/tracks.py"
TRACKS = {
    "obstacle": {"name": "RoadObstacle21", "folder": "dataset_ObstacleTrack", "extension": ".webp", "public_gt_images": 30},
    "anomaly": {"name": "RoadAnomaly21", "folder": "dataset_AnomalyTrack", "extension": ".jpg", "public_gt_images": 10},
}
NOTICE = ("Evaluation only. Public labelled subsets, not the full hidden test set. "
          "Obstacle/anomaly road-error rates are not official anomaly-detection AP, "
          "collision probabilities, or evidence of safe driving. No deployment is authorized.")


def file_identity(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def road_metrics(confusion) -> dict:
    """Rows=GT, columns=prediction; class 0=road, class 1=nonroad."""
    matrix = np.asarray(confusion)
    if (matrix.shape != (2, 2) or matrix.dtype.kind not in "iu"
            or (matrix < 0).any()):
        raise ValueError("road_surface_v1 requires a 2x2 nonnegative integer confusion matrix")
    tp, fn, fp, tn = (int(value) for value in matrix.ravel())
    return {
        "road_tp_pixels": tp, "road_fn_pixels": fn, "road_fp_pixels": fp, "road_tn_pixels": tn,
        "road_precision": _ratio(tp, tp + fp), "road_recall": _ratio(tp, tp + fn),
        "false_road_rate": _ratio(fp, fp + tn),
        "false_road_rate_definition": "FP / (FP + TN), GT nonroad pixels predicted road",
        "decision": "native evaluate_sources argmax; rows=GT, columns=prediction; road=0",
    }


def _thresholds(runtime_threshold: float):
    if isinstance(runtime_threshold, bool) or not math.isfinite(runtime_threshold) or not 0.5 < runtime_threshold <= 1:
        raise ValueError("runtime_threshold must be finite and in (0.5, 1]")
    return {"road_p_gt_0_5": (0.5, ">"), "runtime_road": (float(runtime_threshold), ">=")}


def obstacle_pixel_metrics(probability, ground_truth, runtime_threshold: float = 0.65) -> dict:
    probability, target = np.asarray(probability), np.asarray(ground_truth)
    if target.ndim != 2 or target.dtype.kind not in "iu" or not np.isin(target, [0, 1, 255]).all():
        raise ValueError("Public obstacle/anomaly GT must be a single-channel 0/1/255 ID mask")
    if (probability.shape != target.shape or probability.dtype.kind not in "fiu"
            or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Road probability must be finite [0,1] at the ORIGINAL GT resolution")
    obstacle = target == 1
    total = int(obstacle.sum())
    decisions = {}
    for name, (threshold, comparison) in _thresholds(runtime_threshold).items():
        predicted = probability > threshold if comparison == ">" else probability >= threshold
        errors = int((predicted & obstacle).sum())
        decisions[name] = {"threshold": threshold, "comparison": comparison,
                           "obstacle_pixels_predicted_road": errors,
                           "labelled_obstacle_pixels": total,
                           "obstacle_as_road_rate": _ratio(errors, total)}
    return {"labelled_obstacle_pixels": total, "usual_pixels_not_used_as_road_gt": int((target == 0).sum()),
            "ignored_pixels": int((target == 255).sum()), "decisions": decisions}


def aggregate_obstacle_metrics(rows: list[dict], runtime_threshold: float = 0.65) -> dict:
    if not rows:
        raise ValueError("No labelled images were evaluated")
    decisions = {}
    for name, (threshold, comparison) in _thresholds(runtime_threshold).items():
        errors = sum(row["decisions"][name]["obstacle_pixels_predicted_road"] for row in rows)
        total = sum(row["labelled_obstacle_pixels"] for row in rows)
        rates = [row["decisions"][name]["obstacle_as_road_rate"] for row in rows
                 if row["decisions"][name]["obstacle_as_road_rate"] is not None]
        if not total:
            raise ValueError("Public GT subset has no GT=1 obstacle/anomaly pixels")
        decisions[name] = {
            "threshold": threshold, "comparison": comparison,
            "obstacle_pixels_predicted_road": errors, "labelled_obstacle_pixels": total,
            "micro_obstacle_as_road_rate": errors / total,
            "mean_per_image_obstacle_as_road_rate": sum(rates) / len(rates),
            "images_with_labelled_obstacles": len(rates),
        }
    return {"images": len(rows), "decisions": decisions,
            "usual_pixels_not_used_as_road_gt": sum(row["usual_pixels_not_used_as_road_gt"] for row in rows),
            "ignored_pixels": sum(row["ignored_pixels"] for row in rows)}


def discover_pairs(root: Path, track: str, *, expected_count: int | None = None) -> dict:
    """Pair only public GT, while explicitly leaving unlabelled test images out."""
    spec = TRACKS[track]
    root = Path(root).expanduser().resolve(strict=True)
    if not (root / "images").is_dir() and (root / spec["folder"]).is_dir():
        root = root / spec["folder"]
    image_dir, mask_dir = root / "images", root / "labels_masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected {root}/images and labels_masks; finish extracting the dataset first")
    suffix = "_labels_semantic.png"
    masks = sorted(mask_dir.glob(f"*{suffix}"))
    required = spec["public_gt_images"] if expected_count is None else expected_count
    if len(masks) != required:
        raise ValueError(f"{spec['name']} requires exactly {required} public GT masks; found {len(masks)}. Incomplete or different splits must not be reported as the expected subset")
    pairs, identities = [], set()
    for target in masks:
        fid = target.name[:-len(suffix)]
        if not fid or fid.casefold() in identities:
            raise ValueError("Empty or duplicate public GT frame ID")
        identities.add(fid.casefold())
        image = image_dir / (fid + spec["extension"])
        if not image.is_file():
            raise FileNotFoundError(f"Missing paired image for public GT: {image}")
        # Ambiguous alternate image files are rejected rather than guessed.
        alternatives = [p for p in image_dir.glob(fid + ".*") if p.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"}]
        if len(alternatives) != 1:
            raise ValueError(f"Ambiguous image pair: {fid}")
        pairs.append({"frame_id": fid, "image": image.resolve(), "target": target.resolve()})
    images = list(image_dir.glob("*" + spec["extension"]))
    return {"root": str(root), "pairs": pairs, "available_images": len(images),
            "unlabelled_images_not_evaluated": len(images) - len(pairs), "expected_public_gt_images": required}


def read_pair(pair: dict):
    with Image.open(pair["image"]) as image:
        rgb = np.array(image.convert("RGB"))
    with Image.open(pair["target"]) as mask:
        target = np.array(mask)
    if target.ndim != 2 or target.shape != rgb.shape[:2]:
        raise ValueError(f"Image/GT original shape mismatch or RGB label mask: {pair['frame_id']}")
    if target.dtype.kind not in "iu" or not np.isin(target, [0, 1, 255]).all():
        raise ValueError(f"Unexpected public GT label IDs: {pair['frame_id']}")
    return rgb, target


def restore_road_probability(logits: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [1,2,H,W] road_surface_v1 logits")
    road = logits.float().softmax(dim=1)[:, 0:1]
    restored = F.interpolate(road, size=original_size, mode="bilinear", align_corners=False)
    return restored[0, 0].detach().cpu().numpy()


class NativeRoadPredictor:
    def __init__(self, config: dict, stored: dict):
        _check_compatible(config, stored["config"])
        self.device = _device(config.get("training", {}).get("device", "auto"))
        self.amp = bool(config.get("training", {}).get("amp", False)) and self.device.type == "cuda"
        self.model = build_model(config).to(self.device).eval()
        self.model.load_state_dict(stored["model"], strict=True)
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite model weights")

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        resized = Image.fromarray(rgb).resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)
        values = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        tensor = torch.from_numpy(values).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            logits = self.model(tensor)
        # GT is never downsampled: a one-pixel obstacle must remain in the denominator.
        return restore_road_probability(logits, tuple(rgb.shape[:2]))


def evaluate_track(discovered: dict, predictor, track: str, runtime_threshold: float = 0.65) -> dict:
    rows = []
    for index, pair in enumerate(discovered["pairs"], start=1):
        rgb, target = read_pair(pair)
        row = obstacle_pixel_metrics(predictor(rgb), target, runtime_threshold)
        row.update(frame_id=pair["frame_id"], original_image_size=list(target.shape),
                   image=file_identity(pair["image"]), ground_truth=file_identity(pair["target"]))
        rows.append(row)
        print(json.dumps({"event": "public_gt_image", "track": track, "image": index,
                          "total": len(discovered["pairs"]), "frame_id": pair["frame_id"]}), flush=True)
    return {"dataset": TRACKS[track]["name"], "role": "evaluation_only_public_labelled_subset",
            "root": discovered["root"], "available_images": discovered["available_images"],
            "unlabelled_images_not_evaluated": discovered["unlabelled_images_not_evaluated"],
            "expected_public_gt_images": discovered["expected_public_gt_images"],
            "labels": {"1": "labelled obstacle/anomaly; only evaluated denominator", "0": "usual; not assumed road", "255": "ignore"},
            "source_format_reference": TRACKS_SOURCE, "ground_truth_resizing": "none",
            "probability_resizing": "bilinear to original GT size, align_corners=False, before thresholding",
            "aggregate": aggregate_obstacle_metrics(rows, runtime_threshold), "per_image": rows}


def _assert_not_training(discovered: dict, training_sources: list[dict]):
    labelled = {str(pair["image"]).casefold() for pair in discovered["pairs"]}
    for source in training_sources:
        for row in source["rows"]:
            if any(str(Path(row[key]).resolve()).casefold() in labelled for key in ("image", "source_image") if row.get(key)):
                raise ValueError("Public obstacle/anomaly evaluation images appear in a training manifest")


def evaluate(config_path: Path, checkpoint: Path, output: Path, *, obstacle_root: Path | None = None,
             anomaly_root: Path | None = None, runtime_threshold: float = 0.65, device: str | None = None) -> dict:
    _thresholds(runtime_threshold)
    config_path, checkpoint, output = (Path(p).expanduser().resolve() for p in (config_path, checkpoint, output))
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    config = load_config(config_path)
    if (config.get("task") != "segmentation" or config.get("model", {}).get("num_classes") != 2
            or config.get("data", {}).get("label_space_id") != LABEL_SPACE
            or tuple(config.get("data", {}).get("image_size", [])) != INPUT_SIZE):
        raise ValueError("Require road_surface_v1 binary segmentation with image_size=[192,320]")
    if device is not None:
        config.setdefault("training", {})["device"] = device
    checkpoint_identity = file_identity(checkpoint)
    stored = _read_checkpoint(checkpoint)
    _check_compatible(config, stored["config"])
    training, validation = load_data_sources(config)
    discovered = {name: discover_pairs(root, name) for name, root in (("obstacle", obstacle_root), ("anomaly", anomaly_root)) if root is not None}
    for value in discovered.values():
        _assert_not_training(value, training)
        for pair in value["pairs"]:
            read_pair(pair)  # Fail on corrupt/misaligned masks before the expensive full native evaluation.
    print(json.dumps({"event": "native_validation_started", "sources": [source["name"] for source in validation]}), flush=True)
    native = evaluate_sources(config, checkpoint)
    for name, result in native.items():
        if result.get("partial_validation"):
            raise ValueError(f"Unexpected partial validation: {name}")
        result["road_safety_pixels"] = road_metrics(result["confusion_matrix"])
    predictor = NativeRoadPredictor(config, stored) if discovered else None
    tracks = {name: evaluate_track(value, predictor, name, runtime_threshold) for name, value in discovered.items()}
    if file_identity(checkpoint) != checkpoint_identity:
        raise RuntimeError("Checkpoint changed during evaluation; evaluate a stable snapshot in a new report")
    report = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "notice": NOTICE,
        "evaluation_only": True, "training_performed": False, "deployment_validated": False,
        "checkpoint": checkpoint_identity, "config": file_identity(config_path),
        "checkpoint_diagnostic": bool(stored.get("diagnostic") or stored.get("diagnostic_only") or stored.get("epoch_complete") is False),
        "input_size": list(INPUT_SIZE), "label_space_id": LABEL_SPACE, "road_class_id": 0,
        "device": str(_device(config.get("training", {}).get("device", "auto"))),
        "amp_requested": bool(config.get("training", {}).get("amp", False)),
        "runtime_threshold": runtime_threshold,
        "validation_sources": [{"name": source["name"], "images": len(source["rows"]),
                                "manifest": file_identity(Path(source["manifest"]))} for source in validation],
        "training_manifests_checked_for_exact_path_overlap": [file_identity(Path(source["manifest"])) for source in training],
        "overlap_check_limit": "Exact image/source_image paths in the supplied config only; cannot prove absence of prior or copied-image training exposure",
        "native_sources": native, "public_obstacle_anomaly_subsets": tracks,
        "native_resolution_note": "Native source metrics use evaluate_sources target resize/argmax; obstacle/anomaly metrics retain original GT pixels. Do not pool these different protocols.",
        "content_identity_note": "Checkpoint/config/manifests and every public obstacle/anomaly image/GT are SHA-256 hashed. Native source image bytes are not hashed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report file; never overwritten")
    parser.add_argument("--obstacle-root", type=Path)
    parser.add_argument("--anomaly-root", type=Path)
    parser.add_argument("--runtime-threshold", type=float, default=0.65, help="Current runtime profile threshold; not a calibrated safety probability")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report = evaluate(args.config, args.checkpoint, args.output, obstacle_root=args.obstacle_root,
                      anomaly_root=args.anomaly_root, runtime_threshold=args.runtime_threshold, device=args.device)
    print(json.dumps({"report": str(args.output.resolve()), "sources": list(report["native_sources"]),
                      "public_subsets": list(report["public_obstacle_anomaly_subsets"]), "deployment_validated": False}), flush=True)


if __name__ == "__main__":
    main()
