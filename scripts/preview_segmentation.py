"""CPU-only visual QA for local road_surface_v1 segmentation checkpoints.

This samples the validation manifests recorded in the checkpoint, not training
metadata. The contact sheet is a diagnostic illustration, not a representative
performance estimate or a substitute for full validation. No weights download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from camera_rc.data import ManifestDataset, read_manifest
from camera_rc.models import build_model
from camera_rc.training import _read_checkpoint


LABEL_SPACE = "road_surface_v1"
NOTICE = "Visual QA only: selected samples are not a representative performance estimate."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validation_rows(config: dict):
    data = config["data"]
    if "val_manifest" in data and "val_manifests" in data:
        raise ValueError("Checkpoint declares both val_manifest and val_manifests")
    specs = data.get("val_manifests", [data.get("val_manifest")])
    if not isinstance(specs, list) or not specs:
        raise ValueError("Checkpoint must declare validation manifests")
    rows, origins, sources, seen = [], [], [], set()
    for source_index, spec in enumerate(specs):
        spec = {"manifest": spec} if isinstance(spec, str) else spec
        if not isinstance(spec, dict) or not isinstance(spec.get("manifest"), str):
            raise ValueError("Validation source requires a manifest path")
        if spec.get("label_space_id", LABEL_SPACE) != LABEL_SPACE:
            raise ValueError("Validation source label_space_id differs from road_surface_v1")
        path = Path(spec["manifest"]).expanduser()
        # The training entry point resolves configuration paths before saving.
        # Guessing a different root here could silently preview the wrong data.
        if not path.is_absolute():
            raise ValueError("Checkpoint validation paths must be absolute, as saved by camera_rc train")
        path = path.resolve()
        source_rows = read_manifest(path)
        name = spec.get("name", f"val_{source_index}_{path.stem}")
        sources.append({"name": name, "manifest": str(path), "samples": len(source_rows),
                        "sha256": _sha256(path)})
        for local_index, row in enumerate(source_rows):
            identity = os.path.normcase(str(Path(row["image"]).resolve()))
            if identity in seen:
                raise ValueError(f"Duplicate image across validation sources: {row['image']}")
            seen.add(identity)
            if row.get("label_space_id", LABEL_SPACE) != LABEL_SPACE:
                raise ValueError("Validation sample label_space_id differs from road_surface_v1")
            rows.append(row)
            origins.append({"source": name, "manifest": str(path), "source_index": local_index})
    return rows, origins, sources


def _select_indices(total: int, samples: int, indices: list[int] | None) -> list[int]:
    if total < 1:
        raise ValueError("Validation manifests are empty")
    if indices is not None:
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Explicit indices must be nonempty and unique")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < total for i in indices):
            raise ValueError(f"Every index must be between 0 and {total - 1}")
        return list(indices)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    count = min(total, samples)
    if count == 1:
        return [(total - 1) // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _road_iou(prediction: np.ndarray, target: np.ndarray) -> dict:
    valid = target != 255
    if not valid.any():
        raise ValueError("Selected target has no supervised pixels")
    truth_road = (target == 0) & valid
    pred_road = (prediction == 0) & valid
    intersection = int((truth_road & pred_road).sum())
    union = int((truth_road | pred_road).sum())
    return {"road_iou": intersection / union if union else None,
            "road_intersection_pixels": intersection, "road_union_pixels": union,
            "ground_truth_road_pixels": int(truth_road.sum()),
            "predicted_road_pixels_on_valid_target": int(pred_road.sum()),
            "valid_pixels": int(valid.sum()), "ignored_pixels": int((~valid).sum())}


def _overlay(rgb: np.ndarray, labels: np.ndarray, ignored: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    road = (labels == 0) & ~ignored
    result[road] = 0.55 * result[road] + 0.45 * np.array([0, 230, 80], dtype=np.float32)
    # The same GT ignore mask is marked in both overlays and excluded from IoU.
    result[ignored] = 0.40 * result[ignored] + 0.60 * np.array([220, 0, 220], dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def preview(checkpoint: Path, output: Path, samples: int = 6,
            indices: list[int] | None = None) -> dict:
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new preview directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "incomplete.json"
    _write_json(incomplete, {"status": "incomplete", "checkpoint": str(checkpoint), "device": "cpu"})
    figure = None
    pyplot = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = _sha256(checkpoint)
        stored = _read_checkpoint(checkpoint)
        config = stored.get("config")
        if not isinstance(config, dict) or config.get("task") != "segmentation":
            raise ValueError("Expected a native segmentation checkpoint")
        if config.get("data", {}).get("label_space_id") != LABEL_SPACE:
            raise ValueError("This preview requires explicit label_space_id=road_surface_v1 (0 road, 1 nonroad)")
        if config.get("model", {}).get("num_classes") != 2:
            raise ValueError("road_surface_v1 preview requires exactly two model classes")
        size = config["data"].get("image_size")
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 16 for v in size
        ):
            raise ValueError("Checkpoint image_size must be [height,width] integers >=16")
        rows, origins, sources = _validation_rows(config)
        selected = _select_indices(len(rows), samples, indices)
        dataset = ManifestDataset(rows, config)
        # No automatic device selection and no calls into torch.cuda.
        model = build_model(config).cpu()
        model.load_state_dict(stored["model"], strict=True)
        if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        model.eval()
        panels, records = [], []
        with torch.inference_mode():
            for index in selected:
                sample = dataset[index]
                image, target_tensor = sample["image"].cpu(), sample["target"].cpu()
                logits = model(image.unsqueeze(0))
                if tuple(logits.shape) != (1, 2, *size) or not torch.isfinite(logits).all().item():
                    raise ValueError("Expected finite [1,2,H,W] segmentation logits")
                prediction = logits.argmax(1)[0].numpy()
                target = target_tensor.numpy()
                if not np.isin(target, (0, 1, 255)).all():
                    raise ValueError(f"Unknown target class IDs at validation index {index}")
                rgb = np.clip(image.permute(1, 2, 0).numpy() * 255.0, 0, 255).round().astype(np.uint8)
                ignored = target == 255
                panels.append((rgb, _overlay(rgb, target, ignored), _overlay(rgb, prediction, ignored)))
                row = rows[index]
                records.append({"index": index, "sample_id": row.get("sample_id", Path(row["image"]).name),
                                "dataset_id": row.get("dataset_id"), "group_id": row["group_id"],
                                "image": row["image"], "target": row["target"],
                                **origins[index], **_road_iou(prediction, target)})

        diagnostic = bool(stored.get("diagnostic") or stored.get("diagnostic_only")
                          or config.get("diagnostic_only") or stored.get("epoch_complete") is False)
        figure, axes = pyplot.subplots(len(panels), 3, figsize=(12, 2.65 * len(panels)), squeeze=False)
        for row_index, (views, record) in enumerate(zip(panels, records)):
            score = record["road_iou"]
            score_text = f"{score:.4f}" if score is not None else "undefined (empty union)"
            titles = [f"RGB | index {record['index']}", "GT road (green); ignore (magenta)",
                      f"Predicted road | road IoU {score_text}"]
            for column, view in enumerate(views):
                axes[row_index, column].imshow(view)
                axes[row_index, column].set_title(titles[column], fontsize=9)
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
            sample_id = str(record["sample_id"])
            axes[row_index, 0].set_xlabel(sample_id[:75] + ("..." if len(sample_id) > 75 else ""), fontsize=8)
        figure.suptitle(NOTICE + ("\nDiagnostic checkpoint / random or partial training provenance." if diagnostic else ""), fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.965))
        image_path = output / "preview.png"
        temporary_image = output / "preview.incomplete.png"
        figure.savefig(temporary_image, dpi=150, format="png")
        if _sha256(checkpoint) != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed during preview; rerun after training has finished")
        # Detect changes to sample ordering/paths while the preview was generated.
        if any(_sha256(Path(source["manifest"])) != source["sha256"] for source in sources):
            raise RuntimeError("A validation manifest changed during preview")
        report = {"status": "complete", "notice": NOTICE, "device": "cpu", "precision": "float32",
                  "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
                  "checkpoint_diagnostic": diagnostic, "checkpoint_epoch": stored.get("epoch"),
                  "label_space_id": LABEL_SPACE, "class_ids": {"road": 0, "nonroad": 1, "ignore": 255},
                  "input_image_size_hw": list(size), "full_validation_samples": len(rows),
                  "selection": "explicit_indices" if indices is not None else "evenly_spaced_manifest_indices",
                  "indices": selected, "validation_sources": sources, "samples": records,
                  "road_iou_definition": "Intersection/union for class 0; GT 255 excluded. Null if union is zero. Computed at model input resolution.",
                  "train_validation_leakage_rechecked": False,
                  "checks": "Only validation metadata and selected images/masks loaded. This does not rerun full validation or train/val leakage checks.",
                  "preview": str(image_path)}
        temporary_report = output / "report.incomplete.json"
        _write_json(temporary_report, report)
        os.replace(temporary_image, image_path)
        os.replace(temporary_report, output / "report.json")
        incomplete.unlink()
        return report
    except BaseException as error:
        _write_json(incomplete, {"status": "incomplete", "checkpoint": str(checkpoint), "device": "cpu",
                                 "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if figure is not None and pyplot is not None:
            pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit zero-based validation indices; overrides --samples")
    args = parser.parse_args()
    report = preview(args.checkpoint, args.output, args.samples, args.indices)
    print(json.dumps({"status": report["status"], "preview": report["preview"],
                      "report": str(args.output.resolve() / "report.json"), "indices": report["indices"],
                      "notice": NOTICE}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
