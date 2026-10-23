"""Synthetic-only four-camera training integration check; never driving evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from camera_rc.curriculum import run_curriculum
from prepare_multiview_training import CAMERAS, prepare_multiview


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(output, device="cpu"):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "synthetic_labeled"
    raw.mkdir()
    rows = []
    # Known analytic surfaces exercise image/mask processing, not generalization.
    yy, xx = np.mgrid[:192, :320]
    for split_index, split in enumerate(("train", "val")):
        for camera_index, camera in enumerate(CAMERAS):
            stem = f"{split}_{camera}"
            labels = np.ones((192, 320), dtype=np.uint8)
            center = 160 + camera_index * 8 - 12 + split_index * 3
            road = (yy > 35) & (np.abs(xx - center) < (yy * 0.60 + 12))
            labels[road] = 0
            labels[:3] = 255
            rgb = np.full((192, 320, 3), (35, 65 + camera_index * 10, 70), dtype=np.uint8)
            rgb[road] = (120 + split_index * 8, 110, 105)
            image, mask = raw / f"{stem}.png", raw / f"{stem}_mask.png"
            Image.fromarray(rgb).save(image)
            Image.fromarray(labels).save(mask)
            rows.append({"image": image.name, "mask": mask.name, "camera": camera,
                         "clip_id": f"synthetic_original_{split}", "split": split})
    manifest = raw / "raw.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepared = root / "prepared"
    preparation_report = prepare_multiview(manifest, prepared, dataset_id="synthetic_multiview")
    configs = root / "configs"
    configs.mkdir()
    for stage_name, cameras in (("front_only", ("front",)), ("four_views", CAMERAS)):
        config = {
            "task": "segmentation", "diagnostic_only": True,
            "model": {"num_classes": 2, "width": 16},
            "data": {
                "label_space_id": "road_surface_v1", "image_size": [192, 320],
                "train_sources": [{"name": camera, "manifest": str(prepared / f"train_{camera}.jsonl"),
                                   "weight": 1.0} for camera in cameras],
                "val_manifests": [{"name": camera, "manifest": str(prepared / f"val_{camera}.jsonl")} for camera in cameras],
            },
            "training": {
                "output_dir": str(root / f"unused-{stage_name}"), "epochs": 1,
                "batch_size": 2, "grad_accum_steps": 1, "num_workers": 0,
                "lr": 0.0001, "seed": 42, "device": device, "amp": True,
                "amp_init_scale": 1024.0,
            },
        }
        _write_json(configs / f"{stage_name}.json", config)
    curriculum = configs / "curriculum.json"
    _write_json(curriculum, {"output_dir": "../training", "stages": [
        {"name": "front_only", "config": "front_only.json"},
        {"name": "four_views", "config": "four_views.json", "init_from": "previous"},
    ]})
    plan = run_curriculum(curriculum)
    result = run_curriculum(curriculum, execute=True)
    if not result["training_complete"] or not result.get("diagnostic"):
        raise RuntimeError("Expected completed synthetic diagnostic-only training")
    initial, adapted = result["stages"]
    if adapted["initialization"] != initial["checkpoint"]:
        raise RuntimeError("Second stage did not initialize from the first stage checkpoint")
    expected_holdouts = {str(prepared / f"val_{camera}.jsonl") for camera in CAMERAS}
    if set(adapted["holdout_metrics"]) != expected_holdouts:
        raise RuntimeError("Expected independent evaluation of all four camera holdouts")
    resumed = run_curriculum(curriculum, execute=True, resume_run=True)
    if resumed != result:
        raise RuntimeError("Completed-stage resume unexpectedly changed results")
    verification = {
        "diagnostic_only": True, "real_driving_validated": False,
        "device": device, "training_complete": True, "stages": len(result["stages"]),
        "samples_prepared": preparation_report["samples"],
        "global_holdouts_checked": plan["validation"]["global_holdouts_checked"],
        "shared_model": {"width": 16, "image_size": [192, 320], "num_classes": 2},
        "weights_transferred": True, "camera_holdouts_evaluated": list(CAMERAS),
        "optimizer_steps_by_stage": {
            stage["name"]: {"applied": stage["result"]["final"]["optimizer_steps"],
                            "amp_skipped": stage["result"]["final"]["amp_skipped_steps"]}
            for stage in result["stages"]},
        "resume_skipped_completed_stages": True,
        "report": str(root / "training/report.json"),
        "limits": "Synthetic masks test code integration only; no public or real vehicle dataset was trained.",
    }
    _write_json(root / "verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
