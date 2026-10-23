"""Train synthetic eight-role fixtures on CPU to verify shared-model integration.

This is a diagnostic code check, not real-camera training or driving evidence.
No cameras, serial devices, motor controllers or network services are opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from camera_rc.config import load_config
from camera_rc.models import build_model
from camera_rc.training import run_training, evaluate_sources
from prepare_camera_role_curriculum import ROLES, prepare_curriculum


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_smoke(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "DIAGNOSTIC_INCOMPLETE").write_text("Synthetic diagnostic is incomplete. Never use for driving.\n", encoding="utf-8")
    raw = root / "synthetic_fixtures"
    raw.mkdir()
    rig_id, dataset_id = "synthetic_eight_role_fixture_v1", "synthetic_eight_role_fixture"
    yy, xx = np.mgrid[:192, :320]
    manifests = {}
    for split_index, split in enumerate(("train", "val")):
        rows = []
        clip = f"synthetic_original_clip_{split}"
        for index, role in enumerate(ROLES):
            center = 125 + index * 9 + split_index * 3
            road = (yy > 38) & (abs(xx - center) < yy * .48 + 15)
            labels = np.where(road, 0, 1).astype(np.uint8)
            labels[:3] = 255
            rgb = np.full((192, 320, 3), (25 + index * 4, 65, 80), np.uint8)
            rgb[road] = (115 + split_index * 8, 110, 100)
            # Distinct small nonroad patches exercise ID masks but represent no
            # real obstacle, camera perspective, calibration or operating scene.
            labels[110:124, 145 + index:154 + index] = 1
            rgb[110:124, 145 + index:154 + index] = (190, 40, 40)
            image, target = raw / f"{split}_{role}.png", raw / f"{split}_{role}_target.png"
            Image.fromarray(rgb).save(image)
            Image.fromarray(labels).save(target)
            rows.append({"image": str(image), "target": str(target), "source_image": str(image),
                         "task": "segmentation", "label_space_id": "road_surface_v1",
                         "dataset_id": dataset_id, "group_id": clip, "clip_id": clip,
                         "origin_id": f"{dataset_id}:{clip}", "split": split,
                         "rig_id": rig_id, "camera_role": role, "condition": "synthetic_day",
                         "synthetic": True, "diagnostic_only": True})
        path = raw / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifests[split] = str(path)
    source_spec = root / "source_spec.json"
    _write_json(source_spec, {"rig_id": rig_id, "epochs": 1, "samples_per_epoch": 8,
                              "sources": [{"name": "synthetic_fixture", "kind": "vehicle",
                                           "weight": 1, "manifests": manifests}]})
    preparation = prepare_curriculum(source_spec, root / "prepared", require_vehicle_coverage=True)
    if not preparation["all_roles_have_real_rig_train_and_val_samples"]:
        raise RuntimeError("Expected every synthetic fixture role in train and validation")
    # The vehicle-shaped fixture deliberately tests that interface. Its counts
    # are synthetic and must never be published as real-rig coverage evidence.
    preparation.update(diagnostic_only=True, real_camera_training=False,
                       all_roles_have_fixture_train_and_val_samples=True,
                       all_roles_have_real_rig_train_and_val_samples=False,
                       rig_role_coverage_is_synthetic=True,
                       missing_real_rig_train_val=[f"{split}/{role}" for split in ("train", "val") for role in ROLES])
    preparation["limitations"].insert(0, "All images, masks and camera roles in this run are synthetic diagnostic fixtures; no real-rig coverage is established.")
    _write_json(root / "prepared/report.json", preparation)
    config = load_config(preparation["stage"])
    config["diagnostic_only"] = True
    config["training"].update(device="cpu", batch_size=2, num_workers=0, epochs=1,
                              samples_per_epoch=8, output_dir=str(root / "training"), amp=False)
    _write_json(Path(preparation["stage"]), config)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    try:
        # Match the trainer's initialization seed to confirm parameters really
        # change, in addition to checking the applied optimizer step counter.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["training"]["seed"])
            initial = build_model(config)
            initial_parameters = {name: value.detach().clone() for name, value in initial.named_parameters()}
        result = run_training(config)
        if not result["training_complete"] or result.get("diagnostic") is not True:
            raise RuntimeError("Expected completed diagnostic-only CPU training")
        checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint.get("diagnostic") is not True or checkpoint.get("optimizer_steps", 0) < 1:
            raise RuntimeError("Checkpoint must record diagnostic provenance and applied updates")
        changed = [name for name, value in initial_parameters.items() if not torch.equal(value, checkpoint["model"][name])]
        if not changed:
            raise RuntimeError("Optimizer did not change any shared-model parameters")
        metrics = evaluate_sources(config, Path(result["best_checkpoint"]))
    finally:
        torch.set_num_threads(previous_threads)
    expected_names = {entry["name"] for entry in config["data"]["val_manifests"]}
    if len(expected_names) != 8 or set(metrics) != expected_names:
        raise RuntimeError("Expected separate evaluation of all eight synthetic role holdouts")
    if any(value["partial_validation"] for value in metrics.values()):
        raise RuntimeError("Every synthetic role validation must finish")
    _write_json(root / "synthetic_role_metrics.json", {"diagnostic_only": True,
                "real_camera_training": False, "not_driving_evidence": True, "sources": metrics})
    verification = {"diagnostic_only": True, "training_complete": True, "device": "cpu",
                    "physical_output": False, "real_camera_training": False, "real_driving_validated": False,
                    "models_validated_for_control": False, "rig_id": rig_id,
                    "fixture_samples": {"train": 8, "val": 8},
                    "shared_model": {"instances_trained": 1, "width": 16, "image_size": [192, 320], "num_classes": 2},
                    "optimizer_steps": checkpoint["optimizer_steps"], "changed_parameter_tensors": len(changed),
                    "checkpoint_diagnostic": checkpoint["diagnostic"], "checkpoint": result["best_checkpoint"],
                    "role_holdouts_evaluated": list(ROLES), "validation_sources": sorted(expected_names),
                    "sampling_policy": "Eight replacement draws from eight fixture examples; availability of each role does not guarantee it was drawn during this tiny epoch.",
                    "metrics": str(root / "synthetic_role_metrics.json"),
                    "limits": "Synthetic fixtures verify sampler, optimizer, checkpoint and per-role evaluation interfaces only. No real cameras, synchronized surround perception or collision avoidance were trained or validated."}
    _write_json(root / "verification.json", verification)
    (root / "DIAGNOSTIC_INCOMPLETE").unlink()
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
