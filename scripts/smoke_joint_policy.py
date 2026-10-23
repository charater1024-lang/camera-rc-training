"""Run genuine optimizer updates on fabricated eight-view temporal sequences.

Only code integration is tested. Images, geometry, state and paths are synthetic;
neither accuracy on roads nor any vehicle-control ability is established.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from camera_rc.joint_policy.training import train_joint_policy, write_json


def create_fixture(root, views=8, train_samples=4, val_samples=2):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    optical_to_flu = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    dt = [.5, 1., 1.5, 2., 2.5]
    paths = {}
    for split, count in (("train", train_samples), ("val", val_samples)):
        rows = []
        for sample in range(count):
            frames = []
            speed = 1. + .1 * sample
            for index, offset in enumerate((-.2, -.1, 0.)):
                cameras = []
                for view in range(views):
                    image_path = root / f"{split}_{sample}_{index}_{view}.png"
                    pixels = np.full((64, 96, 3), (40 + view * 12, 50 + index * 20, 90 + sample * 10), dtype=np.uint8)
                    pixels[35:, 30:65] = (120, 120, 120)
                    Image.fromarray(pixels).save(image_path)
                    angle = view * 2 * np.pi / views
                    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                    transform = np.eye(4)
                    transform[:3, :3] = rotation @ optical_to_flu
                    transform[:3, 3] = [offset * speed, .2 * np.sin(angle), 0]
                    cameras.append({"camera_id": f"camera_{view}", "image": str(image_path),
                                    "image_size": [64, 96],
                                    "intrinsic": [[80, 0, 47.5], [0, 80, 31.5], [0, 0, 1]],
                                    "camera_to_anchor": transform.tolist()})
                frames.append({"timestamp_s": 10. + offset, "views": cameras})
            rows.append({"schema_version": 1, "dataset_id": "synthetic_joint_fixture",
                         "origin_id": f"synthetic_joint_fixture:drive_{split}",
                         "group_id": f"drive_{split}", "split": split,
                         "anchor_timestamp_s": 10., "frames": frames, "state": [speed, 0.],
                         "future_dt_s": dt, "targets": [[speed * value, 0., speed] for value in dt],
                         "diagnostic": True, "synthetic": True, "condition": "fabricated",
                         "timing_source": "fabricated_diagnostic"})
        manifest = root / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths[split] = str(manifest)
    return paths


def run_smoke(output, device="cpu"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = create_fixture(output / "fixtures")
    config = {"task": "joint_policy", "diagnostic_only": True, "models_validated_for_control": False,
              "model": {"width": 16, "embed_dim": 64, "token_grid": [3, 5], "horizon": 5,
                        "encoder_chunk_size": 1, "checkpoint_encoder": False},
              "data": {"train_manifest": paths["train"], "val_manifest": paths["val"],
                       "expected_history": 3, "expected_views": 8, "image_size": [64, 96],
                       "camera_names": [f"camera_{index}" for index in range(8)],
                       "future_dt_s": [.5, 1., 1.5, 2., 2.5]},
              "training": {"output_dir": str(output / "training"), "epochs": 1,
                           "batch_size": 1, "num_workers": 0, "lr": .001,
                           "device": device, "seed": 42, "amp": False}}
    write_json(output / "config.json", config)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        summary = train_joint_policy(config)
    finally:
        torch.set_num_threads(previous_threads)
    checks = {"training_complete": summary["training_complete"], "diagnostic_retained": summary["diagnostic"],
              "optimizer_applied": summary["optimizer_steps"] >= 1,
              "parameters_changed": summary["changed_parameter_tensors"] > 0,
              "eight_views_three_times": summary["trained_views"] == 8 and summary["history_frames"] == 3,
              "not_approved_for_control": summary["models_validated_for_control"] is False and summary["deployed"] is False}
    report = {"success": all(checks.values()), "checks": checks,
              "diagnostic_only": True, "real_camera_training": False, "physical_output": False,
              "checkpoint": summary["best_checkpoint"], "optimizer_steps": summary["optimizer_steps"],
              "changed_parameter_tensors": summary["changed_parameter_tensors"],
              "limits": "Fabricated sequences verify joint image/time/geometry model, optimizer and evaluation only; not a Tesla reproduction or evidence of real surround driving."}
    write_json(output / "verification.json", report)
    if not report["success"]:
        raise RuntimeError("Joint temporal smoke failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output, args.device), ensure_ascii=False, indent=2))
