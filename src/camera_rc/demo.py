"""Small generated fixtures to verify code paths; never evidence of driving quality."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def make_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = {}
    for task in ("segmentation", "depth", "trajectory"):
        folder = output / task
        folder.mkdir()
        for split, count in (("train", 6), ("val", 2)):
            records = []
            for i in range(count):
                h, w = 48, 64
                yy, xx = np.mgrid[:h, :w]
                labels = ((xx + i * 2) // 22).clip(0, 2).astype(np.uint8)
                rgb = np.zeros((h, w, 3), np.uint8)
                for c in range(3):
                    rgb[:, :, c] = (labels == c) * 180 + 20
                image = folder / f"{split}_{i}.png"
                Image.fromarray(rgb).save(image)
                row = {"image": str(image), "group_id": f"demo_{split}", "dataset_id": "synthetic_diagnostic", "task": task}
                if task == "trajectory":
                    row.update(state=[0.5, 0.0], target=[[0.25 * (k + 1), 0.0, 0.5] for k in range(5)],
                               future_offsets_s=[0.5, 1, 1.5, 2, 2.5])
                else:
                    target = folder / f"{split}_{i}_target.png"
                    pixels = labels if task == "segmentation" else ((1 + yy / h * 3) * 256).astype(np.uint16)
                    Image.fromarray(pixels).save(target)
                    row["target"] = str(target)
                records.append(row)
            (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        cfg = {"task": task, "diagnostic_only": True, "model": {"width": 8, "num_classes": 3, "horizon": 5},
               "data": {"train_manifest": "train.jsonl", "val_manifest": "val.jsonl", "image_size": [48, 64],
                        "depth_scale": 256, "max_depth_m": 80},
               "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.001,
                            "seed": 42, "device": "auto", "amp": True, "grad_accum_steps": 2,
                            "output_dir": "run"}}
        cp = folder / "config.json"
        cp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        configs[task] = str(cp)
    return configs
