"""One tiny offline-fixture YOLO epoch; random initialization, not accuracy evaluation."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from camera_rc.prepare import prepare_bdd_detection
from camera_rc.detection import run_detection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for split, count in (("train", 6), ("val", 2)):
        images = root / split
        images.mkdir()
        annotations = []
        for i in range(count):
            name = f"{split}_{i}.jpg"
            image = Image.new("RGB", (96, 64), (30, 30, 30))
            ImageDraw.Draw(image).rectangle((20, 20, 60, 48), fill=(180, 60, 30))
            image.save(images / name)
            annotations.append({"name": name, "labels": [{"category": "car", "box2d": {"x1": 20, "y1": 20, "x2": 60, "y2": 48}}]})
        (root / f"{split}.json").write_text(json.dumps(annotations), encoding="utf-8")
    report = prepare_bdd_detection(root / "train.json", root / "val.json", root / "train", root / "val", root / "yolo")
    config = {"task": "detection", "model": "yolo11n.yaml", "data": str(root / "yolo/dataset.yaml"),
              "project": str(root / "training"), "name": "diagnostic", "epochs": 1, "imgsz": 64,
              "batch": 2, "workers": 0, "device": args.device, "amp": False, "pretrained": False,
              "cache": False, "plots": False, "save": True, "verbose": False,
              "mosaic": 0.0, "close_mosaic": 0, "seed": 42}
    result = run_detection(config)
    checkpoint = Path(result["save_dir"]) / "weights/best.pt"
    validation = run_detection(config, mode="val", checkpoint=checkpoint)
    summary = {"diagnostic_only": True, "random_initialization": True, "conversion": report,
               "train": result, "evaluation": validation}
    (root / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
