"""Export only aggregate evidence: no dataset images, masks or sample paths."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_road_safety import road_metrics


def pooled(sources):
    matrix = np.sum([s["runtime_threshold_metrics"]["confusion_matrix"] for s in sources], axis=0)
    tp, fn, fp, tn = map(int, matrix.ravel())
    return {"images": sum(s["images"] for s in sources), "confusion_matrix": matrix.tolist(),
            "road_iou": tp / (tp + fn + fp) if tp + fn + fp else None, **road_metrics(matrix),
            "decision": "road softmax probability >= 0.65 on the native GT grid"}


def aggregate(path):
    raw = Path(path).read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    if report.get("evaluation_complete") is not True or report.get("runtime_threshold") != .65:
        raise ValueError("Completed fixed-threshold frozen report required")
    native = report["native_sources"]
    public = {}
    for name, track in report["public_obstacle_anomaly_subsets"].items():
        rows = track["per_image"]
        rates = [r["decisions"]["runtime_road"]["obstacle_as_road_rate"] for r in rows
                 if r["labelled_obstacle_pixels"] > 0]
        public[name] = {"aggregate": track["aggregate"],
                        "images_all_labelled_pixels_predicted_road": sum(x == 1 for x in rates),
                        "images_at_least_half_labelled_pixels_predicted_road": sum(x >= .5 for x in rates)}
    # Explicit allowlist: no raw per-image records or arbitrary report metadata.
    return {"source_report_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "evaluation_contract_sha256": report["evaluation_contract"]["sha256"],
            "input_size": report["input_size"], "native_gt_size": report["native_gt_size"],
            "precision_condition": report["precision_condition"], "runtime_threshold": .65,
            "native_sources": {name: {"images": s["images"],
                "confusion_matrix_argmax": s["confusion_matrix"],
                "runtime_threshold_metrics": {k: s["runtime_threshold_metrics"][k] for k in
                    ("confusion_matrix", "road_iou", "road_precision", "road_recall", "false_road_rate")}}
                for name, s in native.items()},
            "pooled_native": pooled(list(native.values())),
            "pooled_legacy": pooled([v for k, v in native.items() if not k.startswith("aihub_")]),
            "pooled_aihub": pooled([v for k, v in native.items() if k.startswith("aihub_")]),
            "public_reused_regression": public}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--corrected-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve an existing public evidence version")
    models = {}
    for size in ("192x320", "384x640"):
        for name in ("tinyunet", "yolo_v2", "aihub"):
            models[f"{name}_{size}"] = aggregate(args.evaluation_dir / f"evaluate_{name}_{size}.json")
    if args.corrected_report:
        models["corrected_labels_384x640"] = aggregate(args.corrected_report)
    result = {"schema_version": 1, "models": models, "speed_comparison": "cancelled; not measured",
              "real_vehicle_validated": False, "weights_included": False,
              "notice": "Descriptive pixel metrics on reused reporting sets, not independent safety/collision probabilities. All local file paths and individual dataset samples excluded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
