"""Paired descriptive deltas for reports sharing the exact evaluation contract."""
import argparse
import json
from pathlib import Path

from evaluation_contract import assert_comparable
from summarize_yolo_semantic_experiment import native_summary, public_summary


def compare_reports(baseline, candidate):
    assert_comparable(baseline, candidate)
    before = {"native_argmax": native_summary(baseline), "native_runtime_threshold": native_summary(baseline, threshold=True)}
    after = {"native_argmax": native_summary(candidate), "native_runtime_threshold": native_summary(candidate, threshold=True)}
    if baseline["public_obstacle_anomaly_subsets"]:
        before["public"] = public_summary(baseline)
        after["public"] = public_summary(candidate)
    deltas = []
    def compare(scope, old, new, metrics):
        for metric, direction in metrics:
            previous, current = old[metric], new[metric]
            delta = None if previous is None or current is None else current - previous
            deltas.append({"scope": scope, "metric": metric, "baseline": previous, "candidate": current,
                           "candidate_minus_baseline": delta, "preferred_direction": direction,
                           "regressed": None if delta is None else delta > 0 if direction == "lower" else delta < 0})
    native_fields = [(key, "lower" if key == "false_road_rate" else "higher")
                     for key in ("road_iou", "road_precision", "road_recall", "false_road_rate")]
    for condition in ("native_argmax", "native_runtime_threshold"):
        compare(condition + ":pooled", before[condition]["pooled"], after[condition]["pooled"], native_fields)
        for name, value in before[condition]["per_source"].items():
            compare(condition + ":" + name, value, after[condition]["per_source"][name], native_fields)
    if "public" in before:
        for track, value in before["public"].items():
            compare("public:" + track, value, after["public"][track], [
                (key, "lower") for key in ("micro_obstacle_as_road_rate", "mean_per_image_obstacle_as_road_rate",
                                            "all_obstacle_pixels_predicted_road_images", "at_least_half_obstacle_pixels_predicted_road_images")])
    return {"evaluation_contract": baseline["evaluation_contract"], "input_size": baseline["input_size"],
            "baseline_checkpoint": baseline["checkpoint"], "candidate_checkpoint": candidate["checkpoint"],
            "baseline_training_config": baseline["config"], "candidate_training_config": candidate["config"],
            "baseline": before, "candidate": after, "deltas": deltas,
            "regressions": [item for item in deltas if item["regressed"]],
            "statistical_significance_tested": False, "deployment_validated": False,
            "selection_note": "Descriptive reused development regression only. No threshold tuning or automatic deployment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_reports(json.loads(args.baseline.read_text(encoding="utf-8-sig")),
                             json.loads(args.candidate.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
