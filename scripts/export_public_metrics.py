"""Export only aggregate evidence: no dataset images, masks or sample paths."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from evaluate_road_safety import INPUT_SIZE, LABEL_SPACE, aggregate_obstacle_metrics, road_metrics
from evaluation_contract import PROTOCOL, assert_comparable


NATIVE_DECISION = ("FP32 softmax road=0 probability >= runtime threshold; GT ignore=255 excluded. "
                   "This differs from argmax, whose equal-logit tie chooses road=0.")
PROBABILITY_PROTOCOL = {
    "rgb": "PIL RGB bilinear direct resize; float32 [0,1]",
    "native_gt": "nearest 192x320; ignore255 excluded; FP32 logits argmax and softmax road>=threshold",
    "public_gt": "original resolution GT1 only; GT0 usual is not assumed road",
    "public_probability": "FP32 softmax after input-size logits resize then bilinear probability to original GT",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, name):
    require(type(value) is int and value >= 0, f"Invalid nonnegative count: {name}")
    return value


def ratio_matches(value, expected, name):
    require((value is None and expected is None) or
            (expected is not None and type(value) in (int, float) and math.isfinite(value)
             and math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)),
            f"Stored metric differs from counts: {name}")


def native_metrics(matrix):
    metrics = road_metrics(matrix)
    tp, fn, fp = (metrics[name] for name in ("road_tp_pixels", "road_fn_pixels", "road_fp_pixels"))
    return {"road_iou": tp / (tp + fn + fp) if tp + fn + fp else None, **metrics}


def validate_report(report, expected_input=None):
    """Validate the raw frozen protocol and counts before discarding provenance."""
    require(report.get("evaluation_complete") is True and report.get("training_performed") is False
            and report.get("deployment_validated") is False, "Completed evaluation-only frozen report required")
    require(report.get("protocol_id") == PROTOCOL and report.get("label_space_id") == LABEL_SPACE
            and report.get("class_names") == ["road_surface", "not_road_surface"]
            and type(report.get("road_class_id")) is int and report["road_class_id"] == 0,
            "Canonical frozen road class/protocol required")
    require(report.get("input_size") in ([192, 320], [384, 640])
            and (expected_input is None or report["input_size"] == list(expected_input))
            and report.get("native_gt_size") == list(INPUT_SIZE), "Unexpected model input or native GT size")
    require(type(report.get("runtime_threshold")) in (int, float) and report["runtime_threshold"] == .65,
            "Fixed runtime threshold 0.65 required")
    require(report.get("precision_condition") == "autocast_fp16_logits_fp32_metrics", "Matched FP16/FP32 metric precision required")
    require(report.get("protocol") == PROBABILITY_PROTOCOL, "Actual probability/GT decision protocol differs")
    for name in ("checkpoint", "evaluation_contract"):
        value = report.get(name, {})
        require(isinstance(value, dict) and isinstance(value.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None, f"Invalid {name} SHA256")
    sources = report["native_sources"]
    require(isinstance(sources, dict) and sources, "Nonempty native sources required")
    declared = report["validation_sources"]
    require(len(declared) == len(sources) and {item["name"] for item in declared} == set(sources)
            and set(report["gt_denominators"]) == set(sources), "Native source-set mismatch")
    for name, source in sources.items():
        n = count(source["images"], name + ".images")
        require(n > 0 and next(item["images"] for item in declared if item["name"] == name) == n,
                "Native source image count differs")
        sizes = report["gt_denominators"][name]
        require(set(sizes) == {"0", "1", "255"}, "Canonical GT denominators required")
        sizes = {key: count(value, name + ".GT" + key) for key, value in sizes.items()}
        require(sum(sizes.values()) == n * INPUT_SIZE[0] * INPUT_SIZE[1], "Native GT grid denominator mismatch")
        current = source["runtime_threshold_metrics"]
        require(type(current.get("threshold")) in (int, float) and current["threshold"] == .65
                and current.get("comparison") == ">=" and current.get("decision") == NATIVE_DECISION,
                "Native threshold decision differs")
        for matrix in (source["confusion_matrix"], current["confusion_matrix"]):
            canonical = native_metrics(matrix)
            require([sum(row) for row in matrix] == [sizes["0"], sizes["1"]], "Native confusion denominator mismatch")
        for key in ("road_iou", "road_precision", "road_recall", "false_road_rate"):
            ratio_matches(current[key], canonical[key], name + "." + key)
    tracks = report["public_obstacle_anomaly_subsets"]
    require(set(tracks) == {"obstacle", "anomaly"}, "Public obstacle30/anomaly10 required")
    for name, expected_images in (("obstacle", 30), ("anomaly", 10)):
        track, rows = tracks[name], tracks[name]["per_image"]
        require(len(rows) == expected_images, "Public track image count differs")
        for row in rows:
            labelled = count(row["labelled_obstacle_pixels"], "public labelled pixels")
            require(labelled > 0, "Every public frame requires labelled obstacles")
            count(row["ignored_pixels"], "public ignored pixels")
            count(row["usual_pixels_not_used_as_road_gt"], "public usual pixels")
            for decision_name, threshold, comparison in (("road_p_gt_0_5", .5, ">"), ("runtime_road", .65, ">=")):
                decision = row["decisions"][decision_name]
                require(type(decision.get("threshold")) in (int, float) and decision["threshold"] == threshold
                        and decision.get("comparison") == comparison, "Public threshold decision differs")
                errors = count(decision["obstacle_pixels_predicted_road"], "public false road pixels")
                require(errors <= labelled and type(decision["labelled_obstacle_pixels"]) is int
                        and decision["labelled_obstacle_pixels"] == labelled, "Public per-image denominator mismatch")
                ratio_matches(decision["obstacle_as_road_rate"], errors / labelled, "public per-image rate")
        canonical = aggregate_obstacle_metrics(rows, .65)
        require(track["aggregate"] == canonical, "Public aggregate differs from per-image counts")


def read_report(path, expected_input=None):
    raw = Path(path).read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    validate_report(report, expected_input)
    return report, hashlib.sha256(raw).hexdigest()


def pooled(sources):
    matrix = np.sum([s["runtime_threshold_metrics"]["confusion_matrix"] for s in sources], axis=0)
    tp, fn, fp, tn = map(int, matrix.ravel())
    return {"images": sum(s["images"] for s in sources), "confusion_matrix": matrix.tolist(),
            "road_iou": tp / (tp + fn + fp) if tp + fn + fp else None, **road_metrics(matrix),
            "decision": "road softmax probability >= 0.65 on the native GT grid"}


def aggregate_report(report, source_sha):
    native = report["native_sources"]
    public = {}
    for name, track in report["public_obstacle_anomaly_subsets"].items():
        rows = track["per_image"]
        rates = [r["decisions"]["runtime_road"]["obstacle_as_road_rate"] for r in rows
                 if r["labelled_obstacle_pixels"] > 0]
        public[name] = {"aggregate": aggregate_obstacle_metrics(rows, .65),
                        "images_all_labelled_pixels_predicted_road": sum(x == 1 for x in rates),
                        "images_at_least_half_labelled_pixels_predicted_road": sum(x >= .5 for x in rates)}
    # Explicit allowlist: no raw per-image records or arbitrary report metadata.
    return {"source_report_sha256": source_sha,
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "evaluation_contract_sha256": report["evaluation_contract"]["sha256"],
            "input_size": report["input_size"], "native_gt_size": report["native_gt_size"],
            "precision_condition": report["precision_condition"], "runtime_threshold": .65,
            "native_sources": {name: {"images": s["images"],
                "confusion_matrix_argmax": s["confusion_matrix"],
                "runtime_threshold_metrics": {k: native_metrics(s["runtime_threshold_metrics"]["confusion_matrix"])[k]
                    if k != "confusion_matrix" else s["runtime_threshold_metrics"][k] for k in
                    ("confusion_matrix", "road_iou", "road_precision", "road_recall", "false_road_rate")}}
                for name, s in native.items()},
            "pooled_native": pooled(list(native.values())),
            "pooled_legacy": pooled([v for k, v in native.items() if not k.startswith("aihub_")]),
            "pooled_aihub": pooled([v for k, v in native.items() if k.startswith("aihub_")]),
            "public_reused_regression": public}


def aggregate(path):
    report, source_sha = read_report(path)
    return aggregate_report(report, source_sha)


def build_models(evaluation_dir, corrected_report=None):
    models, originals = {}, {}
    for size in ("192x320", "384x640"):
        expected_input = [int(value) for value in size.split("x")]
        reference, seen_checkpoints = None, set()
        for name in ("tinyunet", "yolo_v2", "aihub"):
            key = f"{name}_{size}"
            report, source_sha = read_report(Path(evaluation_dir) / f"evaluate_{key}.json", expected_input)
            if reference is not None:
                assert_comparable(reference, report)
            reference = report
            require(report["checkpoint"]["sha256"] not in seen_checkpoints, "Distinct comparison checkpoints required")
            seen_checkpoints.add(report["checkpoint"]["sha256"])
            originals[key], models[key] = report, aggregate_report(report, source_sha)
    if corrected_report is not None:
        report, source_sha = read_report(corrected_report, [384, 640])
        assert_comparable(originals["aihub_384x640"], report)
        require(report["checkpoint"]["sha256"] not in {
            item["checkpoint"]["sha256"] for key, item in originals.items() if key.endswith("_384x640")},
            "Distinct corrected candidate checkpoint required")
        models["corrected_labels_384x640"] = aggregate_report(report, source_sha)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--corrected-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Preserve an existing public evidence version")
    models = build_models(args.evaluation_dir, args.corrected_report)
    result = {"schema_version": 1, "models": models, "speed_comparison": "cancelled; not measured",
              "real_vehicle_validated": False, "weights_included": False,
              "notice": "Descriptive pixel metrics on reused reporting sets, not independent safety/collision probabilities. All local file paths and individual dataset samples excluded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
