"""Synthetic report validation only; no image, checkpoint, GPU or network use."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_public_metrics as export


def identity(name):
    return {"path": f"/private/{name}", "sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": 1}


def native_source(matrix=None):
    matrix = matrix or [[80, 20], [40, 60]]
    values = export.native_metrics(matrix)
    return {"images": 1, "confusion_matrix": [[85, 15], [50, 50]],
            "runtime_threshold_metrics": {"confusion_matrix": matrix, **values,
                "threshold": .65, "comparison": ">=", "decision": export.NATIVE_DECISION}}


def public_track(name, images):
    rows = []
    for index in range(images):
        decisions = {}
        for key, threshold, comparison, errors in (("road_p_gt_0_5", .5, ">", 5), ("runtime_road", .65, ">=", 4)):
            decisions[key] = {"threshold": threshold, "comparison": comparison, "obstacle_pixels_predicted_road": errors,
                              "labelled_obstacle_pixels": 10, "obstacle_as_road_rate": errors / 10}
        rows.append({"frame_id": f"{name}_{index}", "image": identity(f"{name}_image_{index}"),
                     "ground_truth": identity(f"{name}_target_{index}"), "labelled_obstacle_pixels": 10,
                     "usual_pixels_not_used_as_road_gt": 20, "ignored_pixels": 30, "decisions": decisions})
    return {"per_image": rows, "aggregate": export.aggregate_obstacle_metrics(rows, .65)}


def report(model, size=(384, 640)):
    sources = {"bdd": native_source(), "aihub_13_day_city": native_source()}
    return {"evaluation_complete": True, "training_performed": False, "deployment_validated": False,
            "protocol_id": export.PROTOCOL, "label_space_id": export.LABEL_SPACE,
            "class_names": ["road_surface", "not_road_surface"], "road_class_id": 0,
            "input_size": list(size), "native_gt_size": list(export.INPUT_SIZE), "runtime_threshold": .65,
            "precision_condition": "autocast_fp16_logits_fp32_metrics", "protocol": dict(export.PROBABILITY_PROTOCOL),
            "checkpoint": identity(model + "_weights"), "config": identity(model + "_training_config"),
            "evaluation_contract": identity("shared_contract"), "native_sources": sources,
            "validation_sources": [{"name": key, "images": 1, "manifest": identity(key)} for key in sources],
            "gt_denominators": {key: {"0": 100, "1": 100, "255": 61240} for key in sources},
            "public_obstacle_anomaly_subsets": {"obstacle": public_track("obstacle", 30), "anomaly": public_track("anomaly", 10)}}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def reports(tmp_path):
    root = tmp_path / "evaluation"
    root.mkdir()
    for size in ((192, 320), (384, 640)):
        for model in ("tinyunet", "yolo_v2", "aihub"):
            write(root / f"evaluate_{model}_{size[0]}x{size[1]}.json", report(model, size))
    candidate = tmp_path / "candidate.json"
    value = report("corrected")
    value["native_sources"]["aihub_13_day_city"] = native_source([[75, 25], [20, 80]])
    write(candidate, value)
    return root, candidate, value


def test_valid_matched_candidate_retains_schema_and_canonical_ratios(reports):
    root, candidate, _ = reports
    models = export.build_models(root, candidate)
    assert len(models) == 7
    current = models["corrected_labels_384x640"]
    assert current["input_size"] == [384, 640]
    assert current["pooled_native"]["confusion_matrix"] == [[155, 45], [60, 140]]
    assert current["pooled_native"]["road_iou"] == 155 / 260
    assert current["public_reused_regression"]["obstacle"]["aggregate"]["decisions"]["runtime_road"]["micro_obstacle_as_road_rate"] == .4
    # Different training configs remain valid; only frozen evaluation conditions
    # and the explicit checkpoint identity are paired.
    assert "/private/" not in json.dumps(current)
    assert '"per_image":' not in json.dumps(current)
    from plot_accuracy_tradeoff import validate_report
    values, _ = validate_report({"schema_version": 1, "models": models,
                                "speed_comparison": "cancelled; not measured", "real_vehicle_validated": False})
    assert values[-1]["key"] == "corrected_labels_384x640"


@pytest.mark.parametrize("mutation", [
    "input", "gt_size", "contract", "top_threshold", "native_threshold", "comparison", "decision",
    "class_id", "classes", "probability_protocol", "precision", "gt_denominator", "confusion_denominator",
    "stored_iou", "stored_recall", "stored_fpr", "nan_metric", "public_comparison", "public_threshold",
    "public_per_image_rate", "public_per_image_denominator", "public_aggregate", "public_identity",
    "same_checkpoint", "source_images", "incomplete", "training_performed",
    "paired_gt_denominator", "paired_public_denominator",
])
def test_mismatched_candidate_rejected_before_public_output(reports, tmp_path, mutation):
    root, candidate, value = reports
    source = value["native_sources"]["bdd"]
    metric = source["runtime_threshold_metrics"]
    public = value["public_obstacle_anomaly_subsets"]["obstacle"]
    decision = public["per_image"][0]["decisions"]["runtime_road"]
    if mutation == "input": value["input_size"] = [192, 320]
    elif mutation == "gt_size": value["native_gt_size"] = [384, 640]
    elif mutation == "contract": value["evaluation_contract"] = identity("other_contract")
    elif mutation == "top_threshold": value["runtime_threshold"] = .6
    elif mutation == "native_threshold": metric["threshold"] = .6
    elif mutation == "comparison": metric["comparison"] = ">"
    elif mutation == "decision": metric["decision"] = "argmax"
    elif mutation == "class_id": value["road_class_id"] = 1
    elif mutation == "classes": value["class_names"].reverse()
    elif mutation == "probability_protocol": value["protocol"]["native_gt"] = "original GT"
    elif mutation == "precision": value["precision_condition"] = "fp32"
    elif mutation == "gt_denominator": value["gt_denominators"]["bdd"]["0"] += 1
    elif mutation == "confusion_denominator": metric["confusion_matrix"][0][0] += 1
    elif mutation == "stored_iou": metric["road_iou"] = .99
    elif mutation == "stored_recall": metric["road_recall"] = .99
    elif mutation == "stored_fpr": metric["false_road_rate"] = .99
    elif mutation == "nan_metric": metric["road_iou"] = float("nan")
    elif mutation == "public_comparison": decision["comparison"] = ">"
    elif mutation == "public_threshold": decision["threshold"] = .6
    elif mutation == "public_per_image_rate": decision["obstacle_as_road_rate"] = .99
    elif mutation == "public_per_image_denominator": decision["labelled_obstacle_pixels"] += 1
    elif mutation == "public_aggregate": public["aggregate"]["decisions"]["runtime_road"]["micro_obstacle_as_road_rate"] = .99
    elif mutation == "public_identity": public["per_image"][0]["image"] = identity("different_image_same_pixel_counts")
    elif mutation == "same_checkpoint": value["checkpoint"] = report("aihub")["checkpoint"]
    elif mutation == "source_images": source["images"] = 2
    elif mutation == "paired_gt_denominator":
        value["native_sources"]["bdd"] = native_source([[90, 20], [40, 50]])
        value["native_sources"]["bdd"]["confusion_matrix"] = [[95, 15], [50, 40]]
        value["gt_denominators"]["bdd"] = {"0": 110, "1": 90, "255": 61240}
        export.validate_report(value)  # Internally valid; differs from the paired cohort.
    elif mutation == "paired_public_denominator":
        row = public["per_image"][0]
        row["labelled_obstacle_pixels"] = 11
        for current in row["decisions"].values():
            current["labelled_obstacle_pixels"] = 11
            current["obstacle_as_road_rate"] = current["obstacle_pixels_predicted_road"] / 11
        public["aggregate"] = export.aggregate_obstacle_metrics(public["per_image"], .65)
        export.validate_report(value)
    elif mutation == "incomplete": value["evaluation_complete"] = False
    else: value["training_performed"] = True
    write(candidate, value)
    output = tmp_path / "not_created" / "public.json"
    with pytest.raises(ValueError):
        export.main(["--evaluation-dir", str(root), "--corrected-report", str(candidate), "--output", str(output)])
    assert not output.parent.exists()


def test_unmatched_existing_baseline_also_rejected(reports):
    root, _, _ = reports
    value = report("yolo_v2", (192, 320))
    value["evaluation_contract"] = identity("different_contract")
    write(root / "evaluate_yolo_v2_192x320.json", value)
    with pytest.raises(ValueError, match="evaluation_contract"):
        export.build_models(root)


def test_valid_export_does_not_rewrite_reports_or_overwrite_existing_output(reports, tmp_path):
    root, candidate, _ = reports
    before = {path: path.read_bytes() for path in [*root.glob("*.json"), candidate]}
    output = tmp_path / "public.json"
    argv = ["--evaluation-dir", str(root), "--corrected-report", str(candidate), "--output", str(output)]
    export.main(argv)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1 and len(result["models"]) == 7
    assert result["speed_comparison"] == "cancelled; not measured"
    assert result["real_vehicle_validated"] is False
    with pytest.raises(FileExistsError):
        export.main(argv)
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_counts_recompute_nearby_stored_ratio_without_mutating_report(tmp_path):
    value = report("aihub")
    value["native_sources"]["bdd"]["runtime_threshold_metrics"]["road_recall"] += 1e-13
    path = tmp_path / "report.json"
    write(path, value)
    result = export.aggregate(path)
    assert result["native_sources"]["bdd"]["runtime_threshold_metrics"]["road_recall"] == .8
    assert json.loads(path.read_text())["native_sources"]["bdd"]["runtime_threshold_metrics"]["road_recall"] != .8
