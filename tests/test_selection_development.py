import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_selection_development import check_model_partition, check_candidate_export_partitions, compare_reports, PROTOCOL


def test_baseline_exposure_is_explicit_and_candidate_forbidden():
    train, dev = [{"image_sha256": "a"}], [{"image_sha256": "b"}]
    assert check_model_partition("baseline", {"a", "b"}, train, dev) == 1
    assert check_model_partition("candidate", {"a"}, train, dev) == 0
    with pytest.raises(ValueError, match="exactly"):
        check_model_partition("candidate", {"a", "b"}, train, dev)
    with pytest.raises(ValueError, match="overlap"):
        check_model_partition("baseline", {"a"}, train, train)
    with pytest.raises(ValueError, match="cover"):
        check_model_partition("baseline", {"a"}, train, dev)


def metrics(matrix):
    (tp, fn), (fp, tn) = matrix
    return {"confusion_matrix": matrix, "valid_pixels": tp + fn + fp + tn,
            "road_iou": tp / (tp + fn + fp), "road_recall": tp / (tp + fn),
            "road_precision": tp / (tp + fp) if tp + fp else None,
            "false_road_rate": fp / (fp + tn), "road_fp_pixels": fp,
            "class_iou": [tp / (tp + fn + fp), tn / (tn + fn + fp)], "threshold": .65}


def set_matrix(report, name, matrix):
    report["native_sources"][name]["runtime_threshold_metrics"] = metrics(matrix)


def pair():
    base = {"protocol_id": PROTOCOL, "role": "baseline", "evaluation_complete": True,
            "prepared_report_sha256": "x", "dev_manifest_sha256": "y", "input_size": [384, 640],
            "native_gt_size": [192, 320], "runtime_threshold": .65, "precision_condition": "fp32",
            "native_sources": {"night": {"images": 2, "runtime_threshold_metrics": metrics([[90, 10], [10, 90]])}}}
    candidate = copy.deepcopy(base)
    candidate["role"] = "candidate"
    return base, candidate


def test_gate_requires_obstacle_rejection_and_road_coverage():
    a, b = pair()
    set_matrix(b, "night", [[90, 10], [11, 89]])
    assert not compare_reports(a, b)["development_gate_passed"]
    set_matrix(b, "night", [[0, 100], [0, 100]])
    assert not compare_reports(a, b)["development_gate_passed"]
    a, b = pair()
    result = compare_reports(a, b)
    assert not result["development_gate_passed"]
    assert any("strict" in reason for reason in result["failures"])
    set_matrix(b, "night", [[90, 10], [9, 91]])
    result = compare_reports(a, b)
    assert result["development_gate_passed"]
    assert result["sources_with_strict_false_road_decrease"] == ["night"]


def partitions():
    train = [{"image_sha256": "train-image", "target_sha256": "corrected-mask"}]
    dev = [{"image_sha256": "dev-image", "target_sha256": "dev-corrected-mask"}]
    exported = {split: [{"image_sha256": row["image_sha256"], "input_target_sha256": row["target_sha256"]}
                        for row in rows] for split, rows in (("train", train), ("val", dev))}
    return train, dev, exported


def test_candidate_input_targets_bind_both_corrected_training_and_selection_dev():
    train, dev, exported = partitions()
    check_candidate_export_partitions(exported, train, dev)
    exported["train"][0]["input_target_sha256"] = "old-uncorrected-mask"
    with pytest.raises(ValueError, match="corrected training"):
        check_candidate_export_partitions(exported, train, dev)
    train, dev, exported = partitions()
    exported["val"][0]["image_sha256"] = "official-validation-image"
    with pytest.raises(ValueError, match="selection development"):
        check_candidate_export_partitions(exported, train, dev)
    train, dev, exported = partitions()
    exported["val"][0]["input_target_sha256"] = "old-dev-mask"
    with pytest.raises(ValueError, match="selection development"):
        check_candidate_export_partitions(exported, train, dev)


@pytest.mark.parametrize("split", ["train", "val"])
def test_candidate_partition_preserves_multiplicity_and_rejects_missing_identities(split):
    train, dev, exported = partitions()
    exported[split].append(dict(exported[split][0]))
    with pytest.raises(ValueError, match="multiplicity"):
        check_candidate_export_partitions(exported, train, dev)
    train, dev, exported = partitions()
    exported[split][0].pop("input_target_sha256")
    with pytest.raises(ValueError, match="identities required"):
        check_candidate_export_partitions(exported, train, dev)


def test_one_source_improvement_cannot_hide_another_source_regression():
    a, b = pair()
    for item in (a, b):
        item["native_sources"]["day"] = copy.deepcopy(item["native_sources"]["night"])
    set_matrix(b, "night", [[90, 10], [9, 91]])
    set_matrix(b, "day", [[90, 10], [11, 89]])
    result = compare_reports(a, b)
    assert not result["development_gate_passed"]
    assert result["sources_with_strict_false_road_decrease"] == ["night"]


@pytest.mark.parametrize("key,value", [("input_size", [192, 320]), ("dev_manifest_sha256", "z"),
                                     ("runtime_threshold", .8), ("precision_condition", "fp16")])
def test_incomparable_reports_rejected(key, value):
    a, b = pair()
    b[key] = value
    with pytest.raises(ValueError, match="Unmatched"):
        compare_reports(a, b)


def test_denominator_and_undefined_metric_rejected():
    a, b = pair()
    b["native_sources"]["night"]["runtime_threshold_metrics"]["confusion_matrix"][0][0] = 91
    with pytest.raises(ValueError, match="denominator"):
        compare_reports(a, b)
    a, b = pair()
    b["native_sources"]["night"]["runtime_threshold_metrics"]["road_recall"] = None
    with pytest.raises(ValueError, match="Finite"):
        compare_reports(a, b)


@pytest.mark.parametrize("field,value", [("road_iou", .8), ("road_recall", .95),
                                        ("false_road_rate", .09), ("road_precision", .99),
                                        ("valid_pixels", 201), ("road_fp_pixels", 0),
                                        ("class_iou", [.8181818181818182, .99]), ("threshold", .7)])
def test_inconsistent_derived_json_cannot_change_gate(field, value):
    a, b = pair()
    b["native_sources"]["night"]["runtime_threshold_metrics"][field] = value
    with pytest.raises(ValueError, match="inconsistent|differs"):
        compare_reports(a, b)


@pytest.mark.parametrize("matrix", [[[90, 10], [10, -1]], [[90., 10], [10, 90]],
                                   [[True, 99], [10, 90]], [[90, 10]], [[90, 10, 0], [10, 90]],
                                   [[0, 0], [10, 90]], [[90, 10], [0, 0]]])
def test_invalid_confusion_and_absent_gt_denominators_rejected(matrix):
    a, b = pair()
    b["native_sources"]["night"]["runtime_threshold_metrics"]["confusion_matrix"] = matrix
    with pytest.raises(ValueError, match="confusion matrix|GT denominators"):
        compare_reports(a, b)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.1, 1.1])
def test_invalid_metric_values_rejected(value):
    a, b = pair()
    b["native_sources"]["night"]["runtime_threshold_metrics"]["false_road_rate"] = value
    with pytest.raises(ValueError, match="Finite"):
        compare_reports(a, b)


def test_canonical_rates_and_percentage_point_deltas_use_counts():
    a, b = pair()
    set_matrix(b, "night", [[89, 11], [9, 91]])  # exactly 1 pp recall decrease is allowed
    result = compare_reports(a, b)
    assert result["development_gate_passed"] and result["derived_metrics_recomputed_from_confusion"]
    actual = result["sources"]["night"]
    assert actual["road_iou"]["candidate"] == 89 / 109
    assert actual["false_road_rate"]["difference_pp"] == pytest.approx(-1.)
    assert actual["road_recall"]["difference_pp"] == pytest.approx(-1.)


def test_empty_sources_and_nonboolean_completion_rejected():
    a, b = pair()
    a["native_sources"] = b["native_sources"] = {}
    with pytest.raises(ValueError, match="Nonempty"):
        compare_reports(a, b)
    a, b = pair()
    b["evaluation_complete"] = "true"
    with pytest.raises(ValueError, match="Completed"):
        compare_reports(a, b)
