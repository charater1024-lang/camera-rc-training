"""Synthetic report consistency fixtures; no model, GPU, or measured results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("semantic_summary_test", SCRIPTS / "summarize_yolo_semantic_experiment.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def metric(matrix):
    result = summary.confusion_metrics(matrix)
    tp, fn, fp, tn = [value for row in matrix for value in row]
    result.update(class_iou=[result["road_iou"], summary.ratio(tn, tn + fn + fp)], partial_validation=False)
    result["road_safety_pixels"] = {"false_road_rate": result["false_road_rate"]}
    return result


def public_track(name, images, errors):
    rows = []
    for i in range(images):
        decision = {"threshold": .65, "comparison": ">=", "obstacle_pixels_predicted_road": errors,
                    "labelled_obstacle_pixels": 100, "obstacle_as_road_rate": errors / 100}
        rows.append({"frame_id": str(i), "image": {"path": f"fixture_{name}_{i}.png", "sha256": "a" * 64, "bytes": 10},
                     "ground_truth": {"path": f"fixture_gt_{name}_{i}.png", "sha256": "b" * 64, "bytes": 10},
                     "labelled_obstacle_pixels": 100, "decisions": {"runtime_road": decision}})
    return {"dataset": name, "aggregate": {"images": images, "decisions": {"runtime_road": {
        "threshold": .65, "comparison": ">=", "obstacle_pixels_predicted_road": errors * images,
        "labelled_obstacle_pixels": 100 * images, "micro_obstacle_as_road_rate": errors / 100,
        "mean_per_image_obstacle_as_road_rate": errors / 100, "images_with_labelled_obstacles": images}}}, "per_image": rows}


def set_public_counts(track, counts):
    """Independent fixture construction, including empty GT and unequal mask sizes."""
    rates = []
    errors = total = 0
    assert len(track["per_image"]) == len(counts)
    for row, (numerator, denominator) in zip(track["per_image"], counts):
        row["labelled_obstacle_pixels"] = denominator
        row["decisions"]["runtime_road"].update(
            obstacle_pixels_predicted_road=numerator, labelled_obstacle_pixels=denominator,
            obstacle_as_road_rate=numerator / denominator if denominator else None)
        if denominator:
            rates.append(numerator / denominator)
        errors += numerator
        total += denominator
    track["aggregate"]["decisions"]["runtime_road"].update(
        obstacle_pixels_predicted_road=errors, labelled_obstacle_pixels=total,
        micro_obstacle_as_road_rate=errors / total,
        mean_per_image_obstacle_as_road_rate=sum(rates) / len(rates),
        images_with_labelled_obstacles=len(rates))


def test_tiny_completely_missed_obstacle_is_not_hidden_by_large_clean_mask():
    obstacle = public_track("RoadObstacle21", 30, 0)
    set_public_counts(obstacle, [(0, 1_000_000), (1, 1)] + [(0, 0)] * 28)
    report = {"runtime_threshold": .65, "public_obstacle_anomaly_subsets": {
        "obstacle": obstacle, "anomaly": public_track("RoadAnomaly21", 10, 0)}}
    result = summary.public_summary(report)["obstacle"]
    assert result["micro_obstacle_as_road_rate"] == pytest.approx(1 / 1_000_001)
    assert result["mean_per_image_obstacle_as_road_rate"] == .5
    assert result["images_with_labelled_obstacles"] == 2
    assert result["empty_gt_images_excluded_from_image_statistics"] == 28
    assert result["all_obstacle_pixels_predicted_road_images"] == 1
    assert result["at_least_half_obstacle_pixels_predicted_road_images"] == 1
    assert result["worst_frames"][0] == {"frame_id": "1", "obstacle_pixels_predicted_road": 1,
        "labelled_obstacle_pixels": 1, "obstacle_as_road_rate": 1.0}
    assert len(result["worst_frames"]) == 2


def test_image_half_wrong_count_includes_exact_half_and_excludes_empty_gt():
    obstacle = public_track("RoadObstacle21", 30, 0)
    set_public_counts(obstacle, [(1, 2), (1, 3), (2, 2)] + [(0, 0)] * 27)
    result = summary.public_summary({"runtime_threshold": .65, "public_obstacle_anomaly_subsets": {
        "obstacle": obstacle, "anomaly": public_track("RoadAnomaly21", 10, 0)}})["obstacle"]
    assert result["at_least_half_obstacle_pixels_predicted_road_images"] == 2
    assert result["all_obstacle_pixels_predicted_road_images"] == 1
    assert result["images_with_labelled_obstacles"] == 3
    assert [row["frame_id"] for row in result["worst_frames"]] == ["2", "0", "1"]


@pytest.mark.parametrize("change", ["mean", "nonempty_count", "frame_rate", "empty_rate", "duplicate_frame"])
def test_image_statistics_reject_inconsistent_evidence(change):
    obstacle = public_track("RoadObstacle21", 30, 0)
    set_public_counts(obstacle, [(0, 1_000_000), (1, 1)] + [(0, 0)] * 28)
    aggregate = obstacle["aggregate"]["decisions"]["runtime_road"]
    if change == "mean": aggregate["mean_per_image_obstacle_as_road_rate"] = aggregate["micro_obstacle_as_road_rate"]
    if change == "nonempty_count": aggregate["images_with_labelled_obstacles"] = 30
    if change == "frame_rate": obstacle["per_image"][1]["decisions"]["runtime_road"]["obstacle_as_road_rate"] = 0
    if change == "empty_rate": obstacle["per_image"][2]["decisions"]["runtime_road"]["obstacle_as_road_rate"] = 0
    if change == "duplicate_frame": obstacle["per_image"][1]["frame_id"] = "0"
    with pytest.raises(ValueError):
        summary.public_summary({"runtime_threshold": .65, "public_obstacle_anomaly_subsets": {
            "obstacle": obstacle, "anomaly": public_track("RoadAnomaly21", 10, 0)}})


def benchmark_fixture(training_id, best, native, config, size, views):
    modes = ["model_only", "host_to_road_map"] + (["batched_model_only", "batched_host_to_road_map"] if views == 8 else [])
    models = {}
    for name in ("yolo26x_sem", "tinyunet"):
        item = {"status": "completed", "out_of_memory": False,
                "resources": {"torch_peak_allocated_gib": 2., "torch_peak_reserved_gib": 2.1,
                              "process_gpu_peak_gib_sampled": None, "process_rss_peak_gib_sampled": 3.}}
        for mode in modes:
            item[mode] = {"measured_cycles": 50, "views_per_cycle": views, "mean_cycle_ms": 100.,
                          "p50_cycle_ms": 90., "p95_cycle_ms": 110., "p99_cycle_ms": 120., "max_cycle_ms": 130.,
                          "images_per_second": views * 10., "camera_work_cycles_per_second": 10.,
                          "batch_size_per_call": views if mode.startswith("batched_") else 1,
                          "model_calls_per_cycle": 1 if mode.startswith("batched_") else views,
                          "cuda_memory": {"torch_peak_allocated_gib": .5, "torch_peak_reserved_gib": .75}}
        models[name] = item
    return {"schema_version": 1, "status": "completed", "models": models,
            **{key: False for key in ("physical_output", "accuracy_evaluated", "driving_validated", "evidence_for_80kmh", "target_hardware_emulated", "training_performed")},
            "provenance": {"yolo": {"checkpoint": best, "training_report": training_id}, "native_checkpoint": native, "native_config": config},
            "views": views, "input_size_hw": list(size), "precision": "fp16", "iterations": 50,
            "input_images": [{"fixture": i} for i in range(views)],
            "hardware": {"actual_gpu_name": "Synthetic contract fixture GPU (not measured)", "requested_allocator_budget_gib": 3}}


def experiment_fixture(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    paths = {}
    for name in ("best.pt", "last.pt", "native.pt", "config.json", "dataset_report.json"):
        path = root / name
        path.write_bytes(name.encode())
        paths[name] = summary.file_identity(path)
    save(root / "status.json", {"status": "complete", "deployment_approved": False,
                               "suspended_queue": {"resumed": True}, "queue_temporarily_suspended": False})
    training = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic", "class_names": summary.NAMES,
                "road_class_id": 0, "training_complete": True, "diagnostic": False, "history": [{"epoch": 1}],
                "actual_args": {"epochs": 1}, "progress": {"optimizer_updates": 2, "train_batches": 2, "amp_skipped_updates": 0},
                "batches_per_epoch": 2, "checkpoints": {"best": paths["best.pt"], "last": paths["last.pt"]},
                "source_config": paths["config.json"], "dataset": {"provenance_path": paths["dataset_report.json"]["path"],
                    "provenance_sha256": paths["dataset_report.json"]["sha256"], "counts": {"train": 10, "val": 3}}}
    training_path = root / "training/training_report.json"
    save(training_path, training)
    training_id = summary.file_identity(training_path)
    metadata = [{"name": "a", "images": 1, "manifest": {"fixture": "a"}}, {"name": "b", "images": 2, "manifest": {"fixture": "b"}}]
    reports = {}
    for name, matrices, errors, size in (
        ("baseline", [[[20, 20], [5, 55]], [[100, 0], [50, 50]]], 20, [192, 320]),
        ("paired", [[[35, 5], [3, 57]], [[70, 30], [20, 80]]], 15, [192, 320]),
        ("higher", [[[38, 2], [2, 58]], [[90, 10], [15, 85]]], 14, [384, 640]),
    ):
        report = {"evaluation_only": True, "training_performed": False, "deployment_validated": False,
                  "checkpoint_diagnostic": False, "checkpoint": paths["native.pt"] if name == "baseline" else paths["best.pt"],
                  "config": paths["config.json"], "input_size": size, "label_space_id": "road_surface_v1", "road_class_id": 0,
                  "runtime_threshold": .65, "validation_sources": metadata, "amp_requested": True,
                  "native_sources": {key: metric(matrix) for key, matrix in zip(("a", "b"), matrices)},
                  "public_obstacle_anomaly_subsets": {"obstacle": public_track("RoadObstacle21", 30, errors), "anomaly": public_track("RoadAnomaly21", 10, errors)}}
        if name != "baseline":
            report.update(model_family="yolo26x-sem", task="semantic", class_names=summary.NAMES, native_gt_size=[192, 320],
                          models_validated_for_control=False, real_performance_validated=False,
                          training_completion={"report": training_id}, amp_effective=True)
            for source, matrix in zip(report["native_sources"].values(), ([[20, 20], [1, 59]], [[60, 40], [2, 98]])):
                source["runtime_threshold_metrics"] = {**metric(matrix), "threshold": .65, "comparison": ">="}
        reports[name] = report
    save(root / "baseline.json", reports["baseline"])
    save(root / "evaluation/paired_192x320.json", reports["paired"])
    save(root / "evaluation/candidate_384x640.json", reports["higher"])
    comparison = summary.compare_reports(reports["baseline"], reports["paired"])
    comparison.update(baseline_report=str(root / "baseline.json"), candidate_report=str(root / "evaluation/paired_192x320.json"))
    save(root / "evaluation/comparison.json", comparison)
    for h, w in ((192, 320), (384, 640)):
        for views in (1, 8):
            save(root / f"benchmarks/benchmark_{h}x{w}_views{views}_3gib.json", benchmark_fixture(training_id, paths["best.pt"], paths["native.pt"], paths["config.json"], (h, w), views))
    return root


def test_pooled_confusion_is_pixel_weighted_not_mean_source_iou(tmp_path):
    root = experiment_fixture(tmp_path)
    result = summary.collect(root)
    native = result["accuracy"]["tinyunet_192x320"]["native"]
    assert native["pooled"]["confusion_matrix"] == [[120, 20], [55, 105]]
    assert native["pooled"]["road_iou"] == pytest.approx(120 / 195)
    assert native["pooled"]["road_iou"] != pytest.approx(sum(x["road_iou"] for x in native["per_source"].values()) / 2)
    assert result["accuracy"]["yolo26x_sem_192x320"]["public_obstacle_pixels"]["obstacle"]["obstacle_pixels_predicted_road"] == 450
    assert result["accuracy"]["yolo26x_sem_192x320"]["public_obstacle_pixels"]["obstacle"]["labelled_obstacle_pixels"] == 3000
    assert len(result["benchmarks"]) == 24
    assert all(row["memory"]["torch_peak_allocated_gib"] == .5 for row in result["benchmarks"])
    assert all(row["cuda_memory_scope"] == "mode-specific reset peak" for row in result["benchmarks"])
    assert not result["msi_prestige_speed_measured"] and not result["zero_accident_guarantee"]


def test_source_iou_and_recall_regressions_are_visible_despite_better_pooled_iou(tmp_path):
    result = summary.collect(experiment_fixture(tmp_path))
    baseline = result["accuracy"]["tinyunet_192x320"]
    candidate = result["accuracy"]["yolo26x_sem_192x320"]
    assert candidate["native"]["pooled"]["road_iou"] > baseline["native"]["pooled"]["road_iou"]
    # The legacy comparator flags FPR, which improved; its behavior remains unchanged.
    assert result["paired_comparison"]["regressions"] == []
    regressions = {(row["scope"], row["metric"]): row for row in result["all_metric_regressions"]}
    assert ("native:b", "road_iou") in regressions
    assert ("native:b", "road_recall") in regressions
    assert regressions[("native:b", "road_recall")]["baseline"] == 1
    assert regressions[("native:b", "road_recall")]["candidate"] == .7
    assert not any(row["statistical_significance_tested"] for row in regressions.values())
    text = summary.korean_markdown(result)
    assert "TinyUNet IoU / 재현율" in text
    assert "b · 도로 IoU: 66.67% → 58.33%" in text
    assert "b · 도로 재현율: 100.00% → 70.00%" in text


def test_public_regressions_include_image_statistics_even_when_micro_improves(tmp_path):
    result = summary.collect(experiment_fixture(tmp_path))
    baseline = result["accuracy"]["tinyunet_192x320"]
    candidate = deepcopy(result["accuracy"]["yolo26x_sem_192x320"])
    candidate["public_obstacle_pixels"]["obstacle"].update(
        mean_per_image_obstacle_as_road_rate=.8,
        all_obstacle_pixels_predicted_road_images=5,
        at_least_half_obstacle_pixels_predicted_road_images=7)
    regressions = {(row["scope"], row["metric"]) for row in summary.paired_metric_regressions(baseline, candidate)}
    assert ("public:obstacle", "micro_obstacle_as_road_rate") not in regressions
    assert ("public:obstacle", "mean_per_image_obstacle_as_road_rate") in regressions
    assert ("public:obstacle", "all_obstacle_pixels_predicted_road_images") in regressions
    assert ("public:obstacle", "at_least_half_obstacle_pixels_predicted_road_images") in regressions


@pytest.mark.parametrize("change", ["running", "queue_suspended", "training_incomplete", "diagnostic", "no_updates", "different_checkpoint", "partial_eval", "missing_benchmark", "bogus_fps", "bad_public_count", "wrong_threshold_denominator"])
def test_incomplete_or_inconsistent_evidence_never_creates_summary(tmp_path, change):
    root = experiment_fixture(tmp_path)
    if change in {"running", "queue_suspended"}:
        path = root / "status.json"
        value = json.loads(path.read_text())
        if change == "running": value["status"] = "running"
        else: value["suspended_queue"]["resumed"] = False
    elif change in {"training_incomplete", "diagnostic", "no_updates"}:
        path = root / "training/training_report.json"
        value = json.loads(path.read_text())
        if change == "training_incomplete": value["training_complete"] = False
        if change == "diagnostic": value["diagnostic"] = True
        if change == "no_updates": value["progress"]["optimizer_updates"] = 0
    elif change == "different_checkpoint":
        (root / "best.pt").write_bytes(b"changed weights")
        path = None
    elif change == "missing_benchmark":
        (root / "benchmarks/benchmark_192x320_views8_3gib.json").unlink()
        path = None
    elif change == "bogus_fps":
        path = root / "benchmarks/benchmark_192x320_views8_3gib.json"
        value = json.loads(path.read_text())
        value["models"]["yolo26x_sem"]["batched_host_to_road_map"]["images_per_second"] = 100000
    else:
        path = root / "evaluation/candidate_384x640.json"
        value = json.loads(path.read_text())
        if change == "partial_eval": value["native_sources"]["a"]["partial_validation"] = True
        if change == "bad_public_count": value["public_obstacle_anomaly_subsets"]["obstacle"]["aggregate"]["images"] = 29
        if change == "wrong_threshold_denominator":
            value["native_sources"]["a"]["runtime_threshold_metrics"] = {**metric([[20, 10], [10, 60]]), "threshold": .65, "comparison": ">="}
    if path is not None:
        save(path, value)
    output = tmp_path / "new_summary"
    with pytest.raises((ValueError, KeyError)):
        summary.write_summary(root, output)
    assert not output.exists()


def test_oom_attempt_partial_modes_are_explicit_not_promoted_to_full_success(tmp_path):
    root = experiment_fixture(tmp_path)
    path = root / "benchmarks/benchmark_192x320_views8_3gib.json"
    success = json.loads(path.read_text())
    success["hardware"]["requested_allocator_budget_gib"] = 7
    save(root / "benchmarks/benchmark_192x320_views8_7gib.json", success)
    failed = deepcopy(success)
    failed["status"] = "partial_failure"
    failed["hardware"]["requested_allocator_budget_gib"] = 3
    failed["models"]["yolo26x_sem"].update(status="failed", out_of_memory=True, error="synthetic OOM fixture")
    for mode in ("batched_model_only", "batched_host_to_road_map"):
        del failed["models"]["yolo26x_sem"][mode]
    save(path, failed)
    result = summary.collect(root)
    partial = [row for row in result["benchmarks"] if row["report_status"] != "completed"]
    assert partial and all(row["mode_completed"] for row in partial)
    assert not any(row["model"] == "yolo26x_sem" and row["mode"].startswith("batched_") for row in partial)
    text = summary.korean_markdown(result)
    assert "별표(*)" in text and "7GiB" in text


def test_korean_summary_contains_denominators_and_limits_and_never_overwrites(tmp_path):
    root = experiment_fixture(tmp_path)
    output = tmp_path / "summary"
    result = summary.write_summary(root, output)
    text = (output / "summary.ko.md").read_text(encoding="utf-8")
    assert "450/3,000" in text
    assert "이미지별 평균 오인율 15.00%" in text and "GT=1 픽셀 전부 오인 0/30장" in text
    assert "실제 MSI 노트북 속도" in text and "1 epoch" in text
    assert "별도 조건" in text and "과거 TinyUNet" in text
    assert "모든 배포·제어 승인" in text
    assert not result["deployment_approved"] and not result["real_performance_validated"]
    before = (output / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="new summary"):
        summary.write_summary(root, output)
    assert (output / "summary.json").read_bytes() == before
