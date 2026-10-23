"""Summarize a finished experiment without loading a model or touching a GPU.

Usage: python scripts/summarize_yolo_semantic_experiment.py --experiment RUN
       --output NEW_DIRECTORY

Writes summary.json and summary.ko.md only after all required evidence passes
consistency checks. This verifies submitted artifacts, not their authenticity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from run_large_data_queue import compare_reports


NAMES = ["road_surface", "not_road_surface"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, name + " must be finite and nonnegative")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer")
    return value


def ratio(a, b):
    return a / b if b else None


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


class Evidence:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve(strict=True)
        payload = path.read_bytes()
        identity = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        self.files[str(path)] = identity
        result = json.loads(payload.decode("utf-8-sig"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant: " + value)))
        require(isinstance(result, dict), "Report must be a JSON object")
        return result, identity

    def verify(self, expected, base):
        require(isinstance(expected, dict) and isinstance(expected.get("path"), str), "Missing file identity")
        path = Path(expected["path"])
        path = path if path.is_absolute() else Path(base) / path
        actual = file_identity(path)
        require(actual["sha256"] == expected.get("sha256"), "File SHA256 changed: " + str(path))
        if "bytes" in expected:
            require(type(expected["bytes"]) is int and actual["bytes"] == expected["bytes"], "File length changed")
        self.files[actual["path"]] = actual
        return actual

    def unchanged(self):
        for value in self.files.values():
            require(file_identity(value["path"]) == value, "Evidence changed while summarizing: " + value["path"])


def confusion_metrics(matrix):
    require(isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix), "Expected a 2x2 confusion matrix")
    tp, fn, fp, tn = [integer(value, "confusion count") for row in matrix for value in row]
    require(tp + fn + fp + tn > 0, "Empty confusion matrix")
    return {"confusion_matrix": matrix, "valid_pixels": tp + fn + fp + tn,
            "road_tp_pixels": tp, "road_fn_pixels": fn, "road_fp_pixels": fp, "road_tn_pixels": tn,
            "road_iou": ratio(tp, tp + fn + fp), "road_precision": ratio(tp, tp + fp),
            "road_recall": ratio(tp, tp + fn), "false_road_rate": ratio(fp, fp + tn)}


def native_summary(report, *, threshold=False):
    source_metadata = report.get("validation_sources")
    require(isinstance(source_metadata, list) and source_metadata, "No validation sources")
    meta = {source["name"]: source for source in source_metadata}
    require(len(meta) == len(source_metadata) and set(meta) == set(report["native_sources"]), "Validation source identity/count mismatch")
    matrix = [[0, 0], [0, 0]]
    sources, count = {}, 0
    for name, native in report["native_sources"].items():
        require(native.get("partial_validation") is False, "Partial native evaluation cannot be summarized as complete")
        samples = integer(meta[name].get("images"), "validation images", 1)
        if "images" in native:
            require(native["images"] == samples, "Native image count differs from declared source")
        item = native.get("runtime_threshold_metrics") if threshold else native
        require(isinstance(item, dict), "Missing candidate operating-threshold coverage")
        if threshold:
            require(item.get("threshold") == report["runtime_threshold"] and item.get("comparison") == ">=", "Operating threshold differs")
        metrics = confusion_metrics(item["confusion_matrix"])
        require(type(item.get("valid_pixels")) is int and item["valid_pixels"] == metrics["valid_pixels"], "Confusion matrix pixel denominator differs")
        reported_iou = item.get("class_iou", [None])[0]
        require(reported_iou is None if metrics["road_iou"] is None else type(reported_iou) in (int, float) and math.isclose(reported_iou, metrics["road_iou"], rel_tol=1e-10, abs_tol=1e-12), "Reported road IoU conflicts with confusion counts")
        if threshold:
            require([sum(row) for row in item["confusion_matrix"]] == [sum(row) for row in native["confusion_matrix"]], "Threshold GT class denominator differs from argmax")
            for key in ("road_iou", "road_precision", "road_recall", "false_road_rate"):
                expected, actual = metrics[key], item.get(key)
                require(actual is None if expected is None else type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), "Operating-threshold metric conflicts with pixel counts")
        for y in range(2):
            for x in range(2):
                matrix[y][x] += metrics["confusion_matrix"][y][x]
        sources[name] = {"images": samples, **metrics}
        count += samples
    pooled = confusion_metrics(matrix)
    return {"images": count, "aggregation": "sum confusion matrices, then calculate pixel-weighted ratios; not mean per-source IoU",
            "pooled": pooled, "per_source": sources,
            "decision": "softmax road>=runtime threshold" if threshold else "argmax, equal logits choose road class 0"}


def public_summary(report):
    result = {}
    require(set(report["public_obstacle_anomaly_subsets"]) == {"obstacle", "anomaly"}, "Both public regression subsets are required")
    for name, count in (("obstacle", 30), ("anomaly", 10)):
        track = report["public_obstacle_anomaly_subsets"][name]
        require(track["aggregate"]["images"] == count and len(track["per_image"]) == count, "Public GT subset is incomplete")
        errors = total = 0
        nonempty_frames = []
        frame_ids = set()
        for row in track["per_image"]:
            frame_id = row["frame_id"]
            require(isinstance(frame_id, str) and frame_id not in frame_ids, "Duplicate or invalid public frame identity")
            frame_ids.add(frame_id)
            decision = row["decisions"]["runtime_road"]
            require(decision["threshold"] == report["runtime_threshold"] and decision["comparison"] == ">=", "Public threshold changed")
            denominator = integer(row["labelled_obstacle_pixels"], "GT obstacle count")
            numerator = integer(decision["obstacle_pixels_predicted_road"], "Obstacle-as-road count")
            require(decision["labelled_obstacle_pixels"] == denominator and numerator <= denominator, "Invalid public denominator")
            rate = ratio(numerator, denominator)
            reported_rate = decision["obstacle_as_road_rate"]
            require(reported_rate is None if rate is None else type(reported_rate) in (int, float)
                    and math.isclose(reported_rate, rate, rel_tol=1e-10, abs_tol=1e-12), "Public per-image rate differs from pixel counts")
            if denominator:
                nonempty_frames.append({"frame_id": frame_id, "obstacle_pixels_predicted_road": numerator,
                                        "labelled_obstacle_pixels": denominator, "obstacle_as_road_rate": rate})
            errors += numerator
            total += denominator
        aggregate = track["aggregate"]["decisions"]["runtime_road"]
        require(total > 0 and aggregate["labelled_obstacle_pixels"] == total and aggregate["obstacle_pixels_predicted_road"] == errors, "Public aggregate counts differ")
        require(math.isclose(finite(aggregate["micro_obstacle_as_road_rate"], "public error rate"), errors / total, rel_tol=1e-10, abs_tol=1e-12), "Public aggregate rate differs")
        mean_rate = math.fsum(row["obstacle_as_road_rate"] for row in nonempty_frames) / len(nonempty_frames)
        require(integer(aggregate["images_with_labelled_obstacles"], "nonempty public images") == len(nonempty_frames), "Public nonempty-image denominator differs")
        require(math.isclose(finite(aggregate["mean_per_image_obstacle_as_road_rate"], "public image-balanced rate"), mean_rate, rel_tol=1e-10, abs_tol=1e-12), "Public image-balanced mean differs from per-image counts")
        all_wrong = sum(row["obstacle_pixels_predicted_road"] == row["labelled_obstacle_pixels"] for row in nonempty_frames)
        half_wrong = sum(2 * row["obstacle_pixels_predicted_road"] >= row["labelled_obstacle_pixels"] for row in nonempty_frames)
        worst = sorted(nonempty_frames, key=lambda row: (-row["obstacle_as_road_rate"], row["labelled_obstacle_pixels"], row["frame_id"]))[:5]
        result[name] = {"dataset": track["dataset"], "images": count, "labelled_obstacle_pixels": total,
                        "obstacle_pixels_predicted_road": errors, "obstacle_as_road_rate": errors / total,
                        "micro_obstacle_as_road_rate": errors / total,
                        "mean_per_image_obstacle_as_road_rate": mean_rate,
                        "images_with_labelled_obstacles": len(nonempty_frames),
                        "empty_gt_images_excluded_from_image_statistics": count - len(nonempty_frames),
                        "all_obstacle_pixels_predicted_road_images": all_wrong,
                        "at_least_half_obstacle_pixels_predicted_road_images": half_wrong,
                        "worst_frames": worst,
                        "worst_frames_order": "descending per-image rate, then ascending GT pixel count and frame ID; at most 5 nonempty images",
                        "image_statistics_note": "Equal weight per image containing GT=1 pixels; zero-GT images excluded. Image counts are not independent drives or collision counts.",
                        "threshold": report["runtime_threshold"], "comparison": ">=",
                        "denominator_note": "GT=1 original-resolution pixels only; usual GT=0 is not general road GT"}
    return result


def paired_metric_regressions(baseline, candidate):
    """Observed same-input changes, independent of the legacy FPR-only flags.

    These are descriptive comparisons, not significance tests or safety gates.
    Undefined ratios are explicitly outside this list, never inferred improved.
    """
    result = []

    def inspect(scope, old, new, fields):
        for metric, direction, unit in fields:
            before, after = old[metric], new[metric]
            if before is None or after is None:
                continue
            regressed = after < before if direction == "higher" else after > before
            if regressed:
                result.append({"scope": scope, "metric": metric, "direction": direction,
                               "unit": unit, "baseline": before, "candidate": after,
                               "candidate_minus_baseline": after - before,
                               "comparison": "paired_192x320", "statistical_significance_tested": False})

    native_fields = [(key, "lower" if key == "false_road_rate" else "higher", "fraction")
                     for key in ("road_iou", "road_precision", "road_recall", "false_road_rate")]
    inspect("native:pooled", baseline["native"]["pooled"], candidate["native"]["pooled"], native_fields)
    require(set(baseline["native"]["per_source"]) == set(candidate["native"]["per_source"]), "Paired native source sets differ")
    for name, old in baseline["native"]["per_source"].items():
        inspect("native:" + name, old, candidate["native"]["per_source"][name], native_fields)
    public_fields = [("micro_obstacle_as_road_rate", "lower", "fraction"),
                     ("mean_per_image_obstacle_as_road_rate", "lower", "fraction"),
                     ("all_obstacle_pixels_predicted_road_images", "lower", "images"),
                     ("at_least_half_obstacle_pixels_predicted_road_images", "lower", "images")]
    for track, old in baseline["public_obstacle_pixels"].items():
        inspect("public:" + track, old, candidate["public_obstacle_pixels"][track], public_fields)
    return result


def evaluation_checks(report, checkpoint, config, *, expected_input, candidate):
    require(report.get("evaluation_only") is True and report.get("training_performed") is False
            and report.get("deployment_validated") is False and report.get("checkpoint_diagnostic") is False,
            "Evaluation is incomplete, diagnostic or makes unsupported deployment claims")
    require(report.get("checkpoint") == checkpoint and report.get("config") == config, "Evaluation weights/config do not match experiment")
    require(report.get("input_size") == list(expected_input) and report.get("label_space_id") == "road_surface_v1"
            and report.get("road_class_id") == 0, "Evaluation input or label protocol differs")
    if candidate:
        require(report.get("model_family") == "yolo26x-sem" and report.get("task") == "semantic"
                and report.get("class_names") == NAMES and report.get("native_gt_size") == [192, 320], "Wrong semantic model/GT contract")
        require(report.get("models_validated_for_control") is False and report.get("real_performance_validated") is False,
                "Pixel evaluation must not claim control/real-world validation")


def same_cohort(left, right):
    require(left["validation_sources"] == right["validation_sources"] and left["runtime_threshold"] == right["runtime_threshold"], "Higher-resolution evaluation cohort/threshold differs")
    require(set(left["native_sources"]) == set(right["native_sources"]), "Higher-resolution sources differ")
    for name in left["native_sources"]:
        a, b = left["native_sources"][name]["confusion_matrix"], right["native_sources"][name]["confusion_matrix"]
        require([sum(row) for row in a] == [sum(row) for row in b], "Higher-resolution native GT denominator differs")
    for track in ("obstacle", "anomaly"):
        def signatures(report):
            return [(row["frame_id"], row["image"], row["ground_truth"], row["labelled_obstacle_pixels"])
                    for row in report["public_obstacle_anomaly_subsets"][track]["per_image"]]
        require(signatures(left) == signatures(right), "Higher-resolution public images/GT changed")


def benchmark_rows(report, path, training_id, best, native_checkpoint, config, *, allow_partial=False):
    require(report.get("status") in (("completed", "partial_failure", "failed") if allow_partial else ("completed",))
            and set(report.get("models", {})) == {"yolo26x_sem", "tinyunet"}, "Benchmark did not complete both models")
    require(all(report.get(key) is False for key in ("physical_output", "accuracy_evaluated", "driving_validated", "evidence_for_80kmh", "target_hardware_emulated", "training_performed")), "Benchmark has unsupported measurement claims")
    proof = report["provenance"]
    require(proof["yolo"]["checkpoint"] == best and proof["yolo"]["training_report"] == training_id
            and proof["native_checkpoint"] == native_checkpoint and proof["native_config"] == config, "Benchmark model provenance differs")
    views = report.get("views")
    require(views in (1, 8) and type(views) is int, "Unknown view count")
    require(report.get("precision") in ("fp16", "fp32") and report["hardware"].get("actual_gpu_name"), "Missing actual GPU/precision measurement")
    require(len(report.get("input_images", [])) == views, "Benchmark does not identify every input image")
    modes = ["model_only", "host_to_road_map"] + (["batched_model_only", "batched_host_to_road_map"] if views == 8 else [])
    result = []
    for model, entry in report["models"].items():
        complete = entry.get("status") == "completed" and entry.get("out_of_memory") is False
        require(complete or (allow_partial and entry.get("status") == "failed" and entry.get("out_of_memory") is True), "Model benchmark failed for a non-OOM reason")
        if not complete and not any(mode in entry for mode in modes):
            continue
        resources = entry.get("resources")
        require(isinstance(resources, dict), "Benchmark memory report missing")
        for mode in modes:
            if mode not in entry and allow_partial and not complete:
                continue
            item = entry[mode]
            n = integer(item.get("measured_cycles"), "measured benchmark cycles", 1)
            require(n == report["iterations"] and item["views_per_cycle"] == views, "Benchmark sample denominator differs")
            mean = finite(item["mean_cycle_ms"], "mean latency", 1e-12)
            for key in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "max_cycle_ms", "images_per_second", "camera_work_cycles_per_second"):
                finite(item[key], key, 1e-12)
            require(item["p50_cycle_ms"] <= item["p95_cycle_ms"] <= item["p99_cycle_ms"] <= item["max_cycle_ms"], "Latency percentiles are inconsistent")
            require(math.isclose(item["images_per_second"], views * 1000 / mean, rel_tol=1e-8)
                    and math.isclose(item["camera_work_cycles_per_second"], 1000 / mean, rel_tol=1e-8), "FPS denominator differs from complete-cycle timing")
            batched = mode.startswith("batched_")
            require(item.get("batch_size_per_call") == (views if batched else 1)
                    and item.get("model_calls_per_cycle") == (1 if batched else views), "Sequential/batched execution metadata differs")
            memory = {key: resources.get(key) for key in ("torch_peak_allocated_gib", "torch_peak_reserved_gib", "process_gpu_peak_gib_sampled", "process_rss_peak_gib_sampled")}
            mode_memory = item.get("cuda_memory")
            if isinstance(mode_memory, dict):
                for key in ("torch_peak_allocated_gib", "torch_peak_reserved_gib"):
                    memory[key] = mode_memory.get(key)
            for value in memory.values():
                if value is not None:
                    finite(value, "memory")
            require(memory["torch_peak_allocated_gib"] is not None and memory["torch_peak_reserved_gib"] is not None, "CUDA allocator measurements unavailable")
            result.append({"model": model, "input_size": report["input_size_hw"], "views": views,
                           "mode": mode, "precision": report["precision"], "cycles": n,
                           "actual_gpu": report["hardware"]["actual_gpu_name"],
                           "allocator_budget_gib": report["hardware"]["requested_allocator_budget_gib"],
                           "p50_cycle_ms": item["p50_cycle_ms"], "p95_cycle_ms": item["p95_cycle_ms"],
                           "images_per_second": item["images_per_second"], "cycles_per_second": item["camera_work_cycles_per_second"],
                           "memory": memory, "cuda_memory_scope": "mode-specific reset peak" if isinstance(mode_memory, dict) else "legacy per-model peak across all modes",
                           "process_memory_scope": "sampled peak across loading/all model modes, not specific to this mode",
                           "report_status": report["status"], "model_status": entry["status"], "mode_completed": True,
                           "report": str(path)})
    return result


def collect(experiment):
    root = Path(experiment).resolve(strict=True)
    evidence = Evidence()
    state, _ = evidence.read(root / "status.json")
    require(state.get("status") == "complete" and state.get("deployment_approved") is False, "Experiment has not completed")
    if "suspended_queue" in state:
        require(state["suspended_queue"].get("resumed") is True and state.get("queue_temporarily_suspended") is False, "Previous queue has not been resumed")
    training, training_id = evidence.read(root / "training/training_report.json")
    require(training.get("schema_version") == 1 and training.get("model_family") == "yolo26x-sem"
            and training.get("task") == "semantic" and training.get("class_names") == NAMES
            and training.get("road_class_id") == 0 and training.get("training_complete") is True
            and training.get("diagnostic") is False, "Training evidence is incomplete/diagnostic/wrong model")
    require(isinstance(training.get("history"), list) and len(training["history"]) == 1
            and training["actual_args"].get("epochs") == 1, "This summary contract requires one completed epoch")
    progress = training["progress"]
    require(integer(progress["optimizer_updates"], "optimizer updates", 1) > 0
            and integer(progress["train_batches"], "train batches", 1) == training["batches_per_epoch"], "No complete epoch with actual optimizer updates")
    best = evidence.verify(training["checkpoints"]["best"], root / "training")
    evidence.verify(training["checkpoints"]["last"], root / "training")
    config = evidence.verify(training["source_config"], root / "training")
    evidence.verify({"path": training["dataset"]["provenance_path"], "sha256": training["dataset"]["provenance_sha256"]}, root / "training")
    comparison, _ = evidence.read(root / "evaluation/comparison.json")
    baseline, _ = evidence.read(comparison["baseline_report"])
    paired_path = root / "evaluation/paired_192x320.json"
    require(Path(comparison["candidate_report"]).resolve() == paired_path, "Comparison names a different candidate report")
    paired, _ = evidence.read(paired_path)
    higher, _ = evidence.read(root / "evaluation/candidate_384x640.json")
    native_checkpoint = evidence.verify(baseline["checkpoint"], root)
    evaluation_checks(baseline, native_checkpoint, config, expected_input=(192, 320), candidate=False)
    evaluation_checks(paired, best, config, expected_input=(192, 320), candidate=True)
    evaluation_checks(higher, best, config, expected_input=(384, 640), candidate=True)
    for report in (paired, higher):
        require(report["training_completion"]["report"] == training_id, "Evaluation training report changed")
    recomputed = compare_reports(baseline, paired)
    for key in ("compared", "regressions", "baseline_checkpoint", "candidate_checkpoint", "deployment_approved"):
        require(comparison.get(key) == recomputed[key], "Saved comparison conflicts with actual reports")
    same_cohort(paired, higher)
    accuracy = {}
    for key, report in (("tinyunet_192x320", baseline), ("yolo26x_sem_192x320", paired), ("yolo26x_sem_384x640", higher)):
        accuracy[key] = {"input_size": report["input_size"], "native": native_summary(report), "public_obstacle_pixels": public_summary(report),
                         "precision": {"amp_requested": report.get("amp_requested"), "amp_effective": report.get("amp_effective"), "metric_math": "FP32"}}
        if key.startswith("yolo"):
            accuracy[key]["runtime_threshold_native"] = native_summary(report, threshold=True)
    require(all(value["native"]["images"] == training["dataset"]["counts"]["val"] for value in accuracy.values()), "Training/evaluation validation counts differ")
    rows, attempts, conditions = [], [], set()
    for path in sorted((root / "benchmarks").glob("benchmark_*.json")):
        report, _ = evidence.read(path)
        attempts.append({"path": str(path), "status": report.get("status"), "input_size": report.get("input_size_hw"), "views": report.get("views"),
                         "allocator_budget_gib": report.get("hardware", {}).get("requested_allocator_budget_gib"),
                         "models": {name: {"status": item.get("status"), "out_of_memory": item.get("out_of_memory"), "error": item.get("error")} for name, item in report.get("models", {}).items()}})
        if report.get("status") != "completed":
            require(report.get("status") in ("partial_failure", "failed")
                    and any(item.get("out_of_memory") is True for item in report.get("models", {}).values()), "Unexpected non-OOM benchmark failure in completed experiment")
            rows.extend(benchmark_rows(report, path, training_id, best, native_checkpoint, config, allow_partial=True))
            continue
        rows.extend(benchmark_rows(report, path, training_id, best, native_checkpoint, config))
        conditions.add((tuple(report["input_size_hw"]), report["views"]))
    require(conditions == {((192, 320), 1), ((192, 320), 8), ((384, 640), 1), ((384, 640), 8)}, "Missing successful benchmark input/view conditions")
    evidence.unchanged()
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
            "experiment": str(root), "model_family": "yolo26x-sem", "epochs": 1,
            "training": {"images": training["dataset"]["counts"]["train"], "validation_images": training["dataset"]["counts"]["val"],
                         "optimizer_updates": progress["optimizer_updates"], "amp_skipped_updates": progress.get("amp_skipped_updates"), "best_checkpoint": best},
            "accuracy": accuracy, "paired_comparison": recomputed,
            "all_metric_regressions": paired_metric_regressions(accuracy["tinyunet_192x320"], accuracy["yolo26x_sem_192x320"]),
            "all_metric_regressions_scope": "Observed paired 192x320 native pooled/per-source IoU, precision, recall, FPR; public pixel FPR, image mean and all/half-wrong image counts. Undefined ratios skipped; no significance or deployment test. Legacy paired_comparison flags are preserved separately.",
            "benchmarks": rows, "benchmark_attempts": attempts,
            "deployment_approved": False, "models_validated_for_control": False, "zero_accident_guarantee": False,
            "msi_prestige_speed_measured": False, "real_performance_validated": False,
            "limits": ["One-epoch road semantic segmentation experiment, not a completed autonomous-driving system or zero-accident proof.",
                       "Native pooled metrics mix source datasets, including synthetic VKITTI; large pixel cohorts dominate the pooled ratios.",
                       "384x640 is a separate input condition with the same 192x320 native GT; not a matched-resolution architecture comparison.",
                       "Runtime-threshold recall/IoU describes candidate road coverage; historical TinyUNet has no paired threshold metric.",
                       "RoadObstacle21/RoadAnomaly21 are reused public development GT subsets, not blind tests or official anomaly AP.",
                       "Historical TinyUNet native RGB/GT bytes were not hashed, so full historical content identity is not proven.",
                       "Accuracy uses recorded AMP settings and FP32 metrics; benchmark fp16 converts model weights/input and is a separate numerical condition.",
                       "1/8-view sequential/batched timings use recorded frames; no synchronized camera capture, fusion, planning or actuation is measured.",
                       "GPU allocator caps do not emulate MSI/GTX1650 hardware; MSI latency has not been measured.",
                       "CUDA peaks use separately reset per-mode measurements when provided; legacy fallback and sampled process memory span all model modes. Null means unavailable, not zero.",
                       "Artifact consistency checks do not authenticate real-world performance or evidence provenance."],
            "input_reports_and_artifacts": sorted(evidence.files.values(), key=lambda item: item["path"])}


def percent(value):
    return "미정의" if value is None else f"{100 * value:.2f}%"


def decimal(value):
    return "미측정" if value is None else f"{value:.2f}"


def korean_markdown(summary):
    data = summary["accuracy"]
    keys = ["tinyunet_192x320", "yolo26x_sem_192x320", "yolo26x_sem_384x640"]
    labels = ["TinyUNet 192×320", "YOLO26x-sem 192×320", "YOLO26x-sem 384×640 (별도 조건)"]
    lines = ["# YOLO26x-sem 학습·평가 결과", "",
             f"YOLO26x-sem을 **1 epoch**, 학습 {summary['training']['images']:,}장·검증 {summary['training']['validation_images']:,}장으로 실행했습니다. 실제 optimizer 업데이트는 {summary['training']['optimizer_updates']:,}회입니다.", "",
             "아래 수치는 도로 영역 분할 실험 결과입니다. 자율주행 완성도·사고 확률이 아니며, 실제 MSI 노트북 속도와 사고 방지 성능은 검증하지 않았습니다.", "",
             "| 모델·입력 | 도로 IoU | 도로 재현율 | 비도로→도로 오인율 | 평가 픽셀 |", "|---|---:|---:|---:|---:|"]
    for key, label in zip(keys, labels):
        metric = data[key]["native"]["pooled"]
        lines.append(f"| {label} | {percent(metric['road_iou'])} | {percent(metric['road_recall'])} | {percent(metric['false_road_rate'])} | {metric['valid_pixels']:,} |")
    lines += ["", "표는 혼동행렬을 합산한 픽셀 가중치 결과입니다. 합성 VKITTI와 실제 영상이 섞여 있으며, 큰 데이터셋의 픽셀이 전체 지표를 지배합니다. 384×640 입력도 native GT는 기존 192×320으로 평가했습니다.", "",
              "| 데이터셋 | 영상 | TinyUNet IoU / 재현율 | YOLO 공통 입력 IoU / 재현율 | YOLO 큰 입력 IoU / 재현율 (별도 조건) |",
              "|---|---:|---:|---:|---:|"]
    for name, baseline in data[keys[0]]["native"]["per_source"].items():
        parts = []
        for key in keys:
            metric = data[key]["native"]["per_source"][name]
            parts.append(f"{percent(metric['road_iou'])} / {percent(metric['road_recall'])}")
        lines.append(f"| {name} | {baseline['images']:,} | " + " | ".join(parts) + " |")
    native_regressions = [row for row in summary["all_metric_regressions"] if row["scope"].startswith("native:")]
    if native_regressions:
        lines += ["", "공통 192×320 입력에서 관측된 도로 지표 하락:", ""]
        names = {"road_iou": "도로 IoU", "road_recall": "도로 재현율", "road_precision": "도로 정밀도", "false_road_rate": "비도로→도로 오인율"}
        for row in native_regressions:
            lines.append(f"- {row['scope'][7:]} · {names[row['metric']]}: {percent(row['baseline'])} → {percent(row['candidate'])} ({100 * row['candidate_minus_baseline']:+.2f}%p).")
        lines += ["", "전체 평균의 개선과 별개로 위 데이터셋의 회귀가 있습니다. 이는 관측된 변화이며 독립 주행 단위의 통계적 유의성을 검증한 결과는 아닙니다."]
    lines += ["", "| 공개 GT | 영상 | TinyUNet | YOLO 공통 입력 | YOLO 큰 입력 |", "|---|---:|---:|---:|---:|"]
    for track in ("obstacle", "anomaly"):
        parts = []
        for key in keys:
            metric = data[key]["public_obstacle_pixels"][track]
            parts.append(f"{metric['obstacle_pixels_predicted_road']:,}/{metric['labelled_obstacle_pixels']:,} ({percent(metric['obstacle_as_road_rate'])})")
        metric = data[keys[0]]["public_obstacle_pixels"][track]
        lines.append(f"| {metric['dataset']} | {metric['images']} | " + " | ".join(parts) + " |")
    threshold = data[keys[1]]["public_obstacle_pixels"]["obstacle"]["threshold"]
    lines += ["", f"분자는 p(도로)≥{threshold:g}로 잘못 분류한 GT=1 픽셀, 분모는 원본 해상도의 모든 GT=1 픽셀입니다. 재사용하는 공개 평가 자료이며 공식 이상 탐지 AP·충돌률이 아닙니다.", ""]
    for key, label in zip(keys, labels):
        for track in ("obstacle", "anomaly"):
            metric = data[key]["public_obstacle_pixels"][track]
            n = metric["images_with_labelled_obstacles"]
            worst = metric["worst_frames"][0]
            lines.append(f"- {label} · {metric['dataset']}: 이미지별 평균 오인율 {percent(metric['mean_per_image_obstacle_as_road_rate'])}; GT=1 픽셀 전부 오인 {metric['all_obstacle_pixels_predicted_road_images']}/{n}장, 절반 이상 오인 {metric['at_least_half_obstacle_pixels_predicted_road_images']}/{n}장. 최악 프레임 `{worst['frame_id']}`: {worst['obstacle_pixels_predicted_road']:,}/{worst['labelled_obstacle_pixels']:,} ({percent(worst['obstacle_as_road_rate'])}).")
    lines += ["", "이미지별 평균은 GT=1이 있는 각 영상에 같은 가중치를 줍니다. 위 표의 픽셀 가중 평균과 별개이며, 작은 장애물의 실패가 큰 장애물 픽셀에 가려지는지 확인하는 지표입니다. GT=1이 없는 영상은 이 평균·전부 오인·절반 이상 오인 횟수에서 제외합니다. 영상 횟수는 독립 주행 횟수나 충돌 횟수가 아닙니다.", ""]
    for key, label in zip(keys[1:], labels[1:]):
        metric = data[key]["runtime_threshold_native"]["pooled"]
        lines.append(f"- {label} 운영 임계값에서 도로 재현율 {percent(metric['road_recall'])}, IoU {percent(metric['road_iou'])}, 비도로→도로 오인율 {percent(metric['false_road_rate'])}. 도로를 모두 포기한 겉보기 개선인지 함께 확인해야 합니다.")
    lines += ["", "과거 TinyUNet 보고서에는 같은 운영 임계값 지표가 없어 이 항목의 상대 개선은 주장하지 않습니다.", "",
              "| 입력 | 뷰·실행 | 모델 | cap GiB | p50/p95 ms | 이미지/s | 전체 주기/s | GPU allocated/reserved GiB | RAM GiB |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["benchmarks"]:
        if not row["mode"].endswith("host_to_road_map"):
            continue
        mode = ("일괄" if row["mode"].startswith("batched_") else "순차") + ("*" if row["report_status"] != "completed" else "")
        memory = row["memory"]
        lines.append(f"| {'×'.join(map(str, row['input_size']))} | {row['views']}뷰 {mode} | {row['model']} | {row['allocator_budget_gib']} | {row['p50_cycle_ms']:.2f}/{row['p95_cycle_ms']:.2f} | {row['images_per_second']:.2f} | {row['cycles_per_second']:.2f} | {decimal(memory['torch_peak_allocated_gib'])}/{decimal(memory['torch_peak_reserved_gib'])} | {decimal(memory['process_rss_peak_gib_sampled'])} |")
    hardware = ", ".join(sorted({row["actual_gpu"] for row in summary["benchmarks"]}))
    precisions = ", ".join(sorted({row["precision"] for row in summary["benchmarks"]}))
    lines += ["", f"측정 GPU: {hardware}. 실행 정밀도: {precisions}. 표는 호스트 RGB 전처리→GPU→CPU 도로 확률맵까지이며 모델만의 시간은 JSON에 별도로 기록했습니다.", "",
              "8뷰는 녹화 프레임의 순차 처리 또는 배치 처리입니다. 실제 8카메라 동기화·융합·계획·제어 주파수가 아닙니다. GPU 수치는 모드별 peak가 있으면 이를 사용하며, RAM/NVML 프로세스 메모리는 모든 모드를 포함한 표본 최고치입니다. 상세 범위는 JSON에 기록했습니다.", "",
              "학습 GPU의 3/7GiB allocator 제한은 MSI/GTX1650의 속도나 총 VRAM을 재현하지 않습니다. 정확도 평가의 AMP와 실행 벤치마크의 fp16 모델 변환도 서로 다른 수치 조건입니다."]
    failed = [item for item in summary["benchmark_attempts"] if item["status"] != "completed"]
    if failed:
        lines += ["", f"메모리 부족으로 완료하지 못한 벤치마크 시도 {len(failed)}건도 JSON에 보존했습니다. 별표(*)는 그 시도 안에서 완료된 개별 모드입니다. 해당 메모리 조건의 모든 모드가 성공했다는 뜻은 아닙니다. 큰 메모리 조건의 성공을 작은 조건의 성공으로 해석하지 않습니다."]
    lines += ["", "모든 배포·제어 승인 및 사고 없음 보장 플래그는 false입니다. 기존 TinyUNet native 이미지/GT 바이트는 과거 평가에서 해시하지 않아 완전한 과거 내용 동일성은 증명하지 못합니다.", ""]
    return "\n".join(lines)


def write_summary(experiment, output):
    output = Path(output).resolve()
    require(not output.exists(), "Choose a new summary directory")
    summary = collect(experiment)
    markdown = korean_markdown(summary)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.ko.md").write_text(markdown, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for JSON and Korean Markdown")
    args = parser.parse_args()
    result = write_summary(args.experiment, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "deployment_approved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
