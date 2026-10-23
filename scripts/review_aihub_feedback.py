"""CPU-only post-completion review; never trains, tunes thresholds or deploys."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


SIZES = ((192, 320), (384, 640))
LIMITS = [
    "공개 장애물·이상 물체 40장은 재사용 개발 회귀 평가입니다. 학습·임계값 튜닝에 사용하지 않습니다.",
    "현재·기존 모델은 공식 validation을 epoch별 best 체크포인트 선택에 사용했습니다. 다음 실험에서 train 유래 개발셋을 사용해도 현재 고정 평가셋이 독립 비공개 테스트로 바뀌지는 않습니다. 일반화 성능이나 통계적 유의성을 입증하지 않습니다.",
    "AI Hub 13·14는 주간, 15는 야간 데이터셋 구분입니다. 원본 주야간 메타데이터가 없는 자료의 조건은 미상이며 추정하지 않습니다.",
    "작은 장애물의 실제 물리 크기·거리, 시간 연속 누락과 동기화된 다중 카메라 융합은 검증하지 못했습니다.",
    "80km/h 인지·충돌 방지 검증은 미완료입니다. 속도·시간·카메라 보정값을 갖춘 별도 검증이 필요합니다.",
    "RTX 3080의 3/7GiB 할당기 제한은 실제 MSI 4/8GB 장치 속도·전체 메모리를 재현하지 않습니다.",
    "처리 지연은 녹화 프레임에서 도로 확률맵까지이며 노출·USB·추적·계획·제동 지연을 제외합니다. 외부 GPU 작업 간섭도 가능합니다.",
    "오프라인 지표와 합성 제어 테스트는 무사고 보장이나 실차 운용 승인 근거가 아닙니다.",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(entry):
    if identity(entry["path"]) != entry:
        raise ValueError(f"Evidence identity changed: {entry['path']}")
    return Path(entry["path"])


def training_evidence(report):
    args, history, progress = report["actual_args"], report["history"], report["progress"]
    if (report.get("training_complete") is not True or report.get("diagnostic") is not False
            or args.get("epochs") != 3 or args.get("resume") is not False or args.get("fraction") != 1.0
            or [row.get("epoch") for row in history] != [1, 2, 3]
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in history for v in row.values())):
        raise ValueError("Three genuinely completed, finite training epochs required")
    batch, count = args["batch"], report["dataset"]["counts"]["train"]
    if type(batch) is not int or batch < 1 or type(count) is not int or count < 1:
        raise ValueError("Invalid training population/batch")
    batches = math.ceil(count / batch)
    fields = ("train_batches", "optimizer_attempts", "optimizer_updates", "amp_skipped_updates")
    if (report.get("batches_per_epoch") != batches
            or any(type(progress.get(k)) is not int or progress[k] < 0 for k in fields)
            or progress["train_batches"] != batches * 3 or progress["optimizer_updates"] < 1
            or progress["optimizer_attempts"] > progress["train_batches"]
            or progress["optimizer_attempts"] != progress["optimizer_updates"] + progress["amp_skipped_updates"]):
        raise ValueError("Incomplete actual batch/update evidence")
    return {"epochs": 3, "batches_per_epoch": batches, **progress}


def change(old, new, direction, *, rate=False):
    valid = all(type(v) in (int, float) and math.isfinite(v) for v in (old, new))
    delta = new - old if valid else None
    return {"baseline": old, "candidate": new, "candidate_minus_baseline": delta,
            "percentage_point_change": delta * 100 if rate and valid else None,
            "relative_change_percent": delta / abs(old) * 100 if valid and old != 0 else None,
            "relative_improvement_percent": (1 if direction == "higher" else -1) * delta / abs(old) * 100 if valid and old != 0 else None,
            "relative_change_unavailable_reason": "missing_or_nonfinite" if not valid else "zero_baseline" if old == 0 else None,
            "preferred_direction": direction,
            "regressed": None if delta is None else delta > 0 if direction == "lower" else delta < 0}


def enrich_comparison(pair):
    for row in pair["deltas"]:
        metric = row["metric"]
        rate = metric in {"road_iou", "road_precision", "road_recall"} or metric.endswith("_rate")
        row.update(change(row["baseline"], row["candidate"], row["preferred_direction"], rate=rate))
    pair["regressions"] = [row for row in pair["deltas"] if row["regressed"]]
    return pair


def runtime_contract(report, mode, *, require_uuid=True):
    hardware = report["hardware"]
    snapshot = hardware.get("nvidia_smi_before") or {}
    gpu = snapshot.get("matched_device") or {}
    if require_uuid and not gpu.get("uuid"):
        raise ValueError("GPU UUID evidence unavailable")
    return {"gpu_uuid": gpu.get("uuid"), "gpu_name": hardware["actual_gpu_name"],
            "gpu_memory_gib": hardware["actual_gpu_memory_gib"], "device": hardware["device"],
            "cuda_version": hardware["cuda_version"], "cpu_threads": hardware["cpu_intraop_threads"],
            "torch_version": report["torch_version"], "pillow_version": report["pillow_version"],
            "precision": report["precision"], "input_size": report["input_size_hw"],
            "input_images": report["input_images"], "views": report["views"], "mode": mode,
            "allocator_cap_gib": hardware["requested_allocator_budget_gib"],
            "iterations": report["iterations"], "warmup": report["warmup_cycles_per_mode"],
            "probability": report["probability"], "excluded": report["excluded"],
            "label_space_id": report["label_space_id"], "class_names": report["class_names"],
            "road_class_id": report["road_class_id"],
            "input_shape_per_call": report["input_shape_per_call"],
            "batched_input_shape_per_call": report["batched_input_shape_per_call"],
            "shared_instances_per_model": report["shared_instances_per_model"],
            "outputs_retained_until_cycle_check": report["outputs_retained_until_cycle_check"]}


def runtime_pair(before, after, mode, baseline_model="yolo26x_sem"):
    result = {"baseline_model": baseline_model, "candidate_model": "yolo26x_sem", "mode": mode,
              "comparable": False, "input_size": after.get("input_size_hw"), "views": after.get("views")}
    result["raw_measurements"] = {"baseline": before.get("models", {}).get(baseline_model, {}).get(mode),
                                  "candidate": after.get("models", {}).get("yolo26x_sem", {}).get(mode)}
    try:
        same_invocation = before is after
        a = runtime_contract(before, mode, require_uuid=not same_invocation)
        b = runtime_contract(after, mode, require_uuid=not same_invocation)
        mismatch = [key for key in a if a[key] != b[key]]
        if mismatch:
            raise ValueError("Runtime contract differs: " + ", ".join(mismatch))
        old, new = before["models"][baseline_model], after["models"]["yolo26x_sem"]
        if baseline_model == "yolo26x_sem" and old.get("ultralytics_version") != new.get("ultralytics_version"):
            raise ValueError("Ultralytics runtime version differs")
        if before["status"] != "completed" or after["status"] != "completed" or any(x["status"] != "completed" for x in (old, new)):
            raise ValueError("Incomplete runtime product")
        left, right = old[mode], new[mode]
        for measurement in (left, right):
            if measurement.get("measured_cycles") != a["iterations"] or measurement.get("views_per_cycle") != a["views"]:
                raise ValueError("Timing sample count/views differ from protocol")
        for key in ("output_dtype", "output_shape", "batch_size_per_call", "model_calls_per_cycle"):
            if left.get(key) is None or left[key] != right.get(key):
                raise ValueError("Runtime mode differs: " + key)
        metrics = {key: change(left[key], right[key], "higher" if key == "camera_work_cycles_per_second" else "lower")
                   for key in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second")}
        for key in ("torch_peak_allocated_gib", "torch_peak_reserved_gib"):
            metrics[key] = change(left["cuda_memory"].get(key), right["cuda_memory"].get(key), "lower")
        metrics["sampled_process_rss_peak_gib"] = change(old.get("resources", {}).get("process_rss_peak_gib_sampled"),
                                                       new.get("resources", {}).get("process_rss_peak_gib_sampled"), "lower")
        if any(metrics[k]["candidate_minus_baseline"] is None for k in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second")):
            raise ValueError("Missing/nonfinite timing measurement")
        if any(metrics[k][which] <= 0 for k in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second") for which in ("baseline", "candidate")):
            raise ValueError("Timing/throughput must be positive")
        result.update(comparable=True, contract=a, metrics=metrics, same_benchmark_invocation=same_invocation,
                      throughput_unit="complete camera-work cycles/s; never aggregate images/s")
    except (ValueError, KeyError, TypeError) as error:
        result["skip_reason"] = str(error)
    return result


def cancelled_runtime_evidence(run, state):
    """Validate recorded user cancellation and immutable exited-run evidence, not live processes."""
    folder = run / "speed_cancellation_v1"
    confirmation_id = identity(folder / "confirmation.json")
    confirmation = read(confirmation_id["path"])
    if (confirmation.get("status") != "cancelled_by_user"
            or confirmation.get("runner_exited") is not True or confirmation.get("queue_restored") is not True
            or confirmation.get("training_complete") is not True
            or type(confirmation.get("runtime_steps_completed")) is not int or confirmation["runtime_steps_completed"] != 0
            or type(confirmation.get("accuracy_steps_completed")) is not int or confirmation["accuracy_steps_completed"] != 6
            or not isinstance(confirmation.get("run"), str) or not Path(confirmation["run"]).is_absolute()
            or Path(confirmation["run"]).resolve() != run):
        raise ValueError("Explicit runtime cancellation with exited runner, restored queue and completed accuracy required")
    request_path = checked(confirmation["request"])
    snapshot_path = checked(confirmation["gpu_runner_exit_snapshot"])
    if request_path.resolve() != (folder / "request.json").resolve() or snapshot_path.resolve().parent != folder.resolve():
        raise ValueError("Cancellation evidence must belong to this run's cancellation folder")
    request, snapshot = read(request_path), read(snapshot_path)
    process = request.get("runner") or {}
    if (not isinstance(process, dict) or type(process.get("pid")) is not int or process["pid"] < 1
            or type(process.get("created")) not in (int, float) or not math.isfinite(process["created"]) or process["created"] <= 0
            or not isinstance(process.get("command"), list) or not process["command"]
            or any(not isinstance(value, str) or not value for value in process["command"])):
        raise ValueError("Cancellation requires a concrete runner process identity")
    if (not isinstance(request.get("user_request"), str) or not request["user_request"].strip()
            or not isinstance(request.get("scope"), str) or not request["scope"].strip()
            or not isinstance(request.get("runner"), dict) or request["runner"] != snapshot.get("runner")
            or request.get("training_complete") is not True or request.get("accuracy_completed") != 6
            or request.get("benchmarks_completed") != 0 or request.get("children_before") != []):
        raise ValueError("Cancellation request does not identify the completed run and empty runtime scope")
    queue = snapshot.get("suspended_queue")
    if (snapshot != state or snapshot.get("status") != "recovery_required"
            or not str(snapshot.get("error", "")).startswith("InterruptedError:")
            or snapshot.get("child") is not None or snapshot.get("pending_launch") is not None
            or snapshot.get("queue_temporarily_suspended") is not False
            or (queue is not None and (not isinstance(queue, dict) or queue.get("resumed") is not True))):
        raise ValueError("Runner exit snapshot changed, has active work, or lacks restored queue evidence")
    steps = snapshot.get("completed_steps", {})
    expected = {f"evaluate_{model}_{h}x{w}" for model in ("tinyunet", "yolo_v2", "aihub") for h, w in SIZES}
    if ({name for name, step in steps.items() if step.get("kind") == "evaluation"} != expected
            or any(step.get("kind") == "benchmark" for step in steps.values())
            or steps.get("train", {}).get("kind") != "train"):
        raise ValueError("Cancellation counts differ from actual completed products")
    return {"confirmation": confirmation_id, "request": confirmation["request"],
            "gpu_runner_exit_snapshot": confirmation["gpu_runner_exit_snapshot"],
            "user_request": request["user_request"], "scope": request["scope"]}


def review(run, output, *, accuracy_only=False):
    run, output = Path(run).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    summary_path, state_path = run / "gpu_summary.json", run / "gpu_runner.json"
    state_id, state = identity(state_path), read(state_path)
    cancellation = summary = summary_id = None
    if accuracy_only:
        cancellation = cancelled_runtime_evidence(run, state)
        steps, request = state["completed_steps"], state["request"]
    else:
        summary_id, summary = identity(summary_path), read(summary_path)
        if (summary.get("complete") is not True or state.get("status") != "complete"
                or state.get("summary") != summary_id or state.get("request") != summary.get("request")
                or state.get("completed_steps") != summary.get("completed_steps")):
            raise ValueError("A frozen completed runner summary with matching provenance is required")
        steps, request = summary["completed_steps"], summary["request"]
    products = {}
    for name, step in steps.items():
        if step.get("kind") not in {"offline", "train", "evaluation", "benchmark"}:
            raise ValueError("Unknown completed product kind")
        path = checked(step["product"])
        if step["kind"] == "offline":
            suites = list(ET.parse(path).getroot().iter("testsuite"))
            if not suites or sum(int(x.get("tests", 0)) for x in suites) < 1 or any(int(x.get(k, 0)) for x in suites for k in ("failures", "errors")):
                raise ValueError("Offline fixture regression did not pass")
        else:
            products[name] = read(path)
    training = products["train"]
    if steps["train"]["kind"] != "train":
        raise ValueError("Training product kind differs")
    counts = training_evidence(training)
    if not accuracy_only and summary["training"] != training:
        raise ValueError("Summary training differs from original product")
    for value in training["checkpoints"].values():
        checked(value)
    checkpoints = {"aihub": training["checkpoints"]["best"], "yolo_v2": request["initializer"], "tinyunet": request["tiny_checkpoint"]}
    checked(request["evaluation_contract"])
    for value in checkpoints.values():
        checked(value)
    from compare_frozen_road import compare_reports
    comparisons = {}
    for height, width in SIZES:
        reports = {}
        for model in checkpoints:
            key = f"evaluate_{model}_{height}x{width}"
            report = products[key]
            if (steps[key]["kind"] != "evaluation" or report.get("evaluation_complete") is not True
                    or report.get("checkpoint_diagnostic") is not False or report.get("error")
                    or report.get("input_size") != [height, width] or report.get("checkpoint") != checkpoints[model]
                    or report.get("evaluation_contract") != request["evaluation_contract"]
                    or set(report.get("public_obstacle_anomaly_subsets", {})) != {"obstacle", "anomaly"}
                    or any(source.get("partial_validation") is not False for source in report["native_sources"].values())):
                raise ValueError(f"Incomplete/mismatched evaluation: {key}")
            reports[model] = report
        for model in ("tinyunet", "yolo_v2"):
            comparisons[f"{model}_{height}x{width}"] = enrich_comparison(compare_reports(reports[model], reports["aihub"]))
    runtime = []
    for height, width in (() if accuracy_only else SIZES):
        for views in (1, 8):
            matched = {}
            for model in ("yolo_v2", "aihub"):
                prefix = f"benchmark_{model}_{height}x{width}_views{views}_"
                entries = [value for name, value in products.items() if name.startswith(prefix) and steps[name]["kind"] == "benchmark"]
                if len(entries) != 1:
                    raise ValueError(f"Exactly one successful benchmark per condition required: {prefix}")
                matched[model] = entries[0]
                benchmark = entries[0]
                if (benchmark.get("input_size_hw") != [height, width] or benchmark.get("views") != views
                        or benchmark.get("status") != "completed"
                        or benchmark.get("provenance", {}).get("yolo", {}).get("checkpoint") != checkpoints[model]
                        or benchmark.get("provenance", {}).get("native_checkpoint") != checkpoints["tinyunet"]):
                    raise ValueError("Benchmark provenance differs: " + prefix)
            for mode in (("host_to_road_map", "batched_host_to_road_map") if views == 8 else ("host_to_road_map",)):
                runtime.append(runtime_pair(matched["yolo_v2"], matched["aihub"], mode))
                # TinyUNet in the same candidate invocation has identical runtime conditions.
                runtime.append(runtime_pair(matched["aihub"], matched["aihub"], mode, "tinyunet"))
    regressions = [{"pair": pair, **row} for pair, comparison in comparisons.items() for row in comparison["regressions"]]
    unknown = [{"pair": pair, **row} for pair, comparison in comparisons.items() for row in comparison["deltas"] if row["regressed"] is None]
    limitations = LIMITS if not accuracy_only else [*LIMITS[:5], LIMITS[-1],
        "사용자 요청으로 속도 측정을 취소했습니다. 지연·처리율·실행 메모리 비교와 속도 향상 판단을 수행하지 않았습니다."]
    report = {"schema_version": 1, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "accuracy_only" if accuracy_only else "accuracy_and_runtime",
              "accuracy_complete": True, "source_run_complete": state.get("status") == "complete",
              "runtime_status": "cancelled_by_user" if accuracy_only else "completed",
              "runtime_comparison_performed": not accuracy_only, "runtime_cancellation": cancellation,
              "source_summary": summary_id, "source_runner": state_id,
              "verified_completed_products": {name: step["product"] for name, step in steps.items()},
              "training": counts, "comparisons": comparisons, "runtime_pairs": runtime,
              "accuracy_regressions": regressions, "unavailable_accuracy_deltas": unknown, "limitations": limitations,
              "runtime_regressions": [{"pair_index": index, "metric": key, **value} for index, pair in enumerate(runtime) if pair["comparable"] for key, value in pair["metrics"].items() if value["regressed"]],
              "all_accuracy_metrics_nonregressing": not regressions and not unknown,
              "training_performed": False, "threshold_tuning_performed": False, "deployment_approved": False,
              "next_actions": ["악화된 데이터셋·주야간·장애물 지표를 우선 검토하고 오분류 표본을 분류합니다.",
                               "개선 실험은 별도 버전의 학습 데이터와 사전 고정한 조건으로 설계하고 동일 검증 계약으로 재비교합니다.",
                               "공개 40장으로 학습·튜닝하지 않으며 새 독립 장애물·야간·영상 검증셋을 확보합니다.",
                               "속도 저하는 입력 크기·경량화·증류 후보를 별도 실험한 뒤 정확도·지연을 함께 비교합니다."]}
    if accuracy_only:
        report["next_actions"][0] = "악화된 출처·조건의 집계 지표를 검토하고, 학습·개발 분할에서 개선 가설을 세웁니다."
        report["next_actions"][-1] = "속도 측정 취소를 유지하며, 정확도 개선 검토만 이어갑니다. 새로운 속도 측정을 자동으로 예약하지 않습니다."
    # Recheck report products after reading/comparing; never write over an earlier review.
    if summary_id is not None:
        checked(summary_id)
    if cancellation:
        for name in ("confirmation", "request", "gpu_runner_exit_snapshot"):
            checked(cancellation[name])
    checked(state_id)
    for step in steps.values():
        checked(step["product"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "feedback.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["AI Hub 학습 완료 후 비교 및 개선 검토", "", f"실제 3 epoch, {counts['train_batches']:,} 배치, {counts['optimizer_updates']:,}회 유효 업데이트를 확인했습니다.",
             "원본 평가 보고서의 동일 입력·정답·임계값·분모를 확인하고 차이를 다시 계산했습니다.", "", "도로 IoU와 장애물→도로 오인율 (기존 → 추가 학습)", ""]
    if accuracy_only:
        lines[4:4] = ["정확도 평가 6건을 검토했습니다. 속도 측정은 사용자 요청으로 취소되어 전체 실행 성능 측정 완료를 뜻하지 않습니다."]
    for pair, comparison in comparisons.items():
        lines.append(pair)
        for row in comparison["deltas"]:
            if row["metric"] not in {"road_iou", "micro_obstacle_as_road_rate"}:
                continue
            if row["percentage_point_change"] is None:
                lines.append(f"- {row['scope']} / {row['metric']}: 측정 불가")
            else:
                relative = row["relative_change_percent"]
                relative_text = "산출 불가(기준 0 또는 결측)" if relative is None else f"{relative:+.2f}%"
                lines.append(f"- {row['scope']} / {row['metric']}: {row['baseline']:.2%} → {row['candidate']:.2%}, {row['percentage_point_change']:+.2f} %p, 상대 변화 {relative_text}; 악화: {row['regressed']}")
        lines.append("")
    lines += [f"전체 정확도 지표에서 확인한 악화 항목: {len(regressions)}개. 모든 지표 향상을 가정하지 않습니다.", "", "실행 속도 (기존 → 추가 학습)", ""]
    if accuracy_only:
        lines += ["사용자 요청으로 취소했습니다. 속도·실행 메모리 비교와 개선율은 산출하지 않았습니다.", ""]
    for pair in runtime:
        label = f"{pair['baseline_model']} / {pair['input_size']} / {pair['views']}영상 / {pair['mode']}"
        if not pair["comparable"]:
            lines.append(f"- {label}: 비교 제외 — {pair['skip_reason']}")
            for side, raw in pair["raw_measurements"].items():
                if raw:
                    values = {key: raw.get(key) for key in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second")}
                    lines.append(f"  {side} 개별 측정값(변화율 산출 안 함): {values}")
            continue
        metrics = pair["metrics"]
        values = [f"{key}: {metrics[key]['baseline']:.2f} → {metrics[key]['candidate']:.2f}" for key in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second")]
        lines.append(f"- {label}: {'; '.join(values)}. 지연 단위 ms, 처리율 단위 전체 카메라 작업 cycle/s.")
    lines += ["", "전체 정확도 악화 지표와 취소 근거는 feedback.json에 기록했습니다." if accuracy_only else
              "GPU 할당·예약 최대와 RAM RSS 표본 최대의 조건별 비교, 전체 악화 지표는 feedback.json에 기록했습니다.", ""]
    lines += [f"- {line}" for line in limitations]
    lines += ["", "다음 개선 검토", ""] + [f"- {line}" for line in report["next_actions"]]
    lines += ["", "이 검토는 추가 학습·임계값 변경·실차 제어·운용 모델 교체를 자동으로 실행하지 않습니다."]
    (output / "feedback.ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accuracy-only", action="store_true", help="Review six completed accuracy products only, requiring verified speed_cancellation_v1 evidence")
    args = parser.parse_args()
    result = review(args.run, args.output, accuracy_only=args.accuracy_only)
    print(json.dumps({"complete": result["complete"], "output": str(args.output.resolve()),
                      "accuracy_regressions": len(result["accuracy_regressions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
