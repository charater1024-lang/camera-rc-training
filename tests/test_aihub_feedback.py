"""CPU-only post-run evidence and honest delta checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_aihub_feedback as review


def train_report():
    return {"training_complete": True, "diagnostic": False,
            "actual_args": {"epochs": 3, "batch": 8, "resume": False, "fraction": 1.0},
            "history": [{"epoch": i, "loss": .2} for i in (1, 2, 3)],
            "dataset": {"counts": {"train": 17}}, "batches_per_epoch": 3,
            "progress": {"train_batches": 9, "optimizer_attempts": 4, "optimizer_updates": 3, "amp_skipped_updates": 1}}


def benchmark(uuid="GPU-fixture"):
    timing = {"measured_cycles": 50, "views_per_cycle": 1, "p50_cycle_ms": 10., "p95_cycle_ms": 12.,
              "p99_cycle_ms": 14., "camera_work_cycles_per_second": 80.,
              "cuda_memory": {"torch_peak_allocated_gib": 1., "torch_peak_reserved_gib": 2.},
              "output_dtype": "float32", "output_shape": [1, 32, 32], "batch_size_per_call": 1, "model_calls_per_cycle": 1}
    entry = {"status": "completed", "host_to_road_map": timing, "resources": {"process_rss_peak_gib_sampled": 1.5}}
    return {"status": "completed", "hardware": {"device": "cuda:0", "actual_gpu_name": "fixture",
            "actual_gpu_memory_gib": 16., "cuda_version": "12.6", "cpu_intraop_threads": 2,
            "requested_allocator_budget_gib": 3., "nvidia_smi_before": {"matched_device": {"uuid": uuid}}},
            "torch_version": "2.9", "pillow_version": "12", "precision": "fp16", "input_size_hw": [32, 32],
            "input_images": [{"image": {"sha256": "a"}}], "views": 1, "iterations": 50,
            "warmup_cycles_per_mode": 10, "probability": "softmax0", "excluded": ["capture"],
            "label_space_id": "road_surface_v1", "class_names": ["road", "nonroad"], "road_class_id": 0,
            "input_shape_per_call": [1, 3, 32, 32], "batched_input_shape_per_call": None,
            "shared_instances_per_model": 1, "outputs_retained_until_cycle_check": 1,
            "models": {"yolo26x_sem": deepcopy(entry), "tinyunet": deepcopy(entry)}}


@pytest.mark.parametrize("old,new,direction,pp,relative,regressed", [
    (.5, .6, "higher", 10., 20., False), (.2, .1, "lower", -10., -50., False),
    (.1, .2, "lower", 10., 100., True), (0., .1, "lower", 10., None, True),
    (None, .1, "lower", None, None, None)])
def test_percentage_points_relative_and_direction(old, new, direction, pp, relative, regressed):
    result = review.change(old, new, direction, rate=True)
    assert result["percentage_point_change"] == (pytest.approx(pp) if pp is not None else None)
    assert result["relative_change_percent"] == (pytest.approx(relative) if relative is not None else None)
    improvement = relative * (1 if direction == "higher" else -1) if relative is not None else None
    assert result["relative_improvement_percent"] == (pytest.approx(improvement) if improvement is not None else None)
    assert result["regressed"] is regressed


def test_counts_are_not_percentage_points():
    pair = {"deltas": [{"metric": "all_obstacle_pixels_predicted_road_images", "baseline": 2, "candidate": 1, "preferred_direction": "lower"}]}
    result = review.enrich_comparison(pair)
    assert result["deltas"][0]["percentage_point_change"] is None
    assert result["deltas"][0]["candidate_minus_baseline"] == -1


def test_training_requires_actual_three_epochs():
    assert review.training_evidence(train_report())["epochs"] == 3
    for field, value in (("history", [{"epoch": 1}]), ("training_complete", False), ("batches_per_epoch", 4)):
        report = train_report()
        report[field] = value
        with pytest.raises(ValueError):
            review.training_evidence(report)
    report = train_report()
    report["progress"]["optimizer_updates"] = 0
    with pytest.raises(ValueError):
        review.training_evidence(report)


@pytest.mark.parametrize("field,value", [("precision", "fp32"), ("input_size_hw", [64, 64]),
                                           ("views", 8), ("input_images", [])])
def test_runtime_protocol_mismatch_explicitly_skipped(field, value):
    a, b = benchmark(), benchmark()
    b[field] = value
    result = review.runtime_pair(a, b, "host_to_road_map")
    assert not result["comparable"] and field.replace("_hw", "") in result["skip_reason"]
    assert result["raw_measurements"]["candidate"]["p95_cycle_ms"] == 12


@pytest.mark.parametrize("field,value", [("requested_allocator_budget_gib", 7.), ("actual_gpu_name", "other")])
def test_runtime_hardware_mismatch_skipped(field, value):
    a, b = benchmark(), benchmark()
    b["hardware"][field] = value
    assert not review.runtime_pair(a, b, "host_to_road_map")["comparable"]


def test_missing_uuid_crossrun_skip_but_same_invocation_pair_valid():
    a, b = benchmark(None), benchmark(None)
    assert "UUID" in review.runtime_pair(a, b, "host_to_road_map")["skip_reason"]
    result = review.runtime_pair(a, a, "host_to_road_map", "tinyunet")
    assert result["comparable"] and result["same_benchmark_invocation"]


def test_runtime_cycle_metric_not_aggregate_image_fps():
    a, b = benchmark(), benchmark()
    b["models"]["yolo26x_sem"]["host_to_road_map"]["camera_work_cycles_per_second"] = 100
    result = review.runtime_pair(a, b, "host_to_road_map")
    assert result["comparable"]
    assert result["metrics"]["camera_work_cycles_per_second"]["relative_change_percent"] == 25
    assert "images_per_second" not in result["metrics"]


def test_identity_mutation_rejected(tmp_path):
    p = tmp_path / "report.json"
    p.write_text('{}')
    saved = review.identity(p)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity changed"):
        review.checked(saved)


def test_incomplete_summary_and_existing_output_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        review.review(tmp_path, out)
    (tmp_path / "gpu_summary.json").write_text('{"complete":false}')
    (tmp_path / "gpu_runner.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="completed runner"):
        review.review(tmp_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_nonfinite_json_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        review.read(p)


def test_real_comparison_helper_used_and_provenance_preserved(tmp_path, monkeypatch):
    # Existing fake predictor generates real evaluator reports entirely on CPU.
    from test_frozen_road_evaluation import complete_export, make_contract
    from test_yolo_semantic_evaluation import FakePredictor
    import evaluate_frozen_road as frozen
    import evaluate_yolo_semantic as yolo
    import hashlib
    from compare_frozen_road import compare_reports
    monkeypatch.setattr(frozen, "OFFICIAL_SHA256", hashlib.sha256(b"fixture official").hexdigest())
    config, best, training, _ = complete_export(tmp_path)
    original = training.read_bytes()
    contract, _ = make_contract(tmp_path, public=True)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    report = frozen.evaluate(contract, "yolo26x-sem", config, best, training, tmp_path / "eval.json", device="cpu")
    result = review.enrich_comparison(compare_reports(report, report))
    assert result["regressions"] == []
    assert all(row["candidate_minus_baseline"] == 0 for row in result["deltas"])
    assert training.read_bytes() == original


def write_cancellation(run, state):
    folder = run / "speed_cancellation_v1"
    folder.mkdir()
    request = {"user_request": "속도 측정을 취소하자", "scope": "runtime benchmarks only",
               "runner": state["runner"], "training_complete": True, "accuracy_completed": 6,
               "benchmarks_completed": 0, "children_before": []}
    request_path, snapshot_path = folder / "request.json", folder / "gpu_runner.exit.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    snapshot_path.write_text(json.dumps(state), encoding="utf-8")
    confirmation = {"status": "cancelled_by_user", "run": str(run.resolve()), "runner_exited": True,
                    "queue_restored": True, "runtime_steps_completed": 0, "training_complete": True,
                    "accuracy_steps_completed": 6, "request": review.identity(request_path),
                    "gpu_runner_exit_snapshot": review.identity(snapshot_path)}
    (folder / "confirmation.json").write_text(json.dumps(confirmation), encoding="utf-8")
    return folder, confirmation


def stopped_state(steps):
    return {"status": "recovery_required", "error": "InterruptedError: Stopped before starting another GPU command",
            "runner": {"pid": 123, "created": 1234., "command": ["python", "runner.py"]},
            "child": None, "pending_launch": None, "suspended_queue": {"pid": 789, "resumed": True},
            "queue_temporarily_suspended": False, "completed_steps": steps}


@pytest.mark.parametrize("accuracy_only", [False, True])
def test_full_completed_review_and_product_tampering(tmp_path, monkeypatch, accuracy_only):
    from test_frozen_road_evaluation import complete_export, make_contract
    from test_yolo_semantic_evaluation import FakePredictor
    import evaluate_frozen_road as frozen
    import evaluate_yolo_semantic as yolo
    import hashlib
    monkeypatch.setattr(frozen, "OFFICIAL_SHA256", hashlib.sha256(b"fixture official").hexdigest())
    config, best, training_path, _ = complete_export(tmp_path)
    contract, _ = make_contract(tmp_path, public=True)
    monkeypatch.setattr(yolo, "YoloSemanticPredictor", FakePredictor)
    template = frozen.evaluate(contract, "yolo26x-sem", config, best, training_path, tmp_path / "eval.json", device="cpu")
    checkpoints = {}
    for name in ("aihub", "yolo_v2", "tinyunet"):
        path = tmp_path / f"{name}_fixture.pt"
        path.write_bytes(name.encode())
        checkpoints[name] = review.identity(path)
    run = tmp_path / "run"
    run.mkdir()
    steps = {}
    def save_step(name, report, kind):
        path = run / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        steps[name] = {"product": review.identity(path), "kind": kind}
    training = train_report()
    training["checkpoints"] = {"best": checkpoints["aihub"], "last": checkpoints["aihub"]}
    save_step("train", training, "train")
    for height, width in review.SIZES:
        for model in checkpoints:
            value = deepcopy(template)
            value["checkpoint"] = checkpoints[model]
            value["input_size"] = [height, width]
            save_step(f"evaluate_{model}_{height}x{width}", value, "evaluation")
        for views in (1, 8):
            for model in ("yolo_v2", "aihub"):
                value = benchmark()
                value.update(input_size_hw=[height, width], views=views)
                value["provenance"] = {"yolo": {"checkpoint": checkpoints[model]}, "native_checkpoint": checkpoints["tinyunet"]}
                for entry in value["models"].values():
                    entry["host_to_road_map"]["views_per_cycle"] = views
                    if views == 8:
                        entry["batched_host_to_road_map"] = deepcopy(entry["host_to_road_map"])
                        entry["batched_host_to_road_map"]["batch_size_per_call"] = 8
                save_step(f"benchmark_{model}_{height}x{width}_views{views}_3gib", value, "benchmark")
    request = {"initializer": checkpoints["yolo_v2"], "tiny_checkpoint": checkpoints["tinyunet"],
               "evaluation_contract": review.identity(contract)}
    if accuracy_only:
        steps = {name: step for name, step in steps.items() if step["kind"] != "benchmark"}
        state = stopped_state(steps)
        state["request"] = request
        write_cancellation(run, state)
        monkeypatch.setattr(review, "runtime_pair", lambda *a, **k: pytest.fail("Accuracy-only must never compare runtimes"))
    else:
        summary = {"complete": True, "training": training, "request": request, "completed_steps": steps}
        summary_path = run / "gpu_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        state = {"status": "complete", "request": request, "completed_steps": steps, "summary": review.identity(summary_path)}
    (run / "gpu_runner.json").write_text(json.dumps(state), encoding="utf-8")
    source_before = {p: p.read_bytes() for p in run.glob("*.json")}
    report = review.review(run, run / "feedback", accuracy_only=accuracy_only)
    assert report["complete"] and report["training_performed"] is False
    assert len(report["comparisons"]) == 4 and len(report["runtime_pairs"]) == (0 if accuracy_only else 12)
    assert report["runtime_comparison_performed"] is (not accuracy_only)
    assert all(pair["comparable"] for pair in report["runtime_pairs"])
    assert report["accuracy_regressions"] == []
    assert (run / "feedback/feedback.ko.md").is_file()
    if accuracy_only:
        assert report["scope"] == "accuracy_only" and report["accuracy_complete"]
        assert report["runtime_status"] == "cancelled_by_user" and not report["source_run_complete"]
        assert report["source_summary"] is None and not (run / "gpu_summary.json").exists()
        korean = (run / "feedback/feedback.ko.md").read_text(encoding="utf-8")
        assert "사용자 요청으로 취소" in korean and "메모리 비교와 개선율은 산출하지 않았습니다" in korean
        assert "조건별 비교, 전체 악화 지표" not in korean
    assert all(p.read_bytes() == original for p, original in source_before.items())
    changed = Path(steps["evaluate_aihub_192x320"]["product"]["path"])
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        review.review(run, run / "changed_feedback", accuracy_only=accuracy_only)
    assert not (run / "changed_feedback").exists()


@pytest.fixture
def cancellation_evidence(tmp_path):
    steps = {f"evaluate_{model}_{h}x{w}": {"kind": "evaluation"} for model in ("tinyunet", "yolo_v2", "aihub") for h, w in review.SIZES}
    steps["train"] = {"kind": "train"}
    state = stopped_state(steps)
    folder, confirmation = write_cancellation(tmp_path, state)
    return tmp_path, state, folder, confirmation


@pytest.mark.parametrize("field,value", [("status", "stopped"), ("runner_exited", False), ("queue_restored", False),
    ("runtime_steps_completed", 1), ("runtime_steps_completed", False), ("training_complete", False),
    ("accuracy_steps_completed", 5), ("run", "relative-run")])
def test_invalid_cancellation_confirmation_rejected(cancellation_evidence, field, value):
    run, state, folder, confirmation = cancellation_evidence
    confirmation[field] = value
    (folder / "confirmation.json").write_text(json.dumps(confirmation))
    with pytest.raises(ValueError, match="Explicit runtime cancellation"):
        review.cancelled_runtime_evidence(run, state)


@pytest.mark.parametrize("which", ["request", "gpu_runner_exit_snapshot"])
def test_cancellation_proof_hash_mismatch_rejected(cancellation_evidence, which):
    run, state, folder, confirmation = cancellation_evidence
    proof = Path(confirmation[which]["path"])
    proof.write_text(proof.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        review.cancelled_runtime_evidence(run, state)


def test_current_runner_changed_after_cancellation_rejected(cancellation_evidence):
    run, state, _, _ = cancellation_evidence
    state["status"] = "running"
    with pytest.raises(ValueError, match="exit snapshot changed"):
        review.cancelled_runtime_evidence(run, state)


@pytest.mark.parametrize("change", ["suspended_queue", "child", "pending_launch", "queue_temporarily_suspended", "error", "missing_eval", "extra_benchmark"])
def test_cancellation_cannot_hide_active_work_or_missing_accuracy(cancellation_evidence, change):
    run, state, folder, confirmation = cancellation_evidence
    if change == "missing_eval":
        del state["completed_steps"]["evaluate_aihub_192x320"]
    elif change == "extra_benchmark":
        state["completed_steps"]["benchmark"] = {"kind": "benchmark"}
    else:
        state[change] = {"suspended_queue": {"resumed": False}, "child": {"pid": 123}, "pending_launch": {"pid": 123},
                         "queue_temporarily_suspended": True, "error": "OutOfMemoryError"}[change]
    snapshot = Path(confirmation["gpu_runner_exit_snapshot"]["path"])
    snapshot.write_text(json.dumps(state))
    confirmation["gpu_runner_exit_snapshot"] = review.identity(snapshot)
    (folder / "confirmation.json").write_text(json.dumps(confirmation))
    with pytest.raises(ValueError):
        review.cancelled_runtime_evidence(run, state)


def test_accuracy_only_requires_explicit_cancellation_record(tmp_path):
    (tmp_path / "gpu_runner.json").write_text(json.dumps(stopped_state({})))
    with pytest.raises(FileNotFoundError):
        review.review(tmp_path, tmp_path / "feedback", accuracy_only=True)
    assert not (tmp_path / "feedback").exists()
