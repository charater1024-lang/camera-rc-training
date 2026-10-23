"""Queue orchestration checks without starting training, downloads, or a vehicle."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location("large_data_queue_under_test",
    Path(__file__).resolve().parents[1] / "scripts/run_large_data_queue.py")
queue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(queue)


def _identity(name, content="a"):
    return {"path": "/data/" + name, "sha256": content * 64, "bytes": 100}


def _native(matrix):
    tp, fn = matrix[0]
    fp, tn = matrix[1]
    return {"valid_pixels": tp + fn + fp + tn, "partial_validation": False,
        "batches": 1, "confusion_matrix": matrix,
        "class_iou": [tp / (tp + fn + fp), tn / (tn + fn + fp)],
        "road_safety_pixels": {"false_road_rate": fp / (fp + tn),
            "road_tp_pixels": tp, "road_fn_pixels": fn, "road_fp_pixels": fp, "road_tn_pixels": tn}}


def _track(name, count, errors):
    decision = {"threshold": 0.65, "comparison": ">=", "labelled_obstacle_pixels": 10,
        "obstacle_pixels_predicted_road": errors, "obstacle_as_road_rate": errors / 10}
    rows = [{"frame_id": f"validation_{i}", "image": _identity(f"{name}_{i}.png"),
             "ground_truth": _identity(f"{name}_{i}_labels.png", "b"),
             "labelled_obstacle_pixels": 10,
             "decisions": {"runtime_road": deepcopy(decision)}} for i in range(count)]
    return {"per_image": rows, "expected_public_gt_images": count,
            "aggregate": {"images": count, "decisions": {"runtime_road": {
                "threshold": 0.65, "comparison": ">=", "labelled_obstacle_pixels": 10 * count,
                "obstacle_pixels_predicted_road": errors * count,
                "images_with_labelled_obstacles": count,
                "micro_obstacle_as_road_rate": errors / 10,
                "mean_per_image_obstacle_as_road_rate": errors / 10}}}}


def _reports():
    baseline = {"config": _identity("same_config.json"),
        "validation_sources": [{"name": "bdd", "images": 1, "manifest": _identity("val.jsonl")}],
        "input_size": [192, 320], "label_space_id": "road_surface_v1", "runtime_threshold": 0.65,
        "checkpoint_diagnostic": False, "checkpoint": _identity("baseline.pt"),
        "native_sources": {"bdd": _native([[80, 20], [10, 90]])},
        "public_obstacle_anomaly_subsets": {"obstacle": _track("obstacle", 30, 2),
                                            "anomaly": _track("anomaly", 10, 2)}}
    candidate = deepcopy(baseline)
    candidate["checkpoint"] = _identity("candidate.pt", "c")
    candidate["native_sources"]["bdd"] = _native([[80, 20], [5, 95]])
    candidate["public_obstacle_anomaly_subsets"] = {"obstacle": _track("obstacle", 30, 1),
                                                   "anomaly": _track("anomaly", 10, 1)}
    return baseline, candidate


def test_improved_pixel_metrics_still_never_approve_deployment():
    baseline, candidate = _reports()
    result = queue.compare_reports(baseline, candidate)
    assert result["regressions"] == []
    assert result["deployment_approved"] is False
    assert result["decision"] == "research_only_do_not_deploy"
    assert "not fresh blind tests" in result["notice"]


def test_native_and_obstacle_regressions_are_exposed():
    baseline, candidate = _reports()
    candidate["native_sources"]["bdd"] = _native([[90, 10], [20, 80]])
    candidate["public_obstacle_anomaly_subsets"]["obstacle"] = _track("obstacle", 30, 3)
    result = queue.compare_reports(baseline, candidate)
    assert set(result["regressions"]) == {"bdd:nonroad_as_road", "obstacle:obstacle_as_road"}
    assert result["deployment_approved"] is False


@pytest.mark.parametrize("field", ["config", "validation_sources", "input_size", "label_space_id", "runtime_threshold"])
def test_changed_evaluation_protocol_is_rejected(field):
    baseline, candidate = _reports()
    candidate[field] = "changed"
    with pytest.raises(ValueError, match="protocol changed"):
        queue.compare_reports(baseline, candidate)


@pytest.mark.parametrize("which", ["baseline", "candidate"])
def test_diagnostic_checkpoint_cannot_be_claimed_as_full_comparison(which):
    baseline, candidate = _reports()
    (baseline if which == "baseline" else candidate)["checkpoint_diagnostic"] = True
    with pytest.raises(ValueError, match="Diagnostic"):
        queue.compare_reports(baseline, candidate)


def test_same_valid_pixel_total_does_not_hide_changed_class_denominators():
    baseline, candidate = _reports()
    candidate["native_sources"]["bdd"] = _native([[40, 10], [10, 140]])
    assert baseline["native_sources"]["bdd"]["valid_pixels"] == candidate["native_sources"]["bdd"]["valid_pixels"]
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_native_partial_validation_is_rejected():
    baseline, candidate = _reports()
    candidate["native_sources"]["bdd"]["partial_validation"] = True
    with pytest.raises(ValueError, match="partial"):
        queue.compare_reports(baseline, candidate)


@pytest.mark.parametrize("matrix", [[[80, 20], [-1, 101]], [[80, 20], [1.0, 99]],
                                    [[80, 20], [True, 99]], [[80, 20], [10]], [[80, 20], [5, 90]]])
def test_invalid_native_counts_or_total_are_rejected(matrix):
    baseline, candidate = _reports()
    candidate["native_sources"]["bdd"]["confusion_matrix"] = matrix
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_native_reported_rate_must_agree_with_confusion_matrix():
    baseline, candidate = _reports()
    candidate["native_sources"]["bdd"]["road_safety_pixels"]["false_road_rate"] = 0.0
    with pytest.raises(ValueError, match="inconsistent"):
        queue.compare_reports(baseline, candidate)


def test_public_changed_content_hash_is_rejected_even_with_same_counts():
    baseline, candidate = _reports()
    candidate["public_obstacle_anomaly_subsets"]["obstacle"]["per_image"][0]["ground_truth"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identities changed"):
        queue.compare_reports(baseline, candidate)


@pytest.mark.parametrize("bad_rate", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_public_error_rate_cannot_look_like_no_regression(bad_rate):
    baseline, candidate = _reports()
    candidate["public_obstacle_anomaly_subsets"]["obstacle"]["aggregate"]["decisions"]["runtime_road"]["micro_obstacle_as_road_rate"] = bad_rate
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_public_image_count_must_match_per_image_identities():
    baseline, candidate = _reports()
    candidate["public_obstacle_anomaly_subsets"]["obstacle"]["aggregate"]["images"] = 29
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_public_per_image_denominators_must_match_even_when_aggregate_matches():
    baseline, candidate = _reports()
    rows = candidate["public_obstacle_anomaly_subsets"]["obstacle"]["per_image"]
    rows[0]["labelled_obstacle_pixels"] = 9
    rows[1]["labelled_obstacle_pixels"] = 11
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_public_errors_cannot_exceed_ground_truth_pixels():
    baseline, candidate = _reports()
    candidate["public_obstacle_anomaly_subsets"]["obstacle"]["per_image"][0]["decisions"]["runtime_road"]["obstacle_pixels_predicted_road"] = 11
    with pytest.raises(ValueError):
        queue.compare_reports(baseline, candidate)


def test_public_duplicate_frames_are_rejected():
    baseline, candidate = _reports()
    rows = candidate["public_obstacle_anomaly_subsets"]["obstacle"]["per_image"]
    rows[1]["frame_id"] = rows[0]["frame_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        queue.compare_reports(baseline, candidate)


def _queue(tmp_path):
    return queue.Queue(tmp_path / "project", tmp_path / "datasets", tmp_path / "queue", timeout_hours=1)


def test_queue_stop_file_prevents_starting_subprocess(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    (instance.output / "STOP_REQUESTED").touch()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("A stopped queue started a child"))
    with pytest.raises(InterruptedError, match="STOP_REQUESTED"):
        instance.command("train", ["-m", "camera_rc", "train"])
    assert instance.state["completed_steps"] == []


def test_queue_deadline_does_not_start_or_claim_completed_work(tmp_path):
    instance = _queue(tmp_path)
    instance.deadline = 0
    with pytest.raises(TimeoutError, match="time budget"):
        instance.check_stop()
    assert instance.state["deployed"] is False


def test_wait_markers_never_advances_on_unparseable_completion_json(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    marker = tmp_path / "DOWNLOAD_COMPLETE.json"
    marker.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        instance.wait_markers([marker])
    assert instance.state["completed_steps"] == []


def test_waiting_can_be_cancelled_without_download_or_training_mutations(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: (instance.output / "STOP_REQUESTED").touch())
    with pytest.raises(InterruptedError):
        instance.wait_markers([tmp_path / "still-missing.json"])
    assert instance.state["status"] == "waiting_for_verified_downloads"
    assert not instance.project.exists()


def test_incomplete_preparation_is_not_reused(tmp_path):
    instance = _queue(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "report.json").write_text('{"complete": true}', encoding="utf-8")
    (prepared / ".incomplete").touch()
    with pytest.raises(ValueError, match="Incomplete preparation"):
        instance.prepared(prepared)


def test_failed_child_does_not_become_completed_step(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    child = SimpleNamespace(returncode=3, poll=lambda: 3)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="exit 3"):
        instance.command("failed_train", ["-m", "camera_rc", "train"])
    assert instance.state["completed_steps"] == []


def test_windows_stop_kills_child_process_tree_and_waits(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    calls = []

    class Child:
        pid = 12345
        returncode = None

        def poll(self):
            (instance.output / "STOP_REQUESTED").touch()
            return None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 1

    monkeypatch.setattr(queue, "sys", SimpleNamespace(platform="win32", executable="python.exe"))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(InterruptedError):
        instance.command("train", ["-m", "camera_rc", "train"])
    assert ["taskkill", "/PID", "12345", "/T", "/F"] in calls
    assert ("wait", 30) in calls
    assert instance.state["completed_steps"] == []


def test_run_orders_download_preparation_and_training_without_deployment(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    calls = []
    monkeypatch.setattr(instance, "wait_markers", lambda markers: calls.append(("wait", [str(p) for p in markers])))
    monkeypatch.setattr(instance, "prepared", lambda path: False)
    monkeypatch.setattr(instance, "command", lambda name, args: calls.append(("prepare", name, args)))
    monkeypatch.setattr(instance, "stage", lambda name, sources: calls.append(("stage", name, sources)))
    instance.run()
    assert [(entry[0], entry[1] if entry[0] != "wait" else len(entry[1])) for entry in calls] == [
        ("wait", 3), ("prepare", "prepare_vkitti2"), ("stage", "large_vkitti_stage1_v1"),
        ("wait", 2), ("prepare", "prepare_acdc"), ("prepare", "prepare_a2d2"), ("stage", "large_all_stage2_v1")]
    assert "--root" in calls[5][2]  # The A2D2 adapter's actual CLI.
    assert instance.state["status"] == "complete"
    assert instance.state["deployed"] is False


@pytest.mark.parametrize("summary", [{"training_complete": False, "diagnostic": False},
                                      {"training_complete": True, "diagnostic": True}])
def test_incomplete_or_diagnostic_training_never_advances_to_evaluation(tmp_path, monkeypatch, summary):
    instance = _queue(tmp_path)
    run = instance.project / "runs/experiment"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    calls = []
    monkeypatch.setattr(instance, "command", lambda name, arguments: calls.append(name))
    with pytest.raises(ValueError, match="Incomplete or diagnostic"):
        instance.stage("experiment", [tmp_path / "prepared"])
    assert calls == ["experiment_recipe", "experiment_train"]
    assert not (run / "evaluation").exists()
    assert instance.state["deployed"] is False


def test_completed_stage_evaluates_baseline_then_candidate_with_same_config(tmp_path, monkeypatch):
    instance = _queue(tmp_path)
    run = instance.project / "runs/experiment"
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"training_complete": true, "diagnostic": false}', encoding="utf-8")
    baseline, candidate = _reports()
    calls = []

    def command(name, arguments):
        calls.append((name, arguments))
        if "_evaluate_" in name:
            report = baseline if name.endswith("_baseline") else candidate
            output = Path(arguments[arguments.index("--output") + 1])
            output.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(instance, "command", command)
    instance.stage("experiment", [tmp_path / "prepared"])
    assert [name for name, _ in calls] == ["experiment_recipe", "experiment_train",
        "experiment_evaluate_baseline", "experiment_evaluate_candidate"]
    evaluation_configs = [args[args.index("--config") + 1] for name, args in calls if "_evaluate_" in name]
    assert evaluation_configs[0] == evaluation_configs[1]
    recipe_args = calls[0][1]
    assert recipe_args[recipe_args.index("--init-from") + 1] == instance.project / "runs/bdd100k_2018_baseline_v1/best.pt"
    result = json.loads((run / "evaluation/comparison.json").read_text(encoding="utf-8"))
    assert result["deployment_approved"] is False
    assert instance.state["status"] == "stage_complete"
