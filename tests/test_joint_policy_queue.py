"""Ordering/provenance checks for synthetic joint training, without GPU execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("joint_queue_under_test", SCRIPTS / "run_joint_policy_queue.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
finally:
    sys.path.remove(str(SCRIPTS))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def predecessor(project):
    run = project / "runs" / queue.ROLE_STAGE
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"role checkpoint fixture")
    checksum = queue.digest(checkpoint)
    state = {"status": "complete", "training_complete": True, "diagnostic": False,
             "models_validated_for_control": False, "deployed": False,
             "checkpoint": str(checkpoint), "checkpoint_sha256": checksum}
    write(project / "runs/camera_role_queue_v1/status.json", state)
    write(run / "summary.json", {"training_complete": True, "diagnostic": False, "best_checkpoint": str(checkpoint)})
    write(run / "evaluation/comparison.json", {"candidate_checkpoint": {"sha256": checksum},
           "models_validated_for_control": False, "deployment_approved": False})
    return state


def downloads(root):
    for asset in ("metadata", "rgb"):
        extracted = {"asset": asset, "archive_sha256": "a" * 64, "files": 1}
        write(root / (asset + "_complete.json"), {"download": {"verified": True, "sha256": "a" * 64},
                                                  "extraction": extracted})
        write(root / asset / "EXTRACTION_COMPLETE.json", extracted)


def prepared(root):
    root.mkdir(parents=True)
    entries = {}
    for split in ("train", "val", "test"):
        path = root / (split + ".jsonl")
        path.write_text('{"split": "' + split + '"}\n', encoding="utf-8")
        entries[split] = {"manifest": str(path), "sha256": queue.digest(path), "samples": 1}
    report = {"complete": True, "partial_dataset": False, "nominal_timing_assumed": True,
              "configured_frame_period_s": 0.1, "synthetic_simulator_data": True,
              "models_validated_for_control": False, "history": 3, "views": 2,
              "future_dt_s": queue.FUTURE_DT_S, "manifests": entries}
    write(root / "report.json", report)
    return report


def template():
    return {"task": "joint_policy", "models_validated_for_control": False, "diagnostic_only": False,
            "data": {"image_size": [192, 320], "expected_history": 3, "expected_views": 2,
                     "future_dt_s": queue.FUTURE_DT_S}, "training": {}, "model": {"kept": True}}


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_predecessor_failure_propagates_without_training(tmp_path, status):
    write(tmp_path / "runs/camera_role_queue_v1/status.json", {"status": status})
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    with pytest.raises(RuntimeError, match="Role queue " + status):
        q.run()
    assert q.state["completed_steps"] == []
    assert not (tmp_path / "data" / queue.DATA_STAGE).exists()


def test_wait_prevents_simultaneous_gpu_work(tmp_path, monkeypatch):
    path = tmp_path / "runs/camera_role_queue_v1/status.json"
    write(path, {"status": "running"})
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    waits = []
    def finish(seconds):
        waits.append(seconds)
        write(path, {"status": "complete"})
    monkeypatch.setattr(queue.time, "sleep", finish)
    q.wait_role_queue()
    assert waits == [30]
    assert q.state["training_complete"] is False
    assert "trained_views" not in q.state


def test_stop_interrupts_wait(tmp_path):
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    (q.output / "STOP_REQUESTED").touch()
    with pytest.raises(InterruptedError):
        q.wait_role_queue()


def test_complete_role_checkpoint_is_bound_to_evaluation(tmp_path):
    predecessor(tmp_path)
    result = queue.verify_role_predecessor(tmp_path)
    assert result["checkpoint_sha256"] == queue.digest(Path(result["checkpoint"]))


@pytest.mark.parametrize("which", ["status_hash", "evaluation_hash", "checkpoint", "summary_incomplete", "diagnostic", "control"])
def test_wrong_hash_or_incomplete_role_training_rejected(tmp_path, which):
    predecessor(tmp_path)
    run = tmp_path / "runs" / queue.ROLE_STAGE
    status_path = tmp_path / "runs/camera_role_queue_v1/status.json"
    if which == "checkpoint":
        (run / "best.pt").write_bytes(b"changed")
    else:
        path, field, value = {
            "status_hash": (status_path, "checkpoint_sha256", "b" * 64),
            "evaluation_hash": (run / "evaluation/comparison.json", "candidate_checkpoint", {"sha256": "b" * 64}),
            "summary_incomplete": (run / "summary.json", "training_complete", False),
            "diagnostic": (run / "summary.json", "diagnostic", True),
            "control": (status_path, "models_validated_for_control", True)}[which]
        report = queue.read_json(path)
        report[field] = value
        write(path, report)
    with pytest.raises(ValueError):
        queue.verify_role_predecessor(tmp_path)


def test_download_extraction_marker_must_match(tmp_path):
    downloads(tmp_path)
    assert set(queue.verify_download_markers(tmp_path)) == {"metadata", "rgb"}
    path = tmp_path / "rgb/EXTRACTION_COMPLETE.json"
    value = queue.read_json(path)
    value["archive_sha256"] = "b" * 64
    write(path, value)
    with pytest.raises(ValueError, match="inconsistent"):
        queue.verify_download_markers(tmp_path)


@pytest.mark.parametrize("change", ["partial_dataset", "models_validated_for_control", "nominal_timing_assumed", "views", "hash"])
def test_preparation_provenance_cannot_overstate_scope(tmp_path, change):
    root = tmp_path / "prepared"
    report = prepared(root)
    if change == "hash":
        with (root / "train.jsonl").open("a", encoding="utf-8") as f:
            f.write(" ")
    else:
        report[change] = {"partial_dataset": True, "models_validated_for_control": True,
                          "nominal_timing_assumed": False, "views": 8}[change]
        write(root / "report.json", report)
    with pytest.raises(ValueError):
        queue.prepared_manifests(root)


def test_config_keeps_test_reserved_and_uses_three_full_epochs(tmp_path):
    source = template()
    original = deepcopy(source)
    config = queue.training_config(source, tmp_path / "data", tmp_path / "run", tmp_path / "best.pt")
    assert source == original
    assert config["training"]["epochs"] == 3
    assert config["training"]["batch_size"] == 2
    assert config["data"]["test_manifest"] != config["data"]["val_manifest"]
    assert config["model"] == original["model"]
    assert config["training"]["init_encoder"] == str(tmp_path / "best.pt")


@pytest.mark.parametrize("field", ["max_steps", "max_train_steps", "max_val_steps"])
def test_diagnostic_step_caps_cannot_be_silently_used(tmp_path, field):
    config = template()
    config["training"][field] = 1
    with pytest.raises(ValueError, match="step caps"):
        queue.training_config(config, tmp_path, tmp_path, tmp_path / "best.pt")


def test_command_plan_and_training_status_preserve_scope(tmp_path, monkeypatch):
    predecessor(tmp_path)
    dataset_root = tmp_path / "datasets/virtual_kitti2"
    downloads(dataset_root)
    write(tmp_path / "configs/joint_policy_8gb.json", template())
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    calls = []
    def command(name, arguments):
        args = list(map(str, arguments))
        calls.append((name, args))
        if name == "prepare_joint_policy":
            assert args[args.index("--frame-period-s") + 1] == "0.1"
            assert args[args.index("--stride") + 1] == "5"
            assert "--max-train-samples" not in args
            prepared(Path(args[args.index("--output") + 1]))
        else:
            config = queue.read_json(Path(args[args.index("--config") + 1]))
            assert config["data"]["val_manifest"].endswith("val.jsonl")
            run = Path(config["training"]["output_dir"])
            run.mkdir()
            (run / "best.pt").write_bytes(b"joint fixture")
            write(run / "summary.json", {"training_complete": True, "diagnostic": False,
                  "models_validated_for_control": False, "best_checkpoint": str(run / "best.pt"),
                  "checkpoint_sha256": queue.digest(run / "best.pt"), "full_requested_epochs_completed": True,
                  "requested_epochs": 3, "completed_epochs": 3, "trained_views": 2, "history_frames": 3,
                  "test_split_evaluated": False, "deployed": False,
                  "initialization": {"sha256": queue.digest(Path(config["training"]["init_encoder"]))}})
    monkeypatch.setattr(q, "command", command)
    q.run()
    assert [name for name, _ in calls] == ["prepare_joint_policy", "train_joint_policy"]
    assert q.state["status"] == "complete"
    assert q.state["training_complete"] is True
    assert q.state["trained_views"] == 2
    assert q.state["temporal_history"] == 3
    assert q.state["nominal_timing_assumed"] is True
    assert q.state["simulator_data"] is True
    assert q.state["models_validated_for_control"] is False
    assert q.state["deployed"] is False


def test_completed_command_with_incomplete_training_is_not_success(tmp_path, monkeypatch):
    predecessor(tmp_path)
    downloads(tmp_path / "datasets/virtual_kitti2")
    write(tmp_path / "configs/joint_policy_8gb.json", template())
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    def command(name, args):
        args = list(map(str, args))
        if name == "prepare_joint_policy":
            prepared(Path(args[args.index("--output") + 1]))
        else:
            write(tmp_path / "runs" / queue.TRAIN_STAGE / "summary.json", {"training_complete": False})
    monkeypatch.setattr(q, "command", command)
    with pytest.raises(ValueError, match="summary is incomplete"):
        q.run()
    assert q.state["status"] != "complete"
    assert "trained_views" not in q.state


def test_existing_results_are_preserved(tmp_path):
    predecessor(tmp_path)
    downloads(tmp_path / "datasets/virtual_kitti2")
    existing = tmp_path / "data" / queue.DATA_STAGE
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    q = queue.JointPolicyQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    with pytest.raises(FileExistsError):
        q.run()
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"
