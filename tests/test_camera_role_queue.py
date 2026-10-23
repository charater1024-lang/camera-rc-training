"""Follow-up queue provenance and ordering checks; no GPU or actuator is used."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("role_queue_under_test", SCRIPTS / "run_camera_role_queue.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
finally:
    sys.path.remove(str(SCRIPTS))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_upstream(project):
    source_entries, vals = [], []
    for name in sorted(set(queue.SOURCE_ROOTS) | queue.VALIDATION_ROOTS):
        root = project / "data" / name
        root.mkdir(parents=True)
        splits = ("train", "val", "test") if name in queue.SOURCE_ROOTS else ("val", "test")
        manifests, entries = {}, {}
        for split in splits:
            key = "bdd_" + split if name == "public_v2" else split
            path = root / (split + ".jsonl")
            path.write_text('{"fixture": "' + name + "_" + split + '"}\n', encoding="utf-8")
            record = {"manifest": str(path), "sha256": queue.digest(path), "samples": 1}
            manifests[key] = record
            entries[split] = {**record, "report_key": key}
            if split == "val":
                vals.append({"name": name, "manifest": str(path)})
        write(root / "report.json", {"complete": True, "label_space_id": queue.LABEL_SPACE,
                                      "manifests": manifests})
        source_entries.append({"root": str(root), "report_sha256": queue.digest(root / "report.json"),
                               "manifests": entries})
    stage = project / "data" / queue.UPSTREAM
    stage.mkdir()
    combined = stage / "train.jsonl"
    combined.write_text('{"all": true}\n', encoding="utf-8")
    config = project / "configs/local" / (queue.UPSTREAM + ".json")
    write(config, {"data": {"train_manifest": str(combined), "val_manifests": vals}})
    write(stage / "report.json", {"complete": True, "label_space_id": queue.LABEL_SPACE,
           "config_sha256": queue.digest(config), "train_manifest_sha256": queue.digest(combined),
           "source_manifests": source_entries})
    run = project / "runs" / queue.UPSTREAM
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"test fixture checkpoint")
    write(run / "summary.json", {"training_complete": True, "diagnostic": False,
                                  "best_checkpoint": str(run / "best.pt")})
    write(run / "evaluation/comparison.json", {"deployment_approved": False,
          "candidate_checkpoint": {"sha256": queue.digest(run / "best.pt")}})
    return stage


def public_report():
    return {"complete": True, "image_masks_decoded_and_validated": True,
            "models_validated_for_control": False, "synchronized_multiview_samples_created": False,
            "all_roles_have_real_rig_train_and_val_samples": False,
            "missing_real_rig_train_val": [f"{s}/{r}" for s in ("train", "val") for r in queue.REAL_ROLES],
            "rig_role_coverage": {r: {s: {"samples": 0, "conditions": []} for s in ("train", "val", "test")}
                                  for r in queue.REAL_ROLES}}


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_failed_upstream_prevents_any_training(tmp_path, status):
    write(tmp_path / "runs/large_data_queue_v1/status.json", {"status": status})
    q = queue.CameraRoleQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    with pytest.raises(RuntimeError, match="Upstream queue " + status):
        q.run()
    assert q.state["completed_steps"] == []
    assert not (tmp_path / "data" / queue.STAGE).exists()


def test_waits_until_predecessor_finishes_before_gpu_work(tmp_path, monkeypatch):
    path = tmp_path / "runs/large_data_queue_v1/status.json"
    write(path, {"status": "running"})
    q = queue.CameraRoleQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    sleeps = []
    def finish(seconds):
        sleeps.append(seconds)
        write(path, {"status": "complete"})
    monkeypatch.setattr(queue.time, "sleep", finish)
    q.wait_upstream()
    assert sleeps == [30]
    assert q.state["completed_steps"] == []
    assert q.state["models_validated_for_control"] is False


def test_stop_file_interrupts_wait(tmp_path):
    q = queue.CameraRoleQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    (q.output / "STOP_REQUESTED").write_text("stop", encoding="utf-8")
    with pytest.raises(InterruptedError):
        q.wait_upstream()


def test_complete_inputs_are_bound_to_source_hashes(tmp_path):
    make_upstream(tmp_path)
    result = queue.verify_upstream(tmp_path)
    assert set(result["sources"]) == set(queue.SOURCE_ROOTS) | queue.VALIDATION_ROOTS
    assert result["checkpoint_sha256"] == queue.digest(Path(result["checkpoint"]))
    assert "train" not in result["sources"]["camvid_road_v1"]


@pytest.mark.parametrize("changed", ["manifest", "source_report", "config", "combined", "summary", "checkpoint"])
def test_completed_but_changed_upstream_is_rejected(tmp_path, changed):
    stage = make_upstream(tmp_path)
    path = {"manifest": tmp_path / "data/a2d2_road_v1/train.jsonl",
            "source_report": tmp_path / "data/acdc_road_v1/report.json",
            "config": tmp_path / "configs/local" / (queue.UPSTREAM + ".json"),
            "combined": stage / "train.jsonl", "summary": tmp_path / "runs" / queue.UPSTREAM / "summary.json",
            "checkpoint": tmp_path / "runs" / queue.UPSTREAM / "best.pt"}[changed]
    if changed == "summary":
        report = queue.read_json(path)
        report["diagnostic"] = True
        write(path, report)
    else:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n ")
    with pytest.raises(ValueError):
        queue.verify_upstream(tmp_path)


def test_missing_source_report_cannot_be_ignored(tmp_path):
    stage = make_upstream(tmp_path)
    report = queue.read_json(stage / "report.json")
    report["source_manifests"].pop()
    write(stage / "report.json", report)
    with pytest.raises(ValueError, match="reports are missing"):
        queue.verify_upstream(tmp_path)


def test_incomplete_dataset_cannot_advance(tmp_path):
    make_upstream(tmp_path)
    (tmp_path / "data/a2d2_road_v1/.incomplete").write_text("in progress", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete preparation"):
        queue.verify_upstream(tmp_path)


def test_no_real_camera_data_is_explicitly_accepted_as_gap():
    queue.validate_public_role_report(public_report())


@pytest.mark.parametrize("field", ["models_validated_for_control", "synchronized_multiview_samples_created",
                                   "all_roles_have_real_rig_train_and_val_samples"])
def test_public_views_cannot_claim_real_rig_training_or_control(field):
    report = public_report()
    report[field] = True
    with pytest.raises(ValueError, match="cannot claim"):
        queue.validate_public_role_report(report)


def test_camera_report_missing_or_fake_samples_is_rejected():
    report = public_report()
    del report["rig_role_coverage"]["front_near"]
    with pytest.raises(ValueError, match="Eight"):
        queue.validate_public_role_report(report)
    report = public_report()
    report["rig_role_coverage"]["front_near"]["train"]["samples"] = 1
    with pytest.raises(ValueError, match="must not count"):
        queue.validate_public_role_report(report)


def test_reduced_validation_is_not_silently_treated_as_full_validation():
    report = public_report()
    report["image_masks_decoded_and_validated"] = False
    with pytest.raises(ValueError):
        queue.validate_public_role_report(report)


def test_output_overwrite_is_refused(tmp_path):
    path = tmp_path / "record.json"
    queue.save_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        queue.save_new(path, {"original": False})
    assert queue.read_json(path) == {"original": True}


def test_full_orchestration_preserves_validation_and_never_deploys(tmp_path, monkeypatch):
    make_upstream(tmp_path)
    write(tmp_path / "runs/large_data_queue_v1/status.json", {"status": "complete"})
    q = queue.CameraRoleQueue(tmp_path, tmp_path / "datasets", tmp_path / "queue")
    calls = []
    def command(name, arguments):
        args = list(map(str, arguments))
        calls.append((name, args))
        if name == "prepare_camera_roles":
            assert "--metadata-only" not in args
            spec = queue.read_json(Path(args[args.index("--spec") + 1]))
            assert spec["samples_per_epoch"] == 32000
            assert all(s["weight"] == 1 for s in spec["sources"])
            output = Path(args[args.index("--output") + 1])
            write(output / "report.json", public_report())
            write(output / "stage.json", {"data": {"val_manifests": [
                {"name": s["name"], "manifest": str(output / (s["name"] + "_val.jsonl"))}
                for s in spec["sources"]]}, "training": {}})
        elif name == "camera_roles_train":
            config = queue.read_json(Path(args[args.index("--config") + 1]))
            assert len(config["data"]["val_manifests"]) == 6
            checkpoint = Path(args[args.index("--init-from") + 1])
            run = Path(config["training"]["output_dir"])
            write(run / "summary.json", {"training_complete": True, "diagnostic": False,
                 "initialization": {"sha256": queue.digest(checkpoint)}})
            (run / "best.pt").write_bytes(b"new test fixture")
        else:
            checkpoint = Path(args[args.index("--checkpoint") + 1])
            write(Path(args[args.index("--output") + 1]), {"checkpoint": {"sha256": queue.digest(checkpoint)}})
    monkeypatch.setattr(q, "command", command)
    monkeypatch.setattr(queue, "compare_reports", lambda a, b: {"regressions": [], "deployment_approved": False})
    q.run()
    assert [name for name, _ in calls] == ["prepare_camera_roles", "camera_roles_train",
                                           "camera_roles_evaluate_baseline", "camera_roles_evaluate_candidate"]
    assert calls[2][1][calls[2][1].index("--config") + 1] == calls[3][1][calls[3][1].index("--config") + 1]
    assert q.state["status"] == "complete"
    assert q.state["training_complete"] is True
    assert q.state["deployed"] is False
    assert q.state["models_validated_for_control"] is False
    assert q.state["all_real_rig_roles_trained"] is False
