import importlib.util
import json
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts/development.py"
    spec = importlib.util.spec_from_file_location("development_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def entry():
    return load_script()


def test_status_missing_data_is_not_training_readiness(entry, tmp_path):
    (tmp_path / "data").mkdir()
    result = entry.status(tmp_path)
    assert result["training_readiness"] == "prepare_real_datasets_first"
    assert result["runtime_checkpoint_exists"] is False
    assert result["real_model_validation"] == "not_assessed"
    assert result["camera_opened"] is result["serial_port_opened"] is False
    assert result["prepared_data"] == str(tmp_path / "data/public_v1")
    assert result["preparation_state"] == "missing"
    assert result["preparation_report_complete"] is None


def test_status_default_legacy_curriculum_still_requires_preflight(entry, tmp_path):
    prepared = tmp_path / "data/public_v1"
    prepared.mkdir(parents=True)
    (prepared / "curriculum.json").write_text("{}")
    result = entry.status(tmp_path)
    assert result["public_curriculum"] == str(prepared / "curriculum.json")
    assert result["training_readiness"] == "run_curriculum_preflight"
    assert result["preparation_state"] == "completion_not_reported"
    assert result["real_model_validation"] == "not_assessed"


def test_status_explicit_v2_report_is_not_fresh_validation(entry, tmp_path):
    prepared = tmp_path / "data/public_v2"
    prepared.mkdir(parents=True)
    (prepared / "curriculum.json").write_text("{}")
    (prepared / "report.json").write_text(json.dumps({"complete": True}))
    result = entry.status(tmp_path, prepared_data="data/public_v2")
    assert result["prepared_data"] == str(prepared)
    assert result["public_curriculum_exists"] is True
    assert result["preparation_report_complete"] is True
    assert result["preparation_state"] == "reported_complete_pending_preflight"
    assert result["training_readiness"] == "run_curriculum_preflight"
    assert result["real_model_validation"] == "not_assessed"


@pytest.mark.parametrize("marker,complete", [(True, True), (True, False), (False, False)])
def test_status_incomplete_preparation_takes_precedence(entry, tmp_path, marker, complete):
    prepared = tmp_path / "data/public_v2"
    prepared.mkdir(parents=True)
    (prepared / "curriculum.json").write_text("{}")
    (prepared / "report.json").write_text(json.dumps({"complete": complete, "error": "decode failed"}))
    if marker:
        (prepared / ".incomplete").write_text("pending")
    result = entry.status(tmp_path, prepared_data=prepared)
    assert result["preparation_state"] == "incomplete"
    assert result["training_readiness"] == "finish_dataset_preparation_first"
    assert result["preparation_incomplete_marker_exists"] is marker
    assert result["preparation_report_error"] == "decode failed"


def test_status_explicit_missing_directory_is_read_only(entry, tmp_path):
    prepared = tmp_path / "not-created"
    result = entry.status(tmp_path, prepared_data=prepared)
    assert result["prepared_data"] == str(prepared)
    assert result["prepared_data_exists"] is False
    assert result["preparation_state"] == "missing"
    assert result["training_readiness"] == "prepare_real_datasets_first"
    assert not prepared.exists()


@pytest.mark.parametrize("content", ["{partial", '"not an object"', '{"complete": "true"}'])
def test_status_invalid_report_returns_diagnostics(entry, tmp_path, content):
    prepared = tmp_path / "data/public_v2"
    prepared.mkdir(parents=True)
    (prepared / "report.json").write_text(content)
    result = entry.status(tmp_path, prepared_data=prepared)
    assert result["preparation_state"] == "invalid_report"
    assert result["preparation_report_read_error"]
    assert result["training_readiness"] == "inspect_preparation_report"


def test_status_cli_forwards_prepared_directory_without_verify(entry, tmp_path, monkeypatch, capsys):
    directory = tmp_path / "missing v2"
    received = []

    def inspect_status(*, prepared_data):
        received.append(prepared_data)
        return {"prepared_data": str(prepared_data), "preparation_state": "missing"}

    monkeypatch.setattr(entry, "status", inspect_status)
    monkeypatch.setattr(entry, "verify", lambda *a, **kw: pytest.fail("status must not run verify"))
    assert entry.main(["status", "--prepared-data", str(directory)]) == 0
    assert received == [directory]
    assert json.loads(capsys.readouterr().out)["preparation_state"] == "missing"


@pytest.mark.parametrize("kind,transport", [("camera", "dry_run"), ("multicamera", "dry_run"),
                                         ("multireplay", "serial")])
def test_verify_rejects_live_devices_even_with_dry_run(entry, tmp_path, kind, transport):
    import json
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"source": {"kind": kind}, "transport": {"kind": transport},
                                "perception": {"allow_diagnostic": True}}))
    with pytest.raises(ValueError):
        entry.validate_fixture(path)


def test_zero_exit_does_not_hide_failed_benchmark_targets(entry):
    report = {"diagnostic_only": True, "model_diagnostic": True, "physical_output": False,
              "shared_model_instances": 1, "camera_names": list(entry.CAMERAS),
              "pipeline_latency": {"frames": 100}, "valid_plan_frames": 100,
              "ram_process_within_limit": True, "system_available_floor_met_sampled": True,
              "processing_deadline_p99_met_on_this_device": False,
              "group_period_p99_met_on_this_device": True}
    with pytest.raises(ValueError, match="deadline"):
        entry.validate_benchmark(report, 100)


def test_training_without_actual_weight_updates_is_rejected(entry):
    report = {"diagnostic_only": True, "real_driving_validated": False, "training_complete": True,
              "stages": 2, "weights_transferred": True, "global_holdouts_checked": True,
              "camera_holdouts_evaluated": list(entry.CAMERAS), "resume_skipped_completed_stages": True,
              "optimizer_steps_by_stage": {"front_only": {"applied": 1}, "four_views": {"applied": 0}}}
    with pytest.raises(ValueError, match="no optimizer updates"):
        entry.validate_training(report)


def test_replay_requires_stop_and_matching_summary(entry):
    runtime = {"success": True, "termination": "frame_limit_reached", "frames": 50,
               "physical_output": False, "shutdown_stop_sent": False, "latched_fault": None,
               "camera_names": list(entry.CAMERAS), "shared_model_instances": 1}
    launcher = {"success": True, "actuate": False, "runtime": runtime}
    with pytest.raises(ValueError, match="STOP"):
        entry.validate_replay(launcher, runtime, 50)
    runtime["shutdown_stop_sent"] = True
    entry.validate_replay(launcher, runtime, 50)
    with pytest.raises(ValueError, match="disagree"):
        entry.validate_replay(launcher, {**runtime, "frames": 49}, 50)


def test_verify_stops_after_child_failure_and_preserves_log(entry, tmp_path, monkeypatch):
    commands = []

    def failed(name, args, output, **kwargs):
        commands.append(name)
        (output / f"{name}.log").write_text("failure details")
        raise RuntimeError("dependency failure")

    monkeypatch.setattr(entry, "run_step", failed)
    output = tmp_path / "new_run"
    result = entry.verify(output, root=tmp_path)
    assert result["success"] is False
    assert commands == ["environment"]
    assert (output / ".incomplete").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "environment.log").read_text() == "failure details"
    with pytest.raises(FileExistsError):
        entry.verify(output, root=tmp_path)


def test_subprocess_uses_argument_list_and_preserves_failure_output(entry, tmp_path):
    output = tmp_path / "output with spaces"
    output.mkdir()
    with pytest.raises(RuntimeError, match="exited with 3"):
        entry.run_step("failure", ["-c", "print('diagnostic detail'); raise SystemExit(3)"],
                       output, root=tmp_path)
    assert "diagnostic detail" in (output / "failure.log").read_text()


def test_cuda_has_no_silent_cpu_fallback(entry):
    with pytest.raises(ValueError, match="CUDA"):
        entry.validate_environment({"success": True, "gpu_execution_verified": False,
                                    "camera_opened": False, "serial_port_opened": False}, "cuda")
