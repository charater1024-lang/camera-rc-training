"""Headless launch contracts; no installation, SSH, or real device access."""
import importlib.util
import json
from pathlib import Path
import signal

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_headless", ROOT / "scripts" / "run_headless.py")
headless = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(headless)


@pytest.fixture(autouse=True)
def ordinary_user(monkeypatch):
    if hasattr(headless.os, "geteuid"):
        monkeypatch.setattr(headless.os, "geteuid", lambda: 1000)


def fake_config(tmp_path):
    path = tmp_path / "config with spaces.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_default_launcher_uses_dry_run_and_unique_outputs(tmp_path, monkeypatch):
    config = fake_config(tmp_path)
    launches = []
    monkeypatch.setattr(headless, "_load_and_check", lambda path, actuate: launches.append(("validate", actuate)))
    def run(path, output, **kwargs):
        launches.append(("run", kwargs["actuate"]))
        output.mkdir(exist_ok=False)
        return {"success": True, "termination": "duration_reached"}
    monkeypatch.setattr(headless, "_run_drive", run)
    first = headless.launch(config, tmp_path / "logs with spaces", duration_s=1)
    second = headless.launch(config, tmp_path / "logs with spaces", duration_s=1)
    assert first["success"] and second["success"]
    assert first["output"] != second["output"]
    assert launches == [("validate", False), ("run", False)] * 2
    assert json.loads((Path(first["output"]) / "launcher.json").read_text(encoding="utf-8")) == first


def test_configuration_failure_does_not_start_runtime_or_create_logs(tmp_path, monkeypatch):
    def reject(path, actuate):
        raise ValueError("serial config requires --actuate")
    monkeypatch.setattr(headless, "_load_and_check", reject)
    monkeypatch.setattr(headless, "_run_drive", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="serial config"):
        headless.launch(fake_config(tmp_path), tmp_path / "logs")
    assert not (tmp_path / "logs").exists()


def test_real_config_checks_preserve_serial_and_physical_arm_contract(tmp_path, monkeypatch):
    config = tmp_path / "runtime.json"
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    config.write_text(json.dumps({"calibration": "calibration.json", "transport": {"kind": "serial"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="BOTH"):
        headless._load_and_check(config, False)
    # Device presence is verified separately; explicitly simulate a physical
    # device here so this launcher contract reaches the model validation gate.
    physical_checks = []
    def simulated_device(device, *, require_present=False):
        physical_checks.append(require_present)
        return "/dev/video0"
    monkeypatch.setattr("camera_rc.driving.config.camera_device_identity", simulated_device)
    with pytest.raises(ValueError, match="model"):
        headless._load_and_check(config, True)
    assert True in physical_checks


def test_signal_mapping_raises_once_then_restores_handlers():
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with headless.graceful_stop_signals() as requested:
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
        assert requested["signal"] == signal.SIGTERM
        handler(signal.SIGTERM, None)  # Must not interrupt STOP/close during cleanup.
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_sigterm_enters_existing_runtime_shutdown_stop(tmp_path, monkeypatch):
    import torch
    from camera_rc.driving import runtime
    from camera_rc.driving.benchmark import make_runtime_fixture

    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        fixture = make_runtime_fixture(tmp_path / "fixture", device="cpu")
        original_open = runtime.open_source
        def interrupting_source(cfg):
            source = original_open(cfg)
            def read():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            source.read = read
            return source
        monkeypatch.setattr(runtime, "open_source", interrupting_source)
        result = headless.launch(fixture["config"], tmp_path / "runs")
    finally:
        torch.set_num_threads(original_threads)
    assert result["success"] is True
    assert result["termination_signal"] == signal.SIGTERM
    assert result["runtime"]["termination"] == "interrupted"
    assert result["runtime"]["shutdown_stop_sent"] is True
    assert result["runtime"]["physical_output"] is False


def test_launcher_runtime_failure_is_nonzero_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(headless, "_load_and_check", lambda *args: {})
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(headless, "_run_drive", fail)
    code = headless.main(["--config", str(fake_config(tmp_path)), "--output-root", str(tmp_path / "runs")])
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["success"] is False
    assert "CUDA unavailable" in report["runtime"]["error"]


def test_cli_cannot_combine_dry_run_and_actuate(tmp_path):
    with pytest.raises(SystemExit) as exc:
        headless.main(["--config", str(fake_config(tmp_path)), "--dry-run", "--actuate"])
    assert exc.value.code == 2


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_duration_must_be_finite_positive(tmp_path, duration):
    with pytest.raises(ValueError, match="finite and positive"):
        headless.launch(fake_config(tmp_path), tmp_path / "runs", duration_s=duration)


def test_service_is_user_dry_run_without_restart_or_shell_command():
    unit = (ROOT / "deploy" / "systemd" / "camera-rc-dry-run.service").read_text(encoding="utf-8")
    lines = [line.strip() for line in unit.splitlines() if line.strip() and not line.startswith("#")]
    command = next(line for line in lines if line.startswith("ExecStart="))
    assert command.endswith(" --dry-run")
    assert "--actuate" not in command and "sh -c" not in command
    assert 'ExecStart="%h/자율 주행/camera-rc-training/.venv-linux/bin/python"' in command
    assert "Restart=no" in lines and "KillSignal=SIGTERM" in lines
    assert not any(line.startswith("User=") for line in lines)


def test_root_execution_is_rejected_before_opening_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(headless.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(headless, "_load_and_check", lambda *args: pytest.fail("must reject root first"))
    with pytest.raises(PermissionError, match="ordinary account"):
        headless.launch(tmp_path / "missing.json", tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
