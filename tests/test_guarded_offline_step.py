import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_guarded_offline_step as guard


def test_never_treats_waiting_or_running_queue_as_exited(tmp_path):
    path = tmp_path / "runs" / guard.QUEUES[0] / "status.json"
    path.parent.mkdir(parents=True)
    for state in ("running", "waiting_for_verified_downloads", "unknown"):
        path.write_text(json.dumps({"status": state, "pid": 1234}))
        with pytest.raises(RuntimeError, match="not exited"):
            guard.queues_exited(tmp_path)


def test_stale_stop_record_cannot_hide_live_queue(tmp_path, monkeypatch):
    path = tmp_path / "runs" / guard.QUEUES[0] / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "stopped", "pid": 1234}))
    monkeypatch.setattr(guard.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: ["python", "scripts/run_large_data_queue.py"]))
    with pytest.raises(RuntimeError, match="still active"):
        guard.queues_exited(tmp_path)
    monkeypatch.setattr(guard.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: ["other_app.exe"]))
    assert len(guard.queues_exited(tmp_path)) == 3  # PID reused, no app is touched.


def test_unknown_queue_can_be_absent_but_no_state_is_mutated(tmp_path):
    assert all(p["state"] == "not_configured" for p in guard.queues_exited(tmp_path))
    assert list(tmp_path.iterdir()) == []
