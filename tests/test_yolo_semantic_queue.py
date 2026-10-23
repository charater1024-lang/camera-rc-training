import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("semantic_experiment", SCRIPTS / "run_yolo_semantic_experiment.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def experiment(tmp_path):
    return MODULE.Experiment(tmp_path, tmp_path / "dataset", tmp_path / "run")


def test_does_not_resume_queue_while_gpu_child_is_alive(tmp_path):
    run = experiment(tmp_path)
    calls = []
    run.queue = SimpleNamespace(pid=123, resume=lambda: calls.append("resume"))
    run.child = SimpleNamespace(pid=456, poll=lambda: None)
    run.state["suspended_queue"] = {"pid": 123, "resumed": False}
    run.release_gpu_queue()
    assert not calls
    assert run.state["status"] == "recovery_required"
    assert run.state["recovery_child_pid"] == 456


def test_resumes_idle_queue_after_child_exit(tmp_path):
    run = experiment(tmp_path)
    calls = []
    run.queue = SimpleNamespace(pid=123, resume=lambda: calls.append("resume"))
    run.child = SimpleNamespace(pid=456, poll=lambda: 1)
    run.state["suspended_queue"] = {"pid": 123, "resumed": False}
    run.release_gpu_queue()
    assert calls == ["resume"]
    assert run.queue is None
    assert json.loads((run.output / "status.json").read_text())["suspended_queue"]["resumed"] is True


def test_active_previous_queue_is_never_suspended(tmp_path, monkeypatch):
    import psutil
    import pytest
    run = experiment(tmp_path)
    status = tmp_path / "runs/large_data_queue_v1/status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "running", "pid": 123}), encoding="utf-8")
    monkeypatch.setattr(psutil, "Process", lambda _: (_ for _ in ()).throw(AssertionError("must not touch running queue")))
    with pytest.raises(RuntimeError, match="not idle"):
        run.reserve_gpu_queue()
