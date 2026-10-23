import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_aihub_progress as ui


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_training_is_not_started_and_probe_not_counted(tmp_path):
    save(tmp_path / "runs/aihub_followup_v1/capacity_probe_v1/batch4/capacity_report.json",
         {"status":"capacity_probe_completed","successful_updates":4,"optimizer_attempts":4})
    value = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {"available":False}))
    assert value["training"]["started"] is False
    assert value["training"]["fraction"] == 0
    assert value["training"]["optimizer_updates"] == 0
    assert value["probes"][0]["successful_updates"] == 4


def test_compact_progress_overrides_older_training_report(tmp_path):
    root = tmp_path / "runs/aihub_followup_v1/reduced_batch_v1/training"
    save(root / "training_report.json", {"progress":{"train_batches":100},"batches_per_epoch":1000,
         "current_epoch":1,"actual_args":{"epochs":3,"batch":4,"nbs":32}})
    save(root / "progress.json", {"progress":{"train_batches":1250,"optimizer_updates":160},
         "current_epoch":2,"batches_per_epoch":1000,"requested_epochs":3,"batch_in_epoch":250})
    value = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {}))["training"]
    assert value["batches_seen"] == 1250 and value["batch_in_epoch"] == 250
    assert value["fraction"] == 1250/3000 and value["optimizer_updates"] == 160
    assert value["physical_batch"] == 4


def test_log_tail_and_json_reads_are_bounded(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("가"*10000, encoding="utf-8")
    assert 0 < len(ui.tail(log)) <= 4000
    large = tmp_path / "large.json"
    large.write_text('"'+'a'*(3*1024**2)+'"',encoding="utf-8")
    assert ui.small_json(large) == {}
    large.write_text('{invalid',encoding="utf-8")
    assert ui.small_json(large) == {}


def test_gpu_measurement_is_cached_with_timeout(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout="Example GPU, 1024, 16384, 25")
    monkeypatch.setattr(ui.subprocess, "run", run)
    cache = ui.GPUCache()
    assert cache.read()["used_gib"] == 1
    assert cache.read()["utilization"] == 25
    assert len(calls) == 1 and calls[0]["timeout"] == 3


def test_only_fixed_routes_and_no_write_methods(tmp_path):
    page = tmp_path / "assets/aihub_progress.html"
    page.parent.mkdir()
    page.write_text("view",encoding="utf-8")
    cls = ui.handler(tmp_path)
    assert not hasattr(cls, "do_POST") and not hasattr(cls, "do_PUT")
    fake = SimpleNamespace(headers={"Host":"127.0.0.1:8765"},server=SimpleNamespace(server_port=8765),
                           path="/../../training_report.json",send_error=lambda status: errors.append(status))
    errors=[]
    cls.do_GET(fake)
    assert errors == [404]
    fake.headers={"Host":"untrusted.example:8765"}
    cls.do_GET(fake)
    assert errors[-1] == 403


def test_running_hides_stale_waiting_reason_and_phase(tmp_path):
    path = tmp_path / "runs/aihub_followup_v1/reduced_batch_v1/gpu_runner.json"
    save(path, {"status":"running","step":"train","waiting_reason":"old waiting text","waiting_phase":"training"})
    value = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {}))["runner"]
    assert value["reason"] == "" and value["waiting_phase"] is None
    save(path, {"status":"waiting_for_gpu","waiting_reason":"actual wait","waiting_phase":"evaluation"})
    value = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {}))["runner"]
    assert value["reason"] == "actual wait" and value["waiting_phase"] == "evaluation"


def test_pid_reuse_is_not_live_runner(monkeypatch):
    monkeypatch.setattr(ui.psutil,"Process",lambda pid:SimpleNamespace(create_time=lambda:2,cmdline=lambda:["same"]))
    assert ui.runner_liveness({"pid":4,"created":1,"command":["same"]})["state"] == "exited"
    assert ui.runner_liveness({"pid":4,"created":2,"command":["same"]})["state"] == "alive"


def test_inaccessible_process_is_unknown_not_dead(monkeypatch):
    def denied(pid): raise psutil.AccessDenied(pid)
    monkeypatch.setattr(ui.psutil,"Process",denied)
    assert ui.runner_liveness({"pid":4,"created":1,"command":["same"]})["state"] == "unknown"


def test_log_ansi_is_removed(tmp_path):
    path=tmp_path/"log"
    path.write_text("before\x1b[31mred\x1b[0m after",encoding="utf-8")
    assert ui.tail(path) == "beforered after"


def test_verified_benchmark_cancellation_hides_only_expected_stop(tmp_path):
    run = tmp_path / "runs/aihub_followup_v1/reduced_batch_v1"
    state = {"status": "recovery_required",
             "error": "InterruptedError: Stopped before starting another GPU command"}
    save(run / "gpu_runner.json", state)
    save(run / "speed_cancellation_v1/confirmation.json", {
        "status": "cancelled_by_user", "runner_exited": True, "queue_restored": True,
        "gpu_runner_exit_snapshot": {"sha256": hashlib.sha256((run / "gpu_runner.json").read_bytes()).hexdigest()}})
    result = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {}))["runner"]
    assert result["status"] == "benchmark_cancelled_by_user"
    assert result["reason"] == ""
    state["error"] = "unrelated failure after cancellation"
    save(run / "gpu_runner.json", state)
    result = ui.snapshot(tmp_path, SimpleNamespace(read=lambda: {}))["runner"]
    assert result["status"] == "recovery_required"
    assert result["reason"] == state["error"]
