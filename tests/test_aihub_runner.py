import json
from pathlib import Path
import sys
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_aihub_followup as runner


def fixture(tmp_path):
    args = SimpleNamespace(project=tmp_path, output=tmp_path / "followup", dataset=tmp_path / "export",
        config=tmp_path / "config.json", evaluation_contract=tmp_path / "contract.json",
        init_training_report=tmp_path / "parent.json", device=0, minimum_free_gib=12,
        maximum_utilization=10, ready_polls=3, poll_seconds=30, max_wait_hours=1,
        batch=32, workers=4, lr=0.0001)
    return runner.Runner(args)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_live_runner_lock_blocks_duplicate(tmp_path):
    original = fixture(tmp_path)
    original.acquire()
    second = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="already owns"):
        second.acquire()
    assert original.lock_path.exists()
    original.release_lock()


def test_stale_lock_is_archived_not_overwritten(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    save(work.lock_path, {"pid": 99999999, "created": 1, "command": ["old"]})
    monkeypatch.setattr(runner, "matching_process", lambda record: None)
    work.acquire()
    archived = list(work.output.glob(".gpu_runner.lock.stale.*"))
    assert len(archived) == 1 and runner.read(archived[0])["command"] == ["old"]
    work.release_lock()


def test_state_merge_preserves_intake_provenance(tmp_path):
    work = fixture(tmp_path)
    save(work.followup_path, {"download_complete_confirmed": False, "current_data_selection": "existingproof", "training_started": False})
    work.update("waiting_for_gpu", gpu={"free_mib": 5000})
    value = runner.read(work.followup_path)
    assert value["current_data_selection"] == "existingproof"
    assert value["download_complete_confirmed"] is False
    assert value["training_started"] is False
    assert value["gpu_runner"]["pid"] == work.current_process["pid"]


def test_process_creation_time_prevents_pid_reuse(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 99, cmdline=lambda: ["new"])
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: process)
    assert runner.matching_process({"pid": 1, "created": 98, "command": ["old"]}) is None


def test_access_denied_is_not_treated_as_exit(monkeypatch):
    def denied(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(runner.psutil, "Process", denied)
    with pytest.raises(psutil.AccessDenied):
        runner.matching_process({"pid": 1, "created": 98, "command": ["old"]})


@pytest.mark.parametrize("free,util,expected", [(12288,10,True),(12287,0,False),(16000,11,False),(8000,0,False)])
def test_gpu_capacity_and_utilization_are_both_required(free, util, expected):
    assert runner.gpu_is_available({"free_mib": free, "utilization_percent": util}, 12, 10) is expected


def test_gpu_wait_requires_consecutive_good_polls(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    reads = iter([13000,13000,8000,13000,13000,13000,13000])
    samples = []
    reservations = []
    def sample(device):
        value = next(reads)
        samples.append(value)
        return {"free_mib": value, "utilization_percent": 0}
    monkeypatch.setattr(runner, "gpu_sample", sample)
    monkeypatch.setattr(runner.time, "sleep", lambda value: None)
    monkeypatch.setattr(work, "reserve_queue", lambda: reservations.append(len(samples)) or True)
    work.wait_and_reserve()
    assert reservations == [6]


def test_busy_gpu_times_out_without_touching_queue(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    clock = iter([0,1,2,4000])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runner.time, "sleep", lambda value: None)
    monkeypatch.setattr(runner, "gpu_sample", lambda device: {"free_mib": 7000, "utilization_percent": 30})
    monkeypatch.setattr(work, "reserve_queue", lambda: pytest.fail("Busy GPU must not suspend queues"))
    with pytest.raises(TimeoutError):
        work.wait_and_reserve()
    assert runner.read(work.state_path)["status"] == "gpu_wait_timeout"


def test_owned_child_must_exit_before_queue_release(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    queue = SimpleNamespace(resume=lambda: pytest.fail("Child is still alive"))
    child = object()
    work.state.update(suspended_queue={"pid": 1, "resumed": False}, child={"pid": 2})
    monkeypatch.setattr(runner, "matching_process", lambda record: child if record["pid"] == 2 else queue)
    with pytest.raises(RuntimeError, match="child is still alive"):
        work.release_queue()


class FakeProcess:
    def __init__(self, pid, script):
        self.pid, self.script = pid, script
        self.suspended, self.resumed = False, False
        self.descendants = []
    def create_time(self): return self.pid + 100.0
    def cmdline(self): return ["python", "scripts/" + self.script]
    def children(self, recursive=True): return self.descendants
    def status(self): return "running"
    def suspend(self): self.suspended = True
    def resume(self): self.resumed = True


def queues(work, monkeypatch):
    definitions = [(1,"large_data_queue_v1","run_large_data_queue.py","waiting_for_verified_downloads"),
                   (2,"camera_role_queue_v1","run_camera_role_queue.py","waiting_for_upstream_queue"),
                   (3,"joint_policy_queue_v1","run_joint_policy_queue.py","waiting_for_role_queue")]
    processes = {pid: FakeProcess(pid, script) for pid, directory, script, status in definitions}
    for pid, directory, script, status in definitions:
        save(work.project / "runs" / directory / "status.json", {"pid": pid, "status": status})
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: processes[pid])
    return processes


def test_only_idle_upstream_is_reserved_and_resumed(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    processes = queues(work, monkeypatch)
    assert work.reserve_queue()
    assert processes[1].suspended and not processes[2].suspended and not processes[3].suspended
    work.release_queue()
    assert processes[1].resumed and not processes[2].resumed
    assert work.state["suspended_queue"]["resumed"] is True


def test_active_downstream_prevents_reservation(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    processes = queues(work, monkeypatch)
    processes[3].descendants = [object()]
    assert work.reserve_queue() is False
    assert not any(process.suspended for process in processes.values())


def test_queue_race_immediately_resumes_parent_without_touching_child(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    processes = queues(work, monkeypatch)
    def raced_suspend():
        processes[1].suspended = True
        processes[1].descendants = [object()]
    processes[1].suspend = raced_suspend
    assert work.reserve_queue() is False
    assert processes[1].resumed


def test_completed_step_is_revalidated_and_not_launched(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    product = work.output / "benchmark.json"
    save(product, {"status": "completed"})
    arguments = ["scripts/benchmark_yolo_semantic.py", "--output", product]
    command = [sys.executable,"-X","utf8","-B","-u",*map(str,arguments)]
    work.record_product("bench", product, "benchmark", command)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No completed step repeats"))
    assert work.command("bench", arguments, product, "benchmark", gpu=True)
    save(product, {"status":"completed", "tampered":True})
    with pytest.raises(RuntimeError, match="Completed step changed"):
        work.command("bench", arguments, product, "benchmark", gpu=True)


def test_incomplete_training_report_is_never_restarted(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    product = work.output / "training/training_report.json"
    save(product, {"training_complete":False, "diagnostic":False})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No incomplete training restart"))
    with pytest.raises(RuntimeError, match="Unjournaled product"):
        work.command("train", ["scripts/train_yolo_semantic.py"], product, "train", gpu=True)


def test_existing_successful_unjournaled_step_is_adopted(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    product = work.output / "benchmark.json"
    save(product, {"status": "completed"})
    work.state.update(step="bench", command=[sys.executable,"-X","utf8","-B","-u","script"])
    (work.output / "bench.log").write_text("finished", encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No repeated benchmark"))
    assert work.command("bench", ["script"], product, "benchmark", gpu=True)
    assert work.state["completed_steps"]["bench"]["product"] == runner.identity(product)


def test_only_recorded_oom_permits_budget_fallback(tmp_path):
    work = fixture(tmp_path)
    product = work.output / "benchmark.json"
    save(product, {"status":"partial_failure", "models":{"yolo":{"out_of_memory":True}}})
    work.state.update(step="bench", command=[sys.executable,"-X","utf8","-B","-u","script"])
    (work.output / "bench.log").write_text("recorded OOM", encoding="utf-8")
    assert work.command("bench", ["script"], product, "benchmark", gpu=True, allow_oom=True) is False
    save(product, {"status":"failed", "models":{"yolo":{"out_of_memory":False}}})
    with pytest.raises(ValueError, match="incomplete"):
        work.command("bench", ["script"], product, "benchmark", gpu=True, allow_oom=True)


def test_release_checks_live_popen_even_if_child_journal_missing(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.state["suspended_queue"] = {"pid": 1, "resumed": False}
    work.child = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(runner, "matching_process", lambda record: pytest.fail("Live child already blocks release"))
    with pytest.raises(RuntimeError, match="child is still alive"):
        work.release_queue()


def test_release_checks_pending_launch_gap(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.state["suspended_queue"] = {"pid": 1, "resumed": False}
    work.state["pending_launch"] = {"command": ["train"]}
    monkeypatch.setattr(runner, "matching_process", lambda record: None)
    monkeypatch.setattr(runner, "pending_launch_processes", lambda launch: [object()])
    with pytest.raises(RuntimeError, match="child is still alive"):
        work.release_queue()


def test_post_popen_journal_failure_waits_for_child(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.gpu_reserved = True
    monkeypatch.setattr(work, "ensure_gpu_phase", lambda phase: None)
    pending = [None, None, 0]
    child = SimpleNamespace(pid=50, poll=lambda: pending.pop(0) if pending else 0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(runner.psutil, "Process", lambda pid: object())
    monkeypatch.setattr(runner, "process_identity", lambda p: (_ for _ in ()).throw(RuntimeError("journal fixture failure")))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", lambda value: sleeps.append(value))
    monkeypatch.setattr(runner, "pending_launch_processes", lambda launch: [])
    with pytest.raises(RuntimeError, match="journal fixture failure"):
        work.command("bench", ["script"], work.output / "bench.json", "benchmark", gpu=True)
    assert len(sleeps) == 2 and work.child is None and work.state["pending_launch"] is None


def test_offline_existing_report_is_reused(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    xml = work.output / "offline_regression.xml"
    xml.write_text('<testsuites><testsuite tests="15" failures="0" errors="0" skipped="0"/></testsuites>', encoding="utf-8")
    monkeypatch.setattr(work, "command", lambda *a, **k: pytest.fail("Already passed offline tests must not repeat"))
    work.offline()
    assert work.state["offline_regression"]["tests"] == 15


def test_failed_offline_fixture_blocks_gpu_work(tmp_path):
    work = fixture(tmp_path)
    xml = work.output / "offline_regression.xml"
    xml.write_text('<testsuites><testsuite tests="15" failures="1" errors="0"/></testsuites>', encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        work.offline()


def test_frozen_contract_requires_hash_bound_completion_and_public_tracks(tmp_path):
    path = tmp_path / "contract.json"
    save(path, {"complete":True,"protocol_id":"frozen_road_pixels_v1", "public_tracks":{"obstacle":{},"anomaly":{}}})
    save(tmp_path / ".complete", {"contract_sha256": runner.identity(path)["sha256"]})
    assert runner.checked_contract_identity(path) == runner.identity(path)
    save(path, {"complete":True,"protocol_id":"frozen_road_pixels_v1", "public_tracks":{"obstacle":{}}})
    with pytest.raises(ValueError, match="completed independent"):
        runner.checked_contract_identity(path)


def test_existing_incomplete_report_with_journal_is_preserved(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    product = work.output / "training/training_report.json"
    save(product, {"training_complete":False, "diagnostic":False})
    command = [sys.executable,"-X","utf8","-B","-u","scripts/train_yolo_semantic.py"]
    work.state.update(step="train", command=command)
    (work.output / "train.log").write_text("partial checkpoint saved", encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No incomplete training restart"))
    with pytest.raises(ValueError, match="incomplete"):
        work.command("train", ["scripts/train_yolo_semantic.py"], product, "train", gpu=True)
    assert product.exists() and (work.output / "train.log").exists()


def test_optional_training_controls_default_preserves_request_and_flags(tmp_path):
    work = fixture(tmp_path)
    assert runner.optional_training_controls(work.args) == ({}, [])


def test_capped_training_controls_are_recorded_and_forwardable(tmp_path):
    work = fixture(tmp_path)
    work.args.batch = 4
    work.args.minimum_free_gib = 7
    work.args.maximum_utilization = 80
    work.args.gpu_memory_gib = 5.0
    work.args.nominal_batch = 32
    settings, flags = runner.optional_training_controls(work.args)
    assert settings == {"gpu_memory_gib": 5.0, "nominal_batch": 32}
    assert flags == ["--gpu-memory-gib", "5.0", "--nominal-batch", "32"]


@pytest.mark.parametrize("cap,minimum,nominal,maximum", [(5,5,32,20),(float("nan"),7,32,20),(5,7,2,20),(None,7,32,80),(5,7,32,91)])
def test_invalid_training_controls_rejected(tmp_path, cap, minimum, nominal, maximum):
    work = fixture(tmp_path)
    work.args.batch = 4
    work.args.gpu_memory_gib, work.args.minimum_free_gib = cap, minimum
    work.args.nominal_batch, work.args.maximum_utilization = nominal, maximum
    with pytest.raises(ValueError):
        runner.optional_training_controls(work.args)


def test_evaluation_gate_does_not_inherit_relaxed_training_policy(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.args.minimum_free_gib = 7
    work.args.maximum_utilization = 80
    samples = iter([(8000,30),(13000,30),(13000,5),(13000,5),(13000,5),(13000,5)])
    observed = []
    def sample(device):
        free, util = next(samples)
        observed.append((free, util))
        return {"free_mib":free,"utilization_percent":util}
    monkeypatch.setattr(runner, "gpu_sample", sample)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    reservations = []
    monkeypatch.setattr(work, "reserve_queue", lambda: reservations.append(len(observed)) or True)
    work.wait_and_reserve("evaluation")
    assert reservations == [5]
    assert work.state["phase_reservations"][-1]["minimum_free_gib"] == 12
    assert work.state["phase_reservations"][-1]["maximum_utilization_percent"] == 10
    assert work.state["waiting_phase"] == "evaluation"


def test_phase_change_restores_queue_before_new_capacity_wait(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.gpu_reserved, work.gpu_phase = True, "training"
    events = []
    monkeypatch.setattr(work, "release_queue", lambda: events.append("release"))
    monkeypatch.setattr(work, "wait_and_reserve", lambda phase: events.append("wait:"+phase))
    work.ensure_gpu_phase("evaluation")
    assert events == ["release", "wait:evaluation"]


def test_every_benchmark_rechecks_idle_gate(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.gpu_reserved, work.gpu_phase = True, "benchmark"
    events = []
    monkeypatch.setattr(work, "release_queue", lambda: events.append("release"))
    def wait(phase):
        events.append("wait:"+phase)
        work.gpu_reserved, work.gpu_phase = True, phase
    monkeypatch.setattr(work, "wait_and_reserve", wait)
    work.ensure_gpu_phase("benchmark")
    work.ensure_gpu_phase("benchmark")
    assert events == ["release","wait:benchmark","release","wait:benchmark"]


def test_training_command_receives_cap_and_nominal_batch(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    work.args.batch, work.args.minimum_free_gib, work.args.maximum_utilization = 4, 7, 80
    work.args.gpu_memory_gib, work.args.nominal_batch = 5, 32
    work.paths = {"weights":tmp_path / "initial.pt"}
    captured = []
    def capture(name, arguments, product, kind, **kwargs):
        captured.extend(arguments)
        raise RuntimeError("captured training arguments only")
    monkeypatch.setattr(work, "command", capture)
    with pytest.raises(RuntimeError, match="captured training arguments"):
        work.run_gpu_work()
    assert captured[captured.index("--batch")+1] == 4
    assert captured[captured.index("--gpu-memory-gib")+1] == "5"
    assert captured[captured.index("--nominal-batch")+1] == "32"


def test_completed_training_is_visible_while_evaluation_waits(tmp_path, monkeypatch):
    work = fixture(tmp_path)
    product = work.output / "training_report.json"
    save(product, {"training_complete":True})
    monkeypatch.setattr(runner, "validate_product", lambda path, kind: {})
    work.record_product("train", product, "train", ["train"])
    work.update("waiting_for_gpu", waiting_phase="evaluation", waiting_reason="evaluation idle gate")
    value = runner.read(work.followup_path)
    assert value["training_complete"] is True and value["training_started"] is True
    assert value["next_action"] == "evaluation idle gate"
