"""Durable, bounded AI Hub fine-tuning/evaluation after verified CPU preparation.

GPU use waits for available capacity; unrelated applications are never stopped.
Only the identified idle upstream training queue is temporarily suspended.
Completed products are reused by hash. An incomplete training run is preserved
for explicit recovery, never automatically restarted or called a completed run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from train_yolo_semantic import identity, verify_export, verify_initialization


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for attempt in range(4):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.1 * (attempt + 1))


def process_identity(process):
    return {"pid": process.pid, "created": process.create_time(), "command": process.cmdline()}


def matching_process(record):
    if not record:
        return None
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() != record["created"]:
            return None
        if process.cmdline() != record["command"]:
            raise RuntimeError("Live PID command differs from recorded identity")
        return process
    except psutil.NoSuchProcess:
        return None
    # AccessDenied must propagate: invisibility never proves a process exited.


def pending_launch_processes(launch):
    """Recover the Popen-to-child-journal gap after an orchestrator hard exit."""
    if not launch:
        return []
    matches = []
    for process in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            if ("python" in (process.info["name"] or "").lower()
                    and process.info["create_time"] >= launch["started_epoch"] - 2
                    and process.cmdline() == launch["command"]):
                matches.append(process)
        except psutil.NoSuchProcess:
            continue
    return matches


def gpu_sample(index=0):
    completed = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free,utilization.gpu,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Expected one identified GPU in nvidia-smi output")
    values = [float(value.strip()) for value in lines[0].split(",")]
    if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("Invalid GPU capacity/utilization reading")
    return {"free_mib": values[0], "utilization_percent": values[1], "total_mib": values[2], "at": now()}


def gpu_is_available(sample, minimum_free_gib, maximum_utilization):
    return sample["free_mib"] >= minimum_free_gib * 1024 and sample["utilization_percent"] <= maximum_utilization


def optional_training_controls(args):
    """Keep the default request/CLI unchanged unless new controls are selected."""
    settings, flags = {}, []
    cap = getattr(args, "gpu_memory_gib", None)
    nominal = getattr(args, "nominal_batch", None)
    maximum = 90 if cap is not None else 20
    if not math.isfinite(args.maximum_utilization) or not 0 <= args.maximum_utilization <= maximum:
        raise ValueError("Training utilization above 20% requires an explicit allocator cap and must not exceed 90%")
    if cap is not None:
        if not math.isfinite(cap) or cap <= 0 or cap + 1 > args.minimum_free_gib:
            raise ValueError("Training allocator cap requires at least 1GiB additional start-gate headroom")
        settings["gpu_memory_gib"] = cap
        flags += ["--gpu-memory-gib", str(cap)]
    if nominal is not None:
        if type(nominal) is not int or nominal < args.batch:
            raise ValueError("Nominal batch must be an integer at least the physical batch")
        settings["nominal_batch"] = nominal
        flags += ["--nominal-batch", str(nominal)]
    return settings, flags


def offline_result(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not suites or totals["tests"] < 1 or totals["failures"] or totals["errors"]:
        raise ValueError("Offline control/sensor fixture tests did not pass")
    return totals


def checked_contract_identity(path):
    entry = identity(path)
    contract = read(path)
    marker = read(Path(path).parent / ".complete")
    if (marker.get("contract_sha256") != entry["sha256"] or contract.get("complete") is not True
            or contract.get("protocol_id") != "frozen_road_pixels_v1"
            or set(contract.get("public_tracks", {})) != {"obstacle", "anomaly"}):
        raise ValueError("A completed independent evaluation contract including both public regression tracks is required")
    return entry


def validate_product(path, kind):
    path = Path(path)
    if kind == "offline":
        return offline_result(path)
    report = read(path)
    if kind == "train":
        if report.get("training_complete") is not True or report.get("diagnostic") is not False:
            raise ValueError("Existing training is incomplete; inspect logs/checkpoints before recovery")
        if report.get("actual_args", {}).get("epochs") != 3:
            raise ValueError("AI Hub first stage must contain exactly three completed epochs")
        verify_initialization(report["checkpoints"]["best"]["path"], path)
    elif kind == "evaluation":
        if (report.get("evaluation_complete") is not True or report.get("checkpoint_diagnostic") is not False or not report.get("evaluation_contract")
                or not report.get("native_sources") or report.get("error")
                or set(report.get("public_obstacle_anomaly_subsets", {})) != {"obstacle", "anomaly"}
                or any(item.get("partial_validation") is not False for item in report["native_sources"].values())):
            raise ValueError("Existing frozen evaluation is incomplete")
    elif kind == "benchmark":
        if report.get("status") != "completed":
            raise ValueError("Existing runtime benchmark is incomplete")
    else:
        raise ValueError(f"Unknown product kind {kind}")
    return report


class Runner:
    def __init__(self, args):
        self.args = args
        self.project = args.project.resolve(strict=True)
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "gpu_runner.json"
        self.lock_path = self.output / ".gpu_runner.lock"
        self.followup_path = self.output / "status.json"
        self.child = None
        self.owns_lock = False
        self.gpu_reserved = False
        self.gpu_phase = None
        self.current_process = process_identity(psutil.Process())
        self.state = read(self.state_path) if self.state_path.is_file() else {
            "schema_version": 1, "created_utc": now(), "completed_steps": {},
            "deployment_approved": False, "suspended_queue": None, "child": None}

    def update(self, status, **fields):
        self.state.update(status=status, updated_utc=now(), runner=self.current_process, **fields)
        write(self.state_path, self.state)
        followup = read(self.followup_path) if self.followup_path.is_file() else {"schema_version": 1}
        followup.update(updated_utc=now(), gpu_runner={
            "path": str(self.state_path), "pid": self.current_process["pid"],
            "created": self.current_process["created"], "status": status,
            "step": self.state.get("step"), "updated_utc": self.state["updated_utc"]},
            deployment_approved=False)
        if self.state.get("training_complete"):
            followup.update(training_started=True, training_complete=True, training_report=self.state["training_report"])
        if status == "waiting_for_gpu":
            followup.update(status="waiting_for_available_GPU", next_action=self.state.get("waiting_reason",
                            "GPU capacity is occupied; existing applications remain running."))
        elif status in {"failed", "recovery_required", "gpu_wait_timeout"}:
            followup.update(status=status, next_action=self.state.get("error", status))
        elif status == "complete":
            followup.update(status="gpu_evaluation_complete_report_ready", training_complete=True)
        elif status == "running" and self.state.get("step") == "train":
            followup.update(status="additional_GPU_training", training_started=True)
        write(self.followup_path, followup)
        print(json.dumps({"status": status, **fields}, ensure_ascii=False), flush=True)

    def acquire(self):
        if self.lock_path.exists():
            previous = read(self.lock_path)
            if matching_process(previous):
                raise RuntimeError("An AI Hub GPU runner already owns this output")
            archived = self.lock_path.with_name(self.lock_path.name + ".stale." + str(time.time_ns()))
            self.lock_path.rename(archived)
        with self.lock_path.open("x", encoding="utf-8") as stream:
            json.dump(self.current_process, stream, ensure_ascii=False)
        self.owns_lock = True

    def release_lock(self):
        if self.owns_lock and self.lock_path.exists() and read(self.lock_path) == self.current_process:
            self.lock_path.unlink()
        self.owns_lock = False

    def release_queue(self):
        record = self.state.get("suspended_queue")
        if not record or record.get("resumed"):
            return
        if ((self.child is not None and self.child.poll() is None)
                or matching_process(self.state.get("child"))
                or pending_launch_processes(self.state.get("pending_launch"))):
            raise RuntimeError("Cannot release queue while the recorded child is still alive")
        queue = matching_process(record)
        if queue:
            queue.resume()
        record.update(resumed=True, resumed_utc=now(), queue_had_exited=queue is None)
        self.gpu_reserved = False
        self.gpu_phase = None
        self.update(self.state.get("status", "starting"), queue_temporarily_suspended=False)

    def recover_child(self):
        record = self.state.get("child")
        while matching_process(record) or pending_launch_processes(self.state.get("pending_launch")):
            self.update("waiting_for_existing_child", step=self.state.get("step"), existing_child=record)
            time.sleep(self.args.poll_seconds)
        self.state["child"] = None
        self.state["pending_launch"] = None
        self.release_queue()

    def prepare_request(self):
        dataset, config = verify_export(self.args.dataset, self.args.config)
        parent = read(self.args.init_training_report)
        weights = parent["checkpoints"]["best"]["path"]
        verified_weights, initialization = verify_initialization(weights, self.args.init_training_report)
        contract = checked_contract_identity(self.args.evaluation_contract)
        old_config = parent["source_config"]["path"]
        self.paths = {"weights": weights, "old_config": old_config,
                      "tiny_checkpoint": self.project / "runs/large_vkitti_stage1_v1/best.pt",
                      "tiny_report": self.project / "runs/large_vkitti_stage1_v1/summary.json",
                      "old_export": parent["dataset"]["provenance_path"]}
        required_scripts = ["train_yolo_semantic.py", "evaluate_frozen_road.py", "evaluation_contract.py",
                            "compare_frozen_road.py", "summarize_yolo_semantic_experiment.py",
                            "benchmark_yolo_semantic.py", "run_aihub_followup.py"]
        request = {"dataset": dataset, "config": config, "evaluation_contract": contract,
                   "initializer": verified_weights, "initialization": initialization,
                   "tiny_checkpoint": identity(self.paths["tiny_checkpoint"]),
                   "tiny_report": identity(self.paths["tiny_report"]),
                   "benchmark_images_manifest": identity(self.project / "data/public_v2/road/bdd_val/manifest.jsonl"),
                   "scripts": [identity(self.project / "scripts" / name) for name in required_scripts],
                   "training": {"epochs": 3, "image_size": 384, "batch": self.args.batch,
                                "workers": self.args.workers, "lr": self.args.lr, "device": self.args.device},
                   "runtime": {"iterations": 50, "warmup": 10, "precision": "fp16",
                               "inputs": [[192, 320], [384, 640]], "views": [1, 8]}}
        training_options, _ = optional_training_controls(self.args)
        request["training"].update(training_options)
        if training_options:
            request["phase_capacity_policy"] = {
                "training": {"minimum_free_gib": self.args.minimum_free_gib, "maximum_utilization_percent": self.args.maximum_utilization},
                "evaluation": {"minimum_free_gib": 12, "maximum_utilization_percent": 10},
                "each_runtime_benchmark": {"minimum_free_gib": 12, "maximum_utilization_percent": 10},
                "ready_polls": self.args.ready_polls,
                "isolation": "Idle-start samples and own queue coordination only; other applications are not stopped or GPU-isolated."}
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if self.state.get("request_sha256") not in (None, fingerprint):
            raise RuntimeError("Runner inputs/code changed; keep this output and use a new reviewed run")
        self.update("inputs_verified", request=request, request_sha256=fingerprint)
        training = self.output / "training/training_report.json"
        if training.exists():
            validate_product(training, "train")
        elif (self.output / "training").exists() or (self.output / "train.log").exists():
            raise RuntimeError("Prior training has no complete report; inspect its preserved checkpoint/log before recovery")

    def reserve_queue(self):
        selected = None
        for directory, script, allowed in (
            ("large_data_queue_v1", "run_large_data_queue.py", "waiting_for_verified_downloads"),
            ("camera_role_queue_v1", "run_camera_role_queue.py", "waiting_for_upstream_queue"),
            ("joint_policy_queue_v1", "run_joint_policy_queue.py", "waiting_for_role_queue")):
            path = self.project / "runs" / directory / "status.json"
            status = read(path)
            if status.get("status") in {"complete", "failed", "stopped"}:
                if type(status.get("pid")) is not int or psutil.pid_exists(status["pid"]):
                    return False
                continue
            if status.get("status") != allowed:
                return False
            try:
                process = psutil.Process(status["pid"])
                record = process_identity(process)
            except psutil.NoSuchProcess:
                raise RuntimeError("Waiting queue status is stale; investigate before training")
            if (f"scripts/{script}" not in " ".join(record["command"]).replace("\\", "/")
                    or process.children(recursive=True) or process.status() == psutil.STATUS_STOPPED):
                return False
            if selected is None:
                selected = (path, allowed, process, record)
        if selected is None:
            self.gpu_reserved = True
            self.update("GPU_reserved", queue_temporarily_suspended=False, all_known_queues_exited=True)
            return True
        path, allowed, queue, record = selected
        # Journal before suspend so a hard interruption can recover the identity.
        self.state["suspended_queue"] = {**record, "resumed": False}
        self.update("reserving_idle_queue")
        queue.suspend()
        after = read(path)
        if after.get("status") != allowed or queue.children(recursive=True):
            self.release_queue()
            return False
        self.update("GPU_reserved", queue_temporarily_suspended=True)
        self.gpu_reserved = True
        return True

    def wait_and_reserve(self, phase="training"):
        if phase not in {"training", "evaluation", "benchmark"}:
            raise ValueError("Unknown GPU work phase")
        minimum = self.args.minimum_free_gib if phase == "training" else 12
        maximum = self.args.maximum_utilization if phase == "training" else 10
        reason = (f"{phase}: waiting for {minimum:g}GiB free and utilization <= {maximum:g}% "
                  f"for {self.args.ready_polls} consecutive samples; unrelated applications remain running.")
        started = time.monotonic()
        stable = 0
        while True:
            if (self.output / "STOP_REQUESTED").exists():
                raise InterruptedError("Stopped before starting another GPU command")
            if time.monotonic() - started > self.args.max_wait_hours * 3600:
                self.update("gpu_wait_timeout", error=f"GPU/queue did not become available within the bounded {phase} wait; completed products are preserved.")
                raise TimeoutError("Bounded GPU availability wait expired")
            sample = gpu_sample(self.args.device)
            stable = stable + 1 if gpu_is_available(sample, minimum, maximum) else 0
            self.update("waiting_for_gpu", gpu=sample, consecutive_ready_polls=stable,
                        wait_elapsed_seconds=time.monotonic() - started, waiting_phase=phase,
                        waiting_reason=reason, minimum_free_gib=minimum, maximum_utilization_percent=maximum)
            if stable >= self.args.ready_polls and self.reserve_queue():
                # Close the capacity-read/queue-reservation race.
                if gpu_is_available(gpu_sample(self.args.device), minimum, maximum):
                    self.gpu_phase = phase
                    self.state.setdefault("phase_reservations", []).append({"phase": phase, "at": now(),
                        "minimum_free_gib": minimum, "maximum_utilization_percent": maximum,
                        "ready_polls": self.args.ready_polls, "external_applications_stopped": False})
                    return
                self.release_queue()
                stable = 0
            time.sleep(self.args.poll_seconds)

    def ensure_gpu_phase(self, phase):
        # Restore an idle queue before waiting for a different capacity policy.
        # Every new timing command obtains fresh idle-start samples, including a
        # possible 7GiB retry; the reduced training gate never applies to timing.
        if not self.gpu_reserved or self.gpu_phase != phase or phase == "benchmark":
            self.release_queue()
            self.gpu_reserved = False
            self.gpu_phase = None
            self.wait_and_reserve(phase)

    def record_product(self, name, product, kind, command):
        validate_product(product, kind)
        if kind == "train":
            self.state["training_complete"] = True
            self.state["training_report"] = identity(product)
        self.state["completed_steps"][name] = {"product": identity(product), "kind": kind,
                                               "command": command, "completed_utc": now()}
        self.update("step_complete", step=name, child=None)

    def command(self, name, arguments, product, kind, *, gpu=False, allow_oom=False):
        product = Path(product)
        completed = self.state["completed_steps"].get(name)
        command = [sys.executable, "-X", "utf8", "-B", "-u", *map(str, arguments)]
        if completed:
            if completed["command"] != command or identity(product) != completed["product"]:
                raise RuntimeError(f"Completed step changed: {name}")
            validate_product(product, kind)
            return True
        logfile = self.output / f"{name}.log"
        if product.exists():
            attempted = self.state.get("failed_attempts", {}).get(name)
            recovering = self.state.get("step") == name and self.state.get("command") == command and logfile.exists()
            if not recovering and not (attempted and attempted["command"] == command and attempted["product"] == identity(product)):
                raise RuntimeError(f"Unjournaled product has no recorded matching child command: {name}")
            if allow_oom and read(product).get("status") != "completed":
                report = read(product)
                if any(item.get("out_of_memory") for item in report.get("models", {}).values()):
                    self.state.setdefault("failed_attempts", {})[name] = {"product": identity(product), "command": command, "reason": "allocator_OOM"}
                    return False
            # Recover a successful child after an interrupted orchestrator.
            self.record_product(name, product, kind, command)
            return True
        if logfile.exists() or (kind == "train" and product.parent.exists()):
            raise RuntimeError(f"Incomplete prior step {name}: preserve output and inspect checkpoints/logs; automatic restart is disabled")
        if (self.output / "STOP_REQUESTED").exists():
            raise InterruptedError("Stopped at a step boundary; existing products preserved")
        if gpu:
            self.ensure_gpu_phase({"train": "training", "evaluation": "evaluation", "benchmark": "benchmark"}[kind])
            if not self.gpu_reserved:
                raise RuntimeError("GPU step has no upstream queue reservation")
        env = dict(os.environ, PYTHONPATH=str(self.project / "src"), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
        self.update("running", step=name, command=command, log=str(logfile),
                    pending_launch={"command": command, "started_epoch": time.time(), "step": name})
        with logfile.open("x", encoding="utf-8") as stream:
            self.child = subprocess.Popen(command, cwd=self.project, env=env, stdout=stream, stderr=subprocess.STDOUT)
            # No taskkill/terminate: an interrupt waits for this owned child and
            # then releases the idle queue. Hard exits are handled by the journal.
            try:
                self.update("running", step=name, child=process_identity(psutil.Process(self.child.pid)))
                code = self.child.wait()
            except BaseException:
                # Even if persistence failed, never leave a started child outside
                # this wait and then resume an upstream GPU queue underneath it.
                while self.child.poll() is None:
                    time.sleep(self.args.poll_seconds)
                raise
            finally:
                if self.child.poll() is not None:
                    while pending_launch_processes(self.state.get("pending_launch")):
                        time.sleep(self.args.poll_seconds)
                    self.state["child"] = None
                    self.child = None
                    self.state["pending_launch"] = None
        if code:
            if allow_oom and product.exists():
                report = read(product)
                if any(item.get("out_of_memory") for item in report.get("models", {}).values()):
                    self.state.setdefault("failed_attempts", {})[name] = {"product": identity(product), "command": command, "reason": "allocator_OOM"}
                    self.update("benchmark_allocator_OOM", step=name, preserved_output=str(product))
                    return False
            raise RuntimeError(f"{name} exited {code}; inspect {logfile}")
        self.record_product(name, product, kind, command)
        return True

    def offline(self):
        product = self.output / "offline_regression.xml"
        if product.exists():
            totals = offline_result(product)
            self.state["offline_regression"] = {"report": identity(product), **totals,
                "scope": "CPU synthetic offline control/planner/sensor fixtures; not recorded vehicle validation"}
            self.update("offline_regression_verified")
            return
        names = ["tests/test_control.py", "tests/test_planner.py", "tests/test_multiview_planner.py",
                 "tests/test_eight_camera_runtime.py", "tests/test_runtime_contract.py"]
        self.command("offline_regression", ["-m", "pytest", *names, "-q", "-p", "no:cacheprovider",
                     "--junitxml", product, "--basetemp", self.output / "offline_test_tmp"], product, "offline")
        self.state["offline_regression"] = {"report": identity(product), **offline_result(product),
            "scope": "CPU synthetic offline control/planner/sensor fixtures; not recorded vehicle validation"}

    def run_gpu_work(self):
        training = self.output / "training/training_report.json"
        _, controls = optional_training_controls(self.args)
        self.command("train", ["scripts/train_yolo_semantic.py", "--dataset", self.args.dataset,
            "--config", self.args.config, "--weights", self.paths["weights"], "--init-training-report", self.args.init_training_report,
            "--output", training.parent, "--epochs", "3", "--image-size", "384", "--batch", self.args.batch,
            "--workers", self.args.workers, "--lr", self.args.lr, "--device", self.args.device, *controls], training, "train", gpu=True)
        candidate = read(training)["checkpoints"]["best"]["path"]
        evaluation = self.output / "evaluation"
        evaluation.mkdir(exist_ok=True)
        models = [("tinyunet", "tinyunet", self.paths["old_config"], self.paths["tiny_checkpoint"], self.paths["tiny_report"]),
                  ("yolo_v2", "yolo26x-sem", self.paths["old_config"], self.paths["weights"], self.args.init_training_report),
                  ("aihub", "yolo26x-sem", self.args.config, candidate, training)]
        for height, width in ((192, 320), (384, 640)):
            for name, family, config, checkpoint, report in models:
                step = f"evaluate_{name}_{height}x{width}"
                product = evaluation / (step + ".json")
                arguments = ["scripts/evaluate_frozen_road.py", "--evaluation-contract", self.args.evaluation_contract,
                    "--model-family", family, "--config", config, "--checkpoint", checkpoint, "--training-report", report,
                    "--output", product, "--input-size", height, width, "--device", "cuda", "--batch-size", "4"]
                if family == "tinyunet":
                    arguments += ["--training-content-export", self.paths["old_export"]]
                self.command(step, arguments, product, "evaluation", gpu=True)
        benchmarks = self.output / "benchmarks"
        benchmarks.mkdir(exist_ok=True)
        for name, checkpoint, report in (("yolo_v2", self.paths["weights"], self.args.init_training_report), ("aihub", candidate, training)):
            for height, width in ((192, 320), (384, 640)):
                for views in (1, 8):
                    for budget in (3, 7):
                        images_manifest = self.project / "data/public_v2/road/bdd_val/manifest.jsonl"
                        if identity(images_manifest) != self.state["request"]["benchmark_images_manifest"]:
                            raise RuntimeError("Benchmark input manifest changed after request freeze")
                        step = f"benchmark_{name}_{height}x{width}_views{views}_{budget}gib"
                        product = benchmarks / (step + ".json")
                        arguments = ["scripts/benchmark_yolo_semantic.py", "--yolo-checkpoint", checkpoint,
                            "--training-report", report, "--native-config", self.paths["old_config"],
                            "--native-checkpoint", self.paths["tiny_checkpoint"], "--images-manifest",
                            images_manifest, "--output", product,
                            "--iterations", "50", "--warmup", "10", "--input-size", height, width,
                            "--views", views, "--precision", "fp16", "--memory-budget-gib", budget, "--device", "cuda"]
                        if self.command(step, arguments, product, "benchmark", gpu=True, allow_oom=budget == 3):
                            break
        self.update("GPU_steps_complete")

    def summarize(self):
        from compare_frozen_road import compare_reports
        summary_path = self.output / "gpu_summary.json"
        korean_path = self.output / "gpu_summary.ko.md"
        if summary_path.exists() and korean_path.exists():
            previous = read(summary_path)
            if (previous.get("complete") is not True or previous.get("request") != self.state["request"]
                    or previous.get("completed_steps") != self.state["completed_steps"]):
                raise RuntimeError("Existing summary differs from completed products; preserve it for review")
            self.update("complete", summary=identity(summary_path), korean_report=str(korean_path))
            return
        comparisons = {}
        for height, width in ((192, 320), (384, 640)):
            candidate = read(self.output / f"evaluation/evaluate_aihub_{height}x{width}.json")
            for name in ("tinyunet", "yolo_v2"):
                baseline = read(self.output / f"evaluation/evaluate_{name}_{height}x{width}.json")
                comparisons[f"{name}_{height}x{width}"] = compare_reports(baseline, candidate)
        runtime_rows, runtime_inputs = [], {}
        for name, step in self.state["completed_steps"].items():
            if step["kind"] != "benchmark":
                continue
            if identity(step["product"]["path"]) != step["product"]:
                raise RuntimeError("Completed benchmark report changed before summarizing")
            benchmark = read(step["product"]["path"])
            condition = (*benchmark["input_size_hw"], benchmark["views"])
            if condition in runtime_inputs and runtime_inputs[condition] != benchmark["input_images"]:
                raise ValueError("Baseline and candidate runtime inputs differ")
            runtime_inputs[condition] = benchmark["input_images"]
            for model, entry in benchmark["models"].items():
                for mode in ("host_to_road_map", "batched_host_to_road_map"):
                    if mode in entry:
                        measured = entry[mode]
                        runtime_rows.append({"experiment": name, "model": model, "mode": mode,
                            "input_size_hw": benchmark["input_size_hw"], "views": benchmark["views"],
                            "allocator_cap_gib": benchmark["hardware"]["requested_allocator_budget_gib"],
                            **{key: measured[key] for key in ("p50_cycle_ms", "p95_cycle_ms", "p99_cycle_ms", "camera_work_cycles_per_second")},
                            "p95_latency_travel_at_80kmh_m": measured["p95_cycle_ms"] / 1000 * 80 / 3.6,
                            "cuda_memory": measured["cuda_memory"],
                            "sampled_process_rss_peak_gib": entry["resources"].get("process_rss_peak_gib_sampled")})
        report = {"schema_version": 1, "complete": True, "completed_utc": now(), "deployment_approved": False,
                  "request": self.state["request"], "training": read(self.output / "training/training_report.json"),
                  "paired_comparisons": comparisons, "completed_steps": self.state["completed_steps"],
                  "runtime_measured_cycles": runtime_rows,
                  "phase_reservations": self.state.get("phase_reservations", []),
                  "offline_regression": self.state["offline_regression"],
                  "limitations": ["80km/h perception/collision avoidance not validated: no synchronized speed, timestamps or calibration.",
                                  "3/7GiB CUDA allocator caps on RTX3080 do not emulate MSI4/8GB device speed or total memory.",
                                  "Public40 obstacle/anomaly frames are reused development regression tests, not blind validation.",
                                  "Recorded frames test one shared model; not synchronized multi-camera fusion or actuator latency.",
                                  "Capped training may share GPU compute with other applications and be slower; its elapsed time is not an isolated GPU performance measurement.",
                                  "Timing starts only after fresh 12GiB-free / <=10% utilization samples; no exclusive GPU isolation is claimed. Other applications can resume work later.",
                                  "Offline fixture passes and pixel metrics do not prove accident-free operation."]}
        write(self.output / "gpu_summary.json", report)
        lines = ["AI Hub 추가 학습 및 오프라인 검증", "", "실제 CUDA 추가 학습 3 epoch와 같은 검증셋에서의 모델 비교를 완료했습니다.",
                 "실차 제어 및 운용 모델 자동 교체는 수행하지 않았습니다.", "", "검증셋별 도로 IoU / 도로 재현율 / 비도로→도로 오인율 (기존→추가 학습)", ""]
        training_options = self.state["request"]["training"]
        if "gpu_memory_gib" in training_options or "nominal_batch" in training_options:
            lines[4:4] = [f"학습 물리 배치 {training_options['batch']}, nominal batch {training_options.get('nominal_batch', training_options['batch'])}, "
                          f"학습 할당기 제한 {training_options.get('gpu_memory_gib', '미설정')} GiB.",
                          "다른 GPU 작업과 함께 수행한 학습 시간은 단독 GPU 처리속도 측정이 아닙니다. 기울기 누적은 큰 물리 배치와 완전히 동일하지 않습니다.", ""]
        for pair, values in comparisons.items():
            lines += [pair, ""]
            before = values["baseline"]["native_argmax"]["per_source"]
            after = values["candidate"]["native_argmax"]["per_source"]
            for name, row in before.items():
                formatted = []
                for key in ("road_iou", "road_recall", "false_road_rate"):
                    a, b = row[key], after[name][key]
                    formatted.append("미측정" if a is None or b is None else f"{a:.2%} → {b:.2%}")
                regressions = [item["metric"] for item in values["regressions"] if item["scope"] == "native_argmax:" + name]
                lines.append(f"- {name}: {' / '.join(formatted)}; 악화 지표: {', '.join(regressions) or '없음'}")
            for name, row in values["baseline"]["public"].items():
                a, b = row["micro_obstacle_as_road_rate"], values["candidate"]["public"][name]["micro_obstacle_as_road_rate"]
                lines.append(f"- {name} 장애물→도로 오인율: {a:.2%} → {b:.2%}; 악화: {b > a}")
            lines.append("")
        lines += ["녹화 프레임 → 도로 확률맵 전체 처리 지연 (노출·USB 전송·제어기 지연 제외)", "",
                  "| 실험 / 모델 | 입력 / 영상 수 / 방식 | p50 / p95 / p99 (ms) | cycle/s | GPU 할당 / 예약 최대 (GiB) | RAM RSS 표본 최대 (GiB) |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in runtime_rows:
            memory = row["cuda_memory"]
            rss = row["sampled_process_rss_peak_gib"]
            rss_text = "미측정" if rss is None else f"{rss:.3f}"
            lines.append(f"| {row['experiment']} / {row['model']} | {row['input_size_hw']} / {row['views']} / {row['mode']} | "
                         f"{row['p50_cycle_ms']:.2f} / {row['p95_cycle_ms']:.2f} / {row['p99_cycle_ms']:.2f} | "
                         f"{row['camera_work_cycles_per_second']:.2f} | {memory['torch_peak_allocated_gib']:.3f} / "
                         f"{memory['torch_peak_reserved_gib']:.3f} | {rss_text} |")
        lines.append("")
        lines += ["p50/p95/p99 전체 처리 지연과 메모리 측정은 benchmarks/*.json, 학습 횟수·모델 해시는 gpu_summary.json에 보존했습니다.",
                  "각 시간 측정은 여유 12GiB·사용률 10% 이하를 연속 확인한 뒤 시작합니다. GPU를 독점하지 않으므로 외부 작업의 재개 가능성이 남습니다.",
                  "RTX 3080에서의 메모리 제한 실험은 실제 MSI 4/8GB 하드웨어 성능을 의미하지 않습니다.",
                  "80km/h 인지·충돌 방지 검증은 미완료입니다. 공개 장애물 40장은 재사용 개발 회귀 자료입니다.",
                  "계획·제어 테스트는 합성 오프라인 테스트이며 무사고 보장이 아닙니다."]
        (self.output / "gpu_summary.ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.update("complete", summary=identity(self.output / "gpu_summary.json"), korean_report=str(self.output / "gpu_summary.ko.md"))

    def run(self):
        self.acquire()
        try:
            self.recover_child()
            self.prepare_request()
            self.offline()
            try:
                self.run_gpu_work()
            finally:
                self.release_queue()
            self.summarize()
        except BaseException as error:
            status = "gpu_wait_timeout" if isinstance(error, TimeoutError) else "recovery_required"
            self.update(status, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                self.release_queue()
            finally:
                self.release_lock()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "dataset", "config", "evaluation-contract", "init-training-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, help="Optional training CUDA allocator cap; leaves contexts/other libraries outside the cap")
    parser.add_argument("--nominal-batch", type=int, help="Optional nominal training batch for gradient accumulation")
    parser.add_argument("--minimum-free-gib", type=float, default=12)
    parser.add_argument("--maximum-utilization", type=float, default=10)
    parser.add_argument("--ready-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=24)
    args = parser.parse_args()
    if (args.batch < 1 or args.workers < 0 or args.ready_polls < 2 or args.poll_seconds < 5
            or not 0 < args.lr < 1 or not 1 <= args.minimum_free_gib <= 16
            or not 0 < args.max_wait_hours <= 72 or args.device != 0):
        parser.error("Invalid training or bounded GPU wait options")
    try:
        optional_training_controls(args)
    except ValueError as error:
        parser.error(str(error))
    Runner(args).run()


if __name__ == "__main__":
    main()
