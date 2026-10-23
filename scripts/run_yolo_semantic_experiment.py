"""Run GPU training, paired accuracy evaluation and execution benchmarks.

The existing download-dependent training queue is suspended only while idle,
and always resumed on this process's normal/error exit. Downloads keep running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_yolo_semantic import write_json


class Experiment:
    def __init__(self, project, dataset, output):
        self.project, self.dataset, self.output = [Path(p).resolve() for p in (project, dataset, output)]
        self.output.mkdir(parents=True, exist_ok=False)
        self.state = {"pid": os.getpid(), "status": "starting", "completed_steps": [],
                      "started_utc": datetime.now(timezone.utc).isoformat(), "deployment_approved": False}
        self.queue = None
        self.child = None
        self.update("starting")

    def update(self, status, **items):
        self.state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **items)
        write_json(self.output / "status.json", self.state)
        print(json.dumps({"status": status, **items}, ensure_ascii=False), flush=True)

    def reserve_gpu_queue(self):
        import psutil
        path = self.project / "runs/large_data_queue_v1/status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["status"] != "waiting_for_verified_downloads":
            raise RuntimeError("Previous queue is not idle; refuse to interrupt active work")
        process = psutil.Process(status["pid"])
        if "scripts/run_large_data_queue.py" not in " ".join(process.cmdline()).replace("\\", "/") or process.children(recursive=True):
            raise RuntimeError("Queue process identity/idle state changed")
        self.state["suspended_queue"] = {"pid": process.pid, "created": process.create_time(), "resumed": False}
        process.suspend()
        self.queue = process
        # Close the status-read/suspend race without interrupting a child.
        after = json.loads(path.read_text(encoding="utf-8"))
        if after["status"] != "waiting_for_verified_downloads" or process.children(recursive=True):
            raise RuntimeError("Previous queue advanced during reservation; resume and retry later")
        self.update("waiting_for_prepared_data", queue_temporarily_suspended=True)

    def release_gpu_queue(self):
        if self.queue is not None:
            if self.child is not None and self.child.poll() is None:
                self.update("recovery_required", recovery_child_pid=self.child.pid,
                            recovery_queue_pid=self.queue.pid,
                            recovery_notice="Child termination was not confirmed; resume the original queue only after that child exits.")
                return
            self.queue.resume()
            self.state["suspended_queue"]["resumed"] = True
            self.state["suspended_queue"]["resumed_utc"] = datetime.now(timezone.utc).isoformat()
            self.queue = None
            self.update(self.state["status"], queue_temporarily_suspended=False)

    def wait_prepared(self):
        deadline = time.monotonic() + 6 * 3600
        while not (self.dataset / ".complete").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Data preparation did not finish within six hours")
            report = self.dataset / "report.json"
            if report.is_file():
                content = json.loads(report.read_text(encoding="utf-8"))
                if content.get("error"):
                    raise RuntimeError(f"Dataset export failed: {content['error']}")
            time.sleep(10)

    def command(self, name, arguments, *, allow_failure=False):
        logfile = self.output / f"{name}.log"
        command = [sys.executable, "-X", "utf8", "-B", "-u", *map(str, arguments)]
        self.update("running", step=name, command=command, log=str(logfile))
        env = dict(os.environ, PYTHONPATH=str(self.project / "src"),
                   YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
        with logfile.open("x", encoding="utf-8") as log:
            self.child = subprocess.Popen(command, cwd=self.project, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.update("running", child_pid=self.child.pid)
            returncode = self.child.wait()
            self.child = None
        if returncode and not allow_failure:
            raise RuntimeError(f"{name} exited {returncode}; see {logfile}")
        self.state["completed_steps"].append({"step": name, "exit_code": returncode})
        self.update("step_complete", step=name, child_pid=None)
        return returncode

    def run(self):
        self.reserve_gpu_queue()
        self.wait_prepared()
        weights = self.project / "models/pretrained/yolo26x-sem.pt"
        capacity = self.output / "training_capacity_batch32.json"
        self.command("capacity", ["scripts/profile_yolo_semantic_training.py", "--weights", weights,
                     "--output", capacity, "--image-size", "384", "--batches", "32"])
        result = json.loads(capacity.read_text(encoding="utf-8"))["results"][0]
        batch = 32 if (result["status"] == "passed" and result["amp_skipped_updates"] < 7
                       and result["peak_allocated_bytes"] < 12 * 1024**3) else 16
        config = self.project / "configs/local/large_vkitti_stage1_v1.json"
        training = self.output / "training"
        self.command("train", ["scripts/train_yolo_semantic.py", "--dataset", self.dataset, "--config", config,
                     "--weights", weights, "--output", training, "--epochs", "1", "--image-size", "384",
                     "--batch", str(batch), "--workers", "4"])
        training_report = training / "training_report.json"
        complete = json.loads(training_report.read_text(encoding="utf-8"))
        if complete.get("training_complete") is not True or complete.get("diagnostic") is not False:
            raise RuntimeError("Incomplete training cannot enter paired evaluation")
        checkpoint = complete["checkpoints"]["best"]["path"]
        evaluation = self.output / "evaluation"
        evaluation.mkdir()
        datasets = self.project.parent / "datasets"
        common_args = ["scripts/evaluate_yolo_semantic.py", "--config", config, "--checkpoint", checkpoint,
                       "--training-report", training_report, "--device", "cuda", "--batch-size", "8",
                       "--obstacle-root", datasets / "road_obstacle21/dataset_ObstacleTrack/dataset_ObstacleTrack",
                       "--anomaly-root", datasets / "road_anomaly21/dataset_AnomalyTrack/dataset_AnomalyTrack"]
        paired = evaluation / "paired_192x320.json"
        self.command("evaluate_paired", [*common_args, "--input-size", "192", "320", "--output", paired])
        from run_large_data_queue import compare_reports
        baseline_file = self.project / "runs/large_vkitti_stage1_v1/evaluation/candidate.json"
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        candidate = json.loads(paired.read_text(encoding="utf-8"))
        comparison = compare_reports(baseline, candidate)
        comparison["baseline_report"] = str(baseline_file)
        comparison["candidate_report"] = str(paired)
        write_json(evaluation / "comparison.json", comparison)
        # Higher resolution is a separately named input condition, not an equal-resolution architecture comparison.
        self.command("evaluate_384x640", [*common_args, "--input-size", "384", "640",
                     "--output", evaluation / "candidate_384x640.json"])
        native = self.project / "runs/large_vkitti_stage1_v1/best.pt"
        images = self.project / "data/public_v2/road/bdd_val/manifest.jsonl"
        benchmarks = self.output / "benchmarks"
        benchmarks.mkdir()
        for height, width in ((192, 320), (384, 640)):
            for views in (1, 8):
                for budget in (3, 7):
                    name = f"benchmark_{height}x{width}_views{views}_{budget}gib"
                    output = benchmarks / f"{name}.json"
                    self.command(name, ["scripts/benchmark_yolo_semantic.py", "--yolo-checkpoint", checkpoint,
                        "--training-report", training_report, "--native-config", config, "--native-checkpoint", native,
                        "--images-manifest", images, "--output", output, "--iterations", "50", "--warmup", "10",
                        "--input-size", str(height), str(width), "--views", str(views), "--precision", "fp16",
                        "--memory-budget-gib", str(budget), "--device", "cuda"], allow_failure=True)
                    report = json.loads(output.read_text(encoding="utf-8"))
                    if report["status"] == "completed":
                        break
                    if budget == 7 or not any(item.get("out_of_memory") for item in report.get("models", {}).values()):
                        raise RuntimeError(f"Execution benchmark failed: {output}")
        self.update("complete", completed_utc=datetime.now(timezone.utc).isoformat(), comparison=str(evaluation / "comparison.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    experiment = Experiment(**vars(parser.parse_args()))
    try:
        experiment.run()
    except BaseException as error:
        if experiment.child is not None and experiment.child.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(experiment.child.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                experiment.child.terminate()
            experiment.child.wait(timeout=30)
        experiment.update("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        experiment.release_gpu_queue()


if __name__ == "__main__":
    main()
