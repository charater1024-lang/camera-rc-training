"""Run ONE explicitly requested offline command, preserving failures and logs.

Never terminates or suspends another application or queue. Existing project
queues must have exited. No benchmark scheduling and no automatic retry.
STOP_REQUESTED prevents launch; during a running command it stops later work,
not the active command. Inspect its PID/checkpoint before manually stopping it.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

QUEUES = ("large_data_queue_v1", "camera_role_queue_v1", "joint_policy_queue_v1")
GPU_SCRIPTS = ("train_yolo_semantic.py", "run_aihub_followup.py", "run_yolo_semantic_experiment.py",
               "evaluate_selection_development.py", "evaluate_frozen_road.py", "evaluate_yolo_semantic.py",
               "train_joint_policy.py", "run_large_data_queue.py", "run_joint_policy_queue.py", "run_camera_role_queue.py")


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def process_record(process):
    return {"pid": process.pid, "created": process.create_time(), "command": process.cmdline()}


def queues_exited(project):
    proofs = []
    for name in QUEUES:
        path = project / "runs" / name / "status.json"
        if not path.exists():
            proofs.append({"queue": name, "state": "not_configured"})
            continue
        state = json.loads(path.read_text(encoding="utf-8-sig"))
        if state.get("status") not in ("stopped", "complete", "failed", "cancelled_by_user"):
            raise RuntimeError(f"Queue is not exited: {name}")
        pid = state.get("pid")
        if isinstance(pid, int):
            try:
                command = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/")
                if name.removesuffix("_v1") in command:
                    raise RuntimeError(f"Queue PID still active: {name}")
            except psutil.NoSuchProcess:
                pass
        proofs.append({"queue": name, "status": state["status"], "record": identity(path)})
    return proofs


def foreign_gpu_workers(project):
    ours = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    found = []
    project_key = str(project).replace("\\", "/").casefold()
    for p in psutil.process_iter(["pid", "cmdline", "name"]):
        if p.pid in ours or "python" not in (p.info["name"] or "").lower():
            continue
        command = " ".join(p.info["cmdline"] or []).replace("\\", "/")
        if any(script in command for script in GPU_SCRIPTS):
            # Include relative script invocations: conservatively wait rather
            # than assume another training command is safe to overlap.
            found.append({"pid": p.pid, "command": command,
                          "same_project_path_visible": project_key in command.casefold()})
    return found


def gpu_sample():
    run = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True, timeout=15)
    parts = [p.strip() for p in run.stdout.splitlines()[0].split(",")]
    return {"uuid": parts[0], "name": parts[1], "total_mib": float(parts[2]),
            "free_mib": float(parts[3]), "utilization_percent": float(parts[4])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--completion-key", choices=("evaluation_complete", "training_complete", "complete"), required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=7)
    parser.add_argument("--maximum-utilization", type=float, default=80)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.project = args.project.resolve(strict=True)
    args.state_dir = args.state_dir.resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or any("benchmark" in p.lower() for p in command):
        raise ValueError("One non-benchmark command required")
    if args.product.exists() or args.state_dir.exists():
        raise FileExistsError("Preserve prior step and inspect before any recovery")
    args.state_dir.mkdir(parents=True)
    state = {"runner": process_record(psutil.Process()), "command": command,
             "product": str(args.product.resolve()), "other_apps_modified": False,
             "speed_measurement": False, "automatic_retry": False,
             "queues_suspended": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    def update(status, **fields):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temp = args.state_dir / "state.tmp.json"
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(args.state_dir / "state.json")
        print(json.dumps({"status": status, **fields}, ensure_ascii=False), flush=True)
    lock = args.project / "runs" / ".guarded_offline_step.lock"
    child, owns_lock = None, False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(state["runner"], stream)
        owns_lock = True
        stable, started = 0, time.monotonic()
        while True:
            if (args.state_dir / "STOP_REQUESTED").exists() or (args.state_dir.parent / "STOP_REQUESTED").exists():
                raise InterruptedError("Stop requested before launch")
            if time.monotonic() - started > args.wait_hours * 3600:
                raise TimeoutError("Bounded GPU wait expired")
            queue_proofs = queues_exited(args.project)
            foreign = foreign_gpu_workers(args.project)
            gpu = gpu_sample()
            ready = not foreign and gpu["free_mib"] >= args.minimum_free_gib * 1024 and gpu["utilization_percent"] <= args.maximum_utilization
            stable = stable + 1 if ready else 0
            update("waiting_for_gpu", gpu=gpu, stable_samples=stable, foreign_gpu_workers=foreign, queues=queue_proofs)
            if stable >= 3:
                break
            time.sleep(30)
        # Check once more at the launch boundary; all exited queues stay exited.
        queues_exited(args.project)
        if foreign_gpu_workers(args.project):
            raise RuntimeError("Another GPU worker appeared at launch boundary")
        with (args.state_dir / "command.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=args.project, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            update("running", child=process_record(psutil.Process(child.pid)))
            while child.poll() is None:
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"Command exited {child.returncode}; no automatic retry")
        result = json.loads(args.product.read_text(encoding="utf-8-sig"))
        if result.get(args.completion_key) is not True:
            raise RuntimeError("Command did not produce its completed product")
        update("complete", completed_product=identity(args.product), exit_code=child.returncode)
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}",
               child_may_be_alive=child is not None and child.poll() is None)
        raise
    finally:
        if owns_lock and (child is None or child.poll() is not None):
            lock.unlink()


if __name__ == "__main__":
    main()
