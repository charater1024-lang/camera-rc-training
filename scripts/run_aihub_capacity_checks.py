"""One-time bounded diagnostics; reserve only the verified idle project queue."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
from run_aihub_followup import Runner, process_identity, now, write


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "runs/aihub_followup_v1/capacity_probe_v1"
    if output.exists():
        raise RuntimeError("Keep previous diagnostic evidence; inspect it before another attempt")
    runner = Runner(SimpleNamespace(project=project, output=output, poll_seconds=5))
    runner.acquire()
    results = []
    try:
        for batch in (4, 8):
            if not runner.reserve_queue():
                raise RuntimeError("Project queue is busy; no diagnostic launched")
            child_output = output / f"batch{batch}"
            command = [sys.executable, "-X", "utf8", "-B", "-u", "scripts/probe_aihub_capacity.py",
                       "--dataset", "data/yolo26_aihub_semantic_v1", "--config", "data/aihub_replay_cohort_v1/config.json",
                       "--init-training-report", "runs/yolo26x_sem_experiment_v2/training/training_report.json",
                       "--output", str(child_output), "--batch", str(batch), "--steps", "4",
                       "--maximum-utilization", "80", "--headroom-gib", "2", "--extra-double-validation"]
            runner.update("diagnostic_launching", step=f"batch{batch}", pending_launch={
                "command": command, "started_epoch": __import__('time').time(), "parent_pid": psutil.Process().pid})
            with (output / f"batch{batch}.log").open("x", encoding="utf-8") as log:
                runner.child = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
                runner.update("diagnostic_running", child=process_identity(psutil.Process(runner.child.pid)), pending_launch=None)
                code = runner.child.wait()
            runner.child = None
            runner.update("diagnostic_child_exited", child=None, exit_code=code)
            runner.release_queue()
            results.append({"batch": batch, "exit_code": code, "report": str(child_output / "capacity_report.json")})
            if code != 0:
                raise RuntimeError(f"Batch {batch} diagnostic failed; preserve evidence and inspect before proceeding")
        write(output / "checks_complete.json", {"finished_utc": now(), "diagnostic_only": True,
              "training_complete": False, "results": results, "other_applications_stopped": False})
        runner.update("diagnostics_complete")
    except BaseException as error:
        runner.update("diagnostic_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if runner.child is not None:
            runner.child.wait()
            runner.child = None
            runner.update("diagnostic_child_exited", child=None, pending_launch=None)
        runner.release_queue()
        runner.release_lock()


if __name__ == "__main__":
    main()
