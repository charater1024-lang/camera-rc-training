"""Read-only AI Hub progress UI on 127.0.0.1; no control or arbitrary-file API."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import re
import subprocess
import threading
import time

import psutil

ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def small_json(path):
    try:
        if path.stat().st_size > 2 * 1024**2:
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8000))
            return ANSI.sub("", stream.read(8000).decode("utf-8", errors="replace"))[-4000:]
    except OSError:
        return ""


def runner_liveness(record):
    if not record:
        return {"state": "unknown", "reason": "no_recorded_process"}
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() != record["created"] or process.cmdline() != record["command"]:
            return {"state": "exited", "reason": "pid_identity_changed"}
        return {"state": "alive", "reason": "pid_creation_and_command_match"}
    except psutil.NoSuchProcess:
        return {"state": "exited", "reason": "process_not_found"}
    except (psutil.AccessDenied, KeyError, TypeError):
        return {"state": "unknown", "reason": "identity_unavailable"}


class GPUCache:
    def __init__(self):
        self.updated, self.value, self.lock = -100, {}, threading.Lock()

    def read(self):
        with self.lock:
            if time.monotonic() - self.updated < 5:
                return self.value
            try:
                result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
                name, used, total, utilization = result.stdout.strip().split(",")
                self.value = {"available": True, "name": name.strip(), "used_gib": float(used)/1024,
                              "total_gib": float(total)/1024, "utilization": float(utilization)}
            except (OSError, ValueError, subprocess.SubprocessError):
                self.value = {"available": False}
            self.updated = time.monotonic()
            return self.value


def snapshot(project, gpu):
    base = project / "runs/aihub_followup_v1"
    run = base / "reduced_batch_v1"
    state = small_json(run / "gpu_runner.json")
    cancellation = small_json(run / "speed_cancellation_v1/confirmation.json")
    cancelled = (
        cancellation.get("status") == "cancelled_by_user"
        and cancellation.get("runner_exited") is True
        and cancellation.get("queue_restored") is True
        and state.get("status") == "recovery_required"
        and state.get("error") == "InterruptedError: Stopped before starting another GPU command"
        and cancellation.get("gpu_runner_exit_snapshot", {}).get("sha256")
        == hashlib.sha256((run / "gpu_runner.json").read_bytes()).hexdigest()
    )
    report = small_json(run / "training/training_report.json")
    live = small_json(run / "training/progress.json")
    counters = live.get("progress", live) if live else report.get("progress", {})
    actual = report.get("actual_args", {})
    request = state.get("request", {}).get("training", {})
    count = report.get("dataset", {}).get("counts", {})
    batches = live.get("batches_per_epoch", report.get("batches_per_epoch"))
    seen = counters.get("train_batches", report.get("progress", {}).get("train_batches", 0))
    epochs = live.get("requested_epochs", actual.get("epochs", request.get("epochs", 3)))
    current = live.get("current_epoch", report.get("current_epoch", 0))
    complete = report.get("training_complete") is True
    fraction = min(1, seen/(batches*epochs)) if batches and epochs else 0
    probes = []
    for batch in (4, 8):
        item = small_json(base / f"capacity_probe_v1/batch{batch}/capacity_report.json")
        probes.append({"batch": batch, "status": item.get("status", "not_started"),
                       "successful_updates": item.get("successful_updates", 0),
                       "attempts": item.get("optimizer_attempts", 0), "error": str(item.get("error", ""))[:500]})
    step = state.get("step")
    waiting = state.get("status") == "waiting_for_gpu"
    error_state = state.get("status") in {"recovery_required", "failed", "gpu_wait_timeout"}
    # Every readable path is fixed here; no client-selected filesystem paths.
    log = tail(run / "train.log")
    if not log:
        log = tail(base / "capacity_probe_v1/batch8.log") or tail(base / "capacity_probe_v1/batch4.log")
    try:
        stamp = max(p.stat().st_mtime for p in (run / "training/progress.json", run / "training/training_report.json") if p.is_file())
        age = max(0, time.time() - stamp)
    except (OSError, ValueError):
        age = None
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu.read(), "probes": probes,
        "runner": {"status": "benchmark_cancelled_by_user" if cancelled else state.get("status", "not_started"), "step": step,
                   "waiting_phase": state.get("waiting_phase") if waiting else None,
                   "reason": "" if cancelled else str(state.get("waiting_reason", "") if waiting else state.get("error", "") if error_state else "")[:1000],
                   "liveness": runner_liveness(state.get("runner"))},
        "training": {"started": bool(report or live), "complete": complete, "epoch": epochs if complete else current,
                     "epochs": epochs, "batches_seen": seen, "batches_per_epoch": batches,
                     "batch_in_epoch": live.get("batch_in_epoch", max(0, min(batches, seen-max(0,current-1)*batches)) if batches else None),
                     "fraction": 1 if complete else fraction, "optimizer_updates": counters.get("optimizer_updates", 0),
                     "optimizer_attempts": counters.get("optimizer_attempts", 0), "amp_skipped": counters.get("amp_skipped_updates", 0),
                     "physical_batch": actual.get("batch", request.get("batch")),
                     "nominal_batch": actual.get("nbs", request.get("nominal_batch")),
                     "train_images": count.get("train", 37864), "validation_images": count.get("val", 21489),
                     "age_seconds": age, "error": str(report.get("error", ""))[:1000]}, "log_tail": log}


def handler(project):
    gpu = GPUCache()
    page = (project / "assets/aihub_progress.html").read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self.send_error(403)
                return
            if self.path in ("/", "/index.html"):
                body, mime = page, "text/html; charset=utf-8"
            elif self.path == "/api/progress":
                body = json.dumps(snapshot(project, gpu), ensure_ascii=False, allow_nan=False).encode("utf-8")
                mime = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local unprivileged port")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.project.resolve(strict=True)))
    print(f"Read-only progress: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
