"""Run the existing driving pipeline without a display, dry-run by default.

SIGTERM/SIGINT becomes one KeyboardInterrupt, allowing run_drive's existing
finally path to send STOP and close the transport. No service is installed here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_now():
    return datetime.now(timezone.utc)


def _load_and_check(config_path, actuate):
    from camera_rc.driving.config import check_actuation, load_runtime_config
    config = load_runtime_config(config_path)
    # Never silently rewrite a serial config into dry-run or enable a configured MCU.
    check_actuation(config, actuate)
    return config


def _run_drive(config, output, **kwargs):
    from camera_rc.driving.runtime import run_drive
    return run_drive(config, output, **kwargs)


def unique_output_path(output_root):
    """Return a new name; run_drive creates it exclusively, preventing overwrite."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        candidate = root / f"{_utc_now():%Y%m%dT%H%M%S.%fZ}-{uuid.uuid4().hex}"
        if not candidate.exists():
            return candidate
    raise FileExistsError("Could not choose a unique run directory")


@contextmanager
def graceful_stop_signals():
    requested = {"signal": None}
    previous = {}

    def request_stop(signum, frame):
        if requested["signal"] is None:
            requested["signal"] = int(signum)
            raise KeyboardInterrupt("Headless shutdown requested")
        # A repeated stop request must not interrupt the runtime's STOP/close path.

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        yield requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def launch(config, output_root, *, actuate=False, duration_s=None, max_frames=None):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise PermissionError("Use an ordinary account; do not run driving with sudo")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration_s must be finite and positive")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    config = Path(config).expanduser().resolve(strict=True)
    _load_and_check(config, actuate)
    output = unique_output_path(output_root)
    started = _utc_now().isoformat()
    with graceful_stop_signals() as requested:
        try:
            result = _run_drive(config, output, actuate=actuate,
                                duration_s=duration_s, max_frames=max_frames)
        except KeyboardInterrupt:
            # Runtime catches interrupts after it starts its control/transport
            # lifecycle. An uncaught interrupt here means startup/return boundary.
            result = {"success": False, "termination": "interrupted_outside_runtime_loop",
                      "error": "Interrupted before the driving runtime returned a summary"}
        except Exception as exc:
            result = {"success": False, "termination": "launcher_error",
                      "error": f"{type(exc).__name__}: {exc}"}
    report = {"success": bool(result.get("success")), "config": str(config),
              "output": str(output), "actuate": bool(actuate),
              "started_at_utc": started, "finished_at_utc": _utc_now().isoformat(),
              "termination_signal": requested["signal"], "runtime": result,
              "note": "Dry-run does not validate physical driving. SIGKILL/power loss requires the independent MCU watchdog."}
    if output.is_dir():
        with (output / "launcher.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "headless")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="actuate", action="store_false", default=False,
                      help="Require a dry_run config (default; no actuator output)")
    mode.add_argument("--actuate", dest="actuate", action="store_true",
                      help="Manual use only: requires the existing calibrated serial/physical-arm safeguards")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    try:
        report = launch(args.config, args.output_root, actuate=args.actuate,
                        duration_s=args.duration_s, max_frames=args.max_frames)
    except Exception as exc:
        report = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
