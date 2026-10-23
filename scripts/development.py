"""Offline development entry point. It never opens cameras or serial ports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"front", "left", "right", "rear"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def status(root=ROOT, *, prepared_data=None):
    root = Path(root).resolve()
    usage = shutil.disk_usage(root)
    prepared = Path(prepared_data).expanduser() if prepared_data is not None else Path("data/public_v1")
    prepared = (prepared if prepared.is_absolute() else root / prepared).resolve()
    curriculum = prepared / "curriculum.json"
    preparation_report = prepared / "report.json"
    incomplete = (prepared / ".incomplete").exists()
    reported_complete, report_error, preparation_error = None, None, None
    if preparation_report.is_file():
        try:
            report = read_json(preparation_report)
            if type(report.get("complete")) is not bool:
                raise ValueError("Preparation report requires a boolean complete field")
            reported_complete = report["complete"]
            preparation_error = report.get("error")
        except (OSError, ValueError) as exc:
            report_error = f"{type(exc).__name__}: {exc}"
    if incomplete or reported_complete is False:
        preparation_state = "incomplete"
        readiness = "finish_dataset_preparation_first"
    elif report_error is not None:
        preparation_state = "invalid_report"
        readiness = "inspect_preparation_report"
    elif not prepared.is_dir():
        preparation_state = "missing"
        readiness = "prepare_real_datasets_first"
    elif reported_complete is True:
        preparation_state = "reported_complete_pending_preflight" if curriculum.is_file() else "report_complete_missing_curriculum"
        readiness = "run_curriculum_preflight" if curriculum.is_file() else "prepare_real_datasets_first"
    else:
        preparation_state = "completion_not_reported"
        # Preserve legacy folders' next step without claiming they are ready.
        readiness = "run_curriculum_preflight" if curriculum.is_file() else "prepare_real_datasets_first"
    return {
        "project": str(root),
        "python": sys.executable,
        "disk_free_gib": round(usage.free / 2**30, 2),
        "development_target": "Windows / RTX 3080 Laptop 16GB / RAM 32GB",
        "future_runtime_target": "MSI / Ubuntu 24.04 / dedicated GPU 4GB / RAM 16GB assumed",
        "prepared_data": str(prepared),
        "prepared_data_exists": prepared.is_dir(),
        "public_curriculum": str(curriculum),
        "public_curriculum_exists": curriculum.is_file(),
        "preparation_report": str(preparation_report),
        "preparation_report_exists": preparation_report.is_file(),
        "preparation_report_complete": reported_complete,
        "preparation_report_read_error": report_error,
        "preparation_report_error": preparation_error,
        "preparation_incomplete_marker_exists": incomplete,
        "preparation_state": preparation_state,
        "training_readiness": readiness,
        "runtime_checkpoint_exists": (root / "runs/deployment/road_surface.pt").is_file(),
        "real_model_validation": "not_assessed",
        "camera_opened": False,
        "serial_port_opened": False,
        "note": "File existence or a report's complete flag is not fresh model/data validation. Run curriculum preflight for the selected data; use verify for offline synthetic checks.",
    }


def validate_environment(report, device):
    require(report.get("success") is True, "Environment compatibility check failed")
    if device == "cuda":
        require(report.get("gpu_execution_verified") is True, "CUDA execution was not verified")
    require(report.get("camera_opened") is False and report.get("serial_port_opened") is False,
            "Environment check unexpectedly used hardware input/output")


def validate_training(report):
    require(report.get("diagnostic_only") is True and report.get("real_driving_validated") is False,
            "Expected synthetic-only training provenance")
    require(report.get("training_complete") is True and report.get("stages") == 2,
            "Both diagnostic training stages must finish")
    require(report.get("weights_transferred") is True and report.get("global_holdouts_checked") is True,
            "Stage initialization or global holdout check failed")
    require(set(report.get("camera_holdouts_evaluated", [])) == CAMERAS,
            "All four camera holdouts must be evaluated")
    require(report.get("resume_skipped_completed_stages") is True, "Completed-stage resume failed")
    steps = report.get("optimizer_steps_by_stage", {})
    require(set(steps) == {"front_only", "four_views"}, "Missing actual optimizer step counters")
    require(all(type(item.get("applied")) is int and item["applied"] > 0 for item in steps.values()),
            "A training stage applied no optimizer updates")


def validate_fixture(path):
    report = read_json(path)
    require(report.get("source", {}).get("kind") == "multireplay", "Only generated multireplay input is allowed")
    require(report.get("transport", {}).get("kind", "dry_run") == "dry_run", "Only dry_run transport is allowed")
    require(report.get("perception", {}).get("allow_diagnostic") is True, "Expected diagnostic fixture")
    return report


def validate_benchmark(report, frames):
    require(report.get("diagnostic_only") is True and report.get("model_diagnostic") is True,
            "Expected diagnostic benchmark")
    require(report.get("physical_output") is False, "Benchmark reported physical output")
    require(report.get("shared_model_instances") == 1 and set(report.get("camera_names", [])) == CAMERAS,
            "Benchmark must use four views and one shared model")
    require(report.get("pipeline_latency", {}).get("frames") == frames, "Incomplete benchmark")
    require(report.get("valid_plan_frames") == frames, "Diagnostic fixture produced invalid paths")
    for flag in ("ram_process_within_limit", "system_available_floor_met_sampled",
                 "processing_deadline_p99_met_on_this_device", "group_period_p99_met_on_this_device"):
        require(report.get(flag) is True, f"Benchmark did not meet configured target: {flag}")
    # Whole-device VRAM includes other applications; it is reported, not treated
    # as this process's allocation or proof of operation on the future MSI.


def validate_replay(launcher, runtime, frames):
    require(launcher.get("success") is True and launcher.get("actuate") is False, "Headless replay failed")
    require(launcher.get("runtime") == runtime, "Launcher/runtime summaries disagree")
    require(runtime.get("success") is True and runtime.get("termination") == "frame_limit_reached",
            "Replay did not finish at its requested frame limit")
    require(runtime.get("frames") == frames and runtime.get("physical_output") is False,
            "Incomplete replay or physical output reported")
    require(runtime.get("shutdown_stop_sent") is True and runtime.get("latched_fault") is None,
            "Replay must close with STOP and without a latched fault")
    require(set(runtime.get("camera_names", [])) == CAMERAS and runtime.get("shared_model_instances") == 1,
            "Unexpected replay camera/model configuration")


def run_step(name, args, output, *, root=ROOT, runner=subprocess.run):
    log = output / f"{name}.log"
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(Path(root) / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    temporary = output / "temporary"
    temporary.mkdir(exist_ok=True)
    environment["TEMP"] = environment["TMP"] = str(temporary)
    command = [sys.executable, *map(str, args)]
    print(f"{name}: running (log: {log})", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        result = runner(command, cwd=root, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with {result.returncode}; see {log}")
    return {"name": name, "command": command, "exit_code": result.returncode, "log": str(log)}


def verify(output=None, *, device="cuda", with_tests=False, root=ROOT):
    root = Path(root).resolve()
    require(device in {"cuda", "cpu"}, "Choose cuda or cpu explicitly; no silent fallback")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / "runs/development" / f"{stamp}-{uuid.uuid4().hex[:12]}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("Offline verification is incomplete.\n", encoding="utf-8")
    report = {
        "success": False, "diagnostic_only": True, "device": device, "output": str(output),
        "project": str(root), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_opened": False, "serial_port_opened": False, "physical_output": False,
        "real_dataset_training_performed": False, "target_msi_validated": False,
        "training_weights_used_for_runtime": False,
        "stages": [], "error": None,
        "note": "Synthetic training and known-output runtime fixture are separate checks. No real driving accuracy is measured.",
    }

    def save():
        # All writes stay in this exclusively created run directory.
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(output / "summary.json")

    def step(name, args, validator):
        item = run_step(name, args, output, root=root)
        report["stages"].append(item)
        save()
        validator()
        item["report_validated"] = True
        save()

    save()
    try:
        environment_file = output / "environment.json"
        step("environment", [root / "scripts/check_runtime_environment.py",
                            "--require-cuda" if device == "cuda" else "--allow-cpu",
                            "--output", environment_file],
             lambda: validate_environment(read_json(environment_file), device))
        if with_tests:
            import xml.etree.ElementTree as ET
            tests_file = output / "tests.xml"

            def validate_tests():
                suites = ET.parse(tests_file).getroot().iter("testsuite")
                counts = [suite.attrib for suite in suites]
                require(bool(counts) and sum(int(row.get("tests", 0)) for row in counts) > 0,
                        "No regression tests executed")
                require(all(int(row.get("failures", 0)) == 0 and int(row.get("errors", 0)) == 0
                            for row in counts), "Regression tests failed")
                report["tests_passed"] = sum(int(row.get("tests", 0)) - int(row.get("skipped", 0))
                                           for row in counts)

            step("regression_tests", ["-m", "pytest", "tests", "integrations/bevformer/test_run.py",
                                     "-q", "--tb=short", f"--junitxml={tests_file}"], validate_tests)
        training_dir = output / "training"
        step("training", [root / "scripts/smoke_multiview_training.py", "--output", training_dir,
                         "--device", device],
             lambda: validate_training(read_json(training_dir / "verification.json")))
        fixture_dir = output / "fixture"
        config = fixture_dir / "runtime.json"
        step("fixture", ["-m", "camera_rc", "drive-fixture", "--multiview", "--device", device,
                        "--output", fixture_dir], lambda: validate_fixture(config))
        benchmark_file = output / "benchmark.json"
        validate_fixture(config)
        step("benchmark", ["-m", "camera_rc", "drive-benchmark", "--config", config,
                          "--output", benchmark_file, "--frames", "100", "--warmup", "10"],
             lambda: validate_benchmark(read_json(benchmark_file), 100))
        replay_dir = output / "headless"

        def check_replay():
            launchers = list(replay_dir.glob("*/launcher.json"))
            require(len(launchers) == 1, "Expected exactly one headless replay result")
            launcher = read_json(launchers[0])
            runtime = read_json(launchers[0].parent / "summary.json")
            validate_replay(launcher, runtime, 50)
            report["headless_output"] = str(launchers[0].parent)

        validate_fixture(config)
        step("headless", [root / "scripts/run_headless.py", "--config", config,
                          "--output-root", replay_dir, "--dry-run", "--max-frames", "50"], check_replay)
        report["success"] = True
        marker.unlink()
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    inspect = subparsers.add_parser("status", help="Report paths, disk space and missing data; no hardware opened")
    inspect.add_argument("--prepared-data", type=Path,
                         help="Prepared dataset directory; relative paths use the project root (default: data/public_v1)")
    check = subparsers.add_parser("verify", help="Run synthetic learning and four-view replay only")
    check.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    check.add_argument("--with-tests", action="store_true")
    check.add_argument("--output", type=Path, help="New result directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = status(prepared_data=args.prepared_data) if args.action == "status" else verify(
            args.output, device=args.device, with_tests=args.with_tests)
    except Exception as exc:
        report = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report.get("success", args.action == "status") else 1


if __name__ == "__main__":
    raise SystemExit(main())
