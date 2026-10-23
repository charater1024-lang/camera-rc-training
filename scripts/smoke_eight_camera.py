"""Execute eight synthetic camera views with one model; no real driving validation.

The generated images, weights and calibration are diagnostic fixtures. This
does not train on road imagery, measure camera synchronization, or open any
camera/serial device. CPU timing limits are relaxed only in this fixture.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from camera_rc.driving.benchmark import make_multiview_fixture, benchmark_runtime
from camera_rc.driving.runtime import run_drive


ROLE_SOURCE = {
    "front_main": "front", "front_wide": "front", "front_near": "front",
    "left_forward": "left", "right_forward": "right",
    "left_rearward": "left", "right_rearward": "right", "rear": "rear",
}
RUNTIME_GROUPS = 2
BENCHMARK_GROUPS = 20
BENCHMARK_WARMUP = 2


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def make_fixture(output, device="cpu"):
    """Expand the existing four-view diagnostic fixture without claiming geometry."""
    root = Path(output).resolve()
    existing = make_multiview_fixture(root, device=device)
    path = Path(existing["config"])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    originals = {camera["name"]: camera for camera in cfg["source"]["cameras"]}
    cameras = []
    for role, source in ROLE_SOURCE.items():
        camera = copy.deepcopy(originals[source])
        camera["name"] = role
        original_path = root / camera["calibration"]
        calibration = json.loads(original_path.read_text(encoding="utf-8"))
        calibration["diagnostic_only"] = True
        calibration["geometry_note"] = "Duplicated synthetic projection for execution only; not measured camera geometry."
        name = f"synthetic_calibration_{role}.json"
        write_json(root / name, calibration)
        camera["calibration"] = name
        cameras.append(camera)
    cfg["source"].update(cameras=cameras, path="eight_camera_synthetic_replay.jsonl",
                          capture_timing_validated=False)
    cfg["models_validated_for_control"] = False
    cfg["hardware"] = {"profile": "eight_camera_synthetic_diagnostic", "memory_architecture": "discrete"}
    cfg["transport"] = {"kind": "dry_run", "calibration_confirmed": False}
    cfg["runtime"] = {
        "perception_hz": 10, "control_hz": 50,
        "frame_timeout_s": 2, "plan_timeout_s": 2, "max_processing_s": 1,
        "gpu_budget_gib": 4, "torch_allocator_cap_gib": 3,
        "ram_budget_gib": 16, "process_ram_limit_gib": 12, "system_available_min_gib": 1,
    }
    cfg["planner"] = {"reaction_time_s": 2.5}
    records = [{"views": {role: {"image": "frame.png", "offset_s": 0} for role in ROLE_SOURCE},
                "speed_mps": 0, "yaw_rate_rps": 0} for _ in range(RUNTIME_GROUPS)]
    replay = root / cfg["source"]["path"]
    replay.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    write_json(path, cfg)
    write_json(root / "SYNTHETIC_ONLY.json", {
        "diagnostic_only": True, "physical_output": False, "real_driving_validated": False,
        "camera_names": list(ROLE_SOURCE), "training_performed": False,
        "synthetic_image_repeated_across_views": True,
        "geometry": "All eight role projections are fabricated diagnostic fixtures, with duplicated coverage.",
        "timing": "Relaxed 1s processing/2s plan expiry is only for this CPU-capable smoke test, not the MSI profile.",
    })
    return path


def run_smoke(output, device="cpu"):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Bound CPU scheduling overhead; this setting is reported with timings.
    torch.set_num_threads(2)
    config_path = make_fixture(root / "synthetic_fixture", device=device)
    runtime = run_drive(config_path, root / "runtime", actuate=False, max_frames=RUNTIME_GROUPS)
    benchmark = benchmark_runtime(config_path, root / "benchmark.json", frames=BENCHMARK_GROUPS,
                                  warmup=BENCHMARK_WARMUP)
    perception_rows = [json.loads(line) for line in (root / "runtime/perception.jsonl").read_text(encoding="utf-8").splitlines()]
    control_rows = [json.loads(line) for line in (root / "runtime/control.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = set(ROLE_SOURCE)
    checks = {
        "runtime_completed": runtime["success"] and runtime["frames"] == RUNTIME_GROUPS,
        "all_eight_views_processed": len(perception_rows) == RUNTIME_GROUPS and all(set(row["views"]) == expected for row in perception_rows),
        "synthetic_clear_plans_valid": bool(perception_rows) and all(row["plan_valid"] is True for row in perception_rows),
        "independent_control_loop_executed": runtime["control_ticks"] > 0,
        "runtime_shared_one_model": runtime["shared_model_instances"] == 1,
        "benchmark_shared_one_model": benchmark["shared_model_instances"] == 1,
        "benchmark_has_eight_views": set(benchmark["camera_names"]) == expected,
        "synthetic_checkpoint_kept_diagnostic": benchmark["model_diagnostic"] is True,
        "no_physical_output": runtime["physical_output"] is False and benchmark["physical_output"] is False
                              and bool(control_rows) and all(row["physical_output"] is False for row in control_rows),
        "shutdown_stop_sent": runtime["shutdown_stop_sent"] is True,
        "benchmark_measured_twenty_groups": benchmark["pipeline_latency"]["frames"] == BENCHMARK_GROUPS,
    }
    report = {
        "success": all(checks.values()), "diagnostic_only": True, "physical_output": False,
        "real_driving_validated": False, "training_performed": False, "checks": checks,
        "device_requested": device, "cpu_intraop_threads": torch.get_num_threads(),
        "camera_names": list(ROLE_SOURCE), "shared_model_instances": 1,
        "runtime_groups": runtime["frames"], "benchmark_groups": BENCHMARK_GROUPS,
        "benchmark_warmup_groups": BENCHMARK_WARMUP,
        "control_ticks": runtime["control_ticks"], "control_reasons": runtime["control_reasons"],
        "pipeline_latency": benchmark["pipeline_latency"], "resources": benchmark["resources"],
        "synthetic_runtime_limits": {"max_processing_s": 1, "plan_timeout_s": 2, "reaction_time_s": 2.5},
        "actual_msi_deadlines_tested": False,
        "runtime_report": str(root / "runtime/summary.json"), "benchmark_report": str(root / "benchmark.json"),
        "limits": "Eight synthetic images with fabricated calibration and known-output diagnostic weights; no real-world accuracy, synchronization, obstacle recall, collision prevention or MSI performance is established. The shared model is actually loaded and evaluated, not mocked. Benchmark repeats one group and excludes capture/USB/actuators. Memory limits do not emulate another GPU.",
    }
    write_json(root / "verification.json", report)
    if not report["success"]:
        raise RuntimeError("Eight-camera synthetic smoke failed; inspect verification.json and runtime/summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New folder; existing folders are refused")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output, args.device), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
