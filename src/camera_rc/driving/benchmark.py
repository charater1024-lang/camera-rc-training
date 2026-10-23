"""Whole software pipeline measurement; never a substitute for target hardware tests."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from .config import load_runtime_config
from .control import Controller
from .resources import configure_device, ResourceMonitor, latency_report
from .sources import open_source
from .types import VehicleState


def benchmark_runtime(config_path, output, frames=50, warmup=5):
    from .pipeline import DrivingPipeline
    if frames < 1 or warmup < 0:
        raise ValueError("frames>=1 and warmup>=0 required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = load_runtime_config(config_path)
    if cfg["transport"]["kind"] != "dry_run":
        raise ValueError("Benchmark requires dry_run; it never opens an actuator")
    device = configure_device(cfg)
    monitor = ResourceMonitor(device, cfg["hardware"]["memory_architecture"])
    pipeline = DrivingPipeline(cfg, device)
    model, controller = pipeline.model, Controller(cfg["controller"])
    source = open_source(cfg["source"])
    frame = None
    latencies, inference, valid = [], [], 0
    try:
        until = time.monotonic() + 5
        while frame is None and time.monotonic() < until:
            frame = source.read()
            if frame is None:
                if source.exhausted:
                    raise ValueError("No benchmark frame")
                time.sleep(0.005)
        if frame is None:
            raise ValueError("Camera did not deliver a frame")
        # Repeated input isolates model+planner+controller time; explicitly not
        # exposure-to-actuator timing or a closed-loop accuracy evaluation.
        for i in range(warmup + frames):
            begin = time.perf_counter()
            result = pipeline.run(frame, frame.speed_mps)
            plan = result.plan
            now = time.monotonic()
            command = controller.update(plan, VehicleState(now, frame.speed_mps, frame.yaw_rate_rps, True, False), now)
            elapsed = (time.perf_counter() - begin) * 1000
            monitor.sample()
            if i >= warmup:
                latencies.append(elapsed)
                inference.append(result.inference_ms)
                valid += int(plan.valid)
    finally:
        source.close()
    resources = monitor.report()
    shared_memory = cfg["hardware"]["memory_architecture"] == "shared"
    sampled_gpu = resources["process_gpu_peak_gib_sampled"]
    gpu_assessment = None if shared_memory or sampled_gpu is None else sampled_gpu <= cfg["runtime"]["gpu_budget_gib"]
    device_used = resources["whole_device_used_peak_gib_sampled"]
    summary = {"diagnostic_only": True, "physical_output": False,
               "model_diagnostic": model.diagnostic,
               "detector_enabled": model.detector is not None,
               "segmentation_input_hw": [model.height, model.width],
               "camera_input_hw": {name: list(item.rgb.shape[:2]) for name, item in frame.views.items()} if pipeline.multiview else list(frame.rgb.shape[:2]),
               "camera_names": list(pipeline.calibrations), "shared_model_instances": 1,
               "group_rate_target_hz": cfg["runtime"]["perception_hz"],
               "pipeline_latency": latency_report(latencies), "inference_latency": latency_report(inference),
               "valid_plan_frames": valid, "resources": resources, "target_budgets": cfg["runtime"],
               "ram_process_within_limit": resources["process_rss_peak_gib_sampled"] <= cfg["runtime"]["process_ram_limit_gib"],
               "system_available_floor_met_sampled": resources["system_available_min_gib_sampled"] >= cfg["runtime"]["system_available_min_gib"],
               "sampled_process_gpu_within_budget": gpu_assessment,
               "sampled_whole_device_gpu_within_budget": None if shared_memory or device_used is None else device_used <= cfg["runtime"]["gpu_budget_gib"],
               "processing_deadline_p99_met_on_this_device": np.percentile(latencies, 99) <= cfg["runtime"]["max_processing_s"] * 1000,
               "group_period_p99_met_on_this_device": bool(np.percentile(latencies, 99) <= 1000 / cfg["runtime"]["perception_hz"]),
               "note": "Repeated frame/group, one shared model instance, sequential batch1 per camera. Frames means complete groups for multiview. Includes rectification, CPU/GPU transfers, models, planner and controller. Excludes exposure, synchronization wait, transport and actuators. A memory cap does not reproduce another GPU's speed. Null GPU assessment means unavailable. Synthetic checkpoints prove execution only."}
    # numpy bool cannot be JSON serialized portably.
    summary["processing_deadline_p99_met_on_this_device"] = bool(summary["processing_deadline_p99_met_on_this_device"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def make_runtime_fixture(output, *, device="auto", detection_checkpoint=None):
    """Known-output segmentation weights for resource measurement, not driving."""
    import torch
    from PIL import Image
    from camera_rc.models import build_model
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = {"task": "segmentation", "diagnostic_only": True,
           "model": {"width": 16, "num_classes": 3}, "data": {"image_size": [192, 320]}, "training": {}}
    model = build_model(cfg)
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.head.bias[0] = 8.0
    checkpoint = root / "synthetic_segmentation.pt"
    torch.save({"format_version": 1, "diagnostic": True, "config": cfg, "model": model.state_dict()}, checkpoint)
    yy, xx = np.mgrid[:360, :640]
    image = np.stack(((xx % 255), (yy % 255), np.full_like(xx, 100)), axis=-1).astype(np.uint8)
    Image.fromarray(image).save(root / "frame.png")
    # Covers a small rear strip for simulation ONLY; actual camera range is measured.
    calibration = {"image_size": [360, 640], "image_to_ground": [[0, -0.0064, 2.0], [-0.003125, 0, 1], [0, 0, 1]],
                   "supported_range": {"min_forward_m": -0.2, "max_forward_m": 1.9, "max_abs_lateral_m": 0.9},
                   "calibrated": True, "rectified": True, "input_is_rectified": True, "diagnostic_only": True}
    (root / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    records = [{"image": "frame.png", "speed_mps": 0.0, "yaw_rate_rps": 0.0} for _ in range(300)]
    (root / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    runtime = {"device": device, "calibration": "calibration.json",
               "source": {"kind": "replay", "path": "replay.jsonl"},
               "perception": {"segmentation_checkpoint": str(checkpoint), "allow_diagnostic": True}}
    if detection_checkpoint:
        runtime["perception"]["detection_checkpoint"] = str(Path(detection_checkpoint).resolve())
    (root / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return {"config": str(root / "runtime.json"), "diagnostic_only": True}


def make_multiview_fixture(output, *, device="auto", detection_checkpoint=None):
    """Synthetic projections for code execution only; not measured camera geometry."""
    make_runtime_fixture(output, device=device, detection_checkpoint=detection_checkpoint)
    root = Path(output).resolve()
    runtime_path = root / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.pop("calibration")
    bounds = {"front": (0.0, 2.0, -0.9, 0.9), "left": (-0.4, 1.7, 0.0, 1.2),
              "right": (-0.4, 1.7, -1.2, 0.0), "rear": (-1.8, 0.1, -0.9, 0.9)}
    cameras = []
    for name, (xmin, xmax, ymin, ymax) in bounds.items():
        calibration = {
            "image_size": [360, 640], "coordinate_origin": "rear_axle_center",
            "image_to_ground": [[0, -(xmax - xmin) / 359, xmax], [-(ymax - ymin) / 639, 0, ymax], [0, 0, 1]],
            "supported_range": {"min_forward_m": xmin, "max_forward_m": xmax,
                                "min_lateral_m": ymin, "max_lateral_m": ymax},
            "calibrated": True, "rectified": True, "input_is_rectified": True, "diagnostic_only": True}
        filename = f"calibration_{name}.json"
        (root / filename).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        cameras.append({"name": name, "width": 640, "height": 360, "fps": 30, "calibration": filename})
    records = [{"views": {name: {"image": "frame.png", "offset_s": 0} for name in bounds},
                "speed_mps": 0.0, "yaw_rate_rps": 0.0} for _ in range(300)]
    (root / "multireplay.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    runtime["source"] = {"kind": "multireplay", "path": "multireplay.jsonl", "max_skew_s": 0.04, "cameras": cameras}
    runtime["runtime"] = {"perception_hz": 10}
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return {"config": str(runtime_path), "diagnostic_only": True, "camera_names": list(bounds)}
