"""Offline random-weight YOLO26 architecture timing; never an accuracy result.

Builds only an installed local YAML (no pretrained weight path/download). One
shared fused detector processes views sequentially. No cameras, road semantic
model, tracker, planner, actuator, or 80km/h driving validation is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch

from camera_rc.driving.resources import ResourceMonitor, latency_report

MODEL_CHOICES = tuple("yolo26" + scale for scale in "nsmlx")


def validate_options(model, imgsz, views, precision, device, groups, warmup, gpu_budget_gib):
    if model not in MODEL_CHOICES or imgsz not in {640, 960} or views not in {1, 8}:
        raise ValueError("Use model yolo26n/s/m/l/x, imgsz 640/960, and views 1/8")
    if precision not in {"fp32", "fp16"} or device not in {"cpu", "cuda"}:
        raise ValueError("Use precision fp32/fp16 and device cpu/cuda")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU benchmarking supports fp32 only")
    if gpu_budget_gib not in {4, 8} or isinstance(gpu_budget_gib, bool):
        raise ValueError("GPU budget must be explicitly 4 or 8 GiB")
    if isinstance(groups, bool) or not isinstance(groups, int) or not 1 <= groups <= 10000:
        raise ValueError("groups must be an integer from 1 to 10000")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 1000:
        raise ValueError("warmup must be an integer from 0 to 1000")


def output_structure(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [output_structure(item) for item in value]
    if isinstance(value, dict):
        return {str(key): output_structure(item) for key, item in value.items()}
    return {"type": type(value).__name__}


def benchmark(output, *, model="yolo26n", imgsz=640, views=1, precision="fp32",
              device="cpu", groups=20, warmup=3, gpu_budget_gib=8):
    validate_options(model, imgsz, views, precision, device, groups, warmup, gpu_budget_gib)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    # Keep Ultralytics settings inside the chosen output area, not user defaults.
    # Only YAML model construction is used, never YOLO('...pt') or predict().
    settings_dir = output.parent / (output.stem + "_ultralytics_settings")
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    output.parent.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(exist_ok=False)
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({name: False for name in ("sync", "hub", "wandb", "mlflow", "clearml", "neptune", "comet") if name in settings})
    package_root = Path(ultralytics.__file__).resolve().parent
    architecture = package_root / "cfg/models/26/yolo26.yaml"
    if not architecture.is_file():
        raise FileNotFoundError("Installed Ultralytics lacks the local YOLO26 architecture")
    scaled_architecture = architecture.with_name(model + ".yaml")
    # Installed yaml_model_load resolves a scaled name to local yolo26.yaml
    # and extracts the scale; the resolved source file is checked above.
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    resolved = torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")
    allocator_cap = gpu_budget_gib - 1
    actual_total, effective_cap = None, None
    if resolved.type == "cuda":
        actual_total = torch.cuda.get_device_properties(resolved).total_memory / 2**30
        fraction = min(allocator_cap / actual_total, .9)
        effective_cap = fraction * actual_total
        torch.cuda.set_per_process_memory_fraction(fraction, resolved)
        torch.cuda.reset_peak_memory_stats(resolved)
    wrapper = YOLO(str(scaled_architecture), task="detect", verbose=False)
    initial_head = wrapper.model.model[-1]
    default_end2end = bool(getattr(initial_head, "end2end", False))
    if any(getattr(initial_head, name, None) is None for name in ("one2one_cv2", "one2one_cv3")):
        raise ValueError("Expected the installed YOLO26 end-to-end detection head")
    # YAML creates both branches, but recent Ultralytics versions default to
    # one-to-many inference. Select the public model property BEFORE fuse(),
    # which permanently removes whichever branch is not selected.
    wrapper.model.end2end = True
    if not getattr(initial_head, "end2end", False):
        raise ValueError("Could not select YOLO26 end-to-end inference before fusion")
    wrapper.model.eval()
    wrapper.fuse(verbose=False, imgsz=imgsz)
    network = wrapper.model.eval().to(resolved)
    if precision == "fp16":
        network.half()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    head = network.model[-1]
    if (not getattr(head, "end2end", False)
            or any(getattr(head, name, None) is None for name in ("one2one_cv2", "one2one_cv3"))
            or any(getattr(head, name, None) is not None for name in ("cv2", "cv3"))):
        raise ValueError("Fusion did not preserve only the YOLO26 end-to-end detection branch")
    # The installed Model initializes ckpt={} for YAML construction.
    if getattr(wrapper, "ckpt", None):
        raise ValueError("Random architecture benchmark must not contain a pretrained checkpoint")
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand((views, 3, imgsz, imgsz), generator=generator).to(device=resolved, dtype=dtype)
    monitor = ResourceMonitor(str(resolved), "discrete")
    per_view, per_group = [], []
    result = None
    with torch.inference_mode():
        for group_index in range(warmup + groups):
            if resolved.type == "cuda":
                torch.cuda.synchronize(resolved)
            group_started = time.perf_counter()
            view_times = []
            for view in range(views):
                if resolved.type == "cuda":
                    torch.cuda.synchronize(resolved)
                started = time.perf_counter()
                result = network(images[view:view + 1])
                if resolved.type == "cuda":
                    torch.cuda.synchronize(resolved)
                view_times.append((time.perf_counter() - started) * 1000)
            elapsed_group = (time.perf_counter() - group_started) * 1000
            # Validation and memory sampling are outside the timed graph loop.
            detections = result[0] if isinstance(result, (tuple, list)) else result
            if (not isinstance(detections, torch.Tensor) or detections.ndim != 3
                    or detections.shape[0] != 1 or detections.shape[-1] != 6
                    or not torch.isfinite(detections).all()):
                raise FloatingPointError("Nonfinite or unexpected detector output")
            monitor.sample()
            if group_index >= warmup:
                per_view.extend(view_times)
                per_group.append(elapsed_group)
    resources = monitor.report()
    report = {
        "diagnostic_only": True, "model_random_initialized": True, "pretrained_weights_loaded": False,
        "accuracy_evaluated": False, "physical_output": False, "real_driving_validated": False,
        "evidence_for_80kmh": False, "model": model, "ultralytics_version": importlib.metadata.version("ultralytics"),
        "torch_version": torch.__version__, "torchvision_version": importlib.metadata.version("torchvision"),
        "architecture_yaml": str(architecture), "architecture_yaml_sha256": hashlib.sha256(architecture.read_bytes()).hexdigest(),
        "selected_scale": model[-1], "seed": seed, "fused": True,
        "parameter_count_fused": sum(value.numel() for value in network.parameters()),
        "device": str(resolved), "precision": precision, "cpu_intraop_threads": torch.get_num_threads(),
        "actual_gpu_name": resources["gpu_name"], "actual_gpu_memory_gib": actual_total,
        "declared_gpu_budget_gib": gpu_budget_gib, "requested_allocator_cap_gib": allocator_cap,
        "effective_allocator_cap_gib": effective_cap, "cap_emulates_target_hardware": False,
        "views": views, "history_frames": 1, "shared_model_instances": 1, "per_call_batch_size": 1,
        "input_pattern": "Seeded uniform random RGB [0,1], independent per view; same preloaded tensors repeated per group",
        "input_shape": list(images.shape), "input_tensor_bytes": images.numel() * images.element_size(),
        "graph": {"head_class": type(head).__name__, "end2end": bool(head.end2end),
                  "end2end_default_before_selection": default_end2end,
                  "end2end_explicitly_selected_before_fusion": True,
                  "export": bool(getattr(head, "export", False)),
                  "one_to_many_branch_present": getattr(head, "cv2", None) is not None,
                  "one_to_one_branch_present": getattr(head, "one2one_cv2", None) is not None,
                  "output_structure": output_structure(result)},
        "measured_groups": groups, "warmup_groups": warmup,
        "single_view_latency": latency_report(per_view), "all_views_group_latency": latency_report(per_group),
        "resources": resources,
        "scope": "Actual fused PyTorch detector graph including its internal head decoding/top-k when enabled. External confidence filtering/NMS, CPU-to-GPU copies, image resize/rectification, capture/USB, PIDNet, tracking, planning and control are excluded. This is not a complete perception pipeline benchmark. Sequential calls emulate a shared detector's per-group work, not exposure synchronization or a real latest-frame source.",
        "limits": "Random initialized weights provide architecture-cost evidence only. No recognition, distance, obstacle recall, or 80km/h safety claim is possible. A GPU memory cap does not emulate another GPU; sampled memory can miss peaks outside the PyTorch allocator.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--imgsz", type=int, choices=(640, 960), required=True)
    parser.add_argument("--views", type=int, choices=(1, 8), default=8)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-budget-gib", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
