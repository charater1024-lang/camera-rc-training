"""Synthetic joint-policy resource benchmark. No cameras, training, or actuation.

Changing --views for this diagnostic workload does not train that camera layout.
An allocator cap on a larger GPU does not emulate a smaller GPU's performance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from camera_rc.driving.resources import ResourceMonitor, latency_report
from camera_rc.joint_policy import build_joint_policy, parameter_count
from camera_rc.joint_policy.model import MODEL_TYPE


def _same_semantics(left, right):
    fields = ("width", "embed_dim", "token_grid", "horizon", "image_size", "max_history_s")
    return (all(getattr(left, key) == getattr(right, key) for key in fields)
            and left.blocks[0].attention.num_heads == right.blocks[0].attention.num_heads
            and torch.equal(left.future_dt_s, right.future_dt_s))


def synthetic_inputs(model, views, history, image_size, device):
    """Fabricated geometry and random pixels solely for workload measurement."""
    height, width = image_size
    generator = torch.Generator().manual_seed(42)
    images = torch.rand((1, history, views, 3, height, width), generator=generator).to(device)
    k = torch.tensor([[width * .8, 0, (width - 1) / 2], [0, width * .8, (height - 1) / 2],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    k = k.expand(1, history, views, 3, 3).clone()
    transforms = torch.eye(4, device=device).expand(1, history, views, 4, 4).clone()
    history_duration = min(model.max_history_s, .2 * (history - 1))
    times = torch.linspace(-history_duration, 0, history, device=device).unsqueeze(0)
    optical_to_flu = torch.tensor([[0., 0, 1], [-1, 0, 0], [0, -1, 0]], device=device)
    for view in range(views):
        yaw = 2 * math.pi * view / views
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.]], device=device)
        transforms[:, :, view, :3, :3] = rotation @ optical_to_flu
        transforms[:, :, view, 0, 3] = .2 * c + .2 * times
        transforms[:, :, view, 1, 3] = .2 * s
        transforms[:, :, view, 2, 3] = .3
    return images, k, transforms, times, torch.tensor([[.2, 0]], device=device)


def benchmark(config_path, output, *, checkpoint=None, device="cpu", views=8, history=3,
              frames=5, gpu_budget_gib=None):
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 10000:
        raise ValueError("frames must be an integer from 1 to 10000")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = Path(config_path).resolve()
    original = json.loads(config_path.read_text(encoding="utf-8-sig"))
    # Validate the original configuration before explicit synthetic overrides.
    original_model = build_joint_policy(original)
    if original_model.image_size is None:
        raise ValueError("Synthetic benchmark requires explicit data.image_size")
    runtime = original.get("runtime", {})
    budget = gpu_budget_gib if gpu_budget_gib is not None else runtime.get("gpu_budget_gib")
    if isinstance(budget, bool) or budget not in (4, 8):
        raise ValueError("Declare runtime.gpu_budget_gib=4 or 8, or pass --gpu-budget-gib")
    cap = runtime.get("torch_allocator_cap_gib", budget - 1)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 < cap < budget:
        raise ValueError("torch_allocator_cap_gib must be positive and below the declared GPU budget")
    cfg = copy.deepcopy(original)
    cfg.setdefault("data", {}).update(expected_views=views, expected_history=history,
                                      camera_names=[f"synthetic_view_{i}" for i in range(views)])
    if "expected_views" in cfg.get("model", {}):
        cfg["model"]["expected_views"] = views
    model = build_joint_policy(cfg)
    provenance = {"checkpoint": None, "weights": "random_initialization", "training_performed": False}
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(stored, dict) or stored.get("format_version") != 1 or stored.get("model_type") != MODEL_TYPE:
            raise ValueError("Expected a joint_camera_temporal_waypoints checkpoint format_version=1")
        source_model = build_joint_policy(stored.get("config", {}))
        if not _same_semantics(source_model, original_model):
            raise ValueError("Checkpoint architecture, image size, history-age limit or future query times differ from config")
        state = stored.get("model")
        if not isinstance(state, dict) or "future_dt_s" not in state or not torch.equal(state["future_dt_s"], model.future_dt_s):
            raise ValueError("Checkpoint future query times must match configuration")
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Checkpoint contains invalid or nonfinite weights")
        model.load_state_dict(state, strict=True)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance.update(checkpoint=str(checkpoint), sha256=digest.hexdigest(), weights="provided_checkpoint",
                          source_diagnostic=bool(stored.get("diagnostic")),
                          checkpoint_expected_views=source_model.expected_views,
                          checkpoint_expected_history=source_model.expected_history)
        del source_model, stored, state
    del original_model
    if device not in ("cpu", "cuda"):
        raise ValueError("Benchmark device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    resolved_device = torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")
    actual_gpu_memory = None
    if resolved_device.type == "cuda":
        actual_gpu_memory = torch.cuda.get_device_properties(resolved_device).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(cap / actual_gpu_memory, .9), resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    model.to(resolved_device).eval()
    batch = synthetic_inputs(model, views, history, model.image_size, resolved_device)
    monitor = ResourceMonitor(str(resolved_device), "discrete")
    milliseconds = []
    output_shape = None
    with torch.inference_mode():
        for index in range(frames + 2):
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            started = time.perf_counter()
            prediction = model(*batch)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            elapsed = (time.perf_counter() - started) * 1000
            output_shape = list(prediction.shape)
            if index >= 2:
                milliseconds.append(elapsed)
            monitor.sample()
    resources = monitor.report()
    report = {
        "diagnostic_only": True, "synthetic_inputs": True, "physical_output": False,
        "real_driving_validated": False, "trained_camera_layout_claim": False,
        "model_type": MODEL_TYPE, "parameter_count": parameter_count(model),
        "device": str(resolved_device), "actual_gpu_name": resources["gpu_name"],
        "actual_gpu_memory_gib": actual_gpu_memory, "declared_gpu_budget_gib": budget,
        "torch_allocator_cap_gib": cap, "cap_emulates_target_gpu": False,
        "precision": "float32", "cpu_intraop_threads": torch.get_num_threads(),
        "input_shape": list(batch[0].shape), "output_shape": output_shape,
        "future_dt_s": model.future_dt_s.detach().cpu().tolist(),
        "original_config_camera_names": original.get("data", {}).get("camera_names"),
        "original_expected_views": original.get("data", {}).get("expected_views"),
        "original_expected_history": original.get("data", {}).get("expected_history"),
        "synthetic_overrides": {"expected_views": views, "expected_history": history,
                                "camera_names": cfg["data"]["camera_names"]},
        "shared_image_encoder_instances": 1, "encoder_chunk_size": model.encoder_chunk_size,
        "history_and_views_jointly_attended": True, "measured_groups": frames, "warmup_groups": 2,
        "latency": latency_report(milliseconds), "resources": resources, "provenance": provenance,
        "limits": "Random pixels and fabricated calibration measure one offline joint model workload only. No camera capture, synchronization, USB transfer, path safety check, control or actuator is measured. Repeating inputs is not an accuracy or collision-avoidance evaluation. Changing camera count does not train those roles. An allocator cap on this GPU does not reproduce target-GPU speed or all process memory.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; never overwrite")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--gpu-budget-gib", type=int, choices=(4, 8))
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config, args.output, checkpoint=args.checkpoint, device=args.device,
                               views=args.views, history=args.history, frames=args.frames,
                               gpu_budget_gib=args.gpu_budget_gib), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
