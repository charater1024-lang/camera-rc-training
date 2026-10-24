"""Explicit offline simulator entry points; default demo never loads weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def load_perception(config_path, device="cpu", allocator_gib=3.):
    """A local trusted model, never implicit download or oracle fallback."""
    if config_path is None:
        return None
    path = Path(config_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = dict(payload.get("perception", payload))
    if cfg.get("detection_checkpoint"):
        raise ValueError("Simulator supports image semantics only; omit detection_checkpoint because metric objects/tracks remain synthetic")
    for name in ("segmentation_checkpoint", "segmentation_training_report", "detection_checkpoint", "yolo_settings_dir"):
        if name in cfg:
            target = Path(cfg[name]).expanduser()
            cfg[name] = str(target if target.is_absolute() else (path.parent / target).resolve())
    cfg.setdefault("yolo_settings_dir", str(path.parent / ".ultralytics-simulator"))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from camera_rc.driving.perception import CameraPerception
    torch.set_num_threads(2)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; CPU fallback is not automatic")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(allocator_gib*2**30/total, .9), 0)
    model = CameraPerception(cfg, device=device)
    checkpoint = Path(cfg["segmentation_checkpoint"])
    model.simulator_identity = {"backend": cfg.get("segmentation_backend", "native"),
                                "checkpoint_sha256": _sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                                "config_sha256": _sha(path), "device": device,
                                "diagnostic": model.diagnostic, "physical_output": False,
                                "synthetic_render_domain": True, "road_accuracy_validation": False}
    return model


def main(argv=None):
    from .engine import Engine, SCENARIOS
    parser = argparse.ArgumentParser(description="Interactive closed-loop synthetic driving simulator; no physical output")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True, help="New output folder; previous episodes are preserved")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--scenario", choices=[s["id"] for s in SCENARIOS], default=None)
        sub.add_argument("--perception-config", type=Path, help="Trusted local perception JSON; absent means synthetic demo")
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        sub.add_argument("--allocator-gib", type=float, default=3.)
        sub.add_argument("--speed-mode", choices=["automatic", "user"], default="automatic")
        sub.add_argument("--target-speed-kmh", type=float, default=None, help="Requested synthetic speed ceiling, 0..150 km/h")
        if name == "serve":
            sub.add_argument("--port", type=int, default=8766)
        else:
            sub.add_argument("--steps", type=int, default=600)
    args = parser.parse_args(argv)
    from .speed import validate_speed
    try:
        validate_speed(args.speed_mode,150. if args.target_speed_kmh is None else args.target_speed_kmh)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.seed <= 2**31-1:
        parser.error("seed must be in 0..2147483647")
    if not .5 <= args.allocator_gib <= 16:
        parser.error("allocator-gib must be in 0.5..16")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh folder")
    if args.command == "evaluate" and not 1 <= args.steps <= 6000:
        parser.error("steps must be in 1..6000")
    model = load_perception(args.perception_config, args.device, args.allocator_gib)
    if args.command == "serve":
        from .server import serve
        engine = Engine(scenario=args.scenario or "clear", seed=args.seed, perception=model, output_dir=args.output,
                        speed_mode=args.speed_mode,target_speed_kmh=args.target_speed_kmh)
        try:
            serve(engine, SCENARIOS, args.port)
        finally:
            engine.close()
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    names = [args.scenario] if args.scenario else [s["id"] for s in SCENARIOS]
    reports = []
    for name in names:
        engine = Engine(scenario=name, seed=args.seed, perception=model, output_dir=args.output/name,
                        speed_mode=args.speed_mode,target_speed_kmh=args.target_speed_kmh)
        try:
            for _ in range(args.steps):
                if engine.step().get("finished"):
                    break
            report = engine.report()
            reports.append(report)
            print(json.dumps({"scenario": name, "metrics": engine.snapshot()["metrics"]}, ensure_ascii=False), flush=True)
        finally:
            engine.close()
    result = {"schema_version": 1, "scope": "synthetic_closed_loop_not_real_driving_validation",
              "physical_output": False, "real_time_benchmark": False, "seed": args.seed,
              "requested_steps": args.steps, "perception": "learned" if model else "synthetic_demo",
              "model_identity": getattr(model, "simulator_identity", None), "scenarios": reports}
    with (args.output/"suite.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0
