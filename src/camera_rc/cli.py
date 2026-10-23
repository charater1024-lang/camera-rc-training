import argparse
import json
import sys
from pathlib import Path
from .config import load_config


def doctor():
    import platform
    import torch
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_name(i), "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                     for i in range(torch.cuda.device_count())]}


def main(argv=None):
    p = argparse.ArgumentParser(description="Camera driving research: datasets, staged training, bounded runtime and validation")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Report Python, CUDA and GPU memory")
    for command in ("train", "evaluate", "check-data"):
        s = subs.add_parser(command)
        s.add_argument("--config", required=True, type=Path)
        if command == "train":
            s.add_argument("--resume", type=Path)
            s.add_argument("--init-from", type=Path, help="Weights-only initialization for a new training stage")
            s.add_argument("--max-steps", type=int, help="Diagnostic cap, not a complete training run")
        elif command == "evaluate":
            s.add_argument("--checkpoint", required=True, type=Path)
    s = subs.add_parser("prepare-bdd-detection")
    for flag in ("train-labels", "val-labels", "train-images", "val-images", "output"):
        s.add_argument("--" + flag, required=True, type=Path)
    s.add_argument("--overwrite", action="store_true")
    s = subs.add_parser("prepare-bdd-drivable")
    for flag in ("train-images", "val-images", "train-masks", "val-masks", "output"):
        s.add_argument("--" + flag, required=True, type=Path)
    s = subs.add_parser("prepare-kitti-depth")
    for flag in ("raw-root", "depth-root", "output"):
        s.add_argument("--" + flag, required=True, type=Path)
    s = subs.add_parser("pair-manifest", help="Pair aligned image/target files by matching relative stem")
    s.add_argument("--images", required=True, type=Path)
    s.add_argument("--targets", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--task", choices=["segmentation", "depth"], required=True)
    s.add_argument("--group-pattern", required=True, help="Regex on relative path with a group/first capture for original drive ID")
    s = subs.add_parser("remap-masks", help="Map source class IDs explicitly; unknown pixels become ignore=255")
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--mapping", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s = subs.add_parser("prepare-cityscapes", help="Pair official Cityscapes leftImg8bit and gtFine_labelIds")
    s.add_argument("--images", required=True, type=Path)
    s.add_argument("--labels", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s = subs.add_parser("split")
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--val-fraction", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s = subs.add_parser("prepare-rc")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--offsets", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    s.add_argument("--max-gap", type=float, default=0.2)
    s = subs.add_parser("demo-data")
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("smoke", help="Generate tiny fixtures; exercise training, checkpoint and evaluation")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s = subs.add_parser("curriculum", help="Validate a staged curriculum; --execute explicitly starts training")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--execute", action="store_true")
    s.add_argument("--resume-run", action="store_true")
    s = subs.add_parser("drive", help="Camera/replay perception, planning and control; dry-run by default")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--actuate", action="store_true", help="Enable an explicitly configured serial transport after calibration")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--duration-s", type=float)
    s = subs.add_parser("drive-benchmark", help="Measure batch1 perception+planning+control without actuators")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--frames", type=int, default=50)
    s.add_argument("--warmup", type=int, default=5)
    s = subs.add_parser("drive-fixture", help="Generate diagnostic-only runtime weights, calibration and replay")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s.add_argument("--detection-checkpoint", type=Path)
    s.add_argument("--multiview", action="store_true", help="Generate a four-camera diagnostic replay using one shared model")
    s = subs.add_parser("drive-sim", help="Ideal-sensor closed-loop software integration test; no hardware")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--steps", type=int, default=250)
    s = subs.add_parser("calibrate-ground", help="Fit ground projection from measured rectified pixel/metre pairs")
    s.add_argument("--points", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s = subs.add_parser("export-runtime", help="Compact native checkpoint without optimizer or training RNG")
    s.add_argument("--checkpoint", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        p.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if isinstance(result, dict) and result.get("success") is False:
        p.exit(1, "Run reported a fault; inspect the saved summary and logs.\n")


def dispatch(args):
    if args.command == "doctor":
        return doctor()
    if args.command == "curriculum":
        from .curriculum import run_curriculum
        return run_curriculum(args.config, execute=args.execute, resume_run=args.resume_run)
    if args.command == "drive":
        from .driving.runtime import run_drive
        return run_drive(args.config, args.output, actuate=args.actuate, max_frames=args.max_frames, duration_s=args.duration_s)
    if args.command == "drive-benchmark":
        from .driving.benchmark import benchmark_runtime
        return benchmark_runtime(args.config, args.output, args.frames, args.warmup)
    if args.command == "drive-fixture":
        from .driving.benchmark import make_runtime_fixture, make_multiview_fixture
        create = make_multiview_fixture if args.multiview else make_runtime_fixture
        return create(args.output, device=args.device, detection_checkpoint=args.detection_checkpoint)
    if args.command == "drive-sim":
        from .driving.simulation import run_simulation
        return run_simulation(args.output, args.steps)
    if args.command == "calibrate-ground":
        from .driving.calibration import fit_ground_calibration
        return fit_ground_calibration(args.points, args.output)
    if args.command == "export-runtime":
        from .driving.export import export_runtime_checkpoint
        return export_runtime_checkpoint(args.checkpoint, args.output)
    if args.command == "prepare-bdd-detection":
        from .prepare import prepare_bdd_detection
        return prepare_bdd_detection(args.train_labels, args.val_labels, args.train_images, args.val_images, args.output, overwrite=args.overwrite)
    if args.command == "prepare-bdd-drivable":
        from .prepare import prepare_bdd_drivable
        return prepare_bdd_drivable(args.train_images, args.val_images, args.train_masks, args.val_masks, args.output)
    if args.command == "prepare-kitti-depth":
        from .prepare import prepare_kitti_depth
        return prepare_kitti_depth(args.raw_root, args.depth_root, args.output)
    if args.command == "pair-manifest":
        from .prepare import make_paired_manifest
        return make_paired_manifest(args.images, args.targets, args.output, args.task, args.group_pattern)
    if args.command == "remap-masks":
        from .data import remap_masks
        return remap_masks(args.manifest, args.mapping, args.output)
    if args.command == "prepare-cityscapes":
        from .data import prepare_cityscapes
        return prepare_cityscapes(args.images, args.labels, args.output)
    if args.command == "split":
        from .prepare import split_manifest
        return split_manifest(args.manifest, args.output, val_fraction=args.val_fraction, seed=args.seed)
    if args.command == "prepare-rc":
        from .rc_targets import prepare_rc
        return prepare_rc(args.csv, args.output, args.offsets, args.max_gap)
    if args.command in ("demo-data", "smoke"):
        from .demo import make_demo
        configs = make_demo(args.output)
        if args.command == "demo-data":
            return configs
        from .training import run_training, evaluate_checkpoint
        results = {}
        for task, path in configs.items():
            cfg = load_config(path)
            cfg["training"]["device"] = args.device
            run = run_training(cfg)
            results[task] = {"train": run, "evaluation": evaluate_checkpoint(cfg, Path(cfg["training"]["output_dir"]) / "best.pt")}
        summary = {"diagnostic_only": True, "hardware": doctor(), "tasks": results}
        (args.output / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return summary
    cfg = load_config(args.config)
    if cfg["task"] == "detection":
        from .detection import run_detection
        if args.command == "check-data":
            raise ValueError("Detection data is validated by prepare-bdd-detection and the YOLO loader")
        checkpoint = getattr(args, "checkpoint", None)
        if args.command == "train":
            if args.init_from and args.resume:
                raise ValueError("--init-from and --resume are mutually exclusive")
            if args.init_from:
                checkpoint = args.init_from
            if args.max_steps:
                raise ValueError("For detection set epochs in the config; max-steps is only for native tasks")
            if args.resume:
                cfg["resume"] = True
                checkpoint = args.resume
        return run_detection(cfg, mode="train" if args.command == "train" else "val", checkpoint=checkpoint)
    if args.command == "check-data":
        from .data import build_dataloaders
        train, val = build_dataloaders(cfg)
        for ds in (train.dataset, val.dataset):
            for i in range(len(ds)):
                ds[i]
        return {"train": len(train.dataset), "val": len(val.dataset), "all_samples_checked": True}
    from .training import run_training, evaluate_checkpoint
    if args.command == "train":
        return run_training(cfg, resume=args.resume, max_steps=args.max_steps, init_from=args.init_from)
    return evaluate_checkpoint(cfg, args.checkpoint)
