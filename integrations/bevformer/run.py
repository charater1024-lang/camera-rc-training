"""Prepare or launch the official BEVFormer-tiny on one GPU.

Config preparation uses only the Python standard library. Training needs the
separate, legacy BEVFormer environment documented in docs/BEVFORMER.md.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


REFERENCE_COMMIT = "66b65f3a1f58caf0507cb2a971b9c0e7f842376c"
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
VARIANTS = {
    "fp32": ("projects/configs/bevformer/bevformer_tiny.py", "tools/train.py"),
    "fp16": (
        "projects/configs/bevformer_fp16/bevformer_tiny_fp16.py",
        "tools/fp16/train.py",
    ),
}


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return result


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--upstream", type=Path, required=True,
                     help="existing official BEVFormer checkout; never cloned automatically")
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--dataset-version", choices=("v1.0-mini", "v1.0-trainval"),
                     default="v1.0-trainval")
    cli.add_argument("--train-info", type=Path,
                     help="defaults to DATA_ROOT/nuscenes_infos_temporal_train.pkl")
    cli.add_argument("--val-info", type=Path,
                     help="defaults to DATA_ROOT/nuscenes_infos_temporal_val.pkl")
    cli.add_argument("--output-dir", type=Path, default=Path("runs/bevformer_tiny"))
    cli.add_argument("--lr", type=positive_float, required=True,
                     help="explicit learning rate for global batch 1; no automatic scaling")
    cli.add_argument("--epochs", type=positive_int, default=24)
    cli.add_argument("--workers", type=nonnegative_int, default=2)
    cli.add_argument("--gpu", type=nonnegative_int, default=0,
                     help="one physical GPU exposed through CUDA_VISIBLE_DEVICES")
    cli.add_argument("--seed", type=nonnegative_int, default=42)
    cli.add_argument("--precision", choices=tuple(VARIANTS), default="fp32")
    cli.add_argument("--log-interval", type=positive_int, default=50)
    cli.add_argument("--checkpoint-interval", type=positive_int, default=1)
    cli.add_argument("--eval-interval", type=positive_int, default=1)
    initialization = cli.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--backbone-weights", type=Path,
                                help="trusted local ResNet50 ImageNet checkpoint")
    initialization.add_argument("--load-from", type=Path,
                                help="trusted local BEVFormer checkpoint; optimizer starts fresh")
    initialization.add_argument("--resume-from", type=Path,
                                help="trusted local BEVFormer checkpoint; restores training state")
    initialization.add_argument("--random-init", action="store_true",
                                help="no pretrained backbone; useful for pipeline checks")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="default: validate paths, write config and launch plan only")
    mode.add_argument("--execute", action="store_true",
                      help="actually run training in the current Python environment")
    cli.add_argument("--allow-unpinned", action="store_true",
                     help="allow an unverified/modified upstream checkout; record this in plan")
    cli.add_argument("--overwrite", action="store_true",
                     help="replace different generated config/plan files, never checkpoints")
    return cli


def require_file(path, label):
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("{} is missing or empty: {}".format(label, path))
    return path


def git_state(upstream):
    if not (upstream / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(upstream), check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(upstream), check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def make_plan(args):
    """Validate inputs without importing torch, loading pickle, or writing files."""
    upstream = args.upstream.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    base_relative, trainer_relative = VARIANTS[args.precision]
    base = require_file(upstream / base_relative, "official tiny config")
    trainer = require_file(upstream / trainer_relative, "official training entry point")
    require_file(upstream / "projects/configs/datasets/custom_nus-3d.py", "dataset base config")
    require_file(upstream / "projects/configs/_base_/default_runtime.py", "runtime base config")
    require_file(upstream / "projects/mmdet3d_plugin/__init__.py", "BEVFormer plugin")
    require_file(upstream / "LICENSE", "upstream license")
    state = git_state(upstream)
    verified = state["commit"] == REFERENCE_COMMIT and state["dirty"] is False
    if not verified and not args.allow_unpinned:
        raise ValueError(
            "Use a clean checkout of {}. Observed commit={}, dirty={}. "
            "If intentionally using another checkout, pass --allow-unpinned."
            .format(REFERENCE_COMMIT, state["commit"], state["dirty"])
        )
    if not data_root.is_dir():
        raise ValueError("data root is missing: {}".format(data_root))
    for camera in CAMERAS:
        if not (data_root / "samples" / camera).is_dir():
            raise ValueError("camera directory is missing: {}".format(data_root / "samples" / camera))
    for name in ("scene.json", "sample.json", "sample_data.json", "ego_pose.json",
                 "calibrated_sensor.json", "sample_annotation.json"):
        require_file(data_root / args.dataset_version / name, "nuScenes metadata")
    train_info = require_file(
        args.train_info or data_root / "nuscenes_infos_temporal_train.pkl", "training annotations"
    )
    val_info = require_file(
        args.val_info or data_root / "nuscenes_infos_temporal_val.pkl", "validation annotations"
    )
    if train_info == val_info:
        raise ValueError("train and validation annotations must be different files")
    weight_path = None
    for attribute in ("backbone_weights", "load_from", "resume_from"):
        if getattr(args, attribute) is not None:
            weight_path = require_file(getattr(args, attribute), attribute)
    config_path = output / "bevformer_tiny_local.py"
    # A separate work directory prevents upstream Config.dump from replacing our
    # inherited config with a flattened one during training/resume.
    work_dir = output / "work"
    if config_path in (base, trainer, train_info, val_info, weight_path):
        raise ValueError("output config must not overwrite an input file")
    argv = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc_per_node=1", str(trainer), str(config_path),
        "--launcher", "pytorch", "--work-dir", str(work_dir), "--seed", str(args.seed),
    ]
    if args.resume_from:
        argv.extend(["--resume-from", str(weight_path)])
    plan = {
        "reference_commit": REFERENCE_COMMIT,
        "upstream": str(upstream), "upstream_state": state, "upstream_verified": verified,
        "base_config": str(base), "config": str(config_path), "work_dir": str(work_dir),
        "data_root": str(data_root), "dataset_version": args.dataset_version,
        "train_info": str(train_info), "val_info": str(val_info),
        "weight_path": str(weight_path) if weight_path else None,
        "initialization": next((key for key in ("backbone_weights", "load_from", "resume_from")
                                if getattr(args, key)), "random"),
        "precision": args.precision, "epochs": args.epochs,
        "samples_per_gpu": 1, "gpu_count": 1, "global_batch": 1,
        "workers_per_gpu": args.workers, "learning_rate": args.lr,
        "argv": argv, "cwd": str(upstream),
        "environment_overrides": {"CUDA_VISIBLE_DEVICES": str(args.gpu),
                                  "PYTHONPATH_prepend": str(upstream)},
        "annotation_check": "existence only; pickle contents and media are checked by upstream at run time",
    }
    return plan


def config_text(args, plan):
    # repr() preserves Unicode, quotes, spaces, and Windows backslashes as Python
    # string literals. We never interpolate paths into shell source.
    root = Path(plan["data_root"]).as_posix().rstrip("/") + "/"
    backbone = {"img": plan["weight_path"]} if args.backbone_weights else None
    lines = [
        "# Generated by integrations/bevformer/run.py; training uses official upstream code.",
        "# Global batch = 1; learning rate was explicitly supplied, never auto-scaled.",
        "_base_ = {!r}".format([Path(plan["base_config"]).as_posix()]),
        "data_root = {!r}".format(root),
        "data = dict(",
        "    samples_per_gpu=1, workers_per_gpu={},".format(args.workers),
        "    train=dict(data_root=data_root, ann_file={!r}),".format(plan["train_info"]),
        "    val=dict(data_root=data_root, ann_file={!r}, samples_per_gpu=1),".format(plan["val_info"]),
        "    test=dict(data_root=data_root, ann_file={!r}),".format(plan["val_info"]),
        ")",
        "model = dict(pretrained={!r})".format(backbone),
        "optimizer = dict(lr={!r})".format(args.lr),
        "total_epochs = {}".format(args.epochs),
        "runner = dict(max_epochs=total_epochs)",
        "evaluation = dict(interval={})".format(args.eval_interval),
        "checkpoint_config = dict(interval={}, max_keep_ckpts=3)".format(args.checkpoint_interval),
        # Avoid adding TensorBoard as another legacy-environment requirement.
        "log_config = dict(interval={}, hooks=[dict(type='TextLoggerHook')])".format(args.log_interval),
        "work_dir = {!r}".format(plan["work_dir"]),
    ]
    if args.load_from:
        lines.append("load_from = {!r}".format(plan["weight_path"]))
    return "\n".join(lines) + "\n"


def write_plan(args, plan):
    config = Path(plan["config"])
    metadata = config.with_name("launch_plan.json")
    payloads = {
        config: config_text(args, plan),
        metadata: json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
    }
    for path, content in payloads.items():
        if path.exists() and path.read_text(encoding="utf-8") != content and not args.overwrite:
            raise ValueError("output differs: {} (choose another --output-dir or use --overwrite)".format(path))
    config.parent.mkdir(parents=True, exist_ok=True)
    for path, content in payloads.items():
        path.write_text(content, encoding="utf-8")


def check_training_environment():
    if platform.system() != "Linux":
        raise ValueError("Training requires Linux/WSL2 with NVIDIA CUDA. Config preparation works on Windows.")
    if sys.version_info[:2] != (3, 8):
        raise ValueError("Use the separate Python 3.8 BEVFormer environment; see docs/BEVFORMER.md.")
    expected = {"torch": "1.9.1", "torchvision": "0.10.1", "mmcv-full": "1.4.0",
                "mmdet": "2.14.0", "mmsegmentation": "0.14.1", "mmdet3d": "0.17.1"}
    for package, version in expected.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("Missing {}=={} in this environment".format(package, version))
        if installed.split("+")[0] != version:
            raise ValueError("Expected {}=={}; found {}".format(package, version, installed))
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to this Python environment")
    if torch.version.cuda != "11.1":
        raise ValueError("Expected the torch CUDA 11.1 build; found {}".format(torch.version.cuda))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = make_plan(args)
        if args.execute:
            # Restrict GPU visibility before importing torch for the preflight.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            check_training_environment()
        write_plan(args, plan)
        print("Config: {}".format(plan["config"]))
        print("Launch plan: {}".format(Path(plan["config"]).with_name("launch_plan.json")))
        print("Working directory: {}".format(plan["cwd"]))
        print("Learning rate: {}; global batch: 1; epochs: {}".format(args.lr, args.epochs))
        print("Linux command (environment overrides are listed in launch_plan.json):")
        print(shlex.join(plan["argv"]))
        if not plan["upstream_verified"]:
            print("WARNING: upstream commit/clean state is unverified", file=sys.stderr)
        if not args.execute:
            print("Prepared only. No training, package installation, or downloads were started.")
            return 0
        env = os.environ.copy()
        env["PYTHONPATH"] = str(plan["upstream"]) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.run(plan["argv"], cwd=plan["cwd"], env=env, check=False).returncode
    except (ValueError, OSError) as error:
        print("BEVFormer setup error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
