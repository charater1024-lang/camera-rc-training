"""Train YOLO26x semantic weights on a verified, complete export.

This runner records real CUDA work and checkpoint identities. It never deploys
weights or treats an incomplete/diagnostic run as a completed experiment.
The default initializer is pinned official weights. Fine-tuning instead requires
the complete training report for a verified best checkpoint; optimizer state is
always newly initialized, never resumed from that checkpoint.

--gpu-memory-gib limits this process's PyTorch CUDA caching allocator, not total
device memory or other processes. --nominal-batch controls Ultralytics nbs;
accumulation varies during warmup and is not equivalent to a larger physical
batch (including BatchNorm, partial batches and optimizer/loss normalization).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time


OFFICIAL_SHA256 = "eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d"
AMP_HELPER_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
CLASS_NAMES = ["road_surface", "not_road_surface"]


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_export(root, source_config):
    import yaml
    root = Path(root).resolve(strict=True)
    if (root / ".incomplete").exists() or not (root / ".complete").is_file():
        raise ValueError("Semantic export is incomplete")
    marker = json.loads((root / ".complete").read_text(encoding="utf-8"))
    report_id, yaml_id = identity(root / "report.json"), identity(root / "dataset.yaml")
    if (marker.get("report_sha256") != report_id["sha256"] or
            marker.get("dataset_yaml_sha256") != yaml_id["sha256"]):
        raise ValueError("Semantic export integrity mismatch")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("complete") is not True or report.get("diagnostic_only") is not False or marker.get("diagnostic_only") is not False:
        raise ValueError("Only a complete, unrestricted export can train this experiment")
    dataset = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    names = dataset.get("names")
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    if names != dict(enumerate(CLASS_NAMES)) or dataset.get("masks_dir") != "masks":
        raise ValueError("Requires two-class PNG masks: road=0, nonroad=1, ignore=255")
    config_id = identity(source_config)
    export_config = report.get("source_config", report.get("config", {}))
    if not isinstance(export_config, dict) or export_config.get("sha256") != config_id["sha256"]:
        raise ValueError("Export source config does not match the requested experiment")
    counts = {}
    for split in ("train", "val"):
        for section in ("image_lists", "manifest_outputs"):
            entry = report[section][split]
            if identity(entry["path"])["sha256"] != entry["sha256"]:
                raise ValueError(f"Changed export {section}/{split}")
        if Path(dataset[split]).resolve() != Path(report["image_lists"][split]["path"]).resolve():
            raise ValueError("YAML list path differs from verified export")
        entries = Path(dataset[split]).read_text(encoding="utf-8").splitlines()
        entries = [line.strip() for line in entries if line.strip()]
        if not entries or len(entries) != len(set(entries)):
            raise ValueError(f"Empty/duplicate {split} image list")
        counts[split] = len(entries)
        if len(entries) != sum(report["full_input_counts"][split].values()):
            raise ValueError("Export count differs from full input cohort")
    return {"provenance_path": report_id["path"], "provenance_sha256": report_id["sha256"],
            "yaml": yaml_id, "counts": counts}, config_id


def validate_parameters(epochs, image_size, batch, workers, lr):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if image_size < 128 or image_size % 32 or batch < 1 or workers < 0:
        raise ValueError("Use image size >=128 divisible by 32, positive batch and nonnegative workers")
    if not math.isfinite(lr) or not 0 < lr < 1:
        raise ValueError("lr must be finite and in (0,1)")


def batch_policy(batch, nominal_batch=None):
    if type(batch) is not int or batch < 1:
        raise ValueError("physical batch must be a positive integer")
    nominal = batch if nominal_batch is None else nominal_batch
    if type(nominal) is not int or nominal < batch:
        raise ValueError("nominal batch must be an integer >= physical batch")
    steps = max(round(nominal / batch), 1)
    return {"physical_batch_requested": batch, "nominal_batch": nominal,
            "nominal_batch_explicit": nominal_batch is not None,
            "target_accumulation_steps": steps,
            "target_full_batch_samples_per_update": steps * batch,
            "warmup_epochs_requested": .05,
            "note": "Ultralytics rounds nbs/physical_batch; warmup interpolates accumulation from 1 to that target and short runs may clamp it. nbs also affects optimizer iteration estimates and weight decay. Partial batches, BatchNorm and accumulation differ from one physical batch of nbs samples; actual optimizer counters are authoritative."}


def validate_gpu_memory_gib(value):
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("gpu-memory-gib must be finite and positive")
    try:
        amount = float(value)
        scaled = amount * 1024 ** 3
        valid = math.isfinite(scaled) and scaled >= 1
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError("gpu-memory-gib must be finite and positive")
    return amount


def configure_gpu_memory_limit(torch, device, requested_gib=None):
    """Call before CUDA tensor/model/NMS allocation; physical-memory query may initialize CUDA."""
    validate_gpu_memory_gib(requested_gib)
    policy = {"requested_gib": requested_gib, "applied": False,
              "scope": "this process's PyTorch CUDA caching allocator only",
              "limitation": "Not a global VRAM reservation, total-process GPU limit or guarantee of enough free memory. Other processes and non-allocator CUDA/context allocations are outside this cap."}
    if requested_gib is None:
        return policy  # Legacy runs leave the allocator limit unchanged.
    if type(device) is not int or device < 0:
        raise ValueError("An explicit allocator cap requires a nonnegative CUDA device index")
    total = int(torch.cuda.get_device_properties(device).total_memory)
    requested_bytes = int(requested_gib * 1024 ** 3)
    if total <= 0 or requested_bytes > total:
        raise ValueError("gpu-memory-gib exceeds the selected device's physical total VRAM")
    fraction = requested_bytes / total
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=device)
    effective = float(torch.cuda.get_per_process_memory_fraction(device=device))
    if not math.isfinite(effective) or not math.isclose(effective, fraction, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("CUDA allocator did not retain the requested memory fraction")
    policy.update(applied=True, device_index=device, total_vram_bytes=total,
                  requested_limit_bytes=requested_bytes, fraction=effective,
                  effective_allocator_limit_bytes=int(total * effective))
    return policy


def _reported_identity(entry, parent, label):
    if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
            or not entry["path"] or type(entry.get("bytes")) is not int):
        raise ValueError(f"Missing {label} file identity")
    path = Path(entry["path"])
    actual = identity(path if path.is_absolute() else parent / path)
    if actual["sha256"] != entry.get("sha256") or actual["bytes"] != entry["bytes"]:
        raise ValueError(f"Changed {label} file identity")
    return actual


def verify_initialization(weights, training_report=None, *, _ancestors=frozenset()):
    """Validate initializer lineage before deserializing any checkpoint.

    A report is local provenance, not a signature or permission to load an
    untrusted downloaded pickle. Every fine-tune ancestor must terminate in the
    pinned official initializer and retain its completed export/checkpoints.
    """
    selected = identity(weights)
    if training_report is None:
        if selected["sha256"] != OFFICIAL_SHA256:
            raise ValueError("Unexpected official pretrained weight digest; do not unpickle it")
        return selected, {"kind": "official_pretrained", "checkpoint": selected,
                          "root_pretrained": selected, "ancestor_training_reports": [],
                          "optimizer_state": "fresh", "resume": False}

    path = Path(training_report).resolve(strict=True)
    if path in _ancestors or len(_ancestors) >= 16:
        raise ValueError("Cyclic or excessively deep initializer lineage")
    report_identity = identity(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or report.get("model_family") != "yolo26x-sem"
            or report.get("task") != "semantic" or report.get("class_names") != CLASS_NAMES
            or type(report.get("road_class_id")) is not int or report["road_class_id"] != 0
            or report.get("training_complete") is not True or report.get("diagnostic") is not False):
        raise ValueError("Initializer requires completed nondiagnostic two-class YOLO26x semantic training")
    source = _reported_identity(report.get("source_config"), path.parent, "parent source config")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Initializer requires best and last checkpoint provenance")
    recorded = {role: _reported_identity(checkpoints.get(role), path.parent, f"parent {role} checkpoint")
                for role in ("best", "last")}
    if selected != recorded["best"]:
        raise ValueError("Fine-tuning requires the completed report's best checkpoint")
    dataset = report.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("provenance_path"), str):
        raise ValueError("Initializer requires its complete training export provenance")
    provenance_path = Path(dataset["provenance_path"])
    if not provenance_path.is_absolute():
        provenance_path = path.parent / provenance_path
    previous_dataset, previous_source = verify_export(provenance_path.parent, source["path"])
    if (any(dataset.get(key) != value for key, value in previous_dataset.items())
            or previous_source != source):
        raise ValueError("Parent training export differs from its completed report")

    actual, history, progress = (report.get(key) for key in ("actual_args", "history", "progress"))
    if not isinstance(actual, dict) or not isinstance(history, list) or not isinstance(progress, dict):
        raise ValueError("Initializer requires actual training arguments, history and update evidence")
    epochs, batch = actual.get("epochs"), actual.get("batch")
    if (type(epochs) is not int or epochs < 1 or type(batch) is not int or batch < 1
            or actual.get("resume") is not False or actual.get("fraction") != 1.0
            or len(history) != epochs
            or any(not isinstance(row, dict) or type(row.get("epoch")) not in (int, float)
                   or row["epoch"] != index for index, row in enumerate(history, start=1))):
        raise ValueError("Initializer lacks full completed-epoch training evidence")
    batches = math.ceil(previous_dataset["counts"]["train"] / batch)
    keys = ("train_batches", "optimizer_attempts", "optimizer_updates", "amp_skipped_updates")
    if (type(report.get("batches_per_epoch")) is not int or report["batches_per_epoch"] != batches
            or any(type(progress.get(key)) is not int or progress[key] < 0 for key in keys)
            or progress["train_batches"] != batches * epochs
            or progress["optimizer_updates"] < 1
            or progress["optimizer_attempts"] > progress["train_batches"]
            or progress["optimizer_attempts"] != progress["optimizer_updates"] + progress["amp_skipped_updates"]):
        raise ValueError("Initializer lacks complete batch and successful optimizer update evidence")

    parent_weights = _reported_identity(report.get("pretrained"), path.parent, "parent initializer")
    parent_proof = report.get("initialization")
    parent_report = None
    if parent_proof is not None:
        if not isinstance(parent_proof, dict):
            raise ValueError("Invalid parent initializer lineage")
        if parent_proof.get("kind") == "completed_best_checkpoint":
            parent_report = _reported_identity(parent_proof.get("training_report"), path.parent,
                                               "ancestor training report")["path"]
        elif parent_proof.get("kind") != "official_pretrained":
            raise ValueError("Unknown parent initializer lineage")
    _, verified_parent = verify_initialization(parent_weights["path"], parent_report,
                                               _ancestors=_ancestors | {path})
    if parent_proof is not None and parent_proof != verified_parent:
        raise ValueError("Parent initializer lineage changed")
    if identity(path) != report_identity:
        raise ValueError("Parent training report changed during initialization verification")
    return selected, {"kind": "completed_best_checkpoint", "checkpoint": selected,
                      "training_report": report_identity, "parent_source_config": source,
                      "parent_dataset": previous_dataset, "parent_completed_epochs": epochs,
                      "parent_optimizer_updates": progress["optimizer_updates"],
                      "root_pretrained": verified_parent["root_pretrained"],
                      "ancestor_training_reports": [report_identity, *verified_parent["ancestor_training_reports"]],
                      "optimizer_state": "fresh", "resume": False}


def validate_finetune_model(wrapper, *, model_class=None, head_class=None):
    """Check the loaded initializer still has the canonical native two-class head."""
    if model_class is None or head_class is None:
        from ultralytics.nn.tasks import SemanticSegmentationModel
        from ultralytics.nn.modules.head import SemanticSegment
        model_class, head_class = SemanticSegmentationModel, SemanticSegment
    model = wrapper.model
    if wrapper.task != "semantic" or not isinstance(model, model_class):
        raise ValueError("Fine-tuning requires a native semantic model")
    head = model.model[-1]
    if (not isinstance(head, head_class) or head.nc != 2 or head.export or head.bake_argmax
            or model.names not in (dict(enumerate(CLASS_NAMES)), CLASS_NAMES)):
        raise ValueError("Fine-tuning requires canonical road/nonroad names and an unexported two-class head")
    yaml_stem = str(model.yaml.get("yaml_file", "")).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if model.yaml.get("scale") != "x" or yaml_stem not in {"yolo26-sem", "yolo26x-sem"}:
        raise ValueError("Fine-tuning requires the YOLO26x semantic architecture")
    return model


def configure_amp_helper(path, settings, utils):
    """Pin AMP's auxiliary detector without changing the initializer directory."""
    helper = identity(path)
    if Path(helper['path']).name != 'yolo26n.pt' or helper['sha256'] != AMP_HELPER_SHA256:
        raise ValueError('AMP helper must be the verified local yolo26n.pt')
    directory = Path(helper['path']).parent
    settings.update({'weights_dir': str(directory)})
    # Ultralytics captures this at import time; updating SETTINGS alone is insufficient.
    utils.WEIGHTS_DIR = directory
    return helper


def train(args):
    validate_parameters(args.epochs, args.image_size, args.batch, args.workers, args.lr)
    batching = batch_policy(args.batch, getattr(args, "nominal_batch", None))
    memory_gib = validate_gpu_memory_gib(getattr(args, "gpu_memory_gib", None))
    dataset, config_id = verify_export(args.dataset, args.config)
    init_training_report = getattr(args, "init_training_report", None)
    pretrained, initialization = verify_initialization(args.weights, init_training_report)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "ultralytics_settings"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden for this GPU experiment")
    # Apply the cap before NMS, AMP checks, loaded models or CUDA tensors. The
    # required physical-memory query can initialize a CUDA context itself.
    memory_policy = configure_gpu_memory_limit(torch, args.device, memory_gib)
    # Load native vision operators before the trainer's deferred AMP import.
    # Fail here with a real CUDA operation instead of discovering a broken
    # extension only after constructing the full training model.
    import torchvision
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
    settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "comet", "dvc", "neptune", "raytune", "tensorboard") if key in settings})
    # check_amp uses this already downloaded detector solely for numeric support.
    # Never resolve it from a previous training run's checkpoint directory.
    amp_helper = configure_amp_helper(Path(__file__).resolve().parents[1] / 'models/pretrained/yolo26n.pt',
                                      settings, ultralytics.utils)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    probe_boxes = torch.tensor([[0., 0., 5., 5.], [0., 0., 4., 4.]], device=args.device)
    probe_scores = torch.tensor([0.9, 0.8], device=args.device)
    if torchvision.ops.nms(probe_boxes, probe_scores, 0.5).cpu().tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS preflight failed")
    del probe_boxes, probe_scores
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic",
              "class_names": CLASS_NAMES, "road_class_id": 0, "training_complete": False,
              "diagnostic": False, "source_config": config_id, "dataset": dataset,
              "pretrained": pretrained, "initialization": initialization, "amp_helper": amp_helper,
              "optimization_state": "fresh_for_this_run", "checkpoints": {}, "history": [],
              "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "torchvision_cuda_nms_preflight": "passed", "ultralytics": ultralytics.__version__,
                              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(args.device),
                              "total_vram_bytes": torch.cuda.get_device_properties(args.device).total_memory},
              "objective": "CE + Dice + auxiliary CE; CE class weights [1,2], ignore255",
              "gpu_memory_policy": memory_policy, "batch_policy": batching,
              "automatic_batch_retry_allowed": memory_gib is None,
              "observed_accumulation_batches": {},
              "deployment_approved": False}
    counters = {"train_batches": 0, "optimizer_attempts": 0, "optimizer_updates": 0, "amp_skipped_updates": 0}
    report["progress"] = counters

    class RecordedTrainer(SemanticSegmentationTrainer):
        def _build_train_pipeline(self):
            # The installed trainer halves batch_size on first-epoch OOM before
            # calling this hook to rebuild loaders and restart the epoch. A cap
            # requests one bounded physical batch: fail before that retry.
            oom_retries = int(getattr(self, "_oom_retries", 0))
            if memory_gib is not None and (self.batch_size != args.batch or oom_retries > 0):
                report["blocked_batch_change"] = {"requested": args.batch, "attempted": self.batch_size}
                report["blocked_oom_retry_count"] = oom_retries
                raise RuntimeError("Capped training forbids automatic batch reduction/epoch retry; inspect the OOM and select a new explicit batch in a fresh run")
            return super()._build_train_pipeline()

        def set_class_weights(self):
            # Same false-road cost preference as the current TinyUNet recipe.
            # This is fixed from the training design, never fitted on obstacle tests.
            self.model.class_weights = torch.tensor([1.0, 2.0], device=self.device)

        def optimizer_step(self):
            before = self.scaler.get_scale()
            super().optimizer_step()
            counters["optimizer_attempts"] += 1
            if self.scaler.get_scale() < before:
                counters["amp_skipped_updates"] += 1
            else:
                counters["optimizer_updates"] += 1

    def progress(trainer):
        counters["train_batches"] += 1
        accumulation = int(trainer.accumulate)
        key = str(accumulation)
        report["observed_accumulation_batches"][key] = report["observed_accumulation_batches"].get(key, 0) + 1
        report["last_observed_physical_batch"] = int(trainer.batch_size)
        if counters["train_batches"] == 1 or counters["train_batches"] % 100 == 0:
            report["elapsed_seconds"] = time.perf_counter() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["current_epoch"] = trainer.epoch + 1
            report["amp_enabled"] = bool(trainer.amp)
            write_json(output / "training_report.json", report)
            print(json.dumps({"event": "gpu_training_progress", **counters,
                              "elapsed_seconds": report["elapsed_seconds"],
                              "peak_allocated_bytes": report["peak_allocated_bytes"]}), flush=True)

    def setup(trainer):
        if trainer.args.resume or trainer.start_epoch != 0:
            raise ValueError("This runner requires fresh optimizer training, never checkpoint resume")
        if trainer.data["names"] != dict(enumerate(CLASS_NAMES)):
            raise ValueError("Trainer changed semantic class mapping")
        if len(trainer.train_loader.dataset) != dataset["counts"]["train"] or len(trainer.test_loader.dataset) != dataset["counts"]["val"]:
            raise ValueError("Trainer excluded samples; a partial run cannot replace the full experiment")
        if trainer.train_loader.drop_last:
            raise ValueError("Training loader must retain its final incomplete batch")
        report["batches_per_epoch"] = len(trainer.train_loader)
        report["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        report["actual_args"] = vars(trainer.args).copy()
        report["batch_policy"]["initial_actual_physical_batch"] = int(trainer.batch_size)
        report["batch_policy"]["initial_actual_accumulation_steps"] = int(trainer.accumulate)
        report["class_weights"] = [1.0, 2.0]
        report["amp_enabled"] = bool(trainer.amp)
        write_json(output / "training_report.json", report)

    def epoch_finished(trainer):
        write_json(output / "training_report.json", report)

    try:
        write_json(output / "training_report.json", report)
        model = YOLO(str(args.weights), task="semantic")
        if model.task != "semantic" or model.model.yaml.get("scale") != "x":
            raise ValueError("Requested model is not the YOLO26x semantic architecture")
        if init_training_report is not None:
            validate_finetune_model(model)
        model.add_callback("on_pretrain_routine_end", setup)
        model.add_callback("on_train_batch_end", progress)
        model.add_callback("on_fit_epoch_end", epoch_finished)
        model.train(trainer=RecordedTrainer, data=dataset["yaml"]["path"], epochs=args.epochs,
                    imgsz=args.image_size, batch=args.batch, device=args.device, amp=True,
                    workers=args.workers, cache=False, cls_pw=0.0, optimizer="AdamW",
                    lr0=args.lr, lrf=0.1, nbs=batching["nominal_batch"], warmup_epochs=0.05,
                    seed=42, deterministic=True, project=str(output), name="fit", exist_ok=False,
                    plots=False, save=True, save_period=1, val=True, fraction=1.0,
                    patience=100, verbose=False, compile=False, resume=False)
        trainer = model.trainer
        with (Path(trainer.save_dir) / "results.csv").open(encoding="utf-8", newline="") as stream:
            report["history"] = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        if len(report["history"]) != args.epochs or counters["train_batches"] != report["batches_per_epoch"] * args.epochs:
            raise RuntimeError("Training did not complete all requested epochs and batches")
        if counters["optimizer_updates"] == 0:
            raise RuntimeError("No successful CUDA optimizer update")
        # Recheck immutable inputs after training before declaring completion.
        after_dataset, after_config = verify_export(args.dataset, args.config)
        if after_dataset != dataset or after_config != config_id:
            raise ValueError("Experiment provenance changed while training")
        after_pretrained, after_initialization = verify_initialization(args.weights, init_training_report)
        if after_pretrained != pretrained or after_initialization != initialization:
            raise ValueError("Initializer provenance changed while training")
        if identity(amp_helper['path']) != amp_helper:
            raise ValueError('AMP helper changed while training')
        report["checkpoints"] = {"best": identity(trainer.best), "last": identity(trainer.last)}
        report["training_complete"] = True
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / "training_report.json", report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "config", "weights", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--init-training-report", type=Path,
                        help="Completed YOLO26x-sem report for --weights=its best.pt; starts a fresh optimizer, not resume")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--nominal-batch", type=int,
                        help="Ultralytics nbs >= physical --batch; default equals --batch. Warmup/partial batches change accumulation; not exact large-batch equivalence")
    parser.add_argument("--gpu-memory-gib", type=float,
                        help="Optional PyTorch per-process CUDA allocator cap in GiB, applied before CUDA tensors; does not reserve memory or cap other processes. Disables automatic batch-reduction retries")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main():
    report = train(build_parser().parse_args())
    print(json.dumps({"training_complete": report["training_complete"], "checkpoints": report["checkpoints"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
