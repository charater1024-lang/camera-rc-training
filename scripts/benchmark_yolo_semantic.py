"""Compare trained YOLO26x-sem and TinyUNet latency on recorded RGB images.

No training, cameras, control output, accuracy measurement, or hardware
emulation is performed. Models run sequentially, one shared instance at a time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from camera_rc.config import load_config
from camera_rc.driving.resources import ResourceMonitor
from camera_rc.models import build_model
from camera_rc.training import _check_compatible, _read_checkpoint


CLASS_NAMES = {0: "road_surface", 1: "not_road_surface"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_options(iterations, warmup, input_size, views, precision, device, memory_budget_gib):
    for value, name, minimum, maximum in ((iterations, "iterations", 1, 10000), (warmup, "warmup", 0, 1000)):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    if (not isinstance(input_size, (tuple, list)) or len(input_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or not 32 <= v <= 4096 or v % 32 for v in input_size)):
        raise ValueError("input_size must be [height,width], multiples of 32 from 32 to 4096")
    if isinstance(views, bool) or views not in (1, 8):
        raise ValueError("views must be 1 or 8")
    if device not in ("cpu", "cuda") or precision not in ("fp32", "fp16"):
        raise ValueError("device must be cpu/cuda and precision fp32/fp16")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU supports fp32 only")
    if memory_budget_gib is not None and (isinstance(memory_budget_gib, bool) or memory_budget_gib not in (3, 7)):
        raise ValueError("memory_budget_gib must be omitted, 3, or 7")
    if device == "cpu" and memory_budget_gib is not None:
        raise ValueError("A CUDA allocator budget cannot be applied on CPU")


def latency_summary(values, views):
    if (not values or isinstance(views, bool) or views not in (1, 8)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values)):
        raise ValueError("Latency samples must be nonempty, finite positive milliseconds and views 1/8")
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return {"measured_cycles": len(values), "views_per_cycle": views,
            "mean_cycle_ms": mean, "p50_cycle_ms": float(np.percentile(values, 50)),
            "p95_cycle_ms": float(np.percentile(values, 95)), "p99_cycle_ms": float(np.percentile(values, 99)),
            "max_cycle_ms": float(values.max()), "images_per_second": views * 1000 / mean,
            "camera_work_cycles_per_second": 1000 / mean,
            "mean_ms_per_image_amortized": mean / views,
            "percentile_method": "numpy.percentile linear interpolation over measured complete cycles"}


def _evidence_path(value, directory):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Evidence requires a nonempty local path")
    path = Path(value)
    return (path if path.is_absolute() else directory / path).resolve(strict=True)


def verify_training_report(training_report, yolo_checkpoint):
    report_path = Path(training_report).resolve(strict=True)
    report = read_json(report_path)
    if (not isinstance(report, dict) or report.get("model_family") != "yolo26x-sem"
            or report.get("task") != "semantic" or report.get("training_complete") is not True
            or report.get("diagnostic") is not False):
        raise ValueError("Require completed, nondiagnostic yolo26x-sem semantic training report")
    source = report.get("source_config")
    if not isinstance(source, dict):
        raise ValueError("Training report requires source_config provenance")
    source_id = identity(_evidence_path(source.get("path"), report_path.parent))
    if source_id["sha256"] != source.get("sha256", source.get("sha")):
        raise ValueError("Training source_config SHA256 differs from report")
    checkpoint_id = identity(yolo_checkpoint)
    matched = None
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Training report requires checkpoints.best/last provenance")
    for name in ("best", "last"):
        entry = checkpoints.get(name)
        if not isinstance(entry, dict):
            continue
        if _evidence_path(entry.get("path"), report_path.parent) != Path(checkpoint_id["path"]):
            continue
        if entry.get("sha256") != checkpoint_id["sha256"] or entry.get("bytes") != checkpoint_id["bytes"]:
            raise ValueError("Selected YOLO checkpoint SHA256/size differs from training report")
        matched = name
        break
    if matched is None:
        raise ValueError("YOLO checkpoint is not a recorded best/last trained artifact")
    return {"training_report": identity(report_path), "source_config": source_id,
            "checkpoint": checkpoint_id, "selected_checkpoint": matched,
            "training_complete": True, "diagnostic": False}


def validate_yolo_contract(wrapper, model_type, head_type):
    network = wrapper.model
    if getattr(wrapper, "task", None) != "semantic" or not isinstance(network, model_type):
        raise ValueError("Expected a native SemanticSegmentationModel")
    if not isinstance(getattr(network, "yaml", None), dict) or network.yaml.get("scale") != "x":
        raise ValueError("Expected YOLO26x semantic scale x")
    # Official checkpoints retain a scaled or base YAML filename. Reject an
    # explicitly different YOLO generation; a custom filename alone is not
    # reliable architecture evidence (the training report is checked separately).
    yaml_file = network.yaml.get("yaml_file")
    stem = str(yaml_file).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if yaml_file else ""
    if re.fullmatch(r"yolo\d+[nsmlx]?-sem", stem) and stem not in {"yolo26-sem", "yolo26x-sem"}:
        raise ValueError("Checkpoint YAML declares a different YOLO semantic generation or scale")
    head = network.model[-1]
    if (not isinstance(head, head_type) or getattr(head, "nc", None) != 2
            or getattr(head, "export", False) or getattr(head, "bake_argmax", False)):
        raise ValueError("Require an unexported two-class semantic logits head")
    names = getattr(network, "names", None)
    names = dict(enumerate(names)) if isinstance(names, (tuple, list)) else names
    if not isinstance(names, dict) or {str(k): v for k, v in names.items()} != {str(k): v for k, v in CLASS_NAMES.items()}:
        raise ValueError("Checkpoint class names must bind 0=road_surface and 1=not_road_surface")
    return network


def load_yolo(checkpoint):
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.nn.modules import SemanticSegment
    settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "clearml", "neptune", "comet") if key in settings})
    # Existence and training provenance are checked before this local load.
    wrapper = YOLO(str(Path(checkpoint).resolve(strict=True)), task="semantic", verbose=False)
    network = validate_yolo_contract(wrapper, SemanticSegmentationModel, SemanticSegment)
    return network.float().eval(), {"ultralytics_version": ultralytics.__version__,
                                   "model_class": type(network).__name__, "scale": network.yaml["scale"],
                                   "architecture_yaml_file": network.yaml.get("yaml_file"),
                                   "generation_evidence": "completed yolo26x-sem training provenance; explicit official YAML generation checked when present",
                                   "head_class": type(network.model[-1]).__name__,
                                   "fusion": "none; trained native PyTorch model"}


def load_native(config, checkpoint):
    stored = _read_checkpoint(Path(checkpoint))
    if stored.get("diagnostic") is not False or stored.get("epoch_complete") is False:
        raise ValueError("Native comparison requires a nondiagnostic completed-epoch checkpoint")
    _check_compatible(config, stored.get("config", {}))
    if (config.get("task") != "segmentation" or config.get("model", {}).get("num_classes") != 2
            or config.get("data", {}).get("label_space_id") != "road_surface_v1"):
        raise ValueError("Native model must use the two-class road_surface_v1 contract")
    network = build_model(config)
    network.load_state_dict(stored["model"], strict=True)
    return network.float().eval(), {"model_class": type(network).__name__, "fusion": "none; native PyTorch model",
                                   "checkpoint": identity(checkpoint)}


def select_images(config, manifest, views):
    if manifest is not None:
        paths = [Path(manifest).resolve(strict=True)]
    else:
        data = config["data"]
        specs = data.get("val_manifests", [data.get("val_manifest")])
        paths = [Path(item["manifest"] if isinstance(item, dict) else item).resolve(strict=True) for item in specs]
    cached, records, seen = [], [], set()
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("synthetic") is True or str(row.get("dataset_id", "")).startswith("vkitti"):
                    continue
                image_path = _evidence_path(row.get("image"), path.parent)
                if image_path in seen:
                    continue
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB").copy()
                cached.append(image)
                records.append({"manifest": str(path), "image": identity(image_path),
                                "dataset_id": row.get("dataset_id"), "sample_id": row.get("sample_id"),
                                "source_size_hw": [image.height, image.width],
                                "synthetic_declared": row.get("synthetic"),
                                "real_image_note": "Synthetic-marked/VKITTI rows excluded; dataset annotations are not independently authenticated"})
                seen.add(image_path)
                if len(cached) == views:
                    return cached, records
    raise ValueError(f"Need {views} distinct recorded RGB images; found {len(cached)}")


def prepare_image(image, input_size, device, precision):
    height, width = input_size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device=device,
                dtype=torch.float16 if precision == "fp16" else torch.float32, non_blocking=False)


def prepare_batch(images, input_size, device, precision):
    """Resize on the host, stack once, and make one host-to-device transfer."""
    height, width = input_size
    arrays = [np.asarray(image.resize((width, height), Image.Resampling.BILINEAR),
                         dtype=np.float32).transpose(2, 0, 1) for image in images]
    values = np.stack(arrays, axis=0) / 255.0
    return torch.from_numpy(values).to(device=device,
                dtype=torch.float16 if precision == "fp16" else torch.float32, non_blocking=False)


def road_probability(network, tensor, input_size, *, return_raw_metadata=False):
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
            or tensor.shape[0] < 1 or tensor.shape[1] != 3):
        raise ValueError("Expected RGB [batch,3,H,W] input for raw semantic logits")
    logits = network(tensor)
    if (not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits)
            or logits.ndim != 4 or logits.shape[:2] != (tensor.shape[0], 2)):
        raise ValueError("Expected raw [batch,2,h,w] logits matching input batch, not class IDs or predictor Results")
    raw_metadata = {"shape": list(logits.shape), "dtype": str(logits.dtype)}
    logits = logits.float()
    if tuple(logits.shape[-2:]) != tuple(input_size):
        logits = F.interpolate(logits, size=tuple(input_size), mode="bilinear", align_corners=False)
    probability = logits.softmax(dim=1)[:, 0]
    return (probability, raw_metadata) if return_raw_metadata else probability


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_snapshot(device):
    """Allocator counters only; CUDA context/library memory is outside this scope."""
    if device.type != "cuda":
        return {name: None for name in ("torch_peak_allocated_gib", "torch_peak_reserved_gib",
                                        "torch_allocated_at_snapshot_gib", "torch_reserved_at_snapshot_gib")}
    return {"torch_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "torch_allocated_at_snapshot_gib": torch.cuda.memory_allocated(device) / 2**30,
            "torch_reserved_at_snapshot_gib": torch.cuda.memory_reserved(device) / 2**30}


def begin_mode_memory(device):
    # A preceding mode's local inputs/outputs have returned and can be freed.
    # The shared model remains resident, so its memory is included in each peak.
    if device.type == "cuda":
        _sync(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = cuda_memory_snapshot(device)
    return {"torch_allocated_at_mode_start_gib": start["torch_allocated_at_snapshot_gib"],
            "torch_reserved_at_mode_start_gib": start["torch_reserved_at_snapshot_gib"]}


def include_cuda_peak(cumulative, snapshot):
    for name in ("torch_peak_allocated_gib", "torch_peak_reserved_gib"):
        value = snapshot[name]
        if value is not None:
            cumulative[name] = max(cumulative.get(name) or 0.0, value)


def measure_mode(network, cached, *, device, precision, input_size, iterations, warmup, mode, monitor):
    if mode not in ("model_only", "host_to_road_map", "batched_model_only", "batched_host_to_road_map"):
        raise ValueError("Unknown timing mode")
    batched = mode.startswith("batched_")
    host = mode.endswith("host_to_road_map")
    if batched and len(cached) != 8:
        raise ValueError("Batched comparison requires exactly eight recorded views")
    prepared = [prepare_image(image, input_size, device, precision) for image in cached] if mode == "model_only" else None
    prepared_batch = prepare_batch(cached, input_size, device, precision) if mode == "batched_model_only" else None
    elapsed, result = [], None
    with torch.inference_mode():
        for index in range(warmup + iterations):
            outputs = []
            _sync(device)
            started = time.perf_counter()
            if batched:
                tensor = prepared_batch if prepared_batch is not None else prepare_batch(cached, input_size, device, precision)
                result, raw_metadata = road_probability(network, tensor, input_size, return_raw_metadata=True)
                if host:
                    result = result.to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()
                outputs.append(result)
            else:
                for view, image in enumerate(cached):
                    tensor = prepared[view] if prepared is not None else prepare_image(image, input_size, device, precision)
                    result, raw_metadata = road_probability(network, tensor, input_size, return_raw_metadata=True)
                    if host:
                        result = result.to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()
                    outputs.append(result)
            _sync(device)
            duration = (time.perf_counter() - started) * 1000
            # Output checks and resource sampling are not part of either clock.
            if any(not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)) for value in outputs):
                raise FloatingPointError("Road probability contains nonfinite or out-of-range values")
            monitor.sample()
            if index >= warmup:
                elapsed.append(duration)
    return {**latency_summary(elapsed, len(cached)), "output_shape": list(result.shape),
            "output_dtype": str(result.dtype),
            "output_bytes_per_image": result.numel() * result.element_size() // result.shape[0],
            "output_bytes_per_cycle": sum(value.numel() * value.element_size() for value in outputs),
            "batch_size_per_call": len(cached) if batched else 1,
            "model_calls_per_cycle": 1 if batched else len(cached),
            "execution": "one shared model; batch of eight recorded frames, no camera synchronization or cross-view fusion" if batched else "one shared model; sequential batch-one calls",
            "raw_model_logits": raw_metadata}


def nvidia_smi_snapshot(expected_uuid):
    """Optional whole-device telemetry; never pretend it is per-model memory."""
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        rows = []
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 5:
                rows.append(dict(zip(("index", "uuid", "name", "total_mib", "used_mib"), fields)))
        matched = [row for row in rows if expected_uuid is not None and row["uuid"] == str(expected_uuid)]
        return {"available": bool(matched), "matched_device": matched[0] if matched else None,
                "scope": "nvidia-smi whole-device snapshot; includes other processes, not dedicated to this model"}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": type(error).__name__}


def atomic_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-volume hard-link publication is atomic and refuses to replace an
        # existing report, even if another process wins the path concurrently.
        try:
            os.link(temporary, output)
        except PermissionError:
            if os.name != "nt":
                raise
            # Windows rename on the same volume is atomic and fails if the
            # destination exists. Unlike POSIX rename, it never replaces it.
            os.rename(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def benchmark(yolo_checkpoint, training_report, native_config, native_checkpoint, output, *,
              images_manifest=None, iterations=50, warmup=10, input_size=(192, 320),
              views=1, precision="fp16", memory_budget_gib=None, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {"schema_version": 1, "status": "failed", "models": {}, "physical_output": False,
              "accuracy_evaluated": False, "driving_validated": False, "evidence_for_80kmh": False,
              "target_hardware_emulated": False, "training_performed": False,
              "requested_inputs": {"yolo_checkpoint": str(yolo_checkpoint), "training_report": str(training_report),
                                   "native_config": str(native_config), "native_checkpoint": str(native_checkpoint),
                                   "images_manifest": str(images_manifest) if images_manifest is not None else None},
              "scope": "Recorded-image latency experiment; one shared model processes views sequentially, plus one batch of eight when views=8. Different recorded frames are not synchronized cameras and no cross-view fusion is performed.",
              "excluded": ["disk image decoding", "camera exposure/capture", "USB", "rectification", "tracking", "planning", "actuation"],
              "timing_modes": {"model_only": "Preloaded CUDA/CPU RGB tensor through native forward, logit resizing and FP32 softmax. No H2D or D2H.",
                               "host_to_road_map": "Cached host PIL RGB through bilinear resize, NumPy/Torch copy, H2D, forward, logit resize, softmax and float32 road-map D2H. Disk and cameras excluded.",
                               "batched_model_only": "Views=8 only: a pre-stacked, preloaded RGB batch through one forward, logit resize and FP32 softmax. Stacking, H2D and D2H excluded.",
                               "batched_host_to_road_map": "Views=8 only: cached host PIL resize, np.stack, one H2D, one batch forward, logit resize, softmax and batch float32 road-map D2H. No camera capture or fusion."},
              "limits": ["A memory cap on an RTX 3080 Laptop does not emulate GTX 1650 or another GPU's compute, bandwidth, power or latency.",
                         "Allocator limits do not cover every CUDA/library allocation or guarantee prevention of OOM.",
                         "Latency percentiles are descriptive samples, not a worst-case or deadline guarantee.",
                         "Road probability is not an obstacle-free-space or safe-driving certificate."]}
    cached = []
    try:
        validate_options(iterations, warmup, input_size, views, precision, device, memory_budget_gib)
        proof = verify_training_report(training_report, yolo_checkpoint)
        config = load_config(native_config)
        report["provenance"] = {"yolo": proof, "native_config": identity(native_config),
                                "native_checkpoint": identity(native_checkpoint)}
        cached, records = select_images(config, images_manifest, views)
        report.update(input_images=records, input_size_hw=list(input_size), views=views,
                      precision=precision, iterations=iterations, warmup_cycles_per_mode=warmup,
                      input_shape_per_call=[1, 3, *input_size], shared_instances_per_model=1,
                      batched_input_shape_per_call=[8, 3, *input_size] if views == 8 else None,
                      outputs_retained_until_cycle_check=views,
                      same_input_size_as_native_training=list(input_size) == config["data"].get("image_size"),
                      label_space_id="road_surface_v1", class_names=CLASS_NAMES,
                      road_class_id=0, probability="bilinear logits to input size, then FP32 softmax channel 0",
                      torch_version=torch.__version__, pillow_version=importlib.metadata.version("Pillow"))
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        resolved = torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")
        torch.set_num_threads(2)
        properties = torch.cuda.get_device_properties(resolved) if device == "cuda" else None
        total = properties.total_memory / 2**30 if properties else None
        if memory_budget_gib is not None:
            if memory_budget_gib >= total:
                raise ValueError("Requested allocator budget must be below actual GPU memory")
            torch.cuda.set_per_process_memory_fraction(memory_budget_gib / total, resolved)
        report["hardware"] = {"device": str(resolved), "actual_gpu_name": properties.name if properties else None,
                              "actual_gpu_memory_gib": total, "cuda_version": torch.version.cuda,
                              "cpu_intraop_threads": torch.get_num_threads(),
                              "requested_allocator_budget_gib": memory_budget_gib,
                              "budget_interpretation": "3/7 GiB allocator caps reserve nominal headroom within 4/8 GiB; other memory is not capped",
                              "nvidia_smi_before": nvidia_smi_snapshot(getattr(properties, "uuid", None)) if properties else None}
        settings_dir = output.parent / (output.stem + "_ultralytics_settings")
        settings_dir.mkdir(parents=True, exist_ok=False)
        os.environ.update(YOLO_CONFIG_DIR=str(settings_dir), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
        for name in ("yolo26x_sem", "tinyunet"):
            network = monitor = None
            current_mode = "load_model"
            cumulative_cuda_peaks = {"torch_peak_allocated_gib": None, "torch_peak_reserved_gib": None}
            entry = {"status": "failed", "out_of_memory": False}
            report["models"][name] = entry
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(resolved)
                monitor = ResourceMonitor(str(resolved), "discrete")
                network, metadata = load_yolo(yolo_checkpoint) if name == "yolo26x_sem" else load_native(config, native_checkpoint)
                if any(not torch.isfinite(value).all() for value in network.state_dict().values()):
                    raise ValueError("Checkpoint contains nonfinite model values")
                entry.update(metadata, parameters=sum(p.numel() for p in network.parameters()))
                # Cast during the host-to-device move; do not first allocate an
                # FP32 CUDA model and count it as an FP16 loading requirement.
                network = network.to(device=resolved,
                                     dtype=torch.float16 if precision == "fp16" else torch.float32).eval()
                entry["state_tensor_bytes"] = sum(v.numel() * v.element_size() for v in network.state_dict().values())
                entry["model_load_cuda_memory"] = {**cuda_memory_snapshot(resolved),
                    "scope": "Model loading and direct requested-dtype device transfer; before timing inputs are prepared"}
                include_cuda_peak(cumulative_cuda_peaks, entry["model_load_cuda_memory"])
                modes = ["model_only", "host_to_road_map"]
                if views == 8:
                    modes.extend(("batched_model_only", "batched_host_to_road_map"))
                entry["modes_requested"] = modes
                for mode in modes:
                    current_mode = mode
                    memory_start = begin_mode_memory(resolved)
                    try:
                        entry[mode] = measure_mode(network, cached, device=resolved, precision=precision,
                                                  input_size=input_size, iterations=iterations, warmup=warmup,
                                                  mode=mode, monitor=monitor)
                    finally:
                        memory = {**memory_start, **cuda_memory_snapshot(resolved),
                            "scope": "Resident shared model plus this mode's input preparation, warmup, measured cycles and output checks; previous mode cache released and allocator peaks reset",
                            "allocator_peaks_reset": resolved.type == "cuda"}
                        include_cuda_peak(cumulative_cuda_peaks, memory)
                        if mode in entry:
                            entry[mode]["cuda_memory"] = memory
                        else:
                            entry["failed_mode_cuda_memory"] = memory
                entry["status"] = "completed"
            except Exception as error:
                entry.update(error_type=type(error).__name__, error=str(error),
                             failed_mode=current_mode,
                             out_of_memory=isinstance(error, torch.cuda.OutOfMemoryError))
            finally:
                if monitor is not None:
                    try:
                        entry["resources"] = monitor.report()
                        include_cuda_peak(cumulative_cuda_peaks, cuda_memory_snapshot(resolved))
                        entry["resources"].update(cumulative_cuda_peaks,
                            torch_peak_scope="Maximum of separately reset model-load and all attempted timing-mode allocator peaks; includes warmup and output checks",
                            sampled_peak_scope="Cumulative ResourceMonitor samples from before model loading through all attempted timing modes; whole-device usage includes other processes")
                    except Exception as error:
                        entry["resource_report_error"] = type(error).__name__ + ": " + str(error)
                del network, monitor
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        completed = [entry["status"] == "completed" for entry in report["models"].values()]
        report["status"] = "completed" if all(completed) else "partial_failure" if any(completed) else "failed"
        if properties:
            report["hardware"]["nvidia_smi_after"] = nvidia_smi_snapshot(getattr(properties, "uuid", None))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        for image in cached:
            image.close()
        atomic_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("yolo-checkpoint", "training-report", "native-config", "native-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--input-size", type=int, nargs=2, default=(192, 320), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--views", type=int, choices=(1, 8), default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--memory-budget-gib", type=int, choices=(3, 7))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = benchmark(**vars(args))
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "physical_output": False}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
