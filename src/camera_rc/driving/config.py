"""Explicit runtime configuration; paths are relative to the configuration file."""
import copy
import json
import math
import re
import stat
from pathlib import Path

DEFAULTS = {
    "device": "auto",
    "hardware": {"profile": "generic_discrete", "memory_architecture": "discrete"},
    "models_validated_for_control": False,
    "perception": {"segmentation_backend": "native", "drivable_class_ids": [0], "half": True, "allow_diagnostic": False,
                   "detection_image_size": 320, "detection_confidence": 0.35},
    "source": {"kind": "camera", "index": 0, "width": 640, "height": 360, "fps": 30},
    "runtime": {"control_hz": 50, "perception_hz": 20, "frame_timeout_s": 0.25,
                "plan_timeout_s": 0.2, "max_processing_s": 0.15,
                "gpu_budget_gib": 4, "torch_allocator_cap_gib": 3,
                "ram_budget_gib": 16, "process_ram_limit_gib": 12,
                "system_available_min_gib": 1},
    "planner": {"horizon_m": 1.3, "sample_step_m": 0.05, "vehicle_width_m": 0.22,
                "vehicle_length_m": 0.35, "wheelbase_m": 0.26, "max_steering_rad": 0.45,
                "max_speed_mps": 0.4, "max_deceleration_mps2": 0.8,
                "reaction_time_s": 0.5, "clearance_m": 0.06, "min_drivable_probability": 0.65},
    "controller": {"wheelbase_m": 0.26, "max_steering_rad": 0.45,
                   "max_speed_mps": 0.4, "max_accel_mps2": 0.3, "max_decel_mps2": 0.8,
                   "max_throttle": 0.3, "state_timeout_s": 0.25},
    "transport": {"kind": "dry_run", "baudrate": 230400, "command_ttl_ms": 150,
                  "state_timeout_s": 0.25, "write_timeout_s": 0.05,
                  "max_steering_rad": 0.45, "max_throttle": 0.3,
                  "calibration_confirmed": False},
    "route_hint": "straight",
}


def load_runtime_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    for section, keys in (("perception", ("segmentation_checkpoint", "detection_checkpoint", "segmentation_training_report", "yolo_settings_dir")),
                          ("source", ("path",))):
        for key in keys:
            if cfg.get(section, {}).get(key):
                p = Path(cfg[section][key]).expanduser()
                cfg[section][key] = str((p if p.is_absolute() else path.parent / p).resolve())
    def read_calibration(value):
        if not isinstance(value, str) or not value:
            raise ValueError("A measured camera/ground calibration file is required")
        target = Path(value).expanduser()
        target = target if target.is_absolute() else path.parent / target
        return json.loads(target.read_text(encoding="utf-8-sig")), str(target.resolve())

    if cfg["source"]["kind"] in ("multicamera", "multireplay"):
        validate_camera_group(cfg["source"])
        cfg["calibrations"], cfg["calibration_paths"] = {}, {}
        for camera in cfg["source"]["cameras"]:
            name = camera["name"]
            calibration, location = read_calibration(camera.get("calibration"))
            cfg["calibrations"][name], cfg["calibration_paths"][name] = calibration, location
    else:
        cfg["calibration"], cfg["calibration_path"] = read_calibration(cfg.get("calibration"))
    validate_runtime_config(cfg)
    return cfg


def camera_device_identity(device, *, require_present=False):
    """Video files/URLs must never masquerade as a fresh live camera feed."""
    if isinstance(device, bool):
        raise ValueError("Live camera requires a physical camera index or /dev path")
    if isinstance(device, int) and device >= 0:
        # Numeric indices must undergo the same physical-device checks as paths.
        device = f"/dev/video{device}"
    if not isinstance(device, str) or not device.startswith("/dev/") or ".." in Path(device).parts:
        raise ValueError("Live camera requires a physical camera index or /dev path; use replay for recorded input")
    path = Path(device)
    if require_present or path.exists():
        resolved = path.resolve(strict=True)
        if require_present and (not re.fullmatch(r"/dev/video\d+", resolved.as_posix())
                                or not stat.S_ISCHR(resolved.stat().st_mode)):
            raise ValueError("Physical output requires a real V4L2 /dev/videoN character device")
        return resolved.as_posix()
    # Still normalize a missing /dev/videoN path against numeric N during preparation.
    match = re.fullmatch(r"/dev/video(\d+)", device)
    return f"/dev/video{int(match.group(1))}" if match else device


def validate_camera_group(source):
    cameras = source.get("cameras")
    if not isinstance(cameras, list) or not 2 <= len(cameras) <= 8:
        raise ValueError("Multiview source requires 2 to 8 named cameras")
    names, devices = set(), set()
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("Each camera must be an object")
        name = camera.get("name")
        if not isinstance(name, str) or not name or not name.isascii() or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("Camera names must be unique ASCII letters/digits/underscores")
        names.add(name)
        if source["kind"] == "multicamera":
            device = camera.get("device", camera.get("index"))
            if camera.get("kind", "camera") != "camera":
                raise ValueError("Live camera groups cannot contain replay sources")
            identity = camera_device_identity(device)
            if identity in devices:
                raise ValueError("Live cameras must use distinct devices")
            devices.add(identity)
        for field in ("width", "height", "fps"):
            value = camera.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"camera.{field} must be finite and positive")
    skew = source.get("max_skew_s")
    if isinstance(skew, bool) or not isinstance(skew, (int, float)) or not math.isfinite(skew) or skew <= 0:
        raise ValueError("source.max_skew_s must be finite and positive")


def validate_runtime_config(cfg):
    if cfg["perception"].get("segmentation_backend", "native") not in ("native", "yolo26x_semantic"):
        raise ValueError("segmentation_backend must be native or yolo26x_semantic")
    kind = cfg["source"]["kind"]
    if kind not in ("camera", "replay", "multicamera", "multireplay"):
        raise ValueError("Unsupported source.kind")
    if kind == "camera":
        camera_device_identity(cfg["source"].get("device", cfg["source"].get("index", 0)))
    if kind in ("multicamera", "multireplay"):
        validate_camera_group(cfg["source"])
        if cfg["source"]["max_skew_s"] >= cfg["runtime"]["plan_timeout_s"]:
            raise ValueError("Multiview skew limit must be below plan expiry")
    architecture = cfg["hardware"]["memory_architecture"]
    if architecture not in ("discrete", "shared"):
        raise ValueError("hardware.memory_architecture must be discrete or shared")
    for key, value in cfg["runtime"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"runtime.{key} must be finite and positive")
    if cfg["runtime"]["torch_allocator_cap_gib"] >= cfg["runtime"]["gpu_budget_gib"]:
        raise ValueError("Leave GPU memory headroom outside the PyTorch allocator")
    if cfg["runtime"]["process_ram_limit_gib"] >= cfg["runtime"]["ram_budget_gib"]:
        raise ValueError("Leave system RAM for the OS and other processes")
    if cfg["runtime"]["system_available_min_gib"] >= cfg["runtime"]["ram_budget_gib"]:
        raise ValueError("System available memory floor must be below the RAM budget")
    if architecture == "shared" and cfg["runtime"]["gpu_budget_gib"] > cfg["runtime"]["ram_budget_gib"]:
        raise ValueError("Shared GPU memory budget cannot exceed the shared RAM budget")
    for field in ("wheelbase_m", "max_steering_rad", "max_speed_mps"):
        if cfg["planner"][field] != cfg["controller"][field]:
            raise ValueError(f"Planner and controller {field} must agree")
    for field in ("max_steering_rad", "max_throttle"):
        if cfg["transport"][field] != cfg["controller"][field]:
            raise ValueError(f"Transport and controller {field} must agree")
    if cfg["planner"]["max_deceleration_mps2"] > cfg["controller"]["max_decel_mps2"]:
        raise ValueError("Planner cannot assume stronger braking than the controller")
    delay = (cfg["runtime"]["plan_timeout_s"] + cfg["transport"]["command_ttl_ms"] / 1000
             + 1 / cfg["runtime"]["control_hz"] + cfg["transport"]["write_timeout_s"])
    if cfg["planner"]["reaction_time_s"] < delay:
        raise ValueError("Planner reaction_time_s must cover plan expiry, control tick, write timeout and MCU TTL")
    if cfg["route_hint"] not in ("straight", "left", "right", "stop"):
        raise ValueError("route_hint must be straight, left, right or stop")
    if cfg["transport"]["kind"] not in ("dry_run", "serial"):
        raise ValueError("transport.kind must be dry_run or serial")


def check_actuation(cfg, actuate):
    serial = cfg["transport"]["kind"] == "serial"
    if serial != bool(actuate):
        raise ValueError("Physical serial output requires BOTH transport.kind=serial and --actuate; dry-run uses neither")
    if not serial:
        return
    if cfg["perception"].get("segmentation_backend", "native") == "yolo26x_semantic":
        raise ValueError("YOLO semantic backend is dry-run/replay only; a separate offline readiness gate and vehicle validation are not implemented")
    if cfg["source"]["kind"] not in ("camera", "multicamera"):
        raise ValueError("Recorded/demo inputs cannot actuate a vehicle")
    if cfg["source"]["kind"] == "multicamera" and len(cfg["source"].get("cameras", [])) > 6:
        raise ValueError("Seven/eight-camera configurations are research-only: use dry_run; physical output is unsupported")
    if cfg["source"]["kind"] == "multicamera" and cfg["source"].get("capture_timing_validated") is not True:
        raise ValueError("Measure exposure timing and USB buffering before enabling multiview actuation")
    sources = cfg["source"]["cameras"] if cfg["source"]["kind"] == "multicamera" else [cfg["source"]]
    identities = [camera_device_identity(source.get("device", source.get("index", 0)), require_present=True)
                  for source in sources]
    if len(set(identities)) != len(identities):
        raise ValueError("Physical output requires distinct live camera devices")
    if cfg.get("models_validated_for_control") is not True:
        raise ValueError("Validate model behaviour on your vehicle before setting models_validated_for_control=true")
    if cfg["perception"].get("allow_diagnostic"):
        raise ValueError("Diagnostic checkpoints cannot actuate a vehicle")
    calibrations = cfg.get("calibrations", {"front": cfg.get("calibration", {})})
    if any(item.get("diagnostic_only") for item in calibrations.values()):
        raise ValueError("Synthetic calibration cannot actuate a vehicle")
    if cfg["planner"].get("assumed_clearance_m", 0) != 0:
        raise ValueError("Unobserved ground cannot be assumed clear for physical actuation")
    if cfg["transport"].get("calibration_confirmed") is not True:
        raise ValueError("Measure steering, propulsion, brakes and telemetry before confirming actuator calibration")
