"""Read-only software compatibility probes; never opens a camera or serial port.

CUDA is required by default. --allow-cpu permits a CPU-only diagnostic and never
reports that run as GPU verification. Imports may initialize CUDA or temporary
library settings, but no packages, drivers or persistent settings are installed.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile


VERSION_BOUNDS = {
    "torch": ((2, 7, 0), (3, 0, 0)),
    "torchvision": ((0, 22, 0), (1, 0, 0)),
    "numpy": ((1, 26, 0), (3, 0, 0)),
}


def release_tuple(version):
    """NVIDIA prerelease/local build tags are compared by release tuple only."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if not match:
        raise ValueError(f"Cannot parse release version: {version!r}")
    return tuple(int(part or 0) for part in match.groups())


def version_satisfies(name, version):
    lower, upper = VERSION_BOUNDS[name]
    return lower <= release_tuple(version) < upper


def probe_numpy_torch(torch, numpy, device):
    values = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    result = (torch.from_numpy(values).to(device) + 1).sum()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    value = float(result.item())
    if value != 21.0:
        raise ValueError(f"Unexpected NumPy/Torch arithmetic result: {value}")
    return {"device": str(result.device), "result": value}


def probe_opencv(cv2, numpy):
    bgr = numpy.asarray([[[10, 20, 30]]], dtype=numpy.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not numpy.array_equal(rgb, numpy.asarray([[[30, 20, 10]]], dtype=numpy.uint8)):
        raise ValueError("OpenCV BGR-to-RGB conversion returned unexpected pixels")
    return {"result_rgb": rgb.tolist()}


def probe_nms(torch, torchvision, device):
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=device)
    kept = torchvision.ops.nms(boxes, scores, 0.5)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    result = kept.cpu().tolist()
    if result != [0, 2]:
        raise ValueError(f"Unexpected torchvision NMS result: {result}")
    return {"device": device, "kept_indices": result}


@contextlib.contextmanager
def temporary_ultralytics_settings():
    """Avoid changing the user's default Ultralytics settings on import."""
    names = ("YOLO_CONFIG_DIR", "YOLO_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="camera-rc-environment-") as directory:
        os.environ["YOLO_CONFIG_DIR"] = directory
        os.environ["YOLO_OFFLINE"] = "true"
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _check(report, name, action):
    try:
        details = action()
        report["checks"].append({"name": name, "passed": True, "details": details})
        return True
    except Exception as exc:
        report["checks"].append({"name": name, "passed": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
        return False


def _finalize(report):
    report["failures"] = [item["name"] for item in report["checks"] if not item["passed"]]
    report["success"] = not report["failures"]
    return report


def inspect_environment(*, require_cuda=True, with_detection=False, importer=importlib.import_module):
    report = {
        "validation_scope": "software_compatibility_only",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "require_cuda": bool(require_cuda), "with_detection": bool(with_detection),
        "versions": {}, "checks": [], "cuda_available": None, "cuda_device": None,
        "gpu_execution_verified": False, "detection_gpu_execution_verified": False,
        "camera_opened": False, "serial_port_opened": False,
        "linux_device_access_verified": False, "runtime_timing_verified": False,
        "vehicle_control_verified": False,
        "notes": [
            "This is not a memory/FPS benchmark, camera/serial test, or vehicle validation.",
            "Linux camera/serial permissions, JetPack compatibility and physical device operation remain unverified.",
            "Version checks compare release tuples; prerelease/local NVIDIA tags are allowed, not proof of binary compatibility.",
            "CUDA-visible memory on Jetson may be shared system DRAM, not separate dedicated VRAM.",
            "No packages, drivers or persistent system settings are installed or changed.",
        ],
    }
    _check(report, "python_version", lambda: _require(sys.version_info >= (3, 10), "Python >=3.10 required"))
    modules = {}
    # Third-party import chatter must not corrupt the machine-readable stdout.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        for name in ("numpy", "torch", "torchvision", "cv2", "serial", "psutil"):
            def load(name=name):
                module = importer(name)
                modules[name] = module
                version = str(getattr(module, "__version__", getattr(module, "VERSION", "unknown")))
                report["versions"][name] = version
                return {"version": version}
            _check(report, f"import_{name}", load)
        for name in VERSION_BOUNDS:
            if name in modules:
                lower, upper = VERSION_BOUNDS[name]
                _check(report, f"version_{name}", lambda name=name, lower=lower, upper=upper:
                       _require(version_satisfies(name, report["versions"][name]),
                                f"{name} requires release >={'.'.join(map(str, lower))}, <{'.'.join(map(str, upper))}"))

        torch = modules.get("torch")
        numpy = modules.get("numpy")
        device = "cpu"
        if torch is not None:
            def cuda_info():
                available = bool(torch.cuda.is_available())
                report["cuda_available"] = available
                info = {"available": available, "torch_cuda_build": getattr(torch.version, "cuda", None)}
                if available:
                    index = torch.cuda.current_device()
                    properties = torch.cuda.get_device_properties(index)
                    report["cuda_device"] = {
                        "index": index, "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "cuda_visible_memory_gib": properties.total_memory / 2**30,
                    }
                    info["device"] = report["cuda_device"]
                if require_cuda and not available:
                    raise RuntimeError("CUDA required but unavailable; --allow-cpu only permits a CPU diagnostic")
                return info
            _check(report, "cuda_availability", cuda_info)
            if report["cuda_device"] is not None:
                device = f"cuda:{report['cuda_device']['index']}"
            if numpy is not None:
                _check(report, "numpy_torch_cpu", lambda: probe_numpy_torch(torch, numpy, "cpu"))
                if device.startswith("cuda"):
                    report["gpu_execution_verified"] = _check(
                        report, "numpy_torch_cuda", lambda: probe_numpy_torch(torch, numpy, device))
        if numpy is not None and "cv2" in modules:
            _check(report, "opencv_rgb", lambda: probe_opencv(modules["cv2"], numpy))
        if with_detection:
            def load_ultralytics():
                with temporary_ultralytics_settings():
                    module = importer("ultralytics")
                    version = str(getattr(module, "__version__", "unknown"))
                    report["versions"]["ultralytics"] = version
                    return {"version": version}
            detection_imported = _check(report, "import_ultralytics", load_ultralytics)
            if torch is not None and "torchvision" in modules:
                # If CUDA exists this invokes the actual CUDA NMS kernel, not a CPU fallback.
                nms_ok = _check(report, "torchvision_nms", lambda: probe_nms(torch, modules["torchvision"], device))
                report["detection_gpu_execution_verified"] = bool(
                    detection_imported and nms_ok and device.startswith("cuda") and report["gpu_execution_verified"])
    if captured.getvalue():
        report["captured_library_stdout"] = captured.getvalue()[-2048:]
    return _finalize(report)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)
    return {"satisfied": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True,
                       help="Require working CUDA (default)")
    group.add_argument("--allow-cpu", dest="require_cuda", action="store_false",
                       help="Permit a CPU-only diagnostic; it does not verify GPU execution")
    parser.add_argument("--with-detection", action="store_true", help="Also import Ultralytics and execute torchvision NMS")
    parser.add_argument("--output", type=Path, help="Also save JSON to a new file; parent must exist")
    args = parser.parse_args(argv)
    report = inspect_environment(require_cuda=args.require_cuda, with_detection=args.with_detection)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        except (OSError, ValueError) as exc:
            report["checks"].append({"name": "write_output", "passed": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
            _finalize(report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
