"""Compatibility diagnostics use mocked CUDA; tests never open physical devices."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "check_runtime_environment", Path(__file__).resolve().parents[1] / "scripts" / "check_runtime_environment.py")
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


class Tensor:
    def __init__(self, values, device="cpu"):
        self.values, self.device = np.asarray(values), device

    def to(self, device):
        return Tensor(self.values, device)

    def __add__(self, value):
        return Tensor(self.values + value, self.device)

    def sum(self):
        return Tensor(self.values.sum(), self.device)

    def item(self):
        return self.values.item()

    def cpu(self):
        return self.to("cpu")

    def tolist(self):
        return self.values.tolist()


def fake_modules(cuda=True):
    def nms(boxes, scores, threshold):
        assert boxes.device == scores.device
        assert threshold == 0.5
        return Tensor([0, 2], boxes.device)
    torch = SimpleNamespace(
        __version__="2.9.1+cu126", float32=np.float32,
        from_numpy=lambda values: Tensor(values),
        tensor=lambda values, dtype, device: Tensor(np.asarray(values, dtype=dtype), device),
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: cuda, current_device=lambda: 0,
                             synchronize=lambda device: None,
                             get_device_properties=lambda index: SimpleNamespace(
                                 name="Mock GPU", major=8, minor=7, total_memory=8 * 2**30)),
    )
    return {
        "torch": torch,
        "torchvision": SimpleNamespace(__version__="0.24.1", ops=SimpleNamespace(nms=nms)),
        "numpy": np,
        "cv2": SimpleNamespace(__version__="4.10.0", COLOR_BGR2RGB=4,
                               cvtColor=lambda image, code: image[..., ::-1]),
        "serial": SimpleNamespace(__version__="3.5"),
        "psutil": SimpleNamespace(__version__="7.0.0"),
        "ultralytics": SimpleNamespace(__version__="8.4.0"),
    }


@pytest.mark.parametrize("name,version,expected", [
    ("torch", "2.7.0a0+nv25.04", True), ("torch", "2.6.1", False),
    ("torch", "3.0.0", False), ("torchvision", "0.22.0a0+abc", True),
    ("torchvision", "0.21.0", False), ("numpy", "1.26.4", True),
    ("numpy", "1.24.4", False), ("numpy", "3.0.0", False),
])
def test_declared_version_bounds_and_nvidia_build_tags(name, version, expected):
    assert environment.version_satisfies(name, version) is expected


def test_default_requires_cuda_without_silent_cpu_pass():
    report = environment.inspect_environment(importer=fake_modules(cuda=False).__getitem__)
    assert report["success"] is False
    assert "cuda_availability" in report["failures"]
    assert report["gpu_execution_verified"] is False
    assert report["cuda_available"] is False


def test_explicit_cpu_diagnostic_is_not_gpu_verification():
    report = environment.inspect_environment(require_cuda=False, with_detection=True,
                                             importer=fake_modules(cuda=False).__getitem__)
    assert report["success"] is True
    assert report["gpu_execution_verified"] is False
    assert report["detection_gpu_execution_verified"] is False
    assert report["linux_device_access_verified"] is False
    assert report["camera_opened"] is False and report["serial_port_opened"] is False


def test_cuda_and_nms_probe_execute_on_cuda_device():
    modules = fake_modules()
    devices = []
    original = modules["torchvision"].ops.nms
    def nms(boxes, scores, threshold):
        devices.append(boxes.device)
        return original(boxes, scores, threshold)
    modules["torchvision"].ops.nms = nms
    report = environment.inspect_environment(with_detection=True, importer=modules.__getitem__)
    assert report["success"] is True
    assert report["gpu_execution_verified"] is True
    assert report["detection_gpu_execution_verified"] is True
    assert devices == ["cuda:0"]
    assert report["runtime_timing_verified"] is False


def test_missing_binary_dependency_is_reported_without_crashing():
    modules = fake_modules()
    def importer(name):
        if name == "cv2":
            raise ImportError("missing libGL.so.1")
        return modules[name]
    report = environment.inspect_environment(importer=importer)
    assert report["success"] is False
    assert "import_cv2" in report["failures"]
    assert any(check["name"] == "numpy_torch_cuda" for check in report["checks"])


def test_numpy_abi_failure_fails_actual_probe():
    modules = fake_modules()
    def broken_numpy(values):
        raise RuntimeError("Numpy is not available")
    modules["torch"].from_numpy = broken_numpy
    report = environment.inspect_environment(importer=modules.__getitem__)
    assert report["success"] is False
    assert "numpy_torch_cpu" in report["failures"] and "numpy_torch_cuda" in report["failures"]
    assert report["gpu_execution_verified"] is False


def test_missing_cuda_nms_is_not_mistaken_for_detection_support():
    modules = fake_modules()
    def unavailable_nms(*args):
        raise NotImplementedError("torchvision::nms CUDA backend unavailable")
    modules["torchvision"].ops.nms = unavailable_nms
    report = environment.inspect_environment(with_detection=True, importer=modules.__getitem__)
    assert report["gpu_execution_verified"] is True
    assert report["success"] is False
    assert report["detection_gpu_execution_verified"] is False
    assert "torchvision_nms" in report["failures"]


def test_incorrect_cv2_result_is_rejected():
    modules = fake_modules()
    modules["cv2"].cvtColor = lambda image, code: image
    report = environment.inspect_environment(importer=modules.__getitem__)
    assert "opencv_rgb" in report["failures"]


def test_unsupported_torch_version_is_reported():
    modules = fake_modules()
    modules["torch"].__version__ = "2.5.0+nv24.08"
    report = environment.inspect_environment(importer=modules.__getitem__)
    assert "version_torch" in report["failures"]


def test_ultralytics_import_uses_temporary_settings_and_captures_stdout(monkeypatch):
    modules = fake_modules()
    monkeypatch.setenv("YOLO_CONFIG_DIR", "original-config")
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    directories = []
    def importer(name):
        if name == "ultralytics":
            directories.append(Path(os.environ["YOLO_CONFIG_DIR"]))
            assert directories[-1].is_dir()
            assert os.environ["YOLO_OFFLINE"] == "true"
            print("Third-party import chatter")
        return modules[name]
    report = environment.inspect_environment(with_detection=True, importer=importer)
    assert report["success"] is True
    assert os.environ["YOLO_CONFIG_DIR"] == "original-config"
    assert "YOLO_OFFLINE" not in os.environ
    assert not directories[0].exists()
    assert "Third-party import chatter" in report["captured_library_stdout"]


def test_cli_writes_matching_json_and_cpu_intent(monkeypatch, tmp_path, capsys):
    inspect = environment.inspect_environment
    monkeypatch.setattr(environment, "inspect_environment", lambda **kwargs:
                        inspect(**kwargs, importer=fake_modules(cuda=False).__getitem__))
    output = tmp_path / "environment.json"
    assert environment.main(["--allow-cpu", "--output", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(output.read_text(encoding="utf-8"))
    assert stdout["require_cuda"] is False and stdout["gpu_execution_verified"] is False


def test_cli_preserves_existing_report_and_returns_failure(monkeypatch, tmp_path, capsys):
    inspect = environment.inspect_environment
    monkeypatch.setattr(environment, "inspect_environment", lambda **kwargs:
                        inspect(**kwargs, importer=fake_modules().__getitem__))
    output = tmp_path / "environment.json"
    output.write_text("previous report", encoding="utf-8")
    assert environment.main(["--output", str(output)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "write_output" in report["failures"]
    assert output.read_text(encoding="utf-8") == "previous report"


def test_cli_default_cuda_failure_returns_nonzero(monkeypatch, capsys):
    inspect = environment.inspect_environment
    monkeypatch.setattr(environment, "inspect_environment", lambda **kwargs:
                        inspect(**kwargs, importer=fake_modules(cuda=False).__getitem__))
    assert environment.main([]) == 1
    assert json.loads(capsys.readouterr().out)["require_cuda"] is True
