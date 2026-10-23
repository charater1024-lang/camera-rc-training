import copy
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import torch

from camera_rc.driving.config import DEFAULTS, load_runtime_config, validate_runtime_config
from camera_rc.driving.resources import ResourceMonitor


@pytest.mark.parametrize("filename,architecture,ram", [
    ("runtime_jetson_orin_nano_8gb.json", "shared", 8),
    ("runtime_msi_prestige15_ubuntu.json", "discrete", 16),
])
def test_profiles_load_with_measured_calibration_placeholder(tmp_path, filename, architecture, ram):
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "configs" / filename).read_text(encoding="utf-8"))
    # Configuration test only: never substitute this for geometric validation.
    (tmp_path / "calibration_measured.json").write_text("{}")
    (tmp_path / filename).write_text(json.dumps(raw))
    cfg = load_runtime_config(tmp_path / filename)
    assert cfg["device"] == "cuda"
    assert cfg["hardware"]["memory_architecture"] == architecture
    assert cfg["runtime"]["ram_budget_gib"] == ram
    assert cfg["transport"]["kind"] == "dry_run"
    assert cfg["models_validated_for_control"] is False
    assert cfg["runtime"]["plan_timeout_s"] == 0.2
    assert cfg["runtime"]["max_processing_s"] == 0.15


@pytest.mark.parametrize("architecture", ["unified_typo", None, 1])
def test_reject_unknown_memory_architecture(architecture):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["hardware"]["memory_architecture"] = architecture
    with pytest.raises(ValueError, match="memory_architecture"):
        validate_runtime_config(cfg)


def test_reject_impossible_shared_budget_and_available_floor():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["hardware"]["memory_architecture"] = "shared"
    cfg["runtime"].update(ram_budget_gib=8, process_ram_limit_gib=4, gpu_budget_gib=16)
    with pytest.raises(ValueError, match="Shared GPU"):
        validate_runtime_config(cfg)
    cfg["runtime"].update(gpu_budget_gib=4, system_available_min_gib=8)
    with pytest.raises(ValueError, match="floor"):
        validate_runtime_config(cfg)


def test_memory_pressure_guard_uses_available_ram_and_preserves_minimum(monkeypatch):
    memory = SimpleNamespace(total=8 * 2**30, available=3 * 2**30)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1 * 2**30))
    monkeypatch.setattr(psutil, "Process", lambda: process)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    monitor = ResourceMonitor("cpu", "shared")
    limits = {"process_ram_limit_gib": 4, "system_available_min_gib": 1.5}
    monitor.check_limits(limits)
    memory.available = 1 * 2**30  # OS or another process can exhaust shared memory.
    with pytest.raises(RuntimeError, match="system_available_ram_below_floor"):
        monitor.check_limits(limits)
    memory.available = 3 * 2**30
    monitor.check_limits(limits)
    assert monitor.report()["system_available_min_gib_sampled"] == 1
    process.memory_info = lambda: SimpleNamespace(rss=5 * 2**30)
    with pytest.raises(RuntimeError, match="process_ram_budget_exceeded"):
        monitor.check_limits(limits)


@pytest.mark.parametrize("architecture,physical_vram", [("shared", None), ("discrete", 8)])
def test_cuda_shared_memory_is_not_reported_as_dedicated_vram(monkeypatch, architecture, physical_vram):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=8 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (6 * 2**30, 8 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    monitor = ResourceMonitor("cuda:0", architecture)
    report = monitor.report()
    assert report["cuda_addressable_memory_gib"] == 8
    assert report["physical_vram_gib"] == physical_vram
    assert report["whole_device_used_peak_gib_sampled"] == 2
    assert "configuration" in report["memory_architecture_source"]


def test_memory_pressure_during_drive_stops_and_records_fault(tmp_path, monkeypatch):
    from camera_rc.driving.benchmark import make_runtime_fixture
    from camera_rc.driving.runtime import run_drive

    torch.set_num_threads(2)
    fixture = make_runtime_fixture(tmp_path / "fixture", device="cpu")
    calls = []

    def fail_after_start(self, runtime):
        calls.append(True)
        if len(calls) == 3:  # Before load, after warmup, then first active loop.
            raise RuntimeError("system_available_ram_below_floor")

    monkeypatch.setattr(ResourceMonitor, "check_limits", fail_after_start)
    result = run_drive(fixture["config"], tmp_path / "run", max_frames=1)
    assert not result["success"]
    assert result["shutdown_stop_sent"]
    assert "system_available_ram_below_floor" in result["error"]
    assert result["frames"] == 0
