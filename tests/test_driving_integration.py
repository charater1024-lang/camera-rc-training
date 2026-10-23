import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from camera_rc.driving.benchmark import make_runtime_fixture, benchmark_runtime
from camera_rc.driving.calibration import fit_ground_calibration
from camera_rc.driving.config import DEFAULTS, check_actuation, load_runtime_config, validate_runtime_config
from camera_rc.driving.runtime import run_drive
from camera_rc.driving.simulation import run_simulation
from camera_rc.driving.sources import Rectifier
from camera_rc.driving.resources import configure_device


def test_closed_loop_moves_stops_and_latches(tmp_path):
    report = run_simulation(tmp_path / "sim", steps=180)
    assert report["success"], report
    assert report["physical_output"] is False
    assert report["scenarios"]["clear_straight"]["progress_m"] > 0.5


def test_replay_native_inference_to_control_and_stop(tmp_path):
    torch.set_num_threads(2)
    fixture = make_runtime_fixture(tmp_path / "fixture", device="cpu")
    path = Path(fixture["config"])
    cfg = json.loads(path.read_text())
    cfg["runtime"] = {"perception_hz": 10, "max_processing_s": 1, "plan_timeout_s": 2}
    cfg["planner"] = {"reaction_time_s": 2.5}
    path.write_text(json.dumps(cfg))
    result = run_drive(path, tmp_path / "run", max_frames=4)
    assert result["success"], result
    assert result["shutdown_stop_sent"]
    assert result["frames"] == 4
    assert result["physical_output"] is False
    logs = [json.loads(l) for l in (tmp_path / "run/control.jsonl").read_text().splitlines()]
    assert any(r["command"]["enabled"] for r in logs)
    assert all(not r["physical_output"] for r in logs)


def test_benchmark_reports_real_scope(tmp_path):
    torch.set_num_threads(2)
    fixture = make_runtime_fixture(tmp_path / "fixture", device="cpu")
    result = benchmark_runtime(fixture["config"], tmp_path / "benchmark.json", frames=2, warmup=1)
    assert result["pipeline_latency"]["frames"] == 2
    assert result["sampled_process_gpu_within_budget"] is None
    assert result["model_diagnostic"] is True
    assert result["valid_plan_frames"] == 2


def test_physical_output_rejects_fixture_and_unobserved_clearance(tmp_path, monkeypatch):
    fixture = make_runtime_fixture(tmp_path / "fixture", device="cpu")
    cfg = load_runtime_config(fixture["config"])
    with pytest.raises(ValueError):
        check_actuation(cfg, True)
    cfg["transport"].update(kind="serial", calibration_confirmed=True)
    cfg["source"]["kind"] = "camera"
    cfg["models_validated_for_control"] = True
    cfg["perception"]["allow_diagnostic"] = False
    monkeypatch.setattr("camera_rc.driving.config.camera_device_identity",
                        lambda device, **kwargs: f"/dev/video{device}")
    with pytest.raises(ValueError, match="Synthetic calibration"):
        check_actuation(cfg, True)
    cfg["calibration"]["diagnostic_only"] = False
    cfg["planner"]["assumed_clearance_m"] = 0.05
    with pytest.raises(ValueError, match="Unobserved"):
        check_actuation(cfg, True)


def test_reaction_budget_includes_write_and_control_tick():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["planner"]["reaction_time_s"] = 0.4
    with pytest.raises(ValueError, match="reaction_time"):
        validate_runtime_config(cfg)


def test_calibration_fit_requires_independent_checks_and_preserves_diagnostic(tmp_path):
    points = {"image_size": [360, 640], "input_is_rectified": True,
              "supported_range": {"min_forward_m": 0.1, "max_forward_m": 1.8, "max_abs_lateral_m": 0.8},
              "points": [[64, 20, 1.9, 0.8], [576, 20, 1.9, -0.8], [64, 340, 0.05, 0.8], [576, 340, 0.05, -0.8]],
              "validation_points": [[320, 180, 0.975, 0], [160, 260, 0.5125, 0.5]], "diagnostic_only": True}
    path = tmp_path / "points.json"
    path.write_text(json.dumps(points))
    report = fit_ground_calibration(path, tmp_path / "calibration.json")
    assert report["max_validation_error_m"] < 1e-5
    assert json.loads((tmp_path / "calibration.json").read_text())["diagnostic_only"] is True
    points["validation_points"][0][2] += 1
    path.write_text(json.dumps(points))
    with pytest.raises(ValueError, match="Independent calibration error"):
        fit_ground_calibration(path, tmp_path / "bad.json")


def test_invalid_rectification_pixels_are_identified():
    rectifier = Rectifier({"image_size": [64, 96], "camera_matrix": [[60, 0, 48], [0, 60, 32], [0, 0, 1]],
                          "distortion_coefficients": [0.5, 0, 0, 0, 0]})
    assert not rectifier.valid_pixels.all()
    assert rectifier.valid_pixels[32, 48]
    assert rectifier(np.zeros((64, 96, 3), np.uint8)).shape == (64, 96, 3)


def test_cuda_default_resolves_index_before_allocator_limit(monkeypatch):
    from types import SimpleNamespace
    recorded = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=4 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: recorded.append((fraction, device.index)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    assert configure_device(DEFAULTS) == "cuda:0"
    assert recorded == [(0.75, 0)]
