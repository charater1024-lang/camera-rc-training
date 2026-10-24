import copy
import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from camera_rc.driving.benchmark import make_multiview_fixture, benchmark_runtime
from camera_rc.driving.config import camera_device_identity, check_actuation, load_runtime_config, validate_camera_group
from camera_rc.driving.multisource import FrameBundle, MultiCameraSource, MultiReplaySource
from camera_rc.driving.runtime import run_drive
from camera_rc.driving.sources import Frame


def settings():
    return {"kind": "multicamera", "max_skew_s": 0.04, "cameras": [
        {"name": name, "index": index, "width": 64, "height": 32, "fps": 30}
        for index, name in enumerate(("front", "left"))]}


class FakeCamera:
    def __init__(self):
        self.frames = deque()
        self.exhausted, self.closed = False, False
        self.error = None

    def read(self):
        return self.frames.popleft() if self.frames else None

    def close(self):
        self.closed = True


def live_group(monkeypatch):
    monkeypatch.setattr("camera_rc.driving.multisource.time.monotonic", lambda: 10.1)
    cameras = {name: FakeCamera() for name in ("front", "left")}
    source = MultiCameraSource(settings(), camera_factory=lambda cfg: cameras[cfg["name"]])
    return source, cameras


def frame(stamp=10.0, seq=1):
    return Frame(np.zeros((32, 64, 3), np.uint8), stamp, seq)


def test_waits_for_all_views_and_never_reuses_previous_view(monkeypatch):
    source, cameras = live_group(monkeypatch)
    cameras["front"].frames.append(frame())
    assert source.read() is None
    cameras["left"].frames.append(frame(10.02))
    bundle = source.read()
    assert bundle.received_at == 10.0
    assert bundle.skew_s == pytest.approx(0.02)
    cameras["front"].frames.append(frame(10.04, 2))
    assert source.read() is None
    cameras["left"].frames.append(frame(10.05, 2))
    assert source.read().sequence == 2
    assert len(source.latest) == 2


def test_waits_for_skew_to_recover_without_growing_queue(monkeypatch):
    source, cameras = live_group(monkeypatch)
    cameras["front"].frames.append(frame(10.0))
    cameras["left"].frames.append(frame(10.08))
    assert source.read() is None
    cameras["front"].frames.append(frame(10.09, 2))
    assert source.read().skew_s == pytest.approx(0.01)
    assert len(source.latest) == 2


def test_failed_camera_is_not_removed_from_required_set(monkeypatch):
    source, cameras = live_group(monkeypatch)
    cameras["left"].exhausted = True
    cameras["left"].error = "unplugged"
    assert source.read() is None
    assert source.exhausted
    assert "left:unplugged" in source.error
    source.close()
    assert all(camera.closed for camera in cameras.values())


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), 11.0])
def test_bad_live_timestamp_faults(monkeypatch, stamp):
    source, cameras = live_group(monkeypatch)
    cameras["front"].frames.append(frame(stamp))
    assert source.read() is None
    assert source.exhausted


def test_failed_group_initialization_closes_previously_opened_camera():
    first = FakeCamera()
    def factory(cfg):
        if cfg["name"] == "front":
            return first
        raise ValueError("camera unavailable")
    with pytest.raises(ValueError, match="unavailable"):
        MultiCameraSource(settings(), camera_factory=factory)
    assert first.closed


@pytest.mark.parametrize("field,value", [("name", "front"), ("index", 0), ("fps", 0)])
def test_reject_ambiguous_camera_group(field, value):
    cfg = settings()
    cfg["cameras"][1][field] = value
    with pytest.raises(ValueError):
        validate_camera_group(cfg)


@pytest.mark.parametrize("device", ["recording.mp4", "https://example.org/video", "rtsp://camera", -1, True, "/dev/../recording.mp4"])
def test_live_camera_rejects_recordings_and_nonphysical_sources(device):
    with pytest.raises(ValueError, match="physical camera"):
        camera_device_identity(device)


def test_numeric_index_alias_cannot_supply_two_views():
    cfg = settings()
    cfg["cameras"][1]["device"] = "/dev/video0"
    with pytest.raises(ValueError, match="distinct"):
        validate_camera_group(cfg)


def replay(tmp_path, row):
    Image.fromarray(np.zeros((32, 64, 3), np.uint8)).save(tmp_path / "frame.png")
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(row) + "\n")
    cfg = settings()
    cfg.update(kind="multireplay", path=str(path))
    return MultiReplaySource(cfg)


def test_replay_uses_oldest_camera_time(tmp_path):
    source = replay(tmp_path, {"views": {"front": {"image": "frame.png"},
                                       "left": {"image": "frame.png", "offset_s": -0.03}}})
    bundle = source.read()
    assert bundle.skew_s == pytest.approx(0.03)
    assert bundle.received_at == bundle.views["left"].received_at
    assert source.read() is None and source.exhausted
    source.close()


@pytest.mark.parametrize("views", [
    {"front": {"image": "frame.png"}},
    {"front": {"image": "frame.png"}, "left": {"image": "frame.png", "offset_s": -0.05}},
    {"front": {"image": "frame.png"}, "left": {"image": "frame.png", "offset_s": 0.01}},
])
def test_replay_rejects_missing_skewed_or_future_views(tmp_path, views):
    source = replay(tmp_path, {"views": views})
    try:
        with pytest.raises(ValueError):
            source.read()
    finally:
        source.close()


def test_multiview_pipeline_and_independent_control_integration(tmp_path):
    torch.set_num_threads(2)
    fixture = make_multiview_fixture(tmp_path / "fixture", device="cpu")
    path = Path(fixture["config"])
    cfg = json.loads(path.read_text())
    cfg["runtime"].update(max_processing_s=1, plan_timeout_s=2)
    cfg["planner"] = {"reaction_time_s": 2.5}
    path.write_text(json.dumps(cfg))
    report = run_drive(path, tmp_path / "run", max_frames=2)
    assert report["success"], report
    assert report["shutdown_stop_sent"] and not report["physical_output"]
    assert set(report["camera_names"]) == {"front", "left", "right", "rear"}
    assert report["shared_model_instances"] == 1
    rows = [json.loads(row) for row in (tmp_path / "run/perception.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(len(row["views"]) == 4 for row in rows)
    assert any(row["plan_valid"] for row in rows)
    benchmark = benchmark_runtime(path, tmp_path / "benchmark.json", frames=2, warmup=1)
    assert benchmark["pipeline_latency"]["frames"] == 2
    assert len(benchmark["camera_input_hw"]) == 4
    assert benchmark["model_diagnostic"]


def test_multiview_actuation_requires_timing_and_every_real_calibration(tmp_path, monkeypatch):
    fixture = make_multiview_fixture(tmp_path / "fixture", device="cpu")
    cfg = load_runtime_config(fixture["config"])
    cfg["transport"].update(kind="serial", calibration_confirmed=True)
    cfg["source"]["kind"] = "multicamera"
    for index, camera in enumerate(cfg["source"]["cameras"]):
        camera["index"] = index
    cfg["models_validated_for_control"] = True
    cfg["perception"]["allow_diagnostic"] = False
    with pytest.raises(ValueError, match="exposure timing"):
        check_actuation(cfg, True)
    cfg["source"]["capture_timing_validated"] = True
    # This contract test supplies simulated physical devices; dedicated device
    # identity tests verify filesystem character-device checks independently.
    monkeypatch.setattr("camera_rc.driving.config.camera_device_identity",
                        lambda device, **kwargs: f"/dev/video{device}")
    with pytest.raises(ValueError, match="Synthetic"):
        check_actuation(cfg, True)
    for calibration in cfg["calibrations"].values():
        calibration["diagnostic_only"] = False
    with pytest.raises(ValueError, match="Physical evidence producers"):
        check_actuation(cfg, True)
