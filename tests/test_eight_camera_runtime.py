"""Eight-view research contracts; no cameras, serial devices or real geometry."""
import copy
import json
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from camera_rc.driving import config as config_module
from camera_rc.driving.config import DEFAULTS, camera_device_identity, check_actuation, validate_camera_group
from camera_rc.driving.multisource import FrameBundle, MultiCameraSource, MultiReplaySource
from camera_rc.driving.multiview import MultiViewPlanner
from camera_rc.driving.pipeline import DrivingPipeline
from camera_rc.driving.runtime import ControlLoop
from camera_rc.driving.sources import Frame
from camera_rc.driving.transport import DryRunTransport
from camera_rc.driving.types import Detection, Perception, Plan, VehicleState


NAMES = ("front_main", "front_wide", "front_near", "left_forward", "right_forward",
         "left_rearward", "right_rearward", "rear")


def group(kind="multicamera", count=8):
    return {"kind": kind, "max_skew_s": .04, "capture_timing_validated": False,
            "cameras": [{"name": name, "index": i, "width": 64, "height": 32, "fps": 30}
                        for i, name in enumerate(NAMES[:count])]}


def synthetic_calibrations():
    # Intentionally overlapping synthetic views only exercise software behavior.
    # They are not a proposed physical arrangement or a coverage measurement.
    return {name: {"image_size": [32, 64], "coordinate_origin": "rear_axle_center",
                   "calibrated": True, "rectified": True, "input_is_rectified": True,
                   "diagnostic_only": True,
                   "image_to_ground": [[0, -2.5 / 31, 2], [-2 / 63, 0, 1], [0, 0, 1]],
                   "supported_range": {"min_forward_m": -.49, "max_forward_m": 1.99,
                                       "min_lateral_m": -.99, "max_lateral_m": .99}}
            for name in NAMES}


def bundle(stamp=10):
    return FrameBundle({name: Frame(np.zeros((32, 64, 3), np.uint8), stamp, 1)
                        for name in NAMES}, 1)


def test_shipped_eight_camera_profile_is_inert_and_has_no_invented_calibration():
    project = Path(__file__).resolve().parents[1]
    path = project / "configs/runtime_msi_8cam_dryrun.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(view["name"] for view in cfg["source"]["cameras"]) == NAMES
    validate_camera_group(cfg["source"])
    assert cfg["transport"]["kind"] == "dry_run"
    assert not cfg["source"]["capture_timing_validated"]
    assert not cfg["models_validated_for_control"]
    assert not cfg["transport"]["calibration_confirmed"]
    assert cfg["runtime"]["gpu_budget_gib"] == 4
    assert cfg["runtime"]["ram_budget_gib"] == 16
    assert len({camera["calibration"] for camera in cfg["source"]["cameras"]}) == 8
    assert all("REPLACE_" in camera["device"] for camera in cfg["source"]["cameras"])
    assert all("image_to_ground" not in camera for camera in cfg["source"]["cameras"])


@pytest.mark.parametrize("count", [7, 8])
def test_more_than_six_views_cannot_actuate_even_with_all_flags_true(count):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(source=group(count=count), models_validated_for_control=True,
               calibrations={name: {"diagnostic_only": False} for name in NAMES[:count]})
    cfg["source"]["capture_timing_validated"] = True
    cfg["transport"].update(kind="serial", calibration_confirmed=True)
    with pytest.raises(ValueError, match="research-only.*dry_run"):
        check_actuation(cfg, True)
    cfg["transport"]["kind"] = "dry_run"
    check_actuation(cfg, False)


def test_ninth_camera_is_rejected():
    source = group()
    source["cameras"].append({"name": "ninth", "index": 8, "width": 64, "height": 32, "fps": 30})
    with pytest.raises(ValueError, match="2 to 8"):
        validate_camera_group(source)


def mock_physical_paths(monkeypatch, *, present=True, character=True):
    class PhysicalPath:
        def __init__(self, value):
            self.value = value
            self.parts = PurePosixPath(value).parts

        def exists(self):
            return present

        def resolve(self, strict=False):
            assert strict is True
            if not present:
                raise FileNotFoundError(self.value)
            return PhysicalPath("/dev/video7")

        def as_posix(self):
            return self.value

        def stat(self):
            return SimpleNamespace(st_mode=stat.S_IFCHR if character else stat.S_IFREG)

    monkeypatch.setattr(config_module, "Path", PhysicalPath)


def test_missing_numeric_physical_camera_no_longer_bypasses_presence(monkeypatch):
    mock_physical_paths(monkeypatch, present=False)
    assert camera_device_identity(7) == "/dev/video7"
    with pytest.raises(FileNotFoundError):
        camera_device_identity(7, require_present=True)


@pytest.mark.parametrize("device", [7, "/dev/video7", "/dev/v4l/by-id/example"])
def test_numeric_and_named_physical_devices_require_character_device(monkeypatch, device):
    mock_physical_paths(monkeypatch, character=False)
    with pytest.raises(ValueError, match="character device"):
        camera_device_identity(device, require_present=True)


def test_numeric_and_symbolic_alias_resolve_to_same_physical_identity(monkeypatch):
    mock_physical_paths(monkeypatch)
    assert camera_device_identity(7, require_present=True) == "/dev/video7"
    assert camera_device_identity("/dev/v4l/by-id/example", require_present=True) == "/dev/video7"
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(source=group(count=2), models_validated_for_control=True)
    cfg["source"]["cameras"][0]["index"] = 7
    cfg["source"]["cameras"][1]["device"] = "/dev/v4l/by-id/example"
    cfg["source"]["capture_timing_validated"] = True
    cfg["transport"].update(kind="serial", calibration_confirmed=True)
    with pytest.raises(ValueError, match="distinct live"):
        check_actuation(cfg, True)


def test_eight_replay_requires_exact_view_set_and_uses_oldest_time(tmp_path):
    Image.new("RGB", (64, 32)).save(tmp_path / "frame.png")
    views = {name: {"image": "frame.png", "offset_s": -.03 if name == "rear" else 0}
             for name in NAMES}
    manifest = tmp_path / "replay.jsonl"
    manifest.write_text(json.dumps({"views": views}) + "\n", encoding="utf-8")
    source = MultiReplaySource({**group("multireplay"), "path": str(manifest)})
    try:
        received = source.read()
        assert len(received.views) == 8 and received.skew_s == pytest.approx(.03)
        assert received.received_at == received.views["rear"].received_at
    finally:
        source.close()
    del views["front_near"]
    manifest.write_text(json.dumps({"views": views}) + "\n", encoding="utf-8")
    source = MultiReplaySource({**group("multireplay"), "path": str(manifest)})
    try:
        with pytest.raises(ValueError, match="exactly all"):
            source.read()
    finally:
        source.close()


def test_eight_camera_stale_view_never_reuses_a_previous_bundle(monkeypatch):
    monkeypatch.setattr("camera_rc.driving.multisource.time.monotonic", lambda: 10.1)

    class Camera:
        exhausted = False
        def __init__(self):
            self.frame = Frame(np.zeros((32, 64, 3), np.uint8), 10, 1)
        def read(self):
            item, self.frame = self.frame, None
            return item
        def close(self):
            pass

    cameras = {name: Camera() for name in NAMES}
    source = MultiCameraSource(group(), camera_factory=lambda cfg: cameras[cfg["name"]])
    assert source.read() is not None
    for name in NAMES[:-1]:
        cameras[name].frame = Frame(np.zeros((32, 64, 3), np.uint8), 10.03, 2)
    assert source.read() is None
    assert len(source.latest) == 8


def test_eight_views_share_one_model_and_missing_view_rejected_before_inference(monkeypatch):
    instances = []
    class Model:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            instances.append(self)
        def infer(self, rgb):
            self.calls += 1
            return Perception(np.ones(rgb.shape[:2], dtype=np.float32), inference_ms=1)
    monkeypatch.setattr("camera_rc.driving.pipeline.CameraPerception", Model)
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(source=group("multireplay"), calibrations=synthetic_calibrations())
    pipeline = DrivingPipeline(cfg, "cpu")
    result = pipeline.run(bundle(), 0)
    assert len(instances) == 1 and instances[0].calls == 8
    assert len(result.views) == 8 and result.inference_ms == 8 and result.plan.valid
    missing = bundle()
    del missing.views["front_near"]
    with pytest.raises(ValueError, match="Every configured view"):
        pipeline.run(missing, 0)
    assert instances[0].calls == 8


def test_obstacle_in_one_of_eight_views_is_not_overruled_by_seven_clear_views():
    calibrations = synthetic_calibrations()
    planner = MultiViewPlanner(DEFAULTS["planner"], calibrations)
    views = {name: Perception(np.ones((32, 64), dtype=np.float32)) for name in NAMES}
    assert planner.plan(views, 0).valid
    pixels = planner.projectors["front_near"].ground_to_pixel([[.3, -.2], [.3, .2]])
    x1, x2 = sorted(pixels[:, 0])
    bottom = float(pixels[0, 1])
    views["front_near"].detections = [Detection((float(x1), bottom - 2, float(x2), bottom), .99, "person")]
    assert not planner.plan(views, 0).valid
    views["front_near"].detections = []
    views["front_near"].drivable[:] = 0
    assert not planner.plan(views, 0).valid


def test_oldest_eight_camera_view_expires_control_without_physical_output(tmp_path):
    class Feedback(DryRunTransport):
        def read_state(self, now):
            return VehicleState(now, 0, 0, True, False)
    frame = bundle(1.03)
    frame.views["rear"].received_at = 1.0
    loop = ControlLoop(copy.deepcopy(DEFAULTS), Feedback(), tmp_path)
    loop.publish(Plan(np.array([[0, 0], [.5, 0], [1, 0]]), .4, True, "clear_path"), frame.received_at)
    assert loop.step(1.1)["command"]["enabled"]
    result = loop.step(1.21)
    assert not result["physical_output"] and not result["command"]["enabled"]
    assert result["command"]["reason"] == "plan_expired_restart_required"
