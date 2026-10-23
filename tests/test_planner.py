import numpy as np
import pytest

from camera_rc.driving.geometry import GroundProjector, constant_curvature_path
from camera_rc.driving.planner import LocalPlanner
from camera_rc.driving.types import Detection, Perception


def calibration():
    return {"calibrated": True, "rectified": True, "image_size": [240, 320],
            "image_to_ground": [[0, -1/80, 220/80], [-1/80, 0, 159.5/80], [0, 0, 1]],
            "supported_range": {"min_forward_m": -0.2, "max_forward_m": 2.7, "max_abs_lateral_m": 1.9}}


def planner(**overrides):
    return LocalPlanner({"horizon_m": 1.3, **overrides}, calibration())


def clear_perception():
    return Perception(np.ones((240, 320), dtype=np.float32))


def obstacle(projector, forward=0.65, width=1.5):
    pixels = projector.ground_to_pixel([[forward, width / 2], [forward, -width / 2]])
    x1, bottom = pixels[0]
    x2, _ = pixels[1]
    return Detection((float(x1), float(bottom - 30), float(x2), float(bottom)), 0.9, "person")


def test_homography_roundtrip_and_left_sign():
    projection = GroundProjector(calibration())
    ground = np.array([[0.4, -0.2], [1.1, 0.3]])
    assert np.allclose(projection.pixel_to_ground(projection.ground_to_pixel(ground)), ground)
    assert projection.ground_to_pixel([[1, 0.3]])[0, 0] < projection.ground_to_pixel([[1, -0.3]])[0, 0]
    left, _ = constant_curvature_path(0.5, [0, 0.5, 1])
    right, _ = constant_curvature_path(-0.5, [0, 0.5, 1])
    assert left[-1, 1] > 0 and right[-1, 1] < 0
    assert np.allclose(left[:, 0], right[:, 0])
    assert np.allclose(left[:, 1], -right[:, 1])


def test_clear_path_and_route_hint_left_right():
    local = planner()
    straight = local.plan(clear_perception(), 0.2)
    left = local.plan(clear_perception(), 0.2, "left")
    right = local.plan(clear_perception(), 0.2, "right")
    assert straight.valid and straight.curvature == pytest.approx(0)
    assert left.valid and left.curvature > 0 and left.points[-1, 1] > 0
    assert right.valid and right.curvature < 0 and right.points[-1, 1] < 0
    assert 0 < straight.target_speed_mps <= local.max_speed_mps


def test_obstacle_wall_stops_all_paths():
    local = planner()
    perception = clear_perception()
    perception.detections = [obstacle(local.projector)]
    result = local.plan(perception, 0.3)
    assert not result.valid and result.target_speed_mps == 0
    assert result.reason == "all_paths_blocked_or_unknown"


def test_obstacle_close_to_initial_footprint_is_not_exempt():
    local = planner()
    perception = clear_perception()
    perception.detections = [obstacle(local.projector, forward=local.front + 0.01, width=0.15)]
    assert not local.plan(perception, 0).valid


def test_unknown_and_nonfinite_masks_stop():
    local = planner()
    assert not local.plan(Perception(np.zeros((240, 320))), 0).valid
    mask = clear_perception().drivable
    mask[100, 100] = np.nan
    assert local.plan(Perception(mask), 0).reason == "invalid_drivable_probability"
    assert local.plan(Perception(np.ones((120, 160))), 0).reason == "calibration_image_size_mismatch"


def test_unknown_forward_strip_and_insufficient_supported_range_stop():
    local = planner()
    mask = clear_perception().drivable
    row = int(local.projector.ground_to_pixel([[0.8, 0]])[0, 1])
    mask[row - 4:row + 5] = 0
    assert not local.plan(Perception(mask), 0.3).valid
    config = calibration()
    config["supported_range"]["max_forward_m"] = 0.6
    assert not LocalPlanner({"horizon_m": 1.3}, config).plan(clear_perception(), 0).valid


def test_vehicle_width_checks_ground_beside_centerline():
    local = planner(candidate_count=3)
    mask = np.zeros((240, 320), dtype=np.float32)
    # A narrow strip contains the centreline but cannot contain the vehicle.
    mask[:, 157:163] = 1
    assert not local.plan(Perception(mask), 0.1).valid


def test_stopping_visibility_and_bad_inputs_fail_closed():
    local = planner()
    assert local.plan(clear_perception(), 3).reason == "speed_exceeds_visible_stopping_distance"
    assert local.plan(clear_perception(), float("nan")).reason == "invalid_speed"
    assert local.plan(clear_perception(), -0.2).reason == "invalid_speed"
    assert local.plan(clear_perception(), 0, "stop").target_speed_mps == 0
    assert not local.plan(clear_perception(), 0, "roundabout").valid
    perception = clear_perception()
    perception.detections = [Detection((2, 3, float("nan"), 10), .9, "car")]
    assert local.plan(perception, 0).reason == "invalid_detection"


@pytest.mark.parametrize("change", [
    {"calibrated": False}, {"rectified": False}, {"image_size": [0, 320]},
    {"image_to_ground": [[0, 0, 0]] * 3}, {"image_to_ground": [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"supported_range": {"min_forward_m": 1, "max_forward_m": .5, "max_abs_lateral_m": 1}},
])
def test_invalid_calibration_rejected(change):
    with pytest.raises(ValueError):
        GroundProjector({**calibration(), **change})


def test_bad_planning_limits_rejected():
    for change in ({"max_speed_mps": float("nan")}, {"candidate_count": 4}, {"sample_step_m": 1},
                   {"wheelbase_m": 2}, {"max_steering_rad": 2}, {"min_drivable_probability": 1.1}):
        with pytest.raises(ValueError):
            planner(**change)
