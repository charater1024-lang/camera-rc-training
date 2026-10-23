import copy
import json

import numpy as np
import pytest

from camera_rc.driving.calibration import fit_ground_calibration
from camera_rc.driving.geometry import GroundProjector
from camera_rc.driving.multiview import MultiViewPlanner
from camera_rc.driving.types import Detection, Perception


def calibration(view, bounds):
    """Synthetic orthographic views with different camera orientations."""
    xmin, xmax, ymin, ymax = bounds
    dx, dy = (xmax - xmin) / 319, (ymax - ymin) / 239
    if view == "front":
        matrix = [[0, -(xmax - xmin) / 239, xmax], [-(ymax - ymin) / 319, 0, ymax], [0, 0, 1]]
    elif view == "left":
        matrix = [[dx, 0, xmin], [0, -dy, ymax], [0, 0, 1]]
    elif view == "rear":
        matrix = [[0, (xmax - xmin) / 239, xmin], [(ymax - ymin) / 319, 0, ymin], [0, 0, 1]]
    else:
        raise ValueError(view)
    return {"calibrated": True, "rectified": True, "input_is_rectified": True,
            "coordinate_origin": "rear_axle_center", "image_size": [240, 320],
            "image_to_ground": matrix,
            "supported_range": {"min_forward_m": xmin + .01, "max_forward_m": xmax - .01,
                                "min_lateral_m": ymin + .01, "max_lateral_m": ymax - .01}}


def calibrations():
    return {"front": calibration("front", (-.02, 2.0, -.45, .45)),
            "left": calibration("left", (-.5, 2.0, .05, 1.8)),
            "rear": calibration("rear", (-1.5, .15, -.6, .6))}


def clear_views(configs=None):
    return {name: Perception(np.ones(config["image_size"], dtype=np.float32))
            for name, config in (configs or calibrations()).items()}


def planner(configs=None, **options):
    return MultiViewPlanner({"horizon_m": 1.3, **options}, configs or calibrations())


def test_front_side_and_rear_each_contribute_projected_coverage():
    configs = calibrations()
    points = np.array([[.8, 0], [.2, .8], [-.5, 0]])
    masks = {name: perception.drivable for name, perception in clear_views().items()}
    assert planner()._ground_clear(points, masks) == (True, 1.0)
    for omitted in configs:
        partial = {name: value for name, value in configs.items() if name != omitted}
        assert not planner(partial)._ground_clear(points, {name: masks[name] for name in partial})[0]


def test_multiview_path_search_handles_straight_and_turn_footprints():
    multi = planner()
    straight = multi.plan(clear_views(), .2)
    left = multi.plan(clear_views(), .2, "left")
    assert straight.valid and straight.curvature == pytest.approx(0)
    assert left.valid and left.curvature > 0 and left.points[-1, 1] > 0
    assert 0 < left.target_speed_mps <= multi.max_speed_mps


def test_overlap_blocked_observation_cannot_be_overwritten_by_clear_view():
    configs = {"front": calibration("front", (-.3, 2.0, -1.3, 1.3)),
               "second": calibration("front", (-.3, 2.0, -1.3, 1.3))}
    views = clear_views(configs)
    views["second"].drivable[:] = 0
    for ordered in (configs, dict(reversed(list(configs.items())))):
        result = planner(ordered).plan(views, 0)
        assert not result.valid and result.reason == "all_paths_blocked_or_unknown"


@pytest.mark.parametrize("change", ["missing", "extra", "none"])
def test_exact_camera_set_is_required(change):
    views = clear_views()
    if change == "missing":
        del views["rear"]
    elif change == "extra":
        views["extra"] = views["front"]
    else:
        views = None
    assert planner().plan(views, 0).reason == "missing_or_unexpected_camera_views"


def test_side_detection_is_projected_and_blocks_the_swept_vehicle():
    configs = calibrations()
    multi = planner(configs)
    views = clear_views(configs)
    assert multi.plan(views, 0).valid
    # In this left-facing view, the bbox bottom maps to a segment along x.
    # It is close enough to the initial left side to block every candidate,
    # even though the front view contains no detections and is all drivable.
    pixels = multi.projectors["left"].ground_to_pixel([[.2, .15], [.5, .15]])
    x1, bottom = pixels[0]
    x2, _ = pixels[1]
    views["left"].detections = [Detection((float(x1), float(bottom - 20), float(x2), float(bottom)), .9, "person")]
    result = multi.plan(views, .1)
    assert not result.valid and result.reason == "all_paths_blocked_or_unknown"


def test_rear_detection_is_not_ignored_based_on_forward_camera_rules():
    configs = {"front": calibration("front", (-.3, 2, -1, 1)),
               "rear": calibration("rear", (-1.0, -.01, -.5, .5))}
    multi = planner(configs)
    views = clear_views(configs)
    pixels = multi.projectors["rear"].ground_to_pixel([[-.04, -.05], [-.04, .05]])
    x1, bottom = pixels[0]
    x2, _ = pixels[1]
    views["rear"].detections = [Detection((float(x1), float(bottom - 10), float(x2), float(bottom)), .9, "person")]
    assert not multi.plan(views, 0).valid


def test_all_unknown_or_insufficient_coverage_stops():
    views = clear_views()
    for perception in views.values():
        perception.drivable[:] = 0
    assert not planner().plan(views, 0).valid
    configs = {"rear": calibration("rear", (-1.5, -.1, -.8, .8))}
    assert not planner(configs).plan(clear_views(configs), 0).valid


def test_supported_coordinates_outside_actual_pixels_are_unknown():
    config = calibration("front", (-.3, .6, -.5, .5))
    config["supported_range"]["max_forward_m"] = 3
    configs = {"front": config}
    # A declared range alone cannot extend the camera's actual field of view.
    assert not planner(configs).plan(clear_views(configs), 0).valid


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_probability_anywhere_in_any_view_stops(value):
    views = clear_views()
    views["rear"].drivable[10, 10] = value
    assert planner().plan(views, 0).reason == "invalid_drivable_probability:rear"


def test_invalid_mask_shape_and_type_stop():
    views = clear_views()
    views["left"].drivable = np.ones((20, 20))
    assert planner().plan(views, 0).reason == "calibration_image_size_mismatch:left"
    views["left"].drivable = np.full((240, 320), "invalid", dtype=object)
    assert planner().plan(views, 0).reason == "invalid_drivable_probability:left"
    views["left"].drivable = [[1, 1], [1]]
    assert planner().plan(views, 0).reason == "invalid_drivable_probability:left"


@pytest.mark.parametrize("detection,reason", [
    (Detection((1, 2, float("nan"), 5), .9, "car"), "invalid_detection"),
    (Detection((-1, 2, 4, 5), .9, "car"), "invalid_detection"),
    (Detection((1, 2, 4, 5), 2, "car"), "invalid_detection"),
    (Detection((0, 2, 4, 5), .9, "car"), "obstacle_projection_unknown"),
])
def test_invalid_or_unprojectable_detection_stops(detection, reason):
    views = clear_views()
    views["left"].detections = [detection]
    assert planner().plan(views, 0).reason == f"{reason}:left"


def test_request_validation_and_stopping_distance():
    multi = planner()
    assert multi.plan(clear_views(), -1).reason == "invalid_speed"
    assert multi.plan(clear_views(), float("nan")).reason == "invalid_speed"
    assert multi.plan(clear_views(), 0, "reverse").reason == "invalid_route_hint"
    assert multi.plan(clear_views(), 0, "stop").reason == "route_stop"
    assert multi.plan(clear_views(), 5).reason == "speed_exceeds_visible_stopping_distance"


def test_shared_rear_axle_origin_must_be_explicit():
    configs = calibrations()
    del configs["left"]["coordinate_origin"]
    with pytest.raises(ValueError, match="rear_axle_center"):
        planner(configs)


def test_rear_only_and_asymmetric_lateral_calibrations():
    rear = GroundProjector(calibration("rear", (-2, -.1, -.8, .8)))
    assert rear.in_supported_range(np.array([[-1, 0], [.5, 0]])).tolist() == [True, False]
    left = GroundProjector(calibration("left", (-.5, 2, .2, 1.5)))
    assert left.in_supported_range(np.array([[0, .5], [0, -.5]])).tolist() == [True, False]
    partial = calibration("left", (-.5, 2, .2, 1.5))
    del partial["supported_range"]["min_lateral_m"]
    with pytest.raises(ValueError, match="supported_range"):
        GroundProjector(partial)


@pytest.mark.parametrize("view,bounds", [("rear", (-2, -.1, -.8, .8)), ("left", (-.5, 2, .2, 1.5))])
def test_measured_calibration_fitting_supports_rear_and_side(tmp_path, view, bounds):
    cfg = calibration(view, bounds)
    projector = GroundProjector(cfg)
    pixels = np.array([[0, 0], [319, 0], [319, 239], [0, 239]])
    heldout = np.array([[90, 80], [220, 150]])
    raw = copy.deepcopy(cfg)
    raw["points"] = np.column_stack((pixels, projector.pixel_to_ground(pixels))).tolist()
    raw["validation_points"] = np.column_stack((heldout, projector.pixel_to_ground(heldout))).tolist()
    source, output = tmp_path / "points.json", tmp_path / "calibration.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    fit_ground_calibration(source, output)
    fitted = GroundProjector(output)
    assert np.allclose(fitted.pixel_to_ground(heldout), projector.pixel_to_ground(heldout), atol=1e-5)
