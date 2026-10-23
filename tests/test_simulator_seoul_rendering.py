"""Synthetic map geometry checks; no map download or learned inference."""

import numpy as np
import pytest

from camera_rc.simulator import rendering


VEHICLE = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0, "speed_mps": 0.0}
ROUTE = np.array([[-20.0, 0.0], [100.0, 0.0]])
MAIN_ROAD = {"id": "main", "points": ROUTE.tolist(), "half_width_m": 3.0}


def render(*, obstacles=(), condition="day", **kwargs):
    return rendering.render_observation(VEHICLE, ROUTE, 3.0, obstacles, condition, 17, **kwargs)


def rectangle(x0, x1, y0, y1, *, height=8.0, identity="building"):
    return {"id": identity, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "height_m": height}


def actor(x, y=0.0):
    return {"id": "person", "x_m": x, "y_m": y, "radius_m": 0.4, "kind": "pedestrian"}


def pixel(x, y):
    u, v, valid = rendering.project_ground(np.asarray(x), np.asarray(y))
    assert valid
    return round(float(v)), round(float(u))


def test_none_map_arguments_preserve_original_scene():
    for condition in ("day", "night", "fog"):
        original = render(condition=condition, obstacles=[actor(12)])
        explicit = render(condition=condition, obstacles=[actor(12)], roads=None, buildings=None)
        for first, second in zip(original, explicit):
            np.testing.assert_array_equal(first, second)


def test_map_roads_form_union_and_replace_route_only_ground():
    cross = {"id": "cross", "points": [[10, -25], [10, 25]], "half_width_m": 2.0}
    _, union = render(roads=[MAIN_ROAD, cross])
    _, only_cross = render(roads=[cross])
    assert union[pixel(10, 7)] == 1.0
    assert union[pixel(3, 0)] == 1.0
    assert only_cross[pixel(3, 0)] == 0.0
    assert union[pixel(3, 7)] == 0.0
    _, empty = render(roads=[])
    assert not empty.any()


def test_unsupported_elevation_is_omitted_and_footways_are_nonroad():
    bridge = {**MAIN_ROAD, "ground_geometry_supported": False}
    empty = render(roads=[])
    unsupported = render(roads=[bridge])
    for first, second in zip(empty, unsupported):
        np.testing.assert_array_equal(first, second)
    footway = {**MAIN_ROAD, "drivable": False}
    image, mask = render(roads=[footway])
    assert not mask.any()
    assert np.any(image != empty[0])
    _, overlapping = render(roads=[footway, MAIN_ROAD])
    assert overlapping[pixel(10, 0)] == 1.0


def test_building_facade_and_footprint_remove_road_probability():
    building = rectangle(10, 16, -3, 3)
    image, mask = render(roads=[MAIN_ROAD], buildings=[building])
    _, clear_mask = render(roads=[MAIN_ROAD])
    assert clear_mask[pixel(12, 0)] == 1.0
    assert mask[pixel(12, 0)] == 0.0
    # A facade also hides road beyond the physical building footprint.
    assert clear_mask[pixel(30, 0)] == 1.0
    assert mask[pixel(30, 0)] == 0.0
    assert mask[pixel(4, 0)] == 1.0
    assert image.shape == (180, 320, 3) and image.dtype == np.uint8
    assert mask.dtype == np.float32 and np.isfinite(mask).all()


@pytest.mark.parametrize("condition", ["day", "night", "fog"])
def test_building_occludes_actor_behind_but_actor_in_front_remains_visible(condition):
    building = rectangle(10, 16, -4, 4)
    clean = render(roads=[MAIN_ROAD], buildings=[building], condition=condition)
    behind = render(roads=[MAIN_ROAD], buildings=[building], obstacles=[actor(24)], condition=condition)
    front = render(roads=[MAIN_ROAD], buildings=[building], obstacles=[actor(5)], condition=condition)
    np.testing.assert_array_equal(clean[0], behind[0])
    np.testing.assert_array_equal(clean[1], behind[1])
    assert np.any(front[0] != clean[0])


def test_depth_result_does_not_depend_on_building_order():
    near = rectangle(9, 13, -3, 3, height=5, identity="near")
    far = rectangle(20, 25, -5, 5, height=16, identity="far")
    forward = render(roads=[MAIN_ROAD], buildings=[near, far], obstacles=[actor(6), actor(30)])
    reversed_order = render(roads=[MAIN_ROAD], buildings=[far, near], obstacles=[actor(30), actor(6)])
    for first, second in zip(forward, reversed_order):
        np.testing.assert_array_equal(first, second)


def test_concave_footprint_and_low_roof_keep_notch_open():
    polygon = np.array([[8, -4], [16, -4], [16, 4], [12, 4], [12, 0], [8, 0]])
    np.testing.assert_array_equal(
        rendering.point_in_polygon(np.array([10, 10, 14, 12]), np.array([-2, 2, 2, 2]), polygon),
        [True, False, True, True],
    )
    building = {"id": "low-concave", "polygon": polygon.tolist(), "height_m": 1.0}
    image, mask = render(roads=[{"id": "wide", "points": ROUTE.tolist(), "half_width_m": 6}], buildings=[building])
    assert mask[pixel(10, -2)] == 0
    assert mask[pixel(10, 2)] == 1
    assert image.dtype == np.uint8


def test_near_plane_crossing_building_is_clipped_without_invalid_projection():
    crossing = rectangle(-9, -3, -1, 3, height=9)
    clear = render(roads=[MAIN_ROAD])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        image, mask = render(roads=[MAIN_ROAD], buildings=[crossing])
    assert np.isfinite(image).all() and np.isfinite(mask).all()
    assert np.any(image != clear[0])
    assert np.count_nonzero(mask) < np.count_nonzero(clear[1])


def test_hundreds_of_remote_features_do_not_change_local_render():
    local = rectangle(15, 20, 6, 12, identity="local")
    expected = render(roads=[MAIN_ROAD], buildings=[local])
    buildings = [rectangle(1000 + i * 4, 1002 + i * 4, -4, 4, identity=f"remote-{i}") for i in range(300)]
    roads = [{"id": f"remote-road-{i}", "points": [[1000 + i * 4, -40], [1000 + i * 4, 40]], "half_width_m": 2} for i in range(300)]
    actual = render(roads=[MAIN_ROAD, *roads], buildings=[local, *buildings])
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("building", [
    {"polygon": [[0, 0], [1, 0], [1, float("nan")]], "height_m": 8},
    {"polygon": [[0, 0], [1, 0], [1, 1]], "height_m": float("inf")},
    {"polygon": [[0, 0], [1, 0], [1, 1]], "height_m": -1},
])
def test_invalid_building_geometry_is_rejected(building):
    with pytest.raises(ValueError):
        render(roads=[MAIN_ROAD], buildings=[building])
