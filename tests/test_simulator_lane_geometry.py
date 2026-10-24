"""Assumed lane/body geometry, independent from learned road probabilities."""
import math

import numpy as np
import pytest

from camera_rc.simulator.lane import LaneCorridor


def test_centered_assumed_rectangle_fits_and_margin_only_changes_path_guard():
    lane = LaneCorridor([(0., 0.), (100., 0.)])
    measured = lane.measure(10., 0., 0.)
    assert measured == {"outside": False, "center_offset_m": 0.,
                        "max_body_offset_m": .9, "clearance_m": .9}
    assert lane.allows_path([[10., 0.], [20., 0.]])
    at_boundary = lane.measure(10., .9, 0.)
    assert not at_boundary["outside"]
    assert at_boundary["clearance_m"] == pytest.approx(0.)
    assert not lane.allows_path([[10., .9]], [0.])
    assert lane.measure(10., .9001, 0.)["outside"]
    assert lane.allows_path([[10., .8]], [0.])


def test_rectangle_sampling_includes_corners_and_never_exceeds_point_one_spacing():
    lane = LaneCorridor([(0., 0.), (100., 0.)])
    samples = lane.sample_points(0., 0., 0.)
    assert np.max(np.linalg.norm(np.roll(samples, -1, axis=0)-samples, axis=1)) <= .1+1e-12
    for corner in ((2.4, .9), (-1., .9), (-1., -.9), (2.4, -.9)):
        assert np.any(np.all(np.isclose(samples, corner), axis=1))
    assert np.max(np.linalg.norm(samples, axis=1)) < 2.7


def test_endpoint_tangent_extensions_cover_normal_goal_overshoot_but_are_bounded():
    lane = LaneCorridor([(0., 0.), (10., 0.)])
    assert np.array_equal(lane.extended_centerline[[0, -1]], [[-20., 0.], [30., 0.]])
    assert not lane.measure(-1., 0., 0.)["outside"]
    assert not lane.measure(11., 0., 0.)["outside"]
    assert lane.allows_path([[9., 0.], [11., 0.]])
    assert lane.measure(31., 0., 0.)["outside"]
    assert not lane.allows_path([[31., 0.]], [0.])


def test_diagonal_heading_detects_outside_front_corner_while_center_is_inside():
    lane = LaneCorridor([(0., 0.), (100., 0.)])
    result = lane.measure(10., 0., math.pi/4)
    assert result["center_offset_m"] == 0.
    assert result["outside"]
    assert result["max_body_offset_m"] == pytest.approx((2.4+.9)/math.sqrt(2.))
    assert not lane.allows_path([[10., 0.]], [math.pi/4])
    assert lane.measure(10., 0., -math.pi/4)["max_body_offset_m"] == pytest.approx(result["max_body_offset_m"])


def test_vertical_and_reversed_headings_use_world_rectangle_orientation():
    lane = LaneCorridor([(0., 0.), (0., 100.)])
    assert not lane.measure(0., 10., math.pi/2)["outside"]
    assert not lane.measure(0., 10., -math.pi/2)["outside"]
    assert lane.measure(0., 10., 0.)["outside"]
    assert lane.allows_path([[0., 10.], [0., 20.]])
    assert lane.allows_path([[0., 20.], [0., 10.]])


def test_full_perimeter_is_checked_even_when_corners_fit_a_nonconvex_buffer():
    lane = LaneCorridor([(-1., -.9), (-1., .9), (2.4, .9), (2.4, -.9)], width_m=1.2, margin_m=0.)
    corners = np.array(((2.4, .9), (-1., .9), (-1., -.9), (2.4, -.9)))
    assert np.max(lane.distance_to_centerline(corners)) == 0.
    assert lane.measure(0., 0., 0.)["outside"]
    assert not lane.allows_path([[0., 0.]], [0.])


def test_repeated_segments_do_not_change_geometry_or_produce_nan():
    lane = LaneCorridor([(0., 0.), (0., 0.), (10., 0.), (10., 0.), (20., 0.)])
    assert len(lane.centerline) == 3
    assert lane.measure(10., 0., 0.)["clearance_m"] == pytest.approx(.9)
    assert lane.allows_path([[5., 0.], [5., 0.], [10., 0.]])
    with pytest.raises(ValueError, match="nonzero"):
        LaneCorridor([(1., 1.), (1., 1.)])


def test_boundaries_are_assumed_display_geometry_and_defensive():
    lane = LaneCorridor([(0., 0.), (10., 0.)])
    boundaries = lane.boundaries
    assert np.allclose(np.asarray(boundaries["left"])[:, 1], 1.8)
    assert np.allclose(np.asarray(boundaries["right"])[:, 1], -1.8)
    boundaries["left"][0][0] = -999.
    assert lane.boundaries["left"][0][0] == -20.


def test_batched_path_checks_every_pose_and_handles_stationary_and_invalid_candidates():
    lane = LaneCorridor([(0., 0.), (4000., 0.)])
    points = np.column_stack((np.linspace(0., 3000., 3000), np.zeros(3000)))
    assert lane.allows_path(points)
    points[-1, 1] = 1.
    assert not lane.allows_path(points, np.zeros(3000))
    assert lane.allows_path([[10., 0.]])
    assert not lane.allows_path([])
    assert not lane.allows_path([[math.nan, 0.]])
    assert not lane.allows_path([[10., 0.]], [math.inf])


def test_nearest_culling_matches_all_segments_at_bends_ties_and_distant_points():
    # Many segments, a reversal, and nearby parallel portions exercise the box
    # culling without assuming that the closest segment follows route order.
    x = np.linspace(0., 120., 81)
    route = np.column_stack((x, 8.*np.sin(x/12.)))
    route = np.vstack((route, route[::-1]+[0., 3.]))
    lane = LaneCorridor(route)
    rng = np.random.default_rng(7)
    points = np.vstack((route, lane.extended_centerline,
                        rng.uniform([-25., -20.], [145., 20.], (1500, 2)),
                        [[-1000., 1000.], [1000., -1000.]]))
    delta = points[:, None, :]-lane._starts[None, :, :]
    along = np.clip(np.sum(delta*lane._vectors[None, :, :], axis=2)/lane._squared_lengths, 0., 1.)
    residual = delta-along[:, :, None]*lane._vectors[None, :, :]
    squared = np.sum(residual*residual, axis=2)
    expected_indices = np.argmin(squared, axis=1)
    expected_distances = np.sqrt(squared[np.arange(len(points)), expected_indices])
    actual, indices = lane._nearest(points)
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(actual, expected_distances)
    scalar, index = lane._nearest(points[0])
    assert scalar == expected_distances[0] and index == expected_indices[0]
    empty, indices = lane._nearest(np.empty((0, 2)))
    assert empty.shape == indices.shape == (0,)
