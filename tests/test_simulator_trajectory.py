"""Synthetic route rollout geometry; no learned model or hardware is involved."""
import math

import numpy as np
import pytest

from camera_rc.simulator.lane import LaneCorridor
from camera_rc.simulator.speed import stopping_distance
from camera_rc.simulator.terrain import load_course
from camera_rc.simulator.trajectory import (
    LATERAL_ACCELERATION_MPS2, MAX_CURVATURE_PER_M, STEERING_RATE_RAD_S,
    WHEELBASE_M, pure_pursuit_rollout,
)


def _distances(length):
    return np.linspace(0., length, math.ceil(length/.1)+1)


def test_straight_rollout_preserves_actual_start_and_extends_final_tangent():
    distances = _distances(20.)
    world, headings = pure_pursuit_rollout([[0., 0.], [10., 0.]], 5., 0., 0., 0., distances, 10.)
    assert np.allclose(world[:, 0], 5.+distances, atol=1e-12)
    assert np.array_equal(world[:, 1], np.zeros(len(world)))
    assert np.array_equal(headings, np.zeros(len(world)))
    assert world[-1, 0] == pytest.approx(25.)


def test_vertical_repeated_route_points_and_heading_are_world_coordinates():
    distances = _distances(12.)
    world, headings = pure_pursuit_rollout([[0., 0.], [0., 0.], [0., 10.]],
                                         0., 2., math.pi/2, 0., distances, 0.)
    assert np.allclose(world[:, 0], 0., atol=1e-12)
    assert np.allclose(world[:, 1], 2.+distances, atol=1e-12)
    assert np.allclose(headings, math.pi/2)


def test_actual_offset_and_heading_are_not_snapped_to_reference():
    distances = _distances(5.)
    world, headings = pure_pursuit_rollout([[0., 0.], [100., 0.]],
                                         5., .4, -.05, .1, distances, 8.)
    assert np.array_equal(world[0], [5., .4]) and headings[0] == -.05
    assert world[1, 0] > 5. and world[1, 1] < .4
    assert np.allclose(np.linalg.norm(np.diff(world, axis=0), axis=1), np.diff(distances), atol=1e-12)
    assert np.max(np.abs(np.diff(headings))) <= MAX_CURVATURE_PER_M*.1+1e-12


def test_rollout_respects_steering_slew_from_actual_steering():
    distances = _distances(5.)
    initial = .2
    speed = 20.
    world, headings = pure_pursuit_rollout([[0., 0.], [100., 0.]],
                                         0., 0., 0., initial, distances, speed)
    # Midpoint bicycle headings recover the steering used for each spatial
    # step, including the first transition from the actual wheel angle.
    steering = np.arctan(WHEELBASE_M*np.diff(headings)/np.diff(distances))
    changes = np.diff(np.r_[initial, steering])
    allowed = STEERING_RATE_RAD_S*np.diff(distances)/speed
    assert np.all(np.abs(changes) <= allowed+1e-12)
    assert steering[0] == pytest.approx(initial-allowed[0])
    # The actual wheel angle may already exceed the requested lateral cap.
    # Keep that transition visible for the final safety review.
    assert abs(math.tan(steering[0])/WHEELBASE_M) > LATERAL_ACCELERATION_MPS2/(speed+.12)**2
    assert np.isfinite(world).all()


@pytest.mark.parametrize("speed", [12., 19.4])
def test_gangwon_variable_curve_rollout_stays_inside_assumed_lane(speed):
    route = np.asarray(load_course("gangwon")["route"]["points"], dtype=float)
    vectors = np.diff(route, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    progress = 1300.
    segment = int(np.searchsorted(cumulative, progress, side="right")-1)
    direction = vectors[segment]/lengths[segment]
    position = route[segment]+direction*(progress-cumulative[segment])
    yaw = math.atan2(direction[1], direction[0])
    distances = _distances(stopping_distance(speed)+2.)
    world, headings = pure_pursuit_rollout(route, *position, yaw, 0., distances, speed)
    assert LaneCorridor(route).allows_path(world, headings)
    assert np.array_equal(world[0], position) and headings[0] == yaw
    curvature = np.diff(headings)/np.diff(distances)
    limit = min(MAX_CURVATURE_PER_M, LATERAL_ACCELERATION_MPS2/(speed+.12)**2)
    assert np.max(np.abs(curvature)) <= limit+1e-12
    assert np.isfinite(world).all() and np.isfinite(headings).all()


@pytest.mark.parametrize("route, distances, state", [
    ([[0., 0.]], [0., .1], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [0., 0.]], [0., .1], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [float("nan"), 1.]], [0., .1], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [1., 0.]], [.1, .2], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [1., 0.]], [0., 0.], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [1., 0.]], [0., .2], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [1., 0.]], [0., float("inf")], (0., 0., 0., 0., 1.)),
    ([[0., 0.], [1., 0.]], [0., .1], (0., 0., float("nan"), 0., 1.)),
    ([[0., 0.], [1., 0.]], [0., .1], (0., 0., 0., math.pi/2, 1.)),
    ([[0., 0.], [1., 0.]], [0., .1], (0., 0., 0., 0., -1.)),
])
def test_invalid_route_samples_and_initial_state_are_rejected(route, distances, state):
    x, y, yaw, steering, speed = state
    with pytest.raises(ValueError):
        pure_pursuit_rollout(route, x, y, yaw, steering, distances, speed)
