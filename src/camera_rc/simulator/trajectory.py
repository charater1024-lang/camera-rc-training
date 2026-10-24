"""Smooth synthetic route proposals using the simulator's assumed bicycle model.

This is a geometric proposal generator. Callers must independently review the
whole path against lane, road, occupancy, stopping and dynamic constraints.
"""
from __future__ import annotations

import math

import numpy as np


WHEELBASE_M = 2.6
STEERING_RATE_RAD_S = .8
MAX_CURVATURE_PER_M = .12
LATERAL_ACCELERATION_MPS2 = 1.4
MAX_SAMPLE_SPACING_M = .1


def pure_pursuit_rollout(route, x_m, y_m, yaw_rad, steering_rad, distances, speed_mps):
    """Return world positions and continuous headings from the actual pose.

    ``distances`` starts at zero and strictly increases at intervals <=0.1 m.
    The initial pose and steering come from the simulated vehicle, rather than
    snapping to the reference route. Steering follows the controller's
    ``4 + 0.2 * speed`` lookahead and its 0.8 rad/s slew bound. A constant
    planning speed determines each spatial step's elapsed time, with a 0.1 m/s
    floor at rest. The final route tangent continues beyond its endpoint.
    Desired curvature also reserves a 1.4 m/s^2 lateral-acceleration budget at
    ``speed + 0.12``. Existing steering is not snapped to this limit: its slew
    transition remains subject to the caller's independent dynamic veto.

    No lane or safety veto is weakened by this function: a path beginning
    outside the lane or requiring an unsupported manoeuvre can still fail the
    caller's independent checks.
    """
    points = np.asarray(route, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError("Route requires at least two finite (x, y) points")
    points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0., axis=1)]]
    if len(points) < 2:
        raise ValueError("Route requires a nonzero length segment")
    samples = np.asarray(distances, dtype=np.float64)
    if (samples.ndim != 1 or len(samples) < 2 or not np.isfinite(samples).all()
            or samples[0] != 0.):
        raise ValueError("Distances require finite increasing samples starting at zero")
    spacing = np.diff(samples)
    if (spacing <= 0.).any() or (spacing > MAX_SAMPLE_SPACING_M+1e-9).any():
        raise ValueError("Distance sample spacing must be positive and at most 0.1 m")
    if any(isinstance(value, (bool, np.bool_)) for value in (x_m, y_m, yaw_rad, steering_rad, speed_mps)):
        raise ValueError("Initial pose, steering and speed must be finite numbers")
    state = np.asarray((x_m, y_m, yaw_rad, steering_rad, speed_mps), dtype=np.float64)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("Initial pose, steering and speed must be finite numbers")
    yaw_rad, steering_rad, speed_mps = map(float, state[2:])
    if speed_mps < 0. or abs(steering_rad) >= math.pi/2.:
        raise ValueError("Forward speed must be nonnegative and steering must be inside +/-pi/2")

    # Cache route geometry once per proposal. Each projection evaluates every
    # segment in NumPy, including nearby or returning route portions.
    starts = points[:-1]
    vectors = np.diff(points, axis=0)
    squared_lengths = np.sum(vectors*vectors, axis=1)
    lengths = np.sqrt(squared_lengths)
    cumulative = np.r_[0., np.cumsum(lengths)]
    lookahead = 4.+.2*float(speed_mps)
    curvature_limit = min(MAX_CURVATURE_PER_M,
                          LATERAL_ACCELERATION_MPS2/(speed_mps+.12)**2)
    elapsed_per_m = 1./max(float(speed_mps), .1)
    position = state[:2].copy()
    yaw, steering = float(yaw_rad), float(steering_rad)
    world = np.empty((len(samples), 2), dtype=np.float64)
    headings = np.empty(len(samples), dtype=np.float64)
    world[0], headings[0] = position, yaw
    for index, step in enumerate(spacing, 1):
        relative = position-starts
        fractions = np.clip(np.sum(relative*vectors, axis=1)/squared_lengths, 0., 1.)
        residual = relative-fractions[:, None]*vectors
        nearest = int(np.argmin(np.sum(residual*residual, axis=1)))
        progress = cumulative[nearest]+fractions[nearest]*lengths[nearest]
        target_progress = progress+lookahead
        target_segment = min(len(lengths)-1, int(np.searchsorted(cumulative, target_progress, side="right")-1))
        target = starts[target_segment]+vectors[target_segment]*(
            (target_progress-cumulative[target_segment])/lengths[target_segment])
        dx, dy = target-position
        local_y = -math.sin(yaw)*dx+math.cos(yaw)*dy
        curvature = float(np.clip(2.*local_y/max(1., dx*dx+dy*dy),
                                  -curvature_limit, curvature_limit))
        desired_steering = math.atan(WHEELBASE_M*curvature)
        steering_delta = STEERING_RATE_RAD_S*float(step)*elapsed_per_m
        steering += float(np.clip(desired_steering-steering, -steering_delta, steering_delta))
        turn = math.tan(steering)/WHEELBASE_M*float(step)
        midpoint_heading = yaw+turn/2.
        position += float(step)*np.array((math.cos(midpoint_heading), math.sin(midpoint_heading)))
        yaw += turn
        world[index], headings[index] = position, yaw
    return world, headings
