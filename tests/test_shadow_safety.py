"""Synthetic evidence contracts only: no vehicle, models, or collision-rate claim."""
from dataclasses import replace
import math

import numpy as np
import pytest

from camera_rc.driving.shadow_safety import ShadowSafetySupervisor, Track
from camera_rc.driving.shadow_scenarios import CASES, altered, example


def review(policy=None, plan=None, state=None, scene=None):
    p, route, vehicle, evidence = example()
    return ShadowSafetySupervisor(policy or p).review(
        plan or route, state or vehicle, scene or evidence, 10.0)


@pytest.mark.parametrize("name,reason", CASES.items())
def test_fault_injection_and_clear_corridor(name, reason):
    policy, _, _, _ = example()
    supervisor = ShadowSafetySupervisor(policy)
    for seq in range(3):
        _, plan, state, scene = altered(name, seq, 10 + seq * .1)
        result = supervisor.review(plan, state, scene, 10 + seq * .1)
        assert result.actuation_enabled is False
    assert result.reasons == (reason,)
    assert result.proposal_eligible == (name == "clear_observed_corridor")


def test_fault_recovery_requires_fresh_evidence_and_explicit_stopped_reset():
    policy, plan, state, scene = altered("failed_detector")
    supervisor = ShadowSafetySupervisor(policy)
    assert supervisor.review(plan, state, scene, 10).mode == "LATCHED_STOP"
    for seq in range(1, 5):
        _, plan, state, scene = example(seq, 10 + seq * .1)
        # Even zero measured speed cannot establish a stop with 0.05 m/s error.
        result = supervisor.review(plan, replace(state, speed_mps=0), scene,
                                   10 + seq * .1, reset_requested=True)
        assert result.mode == "LATCHED_STOP"
    _, plan, state, scene = example(5, 10.5)
    result = supervisor.review(plan, replace(state, speed_mps=0),
                               replace(scene, speed_error_mps=.005), 10.5, reset_requested=True)
    assert result.mode == "LATCHED_STOP"
    _, plan, state, scene = example(6, 10.6)
    result = supervisor.review(plan, replace(state, speed_mps=0),
                               replace(scene, speed_error_mps=.001), 10.6)
    assert result.mode == "LATCHED_STOP"
    _, plan, state, scene = example(7, 10.7)
    result = supervisor.review(plan, replace(state, speed_mps=0),
                               replace(scene, speed_error_mps=.001), 10.7, reset_requested=True)
    assert result.proposal_eligible and not result.actuation_enabled


@pytest.mark.parametrize("reuse_capture", [True, False])
def test_replay_is_not_cured_by_refreshing_other_timestamp(reuse_capture):
    policy, plan, state, scene = example()
    supervisor = ShadowSafetySupervisor(policy)
    assert supervisor.review(plan, state, scene, 10).mode == "WARMUP_STOP"
    _, plan, state, next_scene = example(1, 10.1)
    key, value = ("captured_at_s", 9.97) if reuse_capture else ("sequence", 0)
    next_scene = replace(next_scene, modules=tuple(replace(m, **{key: value}) for m in next_scene.modules))
    result = supervisor.review(plan, state, next_scene, 10.1)
    assert result.reasons == (("frozen_or_reversed_capture" if reuse_capture else "replayed_observation"),)


@pytest.mark.parametrize("now", [10., 9.9, float("nan"), float("inf"), True, -1])
def test_bad_review_clock_latches(now):
    policy, plan, state, scene = example()
    supervisor = ShadowSafetySupervisor(policy)
    supervisor.review(plan, state, scene, 10)
    result = supervisor.review(plan, state, scene, now)
    assert result.mode == "LATCHED_STOP" and not result.actuation_enabled


@pytest.mark.parametrize("target", ["grid", "track", "plan", "state_clock"])
def test_common_ego_anchor_and_clock_are_required(target):
    _, _, state, scene = example()
    if target == "grid":
        scene = replace(scene, grid=replace(scene.grid, anchor_time_s=state.timestamp-.01))
    elif target == "track":
        scene = replace(scene, tracks=(Track("far", state.timestamp-.01, 9, 9, 0, 0, .2, 0, 0, 0),))
    elif target == "plan":
        scene = replace(scene, plan_anchor_time_s=state.timestamp-.01)
    else:
        scene = replace(scene, state_clock_id="other_clock")
    assert review(scene=scene).mode == "LATCHED_STOP"


def test_single_cell_at_body_edge_is_not_hidden_by_sparse_waypoints_or_confidence():
    _, _, _, scene = example()
    scene.grid.road_probability[:] = 1
    scene.grid.occupied[35, 35] = True  # y=0.55 m, away from path centerline
    assert review(scene=scene).reasons == ("occupied_swept_cell",)


def test_far_static_track_does_not_veto_clear_corridor():
    _, _, state, scene = example()
    scene = replace(scene, tracks=(Track("far", state.timestamp, 8, 2, 0, 0, .2, .1, .1, .1),))
    assert review(scene=scene).mode == "WARMUP_STOP"


def test_prediction_must_cover_age_from_ego_anchor():
    policy, plan, state, scene = example()
    delay = .33
    speed = .65
    window = delay + (speed + policy.acceleration_during_delay_mps2 * delay) / policy.min_braking_deceleration_mps2
    assert review(scene=replace(scene, prediction_horizon_s=window+.005)).reasons == ("prediction_horizon_too_short",)


def test_stop_distance_uses_upper_speed_acceleration_and_braking_bound():
    policy, plan, state, scene = example()
    result = review()
    expected = .65*.33 + .5*.3*.33**2 + (.65+.3*.33)**2/(2*.8) + .2
    assert math.isclose(result.required_stopping_distance_m, expected)
    slower_brake = review(policy=replace(policy, min_braking_deceleration_mps2=.5))
    faster = review(plan=replace(plan, target_speed_mps=.8))
    assert slower_brake.required_stopping_distance_m > expected
    assert faster.required_stopping_distance_m > expected


def test_observation_age_can_push_current_speed_above_odd():
    policy, plan, state, scene = example()
    state = replace(state, timestamp=9.91, speed_mps=.98)
    scene = replace(scene, speed_error_mps=.01, plan_anchor_time_s=state.timestamp,
                    grid=replace(scene.grid, anchor_time_s=state.timestamp))
    assert review(state=state, scene=scene).reasons == ("speed_upper_bound_outside_odd",)


def test_delay_acceleration_cannot_violate_speed_odd():
    _, plan, _, _ = example()
    assert review(plan=replace(plan, target_speed_mps=.9)).reasons == ("predicted_speed_outside_odd",)


def test_turning_uses_peak_speed_before_braking():
    policy, plan, _, _ = example()
    policy = replace(policy, max_speed_mps=3, acceleration_during_delay_mps2=4)
    plan = replace(plan, points=np.array([[0., 0.], [2., 0.], [4., .02]]))
    assert review(policy=policy, plan=plan).reasons == ("path_turning_bound",)


def test_long_segments_cannot_dilute_right_angle_corner():
    _, plan, _, _ = example()
    result = review(plan=replace(plan, points=np.array([[0, 0], [10, 0], [10, 10]])))
    assert result.reasons == ("path_turning_bound",)


def test_coarse_grid_cannot_dilute_right_angle_corner():
    _, plan, _, scene = example()
    scene = replace(scene, grid=replace(scene.grid, resolution_m=20., x_min_m=-30., y_min_m=-30.))
    result = review(plan=replace(plan, points=np.array([[0, 0], [10, 0], [10, 10]])), scene=scene)
    assert result.reasons == ("path_turning_bound",)


@pytest.mark.parametrize("field", ["speed_error_mps", "localization_error_m", "motion_compensation_error_m",
                                    "prediction_horizon_s", "plan_created_at_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_invalid_physical_evidence_never_passes(field, value):
    _, _, _, scene = example()
    assert review(scene=replace(scene, **{field: value})).mode == "LATCHED_STOP"


def test_grid_and_sampling_budgets_fail_before_expensive_sweeping():
    policy, _, _, _ = example()
    assert review(policy=replace(policy, max_grid_cells=10)).reasons == ("grid_layer_mismatch",)
    assert review(policy=replace(policy, max_dense_points=10)).reasons == ("path_sampling_budget_exceeded",)


def test_identity_pins_are_copied_and_empty_detections_are_explicitly_healthy():
    policy, plan, state, scene = example()
    supervisor = ShadowSafetySupervisor(policy)
    policy.artifacts["objects"] = "0" * 64
    assert scene.tracks == ()
    assert supervisor.review(plan, state, scene, 10).mode == "WARMUP_STOP"


@pytest.mark.parametrize("points", [[], [[0, 0]], [[0, 0], [float("nan"), 0]], [[0, 0], [-1, 0]],
                                    [[1, 0], [5, 0]], [[0, 0], [0, 0]]])
def test_malformed_or_unsupported_path_stops(points):
    _, plan, _, _ = example()
    assert review(plan=replace(plan, points=points)).mode == "LATCHED_STOP"


def test_legacy_perception_is_not_metric_scene_evidence():
    from camera_rc.driving.types import Perception
    assert review(scene=Perception(np.ones((4, 4), dtype=bool))).reasons == ("invalid_input_contract",)
