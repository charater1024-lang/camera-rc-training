"""Numerical session-metric checks independent from the simulator controller."""
import math

import pytest

from camera_rc.simulator.performance import PerformanceTracker


def record(tracker, **changes):
    sample = dict(time_s=1., dt_s=1., mode="autonomous", position=(5., 0.), yaw_rad=0.,
                  speed_mps=10., applied_target_mps=10., steering_rad=0., distance_delta_m=8.,
                  safety_mode="ALLOW", safety_eligible=True, intervention_delta=0,
                  collision_delta=0, offroad_delta=0)
    sample.update(changes)
    tracker.sample(**sample)


def test_metrics_use_duration_weighting_and_integrated_distance_for_mean_speed():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    record(tracker, position=(4., 3.), yaw_rad=math.pi/6, speed_mps=10., applied_target_mps=8.)
    record(tracker, time_s=4., dt_s=3., position=(8., 4.), yaw_rad=-math.pi/2,
           speed_mps=20., applied_target_mps=16., distance_delta_m=45.,
           safety_mode="LATCHED_STOP", safety_eligible=False, intervention_delta=2,
           collision_delta=1, offroad_delta=3)
    result = tracker.snapshot((8., 4.), -math.pi/2)
    auto = result["autonomous"]
    assert auto["samples"] == 2
    assert auto["duration_s"] == 4.
    assert auto["distance_m"] == 53.
    assert auto["cross_track_rmse_m"] == pytest.approx(math.sqrt((3.**2+3.*4.**2)/4.))
    assert auto["cross_track_max_m"] == 4.
    assert auto["heading_mae_deg"] == pytest.approx((30.+3.*90.)/4.)
    assert auto["speed_rmse_kmh"] == pytest.approx(math.sqrt((7.2**2+3.*14.4**2)/4.))
    assert auto["speed_mae_kmh"] == pytest.approx((7.2+3.*14.4)/4.)
    assert auto["mean_speed_kmh"] == pytest.approx(53./4.*3.6)
    assert auto["peak_speed_kmh"] == 72.
    assert auto["safety_eligible_pct"] == 25.
    assert auto["safety_stop_s"] == 3.
    assert auto["safety_interventions"] == 2
    assert auto["collision_events"] == 1
    assert auto["offroad_events"] == 3
    assert result["current"]["applied_target_kmh"] == 57.6
    assert result["current"]["acceleration_mps2"] == pytest.approx(10./3.)


def test_projection_ignores_repeated_points_and_clamps_progress_with_endpoint_error():
    tracker = PerformanceTracker([(0., 0.), (0., 0.), (10., 0.), (10., 0.), (10., 10.)])
    route = tracker.snapshot((12., 7.), 0.)["route"]
    assert route["length_m"] == 20.
    assert route["progress_m"] == 17.
    assert route["progress_pct"] == 85.
    assert route["remaining_m"] == 3.
    assert route["current_cross_track_error_m"] == 2.
    assert route["current_heading_error_deg"] == -90.
    before = tracker.snapshot((-3., -4.), 0.)["route"]
    assert before["progress_m"] == before["progress_pct"] == 0.
    assert before["current_cross_track_error_m"] == 5.
    after = tracker.snapshot((13., 14.), 0.)["route"]
    assert after["progress_m"] == 20.
    assert after["progress_pct"] == 100.
    assert after["remaining_m"] == 0.
    assert after["current_cross_track_error_m"] == 5.


def test_heading_wraps_across_pi_and_mae_uses_absolute_degrees():
    tracker = PerformanceTracker([(10., 0.), (0., 0.)])
    record(tracker, yaw_rad=math.radians(-179.))
    result = tracker.snapshot((5., 0.), math.radians(-179.))
    assert result["route"]["current_heading_error_deg"] == pytest.approx(1.)
    assert result["autonomous"]["heading_mae_deg"] == pytest.approx(1.)
    assert tracker.snapshot((5., 0.), math.radians(179.))["route"]["current_heading_error_deg"] == pytest.approx(-1.)


def test_manual_samples_do_not_pollute_autonomous_statistics_but_update_current_and_history():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    record(tracker, steering_rad=.1)
    record(tracker, time_s=3., dt_s=2., mode="manual", position=(6., 20.), yaw_rad=2.,
           speed_mps=20., applied_target_mps=0., steering_rad=.3, distance_delta_m=100.,
           safety_eligible=False, safety_mode="LATCHED_STOP", intervention_delta=10,
           collision_delta=10, offroad_delta=10)
    result = tracker.snapshot((6., 20.), 2.)
    auto = result["autonomous"]
    assert auto["samples"] == 1 and auto["duration_s"] == 1. and auto["distance_m"] == 8.
    assert auto["cross_track_rmse_m"] == auto["cross_track_max_m"] == 0.
    assert auto["speed_rmse_kmh"] == 0.
    assert auto["safety_eligible_pct"] == 100.
    assert auto["safety_stop_s"] == auto["safety_interventions"] == auto["collision_events"] == auto["offroad_events"] == 0
    assert [point["mode"] for point in result["history"]] == ["autonomous", "manual"]
    assert result["history"][-1]["target_kmh"] == 0.
    assert result["current"]["acceleration_mps2"] == 5.
    assert result["current"]["steering_rate_deg_s"] == pytest.approx(math.degrees(.2)/2.)
    assert result["route"]["current_cross_track_error_m"] == 20.


def test_unsampled_autonomous_values_are_null_and_new_tracker_resets_all_state():
    reference = [(0., 0.), (100., 0.)]
    tracker = PerformanceTracker(reference)
    initial = tracker.snapshot((0., 0.), 0.)
    auto = initial["autonomous"]
    for field in ("cross_track_rmse_m", "cross_track_max_m", "heading_mae_deg", "speed_rmse_kmh",
                  "speed_mae_kmh", "mean_speed_kmh", "peak_speed_kmh", "safety_eligible_pct"):
        assert auto[field] is None
    assert all(value is None for value in initial["current"].values())
    assert auto["samples"] == auto["duration_s"] == auto["distance_m"] == 0
    record(tracker, mode="manual")
    assert tracker.snapshot((0., 0.), 0.)["autonomous"] == auto
    record(tracker, time_s=2.)
    reset = PerformanceTracker(reference).snapshot((0., 0.), 0.)
    assert reset == initial


def test_warmup_is_in_eligibility_denominator_but_excluded_from_safety_stop_time():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    record(tracker, time_s=.5, dt_s=.5, speed_mps=0., applied_target_mps=0.,
           distance_delta_m=0., safety_eligible=False, safety_mode="WARMUP_STOP")
    record(tracker, time_s=2., dt_s=1.5, speed_mps=.02, applied_target_mps=0.,
           distance_delta_m=.01, safety_eligible=False, safety_mode="LATCHED_STOP")
    record(tracker, time_s=4., dt_s=2., speed_mps=.021, distance_delta_m=.03)
    auto = tracker.snapshot((5., 0.), 0.)["autonomous"]
    assert auto["safety_eligible_pct"] == 50.
    assert auto["safety_stop_s"] == 1.5
    assert auto["stopped_time_s"] == 2.


def test_history_is_bounded_decimated_and_does_not_trim_cumulative_statistics():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    for step in range(1, 2401):
        record(tracker, time_s=step*.1, dt_s=.1, distance_delta_m=.5)
    result = tracker.snapshot((5., 0.), 0.)
    history = result["history"]
    assert len(history) <= 240
    assert all(point["time_s"] >= 120. for point in history)
    assert all(b["time_s"]-a["time_s"] >= .5-1e-9 for a, b in zip(history, history[1:]))
    assert all(all(value is not None for value in point.values()) for point in history)
    assert result["autonomous"]["samples"] == 2400
    assert result["autonomous"]["duration_s"] == pytest.approx(240.)
    assert result["autonomous"]["distance_m"] == 1200.
    record(tracker, time_s=500., dt_s=.1)
    assert len(tracker.snapshot((5., 0.), 0.)["history"]) == 1


def test_short_manual_interval_is_kept_and_mixed_session_survives_history_expiry():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    record(tracker, time_s=.1, dt_s=.1)
    record(tracker, time_s=.2, dt_s=.1, mode="manual", applied_target_mps=0.)
    record(tracker, time_s=.3, dt_s=.1)
    result = tracker.snapshot((5., 0.), 0.)
    assert [point["time_s"] for point in result["history"]] == [.1, .2, .3]
    assert [point["mode"] for point in result["history"]] == ["autonomous", "manual", "autonomous"]
    assert result["session"] == {"total_samples": 3, "manual_samples": 1,
                                 "mixed_mode_session": True, "modes_seen": ["autonomous", "manual"]}
    record(tracker, time_s=130., dt_s=.1)
    expired = tracker.snapshot((5., 0.), 0.)
    assert len(expired["history"]) == 1
    assert expired["history"][0]["mode"] == "autonomous"
    assert expired["session"] == {"total_samples": 4, "manual_samples": 1,
                                  "mixed_mode_session": True, "modes_seen": ["autonomous", "manual"]}
    reset = PerformanceTracker([(0., 0.), (100., 0.)]).snapshot((5., 0.), 0.)
    assert reset["session"] == {"total_samples": 0, "manual_samples": 0,
                                "mixed_mode_session": False, "modes_seen": []}


def test_snapshot_is_pure_and_every_mutable_layer_is_independent():
    tracker = PerformanceTracker([(0., 0.), (100., 0.)])
    record(tracker)
    original = tracker.snapshot((5., 0.), 0.)
    changed = tracker.snapshot((5., 0.), 0.)
    changed["route"]["progress_m"] = -1
    changed["session"]["total_samples"] = -1
    changed["session"]["modes_seen"].append("bad")
    changed["autonomous"]["samples"] = -1
    changed["current"]["applied_target_kmh"] = -1
    changed["history"][0]["speed_kmh"] = -1
    changed["history"].append({"bad": True})
    changed["definitions"]["scope"] = "bad"
    assert tracker.snapshot((5., 0.), 0.) == original
    tracker.snapshot((99., 10.), 2.)
    assert tracker.snapshot((5., 0.), 0.) == original


def test_zero_length_only_reference_cannot_produce_divisions_by_zero():
    with pytest.raises(ValueError, match="nonzero"):
        PerformanceTracker([(0., 0.), (0., 0.)])
