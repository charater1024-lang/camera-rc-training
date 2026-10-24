"""Synthetic lane integration; no learned lane accuracy or vehicle validation."""
import json
import math

import numpy as np
import pytest

from camera_rc.driving.types import Perception
from camera_rc.simulator import Engine


def drive(engine, count, controls=None):
    for _ in range(count):
        snapshot = engine.step(controls)
        if snapshot["finished"]:
            break
    return engine.snapshot()


def departure_events(snapshot):
    return [event for event in snapshot["events"] if event["kind"] == "lane_departure"]


def assert_simulation_only(engine, snapshot):
    assert snapshot["safety"]["actuation_enabled"] is False
    assert engine.report()["configuration"]["real_vehicle_actuation_enabled"] is False


def test_initial_lane_has_assumed_provenance_empty_statistics_and_untrained_identity():
    engine = Engine()
    snapshot = engine.snapshot()
    lane = snapshot["lane"]
    assert lane["source"] == "synthetic_assumed_lane_not_measured"
    assert lane["width_m"] == 3.6
    assert lane["current"]["outside"] is False
    assert lane["current"]["center_offset_m"] == pytest.approx(0.)
    assert lane["current"]["max_body_offset_m"] == pytest.approx(.9)
    assert lane["current"]["clearance_m"] == pytest.approx(.9)
    autonomous = lane["autonomous"]
    assert autonomous["samples"] == autonomous["duration_s"] == 0
    assert autonomous["inside_pct"] is None
    assert autonomous["departure_events"] == autonomous["outside_time_s"] == 0
    assert lane["training"]["status"] == "not_trained"
    assert lane["training"]["model_loaded"] is False
    assert lane["training"]["label_space"] == "ego_lane_v1"
    world_lane = snapshot["world"]["lane"]
    assert world_lane["source"] == lane["source"]
    assert world_lane["width_m"] == lane["width_m"]
    left, right = np.array(world_lane["left_boundary"]), np.array(world_lane["right_boundary"])
    assert left.shape == right.shape and left.shape[1] == 2
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.allclose(left[:, 1], 1.8) and np.allclose(right[:, 1], -1.8)
    assert_simulation_only(engine, snapshot)


@pytest.mark.parametrize("scenario", ["clear", "curved"])
def test_basic_autonomous_goal_regression_retains_lane_and_road_separation(scenario):
    engine = Engine(scenario)
    snapshot = drive(engine, 260)
    assert snapshot["finished"] and snapshot["metrics"]["goal_reached"]
    assert snapshot["metrics"]["collisions"] == snapshot["metrics"]["offroad_events"] == 0
    assert snapshot["vehicle"]["speed_mps"] <= .02
    autonomous = snapshot["lane"]["autonomous"]
    assert autonomous["samples"] == snapshot["step"]
    assert autonomous["duration_s"] == pytest.approx(snapshot["sim_time_s"])
    assert autonomous["inside_pct"] == pytest.approx(100.)
    assert autonomous["departure_events"] == autonomous["outside_time_s"] == 0
    assert not departure_events(snapshot)
    assert_simulation_only(engine, snapshot)


@pytest.mark.parametrize("scenario", ["seoul_day", "busan_day", "gwangju_day", "suncheon_day", "gangwon_day"])
def test_regional_lane_schema_and_short_autonomous_sampling(scenario):
    engine = Engine(scenario)
    snapshot = drive(engine, 5)
    lane = snapshot["lane"]
    assert lane["source"] == "synthetic_assumed_lane_not_measured"
    assert lane["width_m"] == 3.6
    assert lane["autonomous"]["samples"] == 5
    assert lane["autonomous"]["duration_s"] == pytest.approx(.5)
    assert 0. <= lane["autonomous"]["inside_pct"] <= 100.
    assert np.isfinite(np.asarray(snapshot["world"]["lane"]["left_boundary"])).all()
    assert_simulation_only(engine, snapshot)


def test_corner_departure_is_detected_with_center_in_lane_and_whole_vehicle_on_road():
    engine = Engine()
    engine._car.yaw_rad = math.pi/4
    snapshot = engine.step()
    lane = snapshot["lane"]
    assert lane["current"]["center_offset_m"] == pytest.approx(0.)
    assert lane["current"]["max_body_offset_m"] > lane["width_m"]/2
    assert lane["current"]["clearance_m"] < 0
    assert lane["current"]["outside"] is True
    assert snapshot["metrics"]["offroad_events"] == 0
    assert snapshot["metrics"]["collisions"] == 0
    assert lane["autonomous"]["departure_events"] == 1
    assert lane["autonomous"]["outside_time_s"] == pytest.approx(engine.DT_S)
    assert lane["autonomous"]["inside_pct"] == pytest.approx(0.)
    assert len(departure_events(snapshot)) == 1
    assert_simulation_only(engine, snapshot)


def test_departure_counts_entry_transitions_and_resets_without_affecting_road_events():
    engine = Engine()
    engine._car.y_m = 1.2
    first = engine.step()
    second = engine.step()
    assert first["lane"]["autonomous"]["departure_events"] == 1
    assert second["lane"]["autonomous"]["departure_events"] == 1
    engine._car.y_m = 0.
    engine._car.yaw_rad = 0.
    engine.step()
    engine._car.y_m = 1.2
    returned = engine.step()
    assert returned["lane"]["autonomous"]["departure_events"] == 2
    assert returned["metrics"]["offroad_events"] == 0
    assert len(departure_events(returned)) == 2
    reset = engine.reset()
    assert reset["lane"]["autonomous"]["samples"] == 0
    assert reset["lane"]["autonomous"]["inside_pct"] is None
    assert reset["lane"]["autonomous"]["departure_events"] == 0
    assert reset["lane"]["autonomous"]["outside_time_s"] == 0
    assert not departure_events(reset)


def test_manual_override_keeps_departure_visible_but_excludes_manual_time_and_events():
    engine = Engine(mode="manual")
    engine._car.y_m = 1.2
    manual = engine.step({"throttle": 1., "brake": 0.})
    assert manual["lane"]["guard"]["allowed"] is False
    assert manual["lane"]["guard"]["reason"] == "lane_boundary_veto"
    assert manual["command"]["reason"] == "manual_simulation_override"
    assert manual["vehicle"]["speed_mps"] > 0
    assert manual["lane"]["current"]["outside"] is True
    assert len(departure_events(manual)) == 1
    assert manual["lane"]["autonomous"]["samples"] == 0
    assert manual["lane"]["autonomous"]["inside_pct"] is None
    assert manual["lane"]["autonomous"]["outside_time_s"] == 0
    assert manual["lane"]["autonomous"]["departure_events"] == 0
    engine._car.y_m = 0.
    engine._car.yaw_rad = 0.
    engine._car.speed_mps = engine._car.steering_rad = 0.
    engine.set_mode("autonomous")
    returned = engine.step()
    autonomous = returned["lane"]["autonomous"]
    assert autonomous["samples"] == 1
    assert autonomous["duration_s"] == pytest.approx(engine.DT_S)
    assert autonomous["outside_time_s"] == autonomous["departure_events"] == 0
    assert autonomous["inside_pct"] == pytest.approx(100.)
    assert_simulation_only(engine, returned)


def test_outside_duration_uses_substep_crossing_time_instead_of_whole_outer_step():
    engine = Engine()
    # Initially the rectangle is inside (0.7 + 0.9 < 1.8). Its front corner
    # crosses later during this step as the moving body continues turning.
    engine._car.y_m = .7
    engine._car.speed_mps = 8.
    engine._car.steering_rad = .45
    snapshot = engine.step()
    lane = snapshot["lane"]
    assert lane["current"]["outside"] is True
    assert lane["autonomous"]["departure_events"] == 1
    assert 0. < lane["autonomous"]["outside_time_s"] < engine.DT_S
    assert 0. < lane["autonomous"]["inside_pct"] < 100.
    assert lane["autonomous"]["duration_s"] == pytest.approx(engine.DT_S)
    assert snapshot["metrics"]["offroad_events"] == 0


def test_normal_goal_overshoot_is_not_a_lane_departure_or_invalid_guard():
    engine = Engine()
    engine._car.x_m = 37.
    snapshot = engine.step()
    assert snapshot["metrics"]["goal_reached"]
    assert snapshot["lane"]["current"]["outside"] is False
    assert snapshot["lane"]["guard"]["allowed"] is True
    assert snapshot["lane"]["guard"]["reason"] is None
    assert snapshot["lane"]["autonomous"]["departure_events"] == 0
    assert snapshot["lane"]["autonomous"]["inside_pct"] == pytest.approx(100.)


def test_guard_margin_can_veto_before_actual_body_departs_lane():
    engine = Engine()
    engine._car.y_m = .85
    snapshot = engine.step()
    assert snapshot["lane"]["current"]["outside"] is False
    assert snapshot["lane"]["current"]["clearance_m"] == pytest.approx(.05)
    assert snapshot["lane"]["guard"]["allowed"] is False
    assert snapshot["lane"]["guard"]["reason"] == "lane_boundary_veto"
    assert snapshot["command"]["brake"] == 1.
    assert snapshot["lane"]["autonomous"]["departure_events"] == 0
    assert snapshot["lane"]["autonomous"]["outside_time_s"] == 0


def test_positive_learned_road_mask_cannot_bypass_independent_lane_veto():
    engine = Engine(perception=lambda rgb: Perception(np.ones(rgb.shape[:2], dtype=np.float32)))
    engine.step()  # Allow the independent supervisor's normal initial warmup.
    engine._car.y_m = 1.2
    snapshot = engine.step()
    assert snapshot["perception"]["learned"] is True
    assert snapshot["perception"]["model_error"] is None
    assert snapshot["lane"]["training"]["model_loaded"] is False
    assert snapshot["lane"]["guard"]["allowed"] is False
    assert snapshot["lane"]["guard"]["reason"] == "lane_boundary_veto"
    assert snapshot["command"]["brake"] == 1.
    assert snapshot["vehicle"]["speed_mps"] <= .02
    assert snapshot["metrics"]["offroad_events"] == 0
    assert_simulation_only(engine, snapshot)


def test_allowed_lane_path_cannot_bypass_independent_semantic_rejection():
    engine = Engine(perception=lambda rgb: Perception(np.zeros(rgb.shape[:2], dtype=np.float32)))
    snapshot = drive(engine, 3)
    assert snapshot["lane"]["guard"]["allowed"] is True
    assert snapshot["lane"]["guard"]["reason"] is None
    assert snapshot["safety"]["proposal_eligible"] is False
    assert snapshot["safety"]["reasons"] == ["semantic_disagreement_or_nonroad"]
    assert snapshot["command"]["brake"] == 1.
    assert snapshot["vehicle"]["speed_mps"] == 0.
    assert_simulation_only(engine, snapshot)


def test_lane_data_is_serializable_and_report_queries_do_not_accumulate_samples(tmp_path):
    engine = Engine(output_dir=tmp_path/"lane-run")
    snapshot = engine.step()
    for _ in range(3):
        assert engine.snapshot()["lane"] == snapshot["lane"]
        report = json.loads(json.dumps(engine.report(), allow_nan=False))
        assert report["snapshot"]["lane"] == snapshot["lane"]
    records = (tmp_path/"lane-run"/"session_0000"/"steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["lane"] == snapshot["lane"]
    engine.close()


def test_excessive_goal_overshoot_is_vetoed_when_stopping_path_exceeds_lane_runoff():
    from camera_rc.simulator.engine import _polyline_position
    engine = Engine('gwangju_day', speed_mode='user', target_speed_kmh=150)
    point, direction = _polyline_position(engine._road, engine._route_length-15.)
    engine._car.x_m, engine._car.y_m = map(float, point)
    engine._car.yaw_rad = math.atan2(direction[1], direction[0])
    engine._car.speed_mps = 15.
    snapshot = engine.step()
    assert snapshot['lane']['current']['outside'] is False
    assert snapshot['lane']['guard']['reason'] == 'lane_boundary_veto'
    assert snapshot['safety']['proposal_eligible'] is False
    assert snapshot['command']['brake'] == 1.
