"""Speed contracts and synthetic outcomes; no hardware or learned safety claim."""
import math

import numpy as np
import pytest

from camera_rc.driving.types import Perception
from camera_rc.simulator import Engine, SCENARIOS
from camera_rc.simulator.engine import _polyline_position
from camera_rc.simulator.server import SimulatorApp


NEW_SCENARIOS = ("busan_day", "gwangju_day", "suncheon_day", "gangwon_day")


def _change_speed(api, mode, target):
    if api == "constructor":
        return Engine(speed_mode=mode, target_speed_kmh=target)
    engine = Engine()
    if api == "engine":
        return engine.set_speed(mode, target)
    app = SimulatorApp(engine, SCENARIOS)
    return app.action({"action": "speed", "speed_mode": mode, "target_speed_kmh": target})


@pytest.mark.parametrize("api", ["constructor", "engine", "app"])
@pytest.mark.parametrize("target", [math.nan, math.inf, -math.inf, True, False, -.01, 150.001, "100"])
def test_invalid_speed_targets_are_rejected_at_each_boundary(api, target):
    with pytest.raises(ValueError):
        _change_speed(api, "user", target)


@pytest.mark.parametrize("api", ["constructor", "engine", "app"])
@pytest.mark.parametrize("mode", ["manual", "", None, True, 0])
def test_invalid_speed_modes_are_rejected_at_each_boundary(api, mode):
    with pytest.raises(ValueError):
        _change_speed(api, mode, 50)


@pytest.mark.parametrize("mode", ["automatic", "user"])
@pytest.mark.parametrize("target", [0, 7.2, 150])
def test_valid_settings_are_disclosed_in_si_independent_snapshot(mode, target):
    snapshot = Engine(speed_mode=mode, target_speed_kmh=target).snapshot()
    speed = snapshot["speed_control"]
    assert speed["mode"] == mode
    assert speed["target_speed_kmh"] == pytest.approx(target)
    assert 0 <= speed["effective_target_kmh"] <= target
    assert speed["max_speed_kmh"] == 150
    assert speed["synthetic_only"] and speed["user_setting_is_ceiling"]
    assert snapshot["safety"]["actuation_enabled"] is False


@pytest.mark.parametrize("scenario", NEW_SCENARIOS)
def test_new_regional_courses_initialize_and_reset_without_starting(scenario):
    assert scenario in {item["id"] for item in SCENARIOS}
    engine = Engine(scenario=scenario, seed=73)
    initial = engine.snapshot()
    assert initial["step"] == 0 and initial["vehicle"]["speed_mps"] == 0
    assert initial["world"]["route_length_m"] > 3000
    assert initial["world"]["region"]["geometry_source"] == "synthetic_regional_course"
    assert initial["speed_control"]["target_speed_kmh"] == 150
    engine.set_speed("user", 63)
    reset = engine.reset(scenario=scenario, seed=73)
    assert reset["step"] == 0 and reset["vehicle"]["speed_mps"] == 0
    assert reset["speed_control"]["mode"] == "user"
    assert reset["speed_control"]["target_speed_kmh"] == 63
    assert reset["world"]["region"]["geometry_sha256"] == initial["world"]["region"]["geometry_sha256"]


def test_speed_action_updates_setting_without_advancing_or_enabling_physical_output():
    engine = Engine("gwangju_day")
    app = SimulatorApp(engine, SCENARIOS)
    before = app.snapshot()
    changed = app.action({"action": "speed", "speed_mode": "user", "target_speed_kmh": 120})
    assert changed["step"] == before["step"]
    assert changed["sim_time_s"] == before["sim_time_s"]
    assert changed["vehicle"] == before["vehicle"]
    assert not changed["running"] and not changed["physical_output"]
    assert changed["speed_control"]["mode"] == "user"
    assert changed["speed_control"]["target_speed_kmh"] == 120
    with pytest.raises(ValueError):
        app.action({"action": "speed", "speed_mode": "user", "target_speed_kmh": None})
    assert app.snapshot()["speed_control"] == changed["speed_control"]


def test_default_legacy_course_keeps_seven_point_two_kmh_target():
    assert Engine().snapshot()["speed_control"]["effective_target_kmh"] == pytest.approx(7.2)


@pytest.mark.parametrize("scenario,expected", [("gwangju_day", 150), ("gangwon_day", 70)])
def test_automatic_region_advisory_limits_requested_150(scenario, expected):
    speed = Engine(scenario, speed_mode="automatic", target_speed_kmh=150).snapshot()["speed_control"]
    assert speed["effective_target_kmh"] == pytest.approx(expected)
    if expected < 150:
        assert "regional_advisory" in speed["limiting_reasons"]


@pytest.mark.parametrize("mode", ["automatic", "user"])
def test_zero_target_applies_braking_until_stopped(mode):
    engine = Engine(speed_mode=mode, target_speed_kmh=40)
    engine._car.speed_mps = 5
    engine.set_speed(mode, 0)
    for _ in range(10):
        snapshot = engine.step()
        assert snapshot["command"]["brake"] == 1
        assert snapshot["command"]["throttle"] == 0
    assert snapshot["vehicle"]["speed_mps"] == 0
    assert snapshot["speed_control"]["target_speed_kmh"] == 0


def test_150_kmh_substeps_detect_thin_obstacle_between_outer_endpoints():
    engine = Engine(mode="manual", speed_mode="user", target_speed_kmh=150)
    engine.BODY_RADIUS_M = .03
    engine._car.speed_mps = 150 / 3.6
    engine._obstacles = lambda at_time=None: [
        {"id": "thin_high_speed", "x_m": 2., "y_m": 0., "radius_m": .03,
         "kind": "obstacle", "vx_mps": 0., "vy_mps": 0.}
    ]
    snapshot = engine.step({"steering": 0., "throttle": 0., "brake": 0.})
    assert snapshot["vehicle"]["x_m"] > 4
    assert abs(snapshot["vehicle"]["x_m"] - 2.) > .06
    assert snapshot["metrics"]["collisions"] == 1
    assert snapshot["metrics"]["min_clearance_m"] < 0
    assert snapshot["safety"]["actuation_enabled"] is False


def test_physics_never_exceeds_150_kmh_even_with_full_throttle_at_limit():
    engine = Engine(mode="manual", speed_mode="user", target_speed_kmh=150)
    engine._car.speed_mps = 150 / 3.6
    samples = []
    check_geometry = engine._check_geometry

    def record_speed(at_time):
        samples.append(engine._car.speed_mps)
        check_geometry(at_time)

    engine._check_geometry = record_speed
    for _ in range(2):
        snapshot = engine.step({"steering": 0., "throttle": 1., "brake": 0.})
    assert len(samples) > 100
    assert max(samples) <= 150 / 3.6
    assert snapshot["vehicle"]["speed_mps"] == pytest.approx(150 / 3.6)


def test_user_speed_cannot_override_learned_nonroad_veto():
    engine = Engine(speed_mode="user", target_speed_kmh=150,
                    perception=lambda rgb: Perception(np.zeros(rgb.shape[:2], dtype=np.float32)))
    for _ in range(3):
        snapshot = engine.step()
    assert snapshot["safety"]["mode"] == "LATCHED_STOP"
    assert "semantic_disagreement_or_nonroad" in snapshot["safety"]["reasons"]
    assert snapshot["vehicle"]["speed_mps"] == 0
    assert snapshot["command"]["brake"] == 1


@pytest.mark.parametrize("scenario,sensor_range", [("clear", 16), ("gwangju_day", 25)])
def test_learned_long_range_evidence_preserves_nonroad_and_unobserved_cells(scenario, sensor_range):
    engine = Engine(scenario, speed_mode="user", target_speed_kmh=150,
                    perception=lambda rgb: Perception(np.zeros(rgb.shape[:2], dtype=np.float32)))
    # Force a long stopping corridor without pretending the image model saw it.
    engine._car.speed_mps = 150 / 3.6
    capture, sequence, status = engine._observe()
    engine._plan()
    scene = engine._evidence(capture, sequence, status)
    grid = scene.grid
    forward = grid.x_min_m + (np.arange(grid.observed.shape[1]) + .5) * grid.resolution_m
    assert forward.max() > 250
    assert np.any(forward > 100)
    assert not grid.observed[:, forward > sensor_range].any()
    assert not grid.road_probability.any()
    snapshot = engine.snapshot()
    if scenario == "gwangju_day":
        assert snapshot["perception"]["long_range_world_semantics"] is False
        assert snapshot["perception"]["synthetic_sensor_range_m"] == sensor_range
    assert snapshot["perception"]["model_error"] is None


def _place_on_route(engine, progress, speed):
    point, direction = _polyline_position(engine._road, progress)
    engine._car.x_m, engine._car.y_m = map(float, point)
    engine._car.yaw_rad = math.atan2(direction[1], direction[0])
    engine._car.speed_mps = speed


def test_upcoming_bend_reduces_target_before_the_local_path_starts_turning():
    engine = Engine("gangwon_day", speed_mode="user", target_speed_kmh=150)
    _place_on_route(engine, 1125., 25.)
    plan = engine._plan()
    assert abs(plan.curvature) < 1e-8
    assert 0 < plan.target_speed_mps * 3.6 < 150
    assert "upcoming_curve" in engine.snapshot()["speed_control"]["limiting_reasons"]
    # The reduced proposal must actually pass the supervisor after warmup,
    # rather than converting every distant bend into a latched full stop.
    engine.step()
    snapshot = engine.step()
    assert snapshot["safety"]["proposal_eligible"]
    assert snapshot["safety"]["actuation_enabled"] is False


def test_physically_supported_curve_has_room_for_the_complete_stopping_body():
    engine = Engine("gangwon_day", speed_mode="automatic", target_speed_kmh=150)
    _place_on_route(engine, 1300., 19.4)
    plan = engine._plan()
    assert plan.target_speed_mps * 3.6 == pytest.approx(70)
    assert abs(plan.curvature) * (plan.target_speed_mps + .11) ** 2 < 2
    engine.step()
    snapshot = engine.step()
    assert snapshot["safety"]["proposal_eligible"], snapshot["safety"]
    assert snapshot["metrics"]["collisions"] == 0
    assert snapshot["metrics"]["offroad_events"] == 0


def test_remaining_goal_distance_reduces_proposal_and_brakes_before_goal():
    engine = Engine("gwangju_day", speed_mode="user", target_speed_kmh=150)
    _place_on_route(engine, engine._route_length - 75., 15.)
    distant_target = engine._plan().target_speed_mps
    # This pose must also leave stopping room inside the bounded lane runoff.
    # Teleporting here at 15 m/s would exceed its 20 m endpoint extension.
    _place_on_route(engine, engine._route_length - 15., 10.)
    near_target = engine._plan().target_speed_mps
    assert 0 < near_target < distant_target
    assert near_target < engine._car.speed_mps
    assert "goal_distance" in engine.snapshot()["speed_control"]["limiting_reasons"]
    engine.step()
    snapshot = engine.step()
    assert snapshot["safety"]["proposal_eligible"]
    assert snapshot["command"]["brake"] > 0
    assert snapshot["command"]["throttle"] == 0
    assert snapshot["vehicle"]["speed_mps"] < 10
    assert snapshot["world"]["route_progress_m"] < snapshot["world"]["route_length_m"]
