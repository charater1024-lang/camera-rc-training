import math

import numpy as np
import pytest

from camera_rc.driving.control import Controller
from camera_rc.driving.types import Plan, VehicleState


def plan(speed=0.6, left=0.0):
    return Plan(np.array([[0.2, left / 4], [0.6, left], [1.2, left * 2]]), speed, True, "clear")


def state(now=10.0, speed=0.0, **kwargs):
    return VehicleState(now, speed, armed=kwargs.get("armed", True), estop=kwargs.get("estop", False))


def test_pure_pursuit_direction_and_steering_rate():
    left = Controller({"steering_rate_rad_s": 0.4})
    right = Controller({"steering_rate_rad_s": 0.4})
    lc = left.update(plan(left=0.5), state(), 10.0)
    rc = right.update(plan(left=-0.5), state(), 10.0)
    assert lc.steering_rad == pytest.approx(0.02)
    assert rc.steering_rad == pytest.approx(-0.02)
    next_command = left.update(plan(left=0.5), state(10.1), 10.1)
    assert next_command.steering_rad == pytest.approx(0.06)


def test_setpoint_acceleration_deceleration_and_speed_ceiling():
    controller = Controller({"max_speed_mps": 0.2, "max_accel_mps2": 0.5, "max_decel_mps2": 0.8})
    values = []
    for i in range(10):
        now = 10 + i * 0.05
        values.append(controller.update(plan(3), state(now), now).target_speed_mps)
    assert values[0] == pytest.approx(0.025)
    assert max(values) <= 0.2
    command = controller.update(plan(0.05), state(10.5), 10.5)
    assert command.target_speed_mps == pytest.approx(0.16)


def test_measured_speed_drives_braking_and_never_both_actuators():
    controller = Controller()
    command = controller.update(plan(), state(speed=0.8), 10.0)
    assert command.enabled and command.throttle == 0 and command.brake > 0
    command = controller.update(plan(), state(10.05, speed=0), 10.05)
    assert command.throttle > 0 and command.brake == 0


@pytest.mark.parametrize("case,reason", [
    ("absent", "no_vehicle_state"), ("old", "stale_vehicle_state"),
    ("future", "stale_vehicle_state"), ("estop", "estop"),
    ("disarmed", "disarmed"), ("nan", "invalid_vehicle_state"),
    ("reverse", "reverse_motion_unsupported"),
])
def test_unsafe_feedback_stops(case, reason):
    values = {
        "absent": None, "old": state(9), "future": state(11),
        "estop": state(estop=True), "disarmed": state(armed=False),
        "nan": state(speed=float("nan")), "reverse": state(speed=-0.1),
    }
    command = Controller().update(plan(), values[case], 10)
    assert not command.enabled and command.throttle == 0 and command.brake == 1
    assert command.reason == reason


@pytest.mark.parametrize("points", [[], [[1]], [[1, 2, 3], [2, 3, 4]], [[1, float("nan")], [2, 0]], [[-1, 0], [0, 0]]])
def test_invalid_path_never_moves(points):
    command = Controller().update(Plan(points, 0.2, True, "test"), state(), 10)
    assert not command.enabled and command.brake == 1


def test_zero_invalid_and_nonfinite_plan_stop():
    controller = Controller()
    for value in [plan(0), plan(-1), plan(float("nan")), Plan(np.ones((2, 2)), 1, False, "blocked")]:
        command = controller.update(value, state(), 10)
        assert not command.enabled and command.brake == 1


def test_control_deadline_stops_and_reset_starts_slowly():
    controller = Controller()
    controller.update(plan(), state(), 10)
    command = controller.update(plan(), state(11), 11)
    assert not command.enabled and command.reason == "control_deadline_missed"
    command = controller.update(plan(), state(11.05), 11.05)
    assert command.target_speed_mps == pytest.approx(0.015)


def test_integrator_does_not_wind_up_at_throttle_limit():
    controller = Controller({"max_accel_mps2": 100, "kp": 10, "ki": 1, "max_throttle": 0.1})
    for i in range(100):
        now = 10 + i * 0.05
        command = controller.update(plan(), state(now), now)
        assert command.throttle <= 0.1
    assert controller._integral == pytest.approx(0)
    command = controller.update(plan(), state(15, speed=0.7), 15)
    assert command.throttle == 0 and command.brake > 0


def test_stop_retains_bounded_wheel_angle_and_clears_speed():
    controller = Controller()
    moving = controller.update(plan(left=0.2), state(), 10)
    stop = controller.stop("operator")
    assert stop.steering_rad == moving.steering_rad
    assert stop.target_speed_mps == 0 and stop.throttle == 0 and stop.brake == 1
    controller.reset()
    assert controller.stop().steering_rad == 0


def test_abrupt_curvature_change_brakes_before_changing_steering_while_moving():
    controller = Controller()
    initial = controller.update(plan(), state(speed=0.2), 10)
    changed = controller.update(plan(left=0.5), state(10.05, speed=0.2), 10.05)
    assert changed.steering_rad == initial.steering_rad == 0
    assert changed.reason == "alignment_braking"
    assert changed.enabled and changed.brake == 1 and changed.throttle == 0
    assert changed.target_speed_mps == 0
    assert controller._integral == 0 and controller._last_time == 10.05


def test_stationary_alignment_slews_under_brake_and_then_allows_propulsion():
    controller = Controller({"steering_rate_rad_s": 0.4})
    commands = []
    for i in range(30):
        now = 10 + i * 0.05
        commands.append(controller.update(plan(left=0.5), state(now), now))
    aligning = [command for command in commands if command.reason == "alignment_steering"]
    assert len(aligning) > 1
    assert all(command.brake == 1 and command.throttle == 0 and command.target_speed_mps == 0 for command in aligning)
    assert all(command.enabled for command in aligning)
    assert commands[0].steering_rad == pytest.approx(0.02)
    assert max(abs(b.steering_rad - a.steering_rad) for a, b in zip(commands, commands[1:])) <= 0.0200001
    assert commands[-1].reason == "tracking" and commands[-1].throttle > 0
    assert not any(command.reason == "control_deadline_missed" for command in commands)


def test_straight_path_is_unchanged_and_alignment_can_be_explicitly_disabled():
    straight = Controller().update(plan(), state(), 10)
    assert straight.reason == "tracking" and straight.throttle > 0
    disabled = Controller({"stop_to_align": False}).update(plan(left=0.5), state(), 10)
    assert disabled.reason == "tracking" and disabled.throttle > 0


def test_alignment_boolean_must_be_explicit():
    with pytest.raises(ValueError, match="boolean"):
        Controller({"stop_to_align": "false"})


@pytest.mark.parametrize("config", [{"max_throttle": 1.1}, {"wheelbase_m": 0}, {"kp": -1}, {"ki": math.nan}, {"nominal_dt_s": 1}])
def test_invalid_control_config_is_rejected(config):
    with pytest.raises(ValueError):
        Controller(config)
