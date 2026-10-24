"""Cross-component contract checks; never open a camera, GPU or serial port."""
import copy
import json

import numpy as np
import pytest

from camera_rc.driving.config import DEFAULTS
from camera_rc.driving.runtime import ControlLoop
from camera_rc.driving.transport import DryRunTransport
from camera_rc.driving.types import Plan, VehicleState


class RecordedFeedback(DryRunTransport):
    def read_state(self, now):
        return VehicleState(now, 0.0, 0.0, True, False)


def diagnostic_config():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["safety"] = {"mode": "diagnostic_local_only"}
    return cfg


def path():
    return Plan(np.array([[0, 0], [0.5, 0], [1, 0]], dtype=float), 0.4, True, "clear_path")


def test_expired_perception_plan_latches_stop_until_restart(tmp_path):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    stopped = loop.step(1.25)["command"]
    assert stopped["enabled"] is False and stopped["brake"] == 1 and stopped["throttle"] == 0
    assert stopped["reason"] == "plan_expired_restart_required"
    loop.publish(path(), 1.26)
    assert loop.step(1.27)["command"]["enabled"] is False


def test_perception_worker_fault_blocks_new_plan(tmp_path):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    loop.fail("perception_deadline_missed_restart_required")
    loop.publish(path(), 1.02)
    stopped = loop.step(1.03)["command"]
    assert stopped["enabled"] is False and stopped["target_speed_mps"] == 0
    assert stopped["reason"] == "perception_deadline_missed_restart_required"


def test_live_camera_dry_run_does_not_invent_speed_or_arming(tmp_path):
    loop = ControlLoop(diagnostic_config(), DryRunTransport(), tmp_path)
    loop.publish(path(), 1.0)
    item = loop.step(1.01)
    assert item["state"] is None and item["physical_output"] is False
    assert item["command"]["enabled"] is False
    assert item["command"]["reason"] == "no_vehicle_state"


@pytest.mark.parametrize("stamp", [None, float("nan"), float("inf"), -float("inf"),
                                  True, False, "1.02", -1.0, 10 ** 400])
def test_invalid_capture_time_stops_existing_motion_and_cannot_be_replaced(tmp_path, stamp):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    loop.publish(path(), stamp)
    stopped = loop.step(1.03)
    assert stopped["command"]["reason"] == "plan_capture_time_invalid_restart_required"
    assert stopped["command"]["enabled"] is False
    assert stopped["command"]["throttle"] == 0 and stopped["command"]["brake"] == 1
    assert stopped["plan_age_ms"] is None
    json.dumps(stopped, allow_nan=False)
    loop.publish(path(), 1.04)
    assert loop.step(1.05)["command"]["enabled"] is False


@pytest.mark.parametrize("stamp", [None, float("nan"), "missing"])
def test_command_boundary_rechecks_capture_time_even_when_publication_is_bypassed(tmp_path, stamp):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.plan, loop.plan_time = path(), stamp
    item = loop.step(1.0)
    assert item["command"]["reason"] == "plan_capture_time_invalid_restart_required"
    assert item["command"]["enabled"] is False and item["command"]["brake"] == 1
    json.dumps(item, allow_nan=False)


def test_future_capture_time_latches_before_any_enabled_command(tmp_path):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.1)
    item = loop.step(1.0)
    assert item["command"]["reason"] == "plan_from_future_restart_required"
    assert item["command"]["enabled"] is False and item["command"]["brake"] == 1
    loop.publish(path(), 1.11)
    assert loop.step(1.12)["command"]["enabled"] is False


def test_republishing_same_capture_does_not_refresh_plan_age(tmp_path):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    loop.publish(path(), 1.0)
    item = loop.step(1.25)
    assert item["command"]["reason"] == "plan_expired_restart_required"
    assert item["command"]["enabled"] is False and item["command"]["throttle"] == 0


def test_startup_without_a_plan_stays_stopped_without_latching_missing_timestamp(tmp_path):
    loop = ControlLoop(diagnostic_config(), RecordedFeedback(), tmp_path)
    item = loop.step(1.0)
    assert item["command"]["reason"] == "waiting_for_first_plan"
    assert loop.latched_fault is None
    loop.publish(path(), 1.01)
    assert loop.step(1.02)["command"]["enabled"]
