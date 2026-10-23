"""Cross-component contract checks; never open a camera, GPU or serial port."""
import copy

import numpy as np

from camera_rc.driving.config import DEFAULTS
from camera_rc.driving.runtime import ControlLoop
from camera_rc.driving.transport import DryRunTransport
from camera_rc.driving.types import Plan, VehicleState


class RecordedFeedback(DryRunTransport):
    def read_state(self, now):
        return VehicleState(now, 0.0, 0.0, True, False)


def path():
    return Plan(np.array([[0, 0], [0.5, 0], [1, 0]], dtype=float), 0.4, True, "clear_path")


def test_expired_perception_plan_latches_stop_until_restart(tmp_path):
    loop = ControlLoop(copy.deepcopy(DEFAULTS), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    stopped = loop.step(1.25)["command"]
    assert stopped["enabled"] is False and stopped["brake"] == 1 and stopped["throttle"] == 0
    assert stopped["reason"] == "plan_expired_restart_required"
    loop.publish(path(), 1.26)
    assert loop.step(1.27)["command"]["enabled"] is False


def test_perception_worker_fault_blocks_new_plan(tmp_path):
    loop = ControlLoop(copy.deepcopy(DEFAULTS), RecordedFeedback(), tmp_path)
    loop.publish(path(), 1.0)
    assert loop.step(1.01)["command"]["enabled"]
    loop.fail("perception_deadline_missed_restart_required")
    loop.publish(path(), 1.02)
    stopped = loop.step(1.03)["command"]
    assert stopped["enabled"] is False and stopped["target_speed_mps"] == 0
    assert stopped["reason"] == "perception_deadline_missed_restart_required"


def test_live_camera_dry_run_does_not_invent_speed_or_arming(tmp_path):
    loop = ControlLoop(copy.deepcopy(DEFAULTS), DryRunTransport(), tmp_path)
    loop.publish(path(), 1.0)
    item = loop.step(1.01)
    assert item["state"] is None and item["physical_output"] is False
    assert item["command"]["enabled"] is False
    assert item["command"]["reason"] == "no_vehicle_state"
