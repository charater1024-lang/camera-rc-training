import json

import pytest

from camera_rc.driving.mcu_reference import McuSafetyReference
from camera_rc.driving.transport import SerialTransport
from camera_rc.driving.types import Command


SESSION = "a" * 32


def command(seq=1, host_ms=10000, enabled=False, **changes):
    value = {"v": 2, "type": "command", "session": SESSION, "seq": seq,
             "host_ms": host_ms, "ttl_ms": 150, "enabled": enabled,
             "steering_rad": 0.1, "throttle": 0.2 if enabled else 0,
             "brake": 0 if enabled else 1, "target_speed_mps": 0.3 if enabled else 0}
    value.update(changes)
    return json.dumps(value)


def armed_mcu():
    mcu = McuSafetyReference(calibration_confirmed=True)
    assert mcu.receive_line(command(), 1000)
    mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0.0, yaw_rate_rps=0, sensor_seq=1001, sensor_ms=1001, now_ms=1001)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0.0, yaw_rate_rps=0, sensor_seq=1002, sensor_ms=1002, now_ms=1002)
    assert mcu.armed
    assert mcu.receive_line(command(2, 10010, True), 1010)
    assert mcu.output.enabled
    return mcu


def test_startup_requires_safe_handshake_and_physical_arm_transition():
    mcu = McuSafetyReference(calibration_confirmed=True)
    assert not mcu.receive_line(command(enabled=True), 1000)
    assert mcu.output.brake == 1
    assert mcu.receive_line(command(), 1001)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1002, sensor_ms=1002, now_ms=1002)
    assert not mcu.armed
    mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1003, sensor_ms=1003, now_ms=1003)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1004, sensor_ms=1004, now_ms=1004)
    assert mcu.armed


def test_watchdog_stops_without_host_and_latches_until_physical_rearm():
    mcu = armed_mcu()
    assert mcu.tick(1159).enabled
    stopped = mcu.tick(1160)
    assert not stopped.enabled and stopped.brake == 1 and stopped.throttle == 0
    assert not mcu.armed
    assert mcu.receive_line(command(3, 10170, True), 1170)
    assert not mcu.output.enabled
    mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1171, sensor_ms=1171, now_ms=1171)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1172, sensor_ms=1172, now_ms=1172)
    assert mcu.receive_line(command(4, 10180, True), 1180)
    assert mcu.output.enabled


def test_late_receive_cannot_skip_watchdog_even_without_intermediate_tick():
    mcu = armed_mcu()
    assert mcu.receive_line(command(3, 10200, True), 1200)
    assert not mcu.armed and not mcu.output.enabled


def test_queued_command_receives_only_remaining_lifetime():
    mcu = armed_mcu()
    assert mcu.receive_line(command(3, 10020, True), 1120)
    assert mcu.tick(1169).enabled
    assert not mcu.tick(1170).enabled


@pytest.mark.parametrize("changes,reason", [
    ({"seq": 2}, "replayed_command"),
    ({"host_ms": 9999}, "replayed_command"),
    ({"session": "b" * 32}, "session_requires_stop"),
    ({"host_ms": 10200}, "stale_or_future_command"),
    ({"throttle": 2}, "command_out_of_range"),
    ({"target_speed_mps": 2}, "command_out_of_range"),
    ({"ttl_ms": 5000}, "invalid_sequence_or_ttl"),
    ({"steering_rad": float("nan")}, "nonfinite_command"),
])
def test_bad_commands_stop_and_require_rearm(changes, reason):
    mcu = armed_mcu()
    values = {"seq": 3, "host_ms": 10020, "enabled": True}
    values.update(changes)
    assert not mcu.receive_line(command(**values), 1020)
    assert mcu.last_error == reason
    assert not mcu.output.enabled and not mcu.armed


def test_new_session_stop_invalidates_existing_arm():
    mcu = armed_mcu()
    assert mcu.receive_line(command(session="b" * 32), 1020)
    assert mcu.session == "b" * 32
    assert not mcu.armed and mcu.output.brake == 1


def test_estop_is_immediate_and_release_does_not_rearm():
    mcu = armed_mcu()
    mcu.set_inputs(physical_armed=True, estop=True, speed_mps=0.2, yaw_rate_rps=0, sensor_seq=1020, sensor_ms=1020, now_ms=1020)
    assert not mcu.output.enabled and mcu.output.brake == 1
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0.2, yaw_rate_rps=0, sensor_seq=1030, sensor_ms=1030, now_ms=1030)
    assert not mcu.armed


def test_measurement_telemetry_is_exact_and_missing_measurement_disarms():
    mcu = armed_mcu()
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0.27, yaw_rate_rps=-0.12, sensor_seq=1020, sensor_ms=1020, now_ms=1020)
    data = mcu.telemetry(1020)
    assert data["speed_mps"] == 0.27 and data["yaw_rate_rps"] == -0.12
    assert data["session"] == SESSION and data["ack_seq"] == 2 and data["host_ms"] == 10010
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=float("nan"), yaw_rate_rps=0, sensor_seq=1030, sensor_ms=1030, now_ms=1030)
    assert not mcu.armed and mcu.output.brake == 1


def test_calibration_required_independently_on_mcu():
    mcu = McuSafetyReference()
    assert mcu.receive_line(command(), 1000)
    mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1001, sensor_ms=1001, now_ms=1001)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1002, sensor_ms=1002, now_ms=1002)
    assert not mcu.armed
    assert mcu.receive_line(command(2, 10010, True), 1010)
    assert not mcu.output.enabled


def test_sensor_watchdog_expires_with_continuing_commands_and_requires_physical_rearm():
    mcu = armed_mcu()  # Last measured sample was captured at 1002 ms.
    seq = 2
    for now in (1060, 1110, 1160, 1210, 1251):
        seq += 1
        assert mcu.receive_line(command(seq, now + 9000, True), now)
        assert mcu.output.enabled
    seq += 1
    assert mcu.receive_line(command(seq, 10252, True), 1252)
    assert mcu.last_error == "sensor_watchdog_expired"
    assert not mcu.armed and not mcu.output.enabled
    assert mcu.output.brake == 1 and mcu.output.throttle == 0
    # Fresh measurement and commands alone cannot clear a latched sensor fault.
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0,
                   sensor_seq=1253, sensor_ms=1253, now_ms=1253)
    assert not mcu.armed
    seq += 1
    assert mcu.receive_line(command(seq, 10253, True), 1253)
    assert not mcu.output.enabled
    mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0, yaw_rate_rps=0,
                   sensor_seq=1254, sensor_ms=1254, now_ms=1254)
    mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0,
                   sensor_seq=1255, sensor_ms=1255, now_ms=1255)
    assert mcu.armed
    assert mcu.receive_line(command(seq + 1, 10256, True), 1256)
    assert mcu.output.enabled


def test_repolling_same_sensor_sample_does_not_refresh_its_lifetime():
    mcu = armed_mcu()
    for seq, now in enumerate((1060, 1110, 1160, 1210, 1252), start=3):
        mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0,
                       sensor_seq=1002, sensor_ms=1002, now_ms=now)
        assert mcu.receive_line(command(seq, now + 9000, True), now)
    assert not mcu.armed and mcu.last_error == "sensor_watchdog_expired"
    data = mcu.telemetry(1252)
    assert data["sensor_seq"] == 1002 and data["sensor_ms"] == 1002 and data["mcu_ms"] == 1252


@pytest.mark.parametrize("change,reason", [
    ({"sensor_seq": 1001}, "replayed_sensor_sample"),
    ({"sensor_seq": 1002, "sensor_ms": 1003}, "sensor_sample_changed_without_sequence"),
    ({"sensor_seq": 1002, "speed_mps": 0.1}, "sensor_sample_changed_without_sequence"),
    ({"sensor_seq": 1003}, "sensor_sequence_without_new_sample"),
    ({"sensor_seq": True}, "invalid_sensor_sample"),
    ({"sensor_ms": 2000}, "invalid_sensor_sample"),
    ({"sensor_ms": None}, "invalid_sensor_sample"),
])
def test_invalid_sensor_identity_cannot_rejuvenate_feedback(change, reason):
    mcu = armed_mcu()
    values = dict(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0,
                  sensor_seq=1002, sensor_ms=1002, now_ms=1020)
    values.update(change)
    mcu.set_inputs(**values)
    assert not mcu.armed and not mcu.output.enabled and mcu.last_error == reason


def test_v1_command_is_rejected_and_missing_sensor_metadata_is_explicit():
    mcu = armed_mcu()
    assert not mcu.receive_line(command(3, 10020, True, v=1), 1020)
    assert mcu.last_error == "protocol_mismatch" and not mcu.armed
    with pytest.raises(TypeError, match="sensor_seq"):
        mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, now_ms=1021)


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5, 1001])
def test_sensor_timeout_configuration_is_bounded(timeout):
    with pytest.raises(ValueError, match="sensor_timeout_ms"):
        McuSafetyReference(sensor_timeout_ms=timeout)


def test_host_and_mcu_protocol_interoperate_with_independent_clocks():
    class Loopback:
        def __init__(self, **kwargs):
            self.mcu = McuSafetyReference(calibration_confirmed=True)
            self.now_ms = 1000
            self.incoming = bytearray()

        @property
        def in_waiting(self):
            return len(self.incoming)

        def write(self, raw):
            self.mcu.receive_line(raw, self.now_ms)
            self.emit()
            return len(raw)

        def emit(self):
            self.incoming.extend((json.dumps(self.mcu.telemetry(self.now_ms)) + "\n").encode())

        def read(self, count):
            raw = bytes(self.incoming[:count])
            del self.incoming[:count]
            return raw

        def close(self):
            pass

    host = SerialTransport({"port": "LOOPBACK", "calibration_confirmed": True}, serial_factory=Loopback)
    host.send(Command(), 10)
    assert host.read_state(10.001) is None  # No measured sensor sample yet.
    wire = host._serial
    wire.mcu.set_inputs(physical_armed=False, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1002, sensor_ms=1002, now_ms=1002)
    wire.mcu.set_inputs(physical_armed=True, estop=False, speed_mps=0, yaw_rate_rps=0, sensor_seq=1003, sensor_ms=1003, now_ms=1003)
    wire.now_ms = 1004
    wire.emit()
    assert host.read_state(10.004).armed
    host.send(Command(0.1, 0.2, 0, 0.3, True, "tracking"), 10.01)
    assert wire.mcu.output.enabled
    wire.mcu.tick(1154)
    assert not wire.mcu.output.enabled
