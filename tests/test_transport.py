import json

import pytest

from camera_rc.driving.transport import DryRunTransport, SerialTransport, TransportError
from camera_rc.driving.types import Command


class FakeSerial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.incoming = bytearray()
        self.writes = []
        self.closed = False
        self.partial_write = False

    @property
    def in_waiting(self):
        return len(self.incoming)

    def read(self, size):
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result

    def write(self, value):
        self.writes.append(value)
        return len(value) - 1 if self.partial_write else len(value)

    def reset_input_buffer(self):
        self.incoming.clear()

    def close(self):
        self.closed = True


def transport(**kwargs):
    return SerialTransport({"port": "FAKE", "calibration_confirmed": True, **kwargs}, serial_factory=FakeSerial)


def telemetry(value, now=10.01, **changes):
    sent = json.loads(value._serial.writes[-1])
    sample = max(round(now * 1000), (value._last_sensor_ms or 0) + 1)
    data = {"v": 2, "type": "state", "session": value.session,
            "telemetry_seq": value._last_telemetry_seq + 1, "ack_seq": sent["seq"],
            "host_ms": sent["host_ms"], "speed_mps": 0.1, "yaw_rate_rps": 0.0,
            "armed": True, "estop": False, "sensor_seq": (value._last_sensor_seq or 0) + 1,
            "sensor_ms": sample, "mcu_ms": sample}
    data.update(changes)
    value._serial.incoming.extend((json.dumps(data) + "\n").encode())
    return value.read_state(now)


def moving():
    return Command(0.1, 0.2, 0, 0.3, True, "tracking")


def test_opening_device_never_arms_and_requires_stop_handshake():
    value = transport()
    assert value._serial.writes == []
    assert value._serial.kwargs["timeout"] == 0
    assert value._serial.kwargs["write_timeout"] > 0
    assert value.read_state(10) is None
    command = value.send(moving(), 10)
    assert not command.enabled and command.throttle == 0 and command.brake == 1
    assert telemetry(value) is not None
    command = value.send(moving(), 10.02)
    assert command.enabled and command.throttle > 0
    wire = json.loads(value._serial.writes[-1])
    assert wire["seq"] == 2 and wire["ttl_ms"] == 150
    assert value._serial.writes[-1].endswith(b"\n")


def test_telemetry_is_measured_and_requires_recent_ack_even_if_received_now():
    value = transport()
    value.send(Command(), 10)
    received = telemetry(value, speed_mps=0.42)
    assert received.speed_mps == 0.42 and received.timestamp == 10.0  # Conservative ACK-delay bound.
    assert telemetry(value, now=10.3) is None
    assert value.last_error == "stale_ack"
    assert not value.send(moving(), 10.31).enabled


@pytest.mark.parametrize("changes,reason", [
    ({"session": "wrong"}, "session_mismatch"),
    ({"v": 1}, "protocol_mismatch"),
    ({"ack_seq": 99}, "future_ack"),
    ({"ack_seq": True}, "invalid_sequence"),
    ({"host_ms": 5}, "ack_time_mismatch"),
    ({"telemetry_seq": -1}, "invalid_sequence"),
    ({"speed_mps": float("nan")}, "malformed_telemetry"),
    ({"yaw_rate_rps": None}, "invalid_measured_state"),
    ({"speed_mps": 30}, "implausible_measured_state"),
    ({"armed": 1}, "missing_physical_safety_state"),
])
def test_rejects_untrusted_telemetry(changes, reason):
    value = transport()
    value.send(Command(), 10)
    assert telemetry(value, **changes) is None
    assert value.last_error == reason


def test_replayed_telemetry_invalidates_prior_state():
    value = transport()
    value.send(Command(), 10)
    assert telemetry(value, telemetry_seq=1) is not None
    assert telemetry(value, telemetry_seq=1) is None
    assert value.last_error == "replayed_telemetry"
    assert not value.send(moving(), 10.02).enabled


def test_partial_lines_and_invalid_json_fail_closed():
    value = transport()
    value.send(Command(), 10)
    assert telemetry(value) is not None
    value._serial.incoming.extend(b"{bad")
    assert value.read_state(10.02) is not None
    value._serial.incoming.extend(b"}\n")
    assert value.read_state(10.03) is None
    assert value.last_error == "malformed_telemetry"


def test_line_and_buffer_limits_are_bounded():
    value = transport(max_line_bytes=100, max_buffer_bytes=200)
    value._serial.incoming.extend(b"x" * 101)
    assert value.read_state(10) is None
    assert value.last_error == "serial_line_too_long"
    value._serial.incoming.extend(b"x" * 201)
    assert value.read_state(10) is None
    assert value.last_error == "serial_input_overflow"
    assert not value._serial.incoming and not value._buffer


@pytest.mark.parametrize("change", [{"armed": False}, {"estop": True}])
def test_physical_disarm_or_estop_blocks_enabled_output(change):
    value = transport()
    value.send(Command(), 10)
    telemetry(value, **change)
    command = value.send(moving(), 10.02)
    assert not command.enabled and command.brake == 1 and command.throttle == 0


def test_hardware_calibration_must_be_explicit_and_limits_apply():
    value = transport(calibration_confirmed=False)
    value.send(Command(), 10)
    telemetry(value)
    assert value.send(moving(), 10.02).reason == "calibration_required"
    value = transport()
    value.send(Command(), 10)
    telemetry(value)
    command = value.send(Command(100, 100, 0, 0.3, True, "test"), 10.02)
    assert command.steering_rad == 0.45 and command.throttle == 0.3
    command = value.send(Command(0, 0.3, 0.2, 0.3, True, "test"), 10.03)
    assert command.throttle == 0 and command.brake == 0.2


def test_nonfinite_output_is_always_stop():
    value = transport()
    value.send(Command(), 10)
    telemetry(value)
    command = value.send(Command(float("nan"), 1, 0, 0.3, True, "bad"), 10.02)
    assert not command.enabled and command.steering_rad == 0 and command.brake == 1
    assert json.loads(value._serial.writes[-1])["reason"] == "nonfinite_command"


def test_disconnect_timeout_and_partial_write_require_mcu_watchdog():
    value = transport()
    value.send(Command(), 10)
    telemetry(value)
    assert value.read_state(10.3) is None
    assert not value.send(moving(), 10.31).enabled
    value._serial.partial_write = True
    with pytest.raises(TransportError, match="watchdog"):
        value.send(Command(), 10.32)
    value.close()
    assert value._serial.closed
    value.close()
    with pytest.raises(TransportError, match="closed"):
        value.send(Command(), 10.33)


def test_dry_run_has_no_fabricated_feedback():
    value = DryRunTransport()
    assert value.read_state(10) is None
    command = value.send(Command(reason="dry_run"), 10)
    assert value.last_command is command
    value.close()


@pytest.mark.parametrize("change,reason", [
    ({"sensor_seq": None}, "invalid_or_missing_sensor_sample"),
    ({"sensor_ms": None}, "invalid_or_missing_sensor_sample"),
    ({"mcu_ms": None}, "invalid_or_missing_sensor_sample"),
    ({"sensor_seq": True}, "invalid_or_missing_sensor_sample"),
    ({"sensor_ms": 20000, "mcu_ms": 10010}, "invalid_or_missing_sensor_sample"),
    ({"sensor_ms": 9000, "mcu_ms": 10010}, "stale_sensor_sample"),
])
def test_sensor_sample_metadata_is_required_and_age_is_independent_of_packet_age(change, reason):
    value = transport()
    value.send(Command(), 10)
    assert telemetry(value, **change) is None
    assert value.last_error == reason
    assert not value.send(moving(), 10.02).enabled


def test_fresh_packets_and_acks_do_not_refresh_a_frozen_sensor_sample():
    value = transport(sensor_timeout_s=0.1)
    value.send(Command(), 10)
    assert telemetry(value, sensor_seq=1, sensor_ms=1000, mcu_ms=1000) is not None
    for now in (10.03, 10.06, 10.09):
        assert value.send(moving(), now).enabled
        # Even a broken MCU that repeats its old clock cannot rejuvenate the sample.
        assert telemetry(value, now=now + 0.001, sensor_seq=1, sensor_ms=1000, mcu_ms=1000) is not None
    assert not value.send(moving(), 10.101).enabled
    assert value.last_error == "stale_sensor_sample"
    # A new reading cannot auto-resume a previously valid connection after this fault.
    assert telemetry(value, now=10.102, sensor_seq=2, sensor_ms=1102, mcu_ms=1102) is None
    assert not value.send(moving(), 10.103).enabled


def test_sensor_age_includes_ack_delay_and_holds_older_sample_timestamp():
    value = transport(sensor_timeout_s=0.1)
    value.send(Command(), 10)
    assert telemetry(value, now=10.02, sensor_ms=900, mcu_ms=990) is None  # 90 + 20 ms.
    assert value.last_error == "stale_sensor_sample"


@pytest.mark.parametrize("change,reason", [
    ({"sensor_seq": 0}, "invalid_or_missing_sensor_sample"),
    ({"sensor_seq": 1, "sensor_ms": 1001}, "sensor_sample_changed_without_sequence"),
    ({"sensor_seq": 1, "speed_mps": 0.2}, "sensor_sample_changed_without_sequence"),
    ({"sensor_seq": 2}, "sensor_sequence_without_new_sample"),
    ({"sensor_seq": 2, "sensor_ms": 999}, "replayed_sensor_sample"),
    ({"mcu_ms": 999}, "invalid_or_missing_sensor_sample"),
])
def test_sensor_sequence_and_timestamp_are_consistent_and_faults_latch(change, reason):
    value = transport()
    value.send(Command(), 10)
    assert telemetry(value, sensor_seq=1, sensor_ms=1000, mcu_ms=1000) is not None
    fields = dict(sensor_seq=1, sensor_ms=1000, mcu_ms=1010)
    fields.update(change)
    assert telemetry(value, now=10.02, **fields) is None
    assert value.last_error == reason
    assert telemetry(value, now=10.03, sensor_seq=3, sensor_ms=1030, mcu_ms=1030) is None
    assert not value.send(moving(), 10.04).enabled


def test_v2_packet_with_sensor_fields_omitted_is_rejected():
    value = transport()
    value.send(Command(), 10)
    sent = json.loads(value._serial.writes[-1])
    legacy = {"v": 2, "type": "state", "session": value.session, "telemetry_seq": 1,
              "ack_seq": sent["seq"], "host_ms": sent["host_ms"], "speed_mps": 0,
              "yaw_rate_rps": 0, "armed": True, "estop": False}
    value._serial.incoming.extend((json.dumps(legacy) + "\n").encode())
    assert value.read_state(10.01) is None
    assert not value.send(moving(), 10.02).enabled


@pytest.mark.parametrize("config", [{"port": ""}, {"max_throttle": 2}, {"command_ttl_ms": 0}, {"calibration_confirmed": "yes"}])
def test_invalid_transport_config_is_rejected(config):
    with pytest.raises(ValueError):
        transport(**config)
