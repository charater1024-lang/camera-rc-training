"""Fail-closed newline-JSON interface to an independently protected motor MCU."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import secrets
from typing import Any, Callable

from .types import Command, VehicleState


class TransportError(RuntimeError):
    pass


class SerialTransport:
    def __init__(self, config: dict[str, Any], serial_factory: Callable[..., Any] | None = None):
        self.config = {
            "baudrate": 230400, "write_timeout_s": 0.05,
            "state_timeout_s": 0.25, "command_ttl_ms": 150,
            "sensor_timeout_s": 0.25,
            "max_line_bytes": 2048, "max_buffer_bytes": 8192,
            "max_steering_rad": 0.45, "max_throttle": 0.3,
            "max_measured_speed_mps": 10.0, "max_measured_yaw_rate_rps": 20.0,
            "calibration_confirmed": False,
        }
        self.config.update(config)
        c = self.config
        if not isinstance(c.get("port"), str) or not c["port"].strip():
            raise ValueError("serial.port must identify an explicit device")
        for key in ("baudrate", "command_ttl_ms", "max_line_bytes", "max_buffer_bytes"):
            if type(c[key]) is not int or c[key] <= 0:
                raise ValueError(f"serial.{key} must be a positive integer")
        for key in ("write_timeout_s", "state_timeout_s", "sensor_timeout_s", "max_steering_rad", "max_throttle",
                    "max_measured_speed_mps", "max_measured_yaw_rate_rps"):
            if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f"serial.{key} must be finite and positive")
        if not 20 <= c["command_ttl_ms"] <= 1000:
            raise ValueError("command_ttl_ms must be between 20 and 1000")
        if c["max_buffer_bytes"] < c["max_line_bytes"] or c["max_buffer_bytes"] > 1024 * 1024:
            raise ValueError("invalid bounded serial buffer size")
        if c["max_throttle"] > 1 or c["max_steering_rad"] >= math.pi / 2:
            raise ValueError("invalid serial output limits")
        if type(c["calibration_confirmed"]) is not bool:
            raise ValueError("calibration_confirmed must be an explicit boolean")
        if serial_factory is None:
            try:
                import serial
            except ImportError as exc:
                raise TransportError("Install the runtime extra (pyserial) to use a serial device") from exc
            serial_factory = serial.Serial
        self.session = secrets.token_hex(16)
        self.sequence = 0
        self.last_error = "awaiting_telemetry"
        self._buffer = bytearray()
        self._sent: OrderedDict[int, tuple[float, int]] = OrderedDict()
        self._state: VehicleState | None = None
        self._last_telemetry_seq = -1
        self._last_ack = -1
        self._last_sensor_seq = None
        self._last_sensor_ms = None
        self._last_mcu_ms = None
        self._sensor_values = None
        self._sensor_host_time = None
        self._sensor_fault = None
        self._last_send_time: float | None = None
        self._closed = False
        self._serial = serial_factory(
            port=c["port"], baudrate=c["baudrate"], timeout=0,
            write_timeout=c["write_timeout_s"],
        )
        # Construction never sends an enabled command. The first command must
        # establish a session with STOP before matching telemetry can arrive.

    def _reject(self, reason: str) -> None:
        self.last_error = reason
        self._state = None

    def _reject_sensor(self, reason: str) -> None:
        # Before the first sample, startup may legitimately report no reading.
        # A sensor failure after valid feedback requires a new host session.
        if self._last_sensor_seq is not None:
            self._sensor_fault = reason
        self._reject(reason)

    def _fresh_state(self, now: float) -> VehicleState | None:
        if self._sensor_fault:
            self._reject(self._sensor_fault)
            return None
        state = self._state
        if state is None:
            return None
        if self._sensor_host_time is None or not 0 <= now - self._sensor_host_time < self.config["sensor_timeout_s"]:
            self._reject_sensor("stale_sensor_sample")
            return None
        age = now - state.timestamp
        sent = self._sent.get(state.ack_seq)
        if age < 0 or age > self.config["state_timeout_s"] or sent is None or not 0 <= now - sent[0] <= self.config["state_timeout_s"]:
            self._reject("stale_telemetry")
            return None
        return state

    def read_state(self, now: float) -> VehicleState | None:
        if self._closed:
            self._reject("serial_closed")
            return None
        if not math.isfinite(now):
            self._reject("invalid_clock")
            return None
        try:
            available = int(self._serial.in_waiting)
            if available > self.config["max_buffer_bytes"]:
                self._serial.reset_input_buffer()
                self._buffer.clear()
                self._reject("serial_input_overflow")
                return None
            if available:
                self._buffer.extend(self._serial.read(available))
        except Exception as exc:
            self._reject("serial_read_error")
            raise TransportError(f"Serial read failed: {exc}") from exc
        if len(self._buffer) > self.config["max_buffer_bytes"]:
            self._buffer.clear()
            self._reject("serial_input_overflow")
            return None
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            if len(line) > self.config["max_line_bytes"]:
                self._reject("serial_line_too_long")
                continue
            self._accept_line(bytes(line), now)
        if len(self._buffer) > self.config["max_line_bytes"]:
            self._buffer.clear()
            self._reject("serial_line_too_long")
            return None
        return self._fresh_state(now)

    def _accept_line(self, line: bytes, now: float) -> None:
        try:
            value = json.loads(line.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(value, dict):
                raise ValueError()
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
            self._reject("malformed_telemetry")
            return
        if type(value.get("v")) is not int or value["v"] != 2 or value.get("type") != "state":
            self._reject("protocol_mismatch")
            return
        if value.get("session") != self.session:
            self._reject("session_mismatch")
            return
        telemetry_seq, ack = value.get("telemetry_seq"), value.get("ack_seq")
        if type(telemetry_seq) is not int or type(ack) is not int or telemetry_seq < 0 or ack < 1:
            self._reject("invalid_sequence")
            return
        if telemetry_seq <= self._last_telemetry_seq or ack < self._last_ack:
            self._reject("replayed_telemetry")
            return
        if ack > self.sequence:
            self._reject("future_ack")
            return
        sent = self._sent.get(ack)
        if sent is None:
            self._reject("unknown_ack")
            return
        if type(value.get("host_ms")) is not int or value["host_ms"] != sent[1]:
            self._reject("ack_time_mismatch")
            return
        if not 0 <= now - sent[0] <= self.config["state_timeout_s"]:
            self._reject("stale_ack")
            return
        speed, yaw = value.get("speed_mps"), value.get("yaw_rate_rps")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (speed, yaw)):
            self._reject("invalid_measured_state")
            return
        if abs(speed) > self.config["max_measured_speed_mps"] or abs(yaw) > self.config["max_measured_yaw_rate_rps"]:
            self._reject("implausible_measured_state")
            return
        if type(value.get("armed")) is not bool or type(value.get("estop")) is not bool:
            self._reject("missing_physical_safety_state")
            return
        if self._sensor_fault:
            self._reject(self._sensor_fault)
            return
        sensor_seq, sensor_ms, mcu_ms = (value.get(key) for key in ("sensor_seq", "sensor_ms", "mcu_ms"))
        if (type(sensor_seq) is not int or sensor_seq < 1 or type(sensor_ms) is not int
                or type(mcu_ms) is not int or sensor_ms < 0 or mcu_ms < sensor_ms):
            self._reject_sensor("invalid_or_missing_sensor_sample")
            return
        if self._last_sensor_seq is not None:
            if (sensor_seq < self._last_sensor_seq or sensor_ms < self._last_sensor_ms
                    or mcu_ms < self._last_mcu_ms):
                self._reject_sensor("replayed_sensor_sample")
                return
            if sensor_seq == self._last_sensor_seq:
                if sensor_ms != self._last_sensor_ms or (speed, yaw) != self._sensor_values:
                    self._reject_sensor("sensor_sample_changed_without_sequence")
                    return
            elif sensor_ms <= self._last_sensor_ms:
                self._reject_sensor("sensor_sequence_without_new_sample")
                return
        # Telemetry follows its ACK, so the full host ACK round-trip bounds the
        # return-leg delay conservatively without synchronizing the two clocks.
        sensor_host_time = sent[0] - (mcu_ms - sensor_ms) / 1000.0
        if sensor_seq == self._last_sensor_seq:
            # Fresh packets/ACKs must never rejuvenate a repeated sensor sample.
            sensor_host_time = min(sensor_host_time, self._sensor_host_time)
        if not 0 <= now - sensor_host_time < self.config["sensor_timeout_s"]:
            self._reject_sensor("stale_sensor_sample")
            return
        self._last_sensor_seq, self._last_sensor_ms, self._last_mcu_ms = sensor_seq, sensor_ms, mcu_ms
        self._sensor_values, self._sensor_host_time = (speed, yaw), sensor_host_time
        self._last_telemetry_seq, self._last_ack = telemetry_seq, ack
        self._state = VehicleState(sensor_host_time, float(speed), float(yaw), value["armed"], value["estop"], ack)
        self.last_error = ""

    def send(self, command: Command, now: float) -> Command:
        if self._closed:
            raise TransportError("Serial transport is closed")
        if not math.isfinite(now) or now < 0:
            raise TransportError("Invalid host monotonic clock")
        if self._last_send_time is not None and now < self._last_send_time:
            self._reject("host_clock_reversed")
            raise TransportError("Host monotonic clock moved backwards")
        c = self.config
        values = (command.steering_rad, command.throttle, command.brake, command.target_speed_mps)
        state = self._fresh_state(now)
        reason = command.reason
        enabled = bool(command.enabled)
        if not all(math.isfinite(v) for v in values):
            enabled, reason = False, "nonfinite_command"
        elif not c["calibration_confirmed"]:
            enabled, reason = False, "calibration_required"
        elif state is None:
            enabled, reason = False, self.last_error or "no_vehicle_state"
        elif state.estop or not state.armed:
            enabled, reason = False, "estop" if state.estop else "disarmed"
        elif command.target_speed_mps < 0:
            enabled, reason = False, "reverse_command_unsupported"
        steering = max(-c["max_steering_rad"], min(c["max_steering_rad"], command.steering_rad)) if math.isfinite(command.steering_rad) else 0.0
        effective = Command(
            steering_rad=steering,
            throttle=max(0.0, min(c["max_throttle"], command.throttle)) if enabled else 0.0,
            brake=max(0.0, min(1.0, command.brake)) if enabled else 1.0,
            target_speed_mps=command.target_speed_mps if enabled else 0.0,
            enabled=enabled, reason=str(reason)[:160],
        )
        if effective.brake > 0:
            effective.throttle = 0.0
        self.sequence += 1
        host_ms = int(now * 1000)
        payload = {
            "v": 2, "type": "command", "session": self.session, "seq": self.sequence,
            "host_ms": host_ms, "ttl_ms": c["command_ttl_ms"],
            "enabled": effective.enabled, "steering_rad": effective.steering_rad,
            "throttle": effective.throttle, "brake": effective.brake,
            "target_speed_mps": effective.target_speed_mps, "reason": effective.reason,
        }
        raw = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        try:
            written = self._serial.write(raw)
            if written != len(raw):
                raise TransportError("Incomplete serial command write")
        except Exception as exc:
            self._reject("serial_write_error")
            raise TransportError(f"Serial command failed; MCU watchdog must stop: {exc}") from exc
        self._sent[self.sequence] = (now, host_ms)
        self._last_send_time = now
        while len(self._sent) > 256:
            self._sent.popitem(last=False)
        return effective

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._state = None
            self._serial.close()


class DryRunTransport:
    """Records commands only; it never invents speed feedback or arming state."""
    def __init__(self, config: dict[str, Any] | None = None):
        self.last_command: Command | None = None
        self.last_error = "dry_run_no_vehicle_state"
        self.closed = False

    def read_state(self, now: float) -> VehicleState | None:
        return None

    def send(self, command: Command, now: float) -> Command:
        if self.closed:
            raise TransportError("Dry-run transport is closed")
        self.last_command = command
        return command

    def close(self) -> None:
        self.closed = True
