"""Executable MCU safety-protocol reference, NOT deployable GPIO firmware.

Port this state machine to the selected MCU. Its tick/watchdog must run on that
MCU independently from the host; running this Python class on the host does not
provide protection against a host crash or disconnected USB cable.
"""
from __future__ import annotations

import json
import math
import re

from .types import Command


class McuSafetyReference:
    def __init__(self, *, calibration_confirmed=False, max_steering_rad=0.45, max_throttle=0.3,
                 max_speed_mps=0.6, max_ttl_ms=250, max_line_bytes=2048, sensor_timeout_ms=250):
        self.calibration_confirmed = calibration_confirmed is True
        self.max_steering_rad = max_steering_rad
        self.max_throttle = max_throttle
        self.max_speed_mps = max_speed_mps
        self.max_ttl_ms = max_ttl_ms
        self.max_line_bytes = max_line_bytes
        self.sensor_timeout_ms = sensor_timeout_ms
        if type(sensor_timeout_ms) is not int or not 1 <= sensor_timeout_ms <= 1000:
            raise ValueError("sensor_timeout_ms must be an integer between 1 and 1000")
        if not (0 < max_steering_rad < math.pi / 2 and 0 < max_throttle <= 1 and max_speed_mps > 0
                and 20 <= max_ttl_ms <= 1000 and max_line_bytes > 0):
            raise ValueError("Invalid MCU reference limits")
        self.session = None
        self.ack_seq = 0
        self.host_ms = 0
        self.telemetry_seq = 0
        self.last_error = "startup"
        self._fault = "startup"
        self._physical_armed = False
        self._estop = True
        self._arm_low_seen = False
        self._speed = 0.0
        self._yaw = 0.0
        self._sensor_seq = 0
        self._sensor_ms = None
        self._last_rx = None
        self._last_now = None
        self._ttl = 0
        self._origin_host_ms = 0
        self._origin_mcu_ms = 0
        self.output = Command(reason="startup")

    def _stop(self, reason):
        self.output = Command(steering_rad=self.output.steering_rad, reason=reason)

    def _latch(self, reason):
        self._fault = self.last_error = reason
        self._arm_low_seen = False
        self._stop(reason)

    def _check_clock(self, now_ms):
        if type(now_ms) is not int or now_ms < 0:
            # Invalid timer input is a safety fault, not just an API exception:
            # a previous enabled output must not survive a failed clock check.
            self._latch("invalid_mcu_clock")
            return False
        if self._last_now is not None and now_ms < self._last_now:
            self._latch("mcu_clock_reversed")
            return False
        self._last_now = now_ms
        return True

    @property
    def armed(self):
        return self._physical_armed and not self._estop and not self._fault and self.calibration_confirmed

    def set_inputs(self, *, physical_armed, estop, speed_mps, yaw_rate_rps,
                   sensor_seq, sensor_ms, now_ms):
        """Supply measured inputs with the actual oldest sensor acquisition time.

        sensor_seq advances only after both speed and yaw have new validated
        samples. Re-polling switches or sending telemetry must not advance it.
        """
        if not self._check_clock(now_ms):
            return
        self.tick(now_ms)
        if type(physical_armed) is not bool or type(estop) is not bool:
            self._latch("invalid_safety_input")
            return
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in (speed_mps, yaw_rate_rps)):
            self._latch("invalid_measured_state")
            return
        if (type(sensor_seq) is not int or sensor_seq < 1 or type(sensor_ms) is not int
                or sensor_ms < 0 or sensor_ms > now_ms):
            self._latch("invalid_sensor_sample")
            return
        if now_ms - sensor_ms >= self.sensor_timeout_ms:
            self._latch("sensor_watchdog_expired")
            return
        if self._sensor_ms is not None:
            if sensor_seq < self._sensor_seq or sensor_ms < self._sensor_ms:
                self._latch("replayed_sensor_sample")
                return
            if sensor_seq == self._sensor_seq:
                if sensor_ms != self._sensor_ms or (speed_mps, yaw_rate_rps) != (self._speed, self._yaw):
                    self._latch("sensor_sample_changed_without_sequence")
                    return
            elif sensor_ms <= self._sensor_ms:
                self._latch("sensor_sequence_without_new_sample")
                return
        previous_arm = self._physical_armed
        self._physical_armed, self._estop = physical_armed, estop
        self._speed, self._yaw = float(speed_mps), float(yaw_rate_rps)
        self._sensor_seq, self._sensor_ms = sensor_seq, sensor_ms
        if estop:
            self._latch("estop")
            return
        if not physical_armed:
            self._arm_low_seen = True
            self._stop("disarmed")
        elif not previous_arm and self._arm_low_seen and self.session and self._last_rx is not None and now_ms - self._last_rx < self._ttl:
            self._fault = self.last_error = ""
            self._arm_low_seen = False
        if not self.calibration_confirmed:
            self._stop("calibration_required")

    def receive_line(self, line: bytes | str, now_ms: int) -> bool:
        if not self._check_clock(now_ms):
            return False
        self.tick(now_ms)
        try:
            if len(line) > self.max_line_bytes:
                raise ValueError("oversized_command")
            value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_command")))
            if not isinstance(value, dict) or type(value.get("v")) is not int or value["v"] != 2 or value.get("type") != "command":
                raise ValueError("protocol_mismatch")
            session = value.get("session")
            if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f]{32}", session):
                raise ValueError("invalid_session")
            for key in ("seq", "host_ms", "ttl_ms"):
                if type(value.get(key)) is not int:
                    raise ValueError("invalid_sequence_or_ttl")
            if value["seq"] < 1 or value["host_ms"] < 0 or not 20 <= value["ttl_ms"] <= self.max_ttl_ms:
                raise ValueError("invalid_sequence_or_ttl")
            if type(value.get("enabled")) is not bool:
                raise ValueError("invalid_enable")
            for key in ("steering_rad", "throttle", "brake", "target_speed_mps"):
                number = value.get(key)
                if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                    raise ValueError("nonfinite_command")
            if not (0 <= value["throttle"] <= 1 and 0 <= value["brake"] <= 1 and 0 <= value["target_speed_mps"] <= self.max_speed_mps):
                raise ValueError("command_out_of_range")
            new_session = session != self.session
            age_ms = 0
            if new_session and (value["enabled"] or value["throttle"] != 0 or value["brake"] != 1):
                raise ValueError("session_requires_stop")
            if not new_session:
                if value["seq"] <= self.ack_seq or value["host_ms"] < self.host_ms:
                    raise ValueError("replayed_command")
                # Offset established by the safe handshake detects queued old
                # commands. Long-term clock drift may cause a conservative STOP.
                age_ms = (now_ms - self._origin_mcu_ms) - (value["host_ms"] - self._origin_host_ms)
                if age_ms < -50 or age_ms >= value["ttl_ms"]:
                    raise ValueError("stale_or_future_command")
            else:
                self.session = session
                self.telemetry_seq = 0
                self._origin_host_ms, self._origin_mcu_ms = value["host_ms"], now_ms
                self._latch("new_session_requires_physical_rearm")
            self.ack_seq, self.host_ms = value["seq"], value["host_ms"]
            # A command delayed in the USB queue keeps only its remaining TTL;
            # reception must not give an old command a new full lifetime.
            self._last_rx = now_ms
            self._ttl = value["ttl_ms"] - max(0, age_ms)
            if not value["enabled"] or not self.armed:
                self._stop(self._fault or "host_stop")
            else:
                brake = value["brake"]
                self.output = Command(
                    steering_rad=max(-self.max_steering_rad, min(self.max_steering_rad, value["steering_rad"])),
                    throttle=0.0 if brake > 0 else min(self.max_throttle, value["throttle"]),
                    brake=brake, target_speed_mps=value["target_speed_mps"], enabled=True, reason="tracking",
                )
            return True
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError) as exc:
            self._latch(str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "malformed_command")
            return False

    def tick(self, now_ms: int) -> Command:
        """Call from the MCU timer regardless of host activity (e.g. every 5 ms)."""
        if not self._check_clock(now_ms):
            return self.output
        if self._last_rx is not None and now_ms - self._last_rx >= self._ttl:
            self._latch("command_watchdog_expired")
        if self._sensor_ms is not None and now_ms - self._sensor_ms >= self.sensor_timeout_ms:
            self._latch("sensor_watchdog_expired")
        elif self._sensor_ms is None and self._physical_armed:
            self._latch("missing_sensor_sample")
        if self._estop:
            self._stop("estop")
        elif not self.armed:
            self._stop(self._fault or "disarmed")
        return self.output

    def telemetry(self, now_ms: int) -> dict | None:
        if not self._check_clock(now_ms):
            return None
        self.tick(now_ms)
        if self.session is None:
            return None
        self.telemetry_seq += 1
        return {"v": 2, "type": "state", "session": self.session,
                "telemetry_seq": self.telemetry_seq, "ack_seq": self.ack_seq, "host_ms": self.host_ms,
                "mcu_ms": now_ms, "sensor_seq": self._sensor_seq, "sensor_ms": self._sensor_ms,
                "speed_mps": self._speed, "yaw_rate_rps": self._yaw,
                "armed": self.armed, "estop": self._estop}
