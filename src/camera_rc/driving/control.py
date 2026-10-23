"""Measured-speed feedback and pure pursuit for a low-speed Ackermann vehicle.

All angles are radians. Positive steering points left. This module never arms a
vehicle and has no hardware side effects.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .types import Command, Plan, VehicleState


class Controller:
    def __init__(self, config: dict[str, Any] | None = None):
        values = {
            "wheelbase_m": 0.26,
            "max_steering_rad": 0.45,
            "steering_rate_rad_s": 0.8,
            "lookahead_m": 0.6,
            "lookahead_speed_gain_s": 0.5,
            "max_speed_mps": 0.6,
            "max_accel_mps2": 0.3,
            "max_decel_mps2": 0.8,
            "kp": 0.8,
            "ki": 0.2,
            "integral_limit": 1.0,
            "max_throttle": 0.3,
            "state_timeout_s": 0.25,
            "nominal_dt_s": 0.05,
            "stop_to_align": True,
            "alignment_tolerance_rad": 0.04,
            "alignment_speed_mps": 0.03,
        }
        values.update(config or {})
        for key in values:
            value = values[key]
            if key == "stop_to_align":
                if type(value) is not bool:
                    raise ValueError("control.stop_to_align must be a boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"control.{key} must be a finite number")
            if value < 0 or (key not in {"kp", "ki", "lookahead_speed_gain_s"} and value == 0):
                raise ValueError(f"control.{key} is outside its valid range")
        if values["max_throttle"] > 1 or values["max_steering_rad"] >= math.pi / 2:
            raise ValueError("max_throttle must be <= 1 and max_steering_rad < pi/2")
        if values["nominal_dt_s"] > values["state_timeout_s"]:
            raise ValueError("nominal_dt_s must not exceed state_timeout_s")
        self.config = values
        self.reset()

    def reset(self) -> None:
        self._last_time: float | None = None
        self._steering = 0.0
        self._target_speed = 0.0
        self._integral = 0.0

    def stop(self, reason: str = "stop") -> Command:
        # Retain the current wheel angle while stopping to avoid a steering jump.
        steering = self._steering
        self.reset()
        self._steering = steering
        return Command(steering_rad=steering, reason=reason)

    def update(self, plan: Plan, state: VehicleState | None, now: float) -> Command:
        c = self.config
        if not math.isfinite(now):
            return self.stop("invalid_clock")
        if state is None:
            return self.stop("no_vehicle_state")
        if not all(math.isfinite(v) for v in (state.timestamp, state.speed_mps, state.yaw_rate_rps)):
            return self.stop("invalid_vehicle_state")
        age = now - state.timestamp
        if age < 0 or age > c["state_timeout_s"]:
            return self.stop("stale_vehicle_state")
        if state.estop:
            return self.stop("estop")
        if not state.armed:
            return self.stop("disarmed")
        if state.speed_mps < -0.01:
            return self.stop("reverse_motion_unsupported")
        if not plan.valid:
            return self.stop(plan.reason or "invalid_plan")
        if not math.isfinite(plan.target_speed_mps) or not math.isfinite(plan.curvature):
            return self.stop("nonfinite_plan")
        if plan.target_speed_mps <= 0:
            return self.stop(plan.reason or "zero_target_speed")
        try:
            points = np.asarray(plan.points, dtype=np.float64)
        except (TypeError, ValueError):
            return self.stop("invalid_path")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            return self.stop("invalid_path")
        points = points[points[:, 0] > 0.05]
        if not len(points):
            return self.stop("no_forward_path")
        dt = c["nominal_dt_s"] if self._last_time is None else now - self._last_time
        if dt <= 0 or dt > c["state_timeout_s"]:
            return self.stop("control_deadline_missed")
        self._last_time = now

        lookahead = c["lookahead_m"] + c["lookahead_speed_gain_s"] * max(0.0, state.speed_mps)
        distances = np.linalg.norm(points, axis=1)
        candidates = np.flatnonzero(distances >= lookahead)
        target = points[candidates[0] if len(candidates) else -1]
        distance_sq = float(target @ target)
        desired_steering = math.atan2(2.0 * c["wheelbase_m"] * float(target[1]), distance_sq)
        desired_steering = float(np.clip(desired_steering, -c["max_steering_rad"], c["max_steering_rad"]))
        delta = c["steering_rate_rad_s"] * dt
        if c["stop_to_align"] and abs(desired_steering - self._steering) > c["alignment_tolerance_rad"]:
            # The planner's curved path assumes the requested wheel angle. Do
            # not propel the vehicle along an unchecked steering transition.
            self._target_speed = 0.0
            self._integral = 0.0
            if abs(state.speed_mps) > c["alignment_speed_mps"]:
                reason = "alignment_braking"
            else:
                self._steering = float(np.clip(desired_steering, self._steering - delta, self._steering + delta))
                reason = "alignment_steering"
            # Keep enabled so the MCU can apply steering while brake holds.
            # No steering encoder exists in this protocol: alignment refers to
            # the command, not measured physical wheel position.
            return Command(steering_rad=self._steering, brake=1.0, enabled=True, reason=reason)
        self._steering = float(np.clip(desired_steering, self._steering - delta, self._steering + delta))

        desired_speed = min(plan.target_speed_mps, c["max_speed_mps"])
        self._target_speed = float(np.clip(
            desired_speed,
            self._target_speed - c["max_decel_mps2"] * dt,
            self._target_speed + c["max_accel_mps2"] * dt,
        ))
        error = self._target_speed - state.speed_mps
        candidate_integral = float(np.clip(
            self._integral + error * dt, -c["integral_limit"], c["integral_limit"]
        ))
        effort = c["kp"] * error + c["ki"] * candidate_integral
        # Conditional integration prevents sustained saturation winding up the PI.
        if (-1.0 <= effort <= c["max_throttle"]
                or (effort > c["max_throttle"] and error < 0)
                or (effort < -1.0 and error > 0)):
            self._integral = candidate_integral
        effort = c["kp"] * error + c["ki"] * self._integral
        throttle = min(c["max_throttle"], max(0.0, effort))
        brake = min(1.0, max(0.0, -effort))
        return Command(
            steering_rad=self._steering, throttle=throttle, brake=brake,
            target_speed_mps=self._target_speed, enabled=True, reason="tracking",
        )
