"""Conservative, low-speed geometric planning on a calibrated flat ground.

This samples constant-curvature paths and their vehicle footprints. Unknown
ground is blocked. Detection bottom edges are conservatively approximated as
ground obstacles; this is not reliable 3D obstacle perception or a guarantee
of collision avoidance. Rectified input, fixed camera pose, a flat closed
course and external supervision remain necessary.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import GroundProjector, constant_curvature_path
from .types import Perception, Plan


class LocalPlanner:
    def __init__(self, config: dict, calibration: dict | GroundProjector):
        self.projector = calibration if isinstance(calibration, GroundProjector) else GroundProjector(calibration)
        self.config = dict(config)
        defaults = {"horizon_m": 2.0, "sample_step_m": 0.05, "vehicle_width_m": 0.22,
                    "vehicle_length_m": 0.35, "wheelbase_m": 0.26, "max_steering_rad": 0.45,
                    "max_speed_mps": 0.5, "max_deceleration_mps2": 0.8, "reaction_time_s": 0.3,
                    "clearance_m": 0.05, "min_drivable_probability": 0.6,
                    "max_lateral_acceleration_mps2": 0.5, "max_heading_rad": 1.0,
                    "obstacle_padding_m": 0.15, "stopping_margin_m": 0.15}
        for name, default in defaults.items():
            value = float(config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"planner.{name} must be finite and positive")
            setattr(self, name, value)
        self.assumed_clearance_m = float(config.get("assumed_clearance_m", 0.0))
        if not math.isfinite(self.assumed_clearance_m) or self.assumed_clearance_m < 0:
            raise ValueError("assumed_clearance_m must be finite and nonnegative")
        if not 0 < self.min_drivable_probability <= 1:
            raise ValueError("min_drivable_probability must be in (0,1]")
        if self.vehicle_length_m < self.wheelbase_m:
            raise ValueError("vehicle_length_m must include the wheelbase")
        if not self.max_steering_rad < math.pi / 2 or self.max_heading_rad >= math.pi / 2:
            raise ValueError("max_steering_rad and max_heading_rad must be less than pi/2")
        if self.sample_step_m > min(self.horizon_m / 4, self.vehicle_width_m / 2):
            raise ValueError("sample_step_m must densely sample the horizon and vehicle width")
        if self.horizon_m <= self.stopping_margin_m:
            raise ValueError("horizon_m must exceed stopping_margin_m")
        count = config.get("candidate_count", 11)
        if isinstance(count, bool) or not isinstance(count, int) or count < 3 or count % 2 != 1:
            raise ValueError("candidate_count must be an odd integer >=3")
        self.max_curvature = min(math.tan(self.max_steering_rad) / self.wheelbase_m, self.max_heading_rad / self.horizon_m)
        self.curvatures = np.linspace(-self.max_curvature, self.max_curvature, count)
        self.distances = np.linspace(0, self.horizon_m, math.ceil(self.horizon_m / self.sample_step_m) + 1)
        self.rear = (self.vehicle_length_m - self.wheelbase_m) / 2
        self.front = self.wheelbase_m + self.rear
        self.half_width = self.vehicle_width_m / 2
        # Full interior sampling checks obstacles/mask holes inside the footprint,
        # rather than checking only a centreline or four corners.
        xs = np.linspace(-self.rear, self.front,
                         math.ceil(self.vehicle_length_m / self.sample_step_m) + 1)
        ys = np.linspace(-self.half_width, self.half_width,
                         math.ceil(self.vehicle_width_m / self.sample_step_m) + 1)
        self.footprint = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    @staticmethod
    def _stop(reason):
        return Plan(np.empty((0, 2), dtype=np.float64), 0.0, False, reason, 0.0)

    def _swept_points(self, path, heading):
        cos, sin = np.cos(heading)[:, None], np.sin(heading)[:, None]
        x = path[:, 0:1] + cos * self.footprint[None, :, 0] - sin * self.footprint[None, :, 1]
        y = path[:, 1:2] + sin * self.footprint[None, :, 0] + cos * self.footprint[None, :, 1]
        return np.stack((x, y), axis=-1).reshape(-1, 2)

    def _ground_clear(self, points, probabilities):
        # Exempt only the physical current vehicle rectangle by default; it is
        # already occupied. Any newly swept ground, including near-side blind
        # regions during a turn, must be observed. An explicit nonzero assumed
        # reserve is for simulation experiments only and must be rejected live.
        occupied = ((points[:, 0] >= -self.rear - self.assumed_clearance_m - 1e-9)
                    & (points[:, 0] <= self.front + self.assumed_clearance_m + 1e-9)
                    & (np.abs(points[:, 1]) <= self.half_width + self.assumed_clearance_m + 1e-9))
        points = points[~occupied]
        if not len(points) or not self.projector.in_supported_range(points).all():
            return False, 0.0
        pixels = self.projector.ground_to_pixel(points)
        if not np.isfinite(pixels).all():
            return False, 0.0
        u, v = pixels[:, 0], pixels[:, 1]
        if (u < 0).any() or (v < 0).any() or (u >= self.projector.width - 1).any() or (v >= self.projector.height - 1).any():
            return False, 0.0
        x, y = np.floor(u).astype(int), np.floor(v).astype(int)
        # Every bilinear neighbour must be known drivable. Taking their minimum
        # prevents interpolation from smoothing a small non-drivable pixel away.
        confidence = np.minimum.reduce([probabilities[y, x], probabilities[y, x + 1],
                                        probabilities[y + 1, x], probabilities[y + 1, x + 1]])
        return bool((confidence >= self.min_drivable_probability).all()), float(confidence.mean())

    def _obstacles(self, detections):
        obstacles = []
        for detection in detections:
            box = np.asarray(detection.xyxy, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(detection.score) or not 0 <= detection.score <= 1:
                raise ValueError("invalid_detection")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 < self.projector.width and 0 <= y1 < y2 < self.projector.height):
                raise ValueError("invalid_detection")
            feet = self.projector.pixel_to_ground([[x1, y2], [x2, y2], [(x1 + x2) / 2, y2]])
            if not np.isfinite(feet).all():
                raise ValueError("obstacle_projection_unknown")
            # Objects entirely beyond our validated lookahead are left to future
            # frames. An object touching the near blind region triggers stop.
            if (feet[:, 0] > self.projector.max_forward_m).all():
                continue
            if not self.projector.in_supported_range(feet).all():
                raise ValueError("obstacle_projection_unknown")
            # Ground bottom edge + padding accounts only for coarse geometry;
            # overhanging/unsupported objects cannot be reliably modelled here.
            obstacles.append((feet[:2], self.obstacle_padding_m + self.clearance_m))
        return obstacles

    @staticmethod
    def _hits_obstacle(points, obstacles):
        for (start, end), radius in obstacles:
            segment = end - start
            length_squared = float(segment @ segment)
            amount = np.zeros(len(points)) if length_squared < 1e-12 else np.clip((points - start) @ segment / length_squared, 0, 1)
            nearest = start + amount[:, None] * segment
            if (np.linalg.norm(points - nearest, axis=1) <= radius).any():
                return True
        return False

    def plan(self, perception: Perception, speed_mps: float, route_hint: str = "straight") -> Plan:
        if not math.isfinite(speed_mps) or speed_mps < 0:
            return self._stop("invalid_speed")
        if route_hint not in {"straight", "left", "right", "stop"}:
            return self._stop("invalid_route_hint")
        if route_hint == "stop":
            return self._stop("route_stop")
        mask = np.asarray(perception.drivable)
        if mask.shape != (self.projector.height, self.projector.width):
            return self._stop("calibration_image_size_mismatch")
        if not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            return self._stop("invalid_drivable_probability")
        stopping_distance = speed_mps * self.reaction_time_s + speed_mps ** 2 / (2 * self.max_deceleration_mps2) + self.stopping_margin_m
        if stopping_distance > self.horizon_m:
            return self._stop("speed_exceeds_visible_stopping_distance")
        try:
            obstacles = self._obstacles(perception.detections)
        except (TypeError, ValueError) as exc:
            return self._stop(str(exc))
        return self._plan_candidates(mask, obstacles, route_hint)

    def _plan_candidates(self, probabilities, obstacles, route_hint):
        """Shared path search; subclasses may query ground in multiple views."""
        preferred = {"straight": 0.0, "left": self.max_curvature, "right": -self.max_curvature}[route_hint]
        best = None
        for curvature in self.curvatures:
            path, heading = constant_curvature_path(float(curvature), self.distances)
            swept = self._swept_points(path, heading)
            clear, confidence = self._ground_clear(swept, probabilities)
            if not clear or self._hits_obstacle(swept, obstacles):
                continue
            score = abs(curvature - preferred) / self.max_curvature + 0.1 * (1 - confidence)
            if best is None or score < best[0]:
                best = (score, path, float(curvature))
        if best is None:
            return self._stop("all_paths_blocked_or_unknown")
        _, path, curvature = best
        distance = self.horizon_m - self.stopping_margin_m
        braking_limit = (-self.max_deceleration_mps2 * self.reaction_time_s
                         + math.sqrt((self.max_deceleration_mps2 * self.reaction_time_s) ** 2 + 2 * self.max_deceleration_mps2 * distance))
        turning_limit = math.sqrt(self.max_lateral_acceleration_mps2 / abs(curvature)) if abs(curvature) > 1e-9 else self.max_speed_mps
        target = min(self.max_speed_mps, braking_limit, turning_limit)
        return Plan(path, float(target), True, "clear_path", curvature)
