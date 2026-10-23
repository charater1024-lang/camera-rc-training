"""Conservative multi-camera ground queries for a low-speed local planner.

Every calibration maps its rectified image to the same rear-axle frame in
metres (x forward, y left). Ground outside all validated views is unknown;
one clear view never overrides a blocked observation in another view. This
is a static, flat-ground baseline, not 3D reconstruction, temporal fusion,
dynamic-object prediction, or a guarantee of collision avoidance. The caller
must enforce frame freshness and cross-camera timing before calling plan().
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import GroundProjector
from .planner import LocalPlanner
from .types import Perception, Plan


class MultiViewPlanner(LocalPlanner):
    def __init__(self, config: dict, calibrations: dict[str, dict]):
        if (not isinstance(calibrations, dict) or not calibrations
                or any(not isinstance(name, str) or not name for name in calibrations)):
            raise ValueError("MultiViewPlanner requires a nonempty named calibration mapping")
        for name, calibration in calibrations.items():
            if not isinstance(calibration, dict) or calibration.get("coordinate_origin") != "rear_axle_center":
                raise ValueError(f"Camera {name} must declare coordinate_origin=rear_axle_center")
        self.projectors = {name: GroundProjector(calibration) for name, calibration in calibrations.items()}
        super().__init__(config, next(iter(self.projectors.values())))

    def _ground_clear(self, points, probabilities):
        # Only the physical vehicle's current rectangle is exempt. Newly swept
        # side/rear ground needs coverage even while the vehicle moves forward.
        occupied = ((points[:, 0] >= -self.rear - self.assumed_clearance_m - 1e-9)
                    & (points[:, 0] <= self.front + self.assumed_clearance_m + 1e-9)
                    & (np.abs(points[:, 1]) <= self.half_width + self.assumed_clearance_m + 1e-9))
        points = points[~occupied]
        if not len(points):
            return False, 0.0
        observed = np.zeros(len(points), dtype=bool)
        confidence = np.ones(len(points), dtype=np.float64)
        for name, projector in self.projectors.items():
            indices = np.flatnonzero(projector.in_supported_range(points))
            if not len(indices):
                continue
            pixels = projector.ground_to_pixel(points[indices])
            valid = (np.isfinite(pixels).all(axis=1)
                     & (pixels[:, 0] >= 0) & (pixels[:, 0] < projector.width - 1)
                     & (pixels[:, 1] >= 0) & (pixels[:, 1] < projector.height - 1))
            indices, pixels = indices[valid], pixels[valid]
            if not len(indices):
                continue
            x, y = np.floor(pixels).astype(np.int64).T
            mask = probabilities[name]
            # All four interpolation neighbours must be known drivable. This
            # also leaves rectification-invalid pixels, supplied as zero, blocked.
            current = np.minimum.reduce([mask[y, x], mask[y, x + 1],
                                         mask[y + 1, x], mask[y + 1, x + 1]])
            confidence[indices] = np.minimum(confidence[indices], current)
            observed[indices] = True
        if not observed.all():
            return False, 0.0
        return bool((confidence >= self.min_drivable_probability).all()), float(confidence.mean())

    def _view_obstacles(self, detections, projector):
        obstacles = []
        for detection in detections:
            box = np.asarray(detection.xyxy, dtype=np.float64)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or not math.isfinite(detection.score) or not 0 <= detection.score <= 1):
                raise ValueError("invalid_detection")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 < projector.width and 0 <= y1 < y2 < projector.height):
                raise ValueError("invalid_detection")
            feet = projector.pixel_to_ground([[x1, y2], [x2, y2], [(x1 + x2) / 2, y2]])
            # There is no forward-only 'far object' exemption for side/rear
            # views: x need not point along that camera's viewing direction.
            if not np.isfinite(feet).all() or not projector.in_supported_range(feet).all():
                raise ValueError("obstacle_projection_unknown")
            obstacles.append((feet[:2], self.obstacle_padding_m + self.clearance_m))
        return obstacles

    def plan(self, perceptions: dict[str, Perception], speed_mps: float, route_hint: str = "straight") -> Plan:
        if not math.isfinite(speed_mps) or speed_mps < 0:
            return self._stop("invalid_speed")
        if route_hint not in {"straight", "left", "right", "stop"}:
            return self._stop("invalid_route_hint")
        if route_hint == "stop":
            return self._stop("route_stop")
        if not isinstance(perceptions, dict) or set(perceptions) != set(self.projectors):
            return self._stop("missing_or_unexpected_camera_views")
        masks = {}
        obstacles = []
        for name, projector in self.projectors.items():
            perception = perceptions[name]
            if not isinstance(perception, Perception):
                return self._stop(f"invalid_perception:{name}")
            try:
                mask = np.asarray(perception.drivable)
            except (TypeError, ValueError):
                return self._stop(f"invalid_drivable_probability:{name}")
            if mask.shape != (projector.height, projector.width):
                return self._stop(f"calibration_image_size_mismatch:{name}")
            if (mask.dtype.kind not in "biuf" or not np.isfinite(mask).all()
                    or (mask < 0).any() or (mask > 1).any()):
                return self._stop(f"invalid_drivable_probability:{name}")
            masks[name] = mask
            try:
                obstacles.extend(self._view_obstacles(perception.detections, projector))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                reason = str(exc) if str(exc) in {"invalid_detection", "obstacle_projection_unknown"} else "invalid_detection"
                return self._stop(f"{reason}:{name}")
        stopping_distance = (speed_mps * self.reaction_time_s
                             + speed_mps ** 2 / (2 * self.max_deceleration_mps2) + self.stopping_margin_m)
        if stopping_distance > self.horizon_m:
            return self._stop("speed_exceeds_visible_stopping_distance")
        return self._plan_candidates(masks, obstacles, route_hint)
