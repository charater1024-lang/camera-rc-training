"""Flat-ground camera projection in metres, relative to the rear axle.

The homography is valid only for the calibrated camera pose and rectified
image size. Hills, bumps, camera movement and non-ground object pixels violate
this model. It is a restricted-course baseline, not metric depth perception.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GroundProjector:
    def __init__(self, config: dict | str | Path):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("Calibration must be a JSON object")
        if config.get("calibrated") is not True:
            raise ValueError("A measured calibration with calibrated=true is required")
        if config.get("rectified") is not True:
            raise ValueError("Calibration requires rectified=true and rectified input images")
        size = config.get("image_size")
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in size
        ):
            raise ValueError("Calibration image_size must be [height,width], integers >=2")
        self.height, self.width = size
        self.image_to_ground = np.asarray(config.get("image_to_ground"), dtype=np.float64)
        if self.image_to_ground.shape != (3, 3) or not np.isfinite(self.image_to_ground).all():
            raise ValueError("image_to_ground must be a finite 3x3 homography")
        scale = np.max(np.abs(self.image_to_ground))
        if scale == 0:
            raise ValueError("image_to_ground must be invertible")
        self.image_to_ground = self.image_to_ground / scale
        if np.linalg.cond(self.image_to_ground) > 1e12:
            raise ValueError("image_to_ground is singular or ill-conditioned")
        self.ground_to_image = np.linalg.inv(self.image_to_ground)
        limits = config.get("supported_range", {})
        try:
            self.min_forward_m = float(limits["min_forward_m"])
            self.max_forward_m = float(limits["max_forward_m"])
            if "min_lateral_m" in limits or "max_lateral_m" in limits:
                self.min_lateral_m = float(limits["min_lateral_m"])
                self.max_lateral_m = float(limits["max_lateral_m"])
            else:
                lateral = float(limits["max_abs_lateral_m"])
                self.min_lateral_m, self.max_lateral_m = -lateral, lateral
            self.max_abs_lateral_m = max(abs(self.min_lateral_m), abs(self.max_lateral_m))
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("supported_range requires min_forward_m, max_forward_m and either max_abs_lateral_m or min_lateral_m/max_lateral_m") from exc
        if not np.isfinite([self.min_forward_m, self.max_forward_m, self.min_lateral_m, self.max_lateral_m]).all() or not (
            self.min_forward_m < self.max_forward_m and self.min_lateral_m < self.max_lateral_m
        ):
            raise ValueError("supported_range must have finite min_forward<max_forward and min_lateral<max_lateral")

    @staticmethod
    def _project(points, matrix):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Projection expects finite Nx2 coordinates")
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
        valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-10)
        result = np.full((len(points), 2), np.nan, dtype=np.float64)
        result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
        return result

    def pixel_to_ground(self, pixels):
        """Map (pixel u,v) to (forward x,left y); horizon points become NaN."""
        return self._project(pixels, self.image_to_ground)

    def ground_to_pixel(self, points):
        return self._project(points, self.ground_to_image)

    def in_supported_range(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected Nx2 ground coordinates")
        return (np.isfinite(points).all(axis=1) & (points[:, 0] >= self.min_forward_m)
                & (points[:, 0] <= self.max_forward_m)
                & (points[:, 1] >= self.min_lateral_m) & (points[:, 1] <= self.max_lateral_m))


def constant_curvature_path(curvature: float, distances):
    """Rear-axle path; positive curvature turns left (+y)."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or not np.isfinite(distances).all() or not np.isfinite(curvature):
        raise ValueError("Path distances and curvature must be finite")
    heading = distances * curvature
    if abs(curvature) < 1e-9:
        points = np.column_stack((distances, np.zeros_like(distances)))
    else:
        points = np.column_stack((np.sin(heading) / curvature, (1.0 - np.cos(heading)) / curvature))
    return points, heading
