"""Fit a ground-plane mapping from measured, lens-rectified correspondences."""
import json
from pathlib import Path
import numpy as np


def fit_ground_calibration(points_path, output):
    import cv2
    from .geometry import GroundProjector
    from .sources import Rectifier
    raw = json.loads(Path(points_path).read_text(encoding="utf-8-sig"))
    train = np.asarray(raw.get("points"), dtype=np.float64)
    heldout = np.asarray(raw.get("validation_points"), dtype=np.float64)
    for name, values, minimum in (("points", train, 4), ("validation_points", heldout, 2)):
        if values.ndim != 2 or values.shape[1] != 4 or len(values) < minimum or not np.isfinite(values).all():
            raise ValueError(f"{name} requires at least {minimum} finite [pixel_u,pixel_v,forward_m,left_m] points")
    if set(map(tuple, train[:, :2])) & set(map(tuple, heldout[:, :2])):
        raise ValueError("Calibration validation points must be independently measured points")
    h, w = raw["image_size"]
    all_points = np.concatenate((train, heldout))
    if (all_points[:, :2] < 0).any() or (all_points[:, 0] >= w).any() or (all_points[:, 1] >= h).any():
        raise ValueError("Calibration pixels lie outside image_size")
    matrix, _ = cv2.findHomography(train[:, :2], train[:, 2:], method=0)
    if matrix is None:
        raise ValueError("Calibration points are degenerate")
    cfg = {key: raw[key] for key in ("image_size", "supported_range", "input_is_rectified", "camera_matrix", "distortion_coefficients") if key in raw}
    cfg.update(calibrated=True, rectified=True, image_to_ground=matrix.tolist(), coordinate_origin="rear_axle_center", diagnostic_only=bool(raw.get("diagnostic_only", False)))
    projector = GroundProjector(cfg)
    Rectifier(cfg)
    corners = [(x, y) for x in (projector.min_forward_m, projector.max_forward_m)
               for y in (projector.min_lateral_m, projector.max_lateral_m)]
    hull = cv2.convexHull(train[:, 2:].astype(np.float32))
    if any(cv2.pointPolygonTest(hull, (float(x), float(y)), False) < 0 for x, y in corners):
        raise ValueError("supported_range must stay inside the measured ground-point convex hull; do not extrapolate")
    residual = np.linalg.norm(projector.pixel_to_ground(heldout[:, :2]) - heldout[:, 2:], axis=1)
    threshold = float(raw.get("max_validation_error_m", 0.05))
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(residual).all() or residual.max() > threshold:
        raise ValueError(f"Independent calibration error {residual.tolist()} exceeds {threshold} metres")
    cfg["fit"] = {"training_points": len(train), "validation_points": len(heldout),
                  "max_validation_error_m": float(residual.max()), "source": str(Path(points_path).resolve()),
                  "note": "Only valid for this camera pose, lens correction, resolution and flat surface. Recheck after remounting."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(cfg, stream, indent=2)
    return {"calibration": str(output.resolve()), **cfg["fit"]}
