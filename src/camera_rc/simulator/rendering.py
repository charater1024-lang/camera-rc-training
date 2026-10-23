"""Deterministic CPU rendering for the synthetic driving playground.

The camera is a synthetic chase view, six metres behind the vehicle's rear
axle, so that the planner can see its entire ground footprint.  These camera
parameters are a simulation convention, not a physical camera calibration.
The returned road probabilities are simulator ground truth, not inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 320
HEIGHT = 180
FOCAL = 90.0
CAMERA_HEIGHT_M = 3.0
CAMERA_X_M = -6.0
HORIZON = 45.0


def project_ground(
    x_forward_array: np.ndarray, y_left_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project rear-axle-relative ground points to integer-coordinate pixels.

    Inputs broadcast according to NumPy rules.  ``valid`` identifies finite
    points in front of the camera whose projections lie inside the image.
    Invalid or nonpositive depths have NaN image coordinates.
    """
    x_forward, y_left = np.broadcast_arrays(
        np.asarray(x_forward_array, dtype=np.float64),
        np.asarray(y_left_array, dtype=np.float64),
    )
    depth = x_forward - CAMERA_X_M
    in_front = (depth > 0.0) & np.isfinite(depth) & np.isfinite(y_left)
    scale = np.full(depth.shape, np.nan, dtype=np.float64)
    np.divide(FOCAL, depth, out=scale, where=in_front)
    u = WIDTH / 2.0 - scale * y_left
    v = HORIZON + scale * CAMERA_HEIGHT_M
    valid = in_front & (u >= 0.0) & (u < WIDTH) & (v >= 0.0) & (v < HEIGHT)
    return u, v, valid


def ground_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel ground coordinates inverse to :func:`project_ground`.

    Pixel coordinates are integers, without a half-pixel offset.  The first
    two arrays hold vehicle-relative forward and left distances in metres.
    Pixels on or above the horizon are invalid and contain NaN coordinates.
    """
    u, v = np.meshgrid(
        np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32)
    )
    valid = v > HORIZON
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    np.divide(FOCAL * CAMERA_HEIGHT_M, v - HORIZON, out=depth, where=valid)
    x_forward = depth + CAMERA_X_M
    y_left = (WIDTH / 2.0 - u) * depth / FOCAL
    return x_forward, y_left, valid


# Reuse this fixed camera grid.  Render operations only read these arrays.
_GROUND_X, _GROUND_Y, _GROUND_VALID = ground_coordinates()


def _road_distance_squared(
    world_x: np.ndarray, world_y: np.ndarray, road_points: np.ndarray
) -> np.ndarray:
    """Nearest polyline distance, with vectorized batches of 16 segments."""
    starts = road_points[:-1]
    deltas = road_points[1:] - starts
    length_squared = np.sum(deltas * deltas, axis=1)
    nearest = np.full(world_x.shape, np.inf, dtype=np.float32)
    for offset in range(0, len(starts), 16):
        start = starts[offset : offset + 16]
        delta = deltas[offset : offset + 16]
        denominator = length_squared[offset : offset + 16]
        dx = world_x[:, None] - start[None, :, 0]
        dy = world_y[:, None] - start[None, :, 1]
        along = dx * delta[None, :, 0] + dy * delta[None, :, 1]
        # A repeated point is a zero-length segment, which remains a point.
        np.divide(along, denominator, out=along, where=denominator > 0.0)
        along[:, denominator == 0.0] = 0.0
        np.clip(along, 0.0, 1.0, out=along)
        dx -= along * delta[None, :, 0]
        dy -= along * delta[None, :, 1]
        nearest = np.minimum(nearest, np.min(dx * dx + dy * dy, axis=1))
    return nearest


def _lit_object_color(
    color: tuple[int, int, int], depth: float, y_left: float, condition: str
) -> tuple[int, int, int]:
    result = np.asarray(color, dtype=np.float64)
    if condition == "night":
        beam_width = 2.0 + 0.18 * max(0.0, depth + CAMERA_X_M)
        beam = np.exp(-((y_left / beam_width) ** 4))
        exposure = 0.18 + 0.78 * beam * np.exp(-max(depth - 6.0, 0.0) / 55.0)
        result *= exposure
    elif condition == "fog":
        mist = 1.0 - np.exp(-depth / 38.0)
        result = result * (1.0 - mist) + np.array([186, 194, 197]) * mist
    return tuple(np.clip(result, 0, 255).astype(np.uint8).tolist())


def _draw_obstacle(
    drawing: ImageDraw.ImageDraw,
    mask: ImageDraw.ImageDraw,
    x_forward: float,
    y_left: float,
    radius: float,
    kind: str,
    condition: str,
) -> None:
    depth = x_forward - CAMERA_X_M
    if depth <= 0.1:
        return
    scale = FOCAL / depth
    u = WIDTH / 2.0 - scale * y_left
    bottom = HORIZON + scale * CAMERA_HEIGHT_M
    half_width = max(1.0, radius * scale)
    person = kind.lower() in {"person", "pedestrian", "human"}
    cone = kind.lower() in {"cone", "traffic_cone"}
    height_m = 1.7 if person else (0.75 if cone else 1.35)
    top = bottom - height_m * scale
    if u + half_width < 0 or u - half_width >= WIDTH or bottom < 0 or top >= HEIGHT:
        return

    def color(value: tuple[int, int, int]) -> tuple[int, int, int]:
        return _lit_object_color(value, depth, y_left, condition)

    if person:
        head_radius = max(1.0, 0.16 * scale)
        head = (u - head_radius, top, u + head_radius, top + 2 * head_radius)
        drawing.ellipse(head, fill=color((225, 176, 128)))
        mask.ellipse(head, fill=255)
        torso_top = top + 2 * head_radius
        torso_bottom = bottom - 0.65 * scale
        torso = [
            (u - half_width, torso_top),
            (u + half_width, torso_top),
            (u + half_width * 0.7, torso_bottom),
            (u - half_width * 0.7, torso_bottom),
        ]
        drawing.polygon(torso, fill=color((237, 126, 53)))
        mask.polygon(torso, fill=255)
        leg_width = max(1, round(0.12 * scale))
        for direction in (-1, 1):
            leg = [
                (u + direction * half_width * 0.36, torso_bottom),
                (u + direction * half_width * 0.55, bottom),
            ]
            drawing.line(leg, fill=color((33, 45, 66)), width=leg_width)
            mask.line(leg, fill=255, width=leg_width)
    elif cone:
        triangle = [(u, top), (u + half_width, bottom), (u - half_width, bottom)]
        drawing.polygon(triangle, fill=color((244, 126, 39)))
        mask.polygon(triangle, fill=255)
        stripe_y = top + (bottom - top) * 0.62
        drawing.line(
            [(u - half_width * 0.55, stripe_y), (u + half_width * 0.55, stripe_y)],
            fill=color((242, 235, 212)),
            width=max(1, round(scale * 0.11)),
        )
    else:
        # A small offset roof and darker side give the ground-anchored block
        # depth without relying on a 3D renderer or external artwork.
        offset = min(half_width * 0.42, scale * 0.3)
        front = [
            (u - half_width, top + offset),
            (u + half_width - offset, top + offset),
            (u + half_width - offset, bottom),
            (u - half_width, bottom),
        ]
        side = [
            (u + half_width - offset, top + offset),
            (u + half_width, top),
            (u + half_width, bottom - offset),
            (u + half_width - offset, bottom),
        ]
        roof = [
            (u - half_width, top + offset),
            (u - half_width + offset, top),
            (u + half_width, top),
            (u + half_width - offset, top + offset),
        ]
        for polygon, fill in (
            (front, (190, 69, 52)),
            (side, (119, 43, 39)),
            (roof, (223, 112, 80)),
        ):
            drawing.polygon(polygon, fill=color(fill))
            mask.polygon(polygon, fill=255)
        if kind.lower() in {"car", "vehicle", "truck"}:
            window = (
                u - half_width * 0.72,
                top + offset + 0.14 * scale,
                u + half_width - offset - half_width * 0.28,
                top + offset + 0.44 * scale,
            )
            drawing.rectangle(window, fill=color((94, 139, 158)))


def render_observation(
    vehicle: Mapping[str, float],
    road_points: np.ndarray,
    road_half_width_m: float,
    obstacles: Sequence[Mapping[str, object]],
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB pixels and exact visible-road probabilities on the CPU.

    Road points and obstacle centres use world ``(x_m, y_m)`` coordinates.
    The vehicle yaw rotates its forward axis counterclockwise from world x.
    Road paint and texture stay anchored in world coordinates as it moves.
    Obstacle ground footprints and their visible silhouettes are non-road.
    """
    if condition not in {"day", "night", "fog"}:
        raise ValueError("condition must be 'day', 'night', or 'fog'")
    road = np.asarray(road_points, dtype=np.float32)
    if road.ndim != 2 or road.shape[1] != 2 or len(road) < 2 or not np.isfinite(road).all():
        raise ValueError("road_points must contain at least two finite (x, y) points")
    if not np.isfinite(road_half_width_m) or road_half_width_m <= 0:
        raise ValueError("road_half_width_m must be finite and positive")
    pose = np.asarray([vehicle["x_m"], vehicle["y_m"], vehicle["yaw_rad"]], dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError("vehicle position and yaw must be finite")

    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    local_x = _GROUND_X[_GROUND_VALID]
    local_y = _GROUND_Y[_GROUND_VALID]
    world_x = (pose[0] + cosine * local_x - sine * local_y).astype(np.float32)
    world_y = (pose[1] + sine * local_x + cosine * local_y).astype(np.float32)
    distance_squared = _road_distance_squared(world_x, world_y, road)
    road_surface = distance_squared <= road_half_width_m * road_half_width_m

    rng = np.random.default_rng(seed)
    texture_phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    texture = (
        1.8 * np.sin(world_x * 3.1 + world_y * 2.3 + texture_phase[0])
        + 1.2 * np.sin(world_x * 8.3 - world_y * 4.7 + texture_phase[1])
    ).astype(np.float32)
    ground = np.empty((len(world_x), 3), dtype=np.float32)
    ground[:] = (72, 113, 65)
    ground += (texture * 2.2)[:, None]
    ground[road_surface] = np.array((91, 96, 100)) + texture[road_surface, None]

    # Dashed paint follows world x, ensuring forward motion is visible even
    # on a straight, constant-width road.  It is never added outside asphalt.
    dashed = (np.remainder(world_x, 8.0) < 4.0) & (distance_squared < 0.055**2)
    edge = road_surface & (distance_squared > max(0.0, road_half_width_m - 0.09) ** 2)
    ground[dashed & road_surface] = (220, 199, 113)
    ground[edge] = (219, 222, 214)

    obstacle_views: list[tuple[float, float, float, str]] = []
    for obstacle in obstacles:
        values = np.asarray(
            [obstacle["x_m"], obstacle["y_m"], obstacle["radius_m"]], dtype=np.float64
        )
        if not np.isfinite(values).all() or values[2] <= 0:
            raise ValueError("obstacle positions must be finite and radii positive")
        ox, oy, radius = values
        footprint = (world_x - ox) ** 2 + (world_y - oy) ** 2 <= radius * radius
        road_surface[footprint] = False
        ground[footprint] *= 0.46
        dx, dy = ox - pose[0], oy - pose[1]
        obstacle_views.append(
            (cosine * dx + sine * dy, -sine * dx + cosine * dy, radius, str(obstacle.get("kind", "obstacle")))
        )

    depth = local_x - CAMERA_X_M
    row_fraction = np.arange(HEIGHT, dtype=np.float32) / HORIZON
    if condition == "night":
        sky_top, sky_horizon = np.array((7, 13, 30)), np.array((29, 42, 63))
        beam_width = 2.0 + 0.18 * np.maximum(local_x, 0.0)
        beam = np.exp(-((local_y / beam_width) ** 4))
        exposure = 0.14 + 0.86 * beam * np.exp(-np.maximum(local_x, 0.0) / 55.0)
        ground *= exposure[:, None]
        ground *= np.array((0.93, 0.97, 1.07))
    elif condition == "fog":
        sky_top, sky_horizon = np.array((166, 180, 189)), np.array((186, 194, 197))
        mist = (1.0 - np.exp(-depth / 38.0))[:, None]
        ground = ground * (1.0 - mist) + sky_horizon * mist
    else:
        sky_top, sky_horizon = np.array((82, 153, 210)), np.array((182, 211, 226))
        ground *= (0.86 + 0.14 * np.exp(-depth / 100.0))[:, None]

    sky = sky_top + np.minimum(row_fraction, 1.0)[:, None] * (sky_horizon - sky_top)
    rgb = np.broadcast_to(sky[:, None, :], (HEIGHT, WIDTH, 3)).copy()
    rgb[_GROUND_VALID] = ground
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    drawing = ImageDraw.Draw(image)
    if condition == "day":
        drawing.ellipse((252, 9, 267, 24), fill=(249, 237, 190))
    elif condition == "night":
        for star_x, star_y in rng.integers((0, 2), (WIDTH, int(HORIZON) - 4), size=(30, 2)):
            drawing.point((int(star_x), int(star_y)), fill=(117, 137, 168))
        drawing.ellipse((253, 10, 263, 20), fill=(183, 195, 207))

    occlusion = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_drawing = ImageDraw.Draw(occlusion)
    for x_forward, y_left, radius, kind in sorted(obstacle_views, key=lambda item: item[0], reverse=True):
        _draw_obstacle(drawing, mask_drawing, x_forward, y_left, radius, kind, condition)
    probability = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    probability[_GROUND_VALID] = road_surface.astype(np.float32)
    probability[np.asarray(occlusion) != 0] = 0.0
    return np.asarray(image).copy(), probability


def segmentation_overlay(rgb: np.ndarray, probability: np.ndarray) -> Image.Image:
    """Tint road pixels green while retaining the underlying scene details."""
    pixels = np.asarray(rgb)
    probabilities = np.asarray(probability, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    if probabilities.shape != pixels.shape[:2]:
        raise ValueError("probability must match the RGB image height and width")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probability must contain finite values in [0, 1]")
    alpha = (0.42 * probabilities)[..., None]
    blended = pixels.astype(np.float32) * (1.0 - alpha) + np.array((55, 220, 130)) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
