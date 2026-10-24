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
# Mapped scenes use a bounded local view; original fixtures retain their view.
MAP_RENDER_RANGE_M = 100.0
_NEAR_PLANE_M = 0.1


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
_PIXEL_LEFT_RAY = (WIDTH / 2.0 - np.arange(WIDTH)) / FOCAL
_PIXEL_UP_RAY = (HORIZON - np.arange(HEIGHT)) / FOCAL


def point_in_polygon(x, y, polygon: np.ndarray) -> np.ndarray:
    """Even-odd inclusion for a simple, possibly concave footprint.

    Polygon edges count as inside. Coordinates broadcast; vertices use world
    metres. This helper does not assume a particular polygon winding.
    """
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    points = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        dx, dy = b - a
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-16:
            continue
        cross = (x - a[0]) * dy - (y - a[1]) * dx
        along = (x - a[0]) * dx + (y - a[1]) * dy
        boundary |= (np.abs(cross) <= 1e-8 * np.sqrt(length_squared)) & (along >= 0) & (along <= length_squared)
        if abs(dy) > 1e-12:
            intersection = a[0] + (y - a[1]) * dx / dy
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < intersection)
    return inside | boundary


def _feature_points(value, minimum: int, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"{label} must contain at least {minimum} finite (x, y) points")
    return points


def _near_camera(points: np.ndarray, camera_xy: np.ndarray, margin: float = 0.0) -> bool:
    """Conservative AABB cull; long features crossing the view are retained."""
    nearest = np.clip(camera_xy, points.min(axis=0), points.max(axis=0))
    return bool(np.hypot(*(nearest - camera_xy)) <= MAP_RENDER_RANGE_M + margin)


def _mapped_road_masks(world_x, world_y, roads, camera_xy):
    surface = np.zeros(world_x.shape, dtype=bool)
    drivable = np.zeros(world_x.shape, dtype=bool)
    interior = np.zeros(world_x.shape, dtype=bool)
    dashed = np.zeros(world_x.shape, dtype=bool)
    in_range = np.hypot(world_x - camera_xy[0], world_y - camera_xy[1]) <= MAP_RENDER_RANGE_M
    for feature in roads:
        # This renderer has one ground plane. Flattening a bridge or tunnel
        # onto that plane would invent a connection that the map does not have.
        if feature.get("ground_geometry_supported", True) is False:
            continue
        points = _feature_points(feature["points"], 2, "road points")
        width = float(feature["half_width_m"])
        if not np.isfinite(width) or width <= 0:
            raise ValueError("road half_width_m must be finite and positive")
        if not _near_camera(points, camera_xy, width):
            continue
        lower, upper = points.min(axis=0) - width, points.max(axis=0) + width
        candidate = in_range & (world_x >= lower[0]) & (world_x <= upper[0]) & (world_y >= lower[1]) & (world_y <= upper[1])
        if not candidate.any():
            continue
        # Segment culling avoids spending raster work on a long map way's
        # remote bends while preserving segments that cross the local view.
        segment_lower = np.minimum(points[:-1], points[1:]) - width
        segment_upper = np.maximum(points[:-1], points[1:]) + width
        keep = np.all(segment_lower <= camera_xy + MAP_RENDER_RANGE_M, axis=1) & np.all(segment_upper >= camera_xy - MAP_RENDER_RANGE_M, axis=1)
        distance_squared = np.full(np.count_nonzero(candidate), np.inf, dtype=np.float64)
        px, py = world_x[candidate], world_y[candidate]
        for index in np.flatnonzero(keep):
            distance_squared = np.minimum(distance_squared, _road_distance_squared(px, py, points[index : index + 2]))
        surface[candidate] |= distance_squared <= width * width
        interior[candidate] |= distance_squared < max(0.0, width - 0.09) ** 2
        if feature.get("drivable", True) is not False:
            drivable[candidate] |= distance_squared <= width * width
            dashed[candidate] |= (distance_squared < 0.055**2) & (np.remainder(px + py, 8.0) < 4.0)
    return surface, drivable, dashed & drivable, surface & ~interior


def _building_views(buildings, pose, cosine, sine):
    camera_xy = pose[:2] + CAMERA_X_M * np.array((cosine, sine))
    views = []
    for feature in buildings:
        polygon = _feature_points(feature["polygon"], 3, "building polygon")
        height = float(feature["height_m"])
        if not np.isfinite(height) or not 0 < height <= 2000:
            raise ValueError("building height_m must be finite and in (0, 2000]")
        if not _near_camera(polygon, camera_xy):
            continue
        offset = polygon - camera_xy
        local = np.column_stack((cosine * offset[:, 0] + sine * offset[:, 1], -sine * offset[:, 0] + cosine * offset[:, 1]))
        if local[:, 0].max() < _NEAR_PLANE_M:
            continue
        # IDs only choose a stable material; Python's randomized hash is not
        # used, and feature iteration order cannot change facade colors.
        identity = str(feature.get("id", "building"))
        tone = sum((i + 1) * ord(char) for i, char in enumerate(identity)) % 5
        base = np.array(((155, 160, 158), (168, 159, 146), (151, 160, 168), (177, 166, 150), (149, 153, 145))[tone])
        views.append((polygon, local, height, base))
    return views


def _shade_building(colors, depth, left, condition):
    colors = np.asarray(colors, dtype=np.float64)
    if condition == "fog":
        mist = (1.0 - np.exp(-depth / 38.0))[..., None]
        return colors * (1.0 - mist) + np.array((186, 194, 197)) * mist
    if condition == "night":
        beam_width = 2.0 + 0.18 * np.maximum(depth + CAMERA_X_M, 0.0)
        exposure = 0.18 + 0.78 * np.exp(-((left / beam_width) ** 4)) * np.exp(-np.maximum(depth - 6.0, 0.0) / 55.0)
        return colors * exposure[..., None]
    return colors


def _draw_buildings(rgb, probability, depth_buffer, views, condition):
    """Intersect camera rays with extruded polygons and keep nearest hits.

    A vertical facade has one depth per image column. Testing its height for
    each row yields exact planar facade coverage, with near/range clipping
    before projection. This avoids giant screen polygons at the near plane
    and handles concave buildings without convex roof triangulation.
    """
    for polygon, local, height, base in views:
        for a, b in zip(local, np.roll(local, -1, axis=0)):
            delta = b - a
            edge_length = float(np.hypot(*delta))
            if edge_length < 1e-8:
                continue
            denominator = delta[1] - _PIXEL_LEFT_RAY * delta[0]
            depth = np.full(WIDTH, np.inf)
            fraction = np.full(WIDTH, np.inf)
            nonparallel = np.abs(denominator) > 1e-10
            np.divide(a[0] * delta[1] - a[1] * delta[0], denominator, out=depth, where=nonparallel)
            np.divide(a[0] * _PIXEL_LEFT_RAY - a[1], denominator, out=fraction, where=nonparallel)
            valid = nonparallel & (fraction >= 0.0) & (fraction <= 1.0) & (depth >= _NEAR_PLANE_M)
            valid &= np.abs(depth) <= MAP_RENDER_RANGE_M / np.sqrt(1.0 + _PIXEL_LEFT_RAY**2)
            columns = np.flatnonzero(valid)
            if not len(columns):
                continue
            face_depth = depth[columns]
            z = CAMERA_HEIGHT_M + _PIXEL_UP_RAY[:, None] * face_depth[None, :]
            visible = (z >= 0.0) & (z <= height) & (face_depth[None, :] < depth_buffer[:, columns])
            if not visible.any():
                continue
            # The same world-facing material is used on every face, with a
            # modest orientation shade and metre-anchored window pattern.
            shade = 0.78 + 0.20 * abs(delta[1]) / edge_length
            colors = np.broadcast_to(base * shade, (HEIGHT, len(columns), 3)).copy()
            horizontal = np.remainder(fraction[columns] * edge_length, 3.0)
            windows = (z > 1.1) & (np.remainder(z, 3.2) > 1.0) & (np.remainder(z, 3.2) < 2.4)
            windows &= (horizontal[None, :] > 0.65) & (horizontal[None, :] < 2.15)
            colors[windows] = (76, 103, 117) if condition != "night" else (169, 145, 93)
            colors = _shade_building(colors, face_depth[None, :], (_PIXEL_LEFT_RAY[columns] * face_depth)[None, :], condition)
            rows, indices = np.nonzero(visible)
            cols = columns[indices]
            rgb[rows, cols] = np.clip(colors[rows, indices], 0, 255).astype(np.uint8)
            probability[rows, cols] = 0.0
            depth_buffer[rows, cols] = face_depth[indices]
        # Roof tops can only be seen when the camera is above their plane.
        # Even-odd inclusion preserves notches in concave outlines.
        if height < CAMERA_HEIGHT_M:
            roof_depth = np.full(HEIGHT, np.inf)
            np.divide(height - CAMERA_HEIGHT_M, _PIXEL_UP_RAY, out=roof_depth, where=_PIXEL_UP_RAY < 0)
            candidate_rows = np.flatnonzero((roof_depth >= _NEAR_PLANE_M) & (roof_depth <= MAP_RENDER_RANGE_M))
            if not len(candidate_rows):
                continue
            x = np.broadcast_to(roof_depth[candidate_rows, None], (len(candidate_rows), WIDTH))
            y = x * _PIXEL_LEFT_RAY[None, :]
            lower, upper = local.min(axis=0), local.max(axis=0)
            candidate = (x >= lower[0]) & (x <= upper[0]) & (y >= lower[1]) & (y <= upper[1])
            candidate &= np.hypot(x, y) <= MAP_RENDER_RANGE_M
            candidate &= x < depth_buffer[candidate_rows]
            if not candidate.any():
                continue
            inside = point_in_polygon(x[candidate], y[candidate], local)
            rows, cols = np.nonzero(candidate)
            rows, cols = rows[inside], cols[inside]
            actual_rows = candidate_rows[rows]
            colors = _shade_building(base * 1.1, x[rows, cols], y[rows, cols], condition)
            rgb[actual_rows, cols] = np.clip(colors, 0, 255).astype(np.uint8)
            probability[actual_rows, cols] = 0.0
            depth_buffer[actual_rows, cols] = x[rows, cols]


def _draw_depth_actors(rgb, probability, depth_buffer, obstacle_views, condition):
    for x_forward, y_left, radius, kind in obstacle_views:
        depth = x_forward - CAMERA_X_M
        if depth < _NEAR_PLANE_M or np.hypot(depth, y_left) - radius > MAP_RENDER_RANGE_M:
            continue
        actor_image = Image.new("RGB", (WIDTH, HEIGHT))
        actor_mask = Image.new("L", (WIDTH, HEIGHT))
        _draw_obstacle(ImageDraw.Draw(actor_image), ImageDraw.Draw(actor_mask), x_forward, y_left, radius, kind, condition)
        visible = (np.asarray(actor_mask) != 0) & (depth < depth_buffer)
        rgb[visible] = np.asarray(actor_image)[visible]
        probability[visible] = 0.0
        depth_buffer[visible] = depth


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
    *,
    roads: Sequence[Mapping[str, object]] | None = None,
    buildings: Sequence[Mapping[str, object]] | None = None,
    terrain: Mapping[str, object] | None = None,
    lane_layout: Mapping[str, object] | None = None,
    traffic: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB pixels and exact visible-road probabilities on the CPU.

    Road points and obstacle centres use world ``(x_m, y_m)`` coordinates.
    The vehicle yaw rotates its forward axis counterclockwise from world x.
    Road paint and texture stay anchored in world coordinates as it moves.
    Obstacle ground footprints and their visible silhouettes are non-road.
    When supplied, ``roads`` replaces the single route with the union of map
    polylines (``points``, ``half_width_m``). Buildings contain ``polygon`` and
    ``height_m`` and occlude road and actor pixels through a depth buffer.
    Map features beyond the 100-metre camera neighbourhood are culled.
    Unsupported elevated ways are omitted; non-drivable ways are pavement
    visually, but never contribute to the returned road probability.
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
    camera_xy = pose[:2] + CAMERA_X_M * np.array((cosine, sine))
    mapped = roads is not None or buildings is not None
    building_views = _building_views(buildings or (), pose, cosine, sine)
    local_x = _GROUND_X[_GROUND_VALID]
    local_y = _GROUND_Y[_GROUND_VALID]
    world_x = (pose[0] + cosine * local_x - sine * local_y).astype(np.float32)
    world_y = (pose[1] + sine * local_x + cosine * local_y).astype(np.float32)
    if roads is None:
        distance_squared = _road_distance_squared(world_x, world_y, road)
        road_surface = distance_squared <= road_half_width_m * road_half_width_m
        dashed = (np.remainder(world_x, 8.0) < 4.0) & (distance_squared < 0.055**2)
        edge = road_surface & (distance_squared > max(0.0, road_half_width_m - 0.09) ** 2)
        painted_surface = road_surface.copy()
    else:
        painted_surface, road_surface, dashed, edge = _mapped_road_masks(world_x, world_y, roads, camera_xy)
    if mapped:
        in_range = np.hypot(world_x - camera_xy[0], world_y - camera_xy[1]) <= MAP_RENDER_RANGE_M
        road_surface &= in_range
        painted_surface &= in_range
        dashed &= in_range
        edge &= in_range

    rng = np.random.default_rng(seed)
    texture_phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    texture = (
        1.8 * np.sin(world_x * 3.1 + world_y * 2.3 + texture_phase[0])
        + 1.2 * np.sin(world_x * 8.3 - world_y * 4.7 + texture_phase[1])
    ).astype(np.float32)
    ground = np.empty((len(world_x), 3), dtype=np.float32)
    scenery=(terrain or {}).get('scenery',{})
    ground[:] = scenery.get('ground_color',(72,113,65))
    ground += (texture * 2.2)[:, None]
    if scenery.get('kind') in {'coast','wetland'}:
        side=1 if scenery.get('water_side','right')=='left' else -1
        water=(world_y*side>55)&~painted_surface
        ground[water]=np.asarray(scenery.get('water_color',(47,124,158)))+texture[water,None]
    ground[painted_surface] = np.array((91, 96, 100)) + texture[painted_surface, None]
    ground[painted_surface & ~road_surface] = (142, 141, 132)

    # Dashed paint follows world x, ensuring forward motion is visible even
    # on a straight, constant-width road.  It is never added outside asphalt.
    if lane_layout:
        route=np.asarray(lane_layout['lanes'][0]['centerline'],dtype=np.float32)
        guide_zone=_road_distance_squared(world_x,world_y,route)<6.0**2
        dashed[guide_zone]=False
        if not mapped:
            dashed[:]=False
    ground[dashed & road_surface] = (220, 199, 113)
    ground[edge] = (219, 222, 214)
    if lane_layout:
        for key,color in (('divider',(231,195,70)),('ego_outer_boundary',(242,245,240)),('opposing_outer_boundary',(242,245,240))):
            boundary=np.asarray(lane_layout[key],dtype=np.float32)
            paint=(_road_distance_squared(world_x,world_y,boundary)<.085**2)&road_surface
            ground[paint]=color
    if traffic:
        crossing=traffic['crosswalk']
        zebra=(world_x>=crossing['x_min_m'])&(world_x<=crossing['x_max_m'])&(world_y>=crossing['y_min_m'])&(world_y<=crossing['y_max_m'])
        zebra &= (np.floor((world_x-crossing['x_min_m'])/.45).astype(int)%2)==0
        ground[zebra&road_surface]=(238,241,231)
        stop=(np.abs(world_x-29.5)<.12)&(world_y>=-1.8)&(world_y<=1.8)
        ground[stop&road_surface]=(238,241,231)

    for polygon, _local, _height, _base in building_views:
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        candidate = (world_x >= lower[0]) & (world_x <= upper[0]) & (world_y >= lower[1]) & (world_y <= upper[1])
        if not candidate.any():
            continue
        indices = np.flatnonzero(candidate)
        footprint = indices[point_in_polygon(world_x[candidate], world_y[candidate], polygon)]
        road_surface[footprint] = False
        ground[footprint] = (127, 128, 122)

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
    if scenery.get('ridge_peaks'):
        peaks=scenery['ridge_peaks']
        vertices=[(0,int(HORIZON))]
        for index,peak in enumerate(peaks):
            vertices.append((round(index*WIDTH/max(1,len(peaks)-1)),round(HORIZON-float(peak)*30)))
        vertices.append((WIDTH,int(HORIZON)))
        drawing.polygon(vertices,fill=tuple(scenery.get('ridge_color',(71,112,87))))
    if condition == "day":
        drawing.ellipse((252, 9, 267, 24), fill=(249, 237, 190))
    elif condition == "night":
        for star_x, star_y in rng.integers((0, 2), (WIDTH, int(HORIZON) - 4), size=(30, 2)):
            drawing.point((int(star_x), int(star_y)), fill=(117, 137, 168))
        drawing.ellipse((253, 10, 263, 20), fill=(183, 195, 207))

    occlusion = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_drawing = ImageDraw.Draw(occlusion)
    if not mapped:
        for x_forward, y_left, radius, kind in sorted(obstacle_views, key=lambda item: item[0], reverse=True):
            _draw_obstacle(drawing, mask_drawing, x_forward, y_left, radius, kind, condition)
    probability = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    probability[_GROUND_VALID] = road_surface.astype(np.float32)
    if mapped:
        pixels = np.asarray(image).copy()
        depth_buffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
        depth_buffer[_GROUND_VALID] = _GROUND_X[_GROUND_VALID] - CAMERA_X_M
        _draw_buildings(pixels, probability, depth_buffer, building_views, condition)
        _draw_depth_actors(pixels, probability, depth_buffer, obstacle_views, condition)
        return pixels, probability
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
