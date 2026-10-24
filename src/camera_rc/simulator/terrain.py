"""Repository-authored regional courses, without geographic or survey claims.

The four courses evoke regional scenery but every road, building, elevation and
width is invented for simulation. They intentionally use the existing region
schema so the 2D renderer and geometric collision checks share one world. No
network, map download, DEM, or measured vehicle data is used by this module.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import bisect
import math

from .region import canonical_bytes, polyline_length, sha256_bytes


COURSE_IDS = ("busan", "gwangju", "suncheon", "gangwon")
GEOMETRY_SOURCE = "synthetic_regional_course"
_SYNTHETIC_NOTE = "Repository-authored synthetic roads, buildings, scenery and elevation; not a map or measured terrain."

_DEFINITIONS = {
    "busan": {
        "name": "부산 해안·구릉 (합성)", "description": "해안 풍경과 완만한 구릉을 가정한 연구 코스",
        "length_x_m": 3600, "straight_m": 1200, "curve_amplitude_m": 85, "curve_cycles": 1.0,
        "half_width_m": 11.0, "building_step_m": 135, "building_height_m": 30,
        "elevation": ((0., 8.), (.34, 8.), (.52, 44.), (.70, 22.), (.86, 66.), (1., 24.)),
        "scenery": {"kind": "coast", "ground_color": [138, 161, 112], "water_color": [54, 139, 183],
                    "ridge_color": [73, 112, 104], "ridge_peaks": [0.22, 0.47, 0.32, 0.65, 0.36],
                    "water_side": "right", "label": "합성 해안·항만·구릉"},
    },
    "gwangju": {
        "name": "광주 도심·평지 (합성)", "description": "긴 직선과 도심 건물을 가정한 평지 연구 코스",
        "length_x_m": 3400, "straight_m": 2400, "curve_amplitude_m": 36, "curve_cycles": .5,
        "half_width_m": 12.0, "building_step_m": 90, "building_height_m": 17,
        "elevation": ((0., 45.), (.71, 45.), (.84, 50.), (1., 48.)),
        "scenery": {"kind": "city", "ground_color": [143, 158, 115], "water_color": [75, 139, 166],
                    "ridge_color": [110, 140, 133], "ridge_peaks": [0.2, 0.28, 0.18, 0.38, 0.24],
                    "water_side": "none", "label": "합성 도심·평지"},
    },
    "suncheon": {
        "name": "전남 순천 들판·습지 (합성)", "description": "들판과 습지 풍경, 완만한 굽이를 가정한 연구 코스",
        "length_x_m": 3500, "straight_m": 1400, "curve_amplitude_m": 105, "curve_cycles": 1.0,
        "half_width_m": 10.0, "building_step_m": 230, "building_height_m": 7,
        "elevation": ((0., 6.), (.40, 6.), (.60, 14.), (.78, 7.), (1., 18.)),
        "scenery": {"kind": "wetland", "ground_color": [157, 174, 93], "water_color": [82, 152, 160],
                    "ridge_color": [99, 137, 115], "ridge_peaks": [0.28, 0.37, 0.24, 0.42, 0.32],
                    "water_side": "left", "label": "합성 들판·습지"},
    },
    "gangwon": {
        "name": "강원 산지 (합성)", "description": "산림 풍경과 굽이·오르막·내리막을 가정한 연구 코스",
        "length_x_m": 3800, "straight_m": 1200, "curve_amplitude_m": 185, "curve_cycles": 1.6,
        "half_width_m": 9.0, "building_step_m": 310, "building_height_m": 6,
        "elevation": ((0., 320.), (.31, 320.), (.47, 366.), (.65, 430.), (.82, 390.), (1., 454.)),
        "scenery": {"kind": "mountain", "ground_color": [98, 139, 84], "water_color": [65, 131, 153],
                    "ridge_color": [58, 101, 85], "ridge_peaks": [0.64, 0.91, 0.68, 1.0, 0.78],
                    "water_side": "none", "label": "합성 산림·산지 굽이"},
    },
}

COURSE_METADATA = {
    course_id: {"id": course_id, "name": definition["name"], "title": definition["name"],
                "description": definition["description"], "geometry_source": GEOMETRY_SOURCE,
                "synthetic": True}
    for course_id, definition in _DEFINITIONS.items()
}


def list_courses():
    """Return defensive copies of selector metadata in a stable order."""
    return deepcopy([COURSE_METADATA[course_id] for course_id in COURSE_IDS])


def _course_points(definition):
    points = []
    total, straight = definition["length_x_m"], definition["straight_m"]
    for x in range(0, total + 1, 25):
        fraction = max(0., (x - straight) / (total - straight))
        # The envelope starts with zero derivative, connecting the long
        # straight smoothly to invented regional bends.
        envelope = .5 - .5 * math.cos(math.pi * min(1., fraction * 3.))
        y = definition["curve_amplitude_m"] * envelope * math.sin(2. * math.pi * definition["curve_cycles"] * fraction)
        points.append([float(x), round(y, 4)])
    return points


def _road(course_id, suffix, points, half_width):
    return {"id": f"synthetic/{course_id}/{suffix}", "name": f"합성 {suffix}",
            "highway": "synthetic_research_course", "points": points, "half_width_m": half_width,
            "width_source": "repository_authored_simulator_assumption",
            "drivable": True, "ground_geometry_supported": True,
            "geometry_source": GEOMETRY_SOURCE}


def _position(points, distance):
    remaining = distance
    for a, b in zip(points, points[1:]):
        segment = math.dist(a, b)
        if remaining <= segment:
            fraction = remaining / segment
            return [a[k] + fraction * (b[k] - a[k]) for k in (0, 1)]
        remaining -= segment
    return points[-1][:]


@lru_cache(maxsize=4)
def _cached_course(course_id):
    if course_id not in _DEFINITIONS:
        raise ValueError(f"Unknown regional course: {course_id!r}; expected one of {', '.join(COURSE_IDS)}")
    definition = _DEFINITIONS[course_id]
    points = _course_points(definition)
    length = polyline_length(points)
    first, last = points[0], points[-1]
    tail_direction = [(last[k] - points[-2][k]) / math.dist(points[-2], last) for k in (0, 1)]
    # Road surface continues beyond both route ends so the whole circular
    # vehicle body fits at start and when stopping in the goal region.
    extended = [[first[0]-180., first[1]], *points,
                [round(last[k]+180.*tail_direction[k], 4) for k in (0, 1)]]
    route_road = _road(course_id, "reference", extended, definition["half_width_m"])
    roads = [route_road]
    for offset in (-130., -70., 70., 130.):
        roads.append(_road(course_id, f"background_{offset:+.0f}",
                           [[x, round(y+offset, 4)] for x, y in extended], 5.5))
    # Connectors are outside the evaluated route, leaving its long straight
    # and bends free of crossing actors. Background roads support the existing
    # seeded 44-actor placement with a protected route corridor.
    for index in (0, -1):
        x, y = extended[index]
        roads.append(_road(course_id, f"connector_{index}", [[x, y-130.], [x, y+130.]], 5.5))
    buildings = []
    for index, distance in enumerate(range(45, int(length), definition["building_step_m"])):
        center = _position(points, distance)
        for side in (-1, 1):
            x, y = center[0], center[1] + side * 40.
            width, depth = 12.+(index % 3)*3., 12.+(index % 2)*4.
            polygon = [[round(x-width/2, 4), round(y-depth/2, 4)],
                       [round(x+width/2, 4), round(y-depth/2, 4)],
                       [round(x+width/2, 4), round(y+depth/2, 4)],
                       [round(x-width/2, 4), round(y+depth/2, 4)]]
            buildings.append({"id": f"synthetic/{course_id}/building_{index}_{side}",
                              "polygon": polygon+[polygon[0][:]],
                              "height_m": float(definition["building_height_m"] + index % 4 * 2),
                              "height_source": "repository_authored_simulator_assumption",
                              "geometry_source": GEOMETRY_SOURCE})
    route = {"name": definition["name"], "road_ids": [route_road["id"]], "points": points,
             "length_m": round(length, 4), "start_heading_rad": 0.,
             "goal_heading_rad": math.atan2(tail_direction[1], tail_direction[0]),
             "half_width_m": definition["half_width_m"],
             "width_source": "repository_authored_simulator_assumption",
             "legal_driving_route": False, "scope": "synthetic_research_course"}
    terrain = {"synthetic": True, "elevation_source": "repository_authored_assumed_profile_not_DEM",
               "elevation_units": "metres", "elevation_reference": "arbitrary_assumed_datum",
               "progress_units": "route_arc_length_metres", "straight_length_m": definition["straight_m"],
               "elevation_profile": [{"distance_m": round(fraction*length, 4), "elevation_m": height}
                                     for fraction, height in definition["elevation"]],
               "scenery": deepcopy(definition["scenery"]),
               "limitations": ["All scenery and elevation values are invented, not regional measurements.",
                               "Planar road geometry plus a scalar longitudinal grade; no 3D tyre, suspension, banking or vertical-occlusion model."]}
    all_points = [point for road in roads for point in road["points"]]
    region = {"schema_version": 1, "id": course_id, "name": definition["name"],
              "description": definition["description"], "synthetic": True,
              "geometry_source": GEOMETRY_SOURCE,
              "projection": {"method": "synthetic_local_cartesian", "x_axis": "local_forward",
                             "y_axis": "local_left", "units": "metres", "georeferenced": False},
              "bounds_m": {"x_min": min(p[0] for p in all_points)-60.,
                           "y_min": min(p[1] for p in all_points)-60.,
                           "x_max": max(p[0] for p in all_points)+60.,
                           "y_max": max(p[1] for p in all_points)+60.},
              "provenance": {"source": _SYNTHETIC_NOTE, "source_url": None,
                             "source_asset": "camera_rc.simulator.terrain", "license": "AGPL-3.0-only",
                             "attribution": "camera-rc-training repository authors; synthetic geometry",
                             "geographic_data_used": False, "measured_elevation_used": False},
              "roads": roads, "buildings": buildings, "route": route, "full_route": deepcopy(route),
              "terrain": terrain, "actor_target_count": 44,
              "limitations": [_SYNTHETIC_NOTE,
                              "Regional names are scenery themes, not claims about actual street layouts, building footprints, heights or road permissions.",
                              "Synthetic actors do not model real traffic or traffic laws.",
                              "150 km/h is a simulator limit, not real-world driving validation."]}
    region["geometry_sha256"] = sha256_bytes(canonical_bytes({key: region[key] for key in ("roads", "buildings", "route", "full_route", "terrain")}))
    return region


def load_course(course_id):
    """Return an independent deterministic region-shaped synthetic course."""
    return deepcopy(_cached_course(course_id))


def _profile_segment(course_or_id, progress_m):
    if not isinstance(progress_m, (float, int)) or isinstance(progress_m, bool) or not math.isfinite(progress_m):
        raise ValueError("Route progress must be a finite distance in metres")
    course = _cached_course(course_or_id) if isinstance(course_or_id, str) else course_or_id
    profile = course["terrain"]["elevation_profile"]
    distances = [sample["distance_m"] for sample in profile]
    progress = min(distances[-1], max(distances[0], float(progress_m)))
    index = min(len(profile)-2, max(0, bisect.bisect_right(distances, progress)-1))
    return profile[index], profile[index+1], progress


def elevation_at(course_or_id, progress_m):
    """Interpolate assumed elevation at route arc length; clamp at the ends."""
    a, b, progress = _profile_segment(course_or_id, progress_m)
    fraction = (progress-a["distance_m"])/(b["distance_m"]-a["distance_m"])
    return float(a["elevation_m"] + fraction*(b["elevation_m"]-a["elevation_m"]))


def grade_at(course_or_id, progress_m):
    """Return the assumed longitudinal rise/run (dimensionless, not percent)."""
    a, b, _ = _profile_segment(course_or_id, progress_m)
    return float((b["elevation_m"]-a["elevation_m"])/(b["distance_m"]-a["distance_m"]))
