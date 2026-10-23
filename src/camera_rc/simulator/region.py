"""Offline OpenStreetMap geometry for the Seoul Seongsu simulation region.

Geometry is ODbL data; software is covered by the repository licence. Widths,
missing heights, traffic, surface state and route permission are not measured.
The loader performs no network access. See assets/maps/README.md.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math

REGION_ID = "seoul_seongsu"
EARTH_RADIUS_M = 6371008.8
BBOX = {"south": 37.5415, "west": 127.0500, "north": 37.5485, "east": 127.0610}
CENTER = {"latitude": 37.545, "longitude": 127.0555}
QUERY = ('[out:json][timeout:45][maxsize:16777216];'
         '(way["highway"]["area"!="yes"](37.5415,127.0500,37.5485,127.0610);'
         'way["building"](37.5415,127.0500,37.5485,127.0610););out body geom;')
SOURCE_URL = "https://overpass-api.de/api/interpreter"
ATTRIBUTION = "© OpenStreetMap contributors"
LICENSE_URL = "https://www.openstreetmap.org/copyright"
ROUTE_WAY_IDS = (519099440, 519099441)
SOURCE_TAGS = frozenset({"highway", "name", "width", "lanes", "oneway", "building", "height", "building:levels",
                         "layer", "bridge", "tunnel", "access", "vehicle", "motor_vehicle", "motorcar",
                         "access:conditional", "motor_vehicle:conditional", "restriction", "junction",
                         "turn:lanes", "cycleway", "sidewalk"})
# Rendering/physics assumptions only. These do not establish lane permissions.
DEFAULT_HALF_WIDTHS = {"primary": 6.0, "secondary": 5.5, "tertiary": 5.0,
                      "secondary_link": 4.0, "tertiary_link": 3.5,
                      "residential": 3.5, "unclassified": 3.5, "service": 2.5,
                      "footway": 1.5, "path": 1.5, "cycleway": 1.5, "steps": 1.0}
VEHICLE_SURFACES = frozenset({"primary", "secondary", "tertiary", "secondary_link",
                              "tertiary_link", "residential", "unclassified", "service"})


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def project_latlon(latitude, longitude):
    """Local equirectangular coordinates: x east, y north, in metres."""
    if not all(math.isfinite(v) for v in (latitude, longitude)):
        raise ValueError("Coordinates must be finite")
    return [round(EARTH_RADIUS_M * math.radians(longitude - CENTER["longitude"])
                  * math.cos(math.radians(CENTER["latitude"])), 4),
            round(EARTH_RADIUS_M * math.radians(latitude - CENTER["latitude"]), 4)]


def polyline_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def slice_polyline(points, start_m, end_m):
    """Interpolate a distance interval on the source segments, without smoothing."""
    length = polyline_length(points)
    if not 0 <= start_m < end_m <= length + 1e-6:
        raise ValueError("Route interval must fit the source polyline")
    result, travelled = [], 0.0
    for a, b in zip(points, points[1:]):
        segment = math.dist(a, b)
        if segment <= 1e-9:
            continue
        lo, hi = max(start_m, travelled), min(end_m, travelled + segment)
        if hi > lo:
            for distance in (lo, hi):
                fraction = (distance - travelled) / segment
                point = [round(a[k] + fraction * (b[k] - a[k]), 4) for k in (0, 1)]
                if not result or point != result[-1]:
                    result.append(point)
        travelled += segment
    return result


def _clip_segment(a, b, bounds):
    # Liang-Barsky clipping keeps interpolated points on the actual OSM edge.
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0]-bounds[0]), (dx, bounds[2]-a[0]),
                 (-dy, a[1]-bounds[1]), (dy, bounds[3]-a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            ratio = q / p
            if p < 0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
            if low > high:
                return None
    return [[round(a[0]+t*dx, 4), round(a[1]+t*dy, 4)] for t in (low, high)]


def _clip_polyline(points, bounds):
    parts, current = [], []
    for a, b in zip(points, points[1:]):
        clipped = _clip_segment(a, b, bounds)
        if clipped is None or math.dist(*clipped) < 0.001:
            if len(current) > 1:
                parts.append(current)
            current = []
            continue
        if current and math.dist(current[-1], clipped[0]) > .001:
            parts.append(current)
            current = []
        if not current:
            current.append(clipped[0])
        current.append(clipped[1])
    if len(current) > 1:
        parts.append(current)
    return parts


def _positive_tag(value, upper):
    try:
        number = float(str(value).removesuffix(" m").strip())
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and 0 < number <= upper else None


def sanitize_source(payload, raw_response_sha256, fetched_at_utc):
    """Keep original geometry and a tag allowlist; omit people/contact/address data."""
    if payload.get("remark") or not isinstance(payload.get("elements"), list):
        raise ValueError("Incomplete or invalid Overpass response")
    elements = []
    for item in sorted(payload["elements"], key=lambda value: value["id"]):
        if item.get("type") != "way" or not item.get("geometry"):
            continue
        tags = {k: v for k, v in item.get("tags", {}).items() if k in SOURCE_TAGS}
        if "building" in tags:
            # Building names can describe occupants; only retain physical tags.
            tags = {k: v for k, v in tags.items() if k in {"building", "height", "building:levels"}}
        if not ("highway" in tags or "building" in tags):
            continue
        elements.append({"type": "way", "id": item["id"], "nodes": item.get("nodes", []),
                         "geometry": item["geometry"], "tags": tags})
    return {"schema_version": 1, "source_url": SOURCE_URL, "query": QUERY,
            "fetched_at_utc": fetched_at_utc,
            "osm_base_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
            "raw_response_sha256": raw_response_sha256,
            "license": "ODbL-1.0", "license_url": LICENSE_URL, "attribution": ATTRIBUTION,
            "filter": "Original way/node IDs and geometry; allowlisted physical road/building tags only. No user, address, contact, or building-name fields.",
            "elements": elements}


def _route(points, road_ids, name):
    return {"name": name, "road_ids": road_ids, "points": points,
            "length_m": round(polyline_length(points), 4),
            "start_heading_rad": math.atan2(points[1][1]-points[0][1], points[1][0]-points[0][0]),
            "goal_heading_rad": math.atan2(points[-1][1]-points[-2][1], points[-1][0]-points[-2][0]),
            "half_width_m": 5.0, "width_source": "simulator_assumption_by_highway_class:tertiary",
            "legal_driving_route": False, "scope": "centerline_demo_not_lane_route"}


def build_region(source):
    """Deterministically derive the bundled local map from its filtered source."""
    southwest = project_latlon(BBOX["south"], BBOX["west"])
    northeast = project_latlon(BBOX["north"], BBOX["east"])
    bounds = (*southwest, *northeast)
    roads, buildings, route_parts = [], [], {}
    for item in source["elements"]:
        tags = item["tags"]
        points = [project_latlon(p["lat"], p["lon"]) for p in item["geometry"]]
        if "highway" in tags and len(points) >= 2:
            highway = tags["highway"]
            access_supported = not any(tags.get(key, "").lower() in {"no", "private", "permit"}
                                       for key in ("access", "vehicle", "motor_vehicle", "motorcar"))
            grade_supported = (tags.get("layer", "0") in {"0", "0.0"}
                               and tags.get("bridge", "no") == "no" and tags.get("tunnel", "no") == "no")
            conditional_access = any(key in tags for key in ("access:conditional", "motor_vehicle:conditional"))
            explicit_width = _positive_tag(tags.get("width"), 60)
            half_width = explicit_width / 2 if explicit_width else DEFAULT_HALF_WIDTHS.get(highway, 2.0)
            width_source = "osm_width_tag" if explicit_width else "simulator_assumption_by_highway_class:" + highway
            for index, part in enumerate(_clip_polyline(points, bounds)):
                road_id = f"way/{item['id']}/{index}"
                roads.append({"id": road_id, "osm_way_id": item["id"], "name": tags.get("name", ""),
                              "highway": highway, "points": part, "half_width_m": half_width,
                              "width_source": width_source,
                              "drivable": highway in VEHICLE_SURFACES and access_supported and grade_supported and not conditional_access,
                              "ground_geometry_supported": grade_supported, "osm_tags": tags,
                              "oneway_tag": tags.get("oneway"), "lanes_tag": tags.get("lanes")})
                if item["id"] in ROUTE_WAY_IDS:
                    route_parts[item["id"]] = (road_id, part)
        elif "building" in tags and len(points) >= 4 and points[0] == points[-1]:
            height, height_source = _positive_tag(tags.get("height"), 500), "osm_height_tag"
            if height is None:
                levels = _positive_tag(tags.get("building:levels"), 100)
                height = levels * 3.0 if levels else 9.0
                height_source = "osm_levels_times_assumed_3m" if levels else "simulator_assumption_9m"
            buildings.append({"id": f"way/{item['id']}", "osm_way_id": item["id"], "polygon": points,
                              "height_m": height, "height_source": height_source})
    if set(route_parts) != set(ROUTE_WAY_IDS):
        raise ValueError("Expected Seongsu reference road ways are missing")
    first, second = (route_parts[way] for way in ROUTE_WAY_IDS)
    if math.dist(first[1][-1], second[1][0]) > .001:
        raise ValueError("Reference road ways are no longer connected")
    full_points = first[1] + second[1][1:]
    full_route = _route(full_points, [first[0], second[0]], "성수이로 · OSM 연결 도로")
    route = _route(slice_polyline(full_points, 40.0, 120.0), [first[0]], "성수이로 · 80m 기준 구간")
    route["source_interval_m"] = [40.0, 120.0]
    source_hash = sha256_bytes(canonical_bytes(source) + b"\n")
    region = {"schema_version": 1, "id": REGION_ID, "name": "서울 성수동", "center": CENTER, "bbox": BBOX,
              "projection": {"method": "local_equirectangular", "earth_radius_m": EARTH_RADIUS_M,
                             "x_axis": "east", "y_axis": "north", "units": "metres", "rounding_m": .0001},
              "bounds_m": dict(zip(("x_min", "y_min", "x_max", "y_max"), bounds)),
              "provenance": {key: source[key] for key in ("source_url", "query", "fetched_at_utc", "osm_base_timestamp",
                             "raw_response_sha256", "license", "license_url", "attribution")},
              "roads": roads, "buildings": buildings, "route": route, "full_route": full_route,
              "limitations": ["OSM centreline/footprint geometry, not surveyed vehicle clearance or lane geometry.",
                              "Widths without OSM width tags are simulator assumptions; lane/traffic legality is not modelled.",
                              "Building heights often assumed; footprints may be incomplete. Building relations are not included.",
                              "Non-ground ways, explicit restricted access, and conditional access are excluded from the vehicle ground union.",
                              "Turn-restriction relations, signals, lane legality, and vehicle permissions are not validated.",
                              "Vehicles, pedestrians, and obstacle placement are synthetic; no live traffic.",
                              "Reference route follows an OSM polyline; it is not a navigation recommendation."]}
    region["provenance"].update(source="OpenStreetMap via Overpass API",
                                source_asset="seoul_seongsu.source.json", source_sha256=source_hash)
    region["geometry_sha256"] = sha256_bytes(canonical_bytes({key: region[key] for key in ("roads", "buildings", "route", "full_route")}))
    return region


def validate_region(region):
    if region.get("schema_version") != 1 or region.get("id") != REGION_ID:
        raise ValueError("Unsupported region data")
    geometry = {key: region[key] for key in ("roads", "buildings", "route", "full_route")}
    if sha256_bytes(canonical_bytes(geometry)) != region.get("geometry_sha256"):
        raise ValueError("Region geometry digest mismatch")
    for road in region["roads"]:
        if len(road["points"]) < 2 or not 0 < road["half_width_m"] <= 30:
            raise ValueError("Invalid road geometry")
    for building in region["buildings"]:
        if len(building["polygon"]) < 4 or building["polygon"][0] != building["polygon"][-1]:
            raise ValueError("Invalid building ring")
    route = region["route"]
    if not 60 <= polyline_length(route["points"]) <= 100 or route["half_width_m"] < 4:
        raise ValueError("Invalid reference route")


@lru_cache(maxsize=1)
def _cached_region():
    directory = files("camera_rc.simulator").joinpath("assets", "maps")
    region = json.loads(directory.joinpath("seoul_seongsu.json").read_text(encoding="utf-8"))
    source_bytes = directory.joinpath("seoul_seongsu.source.json").read_bytes()
    if sha256_bytes(source_bytes) != region["provenance"]["source_sha256"]:
        raise ValueError("Bundled OSM source digest mismatch")
    validate_region(region)
    return region


def load_region():
    """Return a defensive copy of the validated, cached, bundled OSM region."""
    return deepcopy(_cached_region())
