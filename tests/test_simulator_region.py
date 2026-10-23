"""Offline provenance and geometry checks for the real OSM Seongsu extract."""
from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import math

import pytest

from camera_rc.simulator.region import (
    SOURCE_TAGS, build_region, load_region, polyline_length, project_latlon,
    sanitize_source, slice_polyline, validate_region,
)


def bundled_source():
    return files("camera_rc.simulator").joinpath("assets", "maps", "seoul_seongsu.source.json").read_bytes()


def test_bundled_map_rebuilds_exactly_from_redistributable_source():
    raw = bundled_source()
    region = load_region()
    assert hashlib.sha256(raw).hexdigest() == region["provenance"]["source_sha256"]
    assert build_region(json.loads(raw)) == region
    assert region["provenance"]["raw_response_sha256"] == "13e32047b0389f1dd6c84c729baa8fd9ba51bbcccfa086363c867755c9b829cd"
    assert region["provenance"]["license"] == "ODbL-1.0"
    assert len(region["roads"]) == 161 and len(region["buildings"]) == 554


def test_source_keeps_geometry_but_no_people_address_or_contact_fields():
    source = json.loads(bundled_source())
    for item in source["elements"]:
        assert set(item) == {"type", "id", "nodes", "geometry", "tags"}
        assert set(item["tags"]) <= SOURCE_TAGS
        if "building" in item["tags"]:
            assert set(item["tags"]) <= {"building", "height", "building:levels"}
    payload = {"elements": [{"type": "way", "id": 1, "user": "private", "uid": 12,
                            "geometry": [{"lat": 37.54, "lon": 127.05}],
                            "tags": {"building": "yes", "name": "Resident", "phone": "123", "addr:street": "private", "height": "10"}}]}
    filtered = sanitize_source(payload, "a"*64, "2026-10-23T00:00:00Z")
    assert filtered["elements"][0]["tags"] == {"building": "yes", "height": "10"}


def test_reference_route_is_an_interval_of_actual_connected_osm_roads():
    region = load_region()
    route, full = region["route"], region["full_route"]
    assert route["points"] == slice_polyline(full["points"], *route["source_interval_m"])
    assert route["road_ids"] == ["way/519099440/0"]
    assert 79.99 < polyline_length(route["points"]) < 80.01
    assert 470 < polyline_length(full["points"]) < 475
    assert route["scope"] == "centerline_demo_not_lane_route"
    assert route["legal_driving_route"] is False and route["half_width_m"] >= 4
    source = {item["id"]: item for item in json.loads(bundled_source())["elements"]}
    assert source[519099440]["nodes"][-1] == source[519099441]["nodes"][0]
    assert len(route["points"]) >= 3
    headings = [math.atan2(b[1]-a[1], b[0]-a[0]) for a, b in zip(route["points"], route["points"][1:])]
    assert max(headings) - min(headings) < .1


def test_projection_and_clipped_roads_stay_in_bounded_world():
    assert project_latlon(37.545, 127.0555) == [0., 0.]
    assert project_latlon(37.545, 127.0565)[0] == pytest.approx(88.1638, abs=.01)
    region = load_region()
    bounds = region["bounds_m"]
    for road in region["roads"]:
        for x, y in road["points"]:
            assert bounds["x_min"]-.001 <= x <= bounds["x_max"]+.001
            assert bounds["y_min"]-.001 <= y <= bounds["y_max"]+.001
        assert road["width_source"]
        if road["highway"] in {"footway", "cycleway", "steps"} or not road["ground_geometry_supported"]:
            assert road["drivable"] is False
    assert any(not road["ground_geometry_supported"] for road in region["roads"])


def test_restricted_access_and_non_ground_ways_never_become_vehicle_surface():
    source = json.loads(bundled_source())
    way = next(item for item in source["elements"] if item["id"] == 235572873)
    for key, value in (("access", "private"), ("motor_vehicle", "no"), ("bridge", "yes"), ("layer", "1"), ("access:conditional", "yes @ (Mo-Fr)")):
        changed = deepcopy(source)
        item = next(item for item in changed["elements"] if item["id"] == way["id"])
        item["tags"][key] = value
        roads = [road for road in build_region(changed)["roads"] if road["osm_way_id"] == way["id"]]
        assert roads and not any(road["drivable"] for road in roads)


def test_geometry_tampering_is_detected_and_callers_cannot_mutate_cached_data():
    changed = load_region()
    changed["roads"][0]["points"][0][0] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_region(changed)
    assert changed["roads"][0]["points"][0] != load_region()["roads"][0]["points"][0]
