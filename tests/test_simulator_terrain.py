"""Properties of invented regional courses, not geographic accuracy tests."""
import math

import pytest

from camera_rc.simulator.region import polyline_length
from camera_rc.simulator.terrain import COURSE_IDS, elevation_at, grade_at, list_courses, load_course


@pytest.mark.parametrize("course_id", COURSE_IDS)
def test_regional_course_is_explicitly_synthetic_and_supports_the_region_schema(course_id):
    course = load_course(course_id)
    assert course["geometry_source"] == "synthetic_regional_course"
    assert course["synthetic"] and course["terrain"]["synthetic"]
    assert not course["provenance"]["geographic_data_used"]
    assert not course["provenance"]["measured_elevation_used"]
    assert "합성" in course["name"]
    assert course["route"]["scope"] == "synthetic_research_course"
    assert not course["route"]["legal_driving_route"]
    roads = {road["id"]: road for road in course["roads"]}
    assert set(course["route"]["road_ids"]) <= roads.keys()
    assert course["route"]["half_width_m"] >= 9
    assert course["route"]["length_m"] == pytest.approx(polyline_length(course["route"]["points"]), abs=.0001)
    assert course["route"]["length_m"] >= 2000
    for road in course["roads"]:
        assert road["drivable"] and road["ground_geometry_supported"]
        assert road["width_source"] == "repository_authored_simulator_assumption"
        for x, y in road["points"]:
            assert math.isfinite(x) and math.isfinite(y)
            assert course["bounds_m"]["x_min"] <= x <= course["bounds_m"]["x_max"]
            assert course["bounds_m"]["y_min"] <= y <= course["bounds_m"]["y_max"]
    assert all(building["polygon"][0] == building["polygon"][-1] for building in course["buildings"])
    assert course["actor_target_count"] == 44
    assert len([road for road in course["roads"] if "background" in road["id"]]) == 4


def test_gwangju_has_a_long_level_straight_for_synthetic_speed_experiments():
    course = load_course("gwangju")
    points = course["route"]["points"]
    assert course["terrain"]["straight_length_m"] >= 2000
    assert all(y == 0. for x, y in points if x <= 2400)
    assert grade_at(course, 0.) == grade_at(course, 2300.) == 0.
    assert course["roads"][0]["points"][0][0] < points[0][0]-100.
    assert course["roads"][0]["points"][-1][0] > points[-1][0]+100.


def test_mountain_course_has_bends_uphill_and_downhill_assumptions():
    course = load_course("gangwon")
    points = course["route"]["points"]
    assert max(y for _, y in points)-min(y for _, y in points) > 250.
    profile = course["terrain"]["elevation_profile"]
    elevations = [sample["elevation_m"] for sample in profile]
    assert max(elevations)-min(elevations) > 100.
    segment_grades = [grade_at(course, (a["distance_m"]+b["distance_m"])/2) for a, b in zip(profile, profile[1:])]
    assert min(segment_grades) < 0. < max(segment_grades)
    assert max(abs(grade) for grade in segment_grades) < .12
    assert course["terrain"]["elevation_source"] == "repository_authored_assumed_profile_not_DEM"


@pytest.mark.parametrize("course_id", COURSE_IDS)
def test_profile_interpolation_uses_route_arc_metres_and_clamps_endpoints(course_id):
    course = load_course(course_id)
    profile = course["terrain"]["elevation_profile"]
    assert profile[0]["distance_m"] == 0.
    assert profile[-1]["distance_m"] == course["route"]["length_m"]
    assert all(a["distance_m"] < b["distance_m"] for a, b in zip(profile, profile[1:]))
    assert elevation_at(course, -10.) == profile[0]["elevation_m"]
    assert elevation_at(course_id, 1e8) == profile[-1]["elevation_m"]
    for a, b in zip(profile, profile[1:]):
        middle = (a["distance_m"]+b["distance_m"])/2
        assert elevation_at(course, middle) == pytest.approx((a["elevation_m"]+b["elevation_m"])/2)
    for invalid in (float("nan"), float("inf"), "10", True):
        with pytest.raises(ValueError, match="finite"):
            elevation_at(course, invalid)


def test_regional_catalog_and_courses_are_defensive_and_unknown_ids_fail():
    assert [course["id"] for course in list_courses()] == list(COURSE_IDS)
    course = load_course("busan")
    original_hash = course["geometry_sha256"]
    course["roads"][0]["points"][0][0] = -999999.
    course["terrain"]["elevation_profile"][0]["elevation_m"] = -999999.
    assert load_course("busan")["geometry_sha256"] == original_hash
    assert load_course("busan")["roads"][0]["points"][0][0] == -180.
    assert elevation_at("busan", 0.) == 8.
    metadata = list_courses()
    metadata[0]["name"] = "changed"
    assert "합성" in list_courses()[0]["name"]
    with pytest.raises(ValueError, match="Unknown regional course"):
        load_course("not_a_region")
