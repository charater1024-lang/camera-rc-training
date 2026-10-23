import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from design_high_speed_experiment import design, projected_height, stopping_distance


def test_eighty_kmh_delay_and_braking_are_distinct():
    value = stopping_distance(80, .3, 4, 10)
    assert value["reaction_distance_m"] == pytest.approx(6.6666667)
    assert value["braking_distance_m"] == pytest.approx(61.7283951)
    assert value["required_static_stopping_distance_m"] == pytest.approx(78.3950617)
    assert stopping_distance(80, .3, 3, 10)["required_static_stopping_distance_m"] > value["required_static_stopping_distance_m"]


def test_long_range_small_obstacle_can_be_subpixel():
    assert projected_height(960, 90, .15, 80)["projected_height_pixels"] == pytest.approx(.9)
    assert projected_height(640, 90, .15, 80)["projected_height_pixels"] == pytest.approx(.6)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deceleration_is_not_an_infinite_safe_distance(value):
    with pytest.raises(ValueError):
        stopping_distance(80, .3, value)


def test_full_design_does_not_claim_executed_experiments(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "configs/high_speed_80kmh_experiment.json"
    result = design(config_path, tmp_path / "new")
    assert result["travel_during_100ms_m"] == pytest.approx(2.2222222)
    assert result["scenario_cells"] == 315
    assert result["minimum_runs_before_fault_grid"] == 9450
    assert result["simulation_executed"] is False
    assert result["model_performance_measured"] is False
    assert result["deployment_approved"] is False
    rows = [json.loads(line) for line in Path(result["scenario_manifest"]).read_text().splitlines()]
    assert all(row["executed"] is False for row in rows)
    cfg = json.loads(config_path.read_text())
    sim = cfg["simulation"]
    assert sim["camera_sensor_tick_s"] / sim["fixed_delta_seconds"] == pytest.approx(2)
    assert sim["fixed_delta_seconds"] <= sim["physics_max_substep_delta_time"] * sim["physics_max_substeps"] + 1e-12
