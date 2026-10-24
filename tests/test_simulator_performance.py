"""Engine integration of session metrics; no learned accuracy or driving claim."""
import json
import math

import numpy as np
import pytest

from camera_rc.driving.types import Perception
from camera_rc.simulator import Engine, SCENARIOS
from camera_rc.simulator.engine import _polyline_position
from camera_rc.simulator.server import SimulatorApp


def _performance(engine):
    return engine.snapshot()["performance"]


def _drive_steps(engine, count, controls=None):
    for _ in range(count):
        engine.step(controls)
    return engine.snapshot()


def test_initial_engine_has_zero_counters_and_null_unsampled_metrics():
    performance = _performance(Engine())
    autonomous = performance["autonomous"]
    for key in ("samples", "duration_s", "distance_m", "safety_stop_s", "safety_interventions",
                "collision_events", "offroad_events", "stopped_time_s"):
        assert autonomous[key] == 0
    for key in ("cross_track_rmse_m", "cross_track_max_m", "heading_mae_deg", "speed_rmse_kmh",
                "speed_mae_kmh", "mean_speed_kmh", "peak_speed_kmh", "safety_eligible_pct"):
        assert autonomous[key] is None
    assert performance["history"] == []
    assert performance["current"]["applied_target_kmh"] is None
    assert performance["scope"] == "synthetic_session_only"
    assert performance["session"] == {"total_samples": 0, "manual_samples": 0,
                                      "mixed_mode_session": False, "modes_seen": []}


def test_default_reference_is_finite_zero_to_36_not_the_render_road():
    engine = Engine()
    route = _performance(engine)["route"]
    assert route["length_m"] == 36
    assert route["progress_m"] == route["progress_pct"] == 0
    engine._car.x_m = 18
    route = _performance(engine)["route"]
    assert route["progress_m"] == 18
    assert route["progress_pct"] == 50
    engine._car.x_m = 40
    route = _performance(engine)["route"]
    assert route["progress_m"] == 36 and route["progress_pct"] == 100
    assert route["current_cross_track_error_m"] == 4
    engine._car.x_m = -3
    route = _performance(engine)["route"]
    assert route["progress_m"] == 0
    assert route["current_cross_track_error_m"] == 3
    assert _performance(engine)["autonomous"]["samples"] == 0


def test_curved_reference_ends_at_goal_and_uses_clipped_arc_length():
    engine = Engine("curved")
    xs = np.arange(0., 38., 2.)
    ys = 3.5 * (1 - np.cos(xs / 14.))
    reference = np.column_stack((xs, ys))
    expected_length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    route = _performance(engine)["route"]
    assert route["length_m"] == pytest.approx(expected_length)
    engine._car.x_m = 36
    engine._car.y_m = float(ys[-1])
    route = _performance(engine)["route"]
    assert route["progress_pct"] == pytest.approx(100)
    assert route["remaining_m"] == pytest.approx(0)
    assert route["current_cross_track_error_m"] == pytest.approx(0)


@pytest.mark.parametrize("scenario", ["seoul_day", "gangwon_day"])
def test_regional_progress_uses_route_arc_and_updates_during_manual_session(scenario):
    engine = Engine(scenario, mode="manual")
    total = engine.snapshot()["world"]["route_length_m"]
    position, direction = _polyline_position(engine._road, total * .25)
    engine._car.x_m, engine._car.y_m = map(float, position)
    engine._car.yaw_rad = math.atan2(direction[1], direction[0])
    performance = _performance(engine)
    assert performance["route"]["length_m"] == pytest.approx(total)
    assert performance["route"]["progress_pct"] == pytest.approx(25)
    assert performance["route"]["current_cross_track_error_m"] == pytest.approx(0, abs=1e-9)
    assert performance["autonomous"]["samples"] == 0


def test_only_completed_steps_sample_not_queries_pause_or_settings():
    engine = Engine()
    app = SimulatorApp(engine, SCENARIOS)
    assert engine.step()["performance"]["autonomous"]["samples"] == 1
    before = _performance(engine)
    for _ in range(3):
        assert _performance(engine) == before
        assert engine.report()["snapshot"]["performance"] == before
        engine.frame("camera")
        engine.frame("segmentation")
        app.snapshot()
        app.action({"action": "pause"})
        app.tick()
    engine.set_speed("user", 7.2)
    engine.set_mode("manual")
    engine.set_mode("autonomous")
    assert _performance(engine) == before
    after = engine.step()["performance"]["autonomous"]
    assert after["samples"] == 2
    assert after["duration_s"] == pytest.approx(2 * engine.DT_S)
    assert _performance(engine)["session"]["mixed_mode_session"] is False
    with pytest.raises(ValueError):
        engine.step({"throttle": math.nan})
    assert _performance(engine)["autonomous"]["samples"] == 2


def test_tracking_speed_reference_uses_ramped_command_not_plan_ceiling():
    engine = Engine()
    first = engine.step()
    assert first["performance"]["current"]["applied_target_kmh"] == 0
    snapshot = engine.step()
    assert snapshot["command"]["reason"] == "tracking"
    applied_target = engine._command.target_speed_mps
    assert 0 < applied_target < engine._effective_target_mps
    performance = snapshot["performance"]
    assert performance["current"]["applied_target_kmh"] == pytest.approx(applied_target * 3.6)
    error_kmh = abs(snapshot["vehicle"]["speed_mps"] - applied_target) * 3.6
    assert performance["autonomous"]["speed_mae_kmh"] == pytest.approx(error_kmh / 2)
    assert performance["autonomous"]["speed_rmse_kmh"] == pytest.approx(error_kmh / math.sqrt(2))
    assert performance["autonomous"]["distance_m"] == pytest.approx(snapshot["metrics"]["distance_m"])
    assert performance["autonomous"]["mean_speed_kmh"] == pytest.approx(
        snapshot["metrics"]["distance_m"] / snapshot["sim_time_s"] * 3.6)


@pytest.mark.parametrize("kind", ["warmup", "veto", "requested_stop", "goal"])
def test_stop_commands_have_zero_applied_speed_reference(kind):
    if kind == "veto":
        engine = Engine(perception=lambda rgb: Perception(np.zeros(rgb.shape[:2], dtype=np.float32)))
    elif kind == "requested_stop":
        engine = Engine(speed_mode="user", target_speed_kmh=0)
    else:
        engine = Engine()
    if kind == "goal":
        engine._car.x_m = 36
    snapshot = engine.step()
    assert snapshot["command"]["brake"] == 1
    assert snapshot["performance"]["current"]["applied_target_kmh"] == 0
    autonomous = snapshot["performance"]["autonomous"]
    assert autonomous["samples"] == 1
    if kind == "warmup":
        assert snapshot["safety"]["mode"] == "WARMUP_STOP"
        assert autonomous["safety_eligible_pct"] == 0
        assert autonomous["safety_stop_s"] == 0
        assert autonomous["safety_interventions"] == 0
    elif kind == "veto":
        assert autonomous["safety_stop_s"] == pytest.approx(engine.DT_S)
        assert autonomous["safety_interventions"] == 1


def test_finished_steps_and_reports_do_not_duplicate_terminal_sample():
    engine = Engine()
    engine._car.x_m = 36
    terminal = engine.step()
    assert terminal["finished"] and terminal["metrics"]["goal_reached"]
    performance = terminal["performance"]
    assert performance["autonomous"]["samples"] == 1
    for _ in range(4):
        assert engine.step()["performance"] == performance
        assert engine.report()["snapshot"]["performance"] == performance
    assert engine.snapshot()["step"] == 1


def _obstacle(identity, x, y):
    return {"id": identity, "x_m": x, "y_m": y, "radius_m": .1,
            "kind": "obstacle", "vx_mps": 0., "vy_mps": 0.}


def test_manual_events_stay_out_of_autonomous_totals_across_mode_changes():
    engine = Engine(mode="manual")
    engine._car.x_m, engine._car.y_m = 1., 8.
    engine._obstacles = lambda at_time=None: [_obstacle("manual_contact", 1., 8.)]
    manual = engine.step()
    assert manual["metrics"]["collisions"] == manual["metrics"]["offroad_events"] == 1
    assert manual["metrics"]["interventions"] == 1
    stats = manual["performance"]["autonomous"]
    assert stats["samples"] == stats["collision_events"] == stats["offroad_events"] == 0
    assert stats["safety_interventions"] == 0
    engine._car.x_m, engine._car.y_m = 0., 0.
    engine.set_mode("autonomous")
    returned = engine.step()
    stats = returned["performance"]["autonomous"]
    assert stats["samples"] == 1
    assert stats["collision_events"] == stats["offroad_events"] == stats["safety_interventions"] == 0
    engine._car.x_m, engine._car.y_m = 2., 8.
    engine._obstacles = lambda at_time=None: [_obstacle("autonomous_contact", 2., 8.)]
    autonomous = engine.step()
    stats = autonomous["performance"]["autonomous"]
    assert stats["samples"] == 2
    assert stats["duration_s"] == pytest.approx(2 * engine.DT_S)
    assert stats["collision_events"] == stats["offroad_events"] == 1
    assert autonomous["metrics"]["collisions"] == autonomous["metrics"]["offroad_events"] == 2
    session = autonomous["performance"]["session"]
    assert session["total_samples"] == 3 and session["manual_samples"] == 1
    assert session["mixed_mode_session"] is True
    assert set(session["modes_seen"]) == {"manual", "autonomous"}
    cleared = engine.reset()["performance"]
    assert cleared["autonomous"]["samples"] == 0
    assert cleared["autonomous"]["collision_events"] == cleared["autonomous"]["offroad_events"] == 0
    assert cleared["autonomous"]["safety_interventions"] == 0
    assert cleared["session"]["manual_samples"] == 0
    assert cleared["session"]["mixed_mode_session"] is False


def test_short_manual_interval_is_disclosed_without_contributing_autonomous_sample():
    engine = Engine()
    engine.step()
    before = _performance(engine)["autonomous"]
    engine.set_mode("manual")
    manual = engine.step()["performance"]
    assert manual["autonomous"] == before
    engine.set_mode("autonomous")
    performance = engine.step()["performance"]
    assert performance["autonomous"]["samples"] == 2
    assert performance["session"]["total_samples"] == 3
    assert performance["session"]["manual_samples"] == 1
    assert performance["session"]["mixed_mode_session"] is True
    assert [point["mode"] for point in performance["history"]] == ["autonomous", "manual", "autonomous"]
    assert [point["time_s"] for point in performance["history"]] == pytest.approx([.1, .2, .3])


def test_reset_clears_tracker_and_preserves_previous_performance_summary(tmp_path):
    engine = Engine(output_dir=tmp_path / "sessions")
    _drive_steps(engine, 3)
    engine.set_mode("manual")
    previous = engine.step()["performance"]
    assert previous["session"]["mixed_mode_session"] is True
    assert previous["session"]["manual_samples"] == 1
    first_session = tmp_path / "sessions" / "session_0000"
    steps_before = (first_session / "steps.jsonl").read_bytes()
    reset = engine.reset(scenario="fog", seed=7)
    summary = json.loads((first_session / "summary.json").read_text(encoding="utf-8"))
    assert summary["end_reason"] == "reset"
    assert summary["snapshot"]["performance"] == previous
    assert (first_session / "steps.jsonl").read_bytes() == steps_before
    initial = reset["performance"]
    assert initial["autonomous"]["samples"] == initial["autonomous"]["duration_s"] == 0
    assert initial["autonomous"]["distance_m"] == 0
    assert initial["autonomous"]["speed_rmse_kmh"] is None
    assert initial["autonomous"]["collision_events"] == initial["autonomous"]["offroad_events"] == 0
    assert initial["history"] == []
    assert initial["current"]["applied_target_kmh"] is None
    assert initial["route"]["progress_m"] == 0
    assert initial["session"] == {"total_samples": 0, "manual_samples": 0,
                                  "mixed_mode_session": False, "modes_seen": []}
    engine.step()
    engine.close()
    assert summary["snapshot"]["performance"]["autonomous"]["samples"] == 3
    assert json.loads((first_session / "summary.json").read_text(encoding="utf-8")) == summary


def test_json_report_and_saved_step_expose_the_same_performance(tmp_path):
    engine = Engine(output_dir=tmp_path / "report")
    snapshot = _drive_steps(engine, 6)
    report = json.loads(json.dumps(engine.report(), allow_nan=False))
    assert report["snapshot"]["performance"] == snapshot["performance"]
    assert report["configuration"]["performance_metrics"]["autonomous_samples_only"] is True
    assert report["accuracy_measured"] is False
    records = [json.loads(line) for line in (tmp_path / "report" / "session_0000" / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert records[-1]["performance"] == snapshot["performance"]
    assert [record["performance"]["autonomous"]["samples"] for record in records] == list(range(1, 7))
    engine.close()
