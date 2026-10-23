import copy
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("audit_high_speed_trace", PROJECT / "scripts/audit_high_speed_trace.py")
audit = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(audit)
ROLES = ("front_main", "front_wide", "front_near", "left_forward", "right_forward",
         "left_rearward", "right_rearward", "rear")


def settings():
    return {"schema_version": 1, "camera_roles": list(ROLES), "verified_speed_bin_kmh": [75, 85],
            "provisional_latency_targets": {"front_capture_hz": 30, "exposure_to_command_p999_ms": 100,
                "maximum_exposure_to_command_ms": 150, "latest_frame_max_skew_ms": 10,
                "maximum_deadline_miss_rate": 0, "speed_alignment_tolerance_ms": 50,
                "clock_uncertainty_max_ms": 1, "minimum_samples": 3, "minimum_duration_s": 2 / 30,
                "minimum_effective_capture_hz": 29, "minimum_brake_events": 1,
                "minimum_high_speed_brake_events": 1, "maximum_exposure_to_brake_onset_ms": 300}}


def metadata(kind="hardware_exposure_to_command", diagnostic=True):
    return {"schema_version": 1, "measurement_kind": kind, "clock_domain": "fixture_monotonic_seconds",
            "clock_sync_verified": True, "clock_uncertainty_ms": .5,
            "speed_source": "simulator_ground_truth" if kind == "simulated_exposure_to_command" else "measured_vehicle_telemetry",
            "hardware_id": "synthetic_fixture_no_hardware", "diagnostic": diagnostic}


def row(index, *, start=10, speed=80 / 3.6, latency=.08, unit="drive_1", brake=False):
    exposure = start + index / 30
    result = {"unit_id": unit, "frame_id": index,
              "camera_times_s": {role: exposure for role in ROLES},
              "received_s": exposure + .005, "inference_start_s": exposure + .01,
              "inference_end_s": exposure + .04, "decision_s": exposure + .05,
              "command_s": exposure + latency, "speed_mps": speed,
              "speed_timestamp_s": exposure}
    if brake:
        result["brake_onset_s"] = result["command_s"] + .12
    return result


def execute(tmp_path, rows=None, meta=None, spec=None):
    rows = rows if rows is not None else [row(i, brake=i == 2) for i in range(3)]
    trace, meta_path, spec_path = (tmp_path / name for name in ("trace.jsonl", "metadata.json", "spec.json"))
    trace.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")
    meta_path.write_text(json.dumps(meta if meta is not None else metadata()), encoding="utf-8")
    spec_path.write_text(json.dumps(spec if spec is not None else settings()), encoding="utf-8")
    return audit.audit_trace(trace, meta_path, spec_path, tmp_path / "result.json")


def test_80kmh_and_100ms_travelled_distance_and_percentiles(tmp_path):
    result = execute(tmp_path, [row(i, latency=.1, brake=i == 2) for i in range(3)])
    assert result["all_samples"]["latency"]["p999_ms"] == pytest.approx(100)
    assert result["all_samples"]["distance_travelled_before_command"]["max_m"] == pytest.approx(2.2222222222)
    assert result["all_samples"]["deadline_misses"] == 0
    assert result["sufficient_high_speed_measurements"]
    assert result["brake_response"]["command_to_brake"]["p50_ms"] == pytest.approx(120)
    assert result["brake_response"]["oldest_camera_to_brake"]["p50_ms"] == pytest.approx(220)
    assert result["diagnostic"] and result["decision"] == "blocked"
    assert not result["deployment_approved"] and not result["physical_validation_completed"]
    assert not result["inference_accuracy_evaluated"]
    assert len(result["input_hashes"]["trace_sha256"]) == 64


def test_camera_skew_failure_is_counted_instead_of_discarded(tmp_path):
    rows = [row(i) for i in range(3)]
    for value in rows:
        value["camera_times_s"]["rear"] -= .025
    result = execute(tmp_path, rows)
    assert result["all_samples"]["samples"] == result["speed_bin_samples"]["samples"] == 3
    assert result["all_samples"]["violation_counts"]["camera_skew"] == 3
    assert result["all_samples"]["latency"]["max_ms"] == pytest.approx(105)
    assert result["all_samples"]["newest_camera_to_command"]["max_ms"] == pytest.approx(80)
    assert result["all_samples"]["deadline_miss_fraction"] == 1


def test_slow_high_speed_frames_cannot_hide_in_overall_low_speed_population(tmp_path):
    rows = [row(i, speed=10) for i in range(10)]
    rows.append(row(10, latency=.16))
    result = execute(tmp_path, rows)
    assert result["speed_bin_samples"]["samples"] == 1
    assert result["speed_bin_samples"]["latency"]["p50_ms"] == pytest.approx(160)
    assert result["speed_bin_samples"]["deadline_miss_fraction"] == 1
    assert not result["speed_bin_samples"]["latency_targets_met"]


def test_no_brake_response_blocks_even_other_provisional_targets_met(tmp_path):
    result = execute(tmp_path, [row(i) for i in range(3)], meta=metadata(diagnostic=False))
    assert result["all_samples"]["latency_targets_met"]
    assert "missing_brake_response_evidence" in result["evidence_blockers"]
    assert result["decision"] == "blocked"


def test_no_high_speed_rows_report_missing_coverage_without_nan(tmp_path):
    result = execute(tmp_path, [row(i, speed=10, brake=i == 2) for i in range(3)])
    assert result["speed_bin_samples"]["samples"] == 0
    assert result["speed_bin_samples"]["latency"]["p999_ms"] is None
    assert result["speed_bin_samples"]["deadline_miss_fraction"] is None
    assert not result["sufficient_high_speed_measurements"]
    assert "missing_high_speed_brake_response_evidence" in result["evidence_blockers"]
    assert "NaN" not in (tmp_path / "result.json").read_text()


@pytest.mark.parametrize("field,value", [("clock_sync_verified", False), ("clock_uncertainty_ms", 1.1),
                                        ("clock_uncertainty_ms", float("nan")), ("speed_source", "nominal_command"),
                                        ("diagnostic", "false"), ("hardware_id", ""), ("measurement_kind", "untyped")])
def test_unverified_clock_and_nominal_speed_metadata_rejected(tmp_path, field, value):
    meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError):
        execute(tmp_path, meta=meta)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("mode", ["future_exposure", "inference_backwards", "future_speed", "backwards_command", "duplicate_id", "reverse_id", "missing_camera", "nan_speed", "early_brake"])
def test_corrupt_timestamps_or_identity_rejected(tmp_path, mode):
    rows = [row(i) for i in range(3)]
    if mode == "future_exposure":
        rows[0]["camera_times_s"]["front_main"] += .2
    elif mode == "inference_backwards":
        rows[0]["inference_end_s"] = rows[0]["received_s"]
    elif mode == "future_speed":
        rows[0]["speed_timestamp_s"] = rows[0]["command_s"] + .1
    elif mode == "backwards_command":
        # Each row is internally ordered, but the unit's clock goes backwards.
        rows[2] = row(0)
        rows[2]["frame_id"] = 2
    elif mode == "duplicate_id":
        rows[2]["frame_id"] = rows[1]["frame_id"]
    elif mode == "reverse_id":
        rows[2]["frame_id"] = 0
    elif mode == "missing_camera":
        del rows[0]["camera_times_s"]["rear"]
    elif mode == "nan_speed":
        rows[0]["speed_mps"] = float("inf")
    else:
        rows[0]["brake_onset_s"] = rows[0]["command_s"] - .1
    with pytest.raises(ValueError):
        execute(tmp_path, rows)


def test_dropped_frame_ids_and_camera_reuse_fail_without_becoming_extra_duration(tmp_path):
    rows = [row(0), row(1), row(3)]
    rows[1]["camera_times_s"]["rear"] = rows[0]["camera_times_s"]["rear"]
    result = execute(tmp_path, rows)
    counts = result["all_samples"]["violation_counts"]
    assert counts["missing_frame_ids"] == 1
    assert counts["reused_camera_exposure_timestamp"] == 1
    assert result["units"]["drive_1"]["missing_frame_ids"] == 1
    assert result["duration"]["observed_s"] == pytest.approx(1 / 30)


def test_speed_alignment_failure_is_retained_and_cannot_satisfy_verified_coverage(tmp_path):
    rows = [row(i) for i in range(3)]
    for value in rows:
        value["speed_timestamp_s"] -= .1
    result = execute(tmp_path, rows)
    assert result["speed_bin_samples"]["samples"] == 3
    assert result["all_samples"]["violation_counts"]["speed_timestamp_misalignment"] == 3
    assert result["speed_bin_alignment_verified_samples"] == 0
    assert result["duration"]["high_speed_observed_s"] == 0


def test_cross_session_and_within_session_gaps_do_not_inflate_observed_duration(tmp_path):
    rows = [row(0), row(1), row(2, start=5000), row(0, unit="drive_2", start=10000), row(1, unit="drive_2", start=10000)]
    result = execute(tmp_path, rows)
    assert result["duration"]["observed_s"] == pytest.approx(2 / 30)
    assert result["duration"]["high_speed_observed_s"] == pytest.approx(2 / 30)
    assert result["duration"]["sum_unit_spans_s"] > 4000
    assert result["all_samples"]["violation_counts"]["capture_interval_gap"] == 1


@pytest.mark.parametrize("kind,metric", [("simulated_exposure_to_command", "exposure_to_command"),
                                        ("replay_ingest_to_command", "ingest_to_command")])
def test_simulation_and_replay_cannot_be_reported_as_physical_latency(tmp_path, kind, metric):
    result = execute(tmp_path, meta=metadata(kind, diagnostic=False))
    assert result["timing_metric"] == metric
    assert "not_a_physical_exposure_to_command_measurement" in result["evidence_blockers"]
    assert not result["physical_validation_completed"] and not result["deployment_approved"]


def test_declared_hardware_data_never_authenticates_hardware_or_approves_deployment(tmp_path):
    result = execute(tmp_path, meta=metadata(diagnostic=False))
    assert result["decision"] == "declared_trace_meets_provisional_targets_research_only"
    assert result["provenance_self_reported_not_attested"]
    assert result["research_only"] and not result["physical_validation_completed"]
    assert not result["deployment_approved"]


def test_full_default_sample_and_duration_requirements_are_preserved(tmp_path):
    spec = settings()
    spec["provisional_latency_targets"].update(minimum_samples=30000, minimum_duration_s=1800)
    result = execute(tmp_path, spec=spec)
    assert not result["sufficient_overall_measurements"] and not result["sufficient_high_speed_measurements"]
    assert result["minimum_samples"] == 30000 and result["minimum_duration_s"] == 1800


def test_declared_carla_world_frame_stride_does_not_invent_camera_drops(tmp_path):
    rows = [row(i) for i in range(3)]
    for value in rows:
        value["frame_id"] *= 2
    meta = metadata("simulated_exposure_to_command")
    meta["frame_id_stride"] = 2
    result = execute(tmp_path, rows, meta)
    assert result["units"]["drive_1"]["missing_frame_ids"] == 0
    assert "missing_frame_ids" not in result["all_samples"]["violation_counts"]


def test_output_is_not_overwritten(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("changed_field", ["unit_id", "frame_id"])
def test_renaming_a_duplicate_observation_does_not_create_evidence(tmp_path, changed_field):
    duplicate = copy.deepcopy(row(0))
    duplicate[changed_field] = "drive_2" if changed_field == "unit_id" else 1
    with pytest.raises(ValueError, match="Duplicate timing observation"):
        execute(tmp_path, [row(0), duplicate])


@pytest.mark.parametrize("second_start", [10.01, 10 + 2 / 30])
def test_distinct_units_cannot_overlap_or_touch_exposure_ranges(tmp_path, second_start):
    rows = [row(i) for i in range(3)]
    rows.extend(row(i, start=second_start, unit="drive_2", latency=.09) for i in range(3))
    with pytest.raises(ValueError, match="Exposure time ranges of distinct units overlap"):
        execute(tmp_path, rows)


def test_separate_nonoverlapping_units_add_only_their_observed_intervals(tmp_path):
    rows = [row(i, brake=i == 2) for i in range(3)]
    rows.extend(row(i, unit="drive_2", start=10000, brake=i == 2) for i in range(3))
    result = execute(tmp_path, rows)
    assert result["unit_exposure_ranges_nonoverlapping"]
    assert result["duration"]["observed_s"] == pytest.approx(4 / 30)
    assert result["capture_rate"]["all"]["elapsed_within_units_s"] == pytest.approx(4 / 30)
    assert result["capture_rate"]["all"]["effective_hz"] == pytest.approx(30)
    assert result["brake_response"]["unique_events"] == 2


def test_one_very_late_brake_response_cannot_meet_reaction_targets(tmp_path):
    rows = [row(i, brake=i == 2) for i in range(3)]
    rows[-1]["brake_onset_s"] = rows[-1]["command_s"] + 3600
    result = execute(tmp_path, rows, meta=metadata(diagnostic=False))
    assert result["brake_response"]["events_over_maximum"] == 1
    assert result["brake_response"]["oldest_camera_to_brake"]["max_ms"] > 3600000
    assert "exposure_to_brake_onset_deadline_exceeded" in result["evidence_blockers"]
    assert result["decision"] == "blocked"


def test_repeated_same_brake_onset_counts_once_with_worst_associated_latency(tmp_path):
    rows = [row(i) for i in range(3)]
    for value in rows:
        value["brake_onset_s"] = 10.25
    spec = settings()
    spec["provisional_latency_targets"].update(minimum_brake_events=2, minimum_high_speed_brake_events=2)
    result = execute(tmp_path, rows, spec=spec)
    assert result["brake_response"]["unique_events"] == result["brake_response"]["high_speed_events"] == 1
    assert result["brake_response"]["oldest_camera_to_brake"]["max_ms"] == pytest.approx(250)
    assert "insufficient_unique_brake_events" in result["evidence_blockers"]
    assert "insufficient_unique_high_speed_brake_events" in result["evidence_blockers"]


def test_production_requires_one_hundred_unique_high_speed_brake_events(tmp_path):
    spec = settings()
    spec["provisional_latency_targets"].update(minimum_brake_events=100, minimum_high_speed_brake_events=100)
    rows = [row(i, speed=10 if i == 0 else 80 / 3.6, brake=True) for i in range(100)]
    result = execute(tmp_path, rows, spec=spec)
    assert result["brake_response"]["unique_events"] == 100
    assert result["brake_response"]["high_speed_events"] == 99
    assert "insufficient_unique_brake_events" not in result["evidence_blockers"]
    assert "insufficient_unique_high_speed_brake_events" in result["evidence_blockers"]
    assert "100 events do not establish" in result["brake_response"]["percentile_limit"]


def test_small_trace_does_not_satisfy_production_brake_event_count(tmp_path):
    spec = settings()
    spec["provisional_latency_targets"].update(minimum_brake_events=100, minimum_high_speed_brake_events=100)
    result = execute(tmp_path, spec=spec)
    assert result["brake_response"]["minimum_events"] == 100
    assert result["brake_response"]["minimum_high_speed_events"] == 100
    assert "insufficient_unique_brake_events" in result["evidence_blockers"]
    assert "insufficient_unique_high_speed_brake_events" in result["evidence_blockers"]


def _spaced_row(index, hz, **kwargs):
    return row(index, start=10 + index / hz - index / 30, **kwargs)


def test_twenty_hz_cannot_pass_thirty_hz_protocol_despite_interval_tolerance(tmp_path):
    rows = [_spaced_row(i, 20, brake=i == 2) for i in range(3)]
    result = execute(tmp_path, rows, meta=metadata(diagnostic=False))
    assert "capture_interval_gap" not in result["all_samples"]["violation_counts"]
    assert result["capture_rate"]["all"]["effective_hz"] == pytest.approx(20)
    assert result["capture_rate"]["speed_bin"]["effective_hz"] == pytest.approx(20)
    assert result["units"]["drive_1"]["capture_rate"]["effective_hz"] == pytest.approx(20)
    assert "insufficient_effective_capture_frequency" in result["evidence_blockers"]


def test_slow_unit_cannot_hide_in_high_aggregate_capture_rate(tmp_path):
    rows = [_spaced_row(i, 60) for i in range(100)]
    rows.extend(row(i, start=100 + i / 20 - i / 30, unit="slow_unit", brake=i == 2) for i in range(3))
    result = execute(tmp_path, rows)
    assert result["capture_rate"]["all"]["effective_hz"] > 29
    assert result["capture_rate"]["below_target_units"] == ["slow_unit"]
    assert result["capture_rate"]["below_target_high_speed_units"] == ["slow_unit"]
    assert not result["capture_rate"]["targets_met"]


def test_slow_high_speed_subset_cannot_hide_in_fast_low_speed_frames(tmp_path):
    rows = [_spaced_row(i, 60, speed=10) for i in range(100)]
    last_time = rows[-1]["camera_times_s"]["front_main"]
    rows.extend(row(100 + i, start=last_time + (i + 1) / 20 - (100 + i) / 30,
                    brake=i == 2) for i in range(3))
    result = execute(tmp_path, rows)
    assert result["capture_rate"]["all"]["effective_hz"] > 29
    assert result["capture_rate"]["speed_bin"]["effective_hz"] == pytest.approx(20)
    assert result["capture_rate"]["below_target_units"] == []
    assert result["capture_rate"]["below_target_high_speed_units"] == ["drive_1"]
    assert "insufficient_effective_capture_frequency" in result["evidence_blockers"]


def test_normal_thirty_hz_records_report_rate_and_pixel_evidence_limit(tmp_path):
    result = execute(tmp_path)
    assert result["capture_rate"]["all"]["effective_hz"] == pytest.approx(30)
    assert result["capture_rate"]["speed_bin"]["effective_hz"] == pytest.approx(30)
    assert result["capture_rate"]["targets_met"]
    assert any("no image pixels" in limit for limit in result["limits"])
    assert any("clock resets" in limit for limit in result["limits"])
