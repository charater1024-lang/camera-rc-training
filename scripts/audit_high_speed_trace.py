"""Audit declared timing traces; never certify driving, perception or hardware safety.

frame_id is a per-unit capture sequence counter (default stride 1). A different
source counter stride, such as CARLA world frames, requires metadata.frame_id_stride.
camera_times_s contains only the latest image for each required camera role.
Replay timestamps describe ingestion; they cannot establish exposure latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


KINDS = {"hardware_exposure_to_command", "simulated_exposure_to_command", "replay_ingest_to_command"}
SPEED_SOURCES = {"measured_vehicle_telemetry", "simulator_ground_truth"}
ORDERED_TIMES = ("received_s", "inference_start_s", "inference_end_s", "decision_s", "command_s")


def _number(value, field, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric data")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return float(value)


def _integer(value, field, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be nonempty text without surrounding whitespace")
    return value


def _read(path):
    content = Path(path).read_bytes()
    result = json.loads(content.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result, hashlib.sha256(content).hexdigest()


def _percentiles(values, unit):
    if not values:
        return {"count": 0, **{f"{name}_{unit}": None for name in ("p50", "p95", "p99", "p999", "max")}}
    percentiles = np.percentile(np.asarray(values, np.float64), [50, 95, 99, 99.9], method="linear")
    return {"count": len(values), **{f"{name}_{unit}": float(value) for name, value in zip(("p50", "p95", "p99", "p999"), percentiles)},
            f"max_{unit}": float(max(values))}


def _settings(metadata, spec):
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1 or spec.get("schema_version") != 1:
        raise ValueError("Metadata and specification require schema_version=1")
    kind = metadata.get("measurement_kind")
    if kind not in KINDS:
        raise ValueError("Unsupported measurement_kind")
    speed_source = metadata.get("speed_source")
    if speed_source not in SPEED_SOURCES:
        raise ValueError("speed_source must be measured telemetry or simulator ground truth, never nominal")
    if kind == "hardware_exposure_to_command" and speed_source != "measured_vehicle_telemetry":
        raise ValueError("Hardware traces require measured_vehicle_telemetry")
    if kind == "simulated_exposure_to_command" and speed_source != "simulator_ground_truth":
        raise ValueError("Simulator traces require simulator_ground_truth")
    _text(metadata.get("clock_domain"), "clock_domain")
    _text(metadata.get("hardware_id"), "hardware_id")
    if metadata.get("clock_sync_verified") is not True:
        raise ValueError("clock_sync_verified must be explicitly true")
    if not isinstance(metadata.get("diagnostic"), bool):
        raise ValueError("An explicit diagnostic boolean is required")
    roles = spec.get("camera_roles")
    if not isinstance(roles, list) or len(roles) != 8 or len(set(roles)) != 8 or any(not isinstance(role, str) or not role for role in roles):
        raise ValueError("The experimental specification must declare eight distinct camera roles")
    speed_bin = spec.get("verified_speed_bin_kmh")
    if not isinstance(speed_bin, list) or len(speed_bin) != 2:
        raise ValueError("verified_speed_bin_kmh requires two bounds")
    low, high = [_number(value, "speed bin", 0) for value in speed_bin]
    if low >= high:
        raise ValueError("Speed bin bounds must increase")
    targets = spec.get("provisional_latency_targets")
    if not isinstance(targets, dict):
        raise ValueError("Missing provisional_latency_targets")
    values = {}
    for field in ("front_capture_hz", "exposure_to_command_p999_ms", "maximum_exposure_to_command_ms",
                  "latest_frame_max_skew_ms", "maximum_deadline_miss_rate", "speed_alignment_tolerance_ms",
                  "clock_uncertainty_max_ms", "minimum_duration_s", "minimum_effective_capture_hz",
                  "maximum_exposure_to_brake_onset_ms"):
        values[field] = _number(targets.get(field), field, 0)
    values["minimum_samples"] = _integer(targets.get("minimum_samples"), "minimum_samples")
    for field in ("minimum_brake_events", "minimum_high_speed_brake_events"):
        values[field] = _integer(targets.get(field), field)
    if values["front_capture_hz"] <= 0 or values["minimum_duration_s"] <= 0 or values["maximum_deadline_miss_rate"] > 1:
        raise ValueError("Invalid frequency, duration or deadline miss fraction")
    if values["maximum_exposure_to_command_ms"] < values["exposure_to_command_p999_ms"]:
        raise ValueError("Maximum latency cannot be below the percentile deadline")
    if not 0 < values["minimum_effective_capture_hz"] <= values["front_capture_hz"]:
        raise ValueError("Minimum effective capture frequency must be positive and no larger than nominal capture frequency")
    if values["maximum_exposure_to_brake_onset_ms"] <= 0:
        raise ValueError("Maximum exposure-to-brake-onset latency must be positive")
    uncertainty = _number(metadata.get("clock_uncertainty_ms"), "clock_uncertainty_ms", 0)
    if uncertainty > values["clock_uncertainty_max_ms"]:
        raise ValueError("Declared clock uncertainty exceeds the experiment tolerance")
    values.update(roles=roles, speed_bin=(low, high), frame_id_stride=_integer(metadata.get("frame_id_stride", 1), "frame_id_stride"))
    return values


def _summarize(rows, settings):
    latency = [row["latency_ms"] for row in rows]
    deadline = settings["exposure_to_command_p999_ms"]
    over = sum(value > deadline + 1e-7 for value in latency)
    violations = Counter()
    for row in rows:
        violations.update(row["violations"])
    result = {"samples": len(rows), "latency": _percentiles(latency, "ms"),
              "newest_camera_to_command": _percentiles([r["newest_latency_ms"] for r in rows], "ms"),
              "distance_travelled_before_command": _percentiles([r["travelled_m"] for r in rows], "m"),
              "camera_skew": _percentiles([r["skew_ms"] for r in rows], "ms"),
              "speed_alignment_error": _percentiles([r["speed_alignment_ms"] for r in rows], "ms"),
              "deadline_ms": deadline, "deadline_misses": over,
              "deadline_miss_fraction": over / len(rows) if rows else None,
              "violation_counts": dict(violations)}
    result["latency_targets_met"] = bool(rows) and result["latency"]["p999_ms"] <= deadline + 1e-7 and max(latency) <= settings["maximum_exposure_to_command_ms"] + 1e-7 and result["deadline_miss_fraction"] <= settings["maximum_deadline_miss_rate"]
    result["timing_consistency_targets_met"] = bool(rows) and not violations
    return result


def _capture_rate(rows, high_speed=False):
    field = "high_speed_capture_interval_s" if high_speed else "capture_interval_s"
    intervals = [row[field] for row in rows if row[field] > 0]
    elapsed = sum(intervals)
    return {"effective_hz": len(intervals) / elapsed if elapsed > 0 else None,
            "observed_intervals": len(intervals), "elapsed_within_units_s": elapsed}


def audit_trace(trace, metadata, spec, output):
    trace, metadata_path, spec_path, output = [Path(value).resolve() for value in (trace, metadata, spec, output)]
    if output.exists():
        raise FileExistsError(output)
    meta, metadata_sha = _read(metadata_path)
    experiment, spec_sha = _read(spec_path)
    settings = _settings(meta, experiment)
    rows, units, last = [], {}, {}
    digest = hashlib.sha256()
    period = 1 / settings["front_capture_hz"]
    brake_events, seen_observations = {}, set()
    with trace.open("rb") as source:
        for lineno, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            line = raw_line.decode("utf-8-sig").strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Trace row {lineno} must be an object")
            unit = _text(row.get("unit_id"), "unit_id")
            frame = _integer(row.get("frame_id"), "frame_id", 0)
            camera = row.get("camera_times_s")
            if not isinstance(camera, dict) or set(camera) != set(settings["roles"]):
                raise ValueError("camera_times_s must contain exactly the eight specified latest-frame roles")
            camera = {role: _number(camera[role], "camera timestamp", 0) for role in settings["roles"]}
            oldest, newest = min(camera.values()), max(camera.values())
            times = {field: _number(row.get(field), field, 0) for field in ORDERED_TIMES}
            ordered = [newest] + [times[field] for field in ORDERED_TIMES]
            if any(right < left for left, right in zip(ordered, ordered[1:])):
                raise ValueError("Future exposure or invalid received/inference/decision/command timestamp order")
            speed = _number(row.get("speed_mps"), "speed_mps", 0)
            speed_time = _number(row.get("speed_timestamp_s"), "speed_timestamp_s", 0)
            if speed_time > times["command_s"]:
                raise ValueError("Speed measurement timestamp is after the command")
            observation = tuple(camera[role] for role in settings["roles"]) + tuple(times[field] for field in ORDERED_TIMES)
            if observation in seen_observations:
                raise ValueError("Duplicate timing observation; changing unit_id or frame_id does not create new evidence")
            seen_observations.add(observation)
            previous = last.get(unit)
            violations = []
            camera_skew_ms = (newest - oldest) * 1000
            speed_error_ms = abs(newest - speed_time) * 1000
            if camera_skew_ms > settings["latest_frame_max_skew_ms"] + 1e-7:
                violations.append("camera_skew")
            if speed_error_ms > settings["speed_alignment_tolerance_ms"] + 1e-7:
                violations.append("speed_timestamp_misalignment")
            delta, missing_ids = 0.0, 0
            if previous is not None:
                if frame <= previous["frame_id"]:
                    raise ValueError("frame_id must be unique and strictly increasing within each unit")
                if any(times[field] < previous["times"][field] for field in ORDERED_TIMES) or speed_time < previous["speed_timestamp_s"]:
                    raise ValueError("Trace timestamps must be monotonic within each unit")
                if any(camera[role] < previous["camera_times_s"][role] for role in settings["roles"]):
                    raise ValueError("Camera timestamps must be monotonic within each unit")
                if any(camera[role] == previous["camera_times_s"][role] for role in settings["roles"]):
                    violations.append("reused_camera_exposure_timestamp")
                delta = newest - previous["newest"]
                frame_delta = frame - previous["frame_id"]
                if frame_delta % settings["frame_id_stride"]:
                    raise ValueError("frame_id increment is incompatible with declared frame_id_stride")
                missing_ids = frame_delta // settings["frame_id_stride"] - 1
                if missing_ids:
                    violations.append("missing_frame_ids")
                if delta > 1.5 * period + 1e-9:
                    violations.append("capture_interval_gap")
            latency_ms = (times["command_s"] - oldest) * 1000
            in_bin = settings["speed_bin"][0] - 1e-7 <= speed * 3.6 <= settings["speed_bin"][1] + 1e-7
            entry = {"unit_id": unit, "frame_id": frame, "newest": newest, "oldest": oldest,
                     "latency_ms": latency_ms, "newest_latency_ms": (times["command_s"] - newest) * 1000,
                     "travelled_m": speed * latency_ms / 1000, "skew_ms": camera_skew_ms,
                     "speed_alignment_ms": speed_error_ms, "speed_mps": speed, "in_speed_bin": in_bin,
                     "violations": violations, "delta_s": delta, "missing_frame_ids": missing_ids}
            unit_rows = units.setdefault(unit, [])
            # Duration credits only consecutive observed capture intervals, never
            # cross-session wall gaps, dropped IDs, or unobserved within-unit gaps.
            valid_interval = previous is not None and 0 < delta <= 1.5 * period + 1e-9 and not missing_ids
            verified_high_interval = (previous is not None and in_bin and previous["in_speed_bin"]
                                      and speed_error_ms <= settings["speed_alignment_tolerance_ms"] + 1e-7
                                      and previous["speed_alignment_ms"] <= settings["speed_alignment_tolerance_ms"] + 1e-7)
            entry["observed_interval_s"] = delta if valid_interval else 0.0
            entry["high_speed_observed_interval_s"] = delta if valid_interval and verified_high_interval else 0.0
            # Rate uses elapsed intervals INCLUDING gaps: silently removing slow
            # intervals would falsely inflate the effective capture frequency.
            entry["capture_interval_s"] = delta
            entry["high_speed_capture_interval_s"] = delta if verified_high_interval else 0.0
            if row.get("brake_onset_s") is not None:
                onset = _number(row["brake_onset_s"], "brake_onset_s", 0)
                if onset < times["command_s"]:
                    raise ValueError("brake_onset_s cannot precede its associated command")
                event = {"unit_id": unit, "brake_onset_s": onset,
                         "command_to_brake_ms": (onset - times["command_s"]) * 1000,
                         "oldest_camera_to_brake_ms": (onset - oldest) * 1000,
                         "in_speed_bin": in_bin and speed_error_ms <= settings["speed_alignment_tolerance_ms"] + 1e-7}
                key = (unit, onset)
                if key not in brake_events or event["oldest_camera_to_brake_ms"] > brake_events[key]["oldest_camera_to_brake_ms"]:
                    brake_events[key] = event
            rows.append(entry)
            unit_rows.append(entry)
            last[unit] = {**entry, "times": times, "camera_times_s": camera, "speed_timestamp_s": speed_time}
    if not rows:
        raise ValueError("Trace must contain at least one sample")
    exposure_ranges = sorted((min(row["oldest"] for row in values), max(row["newest"] for row in values), unit)
                             for unit, values in units.items())
    for previous, current in zip(exposure_ranges, exposure_ranges[1:]):
        if current[0] <= previous[1]:
            raise ValueError("Exposure time ranges of distinct units overlap in one clock_domain/hardware_id; audit clock resets separately")
    high_speed = [row for row in rows if row["in_speed_bin"]]
    all_summary, high_summary = _summarize(rows, settings), _summarize(high_speed, settings)
    unit_reports = {}
    for unit, values in units.items():
        unit_reports[unit] = {"all": _summarize(values, settings),
                              "speed_bin": _summarize([row for row in values if row["in_speed_bin"]], settings),
                              "exposure_span_s": values[-1]["newest"] - values[0]["newest"],
                              "observed_duration_s": sum(row["observed_interval_s"] for row in values),
                              "high_speed_observed_duration_s": sum(row["high_speed_observed_interval_s"] for row in values),
                              "capture_rate": _capture_rate(values),
                              "high_speed_capture_rate": _capture_rate(values, True),
                              "missing_frame_ids": sum(row["missing_frame_ids"] for row in values)}
    all_duration = sum(row["observed_interval_s"] for row in rows)
    high_duration = sum(row["high_speed_observed_interval_s"] for row in rows)
    aligned_high_count = sum(row["speed_alignment_ms"] <= settings["speed_alignment_tolerance_ms"] + 1e-7 for row in high_speed)
    enough_all = len(rows) >= settings["minimum_samples"] and all_duration >= settings["minimum_duration_s"] - 1e-7
    enough_high = aligned_high_count >= settings["minimum_samples"] and high_duration >= settings["minimum_duration_s"] - 1e-7
    overall_rate, high_speed_rate = _capture_rate(rows), _capture_rate(rows, True)
    minimum_hz = settings["minimum_effective_capture_hz"]
    rate_failures, high_rate_failures = [], []
    for unit, values in unit_reports.items():
        hz = values["capture_rate"]["effective_hz"]
        high_hz = values["high_speed_capture_rate"]["effective_hz"]
        if hz is None or hz + 1e-7 < minimum_hz:
            rate_failures.append(unit)
        if values["speed_bin"]["samples"] and (high_hz is None or high_hz + 1e-7 < minimum_hz):
            high_rate_failures.append(unit)
    adequate_rate = (overall_rate["effective_hz"] is not None and overall_rate["effective_hz"] + 1e-7 >= minimum_hz
                     and high_speed_rate["effective_hz"] is not None and high_speed_rate["effective_hz"] + 1e-7 >= minimum_hz
                     and not rate_failures and not high_rate_failures)
    high_brake_count = sum(event["in_speed_bin"] for event in brake_events.values())
    late_brake_count = sum(event["oldest_camera_to_brake_ms"] > settings["maximum_exposure_to_brake_onset_ms"] + 1e-7
                           for event in brake_events.values())
    blockers = []
    if not enough_all:
        blockers.append("insufficient_overall_measurements")
    if not enough_high:
        blockers.append("insufficient_verified_high_speed_measurements")
    if not brake_events:
        blockers.append("missing_brake_response_evidence")
    elif not high_brake_count:
        blockers.append("missing_high_speed_brake_response_evidence")
    if len(brake_events) < settings["minimum_brake_events"]:
        blockers.append("insufficient_unique_brake_events")
    if high_brake_count < settings["minimum_high_speed_brake_events"]:
        blockers.append("insufficient_unique_high_speed_brake_events")
    if late_brake_count:
        blockers.append("exposure_to_brake_onset_deadline_exceeded")
    if not adequate_rate:
        blockers.append("insufficient_effective_capture_frequency")
    if not all_summary["latency_targets_met"] or not high_summary["latency_targets_met"]:
        blockers.append("latency_targets_not_met_or_no_high_speed_samples")
    if not all_summary["timing_consistency_targets_met"]:
        blockers.append("timestamp_skew_alignment_or_capture_continuity_failure")
    if meta["diagnostic"]:
        blockers.append("diagnostic_trace_not_operational_evidence")
    if meta["measurement_kind"] != "hardware_exposure_to_command":
        blockers.append("not_a_physical_exposure_to_command_measurement")
    result = {"schema_version": 1, "complete": True, "research_only": True,
              "deployment_approved": False, "physical_validation_completed": False,
              "inference_accuracy_evaluated": False, "metadata": meta,
              "measurement_kind": meta["measurement_kind"], "diagnostic": meta["diagnostic"],
              "provenance_self_reported_not_attested": True,
              "input_hashes": {"trace_sha256": digest.hexdigest(), "metadata_sha256": metadata_sha, "spec_sha256": spec_sha},
              "inputs": {"trace": str(trace), "metadata": str(metadata_path), "spec": str(spec_path)},
              "timing_metric": "ingest_to_command" if meta["measurement_kind"] == "replay_ingest_to_command" else "exposure_to_command",
              "latency_start": "Oldest of the eight latest-camera timestamps; historical model frames are not included.",
              "percentile_method": "NumPy linear empirical interpolation; correlated frames are not independent safety trials.",
              "all_samples": all_summary, "speed_bin_samples": high_summary,
              "verified_speed_bin_kmh": list(settings["speed_bin"]),
              "speed_bin_alignment_verified_samples": aligned_high_count,
              "units": unit_reports,
              "unit_exposure_ranges_nonoverlapping": True,
              "capture_rate": {"all": overall_rate, "speed_bin": high_speed_rate,
                               "minimum_effective_capture_hz": minimum_hz, "targets_met": adequate_rate,
                               "below_target_units": rate_failures, "below_target_high_speed_units": high_rate_failures,
                               "policy": "Positive observed interval count divided by elapsed time within units, including gaps. No elapsed gap between units is included. Each unit and both aggregate sets must meet the minimum; high-speed rates use consecutive aligned in-bin endpoints."},
              "duration": {"observed_s": all_duration, "high_speed_observed_s": high_duration,
                           "sum_unit_spans_s": sum(unit["exposure_span_s"] for unit in unit_reports.values()),
                           "policy": "Sum observed consecutive intervals within each unit. Exclude cross-unit gaps, missing frame IDs and intervals over 1.5/fps; high-speed intervals additionally require both endpoint speeds aligned and in bin."},
              "sufficient_overall_measurements": enough_all, "sufficient_high_speed_measurements": enough_high,
              "minimum_samples": settings["minimum_samples"], "minimum_duration_s": settings["minimum_duration_s"],
              "brake_response": {"unique_events": len(brake_events),
                                 "high_speed_events": high_brake_count,
                                 "minimum_events": settings["minimum_brake_events"],
                                 "minimum_high_speed_events": settings["minimum_high_speed_brake_events"],
                                 "maximum_exposure_to_brake_onset_ms": settings["maximum_exposure_to_brake_onset_ms"],
                                 "events_over_maximum": late_brake_count,
                                 "command_to_brake": _percentiles([event["command_to_brake_ms"] for event in brake_events.values()], "ms"),
                                 "oldest_camera_to_brake": _percentiles([event["oldest_camera_to_brake_ms"] for event in brake_events.values()], "ms"),
                                 "percentile_limit": "Empirical percentiles only. Even 100 events do not establish a reliable p99.9 tail bound or a safety failure probability.",
                                 "association_limit": "Onset-to-command association is declared by the trace producer, not independently verified by this audit."},
              "evidence_blockers": blockers,
              "decision": "blocked" if blockers else "declared_trace_meets_provisional_targets_research_only",
              "limits": ["Metadata and timestamps cannot authenticate real hardware or rule out fabricated traces.",
                         "Simulator clock timing does not measure host wall-clock inference throughput or real brake response.",
                         "Replay ingest timing cannot establish photon/exposure-to-actuation latency.",
                         "Measured speed multiplied by delay is a constant-speed distance estimate, not integrated ground-truth travel or stopping distance.",
                         "No object detection, trajectory accuracy, collision avoidance or roadworthiness conclusion is made.",
                         "Timestamp checks cannot detect black or frozen images delivered with fresh timestamps; no image pixels are present in this trace.",
                         "One audit assumes one continuous clock domain for one hardware_id. Separate audits are required after clock resets; overlapping units cannot be pooled.",
                         "Clock uncertainty is checked against the declared tolerance; reported latency percentiles are point estimates without an uncertainty margin."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_trace(args.trace, args.metadata, args.spec, args.output)
    print(json.dumps({key: result[key] for key in ("decision", "measurement_kind", "deployment_approved", "evidence_blockers")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
