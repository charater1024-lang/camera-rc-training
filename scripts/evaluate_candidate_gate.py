"""Compare paired original-drive results without granting deployment approval.

CLI: python scripts/evaluate_candidate_gate.py --baseline old.json
     --candidate new.json --spec gate.json --output NEW_RESULT.json

Report schema (both reports):
  {"schema_version":1,"evaluation_id":"run-1","protocol_sha256":"<64hex>",
   "test_split_sha256":"<64hex>","hardware_id":"same-target-gpu",
   "evidence_domain":"real_recorded","diagnostic":false,
   "measurement_kind":"hardware_exposure_to_command",
   "unit_type":"original_drive",
   "units":{"drive-1":{"dataset_id":"dataset","original_drive_id":"drive-1",
                        "scenario_ids":["night_wet"]}},
   "metrics":{"road_iou":{"direction":"higher","unit":"fraction",
              "protocol_sha256":"<64hex denominator/aggregation/settings>",
              "units":{"drive-1":{"value":0.8,"denominator":12345}}}}}

Spec schema:
  {"schema_version":1,"required_metrics":{"road_iou":{"direction":"higher",
   "unit":"fraction","min_units":30,"min_improvement":0,"mode":"improve",
   "absolute_min":0.7,"require_no_unit_regression":false}},
   "coverage_required":{"night_wet":30},"critical_failure_metrics":[],
   "bootstrap_repetitions":10000,"seed":42,"require_no_unit_regression":true,
   "required_evidence_domain":"real_recorded",
   "required_measurement_kind":"hardware_exposure_to_command"}

required_evidence_domain and required_measurement_kind are optional. When set,
both reports must contain the exact requested top-level value. Older specs
without these requirements still accept reports without measurement_kind.
If either report supplies measurement_kind, both must supply the same valid
kind. A scope description alone does not restrict admissible evidence.

An independent unit is one original drive, never a pixel/frame/weather clone.
original_drive_id must be globally canonical across source datasets and their
derivatives (for example kitti:2011_09_26_drive_0009, also for its virtual clone).
Every required metric must have matching units, denominators and protocols.
Each required scenario needs its configured drive count within every metric.
The mean is unweighted over drives. Positive paired deltas always mean better.
Internally computed paired bootstrap uses a one-sided Bonferroni-adjusted
lower percentile at 0.05 / number_of_improvement_metrics. It must be strictly
ABOVE min_improvement for mode=improve; mode=maintain uses >= instead. CI95
(2.5%,97.5%) is additionally reported. Absolute bounds apply to EVERY candidate drive, not a pooled
mean. Critical failure metrics must be nonnegative integer event counts with
direction=lower, and every observed candidate count must be zero.

evidence_domain must be real_recorded, simulated or synthetic_fixture. Non-real
or diagnostic evidence can only yield diagnostic_checks_passed, not a qualified
research candidate. real_performance_validated remains false for every outcome:
this audit compares declared evidence, not independently authenticated performance.
Supplied aggregate values or confidence intervals are never used. This checks
the submitted evidence; it cannot certify its authenticity, drive independence,
scenario annotations or freedom from prior test-set tuning. A pass is only a
research candidate, never a safety certification or a zero-accident guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


HEX = re.compile(r"[a-f0-9]{64}\Z")
EVIDENCE_DOMAINS = {"real_recorded", "simulated", "synthetic_fixture"}
MEASUREMENT_KINDS = {"hardware_exposure_to_command", "simulated_exposure_to_command", "replay_ingest_to_command"}


def text(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty text without surrounding whitespace")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def integer(value, name, minimum=1, maximum=100000):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def checksum(value, name):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA256 hex")
    return value


def validate_spec(spec):
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("Gate spec requires schema_version=1")
    for key, choices in (("required_evidence_domain", EVIDENCE_DOMAINS),
                         ("required_measurement_kind", MEASUREMENT_KINDS)):
        if key in spec and (not isinstance(spec[key], str) or spec[key] not in choices):
            raise ValueError(key + " must be one of: " + ", ".join(sorted(choices)))
    required = spec.get("required_metrics")
    if not isinstance(required, dict) or not required:
        raise ValueError("At least one required metric must be configured")
    for name, item in required.items():
        text(name, "metric name")
        if not isinstance(item, dict) or item.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"{name} direction must be higher or lower")
        text(item.get("unit"), name + " unit")
        integer(item.get("min_units", 30), name + " min_units", minimum=2)
        if finite(item.get("min_improvement", 0), name + " min_improvement") < 0:
            raise ValueError("min_improvement must be nonnegative")
        if item.get("mode", "improve") not in {"improve", "maintain"}:
            raise ValueError("Metric mode must be improve or maintain")
        if not isinstance(item.get("require_no_unit_regression", False), bool):
            raise ValueError("require_no_unit_regression must be boolean")
        for bound in ("absolute_min", "absolute_max"):
            if bound in item:
                finite(item[bound], name + " " + bound)
        if "absolute_min" in item and "absolute_max" in item and item["absolute_min"] > item["absolute_max"]:
            raise ValueError("absolute_min cannot exceed absolute_max")
    # A configuration maintaining everything is not evidence of improvement.
    if not any(item.get("mode", "improve") == "improve" for item in required.values()):
        raise ValueError("At least one metric must require improvement")
    coverage = spec.get("coverage_required", {})
    if isinstance(coverage, list):
        if len(set(coverage)) != len(coverage):
            raise ValueError("Duplicate coverage scenarios")
        coverage = {name: 1 for name in coverage}
    if not isinstance(coverage, dict):
        raise ValueError("coverage_required must map scenario IDs to independent-drive counts")
    for scenario, count in coverage.items():
        text(scenario, "scenario ID")
        integer(count, scenario + " coverage count")
    critical = spec.get("critical_failure_metrics", [])
    if not isinstance(critical, list) or len(set(critical)) != len(critical):
        raise ValueError("critical_failure_metrics must be a unique list")
    if any(name not in required or required[name]["direction"] != "lower" for name in critical):
        raise ValueError("Critical failure metrics must be required lower-is-better counts")
    if not isinstance(spec.get("require_no_unit_regression", False), bool):
        raise ValueError("Global require_no_unit_regression must be boolean")
    integer(spec.get("bootstrap_repetitions", 10000), "bootstrap_repetitions", minimum=1000, maximum=100000)
    integer(spec.get("seed", 42), "seed", minimum=0, maximum=2**32-1)
    return coverage


def validate_report(report):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Evaluation report requires schema_version=1")
    for key in ("evaluation_id", "hardware_id"):
        text(report.get(key), key)
    for key in ("protocol_sha256", "test_split_sha256"):
        checksum(report.get(key), key)
    if report.get("unit_type") != "original_drive":
        raise ValueError("Only original-drive independent units are accepted")
    if report.get("evidence_domain") not in EVIDENCE_DOMAINS:
        raise ValueError("evidence_domain must be real_recorded, simulated or synthetic_fixture")
    if "measurement_kind" in report and (not isinstance(report["measurement_kind"], str) or report["measurement_kind"] not in MEASUREMENT_KINDS):
        raise ValueError("measurement_kind must be one of: " + ", ".join(sorted(MEASUREMENT_KINDS)))
    if not isinstance(report.get("diagnostic"), bool):
        raise ValueError("An explicit diagnostic boolean is required")
    units = report.get("units")
    if not isinstance(units, dict) or not units or len(units) > 100000:
        raise ValueError("Report must contain 1..100000 original-drive units")
    origins = set()
    normalized = {}
    for name, unit in units.items():
        text(name, "unit ID")
        if not isinstance(unit, dict):
            raise ValueError("Original-drive metadata must be an object")
        origin = (text(unit.get("dataset_id"), "dataset ID"), text(unit.get("original_drive_id"), "original drive ID"))
        if origin[1] in origins:
            raise ValueError("Repeated original drive masquerades as independent units")
        origins.add(origin[1])
        scenarios = unit.get("scenario_ids")
        if not isinstance(scenarios, list) or any(not isinstance(s, str) for s in scenarios) or len(set(scenarios)) != len(scenarios):
            raise ValueError("scenario_ids must be a unique string list")
        for scenario in scenarios:
            text(scenario, "scenario ID")
        normalized[name] = {"dataset_id": origin[0], "original_drive_id": origin[1], "scenario_ids": sorted(scenarios)}
    if not isinstance(report.get("metrics"), dict):
        raise ValueError("Report metrics must be an object")
    return normalized


def paired_bootstrap(deltas, repetitions=10000, seed=42, lower_quantile=0.05):
    """Paired original-drive bootstrap; no reported CI enters this calculation."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or len(deltas) < 2 or not np.isfinite(deltas).all():
        raise ValueError("Paired bootstrap requires at least two finite drive deltas")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    # Bounded batches avoid allocating repetitions * all_drives at once.
    for start in range(0, repetitions, 16):
        stop = min(start + 16, repetitions)
        indices = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        means[start:stop] = deltas[indices].mean(axis=1)
    return {"ci95": np.quantile(means, [0.025, 0.975], method="linear").tolist(),
            "adjusted_lower_bound": float(np.quantile(means, lower_quantile, method="linear"))}


def metric_result(name, rule, before, after, metadata, coverage, spec):
    for report in (before, after):
        if not isinstance(report, dict):
            raise ValueError("Required metric is missing")
        if report.get("direction") != rule["direction"] or report.get("unit") != rule["unit"]:
            raise ValueError("Metric direction or unit differs from the gate spec")
        checksum(report.get("protocol_sha256"), "metric protocol_sha256")
        if not isinstance(report.get("units"), dict) or not report["units"]:
            raise ValueError("Metric needs per-drive values and denominators")
    if before["protocol_sha256"] != after["protocol_sha256"]:
        raise ValueError("Per-metric evaluation protocols differ")
    ids = sorted(before["units"])
    if set(ids) != set(after["units"]) or not set(ids) <= set(metadata):
        raise ValueError("Paired drive IDs differ or lack original-drive metadata")
    minimum = rule.get("min_units", 30)
    if len(ids) < minimum:
        raise ValueError(f"Need at least {minimum} independent drives; received {len(ids)}")
    coverage_counts = {scenario: sum(scenario in metadata[key]["scenario_ids"] for key in ids) for scenario in coverage}
    missing = {key: {"required": coverage[key], "observed": count} for key, count in coverage_counts.items() if count < coverage[key]}
    if missing:
        raise ValueError("Scenario coverage missing or insufficient: " + json.dumps(missing, sort_keys=True))
    old, new, denominators = [], [], []
    critical = name in spec.get("critical_failure_metrics", [])
    for key in ids:
        a, b = before["units"][key], after["units"][key]
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError("Each metric unit must contain value and denominator")
        x, y = finite(a.get("value"), "baseline value"), finite(b.get("value"), "candidate value")
        denominator = integer(a.get("denominator"), "baseline denominator", maximum=2**63-1)
        if integer(b.get("denominator"), "candidate denominator", maximum=2**63-1) != denominator:
            raise ValueError("Paired metric denominators differ")
        if critical and (x < 0 or y < 0 or not x.is_integer() or not y.is_integer()):
            raise ValueError("Critical failure metrics must contain nonnegative integer event counts")
        old.append(x)
        new.append(y)
        denominators.append(denominator)
    a, b = np.asarray(old), np.asarray(new)
    deltas = b - a if rule["direction"] == "higher" else a - b
    # Reject numeric overflow instead of publishing a malformed successful gate.
    if not np.isfinite(deltas).all() or not math.isfinite(float(a.mean())) or not math.isfinite(float(b.mean())):
        raise ValueError("Metric arithmetic overflow")
    seed = int.from_bytes(hashlib.sha256((str(spec.get("seed", 42)) + ":" + name).encode()).digest()[:8], "big")
    improvement_tests = sum(item.get("mode", "improve") == "improve" for item in spec["required_metrics"].values())
    lower_quantile = 0.05 / improvement_tests
    bootstrap = paired_bootstrap(deltas, spec.get("bootstrap_repetitions", 10000), seed, lower_quantile)
    interval, lower_bound = bootstrap["ci95"], bootstrap["adjusted_lower_bound"]
    mean_delta = float(deltas.mean())
    margin = rule.get("min_improvement", 0)
    mode = rule.get("mode", "improve")
    # A subtraction such as 1.1 - 1.0 must not turn equality to a 0.1
    # margin into a claimed strict improvement through floating-point noise.
    def strictly_above(value):
        return value > margin and not math.isclose(value, margin, rel_tol=1e-12, abs_tol=1e-15)
    acceptable = (strictly_above(mean_delta) and strictly_above(lower_bound)) if mode == "improve" else (mean_delta >= margin and lower_bound >= margin)
    failures = [] if acceptable else ["Required paired mean/adjusted lower bound did not exceed the improvement margin" if mode == "improve" else "Paired mean/adjusted bound permits regression beyond the maintenance requirement"]
    for bound, predicate in (("absolute_min", lambda value, limit: value < limit), ("absolute_max", lambda value, limit: value > limit)):
        if bound in rule and any(predicate(value, rule[bound]) for value in new):
            failures.append("Candidate drive violates " + bound)
    regressed = [key for key, delta in zip(ids, deltas) if delta < 0]
    require_no_regression = spec.get("require_no_unit_regression", False) or rule.get("require_no_unit_regression", False)
    if require_no_regression and regressed:
        failures.append("At least one original drive regressed")
    if critical and any(value != 0 for value in new):
        failures.append("Observed candidate critical failures must all be zero")
    return {"status": "not_qualified" if failures else "passed", "reasons": failures,
            "direction": rule["direction"], "unit": rule["unit"], "mode": mode, "n_units": len(ids),
            "dataset_ids": sorted({metadata[key]["dataset_id"] for key in ids}),
            "paired_unit_ids": ids, "paired_denominators": denominators,
            "baseline_mean": float(a.mean()), "candidate_mean": float(b.mean()),
            "mean_improvement": mean_delta, "improvement_ci95": interval, "min_improvement": margin,
            "gate_adjusted_lower_bound": lower_bound, "gate_lower_quantile": lower_quantile,
            "require_no_unit_regression": require_no_regression,
            "regressed_unit_ids": regressed, "coverage_counts": coverage_counts,
            "critical_failure_metric": critical,
            "observed_candidate_failure_count": sum(int(x) for x in new) if critical else None}


def evaluate_gate(baseline, candidate, spec):
    coverage = validate_spec(spec)
    result = {"schema_version": 1, "status": "blocked", "decision": "insufficient_comparable_evidence",
              "deployment_approved": False, "models_validated_for_control": False,
              "zero_accident_guarantee": False, "real_performance_validated": False,
              "metrics": {}, "blocking_reasons": [],
              "bootstrap": {"method": "paired original-drive bootstrap; one-sided Bonferroni-adjusted percentile lower bound",
                            "aggregation": "unweighted mean over original drives",
                            "familywise_alpha": 0.05,
                            "number_of_improvement_metrics": sum(rule.get("mode", "improve") == "improve" for rule in spec["required_metrics"].values()),
                            "repetitions": spec.get("bootstrap_repetitions", 10000), "seed": spec.get("seed", 42),
                            "confidence": 0.95, "reported_confidence_intervals_used": False},
              "limits": ["Submitted evidence is not an authenticated safety certification.",
                         "Original-drive independence and scenario annotations must be established by the evaluator.",
                         "Previously tuned-on tests and reused drives do not become fresh evidence through bootstrapping.",
                         "Pixel or path metrics and zero failures in observed runs never prove zero collision risk.",
                         "The default 30-drive minimum is an experimental threshold, not evidence of adequate safety exposure.",
                         "Passing applies only to configured metrics and submitted scenarios; deployment is never approved."]}
    try:
        left, right = validate_report(baseline), validate_report(candidate)
        for requirement, key in (("required_evidence_domain", "evidence_domain"),
                                 ("required_measurement_kind", "measurement_kind")):
            if requirement in spec:
                for label, report in (("Baseline", baseline), ("Candidate", candidate)):
                    if report.get(key) != spec[requirement]:
                        raise ValueError(f"{label} {key} must match {requirement}={spec[requirement]}")
        if baseline["evaluation_id"] == candidate["evaluation_id"]:
            raise ValueError("Baseline and candidate need distinct evaluation IDs")
        for key in ("protocol_sha256", "test_split_sha256", "hardware_id", "unit_type", "evidence_domain", "diagnostic"):
            if baseline[key] != candidate[key]:
                raise ValueError(key + " differs between evaluations")
        if baseline.get("measurement_kind") != candidate.get("measurement_kind"):
            raise ValueError("measurement_kind differs between evaluations or is missing from one report")
        if left != right:
            raise ValueError("Original-drive identities, datasets or scenario metadata differ")
    except (ValueError, TypeError, KeyError) as error:
        result["blocking_reasons"].append(str(error))
        return result
    result.update(baseline_evaluation_id=baseline["evaluation_id"], candidate_evaluation_id=candidate["evaluation_id"],
                  protocol_sha256=baseline["protocol_sha256"], test_split_sha256=baseline["test_split_sha256"],
                  hardware_id=baseline["hardware_id"], evidence_domain=baseline["evidence_domain"],
                  diagnostic=baseline["diagnostic"])
    if "measurement_kind" in baseline:
        result["measurement_kind"] = baseline["measurement_kind"]
    for name, rule in spec["required_metrics"].items():
        try:
            result["metrics"][name] = metric_result(name, rule, baseline["metrics"].get(name),
                                                     candidate["metrics"].get(name), left, coverage, spec)
        except (ValueError, TypeError, KeyError) as error:
            result["metrics"][name] = {"status": "blocked", "reasons": [str(error)]}
            result["blocking_reasons"].append(name + ": " + str(error))
    if result["blocking_reasons"]:
        return result
    passed = all(metric["status"] == "passed" for metric in result["metrics"].values())
    real = candidate["evidence_domain"] == "real_recorded" and candidate["diagnostic"] is False
    result.update(status=("passed" if real else "diagnostic_checks_passed") if passed else "not_qualified",
                  decision=("research_candidate_only" if real else "diagnostic_evidence_only") if passed else "required_improvement_not_demonstrated",
                  real_performance_validated=False)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("baseline", "candidate", "spec", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new gate result path; prior evidence is immutable")
    payloads = {name: getattr(args, name).read_bytes() for name in ("baseline", "candidate", "spec")}
    decoded = {name: json.loads(payload.decode("utf-8-sig")) for name, payload in payloads.items()}
    result = evaluate_gate(decoded["baseline"], decoded["candidate"], decoded["spec"])
    result["inputs"] = {name: {"path": str(getattr(args, name).resolve()),
                              "sha256": hashlib.sha256(payloads[name]).hexdigest()}
                        for name in ("baseline", "candidate", "spec")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "deployment_approved": False, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
