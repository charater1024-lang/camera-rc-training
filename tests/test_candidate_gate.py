"""Paired-drive gate contract fixtures; these fabricated data are not model evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_candidate_gate.py"
spec = importlib.util.spec_from_file_location("candidate_gate_under_test", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixtures(n=30):
    """Small complete schema example; deliberately named synthetic fixtures."""
    units = {f"drive_{i:03d}": {"dataset_id": "synthetic_contract_fixture", "original_drive_id": f"original_{i}",
                                "scenario_ids": ["speed_bin_75_85", "night_wet"]} for i in range(n)}
    def metric(direction, unit, value):
        return {"direction": direction, "unit": unit, "protocol_sha256": "b" * 64,
                "units": {key: {"value": value, "denominator": 100} for key in units}}
    baseline = {"schema_version": 1, "evaluation_id": "synthetic_fixture_old", "protocol_sha256": "a" * 64,
                "test_split_sha256": "c" * 64, "hardware_id": "fixture_hardware", "unit_type": "original_drive",
                "evidence_domain": "real_recorded", "diagnostic": False,
                "units": units, "metrics": {"recall": metric("higher", "fraction", .5),
                                             "latency": metric("lower", "ms", 100),
                                             "collision_count": metric("lower", "events", 0)}}
    candidate = deepcopy(baseline)
    candidate["evaluation_id"] = "synthetic_fixture_new"
    for record in candidate["metrics"]["recall"]["units"].values():
        record["value"] = .75
    for record in candidate["metrics"]["latency"]["units"].values():
        record["value"] = 75
    spec = {"schema_version": 1, "required_metrics": {
        "recall": {"direction": "higher", "unit": "fraction", "min_units": 30, "min_improvement": .1, "absolute_min": .7},
        "latency": {"direction": "lower", "unit": "ms", "min_units": 30, "absolute_max": 90},
        "collision_count": {"direction": "lower", "unit": "events", "mode": "maintain", "absolute_max": 0}},
        "coverage_required": {"speed_bin_75_85": 30, "night_wet": 30},
        "critical_failure_metrics": ["collision_count"], "require_no_unit_regression": True,
        "bootstrap_repetitions": 1000, "seed": 42}
    return baseline, candidate, spec


def test_all_improved_plus_zero_events_only_returns_research_candidate():
    result = gate.evaluate_gate(*fixtures())
    assert result["status"] == "passed"
    assert result["decision"] == "research_candidate_only"
    assert result["deployment_approved"] is False
    assert result["models_validated_for_control"] is False
    assert result["zero_accident_guarantee"] is False
    assert result["real_performance_validated"] is False
    assert result["metrics"]["collision_count"]["observed_candidate_failure_count"] == 0
    assert result["metrics"]["recall"]["gate_lower_quantile"] == .025


@pytest.mark.parametrize("domain,diagnostic", [("synthetic_fixture", False), ("simulated", False), ("real_recorded", True)])
def test_synthetic_or_diagnostic_evidence_never_qualifies_real_performance(domain, diagnostic):
    a, b, spec = fixtures()
    for report in (a, b):
        report.update(evidence_domain=domain, diagnostic=diagnostic)
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "diagnostic_checks_passed"
    assert result["decision"] == "diagnostic_evidence_only"
    assert result["real_performance_validated"] is False
    assert result["deployment_approved"] is False


def test_evidence_domain_mismatch_is_blocked():
    a, b, spec = fixtures()
    b["evidence_domain"] = "simulated"
    assert gate.evaluate_gate(a, b, spec)["status"] == "blocked"


@pytest.mark.parametrize("domain", ["real_recorded", "simulated", "synthetic_fixture"])
def test_spec_required_evidence_domain_accepts_exact_declared_domain(domain):
    a, b, spec = fixtures()
    spec["required_evidence_domain"] = domain
    for report in (a, b): report["evidence_domain"] = domain
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == ("passed" if domain == "real_recorded" else "diagnostic_checks_passed")
    assert result["real_performance_validated"] is False
    assert result["deployment_approved"] is False


@pytest.mark.parametrize("mode", ["both_wrong", "baseline_wrong", "candidate_wrong", "baseline_missing", "candidate_missing"])
def test_required_domain_blocks_wrong_or_missing_evidence_even_when_pair_matches(mode):
    a, b, spec = fixtures()
    spec.update(required_evidence_domain="real_recorded", scope="physical_evidence_only")
    if mode in {"both_wrong", "baseline_wrong"}: a["evidence_domain"] = "simulated"
    if mode in {"both_wrong", "candidate_wrong"}: b["evidence_domain"] = "simulated"
    if mode == "baseline_missing": del a["evidence_domain"]
    if mode == "candidate_missing": del b["evidence_domain"]
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert "evidence_domain" in result["blocking_reasons"][0]
    assert result["metrics"] == {}
    assert result["real_performance_validated"] is False


@pytest.mark.parametrize("kind", ["hardware_exposure_to_command", "simulated_exposure_to_command", "replay_ingest_to_command"])
def test_required_measurement_kind_accepts_matching_top_level_fields(kind):
    a, b, spec = fixtures()
    spec["required_measurement_kind"] = kind
    for report in (a, b): report["measurement_kind"] = kind
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "passed"
    assert result["measurement_kind"] == kind
    assert result["real_performance_validated"] is False
    assert result["models_validated_for_control"] is False


@pytest.mark.parametrize("mode", ["both_missing", "baseline_missing", "candidate_missing", "both_wrong", "baseline_wrong", "candidate_wrong"])
def test_required_hardware_kind_cannot_pass_missing_simulated_or_replay_measurements(mode):
    a, b, spec = fixtures()
    spec.update(required_evidence_domain="real_recorded", required_measurement_kind="hardware_exposure_to_command")
    for report in (a, b): report["measurement_kind"] = "hardware_exposure_to_command"
    if mode in {"both_missing", "baseline_missing"}: del a["measurement_kind"]
    if mode in {"both_missing", "candidate_missing"}: del b["measurement_kind"]
    if mode in {"both_wrong", "baseline_wrong"}: a["measurement_kind"] = "replay_ingest_to_command"
    if mode in {"both_wrong", "candidate_wrong"}: b["measurement_kind"] = "replay_ingest_to_command"
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert "required_measurement_kind" in result["blocking_reasons"][0]
    assert result["metrics"] == {}
    assert result["real_performance_validated"] is False
    assert result["deployment_approved"] is False


@pytest.mark.parametrize("field", ["required_evidence_domain", "required_measurement_kind"])
@pytest.mark.parametrize("invalid", [None, "", "unknown", True, []])
def test_optional_spec_evidence_constraints_reject_invalid_values(field, invalid):
    a, b, spec = fixtures()
    spec[field] = invalid
    with pytest.raises(ValueError, match=field):
        gate.evaluate_gate(a, b, spec)


@pytest.mark.parametrize("candidate_kind", [None, "replay_ingest_to_command", "unknown"])
def test_declared_measurement_kinds_must_be_valid_and_comparable_without_new_spec_fields(candidate_kind):
    a, b, spec = fixtures()
    a["measurement_kind"] = "hardware_exposure_to_command"
    if candidate_kind is not None:
        b["measurement_kind"] = candidate_kind
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert "measurement_kind" in result["blocking_reasons"][0]


def test_lower_is_better_delta_sign_and_drive_mean():
    a, b, spec = fixtures()
    result = gate.evaluate_gate(a, b, spec)
    assert result["metrics"]["recall"]["mean_improvement"] == .25
    assert result["metrics"]["latency"]["mean_improvement"] == 25
    assert result["metrics"]["latency"]["improvement_ci95"] == [25, 25]


def test_floating_point_equality_to_margin_is_not_strict_improvement():
    a, b, spec = fixtures()
    for row in a["metrics"]["recall"]["units"].values(): row["value"] = 1.0
    for row in b["metrics"]["recall"]["units"].values(): row["value"] = 1.1
    result = gate.evaluate_gate(a, b, spec)
    assert result["metrics"]["recall"]["status"] == "not_qualified"


def test_more_data_points_inside_one_drive_do_not_satisfy_unit_minimum():
    a, b, spec = fixtures(1)
    for report in (a, b):
        for metric in report["metrics"].values():
            metric["units"]["drive_000"]["denominator"] = 10**9
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert "independent drives" in result["metrics"]["recall"]["reasons"][0]


@pytest.mark.parametrize("field,value", [("protocol_sha256", "f" * 64), ("test_split_sha256", "f" * 64),
                                         ("hardware_id", "other_gpu"), ("unit_type", "pixel")])
def test_incomparable_global_protocols_block(field, value):
    a, b, spec = fixtures()
    b[field] = value
    assert gate.evaluate_gate(a, b, spec)["status"] == "blocked"


@pytest.mark.parametrize("change", ["missing_metric", "missing_unit", "denominator", "unit", "direction", "metric_protocol", "nonfinite"])
def test_missing_or_incomparable_metrics_block(change):
    a, b, spec = fixtures()
    metric = b["metrics"]["recall"]
    if change == "missing_metric": del b["metrics"]["recall"]
    elif change == "missing_unit": del metric["units"]["drive_000"]
    elif change == "denominator": metric["units"]["drive_000"]["denominator"] = 99
    elif change == "unit": metric["unit"] = "percent"
    elif change == "direction": metric["direction"] = "lower"
    elif change == "metric_protocol": metric["protocol_sha256"] = "d" * 64
    elif change == "nonfinite": metric["units"]["drive_000"]["value"] = float("nan")
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert result["metrics"]["recall"]["status"] == "blocked"


def test_repeated_original_drive_cannot_be_bootstrapped_as_independent():
    a, b, spec = fixtures()
    for report in (a, b):
        report["units"]["drive_001"]["original_drive_id"] = report["units"]["drive_000"]["original_drive_id"]
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "blocked"
    assert "masquerades" in result["blocking_reasons"][0]


def test_dataset_rename_cannot_make_same_drive_or_synthetic_clone_independent():
    a, b, spec = fixtures()
    for report in (a, b):
        report["units"]["drive_001"].update(dataset_id="synthetic_clone",
             original_drive_id=report["units"]["drive_000"]["original_drive_id"])
    assert gate.evaluate_gate(a, b, spec)["status"] == "blocked"


def test_scenario_coverage_is_checked_per_metric_not_unused_report_units():
    a, b, spec = fixtures(60)
    for report in (a, b):
        for key in list(report["units"])[:30]:
            report["units"][key]["scenario_ids"] = ["night_wet"]
        report["metrics"]["recall"]["units"] = {k: v for k, v in report["metrics"]["recall"]["units"].items() if k < "drive_030"}
    result = gate.evaluate_gate(a, b, spec)
    assert result["metrics"]["recall"]["status"] == "blocked"
    assert "coverage" in result["metrics"]["recall"]["reasons"][0]


def test_zero_baseline_failure_count_cannot_claim_strict_improvement():
    a, b, spec = fixtures()
    spec["required_metrics"]["collision_count"]["mode"] = "improve"
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "not_qualified"
    assert result["metrics"]["collision_count"]["gate_adjusted_lower_bound"] == 0


def test_candidate_critical_event_fails_even_if_baseline_had_many():
    a, b, spec = fixtures()
    for entry in a["metrics"]["collision_count"]["units"].values(): entry["value"] = 5
    b["metrics"]["collision_count"]["units"]["drive_000"]["value"] = 1
    spec["required_metrics"]["collision_count"].pop("absolute_max")
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "not_qualified"
    assert "Observed candidate critical failures" in " ".join(result["metrics"]["collision_count"]["reasons"])


def test_noninteger_failure_counts_are_blocked():
    a, b, spec = fixtures()
    b["metrics"]["collision_count"]["units"]["drive_000"]["value"] = .1
    assert gate.evaluate_gate(a, b, spec)["status"] == "blocked"


def test_reported_confidence_interval_cannot_override_worse_evidence():
    a, b, spec = fixtures()
    for row in b["metrics"]["latency"]["units"].values(): row["value"] = 200
    b["metrics"]["latency"].update(value=0, n_units=10000000, paired_delta_ci95=[100, 200])
    result = gate.evaluate_gate(a, b, spec)
    assert result["status"] == "not_qualified"
    assert result["metrics"]["latency"]["improvement_ci95"] == [-100, -100]


def test_positive_mean_with_uncertain_improvement_does_not_pass():
    a, b, spec = fixtures()
    spec["require_no_unit_regression"] = False
    spec["required_metrics"]["latency"].pop("absolute_max")
    for i, row in enumerate(b["metrics"]["latency"]["units"].values()):
        row["value"] = 0 if i == 0 else 101
    result = gate.evaluate_gate(a, b, spec)
    metric = result["metrics"]["latency"]
    assert metric["mean_improvement"] > 0
    assert metric["gate_adjusted_lower_bound"] < 0
    assert metric["status"] == "not_qualified"


def test_no_drive_regression_can_be_required_despite_good_mean():
    a, b, spec = fixtures()
    b["metrics"]["latency"]["units"]["drive_000"]["value"] = 101
    spec["required_metrics"]["latency"].pop("absolute_max")
    result = gate.evaluate_gate(a, b, spec)
    assert result["metrics"]["latency"]["gate_adjusted_lower_bound"] > 0
    assert result["metrics"]["latency"]["regressed_unit_ids"] == ["drive_000"]
    assert result["metrics"]["latency"]["status"] == "not_qualified"


def test_absolute_bounds_apply_to_worst_drive_not_mean():
    a, b, spec = fixtures()
    b["metrics"]["recall"]["units"]["drive_000"]["value"] = .69
    result = gate.evaluate_gate(a, b, spec)
    assert result["metrics"]["recall"]["candidate_mean"] > .7
    assert result["metrics"]["recall"]["status"] == "not_qualified"


def test_bootstrap_deterministic_and_more_claims_lower_gate_quantile():
    a, b, spec = fixtures()
    first = gate.evaluate_gate(a, b, spec)
    assert first == gate.evaluate_gate(a, b, spec)
    spec["required_metrics"]["other_recall"] = deepcopy(spec["required_metrics"]["recall"])
    for report in (a, b): report["metrics"]["other_recall"] = deepcopy(report["metrics"]["recall"])
    second = gate.evaluate_gate(a, b, spec)
    assert second["bootstrap"]["number_of_improvement_metrics"] == 3
    assert second["metrics"]["recall"]["gate_lower_quantile"] == pytest.approx(.05 / 3)
    assert second["metrics"]["recall"]["gate_lower_quantile"] < first["metrics"]["recall"]["gate_lower_quantile"]


def test_bootstrap_is_independent_of_input_dictionary_order():
    a, b, spec = fixtures()
    first = gate.evaluate_gate(a, b, spec)
    for report in (a, b):
        report["units"] = dict(reversed(list(report["units"].items())))
        for metric in report["metrics"].values(): metric["units"] = dict(reversed(list(metric["units"].items())))
    assert gate.evaluate_gate(a, b, spec) == first


def test_maintaining_everything_is_not_an_improvement_spec():
    a, b, spec = fixtures()
    for rule in spec["required_metrics"].values(): rule["mode"] = "maintain"
    with pytest.raises(ValueError, match="At least one metric"):
        gate.evaluate_gate(a, b, spec)


def test_cli_missing_evidence_writes_blocked_report_and_preserves_existing_output(tmp_path):
    a, b, spec = fixtures()
    del b["metrics"]["latency"]
    for name, content in (("baseline", a), ("candidate", b), ("spec", spec)):
        (tmp_path / (name + ".json")).write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "result.json"
    args = [sys.executable, "-X", "utf8", "-B", str(SCRIPT), "--baseline", str(tmp_path / "baseline.json"),
            "--candidate", str(tmp_path / "candidate.json"), "--spec", str(tmp_path / "spec.json"), "--output", str(output)]
    completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["deployment_approved"] is False
    assert set(report["inputs"]) == {"baseline", "candidate", "spec"}
    prior = output.read_bytes()
    completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode != 0
    assert output.read_bytes() == prior
