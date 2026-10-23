"""Synthetic contracts for the offline supervisor; not recorded driving data."""
from dataclasses import asdict, replace
import hashlib

import numpy as np

from .shadow_safety import (FRAME, Grid, ModuleEvidence, REQUIRED_MODULES, SceneEvidence,
                            ShadowPolicy, ShadowSafetySupervisor, Track)
from .types import Plan, VehicleState


def example(sequence=0, now=10.0):
    """Small simulated vehicle only. All identities and physical bounds are fixtures."""
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    policy = ShadowPolicy(
        clock_id="synthetic_monotonic", calibration_sha256=digest("synthetic_calibration"),
        artifacts={name: digest("synthetic_" + name) for name in REQUIRED_MODULES},
        allowed_conditions=("synthetic_dry_closed_course",), max_speed_mps=1.0,
        min_braking_deceleration_mps2=0.8, acceleration_during_delay_mps2=0.3,
        reaction_bound_s=0.3, stopping_margin_m=0.2, body_radius_m=0.4,
        lateral_margin_m=0.1, max_localization_error_m=0.2, max_speed_error_mps=0.1,
        max_motion_compensation_error_m=0.1,
        max_observation_age_s=0.2, max_state_age_s=0.1, max_plan_age_s=0.2, max_skew_s=0.05)
    modules = tuple(ModuleEvidence(name, "ok", now - 0.03, sequence, policy.clock_id,
                                  policy.calibration_sha256, policy.artifacts[name])
                    for name in sorted(REQUIRED_MODULES))
    shape = (60, 110)
    grid = Grid(-1.0, -3.0, 0.1, np.ones(shape, dtype=bool), np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=bool), np.full(shape, 0.95, dtype=np.float32),
                now - 0.02, policy.clock_id, policy.calibration_sha256, FRAME)
    scene = SceneEvidence(modules, grid, (), "synthetic_dry_closed_course", True, True, True,
                          0.05, 0.05, 0.02, policy.clock_id, now - 0.02,
                          3.0, now - 0.02, policy.calibration_sha256, policy.clock_id)
    plan = Plan(np.array([[0., 0.], [5., 0.]]), 0.6, True, "synthetic_proposal")
    state = VehicleState(now - 0.02, 0.5, armed=True, estop=False)
    return policy, plan, state, scene


def altered(name, sequence=0, now=10.0):
    policy, plan, state, scene = example(sequence, now)
    modules = list(scene.modules)
    if name == "missing_detector":
        scene = replace(scene, modules=tuple(m for m in modules if m.name != "objects"))
    elif name == "failed_detector":
        scene = replace(scene, modules=tuple(replace(m, status="failed") if m.name == "objects" else m for m in modules))
    elif name in {"stale_observation", "future_observation", "exposure_skew", "nan_stamp", "wrong_clock", "wrong_calibration", "unapproved_model"}:
        updates = {
            "stale_observation": {"captured_at_s": now - 1},
            "future_observation": {"captured_at_s": now + 0.01},
            "exposure_skew": {"captured_at_s": now - 0.15},
            "nan_stamp": {"captured_at_s": float("nan")},
            "wrong_clock": {"clock_id": "wall_clock"},
            "wrong_calibration": {"calibration_sha256": "0" * 64},
            "unapproved_model": {"artifact_sha256": "0" * 64},
        }[name]
        modules[0] = replace(modules[0], **updates)
        scene = replace(scene, modules=tuple(modules))
    elif name in {"unknown_cell", "occupied_cell", "anomaly_cell", "nonroad_cell", "nan_probability"}:
        # A small cell between the original sparse path endpoints must be checked.
        layer, value = {
            "unknown_cell": (scene.grid.observed, False),
            "occupied_cell": (scene.grid.occupied, True),
            "anomaly_cell": (scene.grid.object_or_unknown, True),
            "nonroad_cell": (scene.grid.road_probability, 0.1),
            "nan_probability": (scene.grid.road_probability, float("nan")),
        }[name]
        layer[30, 35] = value
    elif name == "crossing_track":
        scene = replace(scene, tracks=(Track("synthetic_pedestrian", now - 0.02, 2.0, 2.0, 0.0, -2.0, 0.2, 0.1, 0.1, 0.2),))
    elif name == "uncertain_track":
        scene = replace(scene, tracks=(Track("synthetic_uncertain", now - 0.02, 2.0, 2.0, 0.0, 0.0, 0.2, 0.1, 2.0, 0.2),))
    elif name == "no_metric_scale":
        scene = replace(scene, metric_scale_verified=False)
    elif name == "braking_fault":
        scene = replace(scene, braking_available=False)
    elif name == "unvalidated_night":
        scene = replace(scene, condition="night")
    elif name == "outside_area":
        scene = replace(scene, inside_operating_area=False)
    elif name == "short_prediction":
        scene = replace(scene, prediction_horizon_s=0.1)
    elif name == "localization_error":
        scene = replace(scene, localization_error_m=0.5)
    elif name == "speed_error":
        scene = replace(scene, speed_error_mps=0.5)
    elif name == "short_path":
        plan = replace(plan, points=np.array([[0., 0.], [0.1, 0.]]))
    elif name == "grid_boundary":
        plan = replace(plan, points=np.array([[0., 0.], [9.7, 0.]]))
    elif name == "stale_plan":
        scene = replace(scene, plan_created_at_s=now - 1)
    elif name == "estop":
        state = replace(state, estop=True)
    elif name == "overspeed":
        state = replace(state, speed_mps=2.0)
    elif name == "stale_state":
        state = replace(state, timestamp=now - 1)
    elif name != "clear_observed_corridor":
        raise ValueError(f"Unknown synthetic case {name}")
    return policy, plan, state, scene


CASES = {
    "clear_observed_corridor": "supplied_offline_evidence_passed",
    "missing_detector": "missing_or_duplicate_modules",
    "failed_detector": "module_unavailable",
    "stale_observation": "stale_or_future_observation",
    "future_observation": "stale_or_future_observation",
    "exposure_skew": "sensor_exposure_skew",
    "nan_stamp": "stale_or_future_observation",
    "wrong_clock": "clock_domain_mismatch",
    "wrong_calibration": "calibration_mismatch",
    "unapproved_model": "producer_identity_mismatch",
    "unknown_cell": "unobserved_swept_cell",
    "occupied_cell": "occupied_swept_cell",
    "anomaly_cell": "object_or_anomaly_veto",
    "nonroad_cell": "semantic_disagreement_or_nonroad",
    "nan_probability": "invalid_semantic_probability",
    "crossing_track": "reachable_dynamic_obstacle",
    "uncertain_track": "reachable_dynamic_obstacle",
    "no_metric_scale": "metric_scale_unverified",
    "braking_fault": "braking_unavailable",
    "unvalidated_night": "outside_odd",
    "outside_area": "outside_odd",
    "short_prediction": "prediction_horizon_too_short",
    "localization_error": "localization_uncertainty",
    "speed_error": "speed_uncertainty",
    "short_path": "insufficient_stopping_horizon",
    "grid_boundary": "swept_body_outside_observed_grid",
    "stale_plan": "stale_or_future_plan",
    "estop": "estop_or_disarmed",
    "overspeed": "speed_outside_supported_odd",
    "stale_state": "stale_or_future_state",
}


def run_scenarios():
    results = []
    for name, expected in CASES.items():
        policy, _, _, _ = altered(name)
        supervisor = ShadowSafetySupervisor(policy)
        for sequence in range(policy.healthy_cycles_to_reset):
            now = 10.0 + sequence * 0.1
            _, plan, state, scene = altered(name, sequence, now)
            decision = supervisor.review(plan, state, scene, now)
        passed = decision.reasons == (expected,) and not decision.actuation_enabled
        passed = passed and decision.proposal_eligible == (name == "clear_observed_corridor")
        results.append({"case": name, "expected_reason": expected, "matched": passed, "decision": asdict(decision)})
    return {"schema_version": 1, "scope": "synthetic_offline_contract_regression_not_driving_validation",
            "model_inference_performed": False, "real_sensor_inputs": False, "actuation_enabled": False,
            "collision_rate_measured": False, "speed_benchmark_performed": False,
            "passed": sum(item["matched"] for item in results), "total": len(results), "cases": results}
