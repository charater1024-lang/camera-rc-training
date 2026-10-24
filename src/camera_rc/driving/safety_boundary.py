"""Command-boundary reuse of the simulator's independent evidence supervisor.

Legacy local planning can be studied in an explicitly named diagnostic dry-run.
Missing real producers are never replaced with homography-derived oracle range.
"""
from dataclasses import dataclass
import math
from .shadow_safety import ShadowPolicy, ShadowSafetySupervisor, SceneEvidence
from .types import VehicleState
from .behavior import RoadContext, validate_road_plan


@dataclass(frozen=True)
class GeometryObservation:
    captured_at_s: float
    clock_id: str
    road_grade: float
    camera_pitch_delta_rad: float
    exposure_timing_verified: bool
    range_mode: str  # measured_flat_ground or metric_3d


@dataclass(frozen=True)
class SafetyObservation:
    scene: SceneEvidence
    road: RoadContext
    anchor_state: VehicleState
    origin: str  # measured, recorded or synthetic; provenance is not a sensor validation
    geometry: GeometryObservation | None = None


class SafetyBoundary:
    def __init__(self, config, physical=False):
        self.physical = physical
        self.max_flat_grade = .02
        self.max_pitch_delta = .01
        self.diagnostic = config.get("mode") == "diagnostic_local_only" and not physical
        self.policy = ShadowPolicy(**config["policy"]) if config.get("policy") else None
        self.supervisor = ShadowSafetySupervisor(self.policy) if self.policy else None
        self.latched_reason = None
        self._last_observation = None
        self._last_plan = None
        self.last_decision = {"mode": "STOP", "reasons": ["initial_observation_required"], "proposal_eligible": False}

    def review(self, plan, state, observation, now):
        try:
            if type(now) not in (int,float) or not math.isfinite(now) or now < 0:
                raise ValueError("Invalid clock")
            return self._review(plan,state,observation,now)
        except (TypeError,ValueError,AttributeError,KeyError,OverflowError):
            self.latched_reason = "malformed_safety_evidence"
            self.last_decision = {"mode":"LATCHED_STOP","reasons":[self.latched_reason],"proposal_eligible":False}
            return self.latched_reason

    def _review(self, plan, state, observation, now):
        reason = self.latched_reason
        if reason is None and self.diagnostic:
            self.last_decision = {"mode": "DIAGNOSTIC_ONLY", "reasons": ["unverified_local_proposal"], "proposal_eligible": False}
            return None
        if reason is None:
            if self.supervisor is None:
                reason = "safety_policy_missing"
            elif not isinstance(observation, SafetyObservation):
                reason = "safety_observation_missing"
            elif observation.origin not in {"measured", "recorded", "synthetic"} or (self.physical and observation.origin != "measured"):
                reason = "safety_evidence_origin_rejected"
            elif (not isinstance(observation.geometry, GeometryObservation)
                  or observation.geometry.exposure_timing_verified is not True
                  or observation.geometry.clock_id != self.policy.clock_id
                  or type(observation.geometry.captured_at_s) not in (int,float)
                  or not math.isfinite(observation.geometry.captured_at_s)
                  or not 0 <= now-observation.geometry.captured_at_s <= self.policy.max_observation_age_s):
                reason = "geometry_or_exposure_evidence_missing"
            elif (type(observation.geometry.road_grade) not in (int,float)
                  or type(observation.geometry.camera_pitch_delta_rad) not in (int,float)
                  or not math.isfinite(observation.geometry.road_grade)
                  or not math.isfinite(observation.geometry.camera_pitch_delta_rad)
                  or observation.geometry.range_mode not in {"measured_flat_ground","metric_3d"}):
                reason = "invalid_geometry_evidence"
            elif (observation.geometry.range_mode == "measured_flat_ground" and
                  (abs(observation.geometry.road_grade)>self.max_flat_grade
                   or abs(observation.geometry.camera_pitch_delta_rad)>self.max_pitch_delta)):
                reason = "flat_ground_assumption_violated"
            elif state is None:
                reason = "no_vehicle_state"
            elif (not isinstance(observation.anchor_state, VehicleState)
                  or type(state.timestamp) not in (int,float) or not math.isfinite(state.timestamp)
                  or not 0 <= now-state.timestamp <= self.policy.max_state_age_s
                  or state.armed is not True or state.estop is not False
                  or type(state.speed_mps) not in (int,float) or not math.isfinite(state.speed_mps)
                  or not 0 <= state.speed_mps <= self.policy.max_speed_mps
                  or type(state.yaw_rate_rps) not in (int,float) or not math.isfinite(state.yaw_rate_rps)
                  or abs(state.yaw_rate_rps)*state.speed_mps > self.policy.max_lateral_acceleration_mps2):
                reason = "live_telemetry_invalid"
            elif (type(observation.anchor_state.speed_mps) not in (int,float)
                  or type(observation.anchor_state.timestamp) not in (int,float)
                  or not math.isfinite(observation.anchor_state.speed_mps)
                  or not math.isfinite(observation.anchor_state.timestamp)
                  or not 0 <= now-observation.anchor_state.timestamp <= self.policy.max_state_age_s
                  or state.speed_mps > observation.anchor_state.speed_mps + observation.scene.speed_error_mps
                     + self.policy.acceleration_during_delay_mps2*(now-observation.anchor_state.timestamp)):
                reason = "telemetry_exceeds_evidence_speed_bound"
            elif observation is self._last_observation and plan is not self._last_plan:
                reason = "new_plan_requires_new_observation"
            else:
                road_plan = validate_road_plan(plan, observation.road, now,
                    clock_id=self.policy.clock_id, max_age_s=self.policy.max_observation_age_s,
                    body_radius_m=self.policy.body_radius_m+self.policy.lateral_margin_m)
                if not road_plan.valid:
                    reason = road_plan.reason
                    if reason in {"traffic_signal_stop","right_of_way_unknown_or_yield","route_unavailable","route_stop","goal_braking","requested_speed_stop"}:
                        self.last_decision = {"mode":"HOLD_STOP","reasons":[reason],"proposal_eligible":False}
                        return reason
                else:
                    method = self.supervisor.recheck if observation is self._last_observation else self.supervisor.review
                    decision = method(plan,observation.anchor_state,observation.scene,now)
                    self._last_observation, self._last_plan = observation, plan
                    self.last_decision = {"mode": decision.mode, "reasons": list(decision.reasons),
                                          "proposal_eligible": decision.proposal_eligible}
                    if decision.proposal_eligible:
                        return None
                    if decision.mode == "WARMUP_STOP":
                        return decision.reasons[0]
                    reason = decision.reasons[0]
        self.latched_reason = reason
        self.last_decision = {"mode": "LATCHED_STOP", "reasons": [reason], "proposal_eligible": False}
        return reason
