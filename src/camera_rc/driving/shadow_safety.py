"""Offline multi-module evidence gate. This module NEVER enables actuation.

It checks supplied metric observations, not their real-world truth. A passing
proposal is neither a collision-free proof nor approval to drive. Required
range, tracking, localization and timing producers are NOT supplied by the
legacy monocular segmentation pipeline. See docs/SAFETY_ARCHITECTURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import numpy as np

from .types import Plan, VehicleState


REQUIRED_MODULES = frozenset({"semantic", "objects", "metric_occupancy", "tracking", "localization"})
FRAME = "rear_axle_x_forward_y_left_metres"


@dataclass(frozen=True)
class ModuleEvidence:
    name: str
    status: str  # "ok" is distinct from missing, failed, timed_out, etc.
    captured_at_s: float
    sequence: int
    clock_id: str
    calibration_sha256: str
    artifact_sha256: str  # checkpoint or deterministic producer version


@dataclass(frozen=True)
class Grid:
    x_min_m: float
    y_min_m: float
    resolution_m: float
    observed: np.ndarray  # bool: metric ranging supports the cell
    occupied: np.ndarray  # bool: metric occupancy veto
    object_or_unknown: np.ndarray  # bool: projected object/anomaly veto
    road_probability: np.ndarray  # semantic evidence cannot override vetoes
    anchor_time_s: float  # ego frame reference time, not image reception time
    clock_id: str
    calibration_sha256: str
    frame: str = FRAME


@dataclass(frozen=True)
class Track:
    track_id: str
    anchor_time_s: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    radius_m: float
    position_error_m: float
    velocity_error_mps: float
    acceleration_bound_mps2: float


@dataclass(frozen=True)
class SceneEvidence:
    modules: tuple[ModuleEvidence, ...]
    grid: Grid
    tracks: tuple[Track, ...]
    condition: str
    inside_operating_area: bool
    metric_scale_verified: bool
    braking_available: bool
    localization_error_m: float
    speed_error_mps: float
    motion_compensation_error_m: float
    state_clock_id: str
    plan_anchor_time_s: float
    prediction_horizon_s: float
    plan_created_at_s: float
    plan_calibration_sha256: str
    plan_clock_id: str


@dataclass(frozen=True)
class ShadowPolicy:
    """Explicit engineering assumptions; no validated vehicle defaults."""
    clock_id: str
    calibration_sha256: str
    artifacts: dict[str, str]
    allowed_conditions: tuple[str, ...]
    max_speed_mps: float
    min_braking_deceleration_mps2: float
    acceleration_during_delay_mps2: float
    reaction_bound_s: float
    stopping_margin_m: float
    # Max distance from rear axle to ANY body point, including overhangs.
    body_radius_m: float
    lateral_margin_m: float
    max_localization_error_m: float
    max_speed_error_mps: float
    max_motion_compensation_error_m: float
    max_observation_age_s: float
    max_state_age_s: float
    max_plan_age_s: float
    max_skew_s: float
    min_road_probability: float = 0.65
    max_curvature_per_m: float = 0.3
    max_lateral_acceleration_mps2: float = 0.5
    max_path_sampling_m: float = 0.05
    healthy_cycles_to_reset: int = 3
    max_grid_cells: int = 200000
    max_path_points: int = 256
    max_dense_points: int = 2048
    max_tracks: int = 256

    def __post_init__(self):
        if not isinstance(self.clock_id, str) or not self.clock_id:
            raise ValueError("Explicit clock domain required")
        if not _digest(self.calibration_sha256) or set(self.artifacts) != REQUIRED_MODULES:
            raise ValueError("Pin calibration and all required producer artifacts")
        if not all(_digest(v) for v in self.artifacts.values()):
            raise ValueError("Expected SHA256 producer identities")
        if not self.allowed_conditions or not all(isinstance(v, str) and v for v in self.allowed_conditions):
            raise ValueError("Explicit ODD conditions required")
        for name in ("max_speed_mps", "min_braking_deceleration_mps2", "reaction_bound_s",
                     "stopping_margin_m", "body_radius_m", "lateral_margin_m",
                     "max_localization_error_m", "max_speed_error_mps", "max_motion_compensation_error_m",
                     "max_observation_age_s", "max_state_age_s",
                     "max_plan_age_s", "max_skew_s", "max_curvature_per_m", "max_lateral_acceleration_mps2",
                     "max_path_sampling_m"):
            if not _number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not _number(self.acceleration_during_delay_mps2) or self.acceleration_during_delay_mps2 < 0:
            raise ValueError("Invalid acceleration bound")
        if not _number(self.min_road_probability) or not 0 < self.min_road_probability <= 1:
            raise ValueError("Invalid road threshold")
        for name in ("healthy_cycles_to_reset", "max_grid_cells", "max_path_points", "max_dense_points", "max_tracks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ShadowDecision:
    mode: str
    reasons: tuple[str, ...]
    proposal_eligible: bool = False
    required_stopping_distance_m: float | None = None
    checked_cells: int = 0
    # Not constructor arguments: even a passing review is not a drive command.
    actuation_enabled: bool = field(default=False, init=False)
    scope: str = field(default="offline_evidence_contract_only", init=False)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class _Reject(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Reject(reason)


class ShadowSafetySupervisor:
    """Stateful veto with explicit stopped reset; separate from AI inference.

    Road confidence, empty detector output and model votes never create range
    observations. Every swept cell must be observed AND free of every veto.
    Python/NumPy is not a real-time or independent hardware safety channel.
    """
    def __init__(self, policy: ShadowPolicy):
        self.policy = policy
        # Copy pinning data so later caller mutations cannot change the contract.
        self._artifacts = dict(policy.artifacts)
        self._conditions = frozenset(policy.allowed_conditions)
        self._last_now = None
        self._sequences = {}
        self._captures = {}
        self._latched = False
        self._healthy_cycles = 0

    def review(self, plan: Plan, state: VehicleState, scene: SceneEvidence,
               now_s: float, *, reset_requested: bool = False) -> ShadowDecision:
        try:
            _require(_number(now_s) and now_s >= 0, "invalid_clock")
            _require(self._last_now is None or now_s > self._last_now, "nonmonotonic_review_clock")
            self._last_now = now_s
            _require(type(reset_requested) is bool, "invalid_reset_request")
            distance, cells = self._check(plan, state, scene, now_s)
        except _Reject as exc:
            self._latched, self._healthy_cycles = True, 0
            return ShadowDecision("LATCHED_STOP", (str(exc),))
        except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
            self._latched, self._healthy_cycles = True, 0
            return ShadowDecision("LATCHED_STOP", ("malformed_evidence",))
        self._healthy_cycles += 1
        if self._latched:
            current_speed_upper = (state.speed_mps + scene.speed_error_mps
                                   + self.policy.acceleration_during_delay_mps2 * (now_s - state.timestamp))
            if (reset_requested and current_speed_upper <= 0.01
                    and self._healthy_cycles >= self.policy.healthy_cycles_to_reset):
                self._latched = False
            else:
                return ShadowDecision("LATCHED_STOP", ("explicit_stopped_reset_required",),
                                      required_stopping_distance_m=distance, checked_cells=cells)
        if self._healthy_cycles < self.policy.healthy_cycles_to_reset:
            return ShadowDecision("WARMUP_STOP", ("fresh_healthy_cycles_required",),
                                  required_stopping_distance_m=distance, checked_cells=cells)
        return ShadowDecision("PROPOSAL_ELIGIBLE", ("supplied_offline_evidence_passed",), True, distance, cells)

    def _check(self, plan, state, scene, now):
        p = self.policy
        _require(isinstance(scene, SceneEvidence) and isinstance(state, VehicleState)
                 and isinstance(plan, Plan), "invalid_input_contract")
        _require(_number(state.timestamp) and 0 <= now - state.timestamp <= p.max_state_age_s,
                 "stale_or_future_state")
        _require(scene.state_clock_id == p.clock_id, "state_clock_domain_mismatch")
        _require(_number(state.speed_mps) and 0 <= state.speed_mps <= p.max_speed_mps,
                 "speed_outside_supported_odd")
        _require(_number(state.yaw_rate_rps), "invalid_yaw_rate")
        _require(state.estop is False and state.armed is True, "estop_or_disarmed")
        _require(scene.inside_operating_area is True and scene.condition in self._conditions, "outside_odd")
        _require(scene.metric_scale_verified is True, "metric_scale_unverified")
        _require(scene.braking_available is True, "braking_unavailable")
        _require(_number(scene.localization_error_m) and 0 <= scene.localization_error_m <= p.max_localization_error_m,
                 "localization_uncertainty")
        _require(_number(scene.speed_error_mps) and 0 <= scene.speed_error_mps <= p.max_speed_error_mps,
                 "speed_uncertainty")
        _require(_number(scene.motion_compensation_error_m)
                 and 0 <= scene.motion_compensation_error_m <= p.max_motion_compensation_error_m,
                 "motion_compensation_uncertainty")
        _require(_number(scene.plan_anchor_time_s) and scene.plan_anchor_time_s == state.timestamp,
                 "plan_ego_anchor_mismatch")
        _require(scene.plan_clock_id == p.clock_id
                 and scene.plan_calibration_sha256 == p.calibration_sha256, "plan_frame_or_clock_mismatch")
        _require(_number(scene.plan_created_at_s) and 0 <= now - scene.plan_created_at_s <= p.max_plan_age_s,
                 "stale_or_future_plan")
        _require(len(scene.modules) == len(REQUIRED_MODULES), "missing_or_duplicate_modules")
        names, stamps, sequences = set(), [], {}
        for evidence in scene.modules:
            _require(isinstance(evidence, ModuleEvidence) and evidence.name not in names, "missing_or_duplicate_modules")
            names.add(evidence.name)
            _require(evidence.name in REQUIRED_MODULES and evidence.status == "ok", "module_unavailable")
            _require(evidence.clock_id == p.clock_id, "clock_domain_mismatch")
            _require(evidence.calibration_sha256 == p.calibration_sha256, "calibration_mismatch")
            _require(evidence.artifact_sha256 == self._artifacts[evidence.name], "producer_identity_mismatch")
            _require(_number(evidence.captured_at_s) and 0 <= now - evidence.captured_at_s <= p.max_observation_age_s,
                     "stale_or_future_observation")
            _require(evidence.captured_at_s > self._captures.get(evidence.name, -math.inf), "frozen_or_reversed_capture")
            _require(type(evidence.sequence) is int and evidence.sequence >= 0
                     and evidence.sequence > self._sequences.get(evidence.name, -1), "replayed_observation")
            stamps.append(evidence.captured_at_s)
            sequences[evidence.name] = evidence.sequence
        _require(names == REQUIRED_MODULES, "missing_or_duplicate_modules")
        _require(max(stamps) - min(stamps) <= p.max_skew_s, "sensor_exposure_skew")
        # Consume validated identities even if spatial evidence then vetoes.
        self._sequences.update(sequences)
        self._captures.update({e.name: e.captured_at_s for e in scene.modules})
        _require(plan.valid is True and _number(plan.target_speed_mps)
                 and 0 < plan.target_speed_mps <= p.max_speed_mps, "invalid_or_stop_proposal")
        _require(_number(plan.curvature) and abs(plan.curvature) <= p.max_curvature_per_m, "invalid_curvature")
        points = np.array(plan.points, dtype=np.float64, copy=True)
        _require(points.ndim == 2 and points.shape[1] == 2 and 2 <= len(points) <= p.max_path_points
                 and np.isfinite(points).all(), "invalid_path")
        _require(np.linalg.norm(points[0]) <= 1e-8, "path_not_at_rear_axle")
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        _require((lengths > 1e-8).all() and (segments[:, 0] >= 0).all(), "unsupported_reverse_or_zero_segment")
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        _require(abs(headings[0]) <= 0.1, "unchecked_initial_heading_transition")
        ego_age = now - state.timestamp
        current_speed_upper = (state.speed_mps + scene.speed_error_mps
                               + p.acceleration_during_delay_mps2 * ego_age)
        speed = max(current_speed_upper, plan.target_speed_mps + scene.speed_error_mps)
        _require(speed <= p.max_speed_mps, "speed_upper_bound_outside_odd")
        _require(abs(state.yaw_rate_rps) * speed <= p.max_lateral_acceleration_mps2,
                 "measured_lateral_acceleration_bound")
        # Account for age as well as the declared remaining processing/actuation delay.
        delay = p.reaction_bound_s + max(now - min(stamps), now - state.timestamp, now - scene.plan_created_at_s)
        accelerated_speed = speed + p.acceleration_during_delay_mps2 * delay
        _require(accelerated_speed <= p.max_speed_mps, "predicted_speed_outside_odd")
        stopping = (speed * delay + 0.5 * p.acceleration_during_delay_mps2 * delay ** 2
                    + accelerated_speed ** 2 / (2 * p.min_braking_deceleration_mps2) + p.stopping_margin_m)
        window = delay + accelerated_speed / p.min_braking_deceleration_mps2
        track_window = window + now - state.timestamp
        _require(_number(scene.prediction_horizon_s) and scene.prediction_horizon_s >= track_window,
                 "prediction_horizon_too_short")
        _require(float(lengths.sum()) >= stopping, "insufficient_stopping_horizon")
        grid = scene.grid
        _require(isinstance(grid, Grid) and grid.frame == FRAME, "grid_frame_mismatch")
        _require(_number(grid.anchor_time_s) and grid.anchor_time_s == state.timestamp
                 and grid.clock_id == p.clock_id and grid.calibration_sha256 == p.calibration_sha256,
                 "grid_ego_anchor_mismatch")
        _require(_number(grid.resolution_m) and grid.resolution_m > 0
                 and _number(grid.x_min_m) and _number(grid.y_min_m), "invalid_grid_geometry")
        arrays = (grid.observed, grid.occupied, grid.object_or_unknown, grid.road_probability)
        _require(all(isinstance(a, np.ndarray) and a.ndim == 2 for a in arrays), "invalid_grid_layers")
        shape = grid.observed.shape
        _require(min(shape) > 0 and grid.observed.size <= p.max_grid_cells
                 and all(a.shape == shape for a in arrays), "grid_layer_mismatch")
        _require(all(a.dtype == np.bool_ for a in arrays[:3]), "nonboolean_occupancy")
        _require(np.issubdtype(grid.road_probability.dtype, np.floating)
                 and np.isfinite(grid.road_probability).all()
                 and (grid.road_probability >= 0).all() and (grid.road_probability <= 1).all(), "invalid_semantic_probability")
        # Geometry validation must not become arbitrarily coarse when a caller
        # supplies a coarse grid. This independent bound also governs corners.
        step = min(grid.resolution_m / 2, p.max_path_sampling_m)
        counts_float = np.ceil(lengths / step)
        _require(np.isfinite(counts_float).all() and counts_float.sum() + 1 <= p.max_dense_points,
                 "path_sampling_budget_exceeded")
        dense = np.concatenate([points[i] + segments[i] * np.linspace(0, 1, int(n), endpoint=False)[:, None]
                                for i, n in enumerate(counts_float)] + [points[-1:]])
        # Sparse waypoints must not dilute a sharp instantaneous corner into
        # a small curvature. Check at the same bounded spatial resolution.
        dense_segments = np.diff(dense, axis=0)
        dense_lengths = np.linalg.norm(dense_segments, axis=1)
        dense_headings = np.arctan2(dense_segments[:, 1], dense_segments[:, 0])
        curvature = np.abs(np.diff(dense_headings)) / ((dense_lengths[:-1] + dense_lengths[1:]) / 2)
        _require((curvature <= p.max_curvature_per_m).all()
                 and (curvature * accelerated_speed ** 2 <= p.max_lateral_acceleration_mps2).all(), "path_turning_bound")
        # Disk bounds the entire body, margins and localization error. Inflating
        # by half a sample gap covers continuous interpolated segments. Half a
        # cell diagonal includes every grid cell touched by that swept volume.
        age_motion_bound = speed * ego_age + 0.5 * p.acceleration_during_delay_mps2 * ego_age ** 2
        radius = (p.body_radius_m + p.lateral_margin_m + scene.localization_error_m
                  + scene.motion_compensation_error_m + age_motion_bound + step / 2)
        x_max = grid.x_min_m + shape[1] * grid.resolution_m
        y_max = grid.y_min_m + shape[0] * grid.resolution_m
        _require((dense[:, 0] - radius >= grid.x_min_m).all()
                 and (dense[:, 0] + radius <= x_max).all()
                 and (dense[:, 1] - radius >= grid.y_min_m).all()
                 and (dense[:, 1] + radius <= y_max).all(), "swept_body_outside_observed_grid")
        collision_radius = radius + grid.resolution_m / math.sqrt(2)
        checked = np.zeros(shape, dtype=bool)
        for x, y in dense:
            x0 = max(0, math.floor((x - collision_radius - grid.x_min_m) / grid.resolution_m))
            x1 = min(shape[1], math.ceil((x + collision_radius - grid.x_min_m) / grid.resolution_m))
            y0 = max(0, math.floor((y - collision_radius - grid.y_min_m) / grid.resolution_m))
            y1 = min(shape[0], math.ceil((y + collision_radius - grid.y_min_m) / grid.resolution_m))
            xs = grid.x_min_m + (np.arange(x0, x1) + 0.5) * grid.resolution_m
            ys = grid.y_min_m + (np.arange(y0, y1) + 0.5) * grid.resolution_m
            checked[y0:y1, x0:x1] |= (xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2 <= collision_radius ** 2
        _require(not np.any(checked & ~grid.observed), "unobserved_swept_cell")
        _require(not np.any(checked & grid.occupied), "occupied_swept_cell")
        _require(not np.any(checked & grid.object_or_unknown), "object_or_anomaly_veto")
        _require(not np.any(checked & (grid.road_probability < p.min_road_probability)), "semantic_disagreement_or_nonroad")
        _require(len(scene.tracks) <= p.max_tracks, "track_budget_exceeded")
        ids = set()
        for track in scene.tracks:
            _require(isinstance(track, Track) and isinstance(track.track_id, str)
                     and track.track_id and track.track_id not in ids, "invalid_track_identity")
            ids.add(track.track_id)
            _require(_number(track.anchor_time_s) and track.anchor_time_s == state.timestamp, "track_ego_anchor_mismatch")
            _require(all(_number(getattr(track, key)) for key in (
                "x_m", "y_m", "vx_mps", "vy_mps", "radius_m", "position_error_m",
                "velocity_error_mps", "acceleration_bound_mps2")), "invalid_track")
            _require(min(track.radius_m, track.position_error_m, track.velocity_error_mps,
                         track.acceleration_bound_mps2) >= 0, "invalid_track_bounds")
            start = np.array([track.x_m, track.y_m])
            travel = np.array([track.vx_mps, track.vy_mps]) * track_window
            length_sq = float(travel @ travel)
            fraction = np.zeros(len(dense)) if length_sq < 1e-12 else np.clip((dense - start) @ travel / length_sq, 0, 1)
            separation = np.linalg.norm(dense - (start + fraction[:, None] * travel), axis=1)
            bound = (radius + track.radius_m + track.position_error_m
                     + track.velocity_error_mps * track_window
                     + 0.5 * track.acceleration_bound_mps2 * track_window ** 2)
            _require(np.isfinite(separation).all() and _number(bound), "invalid_track_prediction")
            _require((separation > bound).all(), "reachable_dynamic_obstacle")
        return stopping, int(checked.sum())
