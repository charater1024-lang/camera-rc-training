"""Session-local measurements against a synthetic reference centreline.

This tracker measures simulator behaviour, not driving skill, lane accuracy or
real-world safety. Only autonomous samples contribute to cumulative statistics;
the small recent-history buffer also retains manual driving for display.
"""
from __future__ import annotations

from collections import deque
import math


_DEFINITIONS = {
    "scope": "현재 합성 시뮬레이터 세션의 관측값이며 실차 성능·안전 평가나 종합 점수가 아닙니다.",
    "reference": "월드 기준 경로의 중심선이며 실제 차선 정답이 아닙니다. 가장 가까운 선분에 투영합니다.",
    "cross_track_error_m": "기준 폴리라인의 가장 가까운 점까지 거리입니다. 시작·끝 바깥에서는 끝점까지의 거리도 포함합니다.",
    "heading_error_deg": "가장 가까운 경로 선분과 차량 진행각의 차이를 -180~180도로 감싼 값입니다. MAE에는 절댓값을 사용합니다.",
    "autonomous": "자율 모드에서 완료한 단계만 누적합니다. 수동 모드의 시간·거리·사건은 자율 통계에서 제외합니다.",
    "session": "전체·수동 단계 수와 실행된 모드를 세션 전체에 누적합니다. 자율·수동 표본이 모두 있으면 혼합 세션이며 기록 그래프가 만료돼도 유지됩니다.",
    "time_weighting": "RMSE·MAE와 안전 통과율은 시뮬레이션 단계의 dt로 시간 가중합니다. 표본이 없으면 null입니다.",
    "mean_speed_kmh": "자율 모드에서 적분한 실제 이동 거리 / 자율 시뮬레이션 시간 × 3.6입니다.",
    "speed_error_kmh": "단계 종료 속도와 가감속 램프를 적용한 실제 제어 목표의 차이입니다. 정지·거부 때의 목표는 0입니다.",
    "manual_target": "수동 조작에는 폐루프 속도 제어 목표가 없습니다. 전달받은 수동 목표 0은 속도 추종 평가에 사용하지 않습니다.",
    "safety_eligible_pct": "전체 자율 시간 중 독립 안전 계층이 제안을 허용한 시간 비율입니다. 초기 준비 시간도 분모에 포함합니다.",
    "safety_stop_s": "제안이 거부된 자율 시간입니다. WARMUP_STOP 초기 준비 시간은 제외합니다.",
    "stopped_time_s": "자율 모드의 단계 종료 속도가 0.02 m/s 이하인 시뮬레이션 시간입니다.",
    "current": "가속도와 조향각 변화율은 이전 단계 종료 값에서 현재 값으로의 차분 / dt입니다. 초기 속도와 조향각은 0입니다.",
    "history": "전체 모드의 최근 120 시뮬레이션 초, 최대 240점입니다. 기본 0.5초 간격과 운전 모드가 바뀌는 즉시 단계 종료 값을 보관합니다.",
}


class PerformanceTracker:
    """Bounded history plus cumulative autonomous statistics for one run."""

    HISTORY_SECONDS = 120.0
    HISTORY_MAX_POINTS = 240
    HISTORY_INTERVAL_S = .5

    def __init__(self, reference_points):
        points = [tuple(map(float, point)) for point in reference_points]
        if len(points) < 2 or any(len(point) != 2 or not all(map(math.isfinite, point)) for point in points):
            raise ValueError("Reference must contain finite (x, y) points")
        self._segments = []
        self._length_m = 0.
        for a, b in zip(points, points[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length_squared = dx*dx + dy*dy
            if length_squared == 0.:
                continue
            length = math.sqrt(length_squared)
            self._segments.append((a, dx, dy, length_squared, length, self._length_m, math.atan2(dy, dx)))
            self._length_m += length
        if not self._segments:
            raise ValueError("Reference must contain a nonzero length segment")
        self._samples = 0
        self._total_samples = self._manual_samples = 0
        self._modes_seen = set()
        self._duration_s = self._distance_m = 0.
        self._cross_square_time = self._heading_absolute_time = 0.
        self._speed_square_time = self._speed_absolute_time = 0.
        self._cross_max_m = self._peak_speed_kmh = None
        self._eligible_s = self._safety_stop_s = self._stopped_time_s = 0.
        self._interventions = self._collision_events = self._offroad_events = 0
        self._previous_speed_mps = self._previous_steering_rad = 0.
        self._current = {"applied_target_kmh": None, "acceleration_mps2": None, "steering_rate_deg_s": None}
        self._history = deque(maxlen=self.HISTORY_MAX_POINTS)
        self._last_history_s = None
        self._last_history_mode = None

    def _project(self, position, yaw_rad):
        x, y = position
        nearest_squared = math.inf
        progress = heading = 0.
        for a, dx, dy, squared, length, start, direction in self._segments:
            fraction = min(1., max(0., ((x-a[0])*dx + (y-a[1])*dy)/squared))
            error_squared = (x-a[0]-fraction*dx)**2 + (y-a[1]-fraction*dy)**2
            # Stable first-segment selection resolves exact ties at vertices.
            if error_squared < nearest_squared:
                nearest_squared = error_squared
                progress = min(self._length_m, max(0., start+fraction*length))
                heading = math.degrees(math.atan2(math.sin(yaw_rad-direction), math.cos(yaw_rad-direction)))
        return progress, math.sqrt(nearest_squared), heading

    def sample(self, *, time_s, dt_s, mode, position, yaw_rad, speed_mps,
               applied_target_mps, steering_rad, distance_delta_m, safety_mode,
               safety_eligible, intervention_delta, collision_delta, offroad_delta):
        """Record one completed simulator step, using its duration as weight."""
        if not math.isfinite(dt_s) or dt_s <= 0.:
            raise ValueError("Sample duration must be finite and positive")
        _, cross_track, heading = self._project(position, yaw_rad)
        speed_kmh, target_kmh = speed_mps*3.6, applied_target_mps*3.6
        self._total_samples += 1
        self._manual_samples += int(mode == "manual")
        self._modes_seen.add(mode)
        self._current = {
            "applied_target_kmh": target_kmh,
            "acceleration_mps2": (speed_mps-self._previous_speed_mps)/dt_s,
            "steering_rate_deg_s": math.degrees(steering_rad-self._previous_steering_rad)/dt_s,
        }
        self._previous_speed_mps, self._previous_steering_rad = speed_mps, steering_rad
        if mode == "autonomous":
            self._samples += 1
            self._duration_s += dt_s
            self._distance_m += distance_delta_m
            self._cross_square_time += cross_track*cross_track*dt_s
            self._heading_absolute_time += abs(heading)*dt_s
            speed_error = speed_kmh-target_kmh
            self._speed_square_time += speed_error*speed_error*dt_s
            self._speed_absolute_time += abs(speed_error)*dt_s
            self._cross_max_m = cross_track if self._cross_max_m is None else max(self._cross_max_m, cross_track)
            self._peak_speed_kmh = speed_kmh if self._peak_speed_kmh is None else max(self._peak_speed_kmh, speed_kmh)
            if safety_eligible:
                self._eligible_s += dt_s
            elif safety_mode != "WARMUP_STOP":
                self._safety_stop_s += dt_s
            if speed_mps <= .02:
                self._stopped_time_s += dt_s
            self._interventions += intervention_delta
            self._collision_events += collision_delta
            self._offroad_events += offroad_delta
        while self._history and self._history[0]["time_s"] < time_s-self.HISTORY_SECONDS:
            self._history.popleft()
        if (self._last_history_s is None or mode != self._last_history_mode
                or time_s-self._last_history_s >= self.HISTORY_INTERVAL_S-1e-9):
            self._history.append({"time_s": time_s, "mode": mode, "speed_kmh": speed_kmh,
                                  "target_kmh": target_kmh, "cross_track_error_m": cross_track})
            self._last_history_s = time_s
            self._last_history_mode = mode

    def snapshot(self, position, yaw_rad):
        """Pure read: route values reflect the supplied pose, without sampling."""
        progress, cross_track, heading = self._project(position, yaw_rad)
        duration = self._duration_s
        return {
            "scope": "synthetic_session_only",
            "reference": "world_reference_centerline_not_lane_ground_truth",
            "session": {"total_samples": self._total_samples, "manual_samples": self._manual_samples,
                        "mixed_mode_session": {"autonomous", "manual"} <= self._modes_seen,
                        "modes_seen": sorted(self._modes_seen)},
            "route": {"progress_pct": 100.*progress/self._length_m,
                      "progress_m": progress, "length_m": self._length_m,
                      "remaining_m": max(0., self._length_m-progress),
                      "current_cross_track_error_m": cross_track, "current_heading_error_deg": heading},
            "autonomous": {
                "samples": self._samples, "duration_s": duration, "distance_m": self._distance_m,
                "cross_track_rmse_m": math.sqrt(self._cross_square_time/duration) if duration else None,
                "cross_track_max_m": self._cross_max_m,
                "heading_mae_deg": self._heading_absolute_time/duration if duration else None,
                "speed_rmse_kmh": math.sqrt(self._speed_square_time/duration) if duration else None,
                "speed_mae_kmh": self._speed_absolute_time/duration if duration else None,
                "mean_speed_kmh": self._distance_m/duration*3.6 if duration else None,
                "peak_speed_kmh": self._peak_speed_kmh,
                "safety_eligible_pct": 100.*self._eligible_s/duration if duration else None,
                "safety_stop_s": self._safety_stop_s, "safety_interventions": self._interventions,
                "collision_events": self._collision_events, "offroad_events": self._offroad_events,
                "stopped_time_s": self._stopped_time_s,
            },
            "current": self._current.copy(),
            "history": [point.copy() for point in self._history],
            "definitions": _DEFINITIONS.copy(),
        }
