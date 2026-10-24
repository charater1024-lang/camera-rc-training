/* Local, offline-only viewer. The server/engine owns time, safety and metrics. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const number = (value) => typeof value === "number" && Number.isFinite(value);
  const format = (value, digits = 0) => number(value) ? value.toLocaleString("ko-KR", {minimumFractionDigits: digits, maximumFractionDigits: digits}) : "—";
  const text = (id, value) => { $(id).textContent = value; };
  const keyMap = {KeyW: "forward", ArrowUp: "forward", KeyA: "left", ArrowLeft: "left", KeyS: "brake", ArrowDown: "brake", Space: "brake", KeyD: "right", ArrowRight: "right"};
  const keys = new Set();
  const pointerKeys = new Map();
  let state = null, connected = false, actionPending = false, followVehicle = false, mapZoom = 1;
  let scenarioSignature = "", lastSession = null, lastAnnounced = "", eventsSignature = "", actionError = "";
  let actionGeneration = 0;
  let controlsPending = false, controlsDirty = false, lastControlError = false;
  let desiredControls = {steering: 0, throttle: 0, brake: 1};
  let selectedScenarioDirty = false, selectedSeedDirty = false;
  let speedDraftDirty = false;
  let objectOverlayEnabled = true;
  const canvas = $("world-canvas");
  const context = canvas.getContext("2d");
  const images = [{id: "camera-image", placeholder: "camera-placeholder", path: "/frames/camera.png"}, {id: "segmentation-image", placeholder: "segmentation-placeholder", path: "/frames/segmentation.png"}];
  const reasons = {
    wrong_way_veto: "자차 차로의 진행 방향과 반대로 향하는 제안을 거부했습니다.",
    traffic_speed_limit: "교통 상호작용 코스의 가정 속도 상한",
    traffic_signal_stop: "보행자 횡단 신호에 맞춰 정지선 앞에서 제동합니다.",
    initial_observation_required: "첫 관측을 기다리고 있습니다. 시작 또는 한 단계 실행으로 확인하세요.",
    supplied_offline_evidence_passed: "제공된 오프라인 증거가 검사를 통과했습니다.",
    fresh_healthy_cycles_required: "새로운 정상 관측이 연속해서 들어오는지 확인하고 있습니다.",
    explicit_stopped_reset_required: "정지 상태와 새 정상 관측을 확인한 후 명시적인 잠금 해제가 필요합니다.",
    occupied_swept_cell: "차량이 지나갈 영역에 점유된 격자가 있습니다.",
    unobserved_swept_cell: "차량이 지나갈 영역에 관측되지 않은 공간이 있습니다.",
    object_or_anomaly_veto: "객체 또는 미지 영역이 경로를 가로막습니다.",
    reachable_dynamic_obstacle: "이동 장애물이 차량 경로에 도달할 수 있습니다.",
    semantic_disagreement_or_nonroad: "경로의 도로 인식 근거가 부족하거나 비도로 영역이 있습니다.",
    insufficient_stopping_horizon: "제동에 필요한 거리보다 확인된 경로가 짧습니다.",
    stale_or_future_observation: "관측 시각이 오래되었거나 미래 시각입니다.",
    module_unavailable: "필수 인식 모듈의 정상 상태가 확인되지 않았습니다.",
    outside_odd: "허용된 운행 조건 또는 구역을 벗어났습니다.",
    metric_scale_unverified: "미터 단위 거리의 근거가 확인되지 않았습니다.",
    invalid_or_stop_proposal: "정지 또는 유효하지 않은 경로 제안입니다.",
    lane_boundary_veto: "가정한 자차 차로 경계를 차체가 침범할 수 있어 주행 제안을 거부했습니다."
  };
  const scenarioLabels = {
    clear: ["장애물 없는 폐쇄 코스", "넓은 도로를 따라 이동하고 목표 지점에서 정지합니다."],
    curved: ["곡선 도로", "조향이 차량 자세와 다음 카메라 관측을 바꾸는 과정을 확인합니다."],
    static_obstacle: ["정지 장애물", "도로를 막는 장애물 앞에서 자율 모드의 정지 판단을 확인합니다."],
    crossing_pedestrian: ["횡단 보행자", "움직이는 합성 보행자가 경로에 진입할 때 추적 근거로 정지합니다."],
    night: ["야간 장애물", "어두운 합성 영상입니다. 정답 기반 거리 관측은 여전히 제공됩니다."],
    fog: ["제한된 가시거리", "관측 거리가 정지에 필요한 경로보다 짧은 상황을 확인합니다."],
    sensor_freeze: ["카메라 프레임 동결", "시뮬레이션 3초 후 노출 시각과 프레임 식별자가 멈춥니다."],
    sensor_dropout: ["카메라 관측 손실", "시뮬레이션 3초 후 카메라 증거가 사라집니다."],
    braking_degraded: ["제동 성능 저하", "제동 가능 상태가 실패하고 합성 감속 성능도 낮아집니다."],
    seoul_day: ["서울 성수 · 주간 도심", "실제 OSM 도로·건물 지도 위에서 합성 차량·보행자·콘과 주행합니다."],
    seoul_night: ["서울 성수 · 야간 도심", "실제 지도와 어두운 합성 카메라를 사용합니다. 거리·추적은 합성 정답입니다."],
    seoul_hazard: ["서울 성수 · 복합 장애물", "도심 지도에 배치된 합성 교통과 복합 위험 상황의 정지·거부를 확인합니다."],
    busan_day: ["부산 · 해안과 언덕", "부산의 해안·구릉 특성을 반영한 합성 코스입니다. 실제 도로 지도·측량 자료는 아닙니다."],
    gwangju_day: ["광주 · 도심 평지", "광주의 도심·평지 특성을 반영한 합성 코스입니다. 실제 도로 지도·측량 자료는 아닙니다."],
    suncheon_day: ["전남 순천 · 습지와 평야", "순천의 습지·평야 특성을 반영한 합성 코스입니다. 실제 도로 지도·측량 자료는 아닙니다."],
    gangwon_day: ["강원 · 산지 굽잇길", "강원 산지의 고도 변화·굽잇길 특성을 반영한 합성 코스입니다. 실제 도로 지도·측량 자료는 아닙니다."]
  };
  const speedLimitLabels = {
    user_limit: "사용자 상한", user_target: "사용자 지정 속도", requested_speed: "설정 속도", maximum_speed: "시뮬레이터 최대 속도",
    road_speed_limit: "도로 제한 속도", road_limit: "도로 제한 속도", curvature: "도로 곡률", curve_limit: "도로 곡률",
    visibility: "관측·제동 거리", visibility_limit: "가시거리", stopping_distance: "제동 거리", grade: "도로 경사", grade_limit: "도로 경사",
    traffic_speed_limit: "교통 상호작용 코스 상한",
    upcoming_curve: "앞쪽 곡선", regional_advisory: "합성 코스 권장 속도", goal_distance: "목표 정지 거리",
    goal: "목표 지점 접근", goal_approach: "목표 지점 접근", safety_stop: "안전 판단에 따른 정지", safety_veto: "안전 거부", zero_target: "정지 설정"
  };
  function scenarioTitle(item) {
    const id = item?.id;
    return ["busan_day", "gwangju_day", "suncheon_day", "gangwon_day"].includes(id)
      ? item.title || scenarioLabels[id]?.[0] || id
      : scenarioLabels[id]?.[0] || item?.title || id || "장면 대기";
  }
  function showError(message) { text("error-banner", String(message)); $("error-banner").hidden = !message; }
  function announce(message) { if (message !== lastAnnounced) { text("announcer", message); lastAnnounced = message; } }
  async function request(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      const options = {cache: "no-store", credentials: "same-origin", signal: controller.signal};
      if (body !== undefined) Object.assign(options, {method: "POST", headers: {"Content-Type": "application/json", "X-Simulator-Client": "1"}, body: JSON.stringify(body)});
      const response = await fetch(path, options);
      let result;
      try { result = await response.json(); } catch (_) { throw new Error(`서버 응답을 읽을 수 없습니다 (HTTP ${response.status}).`); }
      if (!response.ok) throw new Error(result.error || result.message || `HTTP ${response.status}`);
      return result;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답 시간이 초과되었습니다. 연결과 일시정지 상태를 확인하세요.");
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function updateEnabled() {
    const ready = connected && state && !actionPending;
    for (const id of ["scenario", "seed", "reset", "mode-auto", "mode-manual", "reset-safety"]) $(id).disabled = !ready;
    for (const id of ["speed-mode", "speed-target", "speed-range", "apply-speed"]) $(id).disabled = !ready || !state.speed_control;
    $("start-pause").disabled = !ready || Boolean(state && state.finished);
    $("step").disabled = !ready || Boolean(state && (state.running || state.finished));
    for (const button of document.querySelectorAll(".drive-key")) button.disabled = !ready || state.mode !== "manual" || !state.running || state.finished;
    $("export-report").setAttribute("aria-disabled", connected ? "false" : "true");
  }
  function heldDirections() {
    const directions = new Set([...keys].map((key) => keyMap[key]).filter(Boolean));
    for (const value of pointerKeys.values()) directions.add(value);
    return directions;
  }
  function drawHeld() {
    const directions = heldDirections();
    for (const button of document.querySelectorAll(".drive-key")) button.classList.toggle("active", directions.has(button.dataset.drive));
  }
  function deriveControls() {
    const directions = heldDirections();
    const active = state && state.mode === "manual" && state.running && !state.finished && connected && !document.hidden;
    const forward = active && directions.has("forward") && !directions.has("brake");
    desiredControls = {steering: active ? Number(directions.has("left")) - Number(directions.has("right")) : 0, throttle: forward ? 1 : 0, brake: forward ? 0 : 1};
    drawHeld();
  }
  async function flushControls() {
    controlsDirty = true;
    if (controlsPending || !connected) return;
    controlsPending = true;
    try {
      while (controlsDirty && connected) {
        controlsDirty = false;
        const command = {...desiredControls};
        await request("/api/action", {action: "controls", ...command});
        lastControlError = false;
      }
    } catch (error) {
      keys.clear(); pointerKeys.clear(); desiredControls = {steering: 0, throttle: 0, brake: 1}; drawHeld();
      if (!lastControlError) showError(`수동 명령 전달 실패: ${error.message} 서버는 조작 갱신이 끊기면 제동합니다.`);
      lastControlError = true;
    } finally { controlsPending = false; }
  }
  function releaseControls(send = true) {
    keys.clear(); pointerKeys.clear(); desiredControls = {steering: 0, throttle: 0, brake: 1}; drawHeld();
    if (send && connected) void flushControls();
  }
  async function act(action, extra = {}) {
    if (actionPending || !connected) return;
    if (["pause", "reset", "mode", "reset_safety"].includes(action)) releaseControls();
    actionGeneration += 1;
    actionPending = true; updateEnabled();
    try {
      const result = await request("/api/action", {action, ...extra});
      if (action === "reset") { for (const image of images) image.version = null; }
      if (action === "speed") { speedDraftDirty = false; setSpeedInputError(""); }
      if (result && result.vehicle && result.world) render(result);
      if (action === "reset") { selectedScenarioDirty = false; selectedSeedDirty = false; }
      actionError = ""; showError("");
    } catch (error) { actionError = error.message; showError(actionError); announce(`조작 실패: ${error.message}`); }
    finally { actionPending = false; updateEnabled(); }
  }
  function fillScenarios(list) {
    if (!Array.isArray(list)) return;
    const signature = JSON.stringify(list);
    if (signature === scenarioSignature) return;
    scenarioSignature = signature;
    const previous = $("scenario").value;
    $("scenario").replaceChildren();
    for (const entry of list) {
      const item = typeof entry === "string" ? {id: entry, title: entry} : entry;
      if (!item || typeof item.id !== "string") continue;
      const option = document.createElement("option"); option.value = item.id; option.textContent = scenarioTitle(item);
      option.dataset.description = ["busan_day", "gwangju_day", "suncheon_day", "gangwon_day"].includes(item.id)
        ? item.description || scenarioLabels[item.id]?.[1]
        : scenarioLabels[item.id]?.[1] || item.description || "합성 장면의 도로와 장애물을 관찰합니다.";
      $("scenario").append(option);
    }
    if ([...$("scenario").options].some((option) => option.value === previous)) $("scenario").value = previous;
    describeScenario();
  }
  function describeScenario() { text("scenario-description", $("scenario").selectedOptions[0]?.dataset.description || "시나리오 정보를 기다리고 있습니다."); }
  function setSpeedInputError(message) {
    text("speed-input-error", message); $("speed-input-error").hidden = !message;
    $("speed-target").setAttribute("aria-invalid", String(Boolean(message)));
  }
  function describeSpeedMode() {
    const automatic = $("speed-mode").value === "automatic";
    text("speed-target-label", automatic ? "속도 상한" : "요청 속도");
    text("speed-mode-help", automatic ? "도로·곡률·가시거리에 따라 설정한 상한 이내에서 자동 조절합니다." : "원하는 속도를 요청합니다. 자율 주행에서는 안전 판단·곡률·가시거리 제한을 함께 적용합니다.");
  }
  function markSpeedDraft() {
    speedDraftDirty = true; setSpeedInputError("");
    text("speed-draft-status", "변경사항 미적용 · 적용 버튼을 누르세요.");
  }
  function renderSpeedControl(control) {
    if (!control) { text("speed-draft-status", "엔진에서 속도 설정을 제공하지 않습니다."); return; }
    const maximum = number(control.max_speed_kmh) ? Math.max(0, Math.min(150, control.max_speed_kmh)) : 150;
    for (const id of ["speed-target", "speed-range"]) $(id).max = String(maximum);
    if (!speedDraftDirty) {
      $("speed-mode").value = control.mode === "user" ? "user" : "automatic";
      if (number(control.target_speed_kmh)) for (const id of ["speed-target", "speed-range"]) $(id).value = String(control.target_speed_kmh);
      describeSpeedMode(); text("speed-draft-status", "설정 적용됨 · 주행 중에도 변경할 수 있습니다.");
    }
    const automatic = control.mode !== "user";
    text("applied-speed", `${automatic ? "자동" : "사용자"} · ${format(control.target_speed_kmh, 1)} km/h${automatic ? " 상한" : " 요청"}`);
    text("effective-speed", `${format(control.effective_target_kmh, 1)} km/h`);
    const limits = Array.isArray(control.limiting_reasons) ? control.limiting_reasons : [];
    text("speed-limits", limits.length ? `현재 제한 · ${limits.map((reason) => speedLimitLabels[reason] || String(reason)).join(" · ")}` : "설정 속도 이내에서 주행합니다. 안전 거부 시 정지합니다.");
  }
  function renderTerrain(region, vehicle, progress) {
    const profile = region?.terrain?.elevation_profile;
    const points = Array.isArray(profile) ? profile.filter((point) => number(point.distance_m) && number(point.elevation_m)).sort((a,b) => a.distance_m - b.distance_m) : [];
    const visible = points.length > 1 && points.at(-1).distance_m > points[0].distance_m;
    $("terrain-panel").hidden = !visible;
    $("terrain-telemetry").hidden = !visible;
    if (!visible) return;
    const first = points[0].distance_m, last = points.at(-1).distance_m;
    const minimum = Math.min(...points.map((point) => point.elevation_m)), maximum = Math.max(...points.map((point) => point.elevation_m));
    const x = (distance) => 8 + (distance - first) / (last - first) * 584;
    const y = (elevation) => 73 - (elevation - minimum) / Math.max(10, maximum - minimum) * 61;
    const line = points.map((point,index) => `${index ? "L" : "M"}${x(point.distance_m).toFixed(2)},${y(point.elevation_m).toFixed(2)}`).join(" ");
    $("terrain-line").setAttribute("d", line);
    $("terrain-area").setAttribute("d", `${line} L592,81 L8,81 Z`);
    text("terrain-elevation-range", `합성 고도 ${format(minimum)}–${format(maximum)} m`);
    text("terrain-distance", `경로 ${format(last - first)} m`);
    const current = `${format(vehicle.elevation_m, 1)} m / ${number(vehicle.grade_percent) && vehicle.grade_percent > 0 ? "+" : ""}${format(vehicle.grade_percent, 1)}%`;
    text("terrain-current", `현재 ${current}`); text("vehicle-terrain", current);
    $("terrain-profile").setAttribute("aria-label", `경로 거리 ${format(last-first)}미터의 합성 고도 단면, 최저 ${format(minimum)}미터, 최고 ${format(maximum)}미터. 현재 ${current}`);
    const position = $("terrain-position"), positioned = number(progress) && number(vehicle.elevation_m);
    position.toggleAttribute("hidden", !positioned);
    if (positioned) { position.setAttribute("cx", x(Math.max(first,Math.min(last,progress))).toFixed(2)); position.setAttribute("cy", y(vehicle.elevation_m).toFixed(2)); }
  }
  function renderPerformanceCharts(history) {
    let points = Array.isArray(history) ? history.filter((point) => point && number(point.time_s)).slice(-240) : [];
    points = points.slice().sort((a,b) => a.time_s - b.time_s);
    const lastTime = points.at(-1)?.time_s;
    if (number(lastTime)) points = points.filter((point) => point.time_s >= lastTime - 120);
    const firstTime = points[0]?.time_s;
    const windowLength = number(firstTime) && number(lastTime) ? Math.max(1,lastTime-firstTime) : 1;
    const x = (time) => 42 + (time-firstTime) / windowLength * 546;
    const autonomous = points.filter((point) => point.mode === "autonomous");
    const speeds = autonomous.flatMap((point) => [point.speed_kmh,point.target_kmh]).filter(number);
    const errors = autonomous.map((point) => point.cross_track_error_m).filter(number);
    const speedMaximum = speeds.length ? Math.max(10,Math.ceil(Math.max(...speeds)/25)*25) : 150;
    const errorMaximum = errors.length ? Math.max(1,Math.ceil(Math.max(...errors.map(Math.abs))*2)/2) : 1;
    const speedY = (value) => 135 - value / speedMaximum * 120;
    const errorY = (value) => 135 - value / errorMaximum * 120;
    function series(lineId, dotId, key, y) {
      let drawing = false, line = "", latest = null;
      for (const point of points) {
        if (point.mode !== "autonomous" || !number(point[key])) { drawing = false; continue; }
        const px = x(point.time_s), py = y(point[key]);
        line += `${drawing ? " L" : " M"}${px.toFixed(2)},${py.toFixed(2)}`;
        drawing = true; latest = {x: px,y: py,value: point[key],time: point.time_s};
      }
      $(lineId).setAttribute("d", line.trim()); $(dotId).toggleAttribute("hidden", !latest);
      if (latest) { $(dotId).setAttribute("cx", latest.x.toFixed(2)); $(dotId).setAttribute("cy", latest.y.toFixed(2)); }
      return latest;
    }
    const actual = series("performance-speed-line", "performance-speed-dot", "speed_kmh", speedY);
    const target = series("performance-target-line", "performance-target-dot", "target_kmh", speedY);
    const path = series("performance-path-line", "performance-path-dot", "cross_track_error_m", errorY);
    $("performance-speed-empty").hidden = Boolean(actual || target); $("performance-path-empty").hidden = Boolean(path);
    text("performance-speed-top", speeds.length ? format(speedMaximum) : "—");
    text("performance-path-top", errors.length ? format(errorMaximum, 1) : "—");
    text("performance-path-middle", errors.length ? format(errorMaximum/2, 1) : "—");
    text("performance-path-bottom", "0");
    for (const kind of ["speed", "path"]) {
      text(`performance-${kind}-start`, number(firstTime) ? `${format(firstTime, 1)} s` : "—");
      text(`performance-${kind}-end`, number(lastTime) ? `${format(lastTime, 1)} s` : "—");
    }
    $("performance-speed-chart").setAttribute("aria-label", actual || target ? `최근 자율 주행 속도. 마지막 실제 ${format(actual?.value, 1)} km/h, 제어 목표 ${format(target?.value, 1)} km/h. 수동 구간은 표시하지 않습니다.` : "자율 주행 속도 시계열 미수집");
    $("performance-path-chart").setAttribute("aria-label", path ? `최근 자율 주행 경로 오차. 마지막 ${format(path.value, 2)} m. 수동 구간은 표시하지 않습니다.` : "자율 주행 경로 오차 시계열 미수집");
  }
  function renderPerformance(performance, manual) {
    const available = performance && typeof performance === "object";
    const route = available && performance.route || {}, autonomous = available && performance.autonomous || {}, current = available && performance.current || {};
    const samples = number(autonomous.samples) ? autonomous.samples : 0;
    text("performance-status", !available ? "미수집" : samples > 0 ? `자율 ${format(samples)} 표본` : "자율 표본 없음");
    $("performance-status").className = "badge neutral";
    const modeNote = !available ? "엔진에서 성능 지표를 제공하지 않습니다." : manual ? `수동 조작 중 · 자율 누적은 멈춰 있습니다.${samples ? " 이전 자율 구간의 지표를 유지합니다." : " 자율 주행 표본은 아직 없습니다."}` : samples ? "초기화 이후 자율 주행 단계의 누적값입니다. 현재·경로 진행 값은 전체 운전 모드를 반영합니다." : "자율 주행 표본이 아직 없습니다. 시작 또는 한 단계 실행 후 누적합니다.";
    text("performance-mode-note", `${modeNote}${available && performance.session?.mixed_mode_session === true ? " 혼합 실행 · 수동 구간 제외." : ""}`);
    const progress = number(route.progress_pct) ? Math.max(0,Math.min(100,route.progress_pct)) : null;
    $("performance-progress").value = progress ?? 0;
    $("performance-progress").setAttribute("aria-valuetext", progress === null ? "미수집" : `${format(progress, 1)}%, 목표 도달 판정과 별개`);
    text("performance-progress-value", progress === null ? "—" : `${format(progress, 1)}%`);
    text("performance-progress-distance", `${format(route.progress_m, 1)} / ${format(route.length_m, 1)} m`);
    text("performance-progress-remaining", `남은 경로 ${format(route.remaining_m, 1)} m`);
    for (const [id,key,digits] of [["path-rmse","cross_track_rmse_m",2],["speed-rmse","speed_rmse_kmh",1],["heading-mae","heading_mae_deg",1],["duration","duration_s",1],["safety-eligible","safety_eligible_pct",1],["mean-speed","mean_speed_kmh",1]]) text(`performance-${id}`, format(autonomous[key],digits));
    text("performance-path-max", `최대 ${format(autonomous.cross_track_max_m, 2)} m`);
    text("performance-speed-mae", `평균 절대 오차 ${format(autonomous.speed_mae_kmh, 1)} km/h`);
    text("performance-current-errors", `현재 경로 ${format(route.current_cross_track_error_m, 2)} m / 방향 ${format(route.current_heading_error_deg, 1)}°`);
    text("performance-distance", `주행 거리 ${format(autonomous.distance_m, 1)} m`);
    text("performance-safety-stop", `거부 시간 ${format(autonomous.safety_stop_s, 1)} s`);
    text("performance-peak-speed", `최고 ${format(autonomous.peak_speed_kmh, 1)} km/h`);
    text("performance-events", `자율 구간 · 충돌 ${format(autonomous.collision_events)} · 도로 이탈 ${format(autonomous.offroad_events)} · 개입 ${format(autonomous.safety_interventions)}`);
    text("performance-stopped", `정지 시간 ${format(autonomous.stopped_time_s, 1)} s`);
    text("performance-current-motion", `현재 제어 목표 ${format(current.applied_target_kmh, 1)} km/h · 가속도 ${format(current.acceleration_mps2, 2)} m/s² · 조향 변화율 ${format(current.steering_rate_deg_s, 1)}°/s`);
    renderPerformanceCharts(available ? performance.history : []);
  }
  function assumedLaneBoundaries(worldLane) {
    if (!worldLane || worldLane.source !== "synthetic_assumed_lane_not_measured") return null;
    const boundaries = [worldLane.left_boundary, worldLane.right_boundary].map((boundary) => Array.isArray(boundary) ? boundary.filter((point) => Array.isArray(point) && number(point[0]) && number(point[1])) : []);
    return boundaries.every((boundary) => boundary.length > 1) ? boundaries : null;
  }
  function renderLane(lane, worldLane, manual) {
    const available = lane && typeof lane === "object";
    const current = available && lane.current || {}, autonomous = available && lane.autonomous || {};
    const samples = number(autonomous.samples) ? autonomous.samples : 0;
    const assumed = available && lane.source === "synthetic_assumed_lane_not_measured";
    text("lane-source", !available ? "차선 정보 미제공" : assumed ? `폭 ${format(lane.width_m, 1)} m 가정 중심 차로 · 실측 아님` : "차선 출처를 확인하세요");
    text("lane-inside-pct", samples > 0 ? format(autonomous.inside_pct, 1) : "—");
    text("lane-autonomous-events", `차선 이탈 ${format(autonomous.departure_events)}회 · 이탈 시간 ${format(autonomous.outside_time_s, 1)} s`);
    text("lane-autonomous-samples", !available ? "엔진에서 차선 지표를 제공하지 않습니다." : samples ? `자율 ${format(samples)} 표본 · ${format(autonomous.duration_s, 1)} s${manual ? " · 수동 중 누적 정지" : ""}` : "자율 표본 없음");
    text("lane-clearance", format(current.clearance_m, 2));
    text("lane-current-offset", `중심 편차 ${format(current.center_offset_m, 2)} m · 차체 최대 편차 ${format(current.max_body_offset_m, 2)} m`);
    const outside = typeof current.outside === "boolean" ? current.outside : null;
    text("lane-current-state", outside === null ? "미제공" : outside ? "차선 밖" : "차선 안");
    $("lane-current-state").className = `badge ${outside === null ? "neutral" : outside ? "caution" : "good"}`;
    $("lane-clearance").classList.toggle("lane-outside", outside === true);
    const training = available && lane.training;
    text("lane-training-status", !training ? "차선 학습 상태 미제공" : `차선 모델 ${training.model_loaded === true ? "로드됨" : training.status === "not_trained" ? "미학습" : "미로드"} · ${training.label_space || training.label_space_id || "라벨 공간 미제공"} · 합성 기하 감독 · 학습 실행 중지 · 모델 자동 교체 없음`);
    const guard = available && lane.guard;
    const guardAvailable = assumed && guard?.scope === "synthetic_path_guard" && typeof guard.allowed === "boolean";
    const awaitingCheck = guardAvailable && guard.checked === false;
    text("lane-guard-state", !guardAvailable ? "미제공" : manual ? "수동 · 미적용" : awaitingCheck ? "검사 대기" : guard.allowed ? "통과" : "거부");
    $("lane-guard-state").className = `badge ${!guardAvailable || manual || awaitingCheck ? "neutral" : guard.allowed ? "good" : "caution"}`;
    text("lane-guard-description", !guardAvailable ? "엔진에서 합성 경로 경계 검사 결과를 제공하지 않습니다." : manual ? "수동 조작에는 자율 경로 거부를 적용하지 않습니다. 현재 차체 상태만 관측합니다." : awaitingCheck ? "자율 제안이 아직 없어 경계 검사를 기다립니다." : !guard.allowed && ["lane_boundary_veto","wrong_way_veto"].includes(guard.reason) ? reasons[guard.reason] : guard.allowed ? "자율 제안의 가정 차선 경계 검사를 통과했습니다. 다른 안전 판단도 함께 적용합니다." : "가정 차선 경계 검사에서 자율 제안을 거부했습니다.");
    const mapped = assumedLaneBoundaries(worldLane) !== null;
    $("lane-map-legend").hidden = !mapped; $("lane-map-note").hidden = !mapped;
    $("lane-map-legend").classList.toggle("lane-outside", outside === true);
    text("lane-map-label", `가정 차선 ${format(worldLane?.width_m, 1)} m`);
  }
  function renderTraffic(world) {
    const layout=world.road_lanes, traffic=world.traffic;
    text("lane-assignment-status", layout?.source === "synthetic_lane_assignment_not_measured" ? `왕복 ${layout.lane_count}차로 가정 · 자차: 진행 방향 차로 · 반대 차로 진입/역방향 제안 거부 · 도로 폭과 차로 폭을 구분합니다.` : "차로별 진행 방향 정보 미제공");
    text("traffic-kind", traffic ? "신호·차간 거리 규칙" : "기존 단순 배경 교통");
    if (!traffic) {
      $("traffic-signal-status").className="";
      text("traffic-signal-status","차량·보행자 상호작용은 ‘교통 상호작용 · 왕복 2차로’ 시나리오에서 확인하세요.");
      text("traffic-pedestrian-status","");
      text("traffic-scope-note","기존 배경의 정지·짧은 왕복 움직임을 실제 도심 교통 검증으로 해석하지 않습니다.");
      return;
    }
    const signal=traffic.signal || {};
    text("traffic-signal-status",`차량 신호 ${signal.vehicles === "green" ? "진행" : "정지"} · 보행 신호 ${signal.pedestrians === "walk" ? "횡단" : "대기"} · 주기 ${format(signal.phase_s,1)} / ${format(signal.cycle_s)} s`);
    $("traffic-signal-status").className = signal.vehicles === "red" ? "traffic-red" : "traffic-green";
    text("traffic-pedestrian-status",traffic.pedestrian_waiting_for_vehicle ? "보행자: 횡단 구역의 차량이 빠져나가기를 기다립니다." : `보행자: ${{crossing:"횡단 중",waiting:"신호 대기",arrived:"횡단 완료"}[traffic.pedestrian_state] || "상태 미제공"} · 차량: 차로 유지·차간 거리·적신호 감속`);
    text("traffic-scope-note","작은 직선 교차부의 합성 규칙 테스트입니다. 기존 보수적 원형 안전 검사가 인접 차로 차량에도 정지할 수 있으며, 이를 완화해 통과시키지 않습니다.");
  }
  function renderObjects(view) {
    const available = view?.source === "synthetic_geometry_not_learned_detection";
    const missing = !available || view.status !== "available";
    const actors = !missing && Array.isArray(view.objects) ? view.objects.filter((item) => item && typeof item.id === "string" && number(item.distance_m) && number(item.speed_kmh)).slice().sort((a,b) => a.distance_m-b.distance_m) : [];
    const frozen = available && view.frame_frozen === true;
    text("object-source", !available ? "미제공" : "합성 정답 · 탐지기 미연결");
    text("object-status", !available ? "엔진에서 객체 관측을 제공하지 않습니다." : missing ? "카메라 프레임 손실 · 객체 표시를 중지했습니다. 별도의 합성 거리·안전 추적 상태는 안전 판단에서 확인하세요." : `${frozen ? "고정된 과거 프레임 · " : ""}관측 ${format(view.sampled_at_sim_time_s,1)} s · 경과 ${format(view.age_s,1)} s · 가정 범위 ${format(view.sensor_range_m)} m`);
    $("object-status").className = `object-status ${frozen || available && missing ? "object-stale" : ""}`;
    text("object-count", missing ? "—" : format(actors.length));
    text("object-camera-count", missing ? "—" : format(view.camera_box_count));
    text("object-moving-count", missing ? "—" : format(actors.filter((actor) => actor.speed_kmh > .1).length));
    text("object-route-count", missing ? "—" : format(actors.filter((actor) => actor.near_reference_route === true).length));
    $("object-rows").replaceChildren();
    const names = {vehicle:"차량",pedestrian:"보행자",cone:"콘",obstacle:"장애물"};
    const supplied = new Set(Array.isArray(view?.safety_track_ids) ? view.safety_track_ids : []);
    for (const actor of actors.slice(0,8)) {
      const row = document.createElement("tr");
      const values = [`${names[actor.kind] || "기타 객체"} · ${actor.id}`, `${format(actor.distance_m,1)} m · ${actor.relative_forward_m >= 0 ? "전방" : "후방"}`, `${format(actor.speed_kmh,1)} km/h`, actor.lane_id === "opposing_reverse" ? "반대 차로" : actor.lane_id === "ego_forward" ? "자차 방향 차로" : actor.near_reference_route === true ? "기준 경로 주변" : "기준 경로 바깥", supplied.has(actor.id) ? "합성 추적 제공" : "미제공"];
      for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
      if (actor.near_reference_route === true) row.className = "object-near-route";
      $("object-rows").append(row);
    }
    $("object-empty").hidden = actors.length > 0;
    text("object-empty", missing ? "사용 가능한 객체 표시가 없습니다." : "표시 범위에 관측된 객체가 없습니다. 주변에 객체가 없다는 보장은 아닙니다.");
    text("object-details", !available ? "" : missing ? "신뢰도 — · 학습 탐지 결과 없음" : `가까운 ${Math.min(8,actors.length)} / ${actors.length}개 표시 · 가림/범위 밖 ${format(view.hidden_or_out_of_range_count)}개 · 신뢰도 — (학습 탐지 결과 없음)`);
    const enabled = available && objectOverlayEnabled;
    $("object-overlay-toggle").disabled = !available;
    $("object-overlay-toggle").setAttribute("aria-pressed", String(enabled));
    text("object-overlay-toggle", enabled ? "객체 표시 켜짐" : "객체 표시 꺼짐");
    text("camera-display-label", enabled ? "합성 객체 투영 · 탐지 결과 아님" : "합성 추적 카메라 · 원본");
    $("camera-image").alt = enabled ? "합성 카메라와 가정한 객체의 투영 범위 표시, 학습 탐지 결과 아님" : "차량 뒤에서 바라보는 합성 추적 카메라 원본";
    const imagePath = enabled ? "/frames/objects.png" : "/frames/camera.png";
    if (images[0].path !== imagePath) { images[0].path = imagePath; images[0].version = null; }
    $("object-map-note").hidden = !available;
  }
  function updateImage(item, version) {
    item.nextVersion = version;
    if (item.loading || item.version === version) return;
    item.loading = true; item.version = version;
    const image = $(item.id);
    image.onload = () => { item.loading = false; image.hidden = false; $(item.placeholder).hidden = true; if (item.nextVersion !== item.version) updateImage(item, item.nextVersion); };
    image.onerror = () => { item.loading = false; image.hidden = true; $(item.placeholder).hidden = false; $(item.placeholder).querySelector("small").textContent = "프레임을 불러오지 못했습니다"; item.version = null; };
    image.src = `${item.path}?v=${encodeURIComponent(version)}`;
  }
  function updateEvents(events) {
    if (!Array.isArray(events)) events = [];
    const recent = events.slice(-12).reverse();
    const signature = JSON.stringify(recent);
    if (signature === eventsSignature) return;
    eventsSignature = signature; $("events").replaceChildren();
    if (!recent.length) { const item = document.createElement("li"); item.className = "empty-event"; item.textContent = "기록된 이벤트가 없습니다."; $("events").append(item); return; }
    for (const event of recent) {
      const item = document.createElement("li"), dot = document.createElement("span"), timestamp = document.createElement("time"), detail = document.createElement("span");
      dot.className = "event-dot"; detail.className = "event-detail";
      if (event && typeof event === "object") {
        const value = event.sim_time_s ?? event.time_s ?? event.t;
        timestamp.textContent = number(value) ? `${format(value, 1)}s` : (number(event.step) ? `#${event.step}` : "—");
        detail.textContent = event.message || event.reason || event.type || JSON.stringify(event);
      } else { timestamp.textContent = "—"; detail.textContent = String(event); }
      item.append(dot, timestamp, detail); $("events").append(item);
    }
  }
  let improvementSignature = "";
  function renderImprovements(progress) {
    const signature = JSON.stringify(progress || null);
    if (signature === improvementSignature) return;
    improvementSignature = signature;
    const rows = $("improvement-rows"); rows.replaceChildren();
    if (!progress || !Array.isArray(progress.items)) {
      text("improvement-count", "개선 현황 정보 없음");
      text("improvement-verification", "서버를 최신 코드로 다시 시작하면 개선 현황이 표시됩니다.");
      return;
    }
    text("improvement-count", `코드 완료 ${progress.implemented_items} / 진행 중 ${progress.partial_items} / 전체 ${progress.total_items}`);
    const verification = progress.automated_verification || {};
    text("improvement-verification", `자동 검사: ${verification.status === "passed" ? `${verification.python_passed || 0}개 Python · ${verification.ui_passed || 0}개 UI 통과` : "확인 중"} · 실차 검증 완료 ${progress.measured_validation_complete || 0}개 · ${progress.updated_at}`);
    text("improvement-platforms", (progress.platforms || []).map(p => `${p.title}: ${p.status}`).join(" · "));
    for (const item of progress.items) {
      const row = document.createElement("tr");
      for (const value of [item.title, item.status === "implemented" ? "코드 반영" : "진행 중", item.detail, item.remaining]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      rows.append(row);
    }
  }
  function render(snapshot) {
    if (!snapshot || typeof snapshot !== "object" || !snapshot.vehicle || !snapshot.world) throw new Error("시뮬레이터 상태 형식이 올바르지 않습니다.");
    const previous = state; state = snapshot;
    renderImprovements(snapshot.improvement_progress);
    if (previous && (previous.mode !== state.mode || previous.running && !state.running || previous.session_id !== state.session_id)) releaseControls();
    if (snapshot.session_id !== lastSession) { lastSession = snapshot.session_id; mapZoom = 1; for (const item of images) item.version = null; }
    fillScenarios(snapshot.scenarios);
    if (!selectedScenarioDirty && snapshot.scenario?.id) { $("scenario").value = snapshot.scenario.id; describeScenario(); }
    if (!selectedSeedDirty && number(snapshot.seed)) $("seed").value = snapshot.seed;
    const running = Boolean(snapshot.running), finished = Boolean(snapshot.finished);
    const runLabel = finished ? "실행 종료" : running ? "실행 중" : "일시정지";
    text("run-state", runLabel); $("run-state").className = `badge ${running ? "good" : finished ? "caution" : "neutral"}`;
    text("play-label", running ? "일시정지" : "시작"); text("play-icon", running ? "Ⅱ" : "▶");
    text("sim-time", number(snapshot.sim_time_s) ? `${format(snapshot.sim_time_s, 1)} s` : "—");
    text("step-count", `STEP ${format(snapshot.step)}`); text("world-scenario", scenarioTitle(snapshot.scenario));
    const manual = snapshot.mode === "manual";
    for (const [id, selected] of [["mode-auto", !manual], ["mode-manual", manual]]) { $(id).classList.toggle("selected", selected); $(id).setAttribute("aria-pressed", String(selected)); }
    $("manual-panel").hidden = !manual;
    $("safety-manual-note").hidden = !manual;
    const safety = snapshot.safety || {}, mode = safety.mode || "증거 대기";
    const unsafe = String(mode).includes("STOP") || String(mode).includes("FAULT");
    text("safety-mode", mode); $("safety-mode").className = `badge ${safety.proposal_eligible ? "good" : unsafe ? "caution" : "neutral"}`;
    const reasonList = Array.isArray(safety.reasons) ? safety.reasons : [];
    text("safety-reason", reasonList.length ? reasonList.map((reason) => reasons[reason] || String(reason)).join(" · ") : "판단 근거가 아직 없습니다.");
    if (safety.actuation_enabled !== false) showError("안전 계약 확인 필요: 실제 장치 출력 비활성 상태가 응답에서 확인되지 않았습니다.");
    text("session-label", `SESSION ${snapshot.session_id || "—"}`);
    const vehicle = snapshot.vehicle, command = snapshot.command || {};
    text("speed", number(vehicle.speed_mps) ? format(vehicle.speed_mps * 3.6, 1) : "—"); text("speed-mps", number(vehicle.speed_mps) ? `${format(vehicle.speed_mps, 2)} m/s` : "— m/s");
    renderSpeedControl(snapshot.speed_control);
    text("steering", number(vehicle.steering_rad) ? `${format(vehicle.steering_rad * 180 / Math.PI, 1)}°` : "—");
    for (const name of ["throttle", "brake"]) { const value = command[name]; $(`${name}-meter`).value = number(value) ? Math.min(1, Math.max(0, value)) : 0; text(`${name}-value`, number(value) ? `${format(value * 100)}%` : "—"); }
    text("command-reason", command.reason ? `명령 · ${command.reason}` : "명령 근거 대기");
    const perception = snapshot.perception || {}, kind = perception.kind;
    const learned = kind === "learned";
    text("perception-kind", learned ? "학습 모델" : ["synthetic", "oracle", "synthetic_oracle"].includes(kind) ? "합성 정답" : kind || "—");
    $("perception-kind").className = `badge compact ${learned ? "caution" : "neutral"}`;
    text("perception-label", kind === "synthetic_oracle" ? "합성 정답 센서 · 학습 모델 사용 안 함" : learned ? "학습된 의미 분할 + 정답 거리·추적" : perception.label || "모델 정보 미제공");
    $("perception-label").title = perception.label || "";
    const observedRange = perception.visibility?.idealized_range_m;
    text("range-source", perception.oracle_range === true ? `거리·점유: 합성 세계 정답 기반 (실센서 추정 아님)${number(observedRange) ? ` · 가정한 관측 범위 ${format(observedRange)} m, 가림 적용` : ""}` : perception.oracle_range === false ? "거리·점유: 엔진에 기록된 생산자 근거를 확인하세요." : "거리 관측 출처 미확인");
    text("model-error", perception.model_error || ""); $("model-error").hidden = !perception.model_error;
    const metrics = snapshot.metrics || {};
    for (const [id, value, digits] of [["distance", metrics.distance_m, 1], ["collisions", metrics.collisions, 0], ["offroad", metrics.offroad_events, 0], ["clearance", metrics.min_clearance_m, 2], ["interventions", metrics.interventions, 0]]) text(`metric-${id}`, format(value, digits));
    $("metric-collisions").classList.toggle("alert-value", number(metrics.collisions) && metrics.collisions > 0);
    $("metric-offroad").classList.toggle("alert-value", number(metrics.offroad_events) && metrics.offroad_events > 0);
    text("metric-goal", metrics.goal_reached === true ? "도달" : metrics.goal_reached === false ? "미도달" : "—");
    renderPerformance(snapshot.performance, manual);
    renderLane(snapshot.lane, snapshot.world.lane, manual);
    text("performance-scope", `${scenarioTitle(snapshot.scenario)} · SEED ${format(snapshot.seed)} · 이번 합성 세션의 제어·경로 추종 지표입니다. 실차 성능이나 인지 정확도 평가가 아닙니다.`);
    const region = snapshot.world.region, city = region && typeof region === "object";
    const syntheticRegion = city && (region.geometry_source === "synthetic_regional_course" || region.terrain?.synthetic === true);
    const attribution = city && region.provenance?.attribution;
    for (const id of ["region-summary", "region-assumptions"]) $(id).hidden = !city;
    $("map-attribution").hidden = !city || syntheticRegion || !attribution;
    canvas.setAttribute("aria-label", city ? `${region.name || "지역"} ${syntheticRegion ? "지역 특성 합성 코스" : "실제 지도"}와 합성 차량·장애물·주행 경로` : "차량, 도로, 장애물, 제안 경로의 위에서 본 합성 세계");
    if (city) {
      const roads = Array.isArray(region.roads) ? region.roads : [], buildings = Array.isArray(region.buildings) ? region.buildings : [];
      const actors = Array.isArray(snapshot.world.obstacles) ? snapshot.world.obstacles : [];
      const count = (kind) => actors.filter((item) => item.kind === kind).length;
      const moving = actors.filter((item) => number(item.vx_mps) && number(item.vy_mps) && Math.hypot(item.vx_mps,item.vy_mps) > .01).length;
      text("region-name", region.name || "서울 성수 지역");
      text("region-source", syntheticRegion ? "지역 특성 합성 코스" : attribution ? "OSM 실제 지도" : "지도 출처 미제공");
      text("region-assumptions", syntheticRegion ? "지역의 해안·도심·습지·산지 특성을 반영한 합성 도로·고도입니다. 실제 도로 형상이나 측량 고도가 아니며 교통·센서 관측도 합성입니다." : "실제 지도 윤곽 · 폭·높이는 가정 포함 · 교통은 합성 · 신호 운영 미구현. 거리·추적은 가림과 관측 범위를 적용한 합성 정답 기반이며 현장 측정이 아닙니다.");
      text("region-counts", `도로·보행로 ${format(roads.length)} · 건물 윤곽 ${format(buildings.length)}${number(snapshot.world.route_length_m) ? ` · 주행 경로 ${format(snapshot.world.route_length_m)}m` : ""}`);
      text("actor-counts", `합성 교통 · 차량 ${count("vehicle")} · 보행자 ${count("pedestrian")} · 콘 ${count("cone")} · 움직이는 객체 ${moving}`);
    }
    renderTerrain(region, vehicle, snapshot.world.route_progress_m);
    renderTraffic(snapshot.world);
    renderObjects(snapshot.object_perception);
    updateEvents(snapshot.events); drawWorld();
    const version = `${snapshot.session_id || "local"}-${snapshot.seed}-${snapshot.step}-${snapshot.mode}-${snapshot.scenario?.id || "scene"}`;
    for (const item of images) updateImage(item, version);
    $("canvas-loading").hidden = true; updateEnabled();
    announce(`${runLabel}. ${manual ? "수동" : "자율"} 모드. 안전 상태 ${mode}.`);
  }
  function clipRoadToXRange(road, minimum, maximum) {
    const clipped = [];
    for (let index = 1; index < road.length; index += 1) {
      const start = road[index - 1], end = road[index], dx = end[0] - start[0];
      let from = 0, to = 1;
      if (Math.abs(dx) < 1e-12) {
        if (start[0] < minimum || start[0] > maximum) continue;
      } else {
        const atMinimum = (minimum - start[0]) / dx, atMaximum = (maximum - start[0]) / dx;
        from = Math.max(0, Math.min(atMinimum, atMaximum));
        to = Math.min(1, Math.max(atMinimum, atMaximum));
        if (from > to) continue;
      }
      for (const fraction of [from, to]) {
        clipped.push([start[0] + dx * fraction, start[1] + (end[1] - start[1]) * fraction]);
      }
    }
    return clipped;
  }
  function projectWorldPoint(point, viewport) {
    const {width, height, centerX, centerY, scale, eastNorth} = viewport;
    return eastNorth
      ? [width / 2 + (point[0] - centerX) * scale, height / 2 - (point[1] - centerY) * scale]
      : [width / 2 - (point[1] - centerY) * scale, height / 2 - (point[0] - centerX) * scale];
  }
  function drawWorld() {
    if (!context || !state) return;
    const width = canvas.clientWidth, height = canvas.clientHeight, ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (!width || !height) return;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    const world = state.world, car = state.vehicle, region = world.region;
    const city = region && typeof region === "object", eastNorth = city || world.coordinates === "east_north_m";
    const scenery = region?.terrain?.scenery;
    const sceneryColor = (value, fallback, lighten = 0) => Array.isArray(value) && value.length === 3 && value.every(number) ? `rgb(${value.map((channel) => Math.round(Math.min(255,Math.max(0,channel))*(1-lighten)+255*lighten)).join(",")})` : fallback;
    context.fillStyle = sceneryColor(scenery?.ground_color, city ? "#edf0e9" : "#edf2ef", .76); context.fillRect(0, 0, width, height);
    const validPoint = (point) => Array.isArray(point) && point.length >= 2 && number(point[0]) && number(point[1]);
    const road = Array.isArray(world.road_points) ? world.road_points.filter(validPoint) : [];
    if (road.length < 2 || !number(car.x_m) || !number(car.y_m)) return;
    const halfWidth = number(world.road_half_width_m) ? Math.max(0, world.road_half_width_m) : 0;
    const obstacles = Array.isArray(world.obstacles) ? world.obstacles.filter((item) => number(item.x_m) && number(item.y_m) && number(item.radius_m)) : [];
    const roads = city && Array.isArray(region.roads) ? region.roads.map((item) => ({...item, points: Array.isArray(item.points) ? item.points.filter(validPoint) : []})).filter((item) => item.points.length > 1) : [];
    const buildings = city && Array.isArray(region.buildings) ? region.buildings.map((item) => ({...item, polygon: Array.isArray(item.polygon) ? item.polygon.filter(validPoint) : []})).filter((item) => item.polygon.length > 2) : [];
    const course = city ? road : number(world.goal_x_m) ? clipRoadToXRange(road, Math.min(0,car.x_m)-6, Math.max(world.goal_x_m,car.x_m)+10) : road;
    const bounds = [...course, [car.x_m, car.y_m], ...obstacles.map((item) => [item.x_m, item.y_m])];
    if (city) { for (const item of roads) bounds.push(...item.points); for (const item of buildings) bounds.push(...item.polygon); }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x,y] of bounds) { minX = Math.min(minX,x); maxX = Math.max(maxX,x); minY = Math.min(minY,y); maxY = Math.max(maxY,y); }
    const margin = city ? 12 : halfWidth + 5;
    minX -= margin; maxX += margin; minY -= margin; maxY += margin;
    let centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
    let scale = eastNorth
      ? Math.min((width - 42) / Math.max(1,maxX-minX),(height - 45) / Math.max(1,maxY-minY))
      : Math.min((width - 48) / Math.max(1,maxY-minY),(height - 75) / Math.max(1,maxX-minX));
    if (followVehicle) {
      const yaw = number(car.yaw_rad) ? car.yaw_rad : 0, lookAhead = city ? 9 : 12;
      centerX = car.x_m + Math.cos(yaw) * lookAhead; centerY = car.y_m + Math.sin(yaw) * lookAhead;
      scale = city ? Math.min(width / 110,height / 90) : Math.min(width / 46,height / 54);
    }
    scale *= mapZoom;
    if (!number(scale) || scale <= 0) return;
    const viewport = {width,height,centerX,centerY,scale,eastNorth};
    const project = (x,y) => projectWorldPoint([x,y],viewport);
    const path = (points, close = false) => { context.beginPath(); points.forEach((point,index) => { const [x,y] = project(point[0],point[1]); if (!index) context.moveTo(x,y); else context.lineTo(x,y); }); if (close) context.closePath(); };
    if (scenery?.kind === "coast" || scenery?.kind === "wetland") {
      // A schematic shore band, not a claimed geographic water boundary.
      context.save(); context.globalAlpha = .65; context.fillStyle = sceneryColor(scenery.water_color, "#bdd9dd", .55);
      const right = scenery.water_side === "right";
      context.beginPath(); context.moveTo(right ? width : 0,0); context.lineTo(right ? width*.82 : width*.18,0);
      context.bezierCurveTo(right ? width*.70 : width*.30,height*.35,right ? width*.89 : width*.11,height*.6,right ? width*.77 : width*.23,height);
      context.lineTo(right ? width : 0,height); context.closePath(); context.fill(); context.restore();
    }
    if (scenery?.kind === "mountain" || scenery?.kind === "coast") {
      context.save(); context.strokeStyle = sceneryColor(scenery.ridge_color, "#90aa8c", .2); context.globalAlpha = .22; context.lineWidth = 1;
      for (let index = 0; index < 5; index += 1) {
        context.beginPath(); context.ellipse(width*.77,height*.28,width*(.12+index*.045),height*(.08+index*.055),-.35,0,Math.PI*2); context.stroke();
      }
      context.restore();
    }
    const gridSpacing = city ? (scale > 4 ? 5 : scale > 2 ? 10 : scale > .8 ? 25 : 50) : 5;
    minX = centerX - (eastNorth ? width : height) / scale / 2; maxX = centerX + (eastNorth ? width : height) / scale / 2;
    minY = centerY - (eastNorth ? height : width) / scale / 2; maxY = centerY + (eastNorth ? height : width) / scale / 2;
    context.strokeStyle = city ? "#dfe6db" : "#dde6e2"; context.lineWidth = .6;
    for (let x = Math.ceil(minX/gridSpacing)*gridSpacing; x < maxX; x += gridSpacing) { path([[x,minY],[x,maxY]]); context.stroke(); }
    for (let y = Math.ceil(minY/gridSpacing)*gridSpacing; y < maxY; y += gridSpacing) { path([[minX,y],[maxX,y]]); context.stroke(); }
    context.lineJoin = "round"; context.lineCap = "round";
    if (city) {
      // Each footprint and road is its own path; disconnected OSM ways are never joined.
      for (const building of buildings) {
        path(building.polygon,true); context.fillStyle = "#dce2d6"; context.fill(); context.strokeStyle = "#c4cfc0"; context.lineWidth = .7; context.stroke();
      }
      for (const item of roads) {
        const roadWidth = number(item.half_width_m) ? Math.max(0,item.half_width_m)*2*scale : 1;
        path(item.points); context.lineWidth = roadWidth + 1.6; context.strokeStyle = "#c7d1c8"; context.stroke();
      }
      for (const item of roads) {
        const roadWidth = number(item.half_width_m) ? Math.max(0,item.half_width_m)*2*scale : 1;
        path(item.points); context.lineWidth = Math.max(.7,roadWidth); context.strokeStyle = ["footway","path","pedestrian","steps"].includes(item.highway) ? "#e5eadd" : "#fafbf6"; context.stroke();
      }
      const names = new Set(), labelCenters = [];
      for (const item of roads) {
        if (typeof item.name !== "string" || !item.name || names.has(item.name) || names.size >= 18) continue;
        const middle = item.points[Math.floor(item.points.length/2)], [x,y] = project(middle[0],middle[1]);
        const first = project(item.points[0][0],item.points[0][1]), last = project(item.points.at(-1)[0],item.points.at(-1)[1]);
        if (Math.hypot(first[0]-last[0],first[1]-last[1]) < 40 || x < 50 || x > width-50 || y < 48 || y > height-32 || labelCenters.some((point) => Math.hypot(point[0]-x,point[1]-y)<65)) continue;
        context.font = "9px system-ui"; context.textAlign = "center"; context.lineWidth = 3; context.strokeStyle = "#fafbf4"; context.strokeText(item.name,x,y); context.fillStyle = "#82917f"; context.fillText(item.name,x,y);
        names.add(item.name); labelCenters.push([x,y]);
      }
      path(road); context.lineWidth = 7; context.strokeStyle = "#ffffffd9"; context.stroke();
      path(road); context.lineWidth = 3.8; context.strokeStyle = "#458f847d"; context.stroke();
      path(road); context.lineWidth = 1.5; context.strokeStyle = "#388f83"; context.setLineDash([4,5]); context.stroke(); context.setLineDash([]);
    } else {
      path(road); context.lineWidth = 2*halfWidth*scale+3; context.strokeStyle = "#bacdca"; context.stroke();
      path(road); context.lineWidth = Math.max(.5,2*halfWidth*scale); context.strokeStyle = "#d2dedb"; context.stroke();
      if (!world.road_lanes) { path(road); context.lineWidth = 1; context.strokeStyle = "#f7faf8"; context.setLineDash([7,8]); context.stroke(); context.setLineDash([]); }
    }
    const laneBoundaries = assumedLaneBoundaries(world.lane);
    if (laneBoundaries) {
      context.save(); context.lineWidth = 1.4; context.strokeStyle = state.lane?.current?.outside === true ? "#b96160" : "#8b70ad"; context.setLineDash([6,4]);
      for (const boundary of laneBoundaries) { path(boundary); context.stroke(); }
      context.restore();
    }
    if (world.road_lanes?.source === "synthetic_lane_assignment_not_measured") {
      const layout=world.road_lanes;
      context.save();
      const egoLane=layout.lanes?.find((item)=>item.is_ego);
      if (egoLane?.centerline?.length>1) {path(egoLane.centerline);context.lineWidth=3.6*scale;context.strokeStyle="#15978a18";context.stroke();}
      for (const [key,color] of [["divider","#d3a92c"],["ego_outer_boundary","#ffffff"],["opposing_outer_boundary","#ffffff"]]) {
        if (Array.isArray(layout[key])) {path(layout[key]);context.lineWidth=2;context.strokeStyle=color;context.setLineDash([]);context.stroke();}
      }
      for (const lane of layout.lanes || []) {
        const points=lane.centerline;
        if (!Array.isArray(points)||points.length<2) continue;
        let index=0;
        for (let i=1;i<points.length;i++) if (Math.hypot(points[i][0]-car.x_m,points[i][1]-car.y_m)<Math.hypot(points[index][0]-car.x_m,points[index][1]-car.y_m)) index=i;
        const target=points[index], a=index===points.length-1?points[index-1]:target, b=index===points.length-1?target:points[index+1], length=Math.hypot(b[0]-a[0],b[1]-a[1]);
        if (!length) continue;
        const dx=(b[0]-a[0])/length,dy=(b[1]-a[1])/length;
        const [x,y]=project(target[0],target[1]),[tx,ty]=project(target[0]+dx*4,target[1]+dy*4);
        context.beginPath();context.moveTo(x,y);context.lineTo(tx,ty);context.strokeStyle=lane.is_ego?"#087f78":"#ac705a";context.lineWidth=2;context.stroke();
        const angle=Math.atan2(ty-y,tx-x);context.beginPath();context.moveTo(tx,ty);context.lineTo(tx-6*Math.cos(angle-.5),ty-6*Math.sin(angle-.5));context.lineTo(tx-6*Math.cos(angle+.5),ty-6*Math.sin(angle+.5));context.closePath();context.fillStyle=context.strokeStyle;context.fill();
      }
      context.restore();
    }
    if (world.traffic?.crosswalk) {
      const crossing=world.traffic.crosswalk;
      context.save();context.strokeStyle="#ffffff";context.lineWidth=Math.max(1,.22*scale);
      for (let x=crossing.x_min_m;x<=crossing.x_max_m;x+=.45) {path([[x,crossing.y_min_m],[x,crossing.y_max_m]]);context.stroke();}
      path([[29.5,-1.8],[29.5,1.8]]);context.lineWidth=2;context.strokeStyle=world.traffic.signal?.vehicles==="red"?"#bc4047":"#15978a";context.stroke();context.restore();
    }
    const marker = (point,label,color) => {
      const [x,y] = project(point[0],point[1]); context.beginPath(); context.arc(x,y,5,0,Math.PI*2); context.fillStyle = "#fff"; context.fill(); context.lineWidth = 2; context.strokeStyle = color; context.stroke();
      context.font = "bold 9px system-ui"; context.textAlign = "center"; context.lineWidth = 4; context.strokeStyle = "#fbfcf7"; context.strokeText(label,x,y-10); context.fillStyle = color; context.fillText(label,x,y-10);
    };
    if (city) {
      marker(road[0],"출발", "#648ba3");
      marker(validPoint(world.goal_position_m) ? world.goal_position_m : road.at(-1),"목표", "#197d6f");
    } else if (number(world.goal_x_m)) {
      const candidates = clipRoadToXRange(road,world.goal_x_m,world.goal_x_m);
      const goal = candidates[0] || road.reduce((best,point) => Math.abs(point[0]-world.goal_x_m)<Math.abs(best[0]-world.goal_x_m)?point:best);
      path([[world.goal_x_m,goal[1]-halfWidth],[world.goal_x_m,goal[1]+halfWidth]]); context.lineWidth = 1.5; context.strokeStyle = "#79a793"; context.setLineDash([3,3]); context.stroke(); context.setLineDash([]);
      const [gx,gy] = project(world.goal_x_m,goal[1]); context.fillStyle = "#729687"; context.font = "9px system-ui"; context.textAlign = "center"; context.fillText("GOAL",gx,gy-8);
    }
    const planned = Array.isArray(state.plan_world) ? state.plan_world.filter(validPoint) : [];
    if (planned.length > 1) { path(planned); context.lineWidth = 3; context.strokeStyle = "#159eacd9"; context.stroke(); path(planned); context.lineWidth = 1; context.strokeStyle = "#dcf9f5"; context.setLineDash([3,5]); context.stroke(); context.setLineDash([]); }
    const objectView = state.object_perception;
    const observations = new Map(objectView?.source === "synthetic_geometry_not_learned_detection" && Array.isArray(objectView.objects) ? objectView.objects.map((actor) => [actor.id,actor]) : []);
    for (const item of obstacles) {
      const [x,y] = project(item.x_m,item.y_m), radius = Math.max(3,item.radius_m*scale);
      if (x < -radius || y < -radius || x > width+radius || y > height+radius) continue;
      const moving = number(item.vx_mps) && number(item.vy_mps) && Math.hypot(item.vx_mps,item.vy_mps) > .01;
      if (moving) { const [tx,ty] = project(item.x_m+item.vx_mps*.8,item.y_m+item.vy_mps*.8); context.beginPath(); context.moveTo(x,y); context.lineTo(tx,ty); context.lineWidth = 1.2; context.strokeStyle = item.kind === "pedestrian" ? "#c8806c88" : "#7f9cad88"; context.stroke(); }
      context.save();
      const observed = observations.get(item.id);
      if (objectView?.source === "synthetic_geometry_not_learned_detection") {
        if (observed) {
          const [ox,oy] = project(observed.x_m,observed.y_m);
          context.beginPath(); context.arc(ox,oy,radius+4,0,Math.PI*2); context.strokeStyle = observed.near_reference_route ? "#e89a38" : "#27a391"; context.lineWidth = 2; context.stroke();
          context.font = "9px system-ui"; context.textAlign = "center"; context.fillStyle = "#365c64"; context.fillText(`${item.kind === "pedestrian" ? "보행자" : item.kind === "vehicle" ? "차량" : item.kind === "cone" ? "콘" : "장애물"} ${format(observed.distance_m,1)}m`,ox,oy-radius-8);
        }
        context.globalAlpha = .35;
      }
      context.translate(x,y);
      if (item.kind === "vehicle") {
        const heading = number(item.yaw_rad) ? item.yaw_rad : moving ? Math.atan2(item.vy_mps,item.vx_mps) : 0;
        context.rotate(eastNorth ? Math.PI/2-heading : -heading);
        context.beginPath(); context.roundRect(-radius*.55,-radius,radius*1.1,radius*2,Math.min(2,radius/3)); context.fillStyle = moving ? "#6d91a8" : "#91a6ae"; context.fill(); context.lineWidth = 1; context.strokeStyle = "#fff"; context.stroke();
        if (radius > 4) { context.fillStyle = "#cce0e6"; context.fillRect(-radius*.36,-radius*.55,radius*.72,radius*.42); }
      } else if (item.kind === "cone") {
        context.beginPath(); context.moveTo(0,-radius); context.lineTo(-radius*.8,radius*.7); context.lineTo(radius*.8,radius*.7); context.closePath(); context.fillStyle = "#dca05e"; context.fill(); context.lineWidth = .8; context.strokeStyle = "#fff7e7"; context.stroke();
        context.beginPath(); context.moveTo(-radius*.36,0); context.lineTo(radius*.36,0); context.strokeStyle = "#fff7e7"; context.lineWidth = Math.max(1,radius*.2); context.stroke();
      } else {
        context.beginPath(); context.arc(0,0,radius,0,Math.PI*2); context.fillStyle = item.kind === "pedestrian" ? "#c87f70" : "#d29960"; context.fill(); context.lineWidth = 1.1; context.strokeStyle = "#fff7ee"; context.stroke();
        if (item.kind === "pedestrian" && radius > 3.5) { context.beginPath(); context.arc(0,-radius*.2,radius*.3,0,Math.PI*2); context.fillStyle = "#ffe9dd"; context.fill(); }
      }
      context.restore();
    }
    const [cx,cy] = project(car.x_m,car.y_m);
    context.beginPath(); context.arc(cx,cy,15,0,Math.PI*2); context.fillStyle = "#138a7b16"; context.fill();
    context.save(); context.translate(cx,cy); const yaw = number(car.yaw_rad) ? car.yaw_rad : 0; context.rotate(eastNorth ? Math.PI/2-yaw : -yaw);
    context.fillStyle = "#d7e9e5"; context.strokeStyle = "#fff"; context.lineWidth = 1.2; context.beginPath(); context.roundRect(-6.5,-11,13,22,4); context.fill(); context.stroke();
    context.fillStyle = "#128478"; context.beginPath(); context.roundRect(-5.5,-10,11,20,3); context.fill(); context.fillStyle = "#a9e0d5"; context.beginPath(); context.roundRect(-4,-6,8,5,1); context.fill(); context.fillStyle = "#58aa9a"; context.fillRect(-3.5,5,7,3);
    context.beginPath(); context.moveTo(0,-18); context.lineTo(-3,-14); context.lineTo(3,-14); context.closePath(); context.fillStyle = "#05776d"; context.fill(); context.restore();
    context.strokeStyle = "#94aaa5"; context.fillStyle = "#8da39e"; context.lineWidth = 1; context.beginPath(); context.moveTo(22,height-21); context.lineTo(22,height-37); context.moveTo(19,height-33); context.lineTo(22,height-37); context.lineTo(25,height-33); context.stroke(); context.font = "9px system-ui"; context.textAlign = "left"; context.fillText(eastNorth ? region?.synthetic ? "+y" : "N" : "+x",29,height-27);
    const scaleMetres = [1,2,5,10,20,50,100,200].filter((value) => value*scale <= 75).at(-1) || 1;
    $("world-scale").querySelector("span").style.width = `${scaleMetres*scale}px`; $("world-scale").querySelector("small").textContent = `${scaleMetres} m`;
    text("world-coordinate", eastNorth && !region?.synthetic ? "동쪽 +x / 북쪽 +y · 객체 기호는 도식" : "전방 +x / 좌측 +y · 객체 기호는 도식");
    $("view-fit").textContent = city ? "지역 전체" : "전체 보기";
    $("zoom-out").disabled = mapZoom <= .7; $("zoom-in").disabled = mapZoom >= 8;
  }
  async function poll() {
    const generation = actionGeneration;
    try {
      const snapshot = await request("/api/state"); connected = true;
      $("connection").className = "connection online"; $("connection").querySelector("span").textContent = "엔진 연결됨";
      if (generation !== actionGeneration || actionPending) return;
      if (snapshot.server_error) showError(`엔진 오류: ${snapshot.server_error}`);
      else if (!lastControlError) showError(actionError);
      render(snapshot);
    } catch (error) {
      connected = false; releaseControls(false); updateEnabled();
      $("connection").className = "connection offline"; $("connection").querySelector("span").textContent = "연결 끊김";
      showError(`상태를 확인할 수 없습니다. 화면은 마지막 수신 상태입니다. ${error.message}`); announce("시뮬레이터 연결이 끊겼습니다. 화면의 값은 마지막 수신 상태입니다.");
    } finally { setTimeout(poll, connected ? 250 : 1200); }
  }
  $("start-pause").addEventListener("click", () => { if (state) void act(state.running ? "pause" : "start"); });
  $("step").addEventListener("click", () => { releaseControls(); void act("step"); });
  $("reset").addEventListener("click", () => { const seed = Number($("seed").value); if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) { actionError = "시드는 0부터 2147483647 사이의 정수로 입력하세요."; showError(actionError); $("seed").focus(); return; } void act("reset", {scenario: $("scenario").value, seed}); });
  $("mode-auto").addEventListener("click", () => { void act("mode", {mode: "autonomous"}); });
  $("mode-manual").addEventListener("click", () => { void act("mode", {mode: "manual"}); });
  $("reset-safety").addEventListener("click", () => { void act("reset_safety"); });
  $("scenario").addEventListener("change", () => { selectedScenarioDirty = true; describeScenario(); });
  $("seed").addEventListener("input", () => { selectedSeedDirty = true; });
  $("speed-mode").addEventListener("change", () => { markSpeedDraft(); describeSpeedMode(); });
  $("speed-target").addEventListener("input", () => {
    markSpeedDraft();
    const value = Number($("speed-target").value);
    if ($("speed-target").value.trim() !== "" && number(value) && value >= 0 && value <= Number($("speed-target").max)) $("speed-range").value = String(value);
  });
  $("speed-range").addEventListener("input", () => { markSpeedDraft(); $("speed-target").value = $("speed-range").value; });
  $("apply-speed").addEventListener("click", () => {
    const raw = $("speed-target").value.trim(), target = Number(raw), maximum = Number($("speed-target").max), speedMode = $("speed-mode").value;
    if (!raw || !number(target) || target < 0 || target > maximum || !["automatic", "user"].includes(speedMode)) {
      setSpeedInputError(`속도는 0부터 ${format(maximum)} km/h 사이의 숫자로 입력하세요.`); $("speed-target").focus(); return;
    }
    void act("speed", {speed_mode: speedMode, target_speed_kmh: target});
  });
  $("export-report").addEventListener("click", (event) => { if (!connected) event.preventDefault(); });
  $("object-overlay-toggle").addEventListener("click", () => { objectOverlayEnabled = !objectOverlayEnabled; if (state) render(state); });
  $("view-toggle").addEventListener("click", () => { followVehicle = !followVehicle; mapZoom = 1; $("view-toggle").setAttribute("aria-pressed", String(followVehicle)); drawWorld(); });
  $("view-fit").addEventListener("click", () => { followVehicle = false; mapZoom = 1; $("view-toggle").setAttribute("aria-pressed", "false"); drawWorld(); });
  $("zoom-out").addEventListener("click", () => { mapZoom = Math.max(.7,mapZoom/1.4); drawWorld(); });
  $("zoom-in").addEventListener("click", () => { mapZoom = Math.min(8,mapZoom*1.4); drawWorld(); });
  document.addEventListener("keydown", (event) => {
    if (!keyMap[event.code] || /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) || event.code === "Space" && event.target.tagName === "BUTTON" || event.target.isContentEditable || !state || state.mode !== "manual" || !state.running || actionPending || !connected) return;
    event.preventDefault(); if (event.repeat) return;
    keys.add(event.code); deriveControls(); void flushControls();
  });
  document.addEventListener("keyup", (event) => { if (!keys.has(event.code)) return; event.preventDefault(); releaseControls(); });
  for (const button of document.querySelectorAll(".drive-key")) {
    button.addEventListener("pointerdown", (event) => { if (button.disabled) return; event.preventDefault(); button.setPointerCapture(event.pointerId); pointerKeys.set(event.pointerId,button.dataset.drive); deriveControls(); void flushControls(); });
    const release = (event) => { if (pointerKeys.has(event.pointerId)) releaseControls(); };
    button.addEventListener("pointerup", release); button.addEventListener("pointercancel", release); button.addEventListener("lostpointercapture", release);
    button.addEventListener("keydown", (event) => { if (button.disabled || !["Space","Enter"].includes(event.code)) return; event.preventDefault(); if (event.repeat) return; pointerKeys.set(`keyboard-${button.dataset.drive}`,button.dataset.drive); deriveControls(); void flushControls(); });
    button.addEventListener("keyup", (event) => { if (!["Space","Enter"].includes(event.code)) return; event.preventDefault(); releaseControls(); });
    button.addEventListener("blur", () => { pointerKeys.delete(`keyboard-${button.dataset.drive}`); deriveControls(); if (connected && state?.mode === "manual") void flushControls(); });
  }
  window.addEventListener("blur", () => releaseControls());
  document.addEventListener("visibilitychange", () => { if (document.hidden) releaseControls(); });
  window.addEventListener("pagehide", () => { releaseControls(false); if (connected) { void fetch("/api/action", {method:"POST",credentials:"same-origin",headers:{"Content-Type":"application/json","X-Simulator-Client":"1"},body:JSON.stringify({action:"controls",steering:0,throttle:0,brake:1}),keepalive:true}).catch(() => {}); } });
  setInterval(() => { if (keys.size || pointerKeys.size) { deriveControls(); void flushControls(); } }, 180);
  new ResizeObserver(drawWorld).observe(canvas);
  void request("/api/scenarios").then((data) => fillScenarios(Array.isArray(data) ? data : data.scenarios)).catch(() => {});
  void poll();
})();
