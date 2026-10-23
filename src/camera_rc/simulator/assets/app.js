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
  let state = null, connected = false, actionPending = false, followVehicle = false;
  let scenarioSignature = "", lastSession = null, lastAnnounced = "", eventsSignature = "", actionError = "";
  let actionGeneration = 0;
  let controlsPending = false, controlsDirty = false, lastControlError = false;
  let desiredControls = {steering: 0, throttle: 0, brake: 1};
  let selectedScenarioDirty = false, selectedSeedDirty = false;
  const canvas = $("world-canvas");
  const context = canvas.getContext("2d");
  const images = [{id: "camera-image", placeholder: "camera-placeholder", path: "/frames/camera.png"}, {id: "segmentation-image", placeholder: "segmentation-placeholder", path: "/frames/segmentation.png"}];
  const reasons = {
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
    invalid_or_stop_proposal: "정지 또는 유효하지 않은 경로 제안입니다."
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
    braking_degraded: ["제동 성능 저하", "제동 가능 상태가 실패하고 합성 감속 성능도 낮아집니다."]
  };
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
      const option = document.createElement("option"); option.value = item.id; option.textContent = scenarioLabels[item.id]?.[0] || item.title || item.id;
      option.dataset.description = scenarioLabels[item.id]?.[1] || item.description || "합성 장면의 도로와 장애물을 관찰합니다.";
      $("scenario").append(option);
    }
    if ([...$("scenario").options].some((option) => option.value === previous)) $("scenario").value = previous;
    describeScenario();
  }
  function describeScenario() { text("scenario-description", $("scenario").selectedOptions[0]?.dataset.description || "시나리오 정보를 기다리고 있습니다."); }
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
  function render(snapshot) {
    if (!snapshot || typeof snapshot !== "object" || !snapshot.vehicle || !snapshot.world) throw new Error("시뮬레이터 상태 형식이 올바르지 않습니다.");
    const previous = state; state = snapshot;
    if (previous && (previous.mode !== state.mode || previous.running && !state.running || previous.session_id !== state.session_id)) releaseControls();
    if (snapshot.session_id !== lastSession) { lastSession = snapshot.session_id; for (const item of images) item.version = null; }
    fillScenarios(snapshot.scenarios);
    if (!selectedScenarioDirty && snapshot.scenario?.id) { $("scenario").value = snapshot.scenario.id; describeScenario(); }
    if (!selectedSeedDirty && number(snapshot.seed)) $("seed").value = snapshot.seed;
    const running = Boolean(snapshot.running), finished = Boolean(snapshot.finished);
    const runLabel = finished ? "실행 종료" : running ? "실행 중" : "일시정지";
    text("run-state", runLabel); $("run-state").className = `badge ${running ? "good" : finished ? "caution" : "neutral"}`;
    text("play-label", running ? "일시정지" : "시작"); text("play-icon", running ? "Ⅱ" : "▶");
    text("sim-time", number(snapshot.sim_time_s) ? `${format(snapshot.sim_time_s, 1)} s` : "—");
    text("step-count", `STEP ${format(snapshot.step)}`); text("world-scenario", scenarioLabels[snapshot.scenario?.id]?.[0] || snapshot.scenario?.title || snapshot.scenario?.id || "장면 대기");
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
    text("speed", format(vehicle.speed_mps, 2)); text("speed-kmh", number(vehicle.speed_mps) ? `${format(vehicle.speed_mps * 3.6, 1)} km/h` : "— km/h");
    text("steering", number(vehicle.steering_rad) ? `${format(vehicle.steering_rad * 180 / Math.PI, 1)}°` : "—");
    for (const name of ["throttle", "brake"]) { const value = command[name]; $(`${name}-meter`).value = number(value) ? Math.min(1, Math.max(0, value)) : 0; text(`${name}-value`, number(value) ? `${format(value * 100)}%` : "—"); }
    text("command-reason", command.reason ? `명령 · ${command.reason}` : "명령 근거 대기");
    const perception = snapshot.perception || {}, kind = perception.kind;
    const learned = kind === "learned";
    text("perception-kind", learned ? "학습 모델" : ["synthetic", "oracle", "synthetic_oracle"].includes(kind) ? "합성 정답" : kind || "—");
    $("perception-kind").className = `badge compact ${learned ? "caution" : "neutral"}`;
    text("perception-label", kind === "synthetic_oracle" ? "합성 정답 센서 · 학습 모델 사용 안 함" : learned ? "학습된 의미 분할 + 정답 거리·추적" : perception.label || "모델 정보 미제공");
    $("perception-label").title = perception.label || "";
    text("range-source", perception.oracle_range === true ? "거리·점유: 합성 세계 정답 기반 (실센서 추정 아님)" : perception.oracle_range === false ? "거리·점유: 엔진에 기록된 생산자 근거를 확인하세요." : "거리 관측 출처 미확인");
    text("model-error", perception.model_error || ""); $("model-error").hidden = !perception.model_error;
    const metrics = snapshot.metrics || {};
    for (const [id, value, digits] of [["distance", metrics.distance_m, 1], ["collisions", metrics.collisions, 0], ["offroad", metrics.offroad_events, 0], ["clearance", metrics.min_clearance_m, 2], ["interventions", metrics.interventions, 0]]) text(`metric-${id}`, format(value, digits));
    $("metric-collisions").classList.toggle("alert-value", number(metrics.collisions) && metrics.collisions > 0);
    $("metric-offroad").classList.toggle("alert-value", number(metrics.offroad_events) && metrics.offroad_events > 0);
    text("metric-goal", metrics.goal_reached === true ? "도달" : metrics.goal_reached === false ? "미도달" : "—");
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
  function drawWorld() {
    if (!context || !state) return;
    const width = canvas.clientWidth, height = canvas.clientHeight, ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (!width || !height) return;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    context.fillStyle = "#edf2ef"; context.fillRect(0, 0, width, height);
    const world = state.world, car = state.vehicle;
    const validPoint = (point) => Array.isArray(point) && point.length >= 2 && number(point[0]) && number(point[1]);
    const road = Array.isArray(world.road_points) ? world.road_points.filter(validPoint) : [];
    if (road.length < 2 || !number(car.x_m) || !number(car.y_m)) return;
    const halfWidth = number(world.road_half_width_m) ? Math.max(0, world.road_half_width_m) : 0;
    const obstacles = Array.isArray(world.obstacles) ? world.obstacles.filter((item) => number(item.x_m) && number(item.y_m) && number(item.radius_m)) : [];
    const course = number(world.goal_x_m) ? clipRoadToXRange(road, Math.min(0,car.x_m)-6, Math.max(world.goal_x_m,car.x_m)+10) : road;
    const bounds = [...course, [car.x_m, car.y_m], ...obstacles.map((item) => [item.x_m, item.y_m])];
    const xs = bounds.map((point) => point[0]), ys = bounds.map((point) => point[1]);
    const margin = halfWidth + 5;
    let minX = Math.min(...xs) - margin, maxX = Math.max(...xs) + margin, minY = Math.min(...ys) - margin, maxY = Math.max(...ys) + margin;
    let centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
    let scale = Math.min((width - 48) / Math.max(1, maxY - minY), (height - 75) / Math.max(1, maxX - minX));
    if (followVehicle) { centerX = car.x_m + 12; centerY = car.y_m; scale = Math.min(width / 46, height / 54); }
    if (!number(scale) || scale <= 0) return;
    const project = (x, y) => [width / 2 - (y - centerY) * scale, height / 2 - (x - centerX) * scale];
    const path = (points) => { context.beginPath(); points.forEach((point, index) => { const [x,y] = project(point[0], point[1]); if (!index) context.moveTo(x,y); else context.lineTo(x,y); }); };
    const gridSpacing = 5;
    minX = centerX - height / scale / 2; maxX = centerX + height / scale / 2; minY = centerY - width / scale / 2; maxY = centerY + width / scale / 2;
    context.strokeStyle = "#dde6e2"; context.lineWidth = .65;
    for (let x = Math.ceil(minX / gridSpacing) * gridSpacing; x < maxX; x += gridSpacing) { path([[x,minY],[x,maxY]]); context.stroke(); }
    for (let y = Math.ceil(minY / gridSpacing) * gridSpacing; y < maxY; y += gridSpacing) { path([[minX,y],[maxX,y]]); context.stroke(); }
    context.lineJoin = "round"; context.lineCap = "round";
    path(road); context.lineWidth = 2 * halfWidth * scale + 3; context.strokeStyle = "#bacdca"; context.stroke();
    path(road); context.lineWidth = Math.max(.5, 2 * halfWidth * scale); context.strokeStyle = "#d2dedb"; context.stroke();
    path(road); context.lineWidth = 1; context.strokeStyle = "#f7faf8"; context.setLineDash([7,8]); context.stroke(); context.setLineDash([]);
    if (number(world.goal_x_m)) {
      const nearest = road.reduce((best, point) => Math.abs(point[0] - world.goal_x_m) < Math.abs(best[0] - world.goal_x_m) ? point : best);
      path([[world.goal_x_m,nearest[1] - halfWidth],[world.goal_x_m,nearest[1] + halfWidth]]); context.lineWidth = 1.5; context.strokeStyle = "#79a793"; context.setLineDash([3,3]); context.stroke(); context.setLineDash([]);
      const [gx,gy] = project(world.goal_x_m, nearest[1]); context.fillStyle = "#729687"; context.font = "9px system-ui"; context.textAlign = "center"; context.fillText("GOAL",gx,gy - 8);
    }
    const planned = Array.isArray(state.plan_world) ? state.plan_world.filter(validPoint) : [];
    if (planned.length > 1) { path(planned); context.lineWidth = 3; context.strokeStyle = "#259f9da8"; context.stroke(); path(planned); context.lineWidth = 1; context.strokeStyle = "#d5f4ec"; context.setLineDash([3,5]); context.stroke(); context.setLineDash([]); }
    for (const item of obstacles) {
      const [x,y] = project(item.x_m,item.y_m), radius = Math.max(2.5,item.radius_m * scale);
      context.beginPath(); context.arc(x,y,radius + 3,0,Math.PI * 2); context.fillStyle = "#d4945620"; context.fill();
      context.beginPath(); context.arc(x,y,radius,0,Math.PI * 2); context.fillStyle = item.kind === "pedestrian" ? "#ca785e" : "#d29960"; context.fill(); context.lineWidth = 1.2; context.strokeStyle = "#fff5e8"; context.stroke();
    }
    const [cx,cy] = project(car.x_m,car.y_m);
    context.beginPath(); context.arc(cx,cy,15,0,Math.PI * 2); context.fillStyle = "#138a7b16"; context.fill();
    context.save(); context.translate(cx,cy); context.rotate(number(car.yaw_rad) ? -car.yaw_rad : 0);
    context.fillStyle = "#d7e9e5"; context.strokeStyle = "#fff"; context.lineWidth = 1.2; context.beginPath(); context.roundRect(-6.5,-11,13,22,4); context.fill(); context.stroke();
    context.fillStyle = "#128478"; context.beginPath(); context.roundRect(-5.5,-10,11,20,3); context.fill();
    context.fillStyle = "#a9e0d5"; context.beginPath(); context.roundRect(-4,-6,8,5,1); context.fill(); context.fillStyle = "#58aa9a"; context.fillRect(-3.5,5,7,3);
    context.beginPath(); context.moveTo(0,-18); context.lineTo(-3,-14); context.lineTo(3,-14); context.closePath(); context.fillStyle = "#05776d"; context.fill(); context.restore();
    context.strokeStyle = "#94aaa5"; context.fillStyle = "#8da39e"; context.lineWidth = 1; context.beginPath(); context.moveTo(22,height-21); context.lineTo(22,height-37); context.moveTo(19,height-33); context.lineTo(22,height-37); context.lineTo(25,height-33); context.stroke(); context.font = "8px system-ui"; context.textAlign = "left"; context.fillText("+x",29,height-27);
    const scaleMetres = scale > 8 ? 2 : scale > 3 ? 5 : 10;
    $("world-scale").querySelector("span").style.width = `${scaleMetres * scale}px`; $("world-scale").querySelector("small").textContent = `${scaleMetres} m`;
    text("world-coordinate", `전방 +x / 좌측 +y · 차량 기호는 도식`);
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
  $("export-report").addEventListener("click", (event) => { if (!connected) event.preventDefault(); });
  $("view-toggle").addEventListener("click", () => { followVehicle = !followVehicle; $("view-toggle").setAttribute("aria-pressed", String(followVehicle)); text("view-toggle", followVehicle ? "전체 경로 보기" : "차량 따라가기"); drawWorld(); });
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
