/* Browser protocol regressions, run with: node --test tests/simulator_ui.test.cjs */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const assets = path.join(__dirname, "../src/camera_rc/simulator/assets");

class Element {
  constructor(id = "") {
    this.id = id; this.value = ""; this.textContent = ""; this.hidden = false;
    this.disabled = false; this.dataset = {}; this.style = {}; this.children = [];
    this.listeners = {}; this.attributes = {}; this.max = "150"; this.tagName = "DIV";
    const classes = new Set();
    this.classList = {toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); }, contains(name) { return classes.has(name); }};
  }
  get value() { return this._value; }
  set value(value) { this._value = String(value); }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  emit(type, extra = {}) { this.listeners[type]?.({target: this, ...extra}); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  toggleAttribute(name, present) { if (present) this.attributes[name] = ""; else delete this.attributes[name]; }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  get options() { return this.children; }
  get selectedOptions() { return this.children.filter((child) => child.value === this.value); }
  querySelector(selector) { this.parts ||= {}; return this.parts[selector] ||= new Element(); }
  focus() { this.focused = true; }
  getContext() { return this.context || null; }
}

function snapshot() {
  return {
    session_id: "ui-test", step: 0, seed: 0, running: false, finished: false, mode: "autonomous",
    scenarios: [{id: "gangwon_day", title: "강원 산지 (합성)"}], scenario: {id: "gangwon_day", title: "강원 산지 (합성)"},
    vehicle: {speed_mps: 10, steering_rad: 0, elevation_m: 330, grade_percent: 4},
    speed_control: {mode: "automatic", target_speed_kmh: 150, effective_target_kmh: 72, max_speed_kmh: 150, limiting_reasons: ["curvature"]},
    world: {route_progress_m: 50, region: {name: "강원 산지", geometry_source: "synthetic_regional_course", terrain: {synthetic: true, elevation_profile: [{distance_m: 0, elevation_m: 320}, {distance_m: 100, elevation_m: 340}]}}, obstacles: []},
    safety: {actuation_enabled: false}, perception: {}, metrics: {}, events: []
  };
}
function performanceFixture() {
  return {
    scope: "synthetic_session_only", reference: "reference route polyline",
    route: {progress_pct: 30, progress_m: 30, length_m: 100, remaining_m: 70, current_cross_track_error_m: .25, current_heading_error_deg: 2},
    autonomous: {samples: 3, duration_s: .3, distance_m: 3, cross_track_rmse_m: .32, cross_track_max_m: .5, heading_mae_deg: 2,
      speed_rmse_kmh: 3.2, speed_mae_kmh: 2.5, mean_speed_kmh: 36, peak_speed_kmh: 40, safety_eligible_pct: 66.7,
      safety_stop_s: .1, safety_interventions: 1, collision_events: 0, offroad_events: 0, stopped_time_s: 0},
    current: {applied_target_kmh: 40, acceleration_mps2: 1.25, steering_rate_deg_s: 2.5},
    history: [
      {time_s: 0, mode: "autonomous", speed_kmh: 0, target_kmh: 0, cross_track_error_m: .1},
      {time_s: 1, mode: "autonomous", speed_kmh: 12, target_kmh: 15, cross_track_error_m: .5},
      {time_s: 2, mode: "manual", speed_kmh: 150, target_kmh: 0, cross_track_error_m: 100},
      {time_s: 3, mode: "autonomous", speed_kmh: 9, target_kmh: 0, cross_track_error_m: .2}
    ], definitions: {}
  };
}
function laneFixture() {
  return {
    source: "synthetic_assumed_lane_not_measured", width_m: 3.6,
    current: {outside: false, center_offset_m: .1, max_body_offset_m: 1, clearance_m: .8},
    autonomous: {samples: 10, duration_s: 1, inside_pct: 100, departure_events: 0, outside_time_s: 0},
    training: {status: "not_trained", model_loaded: false, label_space: "ego_lane_v1"},
    guard: {allowed: true, reason: null, scope: "synthetic_path_guard"}
  };
}
function worldLaneFixture() {
  return {source: "synthetic_assumed_lane_not_measured", width_m: 3.6, left_boundary: [[0,1.8],[20,1.8]], right_boundary: [[0,-1.8],[20,-1.8]]};
}
function recordingCanvas() {
  const strokes = [], stack = [];
  const context = {strokes, path: [], dash: [], beginPath() { this.path = []; }, moveTo(x,y) { this.path.push([x,y]); }, lineTo(x,y) { this.path.push([x,y]); },
    setLineDash(dash) { this.dash = [...dash]; }, stroke() { strokes.push({color: this.strokeStyle, dash: [...this.dash], points: this.path.map((point) => [...point])}); },
    save() { stack.push({strokeStyle: this.strokeStyle, dash: [...this.dash]}); }, restore() { Object.assign(this,stack.pop()); }};
  return new Proxy(context, {get(object,key) { return key in object ? object[key] : () => {}; }});
}
const settle = () => new Promise((resolve) => setImmediate(resolve));
async function viewer({initialState, drawMap = false} = {}) {
  const elements = new Map([...fs.readFileSync(path.join(assets, "index.html"), "utf8").matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element(match[1])]));
  const get = (id) => { assert.ok(elements.has(id), `Missing UI element ${id}`); return elements.get(id); };
  const canvas = get("world-canvas");
  if (drawMap) { canvas.clientWidth = 600; canvas.clientHeight = 400; canvas.context = recordingCanvas(); }
  get("speed-mode").value = "automatic";
  const scheduled = new Set(), requests = [], state = {current: initialState || snapshot(), rejectSpeed: false};
  const document = {hidden: false, getElementById: get, createElement: () => new Element(), querySelectorAll: () => [], addEventListener() {}};
  const context = {document, window: {addEventListener() {}}, AbortController, ResizeObserver: class {observe() {}}, setInterval() {},
    setTimeout(callback, delay) { const timer = {callback, delay}; scheduled.add(timer); return timer; }, clearTimeout(timer) { scheduled.delete(timer); },
    async fetch(url, options) {
      const body = options.body && JSON.parse(options.body); requests.push({url, body});
      if (body?.action === "speed" && state.rejectSpeed) return {ok: false, status: 400, json: async () => ({error: "속도 설정 거부"})};
      if (body?.action === "speed") state.current.speed_control = {...state.current.speed_control, mode: body.speed_mode, target_speed_kmh: body.target_speed_kmh};
      return {ok: true, status: 200, json: async () => structuredClone(url === "/api/scenarios" ? state.current.scenarios : state.current)};
    }};
  vm.runInNewContext(fs.readFileSync(path.join(assets, "app.js"), "utf8"), context, {filename: "app.js"});
  await settle();
  const poll = async () => { const timer = [...scheduled].find((item) => item.delay === 250); assert.ok(timer); scheduled.delete(timer); timer.callback(); await settle(); };
  const input = (id, value) => { get(id).value = value; get(id).emit(id === "speed-mode" ? "change" : "input"); };
  return {get, input, poll, requests, state};
}

test("editing speed survives polling and only explicit apply sends the requested mode and speed", async () => {
  const ui = await viewer();
  assert.equal(ui.get("speed").textContent, "36.0");
  assert.equal(ui.get("speed-mps").textContent, "10.00 m/s");
  assert.equal(ui.get("speed-target").value, "150");
  ui.input("speed-mode", "user"); ui.input("speed-target", "83.5");
  assert.equal(ui.get("speed-range").value, "83.5");
  await ui.poll();
  assert.equal(ui.get("speed-mode").value, "user"); assert.equal(ui.get("speed-target").value, "83.5");
  assert.match(ui.get("applied-speed").textContent, /150/);
  assert.equal(ui.requests.filter((request) => request.body).length, 0);
  ui.get("apply-speed").emit("click"); await settle();
  assert.deepEqual(ui.requests.filter((request) => request.body).map((request) => request.body), [{action: "speed", speed_mode: "user", target_speed_kmh: 83.5}]);
  assert.match(ui.get("speed-draft-status").textContent, /설정 적용됨/);
  ui.input("speed-range", "0"); ui.get("apply-speed").emit("click"); await settle();
  assert.equal(ui.requests.at(-1).body.target_speed_kmh, 0);
});

test("invalid numeric targets never send actions and an HTTP failure keeps the draft", async () => {
  const ui = await viewer();
  for (const target of ["", "-1", "150.1", "NaN", "Infinity", "bad"]) {
    ui.input("speed-target", target); ui.get("apply-speed").emit("click"); await settle();
    assert.equal(ui.get("speed-input-error").hidden, false); assert.equal(ui.get("speed-target").attributes["aria-invalid"], "true");
  }
  assert.equal(ui.requests.filter((request) => request.body).length, 0);
  ui.state.rejectSpeed = true; ui.input("speed-target", "120"); ui.get("apply-speed").emit("click"); await settle(); await ui.poll();
  assert.equal(ui.get("speed-target").value, "120"); assert.match(ui.get("speed-draft-status").textContent, /미적용/);
  assert.match(ui.get("error-banner").textContent, /속도 설정 거부/);
});

test("regional courses show invented terrain and reserve OSM attribution for sourced maps", async () => {
  const ui = await viewer();
  assert.equal(ui.get("region-source").textContent, "지역 특성 합성 코스"); assert.equal(ui.get("map-attribution").hidden, true);
  assert.equal(ui.get("terrain-panel").hidden, false); assert.match(ui.get("terrain-line").attributes.d, /^M/);
  assert.equal(ui.get("terrain-position").attributes.cx, "300.00");
  assert.match(ui.get("terrain-current").textContent, /330.0 m \/ \+4.0%/);
  ui.state.current.world.region = {name: "서울 성수동", provenance: {attribution: "© OpenStreetMap contributors"}};
  await ui.poll();
  assert.equal(ui.get("region-source").textContent, "OSM 실제 지도"); assert.equal(ui.get("map-attribution").hidden, false); assert.equal(ui.get("terrain-panel").hidden, true);
  delete ui.state.current.world.region.provenance;
  await ui.poll(); assert.equal(ui.get("map-attribution").hidden, true);
});

test("performance missing or null values remain uncollected and 100 percent never declares a goal", async () => {
  const ui = await viewer();
  assert.equal(ui.get("performance-status").textContent, "미수집");
  assert.equal(ui.get("performance-path-rmse").textContent, "—");
  assert.equal(ui.get("performance-speed-line").attributes.d, "");
  const performance = performanceFixture();
  performance.route.progress_pct = 100;
  performance.autonomous = {samples: 0, duration_s: 0, distance_m: 0, cross_track_rmse_m: null, speed_rmse_kmh: null, heading_mae_deg: null, safety_eligible_pct: null};
  performance.history = [];
  ui.state.current.performance = performance; ui.state.current.metrics.goal_reached = false;
  await ui.poll();
  assert.equal(ui.get("performance-status").textContent, "자율 표본 없음");
  assert.equal(ui.get("performance-progress").value, "100");
  assert.match(ui.get("performance-progress").attributes["aria-valuetext"], /목표 도달 판정과 별개/);
  assert.equal(ui.get("metric-goal").textContent, "미도달");
  for (const id of ["path-rmse", "speed-rmse", "heading-mae", "safety-eligible"]) assert.equal(ui.get(`performance-${id}`).textContent, "—");
  assert.match(ui.get("performance-mode-note").textContent, /표본이 아직 없습니다/);
});

test("performance shows autonomous accumulation in manual mode and gaps the charts at manual samples", async () => {
  const ui = await viewer();
  ui.state.current.performance = performanceFixture(); await ui.poll();
  assert.equal(ui.get("performance-path-rmse").textContent, "0.32");
  assert.equal(ui.get("performance-speed-rmse").textContent, "3.2");
  assert.equal(ui.get("performance-heading-mae").textContent, "2.0");
  assert.match(ui.get("performance-current-motion").textContent, /제어 목표 40.0 km\/h/);
  assert.match(ui.get("performance-events").textContent, /충돌 0 · 도로 이탈 0 · 개입 1/);
  const speedLine = ui.get("performance-speed-line").attributes.d;
  assert.equal((speedLine.match(/M/g) || []).length, 2);
  assert.equal((ui.get("performance-target-line").attributes.d.match(/M/g) || []).length, 2);
  assert.equal(ui.get("performance-speed-top").textContent, "25");
  assert.equal(ui.get("performance-path-top").textContent, "1.0");
  assert.match(ui.get("performance-speed-chart").attributes["aria-label"], /제어 목표 0.0 km\/h/);
  assert.equal(ui.get("performance-target-dot").attributes.cy, "135.00");
  assert.equal(ui.get("performance-speed-empty").hidden, true);
  ui.state.current.mode = "manual"; await ui.poll();
  assert.match(ui.get("performance-mode-note").textContent, /자율 누적은 멈춰 있습니다/);
  assert.equal(ui.get("performance-path-rmse").textContent, "0.32");
  assert.equal(ui.get("performance-speed-line").attributes.d, speedLine);
  ui.state.current.mode = "autonomous";
  ui.state.current.performance.session = {mixed_mode_session: true, manual_samples: 1, modes_seen: ["autonomous", "manual"]};
  ui.state.current.performance.history = ui.state.current.performance.history.filter((point) => point.mode === "autonomous");
  await ui.poll();
  assert.match(ui.get("performance-mode-note").textContent, /혼합 실행 · 수동 구간 제외/);
  assert.doesNotMatch(ui.get("performance-mode-note").textContent, /수동 조작 중/);
});

test("performance reset clears old statistics and paths and bounds rendered history to the recent window", async () => {
  const ui = await viewer();
  const performance = performanceFixture();
  performance.history = Array.from({length: 201}, (_,index) => ({time_s: index,mode: "autonomous",speed_kmh: index % 50,target_kmh: 50,cross_track_error_m: .1}));
  ui.state.current.performance = performance; await ui.poll();
  assert.equal(ui.get("performance-speed-start").textContent, "80.0 s");
  assert.equal(ui.get("performance-speed-end").textContent, "200.0 s");
  assert.doesNotMatch(ui.get("performance-speed-line").attributes.d, /NaN|Infinity/);
  ui.state.current.session_id = "reset-session";
  ui.state.current.performance = {route: {progress_pct: 0}, autonomous: {samples: 0}, current: {}, history: []};
  await ui.poll();
  assert.equal(ui.get("performance-path-rmse").textContent, "—");
  for (const id of ["speed", "target", "path"]) {
    assert.equal(ui.get(`performance-${id}-line`).attributes.d, "");
    assert.ok(Object.hasOwn(ui.get(`performance-${id}-dot`).attributes, "hidden"));
  }
  assert.equal(ui.get("performance-speed-empty").hidden, false);
  assert.equal(ui.get("performance-path-empty").hidden, false);
  assert.equal(ui.get("performance-progress").value, "0");
});

test("missing lane data stays unavailable without declaring a lane model or successful guard", async () => {
  const ui = await viewer();
  assert.equal(ui.get("lane-source").textContent, "차선 정보 미제공");
  assert.equal(ui.get("lane-inside-pct").textContent, "—");
  assert.equal(ui.get("lane-clearance").textContent, "—");
  assert.equal(ui.get("lane-current-state").textContent, "미제공");
  assert.equal(ui.get("lane-guard-state").textContent, "미제공");
  assert.match(ui.get("lane-autonomous-samples").textContent, /제공하지 않습니다/);
  assert.equal(ui.get("lane-training-status").textContent, "차선 학습 상태 미제공");
  assert.equal(ui.get("lane-map-legend").hidden, true);
  assert.equal(ui.get("lane-map-note").hidden, true);
});

test("initial lane statistics are uncollected while the current assumed body clearance is available", async () => {
  const ui = await viewer();
  const lane = laneFixture();
  lane.autonomous = {samples: 0, duration_s: 0, inside_pct: null, departure_events: 0, outside_time_s: 0};
  lane.guard.checked = false;
  ui.state.current.lane = lane; ui.state.current.world.lane = worldLaneFixture(); await ui.poll();
  assert.match(ui.get("lane-source").textContent, /폭 3.6 m 가정 중심 차로 · 실측 아님/);
  assert.equal(ui.get("lane-inside-pct").textContent, "—");
  assert.equal(ui.get("lane-autonomous-samples").textContent, "자율 표본 없음");
  assert.equal(ui.get("lane-guard-state").textContent, "검사 대기");
  assert.equal(ui.get("lane-guard-state").className, "badge neutral");
  assert.match(ui.get("lane-guard-description").textContent, /자율 제안이 아직 없어/);
  assert.equal(ui.get("lane-autonomous-events").textContent, "차선 이탈 0회 · 이탈 시간 0.0 s");
  assert.equal(ui.get("lane-clearance").textContent, "0.80");
  assert.equal(ui.get("lane-current-state").textContent, "차선 안");
  assert.equal(ui.get("lane-map-legend").hidden, false);
  assert.equal(ui.get("lane-map-label").textContent, "가정 차선 3.6 m");
  assert.equal(ui.get("lane-map-note").hidden, false);
  assert.match(ui.get("lane-training-status").textContent, /차선 모델 미학습/);
  assert.match(ui.get("lane-training-status").textContent, /ego_lane_v1/);
  assert.match(ui.get("lane-training-status").textContent, /학습 실행 중지 · 모델 자동 교체 없음/);
});

test("lane statistics and a passing path guard are explicitly synthetic and independent of road departure", async () => {
  const ui = await viewer();
  ui.state.current.lane = laneFixture(); await ui.poll();
  assert.equal(ui.get("lane-inside-pct").textContent, "100.0");
  assert.equal(ui.get("lane-autonomous-samples").textContent, "자율 10 표본 · 1.0 s");
  assert.equal(ui.get("lane-current-offset").textContent, "중심 편차 0.10 m · 차체 최대 편차 1.00 m");
  assert.equal(ui.get("lane-guard-state").textContent, "통과");
  assert.equal(ui.get("lane-guard-state").className, "badge good");
  assert.match(ui.get("lane-guard-description").textContent, /가정 차선 경계 검사/);
  assert.match(ui.get("lane-guard-description").textContent, /다른 안전 판단도 함께 적용/);
  assert.equal(ui.get("metric-offroad").textContent, "—");
});

test("lane boundary intrusion shows body clearance and veto without confusing it with road departure", async () => {
  const ui = await viewer();
  const lane = laneFixture();
  lane.current = {outside: true, center_offset_m: .9, max_body_offset_m: 1.95, clearance_m: -.15};
  lane.autonomous = {samples: 10, duration_s: 1, inside_pct: 80, departure_events: 1, outside_time_s: .2};
  lane.guard = {allowed: false, reason: "lane_boundary_veto", scope: "synthetic_path_guard"};
  ui.state.current.lane = lane; ui.state.current.world.lane = worldLaneFixture();
  ui.state.current.metrics.offroad_events = 0;
  ui.state.current.safety.reasons = ["lane_boundary_veto"]; await ui.poll();
  assert.equal(ui.get("lane-inside-pct").textContent, "80.0");
  assert.equal(ui.get("lane-current-state").textContent, "차선 밖");
  assert.equal(ui.get("lane-current-state").className, "badge caution");
  assert.equal(ui.get("lane-clearance").textContent, "-0.15");
  assert.ok(ui.get("lane-clearance").classList.contains("lane-outside"));
  assert.ok(ui.get("lane-map-legend").classList.contains("lane-outside"));
  assert.equal(ui.get("lane-autonomous-events").textContent, "차선 이탈 1회 · 이탈 시간 0.2 s");
  assert.equal(ui.get("metric-offroad").textContent, "0");
  assert.equal(ui.get("lane-guard-state").textContent, "거부");
  assert.match(ui.get("lane-guard-description").textContent, /차체가 침범할 수 있어/);
  assert.match(ui.get("safety-reason").textContent, /가정한 자차 차로 경계/);
});

test("manual lane observations change while autonomous accumulation remains fixed and the veto is marked unapplied", async () => {
  const ui = await viewer();
  ui.state.current.lane = laneFixture(); await ui.poll();
  const before = ["lane-inside-pct", "lane-autonomous-events"].map((id) => ui.get(id).textContent);
  ui.state.current.mode = "manual";
  ui.state.current.lane.current = {outside: true, center_offset_m: 2, max_body_offset_m: 2.9, clearance_m: -1.1};
  ui.state.current.lane.guard.allowed = false; ui.state.current.lane.guard.reason = "lane_boundary_veto";
  await ui.poll();
  assert.deepEqual(["lane-inside-pct", "lane-autonomous-events"].map((id) => ui.get(id).textContent), before);
  assert.equal(ui.get("lane-clearance").textContent, "-1.10");
  assert.equal(ui.get("lane-current-state").textContent, "차선 밖");
  assert.match(ui.get("lane-autonomous-samples").textContent, /수동 중 누적 정지/);
  assert.equal(ui.get("lane-guard-state").textContent, "수동 · 미적용");
  assert.equal(ui.get("lane-guard-state").className, "badge neutral");
  assert.match(ui.get("lane-guard-description").textContent, /수동 조작에는 자율 경로 거부를 적용하지 않습니다/);
});

test("lane reset removes previous percentages, departures and boundary warning state", async () => {
  const ui = await viewer();
  ui.state.current.lane = laneFixture();
  ui.state.current.lane.current.outside = true;
  ui.state.current.lane.current.clearance_m = -.2;
  ui.state.current.lane.autonomous = {samples: 20, duration_s: 2, inside_pct: 50, departure_events: 3, outside_time_s: 1};
  await ui.poll();
  const lane = laneFixture();
  lane.autonomous = {samples: 0, duration_s: 0, inside_pct: null, departure_events: 0, outside_time_s: 0};
  ui.state.current.session_id = "lane-reset"; ui.state.current.step = 0; ui.state.current.lane = lane;
  await ui.poll();
  assert.equal(ui.get("lane-inside-pct").textContent, "—");
  assert.equal(ui.get("lane-autonomous-samples").textContent, "자율 표본 없음");
  assert.equal(ui.get("lane-autonomous-events").textContent, "차선 이탈 0회 · 이탈 시간 0.0 s");
  assert.equal(ui.get("lane-current-state").textContent, "차선 안");
  assert.equal(ui.get("lane-clearance").textContent, "0.80");
  assert.equal(ui.get("lane-clearance").classList.contains("lane-outside"), false);
  assert.equal(ui.get("lane-map-legend").classList.contains("lane-outside"), false);
  delete ui.state.current.lane; await ui.poll();
  assert.equal(ui.get("lane-inside-pct").textContent, "—");
  assert.equal(ui.get("lane-autonomous-events").textContent, "차선 이탈 —회 · 이탈 시간 — s");
  assert.equal(ui.get("lane-guard-state").textContent, "미제공");
});

test("map draws both provided assumed boundaries and keeps the legend consistent with drawing and source", async () => {
  const initialState = snapshot();
  Object.assign(initialState.vehicle, {x_m: 0, y_m: 0, yaw_rad: 0});
  Object.assign(initialState.world, {region: null, coordinates: "east_north_m", road_points: [[0,0],[20,0]], road_half_width_m: 6, lane: worldLaneFixture()});
  initialState.lane = laneFixture();
  const ui = await viewer({initialState, drawMap: true});
  const canvas = ui.get("world-canvas").context;
  const laneStrokes = () => canvas.strokes.filter((stroke) => ["#8b70ad", "#b96160"].includes(stroke.color));
  assert.equal(ui.get("connection").className, "connection online");
  assert.equal(laneStrokes().length, 2);
  assert.equal(ui.get("lane-map-legend").hidden, false);
  for (const stroke of laneStrokes()) {
    assert.deepEqual(stroke.dash, [6,4]);
    assert.equal(stroke.points.length, 2);
    assert.ok(stroke.points.flat().every(Number.isFinite));
    assert.equal(stroke.points[0][1], stroke.points[1][1]);
  }
  assert.ok(laneStrokes()[0].points[0][1] < laneStrokes()[1].points[0][1]);
  canvas.strokes.length = 0; ui.state.current.lane.current.outside = true; await ui.poll();
  assert.equal(laneStrokes().length, 2);
  assert.ok(laneStrokes().every((stroke) => stroke.color === "#b96160"));
  assert.ok(ui.get("lane-map-legend").classList.contains("lane-outside"));
  for (const invalid of [{...worldLaneFixture(), source: "unverified"}, {...worldLaneFixture(), right_boundary: []}, null]) {
    canvas.strokes.length = 0; ui.state.current.world.lane = invalid; await ui.poll();
    assert.equal(laneStrokes().length, 0);
    assert.equal(ui.get("lane-map-legend").hidden, true);
    assert.equal(ui.get("lane-map-note").hidden, true);
  }
});

function objectFixture() {
  return {source:"synthetic_geometry_not_learned_detection",status:"available",learned_detector_loaded:false,
    sampled_at_sim_time_s:1,age_s:.1,sensor_range_m:16,hidden_or_out_of_range_count:2,camera_box_count:1,
    safety_track_ids:["car-1"],objects:[{id:"car-1",kind:"vehicle",distance_m:10,relative_forward_m:10,speed_kmh:0,
      x_m:10,y_m:0,near_reference_route:true,confidence:null}]};
}
test("object observations remain synthetic and toggle camera projection without sending driving actions", async () => {
  const initial=snapshot(); initial.object_perception=objectFixture();
  const ui=await viewer({initialState:initial});
  assert.match(ui.get("object-source").textContent,/탐지기 미연결/);
  assert.equal(ui.get("object-count").textContent,"1");
  assert.match(ui.get("object-details").textContent,/신뢰도 —/);
  assert.match(ui.get("camera-image").src,/objects.png/);
  ui.get("camera-image").onload();
  ui.get("object-overlay-toggle").emit("click");
  assert.equal(ui.get("object-overlay-toggle").attributes["aria-pressed"],"false");
  assert.match(ui.get("camera-image").src,/camera.png/);
  assert.equal(ui.requests.filter((request)=>request.body).length,0);
  assert.match(ui.get("object-rows").children[0].children[4].textContent,/합성 추적 제공/);
});
test("missing, frozen and dropped object observations do not masquerade as fresh detections", async () => {
  const ui=await viewer();
  assert.equal(ui.get("object-count").textContent,"—");
  assert.equal(ui.get("object-overlay-toggle").disabled,true);
  ui.state.current.object_perception={...objectFixture(),frame_frozen:true,age_s:3}; await ui.poll();
  assert.match(ui.get("object-status").textContent,/고정된 과거 프레임/);
  ui.state.current.object_perception.status="camera_unavailable"; ui.state.current.object_perception.objects=[]; await ui.poll();
  assert.equal(ui.get("object-count").textContent,"—");
  assert.equal(ui.get("object-rows").children.length,0);
  assert.match(ui.get("object-status").textContent,/손실/);
});
test("object rows are bounded, distance sorted, and render actor IDs as text", async () => {
  const initial=snapshot(),view=objectFixture();
  view.objects=Array.from({length:12},(_,index)=>({...view.objects[0],id:index===11?"<img onerror=x>":`actor-${index}`,distance_m:12-index}));
  initial.object_perception=view; const ui=await viewer({initialState:initial});
  assert.equal(ui.get("object-rows").children.length,8);
  assert.match(ui.get("object-rows").children[0].children[0].textContent,/<img onerror=x>/);
  assert.match(ui.get("object-details").textContent,/8 \/ 12/);
  delete ui.state.current.object_perception; await ui.poll();
  assert.equal(ui.get("object-rows").children.length,0);
  assert.equal(ui.get("object-count").textContent,"—");
});

test("traffic signals and assigned lanes are distinct from legacy background actors", async () => {
  const initial=snapshot();
  initial.world.road_lanes={source:"synthetic_lane_assignment_not_measured",lane_count:2};
  initial.world.traffic={signal:{vehicles:"red",pedestrians:"walk",phase_s:11,cycle_s:24},pedestrian_state:"crossing"};
  const ui=await viewer({initialState:initial});
  assert.match(ui.get("lane-assignment-status").textContent,/왕복 2차로/);
  assert.match(ui.get("traffic-signal-status").textContent,/차량 신호 정지.*보행 신호 횡단/);
  assert.match(ui.get("traffic-pedestrian-status").textContent,/횡단 중/);
  assert.match(ui.get("traffic-scope-note").textContent,/보수적 원형/);
  delete ui.state.current.world.traffic; await ui.poll();
  assert.match(ui.get("traffic-kind").textContent,/기존 단순/);
  assert.equal(ui.get("traffic-signal-status").className,"");
});


test("engineering progress is distinct from vehicle validation and renders text safely", async () => {
  const initial = snapshot();
  initial.improvement_progress = {implemented_items: 1, partial_items: 7, total_items: 8, measured_validation_complete: 0,
    updated_at: "2026-10-24", automated_verification: {status: "passed", python_passed: 196, ui_passed: 18},
    items: [{title: "<img onerror=alert(1)>", status: "partial", detail: "code", remaining: "measured test"}]};
  const ui = await viewer({initialState: initial});
  assert.match(ui.get("improvement-count").textContent, /전체 8/);
  assert.match(ui.get("improvement-verification").textContent, /실차 검증 완료 0개/);
  const row = ui.get("improvement-rows").children[0];
  assert.equal(row.children[0].textContent, "<img onerror=alert(1)>");
  delete ui.state.current.improvement_progress; await ui.poll();
  assert.equal(ui.get("improvement-rows").children.length, 0);
  assert.match(ui.get("improvement-count").textContent, /정보 없음/);
});
