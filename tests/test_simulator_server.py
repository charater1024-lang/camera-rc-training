"""Protocol, pause, deadman, and failure boundaries; no model or GPU required."""
import http.client
import json
import threading

from PIL import Image
import pytest

from camera_rc.simulator.server import SimulatorApp, SimulatorHTTPServer


class StubEngine:
    def __init__(self):
        self.steps = 0
        self.finished = False
        self.mode = "manual"
        self.last_controls = None
        self.fail = False
        self.closed = False

    def snapshot(self):
        return {"step": self.steps, "finished": self.finished, "mode": self.mode}

    def step(self, controls=None):
        if self.fail:
            raise RuntimeError("private path or model internals")
        self.steps += 1
        self.last_controls = controls
        return self.snapshot()

    def set_mode(self, mode):
        self.mode = mode

    def reset(self, **kwargs):
        self.steps = 0
        self.finished = False
        self.fail = False
        if kwargs.get("mode"):
            self.mode = kwargs["mode"]

    def report(self):
        return self.snapshot()

    def frame(self, kind):
        return Image.new("RGB", (32, 16), "green")

    def close(self):
        self.closed = True


def make_app():
    clock = [10.]
    engine = StubEngine()
    return SimulatorApp(engine, [{"id": "clear", "title": "Clear"}], clock=lambda: clock[0]), engine, clock


def test_paused_start_step_and_mode_do_not_accidentally_advance():
    app, engine, _ = make_app()
    app.tick()
    assert engine.steps == 0
    app.action({"action": "start"})
    app.tick()
    assert engine.steps == 1
    app.action({"action": "mode", "mode": "autonomous"})
    app.tick()
    assert engine.steps == 1 and not app.running
    app.action({"action": "step"})
    assert engine.steps == 2 and not app.running
    app.action({"action": "start"})
    app.action({"action": "pause"})
    app.tick()
    assert engine.steps == 2


def test_manual_deadman_and_pause_discard_old_throttle():
    app, engine, clock = make_app()
    app.action({"action": "controls", "steering": 1, "throttle": .7, "brake": 0})
    app.action({"action": "start"})
    app.tick()
    assert engine.last_controls["throttle"] == .7
    clock[0] += .751
    app.tick()
    assert engine.last_controls["throttle"] == 0 and engine.last_controls["brake"] == 1
    app.action({"action": "controls", "throttle": .9, "brake": 0})
    app.action({"action": "pause"})
    app.action({"action": "step"})
    assert engine.last_controls["brake"] == 1


def test_failure_stops_without_fallback_and_reset_is_explicit():
    app, engine, _ = make_app()
    engine.fail = True
    app.action({"action": "start"})
    app.tick()
    assert not app.running and app.error and engine.steps == 0
    assert "private path" not in app.error
    with pytest.raises(ValueError):
        app.action({"action": "start"})
    app.action({"action": "reset", "scenario": "clear", "seed": 1})
    assert app.error is None and not app.running
    engine.finished = True
    with pytest.raises(ValueError):
        app.action({"action": "step"})


def test_reset_changes_frame_cache_identity_even_at_same_step():
    app, engine, _ = make_app()
    old = app.snapshot()["session_id"]
    fresh = app.action({"action": "reset", "scenario": "clear", "seed": 42})
    assert fresh["session_id"] != old and fresh["step"] == 0


@pytest.mark.parametrize("message", [
    {"action": "controls", "throttle": float("nan")},
    {"action": "controls", "brake": float("inf")},
    {"action": "controls", "steering": True},
    {"action": "controls", "steering": 2},
    {"action": "controls", "throttle": -1},
    {"action": "reset", "seed": True},
    {"action": "reset", "seed": -1},
    {"action": "reset", "scenario": "../private"},
    {"action": "reset", "output_dir": "../private"},
    {"action": "mode", "mode": "physical"},
    {"action": "start", "actuation_enabled": True},
    {"action": "exec", "command": "anything"}, [], None,
])
def test_reject_unsupported_inputs_without_advancing(message):
    app, engine, _ = make_app()
    with pytest.raises((ValueError, TypeError)):
        app.action(message)
    assert engine.steps == 0


@pytest.fixture
def http_server():
    app, engine, _ = make_app()
    server = SimulatorHTTPServer(app, 0)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    yield server, engine
    server.shutdown()
    server.server_close()
    worker.join()
    app.close()


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    status, data, result_headers = response.status, response.read(), dict(response.getheaders())
    conn.close()
    return status, data, result_headers


def test_loopback_http_state_images_and_download(http_server):
    server, engine = http_server
    assert server.server_address[0] == "127.0.0.1"
    status, data, headers = request(server, "GET", "/api/state")
    assert status == 200 and not json.loads(data)["physical_output"]
    assert "Access-Control-Allow-Origin" not in headers
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    status, data, headers = request(server, "GET", "/frames/camera.png?step=2")
    assert status == 200 and data.startswith(b"\x89PNG")
    status, data, headers = request(server, "GET", "/frames/objects.png?step=2")
    assert status == 200 and data.startswith(b"\x89PNG")
    status, data, headers = request(server, "GET", "/api/report")
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    status, data, headers = request(server, "POST", "/api/action", json.dumps({"action": "step"}),
                                  {"X-Simulator-Client": "1", "Content-Type": "application/json"})
    assert status == 200 and json.loads(data)["step"] == 1 and engine.steps == 1


@pytest.mark.parametrize("headers,expected", [
    ({}, 403),
    ({"X-Simulator-Client": "1", "Content-Type": "text/plain"}, 415),
    ({"X-Simulator-Client": "1", "Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
    ({"X-Simulator-Client": "1", "Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"}, 403),
    ({"X-Simulator-Client": "1", "Content-Type": "application/json", "Host": "evil.example"}, 403),
])
def test_http_cross_origin_actions_are_rejected(http_server, headers, expected):
    server, engine = http_server
    status, _, _ = request(server, "POST", "/api/action", '{"action":"step"}', headers)
    assert status == expected and engine.steps == 0


@pytest.mark.parametrize("body", ['{"action":"controls","throttle":NaN}', '{bad', '[]', ' ' * 8193])
def test_invalid_json_size_does_not_advance(http_server, body):
    server, engine = http_server
    status, _, _ = request(server, "POST", "/api/action", body,
                           {"X-Simulator-Client": "1", "Content-Type": "application/json"})
    assert status == 400 and engine.steps == 0


def test_no_arbitrary_paths_or_rebinding(http_server):
    server, _ = http_server
    for path in ("/../../pyproject.toml", "/assets/../../../private", "/api/serial", "/api/weights"):
        assert request(server, "GET", path)[0] == 404
    assert request(server, "GET", "/api/state", headers={"Host": "evil.example"})[0] == 403


def test_cli_preserves_existing_results_before_model_load(tmp_path, monkeypatch):
    from camera_rc.simulator import cli
    existing = tmp_path / "previous"
    existing.mkdir()
    marker = existing / "evidence.json"
    marker.write_text('{"keep":true}')
    def fail_if_loaded(*args):
        raise AssertionError("Must reject the existing output before model load")
    monkeypatch.setattr(cli, "load_perception", fail_if_loaded)
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--output", str(existing)])
    assert exc.value.code == 2 and marker.read_text() == '{"keep":true}'


def test_cli_rejects_unused_detector_before_loading_weights(tmp_path):
    from camera_rc.simulator.cli import load_perception
    config = tmp_path / "config.json"
    config.write_text('{"perception":{"detection_checkpoint":"missing.pt"}}')
    with pytest.raises(ValueError, match="semantics only"):
        load_perception(config)



def test_improvement_api_is_read_only_and_does_not_claim_vehicle_validation(http_server):
    server,engine=http_server
    status,body,_=request(server,"GET","/api/improvements")
    progress=json.loads(body)
    assert status==200 and progress["total_items"]==8 and engine.steps==0
    assert progress["measured_validation_complete"]==0 and not progress["physical_output"]
    status,body,_=request(server,"GET","/api/state")
    assert "improvement_progress" in json.loads(body)
