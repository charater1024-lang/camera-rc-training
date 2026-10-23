"""Loopback-only interactive simulator. No camera, serial, or hardware routes."""
from __future__ import annotations

import io
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import threading
import time
from urllib.parse import urlsplit
import uuid


class SimulatorApp:
    """One serialized engine; wall-clock pacing never changes fixed physics dt."""

    def __init__(self, engine, scenarios, *, clock=time.monotonic):
        self.engine = engine
        self.scenarios = list(scenarios)
        self.clock = clock
        self.lock = threading.RLock()
        self.running = False
        self.error = None
        self.session_id = uuid.uuid4().hex
        self.controls = {"steering": 0., "throttle": 0., "brake": 1.}
        self.controls_at = -math.inf
        self.reset_pending = False
        self.closed = threading.Event()
        self.thread = None

    def _stop_controls(self):
        self.controls = {"steering": 0., "throttle": 0., "brake": 1.}
        self.controls_at = -math.inf
        self.reset_pending = False

    def snapshot(self):
        with self.lock:
            state = dict(self.engine.snapshot())
            state.update(running=self.running, server_error=self.error,
                         session_id=self.session_id, scenarios=self.scenarios,
                         physical_output=False, manual_deadman_s=0.75)
            return state

    def report(self):
        with self.lock:
            report = dict(self.engine.report())
            report.update(server_error=self.error, session_id=self.session_id, physical_output=False)
            return report

    def frame(self, kind):
        with self.lock:
            frame = self.engine.frame(kind).copy()
        stream = io.BytesIO()
        frame.save(stream, format="PNG")
        return stream.getvalue()

    def _step(self):
        controls = dict(self.controls)
        age = self.clock() - self.controls_at
        if not 0 <= age <= .75:
            controls = {"steering": 0., "throttle": 0., "brake": 1.}
        controls["reset_safety"] = self.reset_pending
        self.reset_pending = False
        try:
            state = self.engine.step(controls)
            if state.get("finished"):
                self.running = False
                self._stop_controls()
        except Exception as exc:
            # Do not mask a learned-model/engine failure with oracle perception.
            self.error = f"{type(exc).__name__}: simulation stopped; reset required"
            self.running = False
            self._stop_controls()

    def tick(self):
        with self.lock:
            if self.running:
                self._step()

    def action(self, message):
        if not isinstance(message, dict) or not isinstance(message.get("action"), str):
            raise ValueError("Expected an action object")
        action = message["action"]
        allowed = {
            "start": {"action"}, "pause": {"action"}, "step": {"action"},
            "reset": {"action", "scenario", "seed", "mode"}, "mode": {"action", "mode"},
            "controls": {"action", "steering", "throttle", "brake"},
            "reset_safety": {"action"},
        }
        if action not in allowed or set(message) - allowed[action]:
            raise ValueError("Unsupported action or field")
        with self.lock:
            if action == "pause":
                self.running = False
                self._stop_controls()
            elif action == "reset":
                scenario = message.get("scenario")
                if scenario is not None and scenario not in {s["id"] for s in self.scenarios}:
                    raise ValueError("Unknown scenario")
                seed = message.get("seed")
                if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**31-1):
                    raise ValueError("Seed must be an integer in 0..2147483647")
                mode = message.get("mode")
                if mode is not None and mode not in {"autonomous", "manual"}:
                    raise ValueError("Unknown mode")
                self.running = False
                self._stop_controls()
                self.engine.reset(scenario=scenario, seed=seed, mode=mode)
                self.session_id = uuid.uuid4().hex
                self.error = None
            elif action == "mode":
                if message.get("mode") not in {"autonomous", "manual"}:
                    raise ValueError("Unknown mode")
                self.running = False
                self._stop_controls()
                self.engine.set_mode(message["mode"])
            elif action == "controls":
                values = {k: message.get(k, default) for k, default in
                          (("steering", 0.), ("throttle", 0.), ("brake", 1.))}
                for key, value in values.items():
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("Controls must be finite numbers")
                    if not (-1 if key == "steering" else 0) <= value <= 1:
                        raise ValueError("Control outside allowed range")
                self.controls, self.controls_at = values, self.clock()
            elif action == "reset_safety":
                self.reset_pending = True
            elif action in {"start", "step"}:
                if self.error or self.engine.snapshot().get("finished"):
                    raise ValueError("Reset the finished or failed episode first")
                if action == "start":
                    self.running = True
                else:
                    self.running = False
                    self._step()
            return self.snapshot()

    def start_worker(self):
        if self.thread is not None:
            raise RuntimeError("Simulation worker already started")
        def worker():
            while not self.closed.wait(.1):
                self.tick()
        self.thread = threading.Thread(target=worker, name="offline-simulator", daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            self.running = False
            self._stop_controls()
            self.engine.close()


class SimulatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, app, port=8766):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid loopback port")
        self.app = app
        super().__init__(("127.0.0.1", port), SimulatorHandler)


class SimulatorHandler(BaseHTTPRequestHandler):
    server_version = "OfflineSimulator/1"
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        pass

    def _respond(self, status, body, mime="application/json; charset=utf-8", *, download=False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="simulation-report.json"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, status, value, *, download=False):
        self._respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), download=download)

    def _valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self._valid_host():
            self._json(403, {"error": "Loopback Host required"})
            return
        path = urlsplit(self.path).path
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        try:
            if path in assets:
                name, mime = assets[path]
                self._respond(200, files("camera_rc.simulator").joinpath("assets", name).read_bytes(), mime)
            elif path == "/api/state":
                self._json(200, self.server.app.snapshot())
            elif path == "/api/scenarios":
                self._json(200, self.server.app.scenarios)
            elif path == "/api/report":
                self._json(200, self.server.app.report(), download=True)
            elif path in {"/frames/camera.png", "/frames/segmentation.png"}:
                self._respond(200, self.server.app.frame(path.split("/")[-1].split(".")[0]), "image/png")
            else:
                self._json(404, {"error": "Not found"})
        except Exception:
            self._json(500, {"error": "Cannot read simulation state"})

    def do_POST(self):
        port = self.server.server_port
        allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        origin = self.headers.get("Origin")
        if (not self._valid_host() or self.headers.get("X-Simulator-Client") != "1"
                or (origin is not None and origin not in allowed_origins)
                or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self._json(403, {"error": "Same-origin simulator client required"})
            return
        if urlsplit(self.path).path != "/api/action":
            self._json(404, {"error": "Not found"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._json(415, {"error": "JSON body required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= 8192 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Body must be 1..8192 bytes")
            self.connection.settimeout(3)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete JSON body")
            message = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            result = self.server.app.action(message)
            self._json(200, result)
        except (ValueError, TypeError, UnicodeError, TimeoutError, RecursionError):
            self._json(400, {"error": "Invalid action or episode requires reset"})
        except Exception:
            with self.server.app.lock:
                self.server.app.running = False
                self.server.app._stop_controls()
            self._json(500, {"error": "Simulation action failed; paused"})


def serve(engine, scenarios, port=8766):
    app = SimulatorApp(engine, scenarios)
    server = SimulatorHTTPServer(app, port)
    app.start_worker()
    print(f"Offline simulator: http://127.0.0.1:{server.server_port}/ (paused; no physical output)", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
