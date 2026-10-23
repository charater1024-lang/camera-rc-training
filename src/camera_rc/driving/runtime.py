"""Camera -> perception -> local path -> feedback controller -> bounded transport.

The 50 Hz control loop expires plans independently of camera/model work. Serial
MCU watchdog enforcement remains necessary if the host or Python process hangs.
"""
from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time

from .config import load_runtime_config, check_actuation
from .control import Controller
from .transport import DryRunTransport, SerialTransport
from .types import VehicleState
from .sources import open_source
from .resources import configure_device, ResourceMonitor, latency_report


class ReplayTransport(DryRunTransport):
    """Dry-run only: recorded feedback is labelled and never sent to an actuator."""
    def __init__(self):
        super().__init__()
        self.state = None

    def set_frame(self, frame):
        self.state = VehicleState(frame.received_at, frame.speed_mps, frame.yaw_rate_rps, True, False)

    def read_state(self, now):
        return self.state


class ControlLoop:
    def __init__(self, cfg, transport, output):
        self.cfg, self.transport, self.output = cfg, transport, Path(output)
        self.controller = Controller(cfg["controller"])
        self.lock, self.done = threading.Lock(), threading.Event()
        self.plan = None
        self.plan_time = None
        self.state = None
        self.command = self.controller.stop("startup")
        self.error = None
        self.latched_fault = None
        self.counts = Counter()
        self.ticks = 0
        self.periods_ms = deque(maxlen=10000)
        self.last_tick = None
        self.shutdown_stop_sent = False
        self.thread = threading.Thread(target=self._run, name="vehicle-control", daemon=True)

    @staticmethod
    def _valid_capture_time(value):
        """Require an actual host-monotonic timestamp, never a missing sentinel."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value) and value >= 0
        except OverflowError:
            return False

    def publish(self, plan, captured_at):
        with self.lock:
            if not self._valid_capture_time(captured_at):
                self.latched_fault = self.latched_fault or "plan_capture_time_invalid_restart_required"
                self.plan, self.plan_time = None, None
                return
            self.plan, self.plan_time = plan, captured_at

    def fail(self, reason):
        with self.lock:
            self.latched_fault = reason

    def snapshot(self):
        with self.lock:
            return self.state, self.command

    def step(self, now):
        if self.last_tick is not None:
            self.periods_ms.append((now - self.last_tick) * 1000)
        self.last_tick = now
        state = self.transport.read_state(now)
        with self.lock:
            plan, stamp = self.plan, self.plan_time
            valid_stamp = self._valid_capture_time(stamp)
            # Check again at the command boundary: a plan without capture time
            # must not silently bypass expiry, even if published by another API.
            if plan is not None and not valid_stamp:
                self.latched_fault = self.latched_fault or "plan_capture_time_invalid_restart_required"
            elif valid_stamp and now < stamp:
                self.latched_fault = self.latched_fault or "plan_from_future_restart_required"
            elif valid_stamp and now - stamp > self.cfg["runtime"]["plan_timeout_s"]:
                self.latched_fault = self.latched_fault or "plan_expired_restart_required"
            if self.state is not None and state is None and self.cfg["transport"]["kind"] == "serial":
                self.latched_fault = "telemetry_lost_restart_required"
            self.state = state
            fault = self.latched_fault
        if fault:
            command = self.controller.stop(fault)
        elif plan is None:
            command = self.controller.stop("waiting_for_first_plan")
        else:
            command = self.controller.update(plan, state, now)
        actual = self.transport.send(command, now)
        with self.lock:
            self.command = actual
        self.counts[actual.reason] += 1
        self.ticks += 1
        return {"monotonic_s": now, "state": asdict(state) if state is not None else None,
                "command": asdict(actual), "plan_age_ms": (now - stamp) * 1000 if valid_stamp else None,
                "physical_output": self.cfg["transport"]["kind"] == "serial"}

    def _run(self):
        period = 1 / self.cfg["runtime"]["control_hz"]
        try:
            self.transport.send(self.controller.stop("startup"), time.monotonic())
            next_tick = time.monotonic()
            with (self.output / "control.jsonl").open("w", encoding="utf-8") as stream:
                while not self.done.is_set():
                    begin = time.monotonic()
                    stream.write(json.dumps(self.step(begin), allow_nan=False) + "\n")
                    stream.flush()
                    # Absolute deadlines avoid accumulating Windows timer rounding
                    # and serialization time on every control tick.
                    next_tick += period
                    now = time.monotonic()
                    if now - next_tick > period:
                        next_tick = now + period
                    self.done.wait(max(0, next_tick - now))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.transport.send(self.controller.stop("shutdown"), time.monotonic())
                self.shutdown_stop_sent = True
            except Exception as exc:
                self.error = self.error or f"Stop write failed: {exc}; MCU watchdog must stop"

    def start(self):
        self.thread.start()

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error = "Control worker did not exit; MCU watchdog must stop"
        self.transport.close()


def run_drive(config_path, output, *, actuate=False, max_frames=None, duration_s=None):
    from .pipeline import DrivingPipeline
    cfg = load_runtime_config(config_path)
    check_actuation(cfg, actuate)
    if max_frames is not None and max_frames < 1 or duration_s is not None and duration_s <= 0:
        raise ValueError("max_frames and duration_s must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    device = configure_device(cfg)
    monitor = ResourceMonitor(device, cfg["hardware"]["memory_architecture"])
    monitor.check_limits(cfg["runtime"])
    pipeline = DrivingPipeline(cfg, device)
    # Warm up before opening any actuator connection.
    pipeline.warmup()
    monitor.check_limits(cfg["runtime"])
    source = open_source(cfg["source"])
    transport = control = None
    started = time.monotonic()
    frames = 0
    latencies, ages = deque(maxlen=10000), deque(maxlen=10000)
    error, termination = None, "source_finished"
    try:
        transport = SerialTransport(cfg["transport"]) if actuate else (ReplayTransport() if cfg["source"]["kind"] in ("replay", "multireplay") else DryRunTransport())
        control = ControlLoop(cfg, transport, output)
        control.start()
        last_received = started
        with (output / "perception.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                begin = time.monotonic()
                if control.error:
                    raise RuntimeError(control.error)
                if duration_s is not None and begin - started >= duration_s:
                    termination = "duration_reached"
                    break
                if max_frames is not None and frames >= max_frames:
                    termination = "frame_limit_reached"
                    break
                monitor.check_limits(cfg["runtime"])
                frame = source.read()
                if frame is None:
                    if source.exhausted:
                        if getattr(source, "error", None):
                            raise RuntimeError(source.error)
                        break
                    # Startup gives the camera time to open, with STOP held.
                    timeout = cfg["runtime"]["frame_timeout_s"] if frames else 5.0
                    if begin - last_received > timeout:
                        raise RuntimeError("camera_timeout_restart_required")
                    time.sleep(0.002)
                    continue
                last_received = frame.received_at
                if begin - frame.received_at > cfg["runtime"]["frame_timeout_s"]:
                    raise RuntimeError("stale_camera_frame")
                if isinstance(transport, ReplayTransport):
                    transport.set_frame(frame)
                measured, _ = control.snapshot()
                speed = measured.speed_mps if measured else frame.speed_mps
                result = pipeline.run(frame, speed)
                plan = result.plan
                monitor.check_limits(cfg["runtime"])
                now = time.monotonic()
                elapsed = now - begin
                age = now - frame.received_at
                latencies.append(elapsed * 1000)
                ages.append(age * 1000)
                if elapsed > cfg["runtime"]["max_processing_s"] or age > cfg["runtime"]["plan_timeout_s"]:
                    control.fail("perception_deadline_missed_restart_required")
                else:
                    control.publish(plan, frame.received_at)
                frames += 1
                stream.write(json.dumps({"frame": frame.sequence, "pipeline_ms": elapsed * 1000,
                                        "frame_age_ms": age * 1000, "inference_ms": result.inference_ms,
                                        "views": result.views, "camera_receive_skew_ms": getattr(frame, "skew_s", 0) * 1000,
                                        "plan_valid": plan.valid, "plan_reason": plan.reason,
                                        "target_speed_mps": plan.target_speed_mps, "points": plan.points.tolist()}, allow_nan=False) + "\n")
                stream.flush()
                delay = 1 / cfg["runtime"]["perception_hz"] - (time.monotonic() - begin)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        termination = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        termination = "fault"
        if control:
            control.fail(error)
    finally:
        if control:
            control.close()
            error = error or control.error
            if control.latched_fault:
                error = error or control.latched_fault
                termination = "fault"
        elif transport:
            transport.close()
        source.close()
        summary = {"success": error is None, "termination": termination, "error": error,
                   "camera_names": list(pipeline.calibrations), "shared_model_instances": 1,
                   "physical_output": bool(actuate), "frames": frames, "duration_s": time.monotonic() - started,
                   "pipeline_latency_last_10000_frames": latency_report(latencies),
                   "frame_age_last_10000_frames": latency_report(ages), "resources": monitor.report(),
                   "control_ticks": control.ticks if control else 0,
                   "control_period_ms_last_10000": latency_report(control.periods_ms) if control else {},
                   "control_reasons": dict(control.counts) if control else {},
                   "latched_fault": control.latched_fault if control else None,
                   "shutdown_stop_sent": control.shutdown_stop_sent if control else False,
                   "note": "Camera timestamp is host receive time, not exposure time. Multiview plan age uses the oldest required view; bounded receive skew does not prove exposure synchronization. Dry-run camera has no measured telemetry and holds STOP. Replay feedback is recorded, not physical."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
