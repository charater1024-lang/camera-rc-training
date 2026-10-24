"""Newest-frame camera acquisition and bounded recorded-input playback."""
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray
    received_at: float
    sequence: int
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    exposure_at_s: float | None = None
    clock_id: str = "host_monotonic"
    timing_source: str = "host_receive_unverified"


class CameraSource:
    def __init__(self, cfg):
        import cv2
        from .config import camera_device_identity
        self.cv2 = cv2
        device = cfg.get("device", cfg.get("index", 0))
        camera_device_identity(device)
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")
        if cfg.get("fourcc"):
            if not isinstance(cfg["fourcc"], str) or len(cfg["fourcc"]) != 4:
                self.cap.release()
                raise ValueError("Camera fourcc must contain four characters")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        for prop, key in ((cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"), (cv2.CAP_PROP_FPS, "fps")):
            if key in cfg:
                self.cap.set(prop, cfg[key])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock, self.done = threading.Lock(), threading.Event()
        self.latest = None
        self.sequence, self.delivered = 0, -1
        self.error = None
        self.thread = threading.Thread(target=self._capture, name="newest-camera-frame", daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.done.is_set():
                ok, bgr = self.cap.read()
                if not ok:
                    self.error = "camera_read_failed"
                    break
                received = time.monotonic()
                rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.sequence += 1
                    self.latest = Frame(rgb, received, self.sequence)
        except Exception as exc:
            self.error = f"camera_error:{exc}"

    def read(self):
        with self.lock:
            if self.latest is None or self.latest.sequence == self.delivered:
                return None
            self.delivered = self.latest.sequence
            return self.latest

    @property
    def exhausted(self):
        return self.error is not None

    def close(self):
        self.done.set()
        self.cap.release()
        self.thread.join(timeout=1)


class ReplaySource:
    """JSONL images with optional measured speed; never permitted for actuation."""
    def __init__(self, cfg):
        self.path = Path(cfg["path"])
        self.stream = self.path.open(encoding="utf-8-sig")
        self.sequence = 0
        self.exhausted = False

    def read(self):
        from PIL import Image
        for line in self.stream:
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["image"])
            path = path if path.is_absolute() else self.path.parent / path
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
            self.sequence += 1
            return Frame(rgb, time.monotonic(), self.sequence, float(row.get("speed_mps", 0)), float(row.get("yaw_rate_rps", 0)))
        self.exhausted = True
        return None

    def close(self):
        self.stream.close()


class Rectifier:
    def __init__(self, calibration):
        self.size = tuple(calibration["image_size"])
        self.k = calibration.get("camera_matrix")
        self.d = calibration.get("distortion_coefficients")
        if (self.k is None) != (self.d is None):
            raise ValueError("camera_matrix and distortion_coefficients must be supplied together")
        self.maps = None
        self.valid_pixels = np.ones(self.size, dtype=bool)
        if self.k is not None:
            import cv2
            k, d = np.asarray(self.k, dtype=np.float64), np.asarray(self.d, dtype=np.float64)
            if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError("Invalid lens calibration")
            h, w = self.size
            self.maps = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_32FC1)
            mx, my = self.maps
            self.valid_pixels = (mx >= 0) & (my >= 0) & (mx < w - 1) & (my < h - 1)
        elif calibration.get("input_is_rectified") is not True:
            raise ValueError("Supply lens calibration or explicitly declare input_is_rectified=true")

    def __call__(self, rgb):
        if tuple(rgb.shape[:2]) != self.size:
            raise ValueError(f"Camera resolution {rgb.shape[:2]} differs from calibration {self.size}; do not resize calibration blindly")
        if self.maps is None:
            return rgb
        import cv2
        return cv2.remap(rgb, *self.maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def open_source(cfg):
    if cfg["kind"] == "camera":
        return CameraSource(cfg)
    if cfg["kind"] == "replay":
        return ReplaySource(cfg)
    if cfg["kind"] in ("multicamera", "multireplay"):
        from .multisource import MultiCameraSource, MultiReplaySource
        return MultiCameraSource(cfg) if cfg["kind"] == "multicamera" else MultiReplaySource(cfg)
    raise ValueError("Unsupported source.kind")
