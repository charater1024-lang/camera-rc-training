"""Bounded latest-frame bundles. Host receive timing is NOT exposure sync."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import validate_camera_group
from .sources import CameraSource, Frame


@dataclass
class FrameBundle:
    views: dict[str, Frame]
    sequence: int
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0

    @property
    def received_at(self):
        return min(frame.received_at for frame in self.views.values())

    @property
    def skew_s(self):
        return max(frame.received_at for frame in self.views.values()) - self.received_at


class MultiCameraSource:
    """Retain at most one frame per camera; never silently omit a required view."""
    def __init__(self, cfg, *, camera_factory=None):
        validate_camera_group(cfg)
        self.sources, self.latest, self.delivered = {}, {}, {}
        self.sequence = 0
        self.max_skew_s = cfg["max_skew_s"]
        self.error = None
        factory = camera_factory or CameraSource
        try:
            for camera in cfg["cameras"]:
                self.sources[camera["name"]] = factory(camera)
        except BaseException:
            self.close()
            raise

    def read(self):
        for name, source in self.sources.items():
            if source.exhausted:
                self.error = f"camera_failed:{name}:{getattr(source, 'error', None)}"
                return None
            frame = source.read()
            if frame is not None:
                if not math.isfinite(frame.received_at) or frame.received_at > time.monotonic():
                    self.error = f"invalid_camera_timestamp:{name}"
                    return None
                self.latest[name] = frame
        if set(self.latest) != set(self.sources):
            return None
        if any(frame.sequence <= self.delivered.get(name, -1) for name, frame in self.latest.items()):
            return None
        bundle = FrameBundle(dict(self.latest), self.sequence + 1)
        if bundle.skew_s > self.max_skew_s:
            return None  # Runtime expiry/timeout holds STOP while waiting.
        self.sequence += 1
        self.delivered = {name: frame.sequence for name, frame in self.latest.items()}
        return bundle

    @property
    def exhausted(self):
        return self.error is not None

    def close(self):
        for source in self.sources.values():
            source.close()


class MultiReplaySource:
    """One JSONL row per capture group; negative offsets simulate receive skew."""
    def __init__(self, cfg):
        validate_camera_group(cfg)
        self.names = {camera["name"] for camera in cfg["cameras"]}
        self.max_skew_s = cfg["max_skew_s"]
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
            views = row.get("views")
            if not isinstance(views, dict) or set(views) != self.names:
                raise ValueError("Replay group must contain exactly all configured camera views")
            images, offsets = {}, {}
            for name, view in views.items():
                offset = view.get("offset_s", 0)
                if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset > 0:
                    raise ValueError("Replay receive offset_s must be finite and <=0")
                offsets[name] = float(offset)
                target = Path(view["image"])
                target = target if target.is_absolute() else self.path.parent / target
                with Image.open(target) as image:
                    images[name] = np.array(image.convert("RGB"))
            if max(offsets.values()) - min(offsets.values()) > self.max_skew_s:
                raise ValueError("Replay camera skew exceeds configured limit")
            self.sequence += 1
            now = time.monotonic()
            frames = {name: Frame(rgb, now + offsets[name], self.sequence) for name, rgb in images.items()}
            speed, yaw = float(row.get("speed_mps", 0)), float(row.get("yaw_rate_rps", 0))
            if not math.isfinite(speed) or not math.isfinite(yaw):
                raise ValueError("Nonfinite replay telemetry")
            return FrameBundle(frames, self.sequence, speed, yaw)
        self.exhausted = True
        return None

    def close(self):
        self.stream.close()
