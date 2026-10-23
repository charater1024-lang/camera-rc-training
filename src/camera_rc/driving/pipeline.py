"""One shared model instance, sequential batch-one inference across camera views."""
from dataclasses import dataclass

import numpy as np

from .multisource import FrameBundle
from .perception import CameraPerception
from .planner import LocalPlanner
from .sources import Rectifier
from .types import Plan


@dataclass
class PipelineResult:
    plan: Plan
    inference_ms: float
    views: dict


class DrivingPipeline:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.multiview = cfg["source"]["kind"] in ("multicamera", "multireplay")
        self.calibrations = cfg["calibrations"] if self.multiview else {"front": cfg["calibration"]}
        self.rectifiers = {name: Rectifier(cal) for name, cal in self.calibrations.items()}
        if self.multiview:
            from .multiview import MultiViewPlanner
            self.planner = MultiViewPlanner(cfg["planner"], self.calibrations)
        else:
            self.planner = LocalPlanner(cfg["planner"], cfg["calibration"])
        self.model = CameraPerception(cfg["perception"], device=device)

    def warmup(self):
        for h, w in {tuple(cal["image_size"]) for cal in self.calibrations.values()}:
            self.model.infer(np.zeros((h, w, 3), dtype=np.uint8))

    def run(self, frame, speed_mps):
        if self.multiview:
            if not isinstance(frame, FrameBundle) or set(frame.views) != set(self.calibrations):
                raise ValueError("Every configured view is required for multiview planning")
            if frame.skew_s > self.cfg["source"]["max_skew_s"]:
                raise ValueError("Camera receive skew exceeds configured limit")
            frames = frame.views
        else:
            if isinstance(frame, FrameBundle):
                raise ValueError("Single-camera pipeline cannot consume a camera group")
            frames = {"front": frame}
        perceptions, details = {}, {}
        for name, item in frames.items():
            rectifier = self.rectifiers[name]
            result = self.model.infer(rectifier(item.rgb))
            result.drivable[~rectifier.valid_pixels] = 0.0
            perceptions[name] = result
            details[name] = {"sequence": item.sequence, "received_at": item.received_at,
                             "inference_ms": result.inference_ms,
                             "detections": len(result.detections)}
        plan_input = perceptions if self.multiview else perceptions["front"]
        plan = self.planner.plan(plan_input, speed_mps, self.cfg["route_hint"])
        return PipelineResult(plan, sum(item.inference_ms for item in perceptions.values()), details)
