"""Runtime units: metres, seconds, radians; vehicle x forward and y left."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class Perception:
    drivable: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    detector_status: str = "unknown"


@dataclass
class Plan:
    points: np.ndarray
    target_speed_mps: float
    valid: bool
    reason: str
    curvature: float = 0.0


@dataclass
class VehicleState:
    timestamp: float
    speed_mps: float
    yaw_rate_rps: float = 0.0
    armed: bool = False
    estop: bool = True
    ack_seq: int = -1


@dataclass
class Command:
    steering_rad: float = 0.0
    throttle: float = 0.0
    brake: float = 1.0
    target_speed_mps: float = 0.0
    enabled: bool = False
    reason: str = "stop"
