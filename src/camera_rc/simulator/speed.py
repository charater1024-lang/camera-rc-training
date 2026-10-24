"""Synthetic speed planning in SI units; these are not vehicle safety limits."""
from __future__ import annotations

import math

MAX_SPEED_KMH = 150.0
MAX_SPEED_MPS = MAX_SPEED_KMH / 3.6


def validate_speed(mode, target_kmh):
    if not isinstance(mode,str) or mode not in {"automatic", "user"}:
        raise ValueError("speed_mode must be automatic or user")
    if (type(target_kmh) not in (int, float) or not math.isfinite(target_kmh)
            or not 0 <= target_kmh <= MAX_SPEED_KMH):
        raise ValueError("target_speed_kmh must be a finite number in 0..150")
    return mode, float(target_kmh)


def stopping_distance(speed_mps):
    """Cover the simulator supervisor's error, delay and braking assumptions."""
    upper = max(0.0, speed_mps) + 0.01
    return upper * .2 + .01 + (upper + .1) ** 2 / 6.0 + .3


def speed_for_distance(distance_m):
    if stopping_distance(MAX_SPEED_MPS)<=distance_m:
        return MAX_SPEED_MPS
    low, high = 0.0, MAX_SPEED_MPS
    for _ in range(40):
        middle = (low + high) / 2
        if stopping_distance(middle) <= max(0.0, distance_m):
            low = middle
        else:
            high = middle
    return low
