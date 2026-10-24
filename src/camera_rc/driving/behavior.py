"""Conservative lane/rule veto for supplied observations, not signal recognition."""
from dataclasses import dataclass
import math
import numpy as np
from .types import Plan


@dataclass(frozen=True)
class RoadContext:
    captured_at_s: float
    clock_id: str
    lane_polygon: tuple[tuple[float, float], ...]
    direction_allowed: bool
    signal: str  # green, none, red, yellow or unknown
    right_of_way: bool
    route_available: bool


def stop_plan(reason):
    return Plan(np.empty((0, 2), dtype=float), 0., False, reason)


def validate_road_plan(plan, context, now_s, *, clock_id, max_age_s, body_radius_m):
    """Check a densely sampled centre path plus a conservative circular body.

    A known simple polygon is required. The body must have clearance from every
    boundary, including concave corners. This is a veto, not a route planner.
    """
    if not isinstance(context, RoadContext):
        return stop_plan("road_context_missing")
    if (type(context.captured_at_s) not in (int, float) or not math.isfinite(context.captured_at_s)
            or context.clock_id != clock_id or not 0 <= now_s-context.captured_at_s <= max_age_s):
        return stop_plan("road_context_stale_or_wrong_clock")
    if context.route_available is not True:
        return stop_plan("route_unavailable")
    if context.direction_allowed is not True:
        return stop_plan("wrong_way_veto")
    if context.signal not in {"green", "none"}:
        return stop_plan("traffic_signal_stop")
    if context.right_of_way is not True:
        return stop_plan("right_of_way_unknown_or_yield")
    try:
        poly = np.asarray(context.lane_polygon, dtype=float)
        points = np.asarray(plan.points, dtype=float)
        if (poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or len(poly) > 256
                or not np.isfinite(poly).all() or points.ndim != 2 or points.shape[1] != 2
                or len(points) < 2 or len(points) > 256 or not np.isfinite(points).all()
                or not math.isfinite(body_radius_m) or body_radius_m <= 0):
            return stop_plan("invalid_lane_geometry")
        # Fail closed on crossing/non-simple polygons instead of trusting ray casting.
        def cross(a,b,c):
            u,v=b-a,c-a
            return float(u[0]*v[1]-u[1]*v[0])
        edges = list(zip(poly, np.roll(poly,-1,axis=0)))
        for i,(a,b) in enumerate(edges):
            if np.linalg.norm(b-a) < 1e-9:
                return stop_plan("invalid_lane_geometry")
            for j,(c,d) in enumerate(edges):
                if j <= i or j == i+1 or (i == 0 and j == len(edges)-1):
                    continue
                if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                    return stop_plan("invalid_lane_geometry")
        dense=[]
        for a,b in zip(points[:-1],points[1:]):
            count=math.ceil(float(np.linalg.norm(b-a)) / min(.025,body_radius_m/4)) + 1
            if count + len(dense) > 4096:
                return stop_plan("lane_sampling_limit")
            dense.extend(np.linspace(a,b,max(2,count)))
        q=np.asarray(dense)
        inside=np.zeros(len(q),dtype=bool)
        distance=np.full(len(q),np.inf)
        for a,b in edges:
            delta=b-a
            t=np.clip((q-a)@delta/float(delta@delta),0,1)
            distance=np.minimum(distance,np.linalg.norm(q-(a+t[:,None]*delta),axis=1))
            if abs(delta[1]) > 1e-12:
                inside ^= ((a[1]>q[:,1]) != (b[1]>q[:,1])) & (q[:,0] < a[0]+(q[:,1]-a[1])*delta[0]/delta[1])
        if not (inside & (distance > body_radius_m + .013)).all():
            return stop_plan("lane_boundary_veto")
    except (TypeError, ValueError, OverflowError):
        return stop_plan("invalid_lane_geometry")
    return plan
