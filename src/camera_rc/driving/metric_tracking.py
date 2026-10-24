"""Diagnostic metric tracker; never estimates range from a 2D bounding box.

Inputs must already be measured world-frame positions with error bounds.
Greedy association is only a baseline; ambiguous association is flagged and
unobserved tracks persist with growing uncertainty instead of becoming free.
"""
from dataclasses import dataclass, replace
import math
from .shadow_safety import Track


@dataclass(frozen=True)
class MetricDetection:
    x_m: float
    y_m: float
    label: str
    radius_m: float
    error_m: float


@dataclass
class _State:
    label: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    error: float
    measured_at: float


class MetricTracker:
    def __init__(self, *, max_speed_mps, acceleration_bound_mps2, max_gap_s=.5):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (max_speed_mps,acceleration_bound_mps2,max_gap_s)):
            raise ValueError("Explicit finite positive tracking bounds required")
        self.max_speed,self.acceleration,self.max_gap=max_speed_mps,acceleration_bound_mps2,max_gap_s
        self.states={}; self.last_time=None; self.next_id=1

    def update(self, detections, now_s, *, ego_world_pose, detector_status):
        if type(now_s) not in (int,float) or not math.isfinite(now_s) or now_s<0 or (self.last_time is not None and now_s<=self.last_time):
            raise ValueError("Tracking requires increasing measurement timestamps")
        if len(ego_world_pose)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in ego_world_pose):
            raise ValueError("Measured ego world x,y,yaw required")
        if detector_status not in {"ok","failed","not_configured","timed_out"}:
            raise ValueError("Explicit detector health required")
        if detector_status!="ok" and detections:
            raise ValueError("Unhealthy detector cannot publish detections")
        if len(detections)>256:
            raise ValueError("Too many metric detections")
        for d in detections:
            if not isinstance(d,MetricDetection) or not isinstance(d.label,str) or not d.label or not all(type(v) in (int,float) and math.isfinite(v) for v in (d.x_m,d.y_m,d.radius_m,d.error_m)) or d.radius_m<=0 or d.error_m<0:
                raise ValueError("Finite measured metric detections required")
        ambiguous=False; used=set(); updated={}
        for d in detections:
            candidates=[]
            for key,old in self.states.items():
                dt=now_s-old.measured_at
                if key in used or old.label!=d.label or dt>self.max_gap:
                    continue
                distance=math.hypot(d.x_m-old.x,d.y_m-old.y)
                if distance<=self.max_speed*dt+old.error+d.error_m:
                    candidates.append((distance,key))
            if len(candidates)>1:
                ambiguous=True
            if candidates:
                _,key=min(candidates); old=self.states[key]; dt=now_s-old.measured_at
                vx,vy=(d.x_m-old.x)/dt,(d.y_m-old.y)/dt
                magnitude=math.hypot(vx,vy)
                if magnitude>self.max_speed:
                    # Measurement noise cannot silently enlarge the declared bound.
                    ambiguous=True; vx=vy=0.
            else:
                key=f"metric-{self.next_id}";self.next_id+=1;vx=vy=0.
            used.add(key)
            updated[key]=_State(d.label,d.x_m,d.y_m,vx,vy,d.radius_m,d.error_m,now_s)
        expired=False
        for key,old in self.states.items():
            if key not in used:
                if now_s-old.measured_at<=self.max_gap:
                    updated[key]=old
                else:
                    expired=True
        self.states=updated; self.last_time=now_s
        x,y,yaw=ego_world_pose; c,s=math.cos(yaw),math.sin(yaw)
        tracks=[]
        for key,old in sorted(updated.items()):
            dt=now_s-old.measured_at
            dx,dy=old.x+old.vx*dt-x,old.y+old.vy*dt-y
            tracks.append(Track(key,now_s,c*dx+s*dy,-s*dx+c*dy,
                c*old.vx+s*old.vy,-s*old.vx+c*old.vy,old.radius,
                old.error+self.max_speed*dt+.5*self.acceleration*dt*dt,
                self.max_speed,self.acceleration))
        return {"tracks":tuple(tracks),"status":"ok" if detector_status=="ok" and not ambiguous and not expired else "degraded",
                "expired_tracks":expired,"association_ambiguous":ambiguous,
                "free_space_verified":False,"physical_output":False}
