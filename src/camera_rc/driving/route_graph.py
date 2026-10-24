"""Directed lane-graph route baseline; consumes declared maps, never invents lanes."""
from dataclasses import dataclass
import heapq
import math
import numpy as np


@dataclass(frozen=True)
class LaneSegment:
    lane_id: str
    centerline: tuple[tuple[float,float],...]
    successors: tuple[str,...]
    speed_limit_mps: float


class LaneGraph:
    def __init__(self, lanes, *, source, max_connection_gap_m=.1):
        if source not in {"measured","synthetic"} or not 0<max_connection_gap_m<=1:
            raise ValueError("Explicit map source and connection tolerance required")
        if not lanes or len(lanes)>1024 or not all(isinstance(x,LaneSegment) for x in lanes):
            raise ValueError("Bounded nonempty directed lane graph required")
        self.lanes={x.lane_id:x for x in lanes};self.lengths={};self.source=source
        if len(self.lanes)!=len(lanes):
            raise ValueError("Duplicate lane identity")
        for lane in lanes:
            p=np.asarray(lane.centerline,dtype=float)
            if (not isinstance(lane.lane_id,str) or not lane.lane_id or p.ndim!=2 or p.shape[1]!=2
                    or not 2<=len(p)<=4096 or not np.isfinite(p).all()
                    or type(lane.speed_limit_mps) not in (int,float)
                    or not math.isfinite(lane.speed_limit_mps) or lane.speed_limit_mps<=0
                    or len(set(lane.successors))!=len(lane.successors)):
                raise ValueError("Invalid lane geometry, speed or successors")
            lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
            if (lengths<1e-8).any():
                raise ValueError("Repeated lane point")
            self.lengths[lane.lane_id]=float(lengths.sum())
        for lane in lanes:
            for successor in lane.successors:
                if successor not in self.lanes:
                    raise ValueError("Unknown successor lane")
                gap=np.linalg.norm(np.asarray(lane.centerline[-1])-np.asarray(self.lanes[successor].centerline[0]))
                if gap>max_connection_gap_m:
                    raise ValueError("Disconnected successor geometry")

    def route(self, start, goal, *, closed_lanes=()):
        closed=set(closed_lanes)
        if start not in self.lanes or goal not in self.lanes or not closed.issubset(self.lanes):
            raise ValueError("Unknown start, goal or closure")
        if start in closed or goal in closed:
            return {"available":False,"reason":"route_blocked","lane_ids":[],"physical_output":False}
        queue=[(self.lengths[start]/self.lanes[start].speed_limit_mps,(start,))];best={start:queue[0][0]}
        while queue:
            cost,path=heapq.heappop(queue);current=path[-1]
            if cost>best[current]:
                continue
            if current==goal:
                points=[]
                for key in path:
                    points.extend(self.lanes[key].centerline if not points else self.lanes[key].centerline[1:])
                return {"available":True,"reason":"declared_lane_route","lane_ids":list(path),
                        "points":points,"speed_limit_mps":min(self.lanes[k].speed_limit_mps for k in path),
                        "source":self.source,"physical_output":False}
            for key in sorted(self.lanes[current].successors):
                if key in closed:
                    continue
                next_cost=cost+self.lengths[key]/self.lanes[key].speed_limit_mps
                if next_cost<best.get(key,math.inf):
                    best[key]=next_cost;heapq.heappush(queue,(next_cost,path+(key,)))
        return {"available":False,"reason":"no_directed_route","lane_ids":[],"physical_output":False}
