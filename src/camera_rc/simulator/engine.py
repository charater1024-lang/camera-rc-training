"""Synthetic world, adjustable speed and independent outcome checks.

The default sensor is an explicitly privileged oracle. A learned callback can
replace image semantics, while metric occupancy/tracks remain synthetic. This
is a closed-loop software demonstration, not a real-vehicle safety assessment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..driving.control import Controller
from ..driving.shadow_safety import (
    Grid, ModuleEvidence, REQUIRED_MODULES, SceneEvidence, ShadowPolicy,
    ShadowSafetySupervisor, Track,
)
from ..driving.types import Command, Perception, Plan, VehicleState
from . import rendering, objects as object_display
from .lane import LaneCorridor, road_lane_layout
from .traffic import TrafficScene, vehicle_clearance
from .performance import PerformanceTracker
from .speed import MAX_SPEED_KMH, MAX_SPEED_MPS, speed_for_distance, stopping_distance, validate_speed
from .trajectory import pure_pursuit_rollout


SCOPE = "synthetic_closed_loop_demo_not_vehicle_validation"
SCENARIOS = [
    {"id":"traffic_interaction","title":"교통 상호작용 · 왕복 2차로","description":"차로를 따라 이동·정지하는 차량과 신호 횡단 보행자의 합성 규칙 시나리오. 실제 교통 검증 아님."},
    {"id": "clear", "title": "Clear closed course", "description": "Follow a clear wide road and stop at the goal."},
    {"id": "curved", "title": "Curved road", "description": "Steering changes pose and the following camera observation."},
    {"id": "static_obstacle", "title": "Stopped obstacle", "description": "An obstacle blocks the route; autonomous mode must stop."},
    {"id": "crossing_pedestrian", "title": "Crossing pedestrian", "description": "A moving pedestrian enters the route; synthetic tracks provide a veto."},
    {"id": "night", "title": "Night obstacle", "description": "Dark synthetic camera imagery; oracle geometry is still privileged."},
    {"id": "fog", "title": "Limited visibility", "description": "A short observed range cannot support the full stopping corridor."},
    {"id": "sensor_freeze", "title": "Frozen camera", "description": "Camera exposure and frame identity freeze after three simulation seconds."},
    {"id": "sensor_dropout", "title": "Camera dropout", "description": "Camera evidence disappears after three simulation seconds."},
    {"id": "braking_degraded", "title": "Degraded braking", "description": "Declared braking capability fails and actual deceleration is reduced."},
    {"id": "seoul_day", "title": "서울 성수동 · 낮", "description": "OSM road/building geometry, 80m reference route and seeded synthetic actors; not a digital twin."},
    {"id": "seoul_night", "title": "서울 성수동 · 밤", "description": "The same mapped course with dark synthetic images and a shorter idealized visible range."},
    {"id": "seoul_hazard", "title": "서울 성수동 · 경로 장애물", "description": "A synthetic blocking vehicle on the mapped route; independent world contacts remain active."},
    {"id": "busan_day", "title": "부산 · 해안과 구릉", "description": "지역 특성을 표현한 합성 해안·구릉 코스. 실제 도로·고도 지도가 아닙니다."},
    {"id": "gwangju_day", "title": "광주 · 도심 평지", "description": "긴 직선과 도심 배경의 합성 코스. 실제 도로·고도 지도가 아닙니다."},
    {"id": "suncheon_day", "title": "전남 순천 · 평야와 완만한 구릉", "description": "평야·수변과 완만한 구릉의 합성 코스. 실제 도로·고도 지도가 아닙니다."},
    {"id": "gangwon_day", "title": "강원 산지 · 산악 곡선", "description": "고도 변화와 굽은 길의 합성 코스. 실제 도로·고도 지도가 아닙니다."},
]
_SCENARIOS = {s["id"]: s for s in SCENARIOS}


@dataclass
class _Vehicle:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    speed_mps: float = 0.0
    steering_rad: float = 0.0


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _distance_to_road(x, y, points):
    """Euclidean distance to the world centerline, independent of image masks."""
    x, y = np.asarray(x), np.asarray(y)
    distance = np.full(np.broadcast_shapes(x.shape, y.shape), np.inf)
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b-a
        if dx*dx+dy*dy<=1e-14:
            distance=np.minimum(distance,np.hypot(x-a[0],y-a[1]))
            continue
        fraction = np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy), 0, 1)
        distance = np.minimum(distance, np.hypot(x-a[0]-fraction*dx, y-a[1]-fraction*dy))
    return distance


def _polyline_position(points, distance):
    """Arc-length interpolation with tangent continuation beyond the last point."""
    vectors=np.diff(points,axis=0)
    lengths=np.linalg.norm(vectors,axis=1)
    cumulative=np.concatenate(([0.],np.cumsum(lengths)))
    index=min(len(lengths)-1,max(0,int(np.searchsorted(cumulative,distance,side='right')-1)))
    direction=vectors[index]/lengths[index]
    return points[index]+direction*(distance-cumulative[index]),direction


def _polyline_projection(point, points):
    vectors=np.diff(points,axis=0)
    squares=np.sum(vectors*vectors,axis=1)
    fractions=np.clip(np.sum((point-points[:-1])*vectors,axis=1)/squares,0,1)
    projections=points[:-1]+vectors*fractions[:,None]
    index=int(np.argmin(np.linalg.norm(projections-point,axis=1)))
    lengths=np.sqrt(squares)
    return float(np.sum(lengths[:index])+fractions[index]*lengths[index]),projections[index]


def _arc_points(curvature, distance):
    if abs(curvature)<1e-7:
        return np.column_stack((distance,np.zeros_like(distance)))
    return np.column_stack((np.sin(curvature*distance)/curvature,
                            (1-np.cos(curvature*distance))/curvature))


def _inside_polygon(x,y,polygon):
    x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y))
    inside=np.zeros(x.shape,dtype=bool)
    for a,b in zip(polygon[:-1],polygon[1:]):
        if a[1] != b[1]:
            inside ^= ((a[1]>y)!=(b[1]>y)) & (x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside


def _polygon_clearance(x,y,polygon):
    distance=float(_distance_to_road(x,y,polygon))
    return -distance if bool(_inside_polygon(x,y,polygon)) else distance


def _blocked_rays(origin, targets, polygons):
    """Conservative opaque 2D footprint rays; targets inside walls are unknown."""
    shape=targets.shape[:-1]
    rays=targets.reshape(-1,2)-np.asarray(origin)
    blocked=np.zeros(len(rays),dtype=bool)
    for polygon in polygons:
        if bool(_inside_polygon(origin[0],origin[1],polygon)):
            return np.ones(shape,dtype=bool)
        for a,b in zip(polygon[:-1],polygon[1:]):
            segment=b-a
            offset=a-origin
            denominator=rays[:,0]*segment[1]-rays[:,1]*segment[0]
            usable=np.abs(denominator)>1e-10
            along=np.full(len(rays),np.inf)
            across=np.full(len(rays),np.inf)
            np.divide(offset[0]*segment[1]-offset[1]*segment[0],denominator,out=along,where=usable)
            np.divide(offset[0]*rays[:,1]-offset[1]*rays[:,0],denominator,out=across,where=usable)
            blocked |= usable & (along>1e-7) & (along<1-1e-7) & (across>=0) & (across<=1)
    return blocked.reshape(shape)


class Engine:
    DT_S = 0.1
    WHEELBASE_M = 2.6
    BODY_RADIUS_M = 2.7
    ROAD_HALF_WIDTH_M = 5.5
    MAX_STEERING_RAD = 0.45
    MAX_SPEED_MPS = MAX_SPEED_MPS
    MAX_STEPS = 900

    def __init__(self, scenario="clear", seed=42, mode="autonomous", perception=None, output_dir=None,
                 speed_mode="automatic", target_speed_kmh=None):
        validate_speed(speed_mode, 150.0 if target_speed_kmh is None else target_speed_kmh)
        self.speed_mode = speed_mode
        self._speed_explicit = target_speed_kmh is not None or speed_mode != "automatic"
        self.target_speed_kmh = target_speed_kmh
        if perception is not None and not callable(perception):
            raise ValueError("perception must be a callable accepting RGB uint8")
        self.perception_callback = perception
        identity = getattr(perception, "simulator_identity", None)
        if identity is not None and not isinstance(identity, dict):
            raise ValueError("simulator_identity must be a JSON object")
        self._model_identity = json.loads(json.dumps(identity, allow_nan=False))
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            if self.output_dir.exists():
                if not self.output_dir.is_dir() or any(self.output_dir.iterdir()):
                    raise ValueError("output_dir must be new or an empty directory")
            else:
                self.output_dir.mkdir(parents=True, exist_ok=False)
        self._run_number = -1
        self._session_dir = None
        self._closed = False
        self.reset(scenario=scenario, seed=seed, mode=mode)

    @staticmethod
    def _validate_seed(seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def reset(self, scenario=None, seed=None, mode=None):
        if self._closed:
            raise RuntimeError("simulator is closed")
        scenario = scenario if scenario is not None else getattr(self, "scenario", "clear")
        seed = seed if seed is not None else getattr(self, "seed", 42)
        mode = mode if mode is not None else getattr(self, "mode", "autonomous")
        if scenario not in _SCENARIOS:
            raise ValueError("unknown scenario")
        self._validate_seed(seed)
        if mode not in {"autonomous", "manual"}:
            raise ValueError("mode must be autonomous or manual")
        if self._session_dir is not None:
            self._write_summary("reset")
        self.scenario, self.seed, self.mode = scenario, seed, mode
        self._car = _Vehicle()
        self._step, self._time = 0, 0.0
        self._finished, self._goal_reached = False, False
        self._distance, self._collisions, self._offroad_events = 0.0, 0, 0
        self._interventions, self._min_clearance = 0, None
        self._contacts, self._offroad = set(), False
        self._events = deque(maxlen=32)
        self._previous_stop = False
        self._camera_frozen = None
        self._frozen_capture, self._frozen_sequence = None, None
        self._model_error = None
        self._model_calls, self._model_successes = 0, 0
        self._capture_time = 0.0
        self._region = None
        self._region_actors = []
        self._regional_visibility = {}
        self._road_near_origin = None
        self._traffic=TrafficScene(seed) if self.scenario=="traffic_interaction" else None
        self.goal_x_m = 110.0 if self._traffic else 36.0
        xs = np.arange(-20.0, 102.0, 2.0) if self.scenario == "curved" else np.array([-20.0,100.0])
        if self._traffic is not None:
            xs=np.array([-40.,180.])
        self._road = np.column_stack((xs, self._road_y(xs)))
        self._course_id = self.scenario.removesuffix('_day')
        self._extended_course = self._course_id in {'busan', 'gwangju', 'suncheon', 'gangwon'}
        self._step_budget=6000 if self._extended_course else self.MAX_STEPS
        if self.scenario.startswith('seoul_') or self._extended_course:
            self._prepare_region()
        if self._region is not None:
            reference=self._road.copy()
        else:
            # The outcome route starts at the initial pose and ends at the
            # goal; the render road also contains extra approach/runoff area.
            inner=self._road[(self._road[:,0]>0)&(self._road[:,0]<self.goal_x_m)]
            reference=np.vstack(([0.,float(self._road_y(0.))],inner,
                                 [self.goal_x_m,float(self._road_y(self.goal_x_m))]))
        self._performance=PerformanceTracker(reference)
        self._lane=LaneCorridor(reference)
        self._road_lanes=road_lane_layout(self._lane,3.0 if self._region and not self._extended_course else 3.6)
        self._lane_direction_allowed=True
        self._lane_outside=False
        self._lane_samples=0
        self._lane_duration=0.
        self._lane_outside_time=0.
        self._lane_departures=0
        self._lane_guard_allowed=True
        self._lane_guard_checked=False
        if not self._speed_explicit or self.target_speed_kmh is None:
            self.target_speed_kmh = 150.0 if self._extended_course else 28.8
        self._effective_target_mps = 2.0
        self._speed_reasons = []
        self._path_length_m = 8.0
        self._render_cache_key, self._render_cache = None, None
        rng = np.random.default_rng(seed)
        self._obstacle_x = 24.0 + float(rng.uniform(-0.4, 0.4))
        self._command = Command(reason="reset")
        self._plan_world = []
        self._safety = {"mode": "WARMUP_STOP", "reasons": ["initial_observation_required"],
                        "proposal_eligible": False, "actuation_enabled": False}
        self._controller = Controller({"wheelbase_m": self.WHEELBASE_M, "max_steering_rad": self.MAX_STEERING_RAD,
            "steering_rate_rad_s": 0.8, "lookahead_m": 4.0, "lookahead_speed_gain_s": 0.2,
            "max_speed_mps": MAX_SPEED_MPS, "max_accel_mps2": 1.0, "max_decel_mps2": 3.0,
            "max_throttle": 1.0, "nominal_dt_s": self.DT_S, "stop_to_align": False})
        self._policy = ShadowPolicy(clock_id="sim_fixed_dt", calibration_sha256=_digest("synthetic_chase_camera_v1"),
            artifacts={n:_digest("synthetic_v1_"+n) for n in REQUIRED_MODULES},
            allowed_conditions=("synthetic_day", "synthetic_night", "synthetic_fog"),
            max_speed_mps=MAX_SPEED_MPS+.12, min_braking_deceleration_mps2=3.0, acceleration_during_delay_mps2=0.5,
            reaction_bound_s=0.2, stopping_margin_m=0.3, body_radius_m=self.BODY_RADIUS_M,
            lateral_margin_m=0.1, max_localization_error_m=0.1, max_speed_error_mps=0.1,
            max_motion_compensation_error_m=0.1, max_observation_age_s=0.15, max_state_age_s=0.15,
            max_plan_age_s=0.15, max_skew_s=0.05, max_curvature_per_m=0.3,
            max_lateral_acceleration_mps2=2.0, max_path_sampling_m=0.12,
            healthy_cycles_to_reset=2, max_path_points=8192, max_dense_points=8192)
        self._supervisor = ShadowSafetySupervisor(self._policy)
        if self._extended_course or self._speed_explicit:
            self._plan()
        self._rgb, self._probability = self._render()
        self._object_view=self._capture_objects()
        self._object_frame_frozen=False
        if self.perception_callback is not None:
            self._probability = np.zeros_like(self._probability)
            self._model_error = "inference_pending"
        self._run_number += 1
        if self.output_dir is not None:
            self._session_dir = self.output_dir/f"session_{self._run_number:04d}"
            self._session_dir.mkdir(exist_ok=False)
            self._write_json(self._session_dir/"configuration.json", self._configuration())
        self._event("reset", "Fresh synthetic session")
        return self.snapshot()

    def _road_y(self, x):
        values = np.asarray(x)
        return 3.5*(1-np.cos(np.maximum(values, 0)/14.0)) if self.scenario == "curved" else np.zeros_like(values)

    def _prepare_region(self):
        from .region import load_region
        if self._extended_course:
            from .terrain import load_course
            self._region=load_course(self._course_id)
        else:
            self._region=load_region()
        self._road=np.asarray(self._region['route']['points'],dtype=float)
        self._route_length=float(np.linalg.norm(np.diff(self._road,axis=0),axis=1).sum())
        self._car.x_m,self._car.y_m=map(float,self._road[0])
        self._car.yaw_rad=float(self._region['route']['start_heading_rad'])
        self.goal_x_m=float(self._road[-1,0])  # compatibility; goal uses arc length and XY region.
        starts,ends,widths=[],[],[]
        for road in self._region['roads']:
            if not road.get('drivable',False) or not road.get('ground_geometry_supported',True):
                continue
            points=np.asarray(road['points'],dtype=float)
            for a,b in zip(points[:-1],points[1:]):
                if np.linalg.norm(b-a)>1e-6:
                    starts.append(a);ends.append(b);widths.append(road['half_width_m'])
        self._segments_a=np.asarray(starts)
        self._segments_b=np.asarray(ends)
        self._segment_widths=np.asarray(widths)
        self._segment_low=np.minimum(self._segments_a,self._segments_b)-self._segment_widths[:,None]
        self._segment_high=np.maximum(self._segments_a,self._segments_b)+self._segment_widths[:,None]
        self._buildings=[]
        for building in self._region['buildings']:
            polygon=np.asarray(building['polygon'],dtype=float)
            self._buildings.append((building['id'],polygon,polygon.min(axis=0),polygon.max(axis=0)))
        self._building_low=np.array([entry[2] for entry in self._buildings])
        self._building_high=np.array([entry[3] for entry in self._buildings])
        # Cover the entire body disk with squares. A cell is certified within
        # the union only if one road capsule contains its circumscribed disk.
        # This is conservative at boundaries/intersections, never a narrowing
        # of the vehicle to force progress through the mapped street.
        spacing=.4
        axis=np.arange(-self.BODY_RADIUS_M,self.BODY_RADIUS_M+spacing,spacing)
        bx,by=np.meshgrid(axis,axis)
        offsets=np.column_stack((bx.ravel(),by.ravel()))
        self._body_cover_radius=spacing/math.sqrt(2)
        self._body_offsets=offsets[np.linalg.norm(offsets,axis=1)<=self.BODY_RADIUS_M+self._body_cover_radius]
        self._region_actors=self._make_region_actors()

    def _near_buildings(self, origin, distance):
        if not self._buildings:
            return []
        delta=np.maximum(np.maximum(self._building_low-origin,origin-self._building_high),0)
        indices=np.flatnonzero(np.sum(delta*delta,axis=1)<=distance*distance)
        return [self._buildings[int(index)] for index in indices]

    def _road_union_margin(self,x,y,search_radius=35.):
        x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y))
        origin=np.array([self._car.x_m,self._car.y_m])
        if (self._road_near_origin is None or np.linalg.norm(origin-self._road_near_origin)>2
                or getattr(self,'_road_search_radius',None)!=search_radius):
            delta=np.maximum(np.maximum(self._segment_low-origin,origin-self._segment_high),0)
            self._near_segment_indices=np.flatnonzero(np.sum(delta*delta,axis=1)<=search_radius**2)
            self._road_near_origin=origin
            self._road_search_radius=search_radius
        margin=np.full(x.shape,-np.inf)
        for index in self._near_segment_indices:
            a,b=self._segments_a[index],self._segments_b[index]
            vector=b-a
            candidate=((x>=self._segment_low[index,0])&(x<=self._segment_high[index,0])
                       &(y>=self._segment_low[index,1])&(y<=self._segment_high[index,1]))
            if not np.any(candidate):continue
            px,py=x[candidate],y[candidate]
            fraction=np.clip(((px-a[0])*vector[0]+(py-a[1])*vector[1])/float(vector@vector),0,1)
            margin[candidate]=np.maximum(margin[candidate],self._segment_widths[index]-np.hypot(px-a[0]-fraction*vector[0],py-a[1]-fraction*vector[1]))
        return margin

    def _body_inside_road_union(self):
        origin=np.array([self._car.x_m,self._car.y_m])
        # A disk contained in one road capsule is already a complete geometric
        # certificate. Only intersections need the more costly union cover.
        delta=np.maximum(np.maximum(self._segment_low-origin,origin-self._segment_high),0)
        indices=np.flatnonzero(np.sum(delta*delta,axis=1)<=self.BODY_RADIUS_M**2)
        if len(indices):
            starts=self._segments_a[indices]
            vectors=self._segments_b[indices]-starts
            fractions=np.clip(np.sum((origin-starts)*vectors,axis=1)/np.sum(vectors*vectors,axis=1),0,1)
            distances=np.linalg.norm(origin-starts-vectors*fractions[:,None],axis=1)
            if np.any(distances+self.BODY_RADIUS_M<=self._segment_widths[indices]):
                return True
        points=self._body_offsets+np.array([self._car.x_m,self._car.y_m])
        return bool(np.all(self._road_union_margin(points[:,0],points[:,1])>=self._body_cover_radius))

    def _make_region_actors(self):
        rng=np.random.default_rng(self.seed)
        actors=[]
        def unobstructed(point,radius):
            return all(_polygon_clearance(*point,b[1])>radius+.25 for b in self._near_buildings(point,radius+.3))
        # Kerbside actors along the reference route are deliberately close
        # enough to appear in the camera. Their placement is synthetic and
        # checked against building footprints, not asserted to be legal parking.
        for index,distance in enumerate(np.linspace(10,self._route_length-8,12)):
            point,direction=_polyline_position(self._road,float(distance))
            normal=np.array([-direction[1],direction[0]])
            side=1 if index%2 else -1
            kind=('vehicle','pedestrian','cone')[index%3]
            radius={'vehicle':2.2,'pedestrian':.3,'cone':.22}[kind]
            point=point+side*normal*(7.0 if kind=='vehicle' else 4.45)
            if unobstructed(point,radius):
                actors.append({'id':f'kerb_{index}','point':point,'direction':direction,'radius':radius,
                               'kind':kind,'amplitude':0.,'speed':0.,'phase':0.})
        roads=[r for r in self._region['roads'] if r.get('drivable',False) and r.get('ground_geometry_supported',True)]
        # Background traffic moves only outside the protected reference-route
        # corridor; the hazard scenario adds an explicit blocking actor below.
        for attempt in range(2400):
            if len(actors)>=44:
                break
            road=roads[int(rng.integers(len(roads)))]
            points=np.asarray(road['points'],dtype=float)
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            if length<8:continue
            path_distance=float(rng.uniform(3,length-3))
            point,direction=_polyline_position(points,path_distance)
            kind=('vehicle','pedestrian','cone')[attempt%3]
            radius={'vehicle':2.2,'pedestrian':.3,'cone':.22}[kind]
            moving=kind!='cone' and attempt%2==0
            amplitude=2.5 if moving else 0.
            if float(_distance_to_road(*point,self._road))<12+amplitude:continue
            if any(np.linalg.norm(point-a['point'])<radius+a['radius']+2*amplitude+1 for a in actors):continue
            if not all(unobstructed(_polyline_position(points,path_distance+d)[0],radius)
                       for d in np.linspace(-amplitude,amplitude,21)):continue
            actors.append({'id':f'area_{attempt}','point':point,'direction':direction,'radius':radius,
                           'kind':kind,'amplitude':amplitude,'speed':.6 if kind=='pedestrian' else 1.2,
                           'phase':float(rng.uniform(0,2*math.pi)),'path':points,'path_distance':path_distance})
        if self.scenario=='seoul_hazard':
            point,direction=_polyline_position(self._road,26.)
            actors.append({'id':'route_blocking_vehicle','point':point,'direction':direction,'radius':2.2,
                           'kind':'vehicle','amplitude':0.,'speed':0.,'phase':0.})
        if len(actors)<30:
            raise ValueError('Mapped region could not support the specified actor count')
        return actors

    def _obstacles(self, at_time=None):
        t = self._time if at_time is None else at_time
        if self._traffic is not None:
            return self._traffic.obstacles()
        if self._region is not None:
            result=[]
            for actor in self._region_actors:
                amplitude=actor['amplitude']
                phase=actor['phase']+t*actor['speed']/amplitude if amplitude else 0.
                if 'path' in actor:
                    point,direction=_polyline_position(actor['path'],actor['path_distance']+amplitude*math.sin(phase))
                else:
                    point,direction=actor['point'],actor['direction']
                velocity=direction*actor['speed']*math.cos(phase) if amplitude else np.zeros(2)
                result.append({'id':actor['id'],'x_m':float(point[0]),'y_m':float(point[1]),
                               'radius_m':actor['radius'],'kind':actor['kind'],
                               'vx_mps':float(velocity[0]),'vy_mps':float(velocity[1])})
            return result
        if self.scenario in {"static_obstacle", "night", "braking_degraded"}:
            return [{"id":"stopped_obstacle", "x_m":self._obstacle_x, "y_m":0.0,
                     "radius_m":0.8, "kind":"obstacle", "vx_mps":0.0, "vy_mps":0.0}]
        if self.scenario == "crossing_pedestrian":
            moving = 3.0 <= t < 15.0
            return [{"id":"pedestrian", "x_m":18.0, "y_m":6.0-min(12.0,max(0.0,t-3.0)),
                     "radius_m":0.4, "kind":"pedestrian", "vx_mps":0.0, "vy_mps":-1.0 if moving else 0.0}]
        return []

    def _condition(self):
        if self.scenario=='seoul_night':return 'night'
        return self.scenario if self.scenario in {"night", "fog"} else "day"

    def _render(self):
        obstacles = self._obstacles()
        if self._region is not None:
            obstacles=[a for a in obstacles if math.hypot(a['x_m']-self._car.x_m,a['y_m']-self._car.y_m)<125]
        key = (self._car.x_m,self._car.y_m,self._car.yaw_rad,self._road.tobytes(),
               self.ROAD_HALF_WIDTH_M,self._condition(),self.seed,
               tuple(tuple(sorted(obstacle.items())) for obstacle in obstacles))
        if key != self._render_cache_key:
            self._render_cache = rendering.render_observation(asdict(self._car), self._road,
                          self.ROAD_HALF_WIDTH_M,obstacles,self._condition(),self.seed,
                          lane_layout=self._road_lanes,traffic=self._traffic.snapshot() if self._traffic else None,
                          **({'roads':self._region['roads'],'buildings':self._region['buildings'],
                              'terrain':self._region.get('terrain')} if self._region else {}))
            self._render_cache_key = key
        # Cache only deterministic image construction. Learned callbacks still
        # run on each fresh observation, with fresh exposure/sequence evidence.
        return tuple(array.copy() for array in self._render_cache)

    def _event(self, kind, message):
        self._events.append({"step":self._step, "sim_time_s":self._time, "kind":kind, "message":message})

    def set_mode(self, mode):
        if self._closed:
            raise RuntimeError("simulator is closed")
        if mode not in {"autonomous", "manual"}:
            raise ValueError("mode must be autonomous or manual")
        if mode != self.mode:
            self.mode = mode
            self._controller.reset()
            self._command = Command(reason="mode_change_brake")
            self._event("mode", mode)

    def set_speed(self, speed_mode, target_speed_kmh):
        if self._closed:
            raise RuntimeError("simulator is closed")
        mode,target=validate_speed(speed_mode,target_speed_kmh)
        self.speed_mode,self.target_speed_kmh=mode,target
        self._speed_explicit=True
        self._controller.reset()
        self._controller._target_speed=self._car.speed_mps
        self._plan()
        self._event('speed_setting',f'{mode}: {target:g} km/h requested')
        return self.snapshot()

    def _sensor_range_m(self):
        if self._traffic is not None and self.perception_callback is None:
            return 80.
        if self._extended_course and self.perception_callback is None:
            return 400.0
        if self._region is not None:
            return 14.0 if self._condition()=='night' else 25.0
        return 7.5 if self.scenario=='fog' else 16.0

    def _terrain_state(self):
        if not self._extended_course:
            return 0.0,0.0
        from .terrain import elevation_at,grade_at
        progress,_=_polyline_projection(np.array([self._car.x_m,self._car.y_m]),self._road)
        return elevation_at(self._region,progress),grade_at(self._region,progress)

    def _capture_objects(self):
        actors=self._obstacles()
        visible=None
        if self._region is not None:
            visible={actor['id'] for actor in actors if bool(self._visible_region_points(
                np.array([actor['x_m'],actor['y_m']]),actors,actor['id']))}
        return object_display.observation(actors,(self._car.x_m,self._car.y_m,self._car.yaw_rad),
            self._sensor_range_m(),visible,self._time,self._road)

    def _observe(self):
        self._object_frame_frozen=False
        self._capture_time = self._time
        sequence = self._step
        frozen = self.scenario == "sensor_freeze" and self._time >= 3.0
        dropout = self.scenario == "sensor_dropout" and self._time >= 3.0
        if frozen and self._camera_frozen is not None:
            self._object_frame_frozen=True
            self._rgb, self._probability = (a.copy() for a in self._camera_frozen)
            self._capture_time = self._frozen_capture
            return self._frozen_capture, self._frozen_sequence, "ok"
        self._rgb, oracle_mask = self._render()
        self._object_view=self._capture_objects()
        self._model_error = None
        if dropout:
            self._rgb = np.zeros_like(self._rgb)
            self._probability = np.zeros(oracle_mask.shape, dtype=np.float32)
            self._model_error = "synthetic_camera_dropout"
            self._object_view.update(status='camera_unavailable',objects=[],camera_box_count=0)
            return self._time, sequence, "failed"
        if self.perception_callback is None:
            self._probability = oracle_mask
        else:
            self._model_calls += 1
            try:
                result = self.perception_callback(self._rgb.copy())
                if not isinstance(result, Perception):
                    raise ValueError("callback must return Perception")
                mask = np.asarray(result.drivable)
                if (mask.shape != oracle_mask.shape or not np.issubdtype(mask.dtype, np.number)
                        or not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1)):
                    raise ValueError("invalid learned probability mask")
                self._probability = mask.astype(np.float32, copy=True)
                self._model_successes += 1
            except Exception as exc:
                self._probability = np.zeros(oracle_mask.shape, dtype=np.float32)
                self._model_error = type(exc).__name__ + ": " + str(exc)[:160]
                return self._time, sequence, "failed"
        if self.scenario == "sensor_freeze":
            self._camera_frozen = (self._rgb.copy(), self._probability.copy())
            self._frozen_capture, self._frozen_sequence = self._time, sequence
        return self._time, sequence, "ok"

    def _to_local(self, x, y):
        dx, dy = np.asarray(x)-self._car.x_m, np.asarray(y)-self._car.y_m
        c, s = math.cos(self._car.yaw_rad), math.sin(self._car.yaw_rad)
        return c*dx+s*dy, -s*dx+c*dy

    def _to_world(self, x, y):
        c, s = math.cos(self._car.yaw_rad), math.sin(self._car.yaw_rad)
        return self._car.x_m+c*x-s*y, self._car.y_m+s*x+c*y

    def _plan(self):
        adaptive=self._extended_course or self._speed_explicit
        target_speed=2.0
        reasons=[]
        lookahead=7.0 if not adaptive else max(7.,self._car.speed_mps*1.2)
        if self._region is not None:
            progress,_=_polyline_projection(np.array([self._car.x_m,self._car.y_m]),self._road)
            remaining=self._route_length-progress
            target,_=_polyline_position(self._road,progress+lookahead)
            x,y=self._to_local(*target)
        else:
            remaining=self.goal_x_m-self._car.x_m
            target_x = self._car.x_m+lookahead
            x,y = self._to_local(target_x,float(self._road_y(target_x)))
        curvature = float(np.clip(2*y/max(1.0,x*x+y*y), -.12, .12))
        if adaptive:
            target_speed=self.target_speed_kmh/3.6
            limits=[('visibility',speed_for_distance(self._sensor_range_m()-8.0)),
                    ('goal_distance',speed_for_distance(remaining-.65))]
            if abs(curvature)>1e-8:
                limits.append(('curvature',max(0.,math.sqrt(1.4/abs(curvature))-.12)))
            if self._region is not None:
                vectors=np.diff(self._road,axis=0)
                lengths=np.linalg.norm(vectors,axis=1)
                headings=np.unwrap(np.arctan2(vectors[:,1],vectors[:,0]))
                turning=np.abs(np.diff(headings))/((lengths[:-1]+lengths[1:])/2)
                ahead=np.cumsum(lengths)[:-1]-progress
                window=max(80.,stopping_distance(max(self._car.speed_mps,target_speed))+40.)
                indices=np.flatnonzero((ahead>=-5.)&(ahead<=window)&(turning>1e-7))
                for index in indices:
                    corner=max(0.,math.sqrt(1.3/turning[index])-.12)
                    cap=math.sqrt(corner*corner+4.*max(0.,ahead[index]-20.-self._car.speed_mps*.2))
                    limits.append(('upcoming_curve',cap))
            if self.speed_mode=='automatic':
                advisory={'busan':90.,'gwangju':150.,'suncheon':100.,'gangwon':70.}.get(self._course_id,7.2)
                limits.append(('regional_advisory',advisory/3.6))
            for reason,limit in limits:
                if limit<target_speed:
                    target_speed=limit
                    reasons.append(reason)
            if self._extended_course:
                # Preview whether the current steering arc can remain on the
                # route for a complete stop. This slows before a distant bend,
                # rather than waiting for the short lookahead to reach it.
                horizon=max(8.,stopping_distance(max(self._car.speed_mps,target_speed))+2.)
                samples=np.linspace(0.,horizon,max(81,math.ceil(horizon/.4)+1))
                preview=_arc_points(curvature,samples)
                px,py=self._to_world(preview[:,0],preview[:,1])
                corridor=self._road_union_margin(px,py,horizon+20.)
                bad=np.flatnonzero(corridor<self.BODY_RADIUS_M+.65)
                if len(bad):
                    route_cap=speed_for_distance(max(0.,samples[bad[0]]-5.))
                    if route_cap<target_speed:
                        target_speed=route_cap
                        reasons.append('upcoming_curve')
        if self._traffic is not None and target_speed>6.:
            target_speed=6.
            reasons.append('traffic_speed_limit')
        self._effective_target_mps=max(0.,target_speed)
        self._speed_reasons=list(dict.fromkeys(reasons))
        self._path_length_m=(max(8.,stopping_distance(max(self._car.speed_mps,target_speed))+2.) if adaptive else 8.)
        distance=np.linspace(0.,self._path_length_m,max(81,math.ceil(self._path_length_m/.1)+1))
        points=_arc_points(curvature,distance)
        if self._extended_course:
            # Re-evaluate steering along the future route, starting at the
            # actual pose and wheel angle. A single fixed-curvature arc over
            # the entire stop distance can leave a lane at changing bends.
            world_points,headings=pure_pursuit_rollout(self._road,
                self._car.x_m,self._car.y_m,self._car.yaw_rad,self._car.steering_rad,
                distance,max(self._car.speed_mps,target_speed))
            lx,ly=self._to_local(world_points[:,0],world_points[:,1])
            points=np.column_stack((lx,ly))
        else:
            wx,wy = self._to_world(points[:,0],points[:,1])
            world_points=np.column_stack((wx,wy))
            headings=self._car.yaw_rad+curvature*distance
        self._plan_y_bounds=(float(points[:,1].min()),float(points[:,1].max()))
        _,nearest=self._lane._nearest(np.array([self._car.x_m,self._car.y_m]))
        tangent=self._lane._vectors[int(nearest)]
        self._lane_direction_allowed=float(tangent@np.array([math.cos(self._car.yaw_rad),math.sin(self._car.yaw_rad)]))>0.
        self._lane_guard_allowed=self._lane.allows_path(world_points,headings) and self._lane_direction_allowed
        self._lane_guard_checked=True
        stride=max(1,math.ceil(len(world_points)/256))
        self._plan_world=world_points[::stride].tolist()
        if self._plan_world[-1]!=world_points[-1].tolist():
            self._plan_world.append(world_points[-1].tolist())
        return Plan(points,max(.1,target_speed),True,"synthetic_route_proposal",curvature)

    def _visible_region_points(self,points,obstacles,exclude_id=None):
        """Idealized 360-degree metric rays, bounded by range and opaque objects.

        No learned depth/range is implied. All geometry still comes from the
        simulator, but hidden cells cannot become evidence of free space.
        """
        origin=np.array([self._car.x_m,self._car.y_m])
        limit=self._sensor_range_m()
        vectors=points-origin
        distance=np.linalg.norm(vectors,axis=-1)
        visible=distance<=limit
        nearby=self._near_buildings(origin,limit)
        visible &= ~_blocked_rays(origin,points,[item[1] for item in nearby])
        squared=np.sum(vectors*vectors,axis=-1)
        for actor in obstacles:
            if actor['id']==exclude_id:continue
            relative=np.array([actor['x_m'],actor['y_m']])-origin
            if np.linalg.norm(relative)>limit+actor['radius_m']:continue
            along=np.sum(vectors*relative,axis=-1)/np.maximum(squared,1e-12)
            closest=np.maximum(0.,float(relative@relative)-along*along*squared)
            radius=actor['radius_m']
            entry=along-np.sqrt(np.maximum(0.,radius*radius-closest))/np.sqrt(np.maximum(squared,1e-12))
            visible &= ~((closest<=radius*radius)&(entry>1e-7)&(entry<1-1e-7))
        return visible

    def _evidence(self, capture, sequence, semantic_status):
        p = self._policy
        modules = tuple(ModuleEvidence(n,semantic_status if n=="semantic" else "ok",
                           capture if n=="semantic" else self._time,
                           sequence if n=="semantic" else self._step,p.clock_id,
                           p.calibration_sha256,p.artifacts[n]) for n in sorted(REQUIRED_MODULES))
        forward=max(16.1,self._path_length_m+6.)
        xs = -4.0+(np.arange(math.ceil((forward+4.)/.3))+.5)*.3
        y_min,y_max=-9.,9.
        if self._extended_course:
            low,high=self._plan_y_bounds
            y_min,y_max=min(y_min,low-4.5),max(y_max,high+4.5)
            if math.ceil((y_max-y_min)/.3)*len(xs)>self._policy.max_grid_cells:
                # An over-budget arc cannot acquire observation evidence by
                # truncating the path. Keep the small grid and let review veto.
                y_min,y_max=-9.,9.
        ys = y_min+(np.arange(math.ceil((y_max-y_min)/.3))+.5)*.3
        gx,gy = np.meshgrid(xs,ys)
        wx,wy = self._to_world(gx,gy)
        occupied = np.zeros(gx.shape,dtype=bool)
        tracks = []
        obstacles=self._obstacles()
        observed=np.hypot(gx,gy)<=self._sensor_range_m()
        visible_actor_ids=None
        if self._region is not None:
            points=np.stack((wx,wy),axis=-1)
            observed=self._visible_region_points(points,obstacles)
            origin=np.array([self._car.x_m,self._car.y_m])
            for building in self._near_buildings(origin,self._sensor_range_m()):
                occupied |= _inside_polygon(wx,wy,building[1])
            visible_actor_ids=set()
            for obstacle in obstacles:
                point=np.array([obstacle['x_m'],obstacle['y_m']])
                if np.linalg.norm(point-origin)<=self._sensor_range_m() and bool(self._visible_region_points(point,obstacles,obstacle['id'])):
                    visible_actor_ids.add(obstacle['id'])
            self._regional_visibility={'idealized_range_m':self._sensor_range_m(),
                                       'visible_actor_ids':sorted(visible_actor_ids),
                                       'hidden_or_out_of_range_actors':len(obstacles)-len(visible_actor_ids),
                                       'unobserved_grid_cells':int(np.count_nonzero(~observed)),
                                       'grid_cells':int(observed.size)}
        for obstacle in obstacles:
            occupied |= np.hypot(wx-obstacle['x_m'],wy-obstacle['y_m']) <= obstacle['radius_m']
            if visible_actor_ids is not None and obstacle['id'] not in visible_actor_ids:
                continue
            x,y = self._to_local(obstacle['x_m'],obstacle['y_m'])
            c,s = math.cos(self._car.yaw_rad),math.sin(self._car.yaw_rad)
            vx,vy=obstacle['vx_mps'],obstacle['vy_mps']
            acceleration_bound=obstacle.get("acceleration_bound_mps2",.05)
            if self._extended_course:
                # These synthetic tracks are generated by known stationary or
                # sinusoidal world trajectories, not a learned motion estimate.
                actor=next(a for a in self._region_actors if a['id']==obstacle['id'])
                acceleration_bound=actor['speed']**2/actor['amplitude'] if actor['amplitude'] else 0.
            tracks.append(Track(obstacle['id'],self._time,float(x),float(y),c*vx+s*vy,-s*vx+c*vy,
                                obstacle['radius_m'],.02,.02,acceleration_bound))
        if self._object_view['sampled_at_sim_time_s']==self._time:
            self._object_view['safety_track_ids']=[track.track_id for track in tracks]
        u,v,valid = rendering.project_ground(gx,gy)
        road = np.zeros(gx.shape,dtype=np.float32)
        ui=np.clip(np.rint(u).astype(int),0,rendering.WIDTH-1)
        vi=np.clip(np.rint(v).astype(int),0,rendering.HEIGHT-1)
        road[valid]=self._probability[vi[valid],ui[valid]]
        if self._extended_course and self.perception_callback is None:
            # Long-range world semantics belong only to the privileged demo.
            # A learned callback is never replaced by this synthetic evidence.
            road=(self._road_union_margin(wx,wy,self._sensor_range_m()+10.)>=0).astype(np.float32)
        if self._region is not None:
            # Semantic painting of footpaths does not grant vehicle passage.
            road[self._road_union_margin(wx,wy,max(35.,self._path_length_m+15.))<0]=0.
            occupied &= observed
        if self.scenario=='fog':
            observed &= gx < 7.5
        grid=Grid(-4.0,y_min,.3,observed,occupied,np.zeros_like(occupied),road,self._time,
                  p.clock_id,p.calibration_sha256)
        return SceneEvidence(modules,grid,tuple(tracks),'synthetic_'+self._condition(),True,True,
              not(self.scenario=='braking_degraded' and self._time>=4.0),
              .01,0.0,.01,p.clock_id,self._time,max(5.,.3+(max(self._car.speed_mps,self._effective_target_mps)+.12)/3.),self._time,p.calibration_sha256,p.clock_id)

    @staticmethod
    def _controls(controls):
        if controls is None:
            return {'steering':0.0,'throttle':0.0,'brake':1.0,'reset_safety':False}
        if not isinstance(controls,dict) or set(controls)-{'steering','throttle','brake','reset_safety'}:
            raise ValueError("invalid simulator controls")
        result={'steering':0.0,'throttle':0.0,'brake':1.0,'reset_safety':False,**controls}
        for key,low,high in [('steering',-1,1),('throttle',0,1),('brake',0,1)]:
            val=result[key]
            if isinstance(val,bool) or not isinstance(val,(float,int)) or not math.isfinite(val) or not low<=val<=high:
                raise ValueError("invalid "+key)
        if type(result['reset_safety']) is not bool:
            raise ValueError("reset_safety must be boolean")
        return result

    def step(self, controls=None):
        if self._closed:
            raise RuntimeError("simulator is closed")
        controls=self._controls(controls)
        if self._finished:
            return self.snapshot()
        before_distance=self._distance
        before_events=(self._interventions,self._collisions,self._offroad_events)
        capture,sequence,status=self._observe()
        plan=self._plan()
        lane_reason='wrong_way_veto' if not self._lane_direction_allowed else 'lane_boundary_veto'
        reviewed_plan=plan if self._lane_guard_allowed else replace(plan,valid=False,reason=lane_reason)
        state=VehicleState(self._time,self._car.speed_mps,
                  self._car.speed_mps*math.tan(self._car.steering_rad)/self.WHEELBASE_M,armed=True,estop=False)
        decision=self._supervisor.review(reviewed_plan,state,self._evidence(capture,sequence,status),self._time,
                                         reset_requested=controls['reset_safety'])
        if not self._lane_guard_allowed:
            decision=replace(decision,reasons=(lane_reason,*decision.reasons))
        self._safety={'mode':decision.mode,'reasons':list(decision.reasons),
                      'proposal_eligible':decision.proposal_eligible,'actuation_enabled':False}
        if self.mode=='manual':
            cap=self.target_speed_kmh/3.6
            speed_brake=1.0 if cap<=0 or self._car.speed_mps>cap else 0.0
            self._effective_target_mps=cap
            self._command=Command(steering_rad=controls['steering']*self.MAX_STEERING_RAD,
                 throttle=controls['throttle'] if controls['brake']<=0 and speed_brake<=0 and self._car.speed_mps<cap else 0.0,
                 brake=max(controls['brake'],speed_brake),enabled=True,reason='manual_simulation_override')
        elif self._traffic is not None and self._traffic.signal()['vehicles']=='red' and self._car.x_m<35.9 and 27.-self._car.x_m<=stopping_distance(self._car.speed_mps)+1.:
            self._command=self._controller.stop('traffic_signal_stop')
        elif self.target_speed_kmh==0 or self._effective_target_mps<.02:
            self._command=self._controller.stop('requested_speed_stop')
        elif self._at_braking_goal():
            self._command=self._controller.stop('goal_braking')
        elif decision.proposal_eligible:
            self._command=self._controller.update(plan,state,self._time)
            if (self._extended_course or self._speed_explicit) and self._command.reason=='tracking' and self._command.brake==0:
                # Synthetic drag/grade feed-forward lets a flat straight course
                # attain its requested ceiling without a permanent PI offset.
                _,grade=self._terrain_state()
                compensation=(.04*self._car.speed_mps+9.81*grade)/5.
                self._command=replace(self._command,throttle=float(np.clip(self._command.throttle+compensation,0.,1.)))
        else:
            self._command=self._controller.stop(decision.reasons[0])
        veto=not decision.proposal_eligible and decision.mode!='WARMUP_STOP'
        if veto and not self._previous_stop:
            self._interventions+=1
            self._event('safety_stop',decision.reasons[0])
        self._previous_stop=veto
        self._integrate(self._command)
        if self.mode=='autonomous':
            self._lane_samples+=1
            self._lane_duration+=self.DT_S
        self._step+=1
        self._time=round(self._step*self.DT_S,10)
        if self._region is not None:
            progress,nearest=_polyline_projection(np.array([self._car.x_m,self._car.y_m]),self._road)
            heading_error=math.atan2(math.sin(self._car.yaw_rad-self._region['route']['goal_heading_rad']),
                                     math.cos(self._car.yaw_rad-self._region['route']['goal_heading_rad']))
            inside_goal=(progress>=self._route_length-1.0 and np.linalg.norm(np.array([self._car.x_m,self._car.y_m])-self._road[-1])<=1.5
                         and abs(heading_error)<=.4 and self._body_inside_road_union())
        else:
            goal_distance=float(_distance_to_road(self._car.x_m,self._car.y_m,self._road))
            inside_goal=(self.goal_x_m<=self._car.x_m<=self.goal_x_m+4.0
                         and goal_distance<=1.0
                         and goal_distance+self.BODY_RADIUS_M<=self.ROAD_HALF_WIDTH_M)
        if (inside_goal and self._car.speed_mps<=.02 and not self._collisions
                and not self._offroad_events and not self._offroad):
            self._goal_reached,self._finished=True,True
            self._event('goal','Stopped in the goal region')
        if self._step>=self._step_budget:
            self._finished=True
            self._event('timeout','Simulation step budget reached')
        self._performance.sample(time_s=self._time,dt_s=self.DT_S,mode=self.mode,
            position=(self._car.x_m,self._car.y_m),yaw_rad=self._car.yaw_rad,
            speed_mps=self._car.speed_mps,
            applied_target_mps=self._command.target_speed_mps if self.mode=='autonomous' and self._command.reason=='tracking' else 0.,
            steering_rad=self._car.steering_rad,distance_delta_m=self._distance-before_distance,
            safety_mode=decision.mode,safety_eligible=decision.proposal_eligible,
            intervention_delta=self._interventions-before_events[0],
            collision_delta=self._collisions-before_events[1],offroad_delta=self._offroad_events-before_events[2])
        snapshot=self.snapshot()
        if self._session_dir is not None:
            logged_snapshot=snapshot
            if self._region is not None:
                logged_snapshot={**snapshot,'world':{**snapshot['world'],'region':{
                    'id':self._region['id'],'geometry_sha256':self._region['geometry_sha256'],
                    'static_geometry_reference':'configuration.json#map_region'}}}
            with (self._session_dir/'steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(logged_snapshot,ensure_ascii=False,allow_nan=False)+'\n')
            if self._finished:self._write_summary('finished')
        return snapshot

    def _at_braking_goal(self):
        if self._region is None:return self._car.x_m>=self.goal_x_m
        progress,_=_polyline_projection(np.array([self._car.x_m,self._car.y_m]),self._road)
        return progress>=self._route_length-.6

    def _integrate(self, command):
        # Bound both time and translation. Collision tests run at every substep,
        # independently from masks, supervisor decisions or command reasons.
        extended=self._extended_course or self._speed_explicit
        _elevation,grade=self._terrain_state()
        count=max(10,math.ceil((self._car.speed_mps+8*self.DT_S)*self.DT_S/.04))
        delta=self.DT_S/count
        for index in range(count):
            before=(self._car.x_m,self._car.y_m)
            target=command.steering_rad
            self._car.steering_rad += float(np.clip(target-self._car.steering_rad,-.8*delta,.8*delta))
            t=self._time+(index+1)*delta
            braking=.7 if self.scenario=='braking_degraded' and t>=4.0 else (6.0 if extended else 3.5)
            acceleration=((5.0 if extended else 3.0)*command.throttle-braking*command.brake
                          -(.04 if extended else .08)*self._car.speed_mps-9.81*grade)
            self._car.speed_mps=float(np.clip(self._car.speed_mps+acceleration*delta,0,self.MAX_SPEED_MPS))
            yaw_rate=self._car.speed_mps*math.tan(self._car.steering_rad)/self.WHEELBASE_M
            mid=self._car.yaw_rad+yaw_rate*delta/2
            self._car.x_m+=self._car.speed_mps*math.cos(mid)*delta
            self._car.y_m+=self._car.speed_mps*math.sin(mid)*delta
            self._car.yaw_rad=math.atan2(math.sin(self._car.yaw_rad+yaw_rate*delta),
                                       math.cos(self._car.yaw_rad+yaw_rate*delta))
            self._distance+=math.hypot(self._car.x_m-before[0],self._car.y_m-before[1])
            if self._traffic is not None:
                self._traffic.advance(delta,self._car)
            self._check_geometry(t)
            lane_outside=self._lane.measure(self._car.x_m,self._car.y_m,self._car.yaw_rad)['outside']
            if lane_outside and not self._lane_outside:
                self._event('lane_departure','Assumed vehicle perimeter crossed the synthetic lane boundary')
                if self.mode=='autonomous':
                    self._lane_departures+=1
            if lane_outside and self.mode=='autonomous':
                self._lane_outside_time+=delta
            self._lane_outside=lane_outside

    def _check_geometry(self, at_time):
        for obstacle in self._obstacles(at_time):
            clearance=math.hypot(self._car.x_m-obstacle['x_m'],self._car.y_m-obstacle['y_m'])-self.BODY_RADIUS_M-obstacle['radius_m']
            if self._traffic is not None:
                clearance=vehicle_clearance(self._car,obstacle)
            self._min_clearance=clearance if self._min_clearance is None else min(self._min_clearance,clearance)
            if clearance<=0 and obstacle['id'] not in self._contacts:
                self._contacts.add(obstacle['id']);self._collisions+=1
                self._event('collision','World-geometry contact: '+obstacle['id'])
        if self._region is not None:
            # AABB distance is a lower bound. Cull only buildings that cannot
            # lower the best recorded clearance, rather than silently omitting
            # all buildings more than one metre from the body.
            search_distance=self.BODY_RADIUS_M+max(0.,self._min_clearance if self._min_clearance is not None else 2000.)
            for building in self._near_buildings(np.array([self._car.x_m,self._car.y_m]),search_distance):
                clearance=_polygon_clearance(self._car.x_m,self._car.y_m,building[1])-self.BODY_RADIUS_M
                self._min_clearance=clearance if self._min_clearance is None else min(self._min_clearance,clearance)
                contact_id='building:'+building[0]
                if clearance<=0 and contact_id not in self._contacts:
                    self._contacts.add(contact_id);self._collisions+=1
                    self._event('collision','World-geometry building contact: '+building[0])
            offroad=not self._body_inside_road_union()
        else:
            offroad=float(_distance_to_road(self._car.x_m,self._car.y_m,self._road))+self.BODY_RADIUS_M>self.ROAD_HALF_WIDTH_M
        if offroad and not self._offroad:
            self._offroad_events+=1
            self._event('offroad','Vehicle disk crossed the geometric road boundary')
        self._offroad=offroad

    def snapshot(self):
        elevation,grade=self._terrain_state()
        result={'scope':SCOPE,'scenario':{k:_SCENARIOS[self.scenario][k] for k in ('id','title')},
                'seed':self.seed,'mode':self.mode,'step':self._step,'sim_time_s':self._time,'finished':self._finished,
                'vehicle':asdict(self._car),'command':{k:asdict(self._command)[k] for k in ('steering_rad','throttle','brake','reason')},
                'safety':self._safety,'metrics':{'distance_m':self._distance,'collisions':self._collisions,
                    'offroad_events':self._offroad_events,'min_clearance_m':self._min_clearance,
                    'interventions':self._interventions,'goal_reached':self._goal_reached},
                'world':{'road_points':self._road.tolist(),'road_half_width_m':self.ROAD_HALF_WIDTH_M,
                         'obstacles':self._obstacles(),'goal_x_m':self.goal_x_m},
                'plan_world':self._plan_world,
                'perception':{'kind':'learned' if self.perception_callback is not None else 'synthetic_oracle',
                     'label':'Learned image semantics + oracle metric range/tracks' if self.perception_callback is not None else 'Synthetic oracle sensors; no learned model',
                     'learned':self.perception_callback is not None,'oracle_range':True,'oracle_tracks':True,
                     'synthetic_route_map':True,'model_identity':self._model_identity,
                     'model_error':self._model_error,'captured_at_sim_time_s':self._capture_time},
                'events':list(self._events)}
        result['world']['road_lanes']=self._road_lanes
        if self._traffic is not None:
            result['world']['traffic']=self._traffic.snapshot()
        result['object_perception']={**self._object_view,
            'age_s':max(0.,self._time-self._object_view['sampled_at_sim_time_s']),
            'frame_frozen':self._object_frame_frozen}
        result['performance']=self._performance.snapshot((self._car.x_m,self._car.y_m),self._car.yaw_rad)
        result['lane']={'source':'synthetic_assumed_lane_not_measured','width_m':self._lane.width_m,
                'current':self._lane.measure(self._car.x_m,self._car.y_m,self._car.yaw_rad),
                'autonomous':{'samples':self._lane_samples,'duration_s':self._lane_duration,
                    'inside_pct':None if self._lane_duration==0 else float(np.clip(
                        100.*(1.-self._lane_outside_time/self._lane_duration),0.,100.)),
                    'departure_events':self._lane_departures,'outside_time_s':self._lane_outside_time},
                'training':{'status':'not_trained','model_loaded':False,'label_space':'ego_lane_v1'},
                'guard':{'allowed':self._lane_guard_allowed,'checked':self._lane_guard_checked,'direction_allowed':self._lane_direction_allowed,
                    'reason':None if self._lane_guard_allowed else 'wrong_way_veto' if not self._lane_direction_allowed else 'lane_boundary_veto',
                    'scope':'synthetic_path_guard'}}
        boundaries=self._lane.boundaries
        result['world']['lane']={'source':'synthetic_assumed_lane_not_measured','width_m':self._lane.width_m,
                'left_boundary':boundaries['left'],'right_boundary':boundaries['right']}
        result['vehicle'].update(elevation_m=elevation,grade_percent=grade*100.)
        result['speed_control']={'mode':self.speed_mode,'target_speed_kmh':self.target_speed_kmh,
                'effective_target_kmh':self.target_speed_kmh if self.mode=='manual' else self._effective_target_mps*3.6,'max_speed_kmh':MAX_SPEED_KMH,
                'limiting_reasons':[] if self.mode=='manual' else self._speed_reasons,'synthetic_only':True,
                'user_setting_is_ceiling':True}
        if self._region is not None:
            progress,_=_polyline_projection(np.array([self._car.x_m,self._car.y_m]),self._road)
            result['world'].update(region=self._region,coordinates='east_north_m',
                    goal_position_m=self._road[-1].tolist(),route_progress_m=progress,route_length_m=self._route_length,
                    road_half_width_m=self._region['route']['half_width_m'])
            result['perception'].update(visibility=self._regional_visibility,
                    label=('Learned image semantics' if self.perception_callback else 'Synthetic oracle image semantics')+
                          ' + bounded, occlusion-filtered oracle metric rays/tracks',
                    omniscient_metric_visibility=False,map_derived_synthetic=True)
            if self._extended_course:
                result['perception'].update(map_derived_synthetic=False,
                    long_range_world_semantics=self.perception_callback is None,
                    synthetic_sensor_range_m=self._sensor_range_m())
            result['metrics']['route_progress_m']=progress
        return json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False))

    def frame(self, kind='camera'):
        if kind=='camera':return Image.fromarray(self._rgb.copy())
        if kind=='segmentation':return rendering.segmentation_overlay(self._rgb,self._probability)
        if kind=='objects':return object_display.overlay(self._rgb,self._object_view)
        raise ValueError('kind must be camera, segmentation or objects')

    def _configuration(self):
        result={'scope':SCOPE,'scenario':self.scenario,'seed':self.seed,'mode':self.mode,
                'fixed_dt_s':self.DT_S,'wheelbase_m':self.WHEELBASE_M,'body_disk_radius_m':self.BODY_RADIUS_M,
                'step_budget':self._step_budget,
                'road_half_width_m':self.ROAD_HALF_WIDTH_M,'autonomous_target_speed_mps':self._effective_target_mps,
                'speed_control':self.snapshot()['speed_control'],
                'normal_braking_deceleration_mps2':6.0 if self._extended_course or self._speed_explicit else 3.5,'degraded_braking_deceleration_mps2':.7,
                'camera':'synthetic chase perspective, 6m behind rear axle and 3m high; not onboard calibration',
                'physics':'2D kinematic bicycle; synthetic route grade adds longitudinal gravity; conservative circular body; no tyre/suspension/contact solver',
                'road_lane_assignment':self._road_lanes,
                'traffic_model':'rule-based substep car following, signal stops and pedestrian clearance' if self._traffic else 'legacy fixture actors',
                'object_display':{'source':object_display.SOURCE,'learned_detector_loaded':False,
                    'display_only':True,'confidence_measured':False,'projection_boxes_are_assumed_extents':True},
                'oracle_metric_range_and_tracks':True,'real_vehicle_actuation_enabled':False,
                'model_identity':self._model_identity,
                'collision_metric':'distinct world-geometry obstacle contacts; no rigid-body response',
                'goal_region':{'x_min_m':self.goal_x_m,'x_max_m':self.goal_x_m+4.0,
                               'max_centerline_distance_m':1.0,'whole_body_inside_road':True,
                               'no_prior_collision_or_offroad_event':True},
                'synthetic_collision_detection_independent_of_supervisor':True,
                'lane_keeping':{'source':'synthetic_assumed_lane_not_measured','width_m':self._lane.width_m,
                    'vehicle_rectangle_m':{'front':self._lane.front_m,'rear':self._lane.rear_m,
                                          'width':self._lane.vehicle_width_m},
                    'reference':'world reference polyline with 20m tangent endpoint extensions',
                    'guard_margin_m':self._lane.margin_m,'perimeter_sample_spacing_m':self._lane.SAMPLE_SPACING_M,
                    'extended_course_path':'pure pursuit bicycle rollout from actual pose and steering; speed=max(current,proposal)',
                    'rollout_desired_lateral_acceleration_mps2':1.4,
                    'sampling':'actual pose every physics substep; autonomous duration and events only',
                    'model_trained':False,'discrete_checks_not_continuous_containment_proof':True},
                'performance_metrics':{'schema_version':1,'scope':'synthetic_session_only',
                    'autonomous_samples_only':True,'sampling':'post-step, weighted by fixed simulation dt',
                    'speed_reference':'controller command after acceleration ramp; zero for stops',
                    'cross_track_reference':'distance to finite world reference polyline, including endpoints',
                    'history':'at most240 points;0.5sim-second spacing plus mode transitions;most recent120sim seconds'},
                'wallclock_inference_changes_playback_speed_only':True}
        if self._traffic is not None:
            result.update(traffic_fixture=self._traffic.snapshot(),
                collision_metric='assumed rectangle SAT separation bound for traffic cars; rectangle-circle contact for pedestrians',
                contact_body_rectangle_m={'front':2.4,'rear':1.,'width':1.8},
                existing_circular_shadow_review_retained=True)
        if self._region is not None:
            result.update(map_region=self._region,map_derived_synthetic_not_digital_twin=True,
                          road_half_width_m=self._region['route']['half_width_m'],
                          synthetic_actor_count=len(self._region_actors),
                          synthetic_vehicle_disk={'radius_m':2.2,'assumed_length_m':4.0,'assumed_width_m':1.8,
                                                  'reference':'geometric vehicle center; conservative circumscribed disk'},
                          actor_dynamics='Seeded circles; parked kerbside or bounded shuttles along actual OSM polylines. No traffic-law or agent-interaction model.',
                          road_union_containment='Conservative complete body-disk square cover (0.4m cells), tested against the union of drivable OSM centerline capsules with disclosed assumed widths.',
                          sensor_visibility='Idealized 360-degree simulator metric rays, range25m day/14m night; opaque building footprints and actor disks occlude. Not camera-derived metric depth.',
                          map_footways_grant_vehicle_passage=False,
                          goal_region={'center_xy_m':self._road[-1].tolist(),'radius_m':1.5,'min_route_progress_m':self._route_length-1.,
                                       'max_heading_error_rad':.4,'whole_body_inside_road_union':True,
                                       'no_prior_collision_or_offroad_event':True})
            if self._extended_course:
                result.update(map_derived_synthetic_not_digital_twin=False,
                    synthetic_regional_terrain_not_geographic_reconstruction=True,
                    sensor_visibility=f'Idealized synthetic metric rays up to {self._sensor_range_m():g}m with occlusion. Extended world semantics only without a learned callback.',
                    actor_dynamics='Seeded synthetic circles on procedural course polylines; no traffic-law or interaction model.',
                    road_union_containment='Conservative body-disk cover against procedural road capsules.')
        return result

    def report(self):
        return {'configuration':self._configuration(),'snapshot':self.snapshot(),
                'model_callback_calls':self._model_calls,'model_callback_successes':self._model_successes,
                'runtime_speed_benchmark':False,'real_sensor_validation':False,'real_driving_validation':False,
                'accuracy_measured':False,'contact_response_modeled':False,
                'hardware_modules_opened':False,'actuation_enabled':False}

    @staticmethod
    def _write_json(path, data):
        with path.open('x',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')

    def _write_summary(self, reason):
        if self._session_dir is not None and not (self._session_dir/'summary.json').exists():
            self._write_json(self._session_dir/'summary.json',{'end_reason':reason,**self.report()})

    def close(self):
        if not self._closed:
            self._write_summary('closed')
            self._closed=True
