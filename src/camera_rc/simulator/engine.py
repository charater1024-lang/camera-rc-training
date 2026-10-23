"""Low-speed synthetic world, vehicle dynamics and independent outcome checks.

The default sensor is an explicitly privileged oracle. A learned callback can
replace image semantics, while metric occupancy/tracks remain synthetic. This
is a closed-loop software demonstration, not a real-vehicle safety assessment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
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
from . import rendering


SCOPE = "synthetic_closed_loop_demo_not_vehicle_validation"
SCENARIOS = [
    {"id": "clear", "title": "Clear closed course", "description": "Follow a clear wide road and stop at the goal."},
    {"id": "curved", "title": "Curved road", "description": "Steering changes pose and the following camera observation."},
    {"id": "static_obstacle", "title": "Stopped obstacle", "description": "An obstacle blocks the route; autonomous mode must stop."},
    {"id": "crossing_pedestrian", "title": "Crossing pedestrian", "description": "A moving pedestrian enters the route; synthetic tracks provide a veto."},
    {"id": "night", "title": "Night obstacle", "description": "Dark synthetic camera imagery; oracle geometry is still privileged."},
    {"id": "fog", "title": "Limited visibility", "description": "A short observed range cannot support the full stopping corridor."},
    {"id": "sensor_freeze", "title": "Frozen camera", "description": "Camera exposure and frame identity freeze after three simulation seconds."},
    {"id": "sensor_dropout", "title": "Camera dropout", "description": "Camera evidence disappears after three simulation seconds."},
    {"id": "braking_degraded", "title": "Degraded braking", "description": "Declared braking capability fails and actual deceleration is reduced."},
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
        fraction = np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy), 0, 1)
        distance = np.minimum(distance, np.hypot(x-a[0]-fraction*dx, y-a[1]-fraction*dy))
    return distance


class Engine:
    DT_S = 0.1
    WHEELBASE_M = 2.6
    BODY_RADIUS_M = 2.7
    ROAD_HALF_WIDTH_M = 5.5
    MAX_STEERING_RAD = 0.45
    MAX_SPEED_MPS = 8.0  # manual negative controls only; autonomous target is 2 m/s.
    MAX_STEPS = 900

    def __init__(self, scenario="clear", seed=42, mode="autonomous", perception=None, output_dir=None):
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
        self.goal_x_m = 36.0
        xs = np.arange(-20.0, 102.0, 2.0) if self.scenario == "curved" else np.array([-20.0,100.0])
        self._road = np.column_stack((xs, self._road_y(xs)))
        self._render_cache_key, self._render_cache = None, None
        rng = np.random.default_rng(seed)
        self._obstacle_x = 24.0 + float(rng.uniform(-0.4, 0.4))
        self._command = Command(reason="reset")
        self._plan_world = []
        self._safety = {"mode": "WARMUP_STOP", "reasons": ["initial_observation_required"],
                        "proposal_eligible": False, "actuation_enabled": False}
        self._controller = Controller({"wheelbase_m": self.WHEELBASE_M, "max_steering_rad": self.MAX_STEERING_RAD,
            "steering_rate_rad_s": 0.8, "lookahead_m": 4.0, "lookahead_speed_gain_s": 0.2,
            "max_speed_mps": 2.0, "max_accel_mps2": 1.0, "max_decel_mps2": 3.0,
            "max_throttle": 1.0, "nominal_dt_s": self.DT_S, "stop_to_align": False})
        self._policy = ShadowPolicy(clock_id="sim_fixed_dt", calibration_sha256=_digest("synthetic_chase_camera_v1"),
            artifacts={n:_digest("synthetic_v1_"+n) for n in REQUIRED_MODULES},
            allowed_conditions=("synthetic_day", "synthetic_night", "synthetic_fog"),
            max_speed_mps=4.0, min_braking_deceleration_mps2=3.0, acceleration_during_delay_mps2=0.5,
            reaction_bound_s=0.2, stopping_margin_m=0.3, body_radius_m=self.BODY_RADIUS_M,
            lateral_margin_m=0.1, max_localization_error_m=0.1, max_speed_error_mps=0.1,
            max_motion_compensation_error_m=0.1, max_observation_age_s=0.15, max_state_age_s=0.15,
            max_plan_age_s=0.15, max_skew_s=0.05, max_curvature_per_m=0.3,
            max_lateral_acceleration_mps2=2.0, max_path_sampling_m=0.12,
            healthy_cycles_to_reset=2)
        self._supervisor = ShadowSafetySupervisor(self._policy)
        self._rgb, self._probability = self._render()
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

    def _obstacles(self, at_time=None):
        t = self._time if at_time is None else at_time
        if self.scenario in {"static_obstacle", "night", "braking_degraded"}:
            return [{"id":"stopped_obstacle", "x_m":self._obstacle_x, "y_m":0.0,
                     "radius_m":0.8, "kind":"obstacle", "vx_mps":0.0, "vy_mps":0.0}]
        if self.scenario == "crossing_pedestrian":
            moving = 3.0 <= t < 15.0
            return [{"id":"pedestrian", "x_m":18.0, "y_m":6.0-min(12.0,max(0.0,t-3.0)),
                     "radius_m":0.4, "kind":"pedestrian", "vx_mps":0.0, "vy_mps":-1.0 if moving else 0.0}]
        return []

    def _condition(self):
        return self.scenario if self.scenario in {"night", "fog"} else "day"

    def _render(self):
        obstacles = self._obstacles()
        key = (self._car.x_m,self._car.y_m,self._car.yaw_rad,self._road.tobytes(),
               self.ROAD_HALF_WIDTH_M,self._condition(),self.seed,
               tuple(tuple(sorted(obstacle.items())) for obstacle in obstacles))
        if key != self._render_cache_key:
            self._render_cache = rendering.render_observation(asdict(self._car), self._road,
                          self.ROAD_HALF_WIDTH_M,obstacles,self._condition(),self.seed)
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

    def _observe(self):
        self._capture_time = self._time
        sequence = self._step
        frozen = self.scenario == "sensor_freeze" and self._time >= 3.0
        dropout = self.scenario == "sensor_dropout" and self._time >= 3.0
        if frozen and self._camera_frozen is not None:
            self._rgb, self._probability = (a.copy() for a in self._camera_frozen)
            self._capture_time = self._frozen_capture
            return self._frozen_capture, self._frozen_sequence, "ok"
        self._rgb, oracle_mask = self._render()
        self._model_error = None
        if dropout:
            self._rgb = np.zeros_like(self._rgb)
            self._probability = np.zeros(oracle_mask.shape, dtype=np.float32)
            self._model_error = "synthetic_camera_dropout"
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
        target_x = self._car.x_m+7.0
        x,y = self._to_local(target_x,float(self._road_y(target_x)))
        curvature = float(np.clip(2*y/max(1.0,x*x+y*y), -.12, .12))
        distance = np.linspace(0.0,8.0,81)
        if abs(curvature) < 1e-7:
            points = np.column_stack((distance,np.zeros_like(distance)))
        else:
            points = np.column_stack((np.sin(curvature*distance)/curvature,
                                       (1-np.cos(curvature*distance))/curvature))
        wx,wy = self._to_world(points[:,0],points[:,1])
        self._plan_world = np.column_stack((wx,wy)).tolist()
        return Plan(points,2.0,True,"synthetic_route_proposal",curvature)

    def _evidence(self, capture, sequence, semantic_status):
        p = self._policy
        modules = tuple(ModuleEvidence(n,semantic_status if n=="semantic" else "ok",
                           capture if n=="semantic" else self._time,
                           sequence if n=="semantic" else self._step,p.clock_id,
                           p.calibration_sha256,p.artifacts[n]) for n in sorted(REQUIRED_MODULES))
        xs = -4.0+(np.arange(67)+.5)*.3
        ys = -9.0+(np.arange(60)+.5)*.3
        gx,gy = np.meshgrid(xs,ys)
        wx,wy = self._to_world(gx,gy)
        occupied = np.zeros(gx.shape,dtype=bool)
        tracks = []
        for obstacle in self._obstacles():
            occupied |= np.hypot(wx-obstacle['x_m'],wy-obstacle['y_m']) <= obstacle['radius_m']
            x,y = self._to_local(obstacle['x_m'],obstacle['y_m'])
            c,s = math.cos(self._car.yaw_rad),math.sin(self._car.yaw_rad)
            vx,vy=obstacle['vx_mps'],obstacle['vy_mps']
            tracks.append(Track(obstacle['id'],self._time,float(x),float(y),c*vx+s*vy,-s*vx+c*vy,
                                obstacle['radius_m'],.02,.02,.05))
        u,v,valid = rendering.project_ground(gx,gy)
        road = np.zeros(gx.shape,dtype=np.float32)
        ui=np.clip(np.rint(u).astype(int),0,rendering.WIDTH-1)
        vi=np.clip(np.rint(v).astype(int),0,rendering.HEIGHT-1)
        road[valid]=self._probability[vi[valid],ui[valid]]
        observed=np.ones(gx.shape,dtype=bool)
        if self.scenario=='fog':
            observed &= gx < 7.5
        grid=Grid(-4.0,-9.0,.3,observed,occupied,np.zeros_like(occupied),road,self._time,
                  p.clock_id,p.calibration_sha256)
        return SceneEvidence(modules,grid,tuple(tracks),'synthetic_'+self._condition(),True,True,
              not(self.scenario=='braking_degraded' and self._time>=4.0),
              .01,0.0,.01,p.clock_id,self._time,5.0,self._time,p.calibration_sha256,p.clock_id)

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
        capture,sequence,status=self._observe()
        plan=self._plan()
        state=VehicleState(self._time,self._car.speed_mps,
                  self._car.speed_mps*math.tan(self._car.steering_rad)/self.WHEELBASE_M,armed=True,estop=False)
        decision=self._supervisor.review(plan,state,self._evidence(capture,sequence,status),self._time,
                                         reset_requested=controls['reset_safety'])
        self._safety={'mode':decision.mode,'reasons':list(decision.reasons),
                      'proposal_eligible':decision.proposal_eligible,'actuation_enabled':False}
        if self.mode=='manual':
            self._command=Command(steering_rad=controls['steering']*self.MAX_STEERING_RAD,
                 throttle=controls['throttle'] if controls['brake']<=0 else 0.0,
                 brake=controls['brake'],enabled=True,reason='manual_simulation_override')
        elif self._car.x_m>=self.goal_x_m:
            self._command=self._controller.stop('goal_braking')
        elif decision.proposal_eligible:
            self._command=self._controller.update(plan,state,self._time)
        else:
            self._command=self._controller.stop(decision.reasons[0])
        veto=not decision.proposal_eligible and decision.mode!='WARMUP_STOP'
        if veto and not self._previous_stop:
            self._interventions+=1
            self._event('safety_stop',decision.reasons[0])
        self._previous_stop=veto
        self._integrate(self._command)
        self._step+=1
        self._time=round(self._step*self.DT_S,10)
        goal_distance=float(_distance_to_road(self._car.x_m,self._car.y_m,self._road))
        inside_goal=(self.goal_x_m<=self._car.x_m<=self.goal_x_m+4.0
                     and goal_distance<=1.0
                     and goal_distance+self.BODY_RADIUS_M<=self.ROAD_HALF_WIDTH_M)
        if (inside_goal and self._car.speed_mps<=.02 and not self._collisions
                and not self._offroad_events and not self._offroad):
            self._goal_reached,self._finished=True,True
            self._event('goal','Stopped in the goal region')
        if self._step>=self.MAX_STEPS:
            self._finished=True
            self._event('timeout','Simulation step budget reached')
        snapshot=self.snapshot()
        if self._session_dir is not None:
            with (self._session_dir/'steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(snapshot,ensure_ascii=False,allow_nan=False)+'\n')
            if self._finished:self._write_summary('finished')
        return snapshot

    def _integrate(self, command):
        # Bound both time and translation. Collision tests run at every substep,
        # independently from masks, supervisor decisions or command reasons.
        count=max(10,math.ceil((self._car.speed_mps+3*self.DT_S)*self.DT_S/.04))
        delta=self.DT_S/count
        for index in range(count):
            before=(self._car.x_m,self._car.y_m)
            target=command.steering_rad
            self._car.steering_rad += float(np.clip(target-self._car.steering_rad,-.8*delta,.8*delta))
            t=self._time+(index+1)*delta
            braking=.7 if self.scenario=='braking_degraded' and t>=4.0 else 3.5
            acceleration=3.0*command.throttle-braking*command.brake-.08*self._car.speed_mps
            self._car.speed_mps=float(np.clip(self._car.speed_mps+acceleration*delta,0,self.MAX_SPEED_MPS))
            yaw_rate=self._car.speed_mps*math.tan(self._car.steering_rad)/self.WHEELBASE_M
            mid=self._car.yaw_rad+yaw_rate*delta/2
            self._car.x_m+=self._car.speed_mps*math.cos(mid)*delta
            self._car.y_m+=self._car.speed_mps*math.sin(mid)*delta
            self._car.yaw_rad=math.atan2(math.sin(self._car.yaw_rad+yaw_rate*delta),
                                       math.cos(self._car.yaw_rad+yaw_rate*delta))
            self._distance+=math.hypot(self._car.x_m-before[0],self._car.y_m-before[1])
            self._check_geometry(t)

    def _check_geometry(self, at_time):
        for obstacle in self._obstacles(at_time):
            clearance=math.hypot(self._car.x_m-obstacle['x_m'],self._car.y_m-obstacle['y_m'])-self.BODY_RADIUS_M-obstacle['radius_m']
            self._min_clearance=clearance if self._min_clearance is None else min(self._min_clearance,clearance)
            if clearance<=0 and obstacle['id'] not in self._contacts:
                self._contacts.add(obstacle['id']);self._collisions+=1
                self._event('collision','World-geometry contact: '+obstacle['id'])
        offroad=float(_distance_to_road(self._car.x_m,self._car.y_m,self._road))+self.BODY_RADIUS_M>self.ROAD_HALF_WIDTH_M
        if offroad and not self._offroad:
            self._offroad_events+=1
            self._event('offroad','Vehicle disk crossed the geometric road boundary')
        self._offroad=offroad

    def snapshot(self):
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
        return json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False))

    def frame(self, kind='camera'):
        if kind=='camera':return Image.fromarray(self._rgb.copy())
        if kind=='segmentation':return rendering.segmentation_overlay(self._rgb,self._probability)
        raise ValueError('kind must be camera or segmentation')

    def _configuration(self):
        return {'scope':SCOPE,'scenario':self.scenario,'seed':self.seed,'mode':self.mode,
                'fixed_dt_s':self.DT_S,'wheelbase_m':self.WHEELBASE_M,'body_disk_radius_m':self.BODY_RADIUS_M,
                'road_half_width_m':self.ROAD_HALF_WIDTH_M,'autonomous_target_speed_mps':2.0,
                'normal_braking_deceleration_mps2':3.5,'degraded_braking_deceleration_mps2':.7,
                'camera':'synthetic chase perspective, 6m behind rear axle and 3m high; not onboard calibration',
                'physics':'2D kinematic bicycle; conservative circular body; no tyre/suspension/contact solver',
                'oracle_metric_range_and_tracks':True,'real_vehicle_actuation_enabled':False,
                'model_identity':self._model_identity,
                'collision_metric':'distinct world-geometry obstacle contacts; no rigid-body response',
                'goal_region':{'x_min_m':self.goal_x_m,'x_max_m':self.goal_x_m+4.0,
                               'max_centerline_distance_m':1.0,'whole_body_inside_road':True,
                               'no_prior_collision_or_offroad_event':True},
                'synthetic_collision_detection_independent_of_supervisor':True,
                'wallclock_inference_changes_playback_speed_only':True}

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
