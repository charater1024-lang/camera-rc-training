"""No camera, torch, serial or training: architecture regression contracts."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np
import pytest
from camera_rc.driving.label_contract import validate_label_contract
from camera_rc.driving.behavior import RoadContext, validate_road_plan
from camera_rc.driving.safety_boundary import SafetyBoundary, SafetyObservation, GeometryObservation
from camera_rc.driving.shadow_scenarios import example, altered
from camera_rc.driving.config import DEFAULTS
from camera_rc.driving.runtime import ControlLoop
from camera_rc.driving.transport import DryRunTransport
from camera_rc.driving.types import Plan, VehicleState
from camera_rc.driving.metric_tracking import MetricDetection, MetricTracker
from camera_rc.driving.trajectory_checks import assess_trajectory
from camera_rc.driving.readiness import audit_bundle, REQUIRED_CHECKS


def road(now=10., **updates):
    return replace(RoadContext(now,"host_monotonic",((-1.,-2.),(8.,-2.),(8.,2.),(-1.,2.)),True,"none",True,True),**updates)


def test_road_and_ego_lane_semantics_are_not_interchangeable():
    for name,ids in (("road_surface_v1",[0]),("ego_lane_v1",[1])):
        stored={"data":{"label_space_id":name},"model":{"num_classes":2}}
        runtime={"expected_label_space_id":name,"drivable_class_ids":ids}
        assert validate_label_contract(runtime,stored)["drivable_class_ids"]==ids
        for override in ({"drivable_class_ids":[1-ids[0]]},{"expected_label_space_id":"other"}):
            with pytest.raises(ValueError):
                validate_label_contract({**runtime,**override},stored)
    with pytest.raises(ValueError):
        validate_label_contract({"drivable_class_ids":[0]}, {"data":{},"model":{"num_classes":2}})


@pytest.mark.parametrize("updates,reason", [({"signal":"red"},"traffic_signal_stop"),({"signal":"unknown"},"traffic_signal_stop"),({"direction_allowed":False},"wrong_way_veto"),({"right_of_way":False},"right_of_way_unknown_or_yield"),({"route_available":False},"route_unavailable"),({"captured_at_s":9.},"road_context_stale_or_wrong_clock")])
def test_rule_veto(updates,reason):
    _,plan,_,_=example()
    result=validate_road_plan(plan,road(**updates),10.,clock_id="host_monotonic",max_age_s=.2,body_radius_m=.3)
    assert not result.valid and result.reason==reason


def test_lane_body_and_polygon_rejection():
    plan=Plan(np.array([[0,0],[1,0],[2,0]]),.3,True,"test")
    options=dict(clock_id="host_monotonic",max_age_s=.2,body_radius_m=.3)
    assert validate_road_plan(plan,road(),10.,**options).valid
    assert validate_road_plan(plan,road(lane_polygon=((-1,-.2),(4,-.2),(4,.2),(-1,.2))),10.,**options).reason=="lane_boundary_veto"
    assert validate_road_plan(plan,road(lane_polygon=((-1,-1),(4,1),(-1,1),(4,-1))),10.,**options).reason=="invalid_lane_geometry"


def observation(seq=0,now=10.):
    policy,plan,state,scene=example(seq,now)
    return policy,plan,state,SafetyObservation(scene,road(now,clock_id=policy.clock_id),state,"synthetic",GeometryObservation(now,policy.clock_id,0.,0.,True,"measured_flat_ground"))


def test_default_command_boundary_stops_without_evidence_and_latches(tmp_path):
    class Feedback(DryRunTransport):
        def read_state(self,now): return VehicleState(now,0.,0.,True,False)
    cfg=copy.deepcopy(DEFAULTS)
    loop=ControlLoop(cfg,Feedback(),tmp_path)
    _,plan,_,_=example()
    loop.publish(plan,10.)
    result=loop.step(10.01)
    assert not result["command"]["enabled"] and result["command"]["brake"]==1.
    assert result["safety"]["mode"]=="LATCHED_STOP"


def test_shared_supervisor_warmup_recheck_and_expiry():
    policy,plan,state,obs=observation()
    boundary=SafetyBoundary({"policy":asdict(policy)})
    assert boundary.review(plan,state,obs,10.)=="fresh_healthy_cycles_required"
    assert boundary.review(plan,state,obs,10.01)=="fresh_healthy_cycles_required"
    for seq in (1,2):
        _,plan,state,obs=observation(seq,10.+seq*.05)
        reason=boundary.review(plan,state,obs,10.+seq*.05)
    assert reason is None
    assert boundary.review(plan,state,obs,10.11) is None
    assert boundary.review(plan,state,obs,10.8) is not None
    assert boundary.latched_reason is not None


def test_synthetic_evidence_cannot_cross_physical_boundary():
    policy,plan,state,obs=observation()
    boundary=SafetyBoundary({"mode":"diagnostic_local_only","policy":asdict(policy)},physical=True)
    assert boundary.review(plan,state,obs,10.)=="safety_evidence_origin_rejected"


def test_failed_detector_reaches_shared_runtime_veto():
    policy,plan,state,scene=altered("failed_detector")
    obs=SafetyObservation(scene,road(clock_id=policy.clock_id),state,"recorded",GeometryObservation(10.,policy.clock_id,0.,0.,True,"measured_flat_ground"))
    boundary=SafetyBoundary({"policy":asdict(policy)})
    assert boundary.review(plan,state,obs,10.)=="module_unavailable"


def test_metric_tracking_motion_compensation_and_occlusion():
    tracker=MetricTracker(max_speed_mps=10.,acceleration_bound_mps2=2.)
    d=lambda x:MetricDetection(x,0.,"car",.5,.05)
    a=tracker.update([d(10.)],1.,ego_world_pose=(0.,0.,0.),detector_status="ok")
    b=tracker.update([d(10.)],1.1,ego_world_pose=(1.,0.,0.),detector_status="ok")
    assert a["tracks"][0].track_id==b["tracks"][0].track_id
    assert b["tracks"][0].x_m==9 and b["tracks"][0].vx_mps==0
    c=tracker.update([],1.2,ego_world_pose=(1.,0.,0.),detector_status="failed")
    assert c["status"]=="degraded" and len(c["tracks"])==1
    assert c["tracks"][0].position_error_m>b["tracks"][0].position_error_m
    assert c["free_space_verified"] is False
    with pytest.raises(ValueError):
        tracker.update([],1.2,ego_world_pose=(1.,0.,0.),detector_status="ok")


def test_expired_track_is_not_free_space_proof():
    t=MetricTracker(max_speed_mps=10.,acceleration_bound_mps2=2.)
    t.update([MetricDetection(1,0,"person",.3,.1)],1.,ego_world_pose=(0,0,0),detector_status="ok")
    result=t.update([],2.,ego_world_pose=(0,0,0),detector_status="ok")
    assert result["status"]=="degraded" and result["expired_tracks"] and not result["free_space_verified"]


def test_trajectory_physical_constraints():
    limits=dict(max_speed_mps=2.,max_accel_mps2=1.,max_decel_mps2=1.,max_curvature_per_m=.5,max_lateral_accel_mps2=.5)
    good=np.array([[0,0,1],[1,0,1],[2,0,1]],dtype=float)
    assert assess_trajectory(good,[0,1,2],**limits)["feasible"]
    fast=good.copy();fast[:,0]*=10
    assert "speed_limit" in assess_trajectory(fast,[0,1,2],**limits)["reasons"]
    assert not assess_trajectory(good,[0,0,2],**limits)["feasible"]
    bend=np.array([[0,0,1],[1,0,1],[1,1,1]],dtype=float)
    assert "curvature_limit" in assess_trajectory(bend,[0,1,2],**limits)["reasons"]


def test_missing_measured_evidence_cannot_pass_readiness(tmp_path):
    path=tmp_path/"bundle.json";path.write_text("{}")
    result=audit_bundle(path)
    assert not result["audit_passed"] and not result["actuation_authorized"]
    assert all(v=="missing_or_invalid" for v in result["checks"].values())


def test_readiness_pins_and_measured_reports_are_bound(tmp_path):
    pins={}
    for key in ("checkpoint","calibration","vehicle_config","code_revision"):
        f=tmp_path/key;f.write_bytes(key.encode())
        pins[key]={"path":key,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
    odd={"area":"test-course","road_type":"flat","conditions":["day"],"max_speed_mps":.4}
    hashes={k:v["sha256"] for k,v in pins.items()};checks={}
    for check in REQUIRED_CHECKS:
        report={"platform":"rc_car","check":check,"passed":True,"measurement_source":"measured","artifact_hashes":hashes,"odd":odd,"independent_test_set":"holdout-1"}
        f=tmp_path/(check+".json");f.write_text(json.dumps(report))
        checks[check]={"report":f.name,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
    path=tmp_path/"bundle.json";path.write_text(json.dumps({"platform":"rc_car","odd":odd,"artifacts":pins,"checks":checks}))
    result=audit_bundle(path)
    assert result["audit_passed"] and not result["actuation_authorized"]
    (tmp_path/"checkpoint").write_bytes(b"changed")
    result=audit_bundle(path)
    assert not result["audit_passed"] and "artifact_hash_mismatch:checkpoint" in result["reasons"]


@pytest.mark.parametrize("change,reason", [({"geometry":None},"geometry_or_exposure_evidence_missing"),({"grade":.08},"flat_ground_assumption_violated"),({"pitch":.03},"flat_ground_assumption_violated"),({"sync":False},"geometry_or_exposure_evidence_missing")])
def test_geometry_and_exposure_contract(change,reason):
    policy,plan,state,obs=observation()
    if "geometry" in change:
        obs=replace(obs,geometry=None)
    else:
        g=obs.geometry
        obs=replace(obs,geometry=replace(g,road_grade=change.get("grade",g.road_grade),camera_pitch_delta_rad=change.get("pitch",g.camera_pitch_delta_rad),exposure_timing_verified=change.get("sync",True)))
    assert SafetyBoundary({"policy":asdict(policy)}).review(plan,state,obs,10.)==reason



def test_progress_is_separate_from_driving_safety_and_has_all_workstreams():
    from camera_rc.simulator.progress import improvement_progress
    p=improvement_progress()
    assert p["total_items"]==8
    assert p["implemented_items"]+p["partial_items"]==8
    assert not p["physical_output"] and p["measured_validation_complete"]==0
    assert p["scope"]=="engineering_progress_not_driving_performance"


def test_malformed_scene_fails_closed():
    policy,plan,state,obs=observation()
    boundary=SafetyBoundary({"policy":asdict(policy)})
    assert boundary.review(plan,state,replace(obs,scene=None),10.)=="malformed_safety_evidence"


def test_current_telemetry_cannot_outpace_old_evidence():
    policy,plan,state,obs=observation()
    boundary=SafetyBoundary({"policy":asdict(policy)})
    assert boundary.review(plan,replace(state,speed_mps=.9),obs,10.)=="telemetry_exceeds_evidence_speed_bound"



def test_lane_graph_respects_direction_closures_and_geometry():
    from camera_rc.driving.route_graph import LaneGraph,LaneSegment
    lanes=[LaneSegment("a",((0,0),(1,0)),("b",),.5),LaneSegment("b",((1,0),(2,0)),(),.4)]
    graph=LaneGraph(lanes,source="synthetic")
    result=graph.route("a","b")
    assert result["available"] and result["lane_ids"]==["a","b"] and result["speed_limit_mps"]==.4
    assert not graph.route("b","a")["available"]
    assert not graph.route("a","b",closed_lanes=["b"])["available"]
    with pytest.raises(ValueError,match="Disconnected"):
        LaneGraph([replace(lanes[0],successors=("c",)),LaneSegment("c",((10,0),(11,0)),(),.4)],source="synthetic")



def test_red_signal_is_a_hold_and_can_resume_with_fresh_green_evidence():
    policy,plan,state,obs=observation()
    boundary=SafetyBoundary({"policy":asdict(policy)})
    red=replace(obs,road=replace(obs.road,signal="red"))
    assert boundary.review(plan,state,red,10.)=="traffic_signal_stop"
    assert boundary.latched_reason is None and boundary.last_decision["mode"]=="HOLD_STOP"
    for seq in (1,2,3):
        _,plan,state,obs=observation(seq,10.+seq*.05)
        result=boundary.review(plan,state,obs,10.+seq*.05)
    assert result is None



def test_control_loop_uses_supplied_scene_and_rechecks_between_camera_frames(tmp_path):
    policy,plan,state,obs=observation()
    class Feedback(DryRunTransport):
        def read_state(self,now): return replace(state,timestamp=now)
    cfg=copy.deepcopy(DEFAULTS);cfg["safety"]={"mode":"required","policy":asdict(policy)}
    loop=ControlLoop(cfg,Feedback(),tmp_path)
    for seq in range(3):
        now=10.+seq*.05
        _,plan,state,obs=observation(seq,now)
        loop.publish(plan,now,obs)
        item=loop.step(now)
    assert item["command"]["enabled"] and not item["physical_output"]
    assert item["safety"]["proposal_eligible"]
    assert loop.step(10.11)["command"]["enabled"]
    item=loop.step(10.5)
    assert not item["command"]["enabled"] and not item["safety"]["proposal_eligible"]


def test_validation_booleans_cannot_enable_unintegrated_physical_output(monkeypatch):
    from camera_rc.driving.config import check_actuation
    cfg=copy.deepcopy(DEFAULTS)
    cfg["transport"].update(kind="serial",calibration_confirmed=True)
    cfg["models_validated_for_control"]=True
    cfg["calibration"]={"diagnostic_only":False}
    monkeypatch.setattr("camera_rc.driving.config.camera_device_identity",lambda *a,**kw:"measured-device")
    with pytest.raises(ValueError,match="Physical evidence producers"):
        check_actuation(cfg,True)
