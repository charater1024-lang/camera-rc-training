"""Explicit synthetic lanes and interactive traffic safety fixtures."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from camera_rc.simulator import Engine
from camera_rc.simulator.lane import LaneCorridor, road_lane_layout
from camera_rc.simulator.traffic import TrafficScene, vehicle_clearance


def stationary_ego(x=0.):
    return SimpleNamespace(x_m=x,y_m=0.,yaw_rad=0.,speed_mps=0.)


def advance(scene,seconds,ego=None):
    ego=ego or stationary_ego()
    for _ in range(round(seconds/.01)): scene.advance(.01,ego)


def test_ego_lane_divider_and_opposite_lane_have_distinct_centers_and_directions():
    layout=road_lane_layout(LaneCorridor([[0.,0.],[100.,0.]]))
    assert layout['lane_count']==2 and layout['ego_lane_id']=='ego_forward'
    assert all(p[1]==pytest.approx(1.8) for p in layout['divider'])
    assert all(p[1]==pytest.approx(-1.8) for p in layout['ego_outer_boundary'])
    opposite=layout['lanes'][1]
    assert opposite['direction']=='reverse'
    assert all(p[1]==pytest.approx(3.6) for p in opposite['centerline'])
    assert opposite['centerline'][0][0]>opposite['centerline'][-1][0]


def test_wrong_way_stops_even_with_body_entirely_inside_the_ego_lane():
    engine=Engine()
    engine._car.yaw_rad=math.pi
    state=engine.step()
    assert state['lane']['current']['outside'] is False
    assert state['lane']['guard']['reason']=='wrong_way_veto'
    assert state['command']['brake']==1. and state['safety']['proposal_eligible'] is False


def test_drivable_road_does_not_allow_crossing_the_assigned_divider():
    engine=Engine();engine._car.y_m=2.
    state=engine.step()
    assert state['metrics']['offroad_events']==0
    assert state['lane']['current']['outside'] is True
    assert state['lane']['guard']['reason']=='lane_boundary_veto'
    assert state['command']['brake']==1.


def test_traffic_is_continuous_directional_and_deterministic():
    a,b=TrafficScene(),TrafficScene()
    before={actor['id']:actor for actor in a.obstacles()}
    advance(a,3.);advance(b,3.)
    assert a.snapshot()==b.snapshot() and a.obstacles()==b.obstacles()
    after={actor['id']:actor for actor in a.obstacles()}
    assert after['lead_car']['x_m']>before['lead_car']['x_m']
    assert after['oncoming_car']['x_m']<before['oncoming_car']['x_m']
    assert after['oncoming_car']['y_m']==3.6
    assert after['crossing_person']['y_m']==before['crossing_person']['y_m']


def test_following_vehicle_does_not_run_into_stationary_ego():
    scene=TrafficScene();advance(scene,20.)
    follower=next(car for car in scene.actors if car['id']=='following_car')
    assert follower['x_m']<=-5.9
    assert follower['speed']<.1


def test_car_brakes_at_red_and_pedestrian_crosses_only_after_wait():
    scene=TrafficScene();scene.time_s=8.
    lead=scene.actors[0];lead.update(x_m=20.,speed=5.)
    advance(scene,1.9)
    assert scene.pedestrian_y==6.2
    assert lead['x_m']<27.5 and lead['speed']<5.
    advance(scene,7.)
    assert scene.pedestrian_y<0. and scene.snapshot()['pedestrian_state']=='crossing'


def test_pedestrian_waits_for_vehicle_still_occupying_crosswalk():
    scene=TrafficScene();scene.time_s=10.
    scene.actors[0].update(x_m=32.,speed=0.,limit=0.)
    advance(scene,1.)
    assert scene.pedestrian_y==6.2
    assert scene.snapshot()['pedestrian_waiting_for_vehicle'] is True


def test_adjacent_lane_vehicle_is_not_a_rectangle_contact_but_overlap_is():
    ego=stationary_ego()
    actor=dict(x_m=.7,y_m=3.6,kind='vehicle',radius_m=2.2)
    assert vehicle_clearance(ego,actor)>0.
    actor['y_m']=0.
    assert vehicle_clearance(ego,actor)<0.
    actor.update(kind='pedestrian',x_m=2.3,y_m=0.,radius_m=.3)
    assert vehicle_clearance(ego,actor)<0.


def test_ego_obeys_red_signal_stop_before_crosswalk():
    engine=Engine('traffic_interaction')
    advance(engine._traffic,9.,engine._car)
    engine._time=9.;engine._step=90
    engine._car.x_m=26.;engine._car.speed_mps=2.
    state=engine.step()
    assert state['command']['reason']=='traffic_signal_stop'
    for _ in range(20):state=engine.step()
    assert state['vehicle']['x_m']<27.
    assert state['vehicle']['speed_mps']<=.02
    assert state['world']['traffic']['signal']['vehicles']=='red'
    assert state['safety']['actuation_enabled'] is False


def test_traffic_updates_with_physics_and_reset_is_clean():
    engine=Engine('traffic_interaction')
    first=engine.snapshot()
    for _ in range(15):state=engine.step()
    assert state['world']['traffic']['time_s']==pytest.approx(state['sim_time_s'])
    assert state['world']['obstacles']!=first['world']['obstacles']
    assert engine.snapshot()==state
    reset=engine.reset()
    assert reset['world']['traffic']['time_s']==0.
    assert reset['world']['obstacles']==first['world']['obstacles']
    assert np.asarray(engine.frame()).shape==(180,320,3)


def test_complete_signal_cycle_has_no_npc_contacts_or_teleporting_pedestrian():
    scene=TrafficScene();positions=[]
    for _ in range(2600):
        scene.advance(.01,stationary_ego());positions.append(scene.pedestrian_y)
    assert scene.actor_contact_events==0
    assert scene.pedestrian_y==pytest.approx(-3.,abs=.01)
    assert max(abs(b-a) for a,b in zip(positions,positions[1:]))<=.0126


def test_red_braking_does_not_release_after_an_existing_stop_line_overshoot():
    engine=Engine('traffic_interaction');engine._traffic.time_s=9.
    engine._car.x_m=28.;engine._car.speed_mps=1.
    state=engine.step()
    assert state['command']['reason']=='traffic_signal_stop'
    assert state['command']['brake']==1.


def test_user_ceiling_does_not_override_assumed_traffic_fixture_speed_limit():
    engine=Engine('traffic_interaction',speed_mode='user',target_speed_kmh=150)
    plan=engine._plan()
    assert plan.target_speed_mps<=6.
    assert 'traffic_speed_limit' in engine.snapshot()['speed_control']['limiting_reasons']
