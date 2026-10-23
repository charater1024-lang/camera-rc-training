"""Map-derived synthetic outcomes, not real Seoul traffic or driving evidence."""
import json
import math

import numpy as np
import pytest

from camera_rc.driving.types import Perception
from camera_rc.simulator import Engine
from camera_rc.simulator.engine import (
    _blocked_rays, _distance_to_road, _inside_polygon, _polygon_clearance,
)


def run(engine,steps,controls=None):
    for _ in range(steps):
        snapshot=engine.step(controls)
        if snapshot['finished']:break
    return engine.snapshot()


def test_seoul_reference_goal_uses_real_route_arc_length():
    engine=Engine('seoul_day')
    start=engine.snapshot()
    assert start['vehicle']['x_m']>100  # The old x>=36 goal rule must not apply.
    assert start['world']['coordinates']=='east_north_m'
    assert 79.9<start['world']['route_length_m']<80.1
    result=run(engine,450)
    assert result['finished'] and result['metrics']['goal_reached']
    assert result['metrics']['route_progress_m']>79
    assert result['metrics']['distance_m']>79
    assert result['metrics']['collisions']==result['metrics']['offroad_events']==0
    assert result['vehicle']['speed_mps']<=.02
    assert math.dist([result['vehicle']['x_m'],result['vehicle']['y_m']],
                     result['world']['goal_position_m'])<1.5
    assert result['safety']['actuation_enabled'] is False


def test_seoul_hazard_moves_then_stops_and_manual_control_can_collide():
    engine=Engine('seoul_hazard')
    result=run(engine,130)
    assert result['metrics']['distance_m']>5
    assert result['vehicle']['speed_mps']<=.02
    assert result['metrics']['collisions']==result['metrics']['offroad_events']==0
    assert result['safety']['mode']=='LATCHED_STOP'
    negative=Engine('seoul_hazard',mode='manual')
    result=run(negative,70,{'throttle':1.,'brake':0.})
    assert 'route_blocking_vehicle' in negative._contacts
    assert result['metrics']['collisions']>=1
    assert result['safety']['actuation_enabled'] is False


def test_regional_actor_geometry_is_plausible_and_motion_follows_source_roads():
    engine=Engine('seoul_day')
    assert len(engine._region_actors)==44
    actors=engine._obstacles(0.)
    assert all(actor['radius_m']>=2.1 for actor in actors if actor['kind']=='vehicle')
    assert any(a['amplitude']>0 and a['kind']=='vehicle' for a in engine._region_actors)
    assert any(a['id'].startswith('kerb_') for a in actors)
    start=np.array([engine._car.x_m,engine._car.y_m])
    for actor in actors:
        assert np.linalg.norm(np.array([actor['x_m'],actor['y_m']])-start)>engine.BODY_RADIUS_M+actor['radius_m']
    for time in np.linspace(0.,30.,61):
        for actor,definition in zip(engine._obstacles(float(time)),engine._region_actors):
            point=np.array([actor['x_m'],actor['y_m']])
            for building in engine._near_buildings(point,actor['radius_m']):
                assert _polygon_clearance(*point,building[1])>actor['radius_m']
            if definition['amplitude']:
                assert float(_distance_to_road(*point,definition['path']))<1e-7


def test_minimum_clearance_matches_all_buildings_not_only_near_contacts():
    engine=Engine('seoul_day',mode='manual')
    engine._car.x_m=173.71417774411657
    engine._car.y_m=-29.289107245020155
    engine._min_clearance=None
    engine._check_geometry(0.)
    point=np.array([engine._car.x_m,engine._car.y_m])
    expected=min([_polygon_clearance(*point,b[1])-engine.BODY_RADIUS_M for b in engine._buildings]+
                 [np.linalg.norm(point-np.array([a['x_m'],a['y_m']]))-a['radius_m']-engine.BODY_RADIUS_M
                  for a in engine._obstacles(0.)])
    assert engine._min_clearance==pytest.approx(expected,abs=1e-8)


def actual_building_approach(engine):
    """Find a long actual wall with clear exterior for a geometry negative test."""
    for building in engine._buildings:
        polygon=building[1]
        for a,b in zip(polygon[:-1],polygon[1:]):
            vector=b-a
            length=np.linalg.norm(vector)
            if length<6:continue
            middle=(a+b)/2
            normal=np.array([-vector[1],vector[0]])/length
            if bool(_inside_polygon(*(middle+normal*.1),polygon)):normal=-normal
            start=middle+normal*(engine.BODY_RADIUS_M+.5)
            if _polygon_clearance(*start,polygon)>engine.BODY_RADIUS_M+.45:
                return building,middle,normal,start
    raise AssertionError('No suitable actual wall in bundled map')


def test_manual_vehicle_contact_with_actual_building_ignores_learned_free_road():
    engine=Engine('seoul_day',mode='manual',
                  perception=lambda rgb:Perception(np.ones(rgb.shape[:2],dtype=np.float32)))
    building,middle,normal,start=actual_building_approach(engine)
    engine._car.x_m,engine._car.y_m=map(float,start)
    engine._car.yaw_rad=math.atan2(-normal[1],-normal[0])
    engine._car.speed_mps=3.
    result=run(engine,4,{'throttle':1.,'brake':0.})
    assert 'building:'+building[0] in engine._contacts
    assert result['metrics']['collisions']>=1
    assert result['perception']['model_error'] is None
    assert result['safety']['actuation_enabled'] is False


def test_actual_building_occludes_metric_targets_and_hidden_tracks():
    engine=Engine('seoul_day')
    building,middle,normal,start=actual_building_approach(engine)
    engine._car.x_m,engine._car.y_m=map(float,start)
    target=middle-normal*.5
    assert bool(_blocked_rays(start,target,[building[1]]))
    hidden={'id':'hidden_fixture','x_m':float(target[0]),'y_m':float(target[1]),
            'radius_m':.2,'kind':'pedestrian','vx_mps':0.,'vy_mps':0.}
    assert not bool(engine._visible_region_points(target,[hidden],hidden['id']))
    engine._obstacles=lambda at_time=None:[hidden]
    scene=engine._evidence(0.,0,'ok')
    assert all(track.track_id!=hidden['id'] for track in scene.tracks)
    assert (~scene.grid.observed).any()


def test_road_union_contains_body_across_intersection_without_shrinking_it():
    engine=Engine('seoul_day')
    engine._car.x_m=engine._car.y_m=0.
    engine._segments_a=np.array([[-20.,0.],[0.,-20.]])
    engine._segments_b=np.array([[20.,0.],[0.,20.]])
    engine._segment_widths=np.array([2.6,2.6])
    engine._segment_low=np.minimum(engine._segments_a,engine._segments_b)-2.6
    engine._segment_high=np.maximum(engine._segments_a,engine._segments_b)+2.6
    engine._road_near_origin=None
    assert engine.BODY_RADIUS_M==2.7
    assert engine._body_inside_road_union()
    engine._car.x_m=engine._car.y_m=6.
    assert not engine._body_inside_road_union()


def test_region_log_stores_static_geometry_once_and_retains_provenance(tmp_path):
    engine=Engine('seoul_day',output_dir=tmp_path/'session')
    engine.step()
    engine.close()
    directory=tmp_path/'session/session_0000'
    configuration=json.loads((directory/'configuration.json').read_text(encoding='utf-8'))
    step=json.loads((directory/'steps.jsonl').read_text(encoding='utf-8'))
    assert len(configuration['map_region']['buildings'])>500
    assert 'roads' not in step['world']['region']
    assert step['world']['region']['geometry_sha256']==configuration['map_region']['geometry_sha256']
    assert configuration['map_derived_synthetic_not_digital_twin'] is True
    assert configuration['real_vehicle_actuation_enabled'] is False
    assert configuration['synthetic_vehicle_disk']['radius_m']==2.2


def test_night_stays_explicitly_synthetic_with_bounded_metric_visibility():
    engine=Engine('seoul_night')
    result=run(engine,3)
    perception=result['perception']
    assert perception['oracle_range'] and perception['oracle_tracks']
    assert not perception['omniscient_metric_visibility']
    assert perception['visibility']['idealized_range_m']==14.
    assert perception['visibility']['hidden_or_out_of_range_actors']>0
    assert perception['visibility']['unobserved_grid_cells']>0
    assert result['metrics']['distance_m']>0
    assert not engine.report()['accuracy_measured']
