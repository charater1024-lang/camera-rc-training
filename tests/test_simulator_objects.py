"""Display-only object observations and camera synchronization."""
import math
import numpy as np
import pytest
from camera_rc.simulator import Engine
from camera_rc.simulator.objects import SOURCE, observation, projected_box


def actor(identifier='a',x=10.,y=0.,kind='vehicle',vx=0.,vy=0.):
    return dict(id=identifier,x_m=x,y_m=y,radius_m=.8,kind=kind,vx_mps=vx,vy_mps=vy)


def test_bounded_observations_use_geometry_not_fake_detector_confidence():
    actors=[actor('hidden',8.),actor('far',40.),actor('visible',10.,kind='pedestrian',vy=-1.)]
    view=observation(actors,(0.,0.,0.),16.,{'visible','far'},2.,[[0.,0.],[100.,0.]])
    assert view['source']==SOURCE and view['learned_detector_loaded'] is False
    assert view['hidden_or_out_of_range_count']==2
    assert [a['id'] for a in view['objects']]==['visible']
    a=view['objects'][0]
    assert a['distance_m']==10. and a['speed_kmh']==3.6
    assert a['confidence'] is None and a['near_reference_route'] is True
    assert a['track_identity_source']=='synthetic_actor_id'


def test_relative_coordinates_rotate_and_off_camera_actors_have_no_box():
    view=observation([actor(x=0.,y=10.),actor('behind',x=0.,y=-10.)],(0.,0.,math.pi/2),16.,None,0.,[[0.,0.],[0.,100.]])
    values={a['id']:a for a in view['objects']}
    assert values['a']['relative_forward_m']==pytest.approx(10.)
    assert values['a']['relative_left_m']==pytest.approx(0.,abs=1e-12)
    assert values['a']['camera_box_xyxy'] is not None
    assert values['behind']['camera_box_xyxy'] is None


@pytest.mark.parametrize('kind',['vehicle','pedestrian','cone','obstacle'])
def test_projection_box_is_clipped_to_camera_pixels(kind):
    box=projected_box(-3.,1.,2.,kind)
    assert box is not None
    assert 0<=box[0]<=box[2]<=319 and 0<=box[1]<=box[3]<=179
    assert projected_box(200.,0.,1.,kind) is None
    assert projected_box(-7.,0.,1.,kind) is None
    assert projected_box(10.,100.,1.,kind) is None


def test_frame_overlay_preserves_raw_rgb_and_snapshot_queries_do_not_sample():
    engine=Engine('static_obstacle')
    engine._car.x_m=10.
    snapshot=engine.step()
    view=snapshot['object_perception']
    assert len(view['objects'])==1
    assert view['safety_track_ids']==['stopped_obstacle']
    assert view['objects'][0]['camera_box_xyxy'] is not None
    raw=np.asarray(engine.frame('camera')).copy()
    annotated=np.asarray(engine.frame('objects'))
    assert annotated.shape==raw.shape and not np.array_equal(annotated,raw)
    assert np.array_equal(np.asarray(engine.frame('camera')),raw)
    assert engine.snapshot()==snapshot
    assert engine.report()['configuration']['object_display']['display_only'] is True
    assert snapshot['safety']['actuation_enabled'] is False
    engine.reset('clear')
    assert engine.snapshot()['object_perception']['objects']==[]


def test_regional_building_occlusion_and_range_are_applied_to_display(monkeypatch):
    engine=Engine('seoul_day')
    actors=[actor('blocked',10.),actor('far',100.)]
    wall=np.array([[4.,-2.],[6.,-2.],[6.,2.],[4.,2.],[4.,-2.]])
    monkeypatch.setattr(engine,'_obstacles',lambda *args: actors)
    monkeypatch.setattr(engine,'_near_buildings',lambda *args:[('wall',wall)])
    view=engine._capture_objects()
    assert view['objects']==[] and view['hidden_or_out_of_range_count']==2


def test_frozen_camera_keeps_object_positions_boxes_and_timestamp_frozen(monkeypatch):
    engine=Engine('sensor_freeze')
    monkeypatch.setattr(engine,'_obstacles',lambda *args:[actor(x=10.+engine._time)])
    for _ in range(30): engine.step()
    captured=engine.snapshot()['object_perception']
    assert captured['frame_frozen'] is False
    image=np.asarray(engine.frame('objects')).copy()
    for _ in range(5): engine.step()
    frozen=engine.snapshot()['object_perception']
    assert frozen['frame_frozen'] is True and frozen['age_s']>.5
    assert frozen['sampled_at_sim_time_s']==captured['sampled_at_sim_time_s']==2.9
    assert frozen['objects']==captured['objects']
    assert np.array_equal(np.asarray(engine.frame('objects')),image)


def test_camera_dropout_does_not_paint_boxes_on_missing_camera(monkeypatch):
    engine=Engine('sensor_dropout')
    monkeypatch.setattr(engine,'_obstacles',lambda *args:[actor()])
    for _ in range(31): engine.step()
    view=engine.snapshot()['object_perception']
    assert view['status']=='camera_unavailable'
    assert view['objects']==[] and view['camera_box_count']==0
    assert not np.asarray(engine.frame('objects')).any()
