"""Closed-loop software outcomes, not learned accuracy or driving validation."""
import json

import numpy as np
import pytest

from camera_rc.driving.types import Perception
from camera_rc.simulator import Engine, SCENARIOS
from camera_rc.simulator import rendering
from camera_rc.simulator.engine import _distance_to_road


def run(engine, steps, controls=None):
    for _ in range(steps):
        snapshot = engine.step(controls)
        if snapshot['finished']:
            break
    return engine.snapshot()


def test_deterministic_closed_loop_and_images_follow_pose():
    first, second = Engine(seed=123), Engine(seed=123)
    image_at_start = np.asarray(first.frame())
    for _ in range(25):
        assert first.step() == second.step()
    assert first.snapshot()['vehicle']['x_m'] > 1
    assert not np.array_equal(image_at_start, np.asarray(first.frame()))
    assert np.array_equal(np.asarray(first.frame()), np.asarray(second.frame()))
    snapshot = first.snapshot()
    snapshot['vehicle']['x_m'] = 999
    assert first.snapshot()['vehicle']['x_m'] != 999
    assert snapshot['safety']['actuation_enabled'] is False


@pytest.mark.parametrize('scenario', ['clear', 'curved'])
def test_clear_and_curved_reach_goal_without_geometric_contacts(scenario):
    engine = Engine(scenario=scenario)
    snapshot = run(engine, 260)
    assert snapshot['finished'] and snapshot['metrics']['goal_reached']
    assert snapshot['metrics']['distance_m'] >= 36
    assert snapshot['vehicle']['speed_mps'] <= .02
    assert snapshot['metrics']['collisions'] == snapshot['metrics']['offroad_events'] == 0
    if scenario == 'curved':
        assert snapshot['vehicle']['y_m'] > 3


@pytest.mark.parametrize('scenario,steps', [
    ('static_obstacle',110), ('crossing_pedestrian',110), ('night',110),
    ('fog',5), ('sensor_freeze',65), ('sensor_dropout',65), ('braking_degraded',95),
])
def test_hazards_and_failures_stop_without_collision(scenario, steps):
    engine = Engine(scenario=scenario)
    snapshot = run(engine, steps)
    assert snapshot['safety']['mode'] == 'LATCHED_STOP'
    assert snapshot['safety']['actuation_enabled'] is False
    assert not snapshot['safety']['proposal_eligible']
    assert snapshot['vehicle']['speed_mps'] <= .02
    assert snapshot['command']['brake'] == 1
    assert snapshot['metrics']['interventions'] >= 1
    assert snapshot['metrics']['collisions'] == snapshot['metrics']['offroad_events'] == 0
    if scenario == 'sensor_freeze':
        assert snapshot['perception']['captured_at_sim_time_s'] < 3.0
        assert snapshot['sim_time_s'] > 6


def test_manual_negative_control_can_collide_independently_of_stop_decision():
    engine = Engine('static_obstacle', mode='manual')
    snapshot = run(engine, 75, {'steering':0, 'throttle':1, 'brake':0})
    assert snapshot['metrics']['collisions'] == 1
    assert snapshot['metrics']['min_clearance_m'] < 0
    assert not snapshot['safety']['actuation_enabled']
    assert snapshot['command']['reason'] == 'manual_simulation_override'


def test_substep_contact_catches_thin_obstacle_between_endpoints():
    engine = Engine(mode='manual')
    # Deliberately small geometry makes the contact interval shorter than one
    # outer step's translation. This negative control must not tunnel through.
    engine.BODY_RADIUS_M = .03
    engine._car.speed_mps = 8
    engine._obstacles = lambda at_time=None: [
        {'id':'thin','x_m':.4,'y_m':0.,'radius_m':.03,'kind':'obstacle','vx_mps':0.,'vy_mps':0.}]
    snapshot = engine.step({'steering':0,'throttle':0,'brake':0})
    assert snapshot['vehicle']['x_m'] > .7
    assert abs(snapshot['vehicle']['x_m']-.4) > .06
    assert snapshot['metrics']['collisions'] == 1


def test_manual_steering_can_leave_road_and_missing_controls_brake():
    engine = Engine(mode='manual')
    snapshot = run(engine, 55, {'steering':1.,'throttle':1.,'brake':0.})
    assert snapshot['metrics']['offroad_events'] > 0
    assert snapshot['vehicle']['steering_rad'] > 0
    assert engine.step()['command']['brake'] == 1


@pytest.mark.parametrize('x,y,prior_offroad', [(36.,8.,0),(42.,0.,0),(36.,0.,1)])
def test_goal_rejects_lateral_escape_overshoot_and_prior_offroad(x,y,prior_offroad):
    engine=Engine(mode='manual')
    engine._car.x_m=x
    engine._car.y_m=y
    engine._offroad_events=prior_offroad
    snapshot=engine.step()
    assert not snapshot['metrics']['goal_reached']
    assert not snapshot['finished']


def test_manual_public_api_offroad_goal_regression():
    engine=Engine(mode='manual')
    run(engine,30,{'steering':.22,'throttle':1.,'brake':0.})
    for _ in range(100):
        if engine.snapshot()['vehicle']['x_m']>=37:
            break
        engine.step({'steering':0.,'throttle':1.,'brake':0.})
    snapshot=run(engine,35,{'steering':0.,'throttle':0.,'brake':1.})
    assert snapshot['vehicle']['x_m']>40
    assert snapshot['vehicle']['speed_mps']==0
    assert snapshot['metrics']['offroad_events']>0
    assert not snapshot['metrics']['goal_reached']


def test_straight_polyline_simplification_preserves_geometry():
    engine=Engine()
    assert len(engine._road)==2
    x=np.array([-25.,-20.,0.,50.,100.,105.])
    y=np.array([1.,0.,-4.,5.,.2,3.])
    dense=np.column_stack((np.arange(-20.,102.,2.),np.zeros(61)))
    np.testing.assert_allclose(_distance_to_road(x,y,engine._road),
                               _distance_to_road(x,y,dense),atol=1e-13)


def test_render_cache_preserves_fresh_learned_calls_and_tracks(monkeypatch):
    original=rendering.render_observation
    calls=[]
    def render(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(rendering,'render_observation',render)
    engine=Engine(mode='manual',perception=lambda rgb:Perception(np.ones(rgb.shape[:2],dtype=np.float32)))
    run(engine,3)
    assert len(calls)==1
    assert engine.report()['model_callback_calls']==3
    assert engine.snapshot()['perception']['captured_at_sim_time_s']==.2
    engine.step({'throttle':1.,'brake':0.})
    engine.step()
    assert len(calls)==2
    engine.reset(scenario='crossing_pedestrian')
    reset_count=len(calls)
    engine._time=4.
    first=engine._render()[0]
    engine._time=5.
    second=engine._render()[0]
    assert len(calls)==reset_count+2
    assert not np.array_equal(first,second)


def test_learned_masks_change_gate_and_initial_frame_has_no_oracle_mask():
    def allowed(rgb):
        assert rgb.dtype == np.uint8 and rgb.shape == (180,320,3)
        return Perception(np.ones(rgb.shape[:2],dtype=np.float32))
    allowed.simulator_identity = {'checkpoint_sha256':'a'*64, 'device':'cpu', 'diagnostic':False}
    good = Engine(perception=allowed)
    blocked = Engine(perception=lambda rgb:Perception(np.zeros(rgb.shape[:2],dtype=np.float32)))
    assert good.snapshot()['perception']['model_error'] == 'inference_pending'
    assert not good._probability.any()
    assert run(good,25)['metrics']['distance_m'] > 1
    rejected = run(blocked,4)
    assert rejected['metrics']['distance_m'] == 0
    assert rejected['safety']['reasons'] == ['semantic_disagreement_or_nonroad']
    assert good.report()['configuration']['model_identity'] == allowed.simulator_identity
    assert good.snapshot()['perception']['oracle_range'] is True


@pytest.mark.parametrize('bad', ['exception','nonfinite','shape','wrong_type'])
def test_bad_learned_output_never_falls_back_to_oracle(bad):
    def callback(rgb):
        if bad == 'exception':
            raise RuntimeError('fixture failure')
        if bad == 'wrong_type':
            return np.ones(rgb.shape[:2])
        return Perception(np.full((2,2) if bad=='shape' else rgb.shape[:2],
                                  np.nan if bad=='nonfinite' else 1,dtype=np.float32))
    engine=Engine(perception=callback)
    snapshot=run(engine,3)
    assert snapshot['perception']['kind']=='learned'
    assert snapshot['perception']['model_error']
    assert snapshot['metrics']['distance_m']==0
    assert snapshot['safety']['reasons']==['module_unavailable']
    assert not engine._probability.any()


def test_reset_preserves_prior_session_logs_and_close_is_idempotent(tmp_path):
    engine=Engine(output_dir=tmp_path/'sessions')
    run(engine,3)
    first=tmp_path/'sessions/session_0000/steps.jsonl'
    original=first.read_bytes()
    engine.reset(scenario='fog',seed=7)
    assert first.read_bytes()==original
    assert (first.parent/'summary.json').is_file()
    run(engine,3)
    engine.close();engine.close()
    assert (tmp_path/'sessions/session_0001/summary.json').is_file()
    with pytest.raises(RuntimeError):engine.step()
    with pytest.raises(ValueError):Engine(output_dir=tmp_path/'sessions')


@pytest.mark.parametrize('controls', [{'throttle':float('nan')},{'steering':2},
                                     {'brake':-1},{'reset_safety':1},{'unexpected':1}])
def test_invalid_controls_are_rejected_without_state_mutation(controls):
    engine=Engine(mode='manual')
    before=engine.snapshot()
    with pytest.raises(ValueError):engine.step(controls)
    assert engine.snapshot()==before
    json.dumps(engine.report(),allow_nan=False)


def test_camera_projection_matches_inverse_ground_pixels():
    x,y,valid=rendering.ground_coordinates()
    u,v,projected=rendering.project_ground(x[valid],y[valid])
    rows,columns=np.nonzero(valid)
    # Float32 inverse rounding may move the exact boundary by a fraction;
    # interior pixels must preserve the documented integer-pixel convention.
    interior=(columns>1)&(columns<rendering.WIDTH-2)&(rows<rendering.HEIGHT-2)
    assert projected[interior].all()
    np.testing.assert_allclose(u[interior],columns[interior],atol=3e-5)
    np.testing.assert_allclose(v[interior],rows[interior],atol=3e-5)
    assert len(SCENARIOS)>=7
