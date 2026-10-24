"""Deterministic small traffic interaction fixture, not calibrated social behavior."""
import math


class TrafficScene:
    CROSSING_X=32.
    PERIOD_S=24.

    def __init__(self,seed=42):
        self.time_s=0.
        self.actors=[
            dict(id='lead_car',kind='vehicle',x_m=17.,y_m=0.,speed=4.,direction=1.,limit=6.,lane_id='ego_forward'),
            dict(id='following_car',kind='vehicle',x_m=-26.,y_m=0.,speed=3.,direction=1.,limit=5.,lane_id='ego_forward'),
            dict(id='oncoming_car',kind='vehicle',x_m=85.,y_m=3.6,speed=6.,direction=-1.,limit=7.,lane_id='opposing_reverse')]
        self.pedestrian_y=6.2
        self.pedestrian_velocity=0.
        self.waiting_for_vehicle=False
        self.actor_contact_events=0
        self._contacts=set()

    def signal(self):
        phase=self.time_s%self.PERIOD_S
        return {'vehicles':'green' if phase<8. else 'red',
                'pedestrians':'walk' if 10.<=phase<18. else 'wait',
                'phase_s':phase,'cycle_s':self.PERIOD_S}

    def advance(self,dt,ego):
        if not math.isfinite(dt) or dt<=0 or dt>.02:
            raise ValueError('Traffic updates require positive physics substeps <=.02s')
        signal=self.signal()
        ped_on_road=-1.8<self.pedestrian_y<5.4
        for car in sorted(self.actors,key=lambda item:item['direction']*item['x_m'],reverse=True):
            direction=car['direction']
            route_end=150. if direction>0 else -30.
            desired=min(car['limit'],math.sqrt(6.*max(0.,direction*(route_end-car['x_m'])-.5)))
            line=27. if direction>0 else 38.
            gap=direction*(line-car['x_m'])
            if (signal['vehicles']=='red' or ped_on_road) and gap>=0:
                desired=min(desired,math.sqrt(2.*3.*max(0.,gap-.4)))
            leaders=[(other['x_m'],other['speed']) for other in self.actors if other is not car and other['lane_id']==car['lane_id']]
            if car['lane_id']=='ego_forward':
                leaders.append((ego.x_m,ego.speed_mps))
            for lead_x,lead_speed in leaders:
                separation=direction*(lead_x-car['x_m'])
                if separation>0:
                    desired=min(desired,max(0.,(separation-6.)/1.5),math.sqrt(lead_speed**2+6.*max(0.,separation-6.)))
            old_speed=car['speed']
            car['speed']+=max(-3.*dt,min(1.5*dt,desired-car['speed']))
            car['x_m']+=direction*(old_speed+car['speed'])*dt/2.
        crossing_blocked=any(abs(car['x_m']-self.CROSSING_X)<4.7 for car in self.actors)
        crossing_blocked |= abs(ego.x_m-self.CROSSING_X)<4.7
        self.waiting_for_vehicle=signal['pedestrians']=='walk' and crossing_blocked and not ped_on_road
        can_walk=(signal['pedestrians']=='walk' or ped_on_road) and not self.waiting_for_vehicle
        previous=self.pedestrian_y
        remaining=max(0.,self.pedestrian_y+3.)
        desired=-min(1.25,math.sqrt(4.*remaining)) if can_walk else 0.
        velocity=self.pedestrian_velocity+max(-2.*dt,min(2.*dt,desired-self.pedestrian_velocity))
        self.pedestrian_y=max(-3.,self.pedestrian_y+(self.pedestrian_velocity+velocity)*dt/2.)
        self.pedestrian_velocity=(self.pedestrian_y-previous)/dt
        contacts=set()
        for i,a in enumerate(self.actors):
            for b in self.actors[i+1:]:
                if abs(a['x_m']-b['x_m'])<=4.4 and abs(a['y_m']-b['y_m'])<=1.8:
                    contacts.add(tuple(sorted((a['id'],b['id']))))
            if math.hypot(max(0.,abs(a['x_m']-self.CROSSING_X)-2.2),max(0.,abs(a['y_m']-self.pedestrian_y)-.9))<=.3:
                contacts.add((a['id'],'crossing_person'))
        self.actor_contact_events+=len(contacts-self._contacts)
        self._contacts=contacts
        self.time_s+=dt

    def obstacles(self):
        result=[]
        for car in self.actors:
            result.append({key:car[key] for key in ('id','kind','x_m','y_m','lane_id')}|
                {'radius_m':2.2,'vx_mps':car['direction']*car['speed'],'vy_mps':0.,
                 'yaw_rad':0. if car['direction']>0 else math.pi,'acceleration_bound_mps2':3.})
        result.append({'id':'crossing_person','kind':'pedestrian','x_m':self.CROSSING_X,
            'y_m':self.pedestrian_y,'radius_m':.3,'vx_mps':0.,'vy_mps':self.pedestrian_velocity,
            'lane_id':'crosswalk','acceleration_bound_mps2':5.})
        return result

    def snapshot(self):
        return {'source':'synthetic_rule_based_traffic_not_social_validation','time_s':self.time_s,
            'signal':self.signal(),'actor_contact_events':self.actor_contact_events,'crosswalk':{'x_m':self.CROSSING_X,'x_min_m':30.5,'x_max_m':33.5,'y_min_m':-1.8,'y_max_m':5.4},
            'ego_stop_line_x_m':27.,'pedestrian_waiting_for_vehicle':self.waiting_for_vehicle,
            'pedestrian_state':'crossing' if abs(self.pedestrian_velocity)>.01 else 'waiting' if self.pedestrian_y>0 else 'arrived',
            'rules':['lane following','bounded acceleration and braking','same-lane following gap','red-signal stopping','pedestrian crossing clearance'],
            'limitations':['One small straight intersection fixture; no calibrated human behavior, route planning, lane changes or urban traffic validation.',
                            'Existing conservative circular safety review can stop for adjacent-lane traffic. It is not relaxed by this fixture.']}


def vehicle_clearance(ego,actor):
    """SAT separation bound for assumed rectangles; circle for a pedestrian."""
    c,s=math.cos(ego.yaw_rad),math.sin(ego.yaw_rad)
    centre=(ego.x_m+.7*c,ego.y_m+.7*s)
    dx,dy=actor['x_m']-centre[0],actor['y_m']-centre[1]
    if actor['kind']!='vehicle':
        local_x,local_y=c*dx+s*dy,-s*dx+c*dy
        return math.hypot(max(0.,abs(local_x)-1.7),max(0.,abs(local_y)-.9))-actor['radius_m']
    axes=((c,s),(-s,c),(1.,0.),(0.,1.))
    gaps=[]
    for ax,ay in axes:
        ego_extent=1.7*abs(c*ax+s*ay)+.9*abs(-s*ax+c*ay)
        other_extent=2.2*abs(ax)+.9*abs(ay)
        gaps.append(abs(dx*ax+dy*ay)-ego_extent-other_extent)
    return max(gaps)
