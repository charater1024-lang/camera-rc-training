"""Display-only synthetic object observations; never a learned detector."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw
from . import rendering

SOURCE = "synthetic_geometry_not_learned_detection"


def projected_box(forward, left, radius, kind):
    """Clipped proxy for the renderer's actor extent, not a detected pixel mask."""
    depth = forward-rendering.CAMERA_X_M
    if depth <= .1 or math.hypot(depth,left)-radius > rendering.MAP_RENDER_RANGE_M:
        return None
    scale = rendering.FOCAL/depth
    u = rendering.WIDTH/2.-scale*left
    bottom = rendering.HORIZON+scale*rendering.CAMERA_HEIGHT_M
    height = 1.7 if kind == 'pedestrian' else .75 if kind == 'cone' else 1.35
    half = max(1., radius*scale)
    bounds = [u-half, bottom-height*scale, u+half, bottom]
    if bounds[2] < 0 or bounds[0] >= rendering.WIDTH or bounds[3] < 0 or bounds[1] >= rendering.HEIGHT:
        return None
    return [max(0.,bounds[0]),max(0.,bounds[1]),min(rendering.WIDTH-1.,bounds[2]),min(rendering.HEIGHT-1.,bounds[3])]


def observation(obstacles, pose, range_m, visible_ids, time_s, route):
    x,y,yaw=pose
    c,s=math.cos(yaw),math.sin(yaw)
    route=np.asarray(route,dtype=float)
    starts=route[:-1]; vectors=np.diff(route,axis=0)
    squared=np.sum(vectors*vectors,axis=1)
    objects=[]
    for actor in obstacles:
        dx,dy=actor['x_m']-x,actor['y_m']-y
        distance=math.hypot(dx,dy)
        if distance > range_m or (visible_ids is not None and actor['id'] not in visible_ids):
            continue
        forward,left=c*dx+s*dy,-s*dx+c*dy
        offset=np.array([actor['x_m'],actor['y_m']])-starts
        fraction=np.clip(np.sum(offset*vectors,axis=1)/np.maximum(squared,1e-12),0.,1.)
        route_distance=float(np.min(np.linalg.norm(offset-fraction[:,None]*vectors,axis=1)))
        kind=actor.get('kind','obstacle')
        objects.append({**actor,'distance_m':distance,'relative_forward_m':forward,'relative_left_m':left,
            'speed_kmh':3.6*math.hypot(actor['vx_mps'],actor['vy_mps']),
            'confidence':None,'track_identity_source':'synthetic_actor_id',
            'near_reference_route':forward>=0 and route_distance<=2.7+actor['radius_m'],
            'camera_box_xyxy':projected_box(forward,left,actor['radius_m'],kind)})
    objects.sort(key=lambda item:(item['distance_m'],item['id']))
    return {'source':SOURCE,'learned_detector_loaded':False,'status':'available',
        'sampled_at_sim_time_s':time_s,'sensor_range_m':range_m,
        'objects':objects,'world_actor_count':len(obstacles),
        'hidden_or_out_of_range_count':len(obstacles)-len(objects),
        'camera_box_count':sum(item['camera_box_xyxy'] is not None for item in objects),
        'safety_track_ids':[],
        'box_definition':'projected assumed actor extent; not detected pixels or ground-truth instance masks',
        'route_proximity_definition':'actor center near reference polyline; not a collision prediction or safety decision'}


def overlay(rgb, view):
    image=Image.fromarray(np.asarray(rgb,dtype=np.uint8).copy())
    drawing=ImageDraw.Draw(image)
    actors=[actor for actor in view['objects'] if actor['camera_box_xyxy'] is not None]
    for actor in reversed(actors):
        color='#e89a38' if actor['near_reference_route'] else '#27c2b1'
        drawing.rectangle(actor['camera_box_xyxy'],outline=color,width=1)
    occupied=[]
    for actor in actors:
        box=actor['camera_box_xyxy']
        color='#e89a38' if actor['near_reference_route'] else '#27c2b1'
        kind={'vehicle':'CAR','pedestrian':'PERSON','cone':'CONE'}.get(actor['kind'],'OBJECT')
        label=f"{kind} {actor['distance_m']:.1f}m"
        bounds=drawing.textbbox((0,0),label)
        width,height=bounds[2]-bounds[0]+4,bounds[3]-bounds[1]+4
        left=min(max(0.,box[0]),max(0.,rendering.WIDTH-width))
        top=max(0.,box[1]-height-2)
        position=None
        candidates=[top]+list(range(0,max(1,rendering.HEIGHT-height),height+3))
        for candidate in candidates:
            rect=(left,float(candidate),left+width,float(candidate)+height)
            if not any(rect[0]<other[2]+2 and rect[2]+2>other[0] and rect[1]<other[3]+2 and rect[3]+2>other[1] for other in occupied):
                position=rect
                break
        if position is None:
            # Keep boxes when the small camera frame has no clear label space.
            continue
        occupied.append(position)
        left,top,right,bottom=position
        drawing.line(((left+right)/2.,bottom,(box[0]+box[2])/2.,box[1]),fill=color,width=1)
        drawing.rectangle(position,fill='#16313b')
        drawing.text((left+2,top+2-bounds[1]),label,fill=color)
    return image
