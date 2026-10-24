"""Offline physical-consistency checks for proposed x/y/speed trajectories."""
import numpy as np


def assess_trajectory(points, times_s, *, max_speed_mps, max_accel_mps2,
                      max_decel_mps2, max_curvature_per_m, max_lateral_accel_mps2):
    limits=np.array([max_speed_mps,max_accel_mps2,max_decel_mps2,max_curvature_per_m,max_lateral_accel_mps2],dtype=float)
    if not np.isfinite(limits).all() or (limits<=0).any():
        raise ValueError("Positive finite vehicle limits required")
    p,t=np.asarray(points,dtype=float),np.asarray(times_s,dtype=float)
    result={"feasible":False,"reasons":[],"physical_output":False}
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or t.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(t).all() or (np.diff(t)<=0).any() or (t<0).any():
        result["reasons"]=["invalid_timed_trajectory"]
        return result
    speed=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)/np.diff(t)
    if (p[:,2]<0).any() or (p[:,2]>max_speed_mps).any() or (speed>max_speed_mps).any():
        result["reasons"].append("speed_limit")
    # Test both coordinate-implied and explicitly requested acceleration.
    for velocity,stamp in ((p[:,2],t),(speed,(t[1:]+t[:-1])/2)):
        acceleration=np.diff(velocity)/np.diff(stamp)
        if (acceleration>max_accel_mps2).any() or (acceleration < -max_decel_mps2).any():
            result["reasons"].append("acceleration_limit")
    a,b=np.diff(p[:,:2],axis=0)[:-1],np.diff(p[:,:2],axis=0)[1:]
    chord=np.linalg.norm(a+b,axis=1)
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*chord
    cross=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    stationary=(np.linalg.norm(a,axis=1)<1e-8)&(np.linalg.norm(b,axis=1)<1e-8)
    if ((denom<1e-10)&~stationary).any():
        result["reasons"].append("undefined_turn_or_reversal")
    curvature=np.divide(2*np.abs(cross),denom,out=np.zeros_like(denom),where=denom>1e-10)
    if (curvature>max_curvature_per_m).any():
        result["reasons"].append("curvature_limit")
    upper_speed=np.maximum.reduce([speed[:-1],speed[1:],p[1:-1,2]])
    if (curvature*upper_speed**2>max_lateral_accel_mps2).any():
        result["reasons"].append("lateral_acceleration_limit")
    expected=(p[:-1,2]+p[1:,2])/2
    if (np.abs(speed-expected)>np.maximum(.15,.3*expected)).any():
        result["reasons"].append("position_speed_inconsistent")
    result["reasons"]=sorted(set(result["reasons"]))
    result["feasible"]=not result["reasons"]
    return result
