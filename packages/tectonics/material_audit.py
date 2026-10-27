"""同一材料轨迹的独立运动学审计，不修改pyGPlates标量或构造引擎。

空间差分/时间积分仅测量给定框架场；不是岩石圈流变求解器。
长度km，速度km/Myr，散度1/Myr，面积因子无量纲。
"""
from __future__ import annotations
import numpy as np
import pygplates as pg
from .geometry import R_KM, unit
from .material_band import point_array, spherical_triangle_area

SECONDS_PER_MYR=365.25*86400*1e6


def tangent_basis(points):
    x=np.asarray(points,float)
    e=np.cross(np.broadcast_to([0.,0.,1.],x.shape),x)
    if np.any(np.linalg.norm(e,axis=1)<1e-8):
        raise ValueError("当前局部审计不适用于极点")
    e=unit(e)
    return e,np.cross(x,e)


def small_material_patches(centers,radius_km):
    """同一中心的四方球面斑块；持久连接关系，不从变形后重新三角剖分。"""
    x=unit(centers)
    if not np.isfinite(radius_km) or not 0<radius_km<100:
        raise ValueError("局部斑块半径必须在0与100km之间")
    e,n=tangent_basis(x)
    ring=np.stack([unit(np.cos(radius_km/R_KM)*x+np.sin(radius_km/R_KM)*v)
                   for v in (e,n,-e,-n)],axis=1)
    points=np.r_[x,ring.reshape(-1,3)]
    tri=np.concatenate([np.column_stack([np.full(4,j),len(x)+4*j+np.arange(4),
                len(x)+4*j+np.roll(np.arange(4),-1)]) for j in range(len(x))])
    return points,tri


def velocity_divergence(snapshot,points,probe_km=.1,velocity_delta_time=.01,past=True):
    """球面切平面的中心差分；使用同一API速度，不把平滑应变当其导数。"""
    x=unit(points); e,n=tangent_basis(x)
    if probe_km<=0 or velocity_delta_time<=0:
        raise ValueError("差分距离和速度窗口须为正")
    probes=np.stack([unit(np.cos(probe_km/R_KM)*x+np.sin(probe_km/R_KM)*v)
                     for v in (e,-e,n,-n)],axis=1)
    velocities=snapshot.get_point_velocities([tuple(v) for v in probes.reshape(-1,3)],
        earth_radius_in_kms=R_KM,velocity_delta_time=velocity_delta_time,
        velocity_delta_time_type=pg.VelocityDeltaTimeType.t_plus_delta_t_to_t if past
          else pg.VelocityDeltaTimeType.t_to_t_minus_delta_t)
    if any(v is None for v in velocities):
        raise ValueError("差分斑块越出给定网络；不填补缺失速度")
    v=np.array([q.to_xyz() for q in velocities]).reshape(-1,4,3)
    return (np.sum((v[:,0]-v[:,1])*e,axis=1)+np.sum((v[:,2]-v[:,3])*n,axis=1))/(2*probe_km)


def cumulative_trapezoid(values,elapsed):
    values=np.asarray(values,float);elapsed=np.asarray(elapsed,float)
    if np.any(np.diff(elapsed)<=0):raise ValueError("积分时间须严格递增")
    return np.r_[np.zeros_like(values[:1]),np.cumsum((values[1:]+values[:-1])*np.diff(elapsed)[:,None]/2,axis=0)]


def audit_trajectories(model,centers,t0=35.,dt=.25,patch_km=1.,probe_km=.1,instant_window=.01):
    points,tri=small_material_patches(centers,patch_km)
    nc=len(centers)
    span=model.reconstruct_geometry([tuple(p) for p in points],t0,time_increment=dt,deactivate_points=None)
    elapsed=np.linspace(0,t0,round(t0/dt)+1);times=t0-elapsed
    a0=spherical_triangle_area(points,tri).reshape(-1,4).sum(1)
    arrays={k:[] for k in ("xyz","beta","J","strain_rate","past_divergence","instant_divergence")}
    for t in times:
        xyz=point_array(span.get_geometry_points(float(t)));x=xyz[:nc]
        snapshot=model.topological_snapshot(float(t))
        rates=snapshot.get_point_strain_rates([tuple(p) for p in x])
        if any(q is None for q in rates):raise ValueError("材料中心越出网络")
        arrays["xyz"].append(x)
        arrays["beta"].append(np.array(span.get_crustal_stretching_factors(float(t)))[:nc])
        arrays["J"].append(spherical_triangle_area(xyz,tri).reshape(-1,4).sum(1)/a0)
        arrays["strain_rate"].append(np.array([r.get_dilatation_rate() for r in rates])*SECONDS_PER_MYR)
        arrays["past_divergence"].append(velocity_divergence(snapshot,x,probe_km,1.,past=True))
        arrays["instant_divergence"].append(velocity_divergence(snapshot,x,probe_km,instant_window,past=t==0))
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    arrays.update(times_ma=times,elapsed_myr=elapsed)
    for key in ("strain_rate","past_divergence","instant_divergence"):
        arrays["integral_"+key]=cumulative_trapezoid(arrays[key],elapsed)
    logb,logj=np.log(arrays["beta"]),np.log(arrays["J"])
    sd=arrays["integral_strain_rate"];db=arrays["integral_past_divergence"];di=arrays["integral_instant_divergence"]
    arrays.update(error_total=logb-logj,error_scalar_integral=logb-sd,
        error_spatial_rate=sd-db,error_temporal_window=db-di,error_velocity_map=di-logj)
    arrays["decomposition_closure"]=arrays["error_total"]-sum(arrays[k] for k in (
        "error_scalar_integral","error_spatial_rate","error_temporal_window","error_velocity_map"))
    return arrays
