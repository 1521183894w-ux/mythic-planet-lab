"""有限点映射与速度接口的一致性审计；仅测量既有pyGPlates网络。

method_class=procedural_constrained；给定运动场=interpolated。
不修改框架标量，不求流变或盆地高程；长度km、时间Myr。
"""
from __future__ import annotations
import numpy as np
import pygplates as pg
from .geometry import R_KM,unit
from .material_band import point_array,spherical_triangle_area
from .material_audit import tangent_basis,small_material_patches


def sphere_log_velocity(points,mapped,elapsed_myr):
    """从实际点映射求切向位移率，atan2保留小角度；非新的运动求解器。"""
    x,y=unit(points),unit(mapped)
    if not np.isfinite(elapsed_myr) or elapsed_myr==0:
        raise ValueError('映射时间须有限且非零')
    dot=np.sum(x*y,axis=-1)
    cross=np.cross(x,y);sine=np.linalg.norm(cross,axis=-1)
    if np.any((dot<0)&(sine<1e-10)):
        raise ValueError('对跖映射没有唯一局部对数')
    tangent=np.cross(cross,x)
    angle=np.arctan2(sine,dot)
    factor=np.divide(R_KM*angle,sine*elapsed_myr,out=np.zeros_like(angle),where=sine>0)
    return tangent*factor[...,None]


def network_map(snapshot,points,target_time):
    """使用框架同一步点映射；网外拒绝，不以零速度或最近宿主补值。"""
    nets=snapshot.get_resolved_topologies(resolve_topology_types=pg.ResolveTopologyType.network)
    if len(nets)!=1:raise ValueError('本地审计要求唯一网络')
    result=[nets[0].reconstruct_point(tuple(p),float(target_time)) for p in np.asarray(points)]
    if any(p is None for p in result):raise ValueError('映射点位于网络之外')
    return point_array(result)


def map_velocity(snapshot,points,window_myr,centered=False):
    t=snapshot.get_reconstruction_time()
    if window_myr<=0:raise ValueError('时间窗口必须为正')
    if centered:
        younger=network_map(snapshot,points,t-window_myr/2)
        older=network_map(snapshot,points,t+window_myr/2)
        return (sphere_log_velocity(points,younger,window_myr/2)+sphere_log_velocity(points,older,-window_myr/2))/2
    return sphere_log_velocity(points,network_map(snapshot,points,t-window_myr),window_myr)


def surface_divergence(field,points,probe_km=.1):
    if probe_km<=0:raise ValueError('空间探针须为正')
    x=unit(points);e,n=tangent_basis(x)
    probes=np.stack([unit(np.cos(probe_km/R_KM)*x+np.sin(probe_km/R_KM)*v) for v in (e,-e,n,-n)],axis=1)
    vel=np.asarray(field(probes.reshape(-1,3))).reshape(-1,4,3)
    return (np.sum((vel[:,0]-vel[:,1])*e,axis=1)+np.sum((vel[:,2]-vel[:,3])*n,axis=1))/(2*probe_km)


def api_velocity(snapshot,points,window_myr,centered=False):
    v=snapshot.get_point_velocities([tuple(p) for p in points],earth_radius_in_kms=R_KM,
        velocity_delta_time=window_myr,velocity_delta_time_type=pg.VelocityDeltaTimeType.t_plus_minus_half_delta_t if centered else pg.VelocityDeltaTimeType.t_to_t_minus_delta_t)
    if any(p is None for p in v):raise ValueError('速度探针位于网络之外')
    return point_array(v)


def midpoint_audit(model,centers,initial_time=35.,dt=.25,patch_km=1.,probe_km=.1,short_window=.01):
    """事件对齐的中点求积；点运动仍由框架完成，不使用速度重建轨迹。

    假定1Myr关键帧且dt整除1Myr，匹配窗口=dt完全位于对应时间段。
    短窗口/API匹配窗口/映射差分都用同一材料中点与权重。
    """
    if dt<=0 or not np.isclose(round(1/dt)*dt,1):raise ValueError('dt须整除1Myr关键帧')
    if not np.isclose(round(initial_time/dt)*dt,initial_time):raise ValueError('时长须整除dt')
    points,tri=small_material_patches(centers,patch_km);nc=len(centers)
    span=model.reconstruct_geometry([tuple(p) for p in points],initial_time,time_increment=dt/2,deactivate_points=None)
    times=initial_time-np.arange(round(initial_time/dt)+1)*dt
    area0=spherical_triangle_area(points,tri).reshape(-1,4).sum(1)
    j=[];beta=[];xyz=[];rates={k:[] for k in ['api_short','api_matched','map_short']};zeros=[]
    for i,t in enumerate(times):
        q=point_array(span.get_geometry_points(float(t)))
        j.append(spherical_triangle_area(q,tri).reshape(-1,4).sum(1)/area0)
        beta.append(np.asarray(span.get_crustal_stretching_factors(float(t)))[:nc]);xyz.append(q[:nc])
        if i==len(times)-1:continue
        mid=t-dt/2;snap=model.topological_snapshot(float(mid))
        x=point_array(span.get_geometry_points(float(mid)))[:nc]
        funcs={'api_short':lambda z:api_velocity(snap,z,short_window,centered=True),
            'api_matched':lambda z:api_velocity(snap,z,dt,centered=True),
            'map_short':lambda z:map_velocity(snap,z,short_window,centered=True)}
        for k,f in funcs.items():rates[k].append(surface_divergence(f,x,probe_km))
        zeros.append(np.count_nonzero(np.linalg.norm(api_velocity(snap,x,short_window,centered=True),axis=1)==0))
    out=dict(times_ma=times,J=np.asarray(j),beta=np.asarray(beta),xyz=np.asarray(xyz),short_zero_count=np.asarray(zeros))
    for k,r in rates.items():
        out[k+'_rate']=np.asarray(r)
        out[k+'_integral']=np.r_[np.zeros((1,nc)),np.cumsum(np.asarray(r)*dt,axis=0)]
        out[k+'_error']=out[k+'_integral']-np.log(out['J'])
    return out
