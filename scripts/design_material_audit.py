"""U36-S5：区分框架时间窗口/空间梯度误差，并按原生宿主裁切材料来源。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
import pygplates as pg
import shapely
from shapely.geometry import Point
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,omega_rotmat,unit
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.crust import CONT
from mythic.tectonics.sampling import native_cover
from mythic.tectonics.material_band import build_network,point_array,reference_strip,spherical_triangle_area
from mythic.tectonics.material_audit import audit_trajectories
from mythic.tectonics.source_clipping import clip_native_sources,triangulate_sources

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/u36_s5';DATA=ROOT/'data/worlds/design_v2'
ERROR_KEYS=['error_total','error_scalar_integral','error_spatial_rate','error_temporal_window','error_velocity_map']


def write_json(path,report):
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
        if isinstance(value,np.integer):return int(value)
        return value
    path.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def compact(arrays):
    out=dict(sample_count=arrays['J'].shape[1],time_count=len(arrays['times_ma']),
        decomposition_closure_max=float(np.max(abs(arrays['decomposition_closure']))),timeline=[])
    for time in [35.,34.,33.,30.,20.,10.,0.]:
        j=int(np.argmin(abs(arrays['times_ma']-time)))
        row=dict(t_ma=float(arrays['times_ma'][j]),J_quantiles=np.quantile(arrays['J'][j],[0,.1,.5,.9,1]).tolist())
        for k in ERROR_KEYS:row[k]=np.quantile(arrays[k][j],[0,.1,.5,.9,1]).tolist()
        out['timeline'].append(row)
    return out


def rotation_control(case):
    h=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        keys=[]
        for t in range(51):
            if case=='constant':amount=t
            elif case=='step':amount=min(t,35)
            else:amount=t if t<=33 else (34 if t>=35 else 33+(35*(t-33)-(t*t-33*33)/2)/2)
            keys.append((t,omega_rotmat(np.array([0,0,sign*5/R_KM]),-amount)))
        h.add(plate,0,f'control_{plate}',keys)
    return pg.RotationModel(str(h.write_rot(CACHE/(case+'.rot'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--part',choices=['all','trajectory','sources'],default='all');args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    a=np.load(ROOT/'.cache/u36_s4/material_band.npz')
    rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s4/material_network.gpml'),rot)
    centers=unit(a['xyz'][a['triangles']].sum(1))
    if args.part in ['all','trajectory']:
        report=dict(checkpoint='U36-S5',part='trajectory',framework_version=pg.__version__,
            version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S5',
            physical_engine_changed=False,method_class='procedural_constrained',framework_field_class='interpolated',
            decomposition='log(beta/J)=[logbeta-intD]+[intD-intDivPast]+[intDivPast-intDivInstant]+[intDivInstant-logJ]',
            units={'rate':'1/Myr','error':'无量纲对数差','length':'km','time':'Myr'},controls=[],sensitivity=[])
        arrays=audit_trajectories(model,centers,dt=.25)
        np.savez_compressed(CACHE/'trajectory.npz',**arrays);report['real']=compact(arrays)
        known=a['known'];weights=a['area0_km2']
        v0=float(np.sum(weights[known]*a['H0_km'][known]))
        bj=np.exp(arrays['integral_instant_divergence'][-1])
        report['window_matched_diagnostic']=dict(
            original_cell_volume_by_framework_relative_residual=float(np.sum(weights[known]*a['timeline_J'][-1][known]*a['H0_km'][known]/arrays['beta'][-1][known])/v0-1),
            original_cell_volume_by_short_window_divergence_relative_residual=float(np.sum(weights[known]*a['timeline_J'][-1][known]*a['H0_km'][known]/bj[known])/v0-1),
            weighted_mean_log_components={k:float(np.average(arrays[k][-1],weights=weights)) for k in ERROR_KEYS},
            note='同一旧材料单元上的诊断性比较；未替换框架beta、未接入全球厚度或B5')
        subset=centers[::16]
        for dt in [.5,.25,.125]:
            for radius in [1.,.1]:
                r=audit_trajectories(model,subset,dt=dt,patch_km=radius)
                report['sensitivity'].append(dict(dt_myr=dt,patch_km=radius,summary=compact(r)))
        # 额外半径0.01km只作有限面积差分精度核对，不叫物理网格细化。
        for probe in [1.,.1,.01]:
            r=audit_trajectories(model,subset,dt=.25,patch_km=1.,probe_km=probe)
            report.setdefault('spatial_difference',[]).append(dict(probe_km=probe,summary=compact(r)))
        control_strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60)
        control_centers=unit(control_strip['xyz'][control_strip['triangles']].sum(1))[::8]
        for case in ['constant','step','ramp']:
            m,_=build_network(control_strip,rotation_control(case),35,plates=(1,2),tag=case)
            r=audit_trajectories(m,control_centers,dt=.125)
            np.savez_compressed(CACHE/f'control_{case}.npz',**r)
            report['controls'].append(dict(case=case,rotation_coverage_ma=[0,50],rotation_key_spacing_myr=1,
                motion_note='ramp为35→34Ma的25%速率、34→33Ma的75%速率、33Ma以后满速；是两级启动，不是连续加速度求解' if case=='ramp' else '显式常速或35Ma启动的分段旋转',summary=compact(r)))
        report['elapsed_s']=time.perf_counter()-start
        write_json(DATA/'u36_s5_trajectory.json',report)
        print('trajectory',report['elapsed_s'],report['real']['timeline'][-1],flush=True)
    if args.part in ['all','sources']:
        begin=time.perf_counter()
        with (ROOT/'.cache/u36_s4/control/reference_35Ma.pkl').open('rb') as f:eng=pickle.load(f)
        boundary=a['xyz'][a['boundary']]
        pieces,projection,summary=clip_native_sources(eng,boundary)
        _,_,broad=clip_native_sources(eng,boundary,search_padding_km=360)
        summary['broad_search_known_volume_difference_km3']=broad['known_cont_volume_km3']-summary['known_cont_volume_km3']
        v,tri,source,degenerate=triangulate_sources(pieces,projection,max_edge_km=15.,return_diagnostics=True)
        # 以独立密集查询检查布尔优先级与现有native_cover相同。查询不修改裁切结果。
        rng=np.random.default_rng(3605)
        bounds=np.array([projection.forward(boundary).min(0),projection.forward(boundary).max(0)])
        queries=rng.uniform(bounds[0],bounds[1],(20000,2))
        target=shapely.Polygon(projection.forward(boundary));inside=shapely.contains_xy(target,queries[:,0],queries[:,1])
        queries=queries[inside];native=native_cover(eng,projection.inverse(queries))
        expected=np.where(native>=0,eng.c.uid[np.maximum(native,0)],-1)
        actual=np.full(len(queries),-2,np.int64)
        for piece in pieces:
            inside=shapely.contains_xy(piece['geometry'],queries[:,0],queries[:,1]);actual[inside]=piece['uid']
        summary.update(independent_query_count=len(queries),native_uid_mismatch_count=int(np.sum(expected!=actual)))
        if summary['native_uid_mismatch_count']:raise RuntimeError('裁切优先级与原生查询不一致')
        # 只追踪已裁切的参考材料，未知厚度不补值。所有面积仍为球面几何。
        centers2=unit(v[tri].sum(1));span=model.reconstruct_geometry([tuple(p) for p in np.r_[v,centers2]],35,time_increment=.25,deactivate_points=None)
        ar0=abs(spherical_triangle_area(v,tri));h0=np.array([pieces[i]['T_km'] if pieces[i]['ctype']==CONT else np.nan for i in source])
        known=np.isfinite(h0);initial=float(np.sum(ar0[known]*h0[known]));timeline=[];out={}
        for t in [35.,30.,20.,10.,0.]:
            xyz=point_array(span.get_geometry_points(t));j=spherical_triangle_area(xyz[:len(v)],tri)/spherical_triangle_area(v,tri)
            beta=np.asarray(span.get_crustal_stretching_factors(t))[len(v):]
            area=ar0*j
            locations=model.topological_snapshot(t).get_point_locations([tuple(p) for p in xyz[len(v):]])
            timeline.append(dict(t_ma=t,known_volume_by_J_km3=float(np.sum(area[known]*h0[known]/j[known])),
                known_volume_by_beta_km3=float(np.sum(area[known]*h0[known]/beta[known])),
                outside_center_count=sum(p.not_located_in_resolved_topology() for p in locations),
                inverted_cells=int(np.sum(j<=0)),H_final_quantiles_km=np.quantile(h0[known]/j[known],[0,.1,.5,.9,1]).tolist()))
            out[f'xyz_{int(t)}Ma']=xyz[:len(v)];out[f'J_{int(t)}Ma']=j
        np.savez_compressed(CACHE/'clipped_material.npz',vertices=v,triangles=tri,source_index=source,H0_km=h0,
            source_uid=np.array([pieces[i]['uid'] for i in source]),source_plate=np.array([pieces[i]['plate'] for i in source]),**out)
        with (CACHE/'clipped_sources.pkl').open('wb') as f:pickle.dump(dict(pieces=pieces,projection=projection,summary=summary),f)
        report=dict(checkpoint='U36-S5',part='sources',method_class='procedural_constrained',input_geometry_class='user_override',
            projection='local_gnomonic；平面只做布尔运算，全部km²回球面计算',summary=summary,
            vertex_count=len(v),triangle_count=len(tri),numerical_geometry=degenerate,triangle_initial_area_km2=float(ar0.sum()),
            triangle_known_volume_km3=initial,clip_vs_triangle_known_volume_difference_km3=initial-summary['known_cont_volume_km3'],
            original_center_quadrature_known_volume_km3=float(np.sum(a['area0_km2'][a['known']]*a['H0_km'][a['known']])),
            timeline=timeline,source_hosts=[{k:v for k,v in p.items() if k!='geometry'} for p in pieces],elapsed_s=time.perf_counter()-begin,
            limitations=['给定的参考线/带宽仍未获得独立物理约束','裁切不补空隙，也不改变原生宿主与全球引擎',
                'H0/J仍为定义性材料体积账；真实减薄仍需速度/面积及成因证据','不是B5地形、高程、水位或沉降求解'])
        write_json(DATA/'u36_s5_sources.json',report)
        print('sources',report['elapsed_s'],summary,flush=True)


if __name__=='__main__':main()
