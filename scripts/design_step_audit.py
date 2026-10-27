"""U36-S6：短窗口小角度截断、事件对齐求积与局部柱体连续性。"""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import pygplates as pg
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,omega_rotmat,unit
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.material_band import build_network,point_array,reference_strip,spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.step_audit import midpoint_audit,api_velocity,map_velocity,network_map
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s6';DATA=ROOT/'data/worlds/design_v2'


def control_rotation(case):
    h=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        keys=[]
        for t in range(51):
            if case=='constant':amount=t
            elif case in ['step','turning','rigid']:amount=min(t,35)
            else:amount=t if t<=33 else (34 if t>=35 else 33+(35*(t-33)-(t*t-33*33)/2)/2)
            s=1 if case=='rigid' else sign
            r=omega_rotmat(np.array([0,0,s*5/R_KM]),-amount)
            if case=='turning':r=omega_rotmat(np.array([0,sign*2/R_KM,0]),-max(min(t,25)-10,0))@r
            keys.append((t,r))
        h.add(plate,0,f'{case}_{plate}',keys)
    return pg.RotationModel(str(h.write_rot(CACHE/(case+'.rot'))))


def summarize(a,weights=None):
    out=dict(sample_count=a['J'].shape[1],time_count=len(a['times_ma']),timeline=[])
    for t in [35.,34.,33.,30.,20.,10.,0.]:
        i=int(np.argmin(abs(a['times_ma']-t)));row=dict(t_ma=float(a['times_ma'][i]),J_quantiles=np.quantile(a['J'][i],[0,.1,.5,.9,1]).tolist())
        for k in ['api_short','api_matched','map_short']:
            val=a[k+'_error'][i];row[k+'_error_quantiles']=np.quantile(val,[0,.1,.5,.9,1]).tolist()
            row[k+'_error_weighted_mean']=float(np.average(val,weights=weights))
        out['timeline'].append(row)
    out['zero_center_short_api_count']=int(a['short_zero_count'].sum())
    out['minimum_J']=float(np.min(a['J']))
    return out


def trajectory():
    start=time.perf_counter();old=np.load(ROOT/'.cache/u36_s4/material_band.npz')
    rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s4/material_network.gpml'),rot)
    centers=unit(old['xyz'][old['triangles']].sum(1))
    report=dict(checkpoint='U36-S6',framework_version=pg.__version__,method_class='procedural_constrained',
        velocity_field_class='interpolated',reference_geometry_class='user_override',physical_engine_changed=False,
        version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S6',
        units={'velocity':'km/Myr','angle':'rad','error':'无量纲对数差','time':'Myr'},
        assumptions=['给定200km参考带；不推断物理带宽','旋转1Myr关键帧；积分步长整除1Myr','不把点映射微分称为流变求解器'])
    angles=[]
    for angle in np.geomspace(1e-9,1e-3,49):
        rotation=pg.FiniteRotation((90,0),float(angle));p=pg.PointOnSphere((0,0));q=rotation*p
        v=np.linalg.norm(pg.calculate_velocities([p],rotation,1.,earth_radius_in_kms=R_KM)[0].to_xyz())
        x,y=np.array(p.to_xyz()),np.array(q.to_xyz())
        angles.append(dict(angle_rad=float(angle),identity=rotation.represents_identity_rotation(),api_speed_km_myr=float(v),
            map_speed_km_myr=float(R_KM*np.arctan2(np.linalg.norm(np.cross(x,y)),x@y)),analytic_speed_km_myr=float(R_KM*angle)))
    report['angle_sweep']=angles
    a=midpoint_audit(model,centers,dt=.25);np.savez_compressed(CACHE/'real.npz',**a);report['real']=summarize(a,old['area0_km2'])
    strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60)
    x=unit(strip['xyz'][strip['triangles']].sum(1))[::8];report['controls']=[]
    for case in ['constant','step','ramp','turning','rigid']:
        m,_=build_network(strip,control_rotation(case),35,plates=(1,2),tag=case)
        for dt in [.25,.125,.0625]:
            r=midpoint_audit(m,x,dt=dt)
            np.savez_compressed(CACHE/f'{case}_{dt}.npz',**r)
            report['controls'].append(dict(case=case,dt_myr=dt,summary=summarize(r)))
            print(case,dt,report['controls'][-1]['summary']['timeline'][-1],flush=True)
    # 时间窗口独立扫描：同一35→34Ma慢速段中点，点映射给出非零速度。
    m,_=build_network(strip,control_rotation('ramp'),35,plates=(1,2),tag='window')
    s=m.reconstruct_geometry([tuple(p) for p in x],35,time_increment=.25,deactivate_points=None)
    t=34.5;q=point_array(s.get_geometry_points(t));snap=m.topological_snapshot(t);report['window_sweep']=[]
    for w in [1.,.5,.25,.1,.05,.025,.01,.005,.001]:
        api=api_velocity(snap,q,w,centered=True);mv=map_velocity(snap,q,w,centered=True)
        report['window_sweep'].append(dict(window_myr=w,zero_api_count=int(np.sum(np.linalg.norm(api,axis=1)==0)),count=len(q),
            api_median_km_myr=float(np.median(np.linalg.norm(api,axis=1))),map_median_km_myr=float(np.median(np.linalg.norm(mv,axis=1))),
            max_vector_difference_km_myr=float(np.max(np.linalg.norm(api-mv,axis=1)))))
    # 明确核查单步reconstruct_point与span，误差按实际坐标距离统计。
    report['one_step_parity']=[]
    full=m.reconstruct_geometry([tuple(p) for p in x],35,time_increment=.125,deactivate_points=None)
    for t in [35.,34.5,34.,33.,20.]:
        q=point_array(full.get_geometry_points(t));mapped=network_map(m.topological_snapshot(t),q,t-.125)
        target=point_array(full.get_geometry_points(t-.125))
        report['one_step_parity'].append(dict(t_ma=t,dt_myr=.125,max_chord_error_km=float(np.max(np.linalg.norm(mapped-target,axis=1))*R_KM)))
    # 相同刚体输入的独立精确旋转基准；一致性不是物理正确性的替代。
    rrigid=control_rotation('rigid');patches,ptri=small_material_patches(x,1.)
    a0=spherical_triangle_area(patches,ptri).reshape(-1,4).sum(1)
    exact=point_array([rrigid.get_rotation(0.,1,35.)*pg.PointOnSphere(tuple(p)) for p in patches])
    exactj=spherical_triangle_area(exact,ptri).reshape(-1,4).sum(1)/a0
    frame=np.load(CACHE/'rigid_0.125.npz')['J'][-1]
    report['rigid_exact_rotation_comparison']=dict(exact_J_quantiles=np.quantile(exactj,[0,.1,.5,.9,1]).tolist(),
        network_J_quantiles=np.quantile(frame,[0,.1,.5,.9,1]).tolist(),
        maximum_network_area_relative_error=float(np.max(abs(frame-exactj))),
        note='网络点映射在共同刚体输入下仍不严格保持面积；同映射散度与面积吻合不能自动证明其刚体客观性')
    report['elapsed_s']=time.perf_counter()-start;write_json(DATA/'u36_s6_trajectory.json',report)
    print('trajectory seconds',report['elapsed_s'],'real',report['real']['timeline'][-1],flush=True)


def sources(edge_km=15.):
    start=time.perf_counter();a=np.load(ROOT/'.cache/u36_s5/clipped_material.npz')
    v,tri,h0=a['vertices'],a['triangles'],a['H0_km'];source_uid=a['source_uid'];suffix=''
    if edge_km!=15.:
        import pickle
        from mythic.tectonics.source_clipping import triangulate_sources
        from mythic.tectonics.crust import CONT
        with (ROOT/'.cache/u36_s5/clipped_sources.pkl').open('rb') as f:clipped=pickle.load(f)
        pieces=clipped['pieces'];v,tri,source,fragment=triangulate_sources(pieces,clipped['projection'],max_edge_km=edge_km,return_diagnostics=True)
        h0=np.array([pieces[i]['T_km'] if pieces[i]['ctype']==CONT else np.nan for i in source])
        source_uid=np.array([pieces[i]['uid'] for i in source]);suffix='_refined'
    centers=unit(v[tri].sum(1));known=np.isfinite(h0)
    rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s4/material_network.gpml'),rot)
    # 固定源三角形与独立1km中心小斑块同时由同一框架推进；柱厚来自局部Jacobian而不是该源三角形面积。
    patches,patch_tri=small_material_patches(centers,1.);points=np.r_[v,patches]
    span=model.reconstruct_geometry([tuple(p) for p in points],35,time_increment=.25,deactivate_points=None)
    signed0=spherical_triangle_area(v,tri);area0=abs(signed0);pa0=spherical_triangle_area(patches,patch_tri).reshape(-1,4).sum(1)
    volume0=float(np.sum(area0[known]*h0[known]));timeline=[];saved={};raw=[];ever_outside=np.zeros(len(tri),bool);source_uids=np.unique(source_uid[known])
    for t in 35-np.arange(141)*.25:
        xyz=point_array(span.get_geometry_points(float(t)));vv=xyz[:len(v)];pp=xyz[len(v):]
        signed=spherical_triangle_area(vv,tri);area=abs(signed);j=signed/signed0
        jac=spherical_triangle_area(pp,patch_tri).reshape(-1,4).sum(1)/pa0
        beta=np.asarray(span.get_crustal_stretching_factors(float(t)))[len(v):len(v)+len(centers)]
        locations=model.topological_snapshot(float(t)).get_point_locations([tuple(p) for p in pp])
        outloc=np.array([p.not_located_in_resolved_topology() for p in locations])
        outside=outloc[:len(centers)]|outloc[len(centers):].reshape(-1,4).any(1);ever_outside|=outside
        h=h0/jac
        # 源UID维持独立；不跨来源调平、不归一化，不补缺值。
        raw.append(dict(t_ma=float(t),area=area,jac=jac,beta=beta,h=h,outside=outside,inverted_triangles=int(np.sum(j<=0)),inverted_patches=int(np.sum(jac<=0))))
        if t in [35.,30.,20.,10.,0.]:saved.update({f'xyz_{int(t)}Ma':vv,f'H_{int(t)}Ma':h,f'jac_{int(t)}Ma':jac})
    # 先检查全部141时点，固定全时段有效子集；无效已知材料保留独立初始库存，不报告伪守恒。
    supported=known&~ever_outside;v_supported=float(np.sum(area0[supported]*h0[supported]))
    for q in raw:
        row=dict(t_ma=q['t_ma'],volume_km3=float(np.sum(q['area'][supported]*q['h'][supported])),
            volume_relative_residual=float(np.sum(q['area'][supported]*q['h'][supported])/v_supported-1),
            unfiltered_volume_relative_residual=float(np.sum(q['area'][known]*q['h'][known])/volume0-1),
            framework_volume_relative_residual=float(np.sum(q['area'][supported]*h0[supported]/q['beta'][supported])/v_supported-1),
            inverted_triangles=q['inverted_triangles'],inverted_patches=q['inverted_patches'],outside_patch_count=int(q['outside'].sum()),
            thickness_quantiles_km=np.quantile(q['h'][supported],[0,.1,.5,.9,1]).tolist(),source_volumes=[])
        for uid in source_uids:
            mask=supported&(source_uid==uid);initial=float(np.sum(area0[mask]*h0[mask]))
            if initial>0:row['source_volumes'].append(dict(uid=int(uid),initial_volume_km3=initial,volume_km3=float(np.sum(q['area'][mask]*q['h'][mask]))))
        timeline.append(row)
    for t in [35,30,20,10,0]:saved[f'H_{t}Ma'][~supported]=np.nan
    np.savez_compressed(CACHE/f'local_material{suffix}.npz',vertices=v,triangles=tri,H0_km=h0,source_uid=source_uid,supported=supported,**saved)
    report=dict(checkpoint='U36-S6',method_class='procedural_constrained',reference_geometry_class='user_override',
        integration='pyGPlates固定源三角形与独立1km中心斑块；H=H0/Jac_center，未按整格面积或总量归一化',
        units={'area':'km²','thickness':'km','volume':'km³','time':'Ma'},dt_myr=.25,initial_volume_km3=volume0,
        maximum_edge_km=edge_km,framework_version=pg.__version__,triangle_count=len(tri),known_triangle_count=int(known.sum()),supported_triangle_count=int(supported.sum()),
        supported_initial_volume_km3=v_supported,unsupported_known_initial_volume_km3=volume0-v_supported,
        unsupported_known_initial_area_km2=float(area0[known&ever_outside].sum()),ever_outside_patch_count=int(ever_outside.sum()),
        boundary_policy='全部141时点中心/四环点均在网络内才可评估；其余已知初始材料保留库存，厚度设缺值；未补料、未总量归一化',source_uid_count=len(source_uids),timeline=timeline,
        source_area_fractions=json.loads((DATA/'u36_s5_sources.json').read_text(encoding='utf-8'))['summary'],
        limitations=['柱体连续性是给定运动下的运动学假设，不是独立流变/热/高程求解','独立小斑块与有限源三角形仍有空间求积误差',
            '源UID不交换物质；不包含真实侧向流动、侵蚀沉积或新增洋壳','物理带宽及古接触尚待约束；非陆壳/无覆盖仍不补陆壳'],
        elapsed_s=time.perf_counter()-start)
    write_json(DATA/f'u36_s6_sources{suffix}.json',report);print('sources seconds',report['elapsed_s'],'final residual',timeline[-1]['volume_relative_residual'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['all','trajectory','sources'],default='all');p.add_argument('--source-edge-km',type=float,default=15.);args=p.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    if args.part in ['all','trajectory']:trajectory()
    if args.part in ['all','sources']:sources(args.source_edge_km)
if __name__=='__main__':main()
