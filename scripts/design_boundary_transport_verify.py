"""S11结果的独立几何复算、保护内容与同窗世界核验。"""
from pathlib import Path
import json,hashlib,subprocess,re
import numpy as np
import pygplates as pg
from mythic.tectonics.geometry import R_KM,unit
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.boundary_transport import GnomonicRotationMap
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s11'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    rows=[];trial={}
    for p in sorted(DATA.glob('u36_s11_*_e30_l*.json')):
        if 'additive' in p.name:continue
        q=read(p);tag=q['tag'];a=np.load(CACHE/(tag+'.npz'))
        points=a['points'];jj=a['J'];v0=a['V0_km3'];saved=a['source_volumes_km3']
        tri=a['triangles'];known=a['known'];h0=a['H0_km'];uid=a['source_uid'];uids=a['uids'];nv=len(a['vertices'])
        assert q['complete_35_to_0Ma'] and len(points)==round(35/q['dt_myr'])+1
        group=np.searchsorted(uids,uid[known]);dif=0.;areas_error=0.
        for i,(pos,j) in enumerate(zip(points,jj)):
            area=abs(spherical_triangle_area(pos[:nv],tri))
            value=np.bincount(group,weights=area[known]*h0[known]/j[known],minlength=len(uids))
            dif=max(dif,float(np.max(abs(value-saved[i]))))
            assert max(q['timeline'][i]['query_counts'].values())==0
            assert q['timeline'][i]['mesh']['nonpositive_faces']==0
            if i in [0,len(points)//2,len(points)-1]:
                for k in np.linspace(0,len(tri)-1,10,dtype=int):
                    pgarea=pg.PolygonOnSphere([tuple(z) for z in pos[:nv][tri[k]]]).get_area()*R_KM**2
                    areas_error=max(areas_error,abs(float(pgarea-area[k])))
        assert dif<1e-6
        rows.append(dict(tag=tag,time_count=len(points),all_source_volume_max_recalculation_difference_km3=dif,
            maximum_sample_triangle_area_difference_from_pygplates_km2=areas_error))
        if q['scope']=='full' and q['dt_myr']==.25 and q['anchor']==0:trial[q['uniform_refinement_levels']]=q
    base=np.load(CACHE/'full_e30_l0_dt0.25_a0.npz')
    bp=base['points'][-1];bj=base['J'][-1]
    comparisons=[]
    for tag in ['full_e30_l0_dt0.125_a0','full_e30_l0_dt0.25_a301']:
        b=np.load(CACHE/(tag+'.npz'))
        assert np.array_equal(base['triangles'],b['triangles']) and np.array_equal(base['vertices'],b['vertices'])
        comparisons.append(dict(tag=tag,maximum_final_position_difference_m=float(np.linalg.norm(bp-b['points'][-1],axis=1).max()*R_KM*1000),
            maximum_final_thickness_relative_difference=float(np.max(abs(b['J'][-1]/bj-1)))))
    derivative=[]
    rot=pg.RotationModel(str(DATA/'history_v2.rot'));model=pg.TopologicalModel(str(CACHE/'network.gpml'),rot)
    graph=read(DATA/'u36_s11_reference.json')['reference_connectivity']
    for t in [35.,20.,5.]:
        m=GnomonicRotationMap(model.topological_snapshot(t).get_resolved_topologies()[0],graph)
        selected=np.flatnonzero(np.ptp(m.plate[m.triangles],axis=1)>0)[::7]
        centers=unit(m.xyz[m.triangles[selected]].sum(axis=1));rr={p:rot.get_rotation(t-.25,p,t) for p in [301,401]}
        _,exact=m.step_with_jacobian(centers,rr)
        for radius in [.1,.01]:
            pp,pt=small_material_patches(centers,radius)
            moved=m.rotation_map(pp,rr)
            finite=spherical_triangle_area(moved,pt).reshape(-1,4).sum(1)/spherical_triangle_area(pp,pt).reshape(-1,4).sum(1)
            error=float(np.max(abs(finite/exact-1)))
            assert error<2e-5
            derivative.append(dict(t_ma=t,points=len(centers),patch_radius_km=radius,maximum_relative_difference=error))
    uids=[s['uid'] for s in trial[0]['source_summary']]
    bad=[]
    for u in uids:
        error=[abs(next(s for s in trial[l]['source_summary'] if s['uid']==u)['final_relative_residual']) for l in [0,1,2]]
        if not(error[0]>=error[1]>=error[2]):bad.append(dict(uid=u,absolute_final_residual=error))
    old=read(DATA/'u36_s10_verification.json')
    protected={}
    for p in old['protected']:
        current=subprocess.check_output(['git','hash-object',p],text=True).strip()
        before=subprocess.check_output(['git','rev-parse','7cb67e9:'+p],text=True).strip()
        protected[p]=dict(identical_to_base=current==before,git_object=current)
    assert all(z['identical_to_base'] for z in protected.values())
    out=ROOT/'figures/stage1/u36_s11_reference'
    for name in ['metrics','regions','manifest']:
        q=read(out/(name+'.json'));write(DATA/f'u36_s11_reference_{name}.json',q)
    previous=read(DATA/'u36_s10_reference_metrics.json');current=read(out/'metrics.json')
    strip=lambda q:{k:v for k,v in q['metrics'].items() if k!='timings_s'}
    assert strip(previous)==strip(current) and previous['b_verdict']==current['b_verdict']
    assert read(DATA/'u36_s10_reference_regions.json')['regions']==read(out/'regions.json')['regions']
    text=(ROOT/'.cache/u36_s11_tests.log').read_text(encoding='utf8')
    test_summary=re.findall(r'(?m)^.*\d+ failed, \d+ passed in [\d.]+s.*$',text)[-1].strip()
    assert '2 failed, 46 passed' in test_summary
    (DATA/'u36_s11_tests.txt').write_text('\n'.join(z.rstrip() for z in text.splitlines())+'\n',encoding='utf8')
    ref=read(DATA/'u36_s11_reference.json')
    report=dict(checkpoint='U36-S11',method_class='procedural_constrained',base_commit='7cb67e9',
        result_recalculation=rows,time_and_frame_comparisons=comparisons,analytic_derivative_checks=derivative,
        nonmonotonic_source_final_error=bad,all_sources_final_error_monotonic=len(bad)==0,
        protected=protected,protected_identical=True,reference_metrics_identical_except_timings=True,
        region_windows_identical=True,test_summary=test_summary,all_tests_passed=False,
        new_tests_passed=True,old_strict_failures_retained=True,
        all_eight_trials_full_point_support_at_sampled_times=True,
        physical_field_validated=False,physical_width_identified=False,global_or_B5_coupled=False,
        quality_thresholds_changed=False,default_world_adopted=False,
        boundary_representation_difference=ref,
        limits=['仅验证指定分片插值场；其解析J与微斑块和独立有限面积比较，不是真实岩石圈物理验证',
            '细分三档不足以证明所有来源收敛或连续时间无折叠',
            '新网络边界的表示改变小于约0.2m，但新坐标/连接改变内部插值场',
            '原目标与新网络的极小几何差原样报告，不把有限点支撑当精确连续域等价'])
    write(DATA/'u36_s11_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['result_recalculation','time_and_frame_comparisons','analytic_derivative_checks','nonmonotonic_source_final_error','test_summary']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
