"""U36-S12：同一S11分片运动场的跨面求积诊断；不接全球/B5。"""
from pathlib import Path
import argparse,json,time,hashlib,pickle,subprocess
import numpy as np
import pygplates as pg
from mythic.tectonics.geometry import unit
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.boundary_transport import GnomonicRotationMap
from mythic.tectonics.piecewise_audit import crossing_reference_faces,split_on_reference_faces,uniform_refine
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s12'
def write(p,q):p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))

def prepare():
    CACHE.mkdir(exist_ok=True)
    base=np.load(ROOT/'.cache/u36_s11/full_e30_l0_dt0.25_a0.npz')
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s11/network.gpml'),rot,topological_snapshot_cache_size=2)
    ref=read(DATA/'u36_s11_reference.json')
    m=GnomonicRotationMap(model.topological_snapshot(35.).get_resolved_topologies()[0],ref['reference_connectivity'])
    v,tri,parent,diag=split_on_reference_faces(m,base['vertices'],base['triangles'])
    h=base['H0_km'][parent];uid=base['source_uid'][parent];known=np.isfinite(h)
    uids=base['uids'];a=abs(spherical_triangle_area(v,tri));g=np.searchsorted(uids,uid[known])
    v0=np.bincount(g,weights=a[known]*h[known],minlength=len(uids))
    diag.update(checkpoint='U36-S12',method_class='procedural_constrained',mapping_class='interpolated',
        input_triangles=len(base['triangles']),split_triangles=len(tri),
        reference_connectivity_sha256=ref['reference_connectivity_sha256'],
        original_source_target_wkb_sha256=ref['source_target_wkb_sha256'],
        maximum_UID_initial_volume_difference_km3=float(abs(v0-base['V0_km3']).max()),
        initial_known_volume_difference_km3=float(v0.sum()-base['V0_km3'].sum()),
        baseline_initial_crossing_count=int(crossing_reference_faces(m,base['vertices'],base['triangles']).sum()),
        split_initial_crossing_count=int(crossing_reference_faces(m,v,tri).sum()),
        note='只切分同一有限来源三角形；没有移动来源、换连接、调带宽或逐UID归一化')
    assert diag['maximum_UID_initial_volume_difference_km3']<1e-5
    for z in diag['excluded_tiny_subfragments']:
        k=z['parent'];z['uid']=int(base['source_uid'][k]);h=base['H0_km'][k]
        z['known_initial_volume_km3']=float(z['area_km2']*h) if np.isfinite(h) else None
    diag['excluded_tiny_known_initial_volume_km3']=sum(z['known_initial_volume_km3'] or 0 for z in diag['excluded_tiny_subfragments'])
    diag['precision_warning']='精度底附近小片面积比仍受浮点影响，正面积不证明局部演化可靠'
    np.savez_compressed(CACHE/'partition.npz',vertices=v,triangles=tri,parent=parent)
    write(DATA/'u36_s12_partition.json',diag)
    print('PARTITION', {k:v for k,v in diag.items() if k not in ['initial_area_difference_by_parent_km2','excluded_tiny_subfragments']},flush=True)

def run(mode,level):
    start=time.perf_counter();CACHE.mkdir(exist_ok=True)
    base=np.load(ROOT/'.cache/u36_s11/full_e30_l0_dt0.25_a0.npz')
    if mode=='original':
        old=np.load(ROOT/f'.cache/u36_s11/full_e30_l{level}_dt0.25_a0.npz')
        v=old['vertices'];tri=old['triangles'];h0=old['H0_km'];uid=old['source_uid']
        # NPZ列只解压一次，不在每一步重复展开整个历史。
        original_positions=old['points'];original_J=old['J']
    else:
        a=np.load(CACHE/'partition.npz');v=a['vertices'];tri=a['triangles'];parent=a['parent']
        for _ in range(level):v,tri,parent=uniform_refine(v,tri,parent)
        h0=base['H0_km'][parent];uid=base['source_uid'][parent]
    precision_ledger=[]
    if mode=='aligned':
        areas=abs(spherical_triangle_area(v,tri));tiny=areas<=1e-9
        precision_ledger=[dict(uid=int(u),area_km2=float(a),known_volume_km3=float(a*h) if np.isfinite(h) else None) for u,a,h in zip(uid[tiny],areas[tiny],h0[tiny])]
        tri,h0,uid=tri[~tiny],h0[~tiny],uid[~tiny]
    known=np.isfinite(h0);uids,group=np.unique(uid[known],return_inverse=True)
    area0=abs(spherical_triangle_area(v,tri));v0=np.bincount(group,weights=area0[known]*h0[known],minlength=len(uids))
    centers=unit(v[tri].sum(axis=1));nv=len(v);points=np.r_[v,centers];cum=np.ones(len(tri))
    times=35.-np.arange(141)*.25
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s11/network.gpml'),rot,topological_snapshot_cache_size=2)
    graph=read(DATA/'u36_s11_reference.json')['reference_connectivity']
    abs_change=np.zeros((2,len(uids)));signed_change=abs_change.copy();exposure=abs_change.copy()
    timeline=[];volumes=[];snapshots=[];j_history=[];crossed_ever=np.zeros(len(tri),bool)
    def volume(pos,j):return abs(spherical_triangle_area(pos[:nv],tri))*h0/j
    vprev=area0*h0
    for i,t in enumerate(times):
        m=GnomonicRotationMap(model.topological_snapshot(float(t)).get_resolved_topologies()[0],graph)
        m.weights(points)
        cellv=volume(points,cum);vs=np.bincount(group,weights=cellv[known],minlength=len(uids))
        volumes.append(vs);snapshots.append(points.copy());j_history.append(cum.copy())
        timeline.append(dict(t_ma=float(t),net_relative_residual=float(vs.sum()/v0.sum()-1),
            weighted_absolute_source_residual=float(abs(vs-v0).sum()/v0.sum())))
        if t==0:break
        cross=crossing_reference_faces(m,points[:nv],tri);crossed_ever|=cross
        if mode=='original':
            next_points=original_positions[i+1];next_cum=original_J[i+1]
        else:
            rr={p:rot.get_rotation(float(t-.25),p,float(t)) for p in [301,401]}
            next_points,step=m.step_with_jacobian(points,rr);next_cum=cum*step[nv:]
        delta=volume(next_points,next_cum)-cellv
        for k,mask in enumerate([~cross,cross]):
            mask=mask&known
            abs_change[k]+=np.bincount(group[mask[known]],weights=abs(delta[mask]),minlength=len(uids))
            signed_change[k]+=np.bincount(group[mask[known]],weights=delta[mask],minlength=len(uids))
            exposure[k]+=np.bincount(group[mask[known]],weights=area0[mask]*h0[mask],minlength=len(uids))
        points,cum=next_points,next_cum
        if i%40==0:print(mode,level,t,flush=True)
    volumes=np.asarray(volumes)
    assert np.max(abs(signed_change.sum(axis=0)-(volumes[-1]-v0)))<1e-6
    summaries=[dict(uid=int(u),initial_volume_km3=float(v0[k]),final_relative_residual=float(volumes[-1,k]/v0[k]-1),
        crossing_volume_time_fraction=float(exposure[1,k]/(v0[k]*140)),
        crossing_step_absolute_volume_change_km3=float(abs_change[1,k]),
        single_face_step_absolute_volume_change_km3=float(abs_change[0,k]),
        signed_step_change_km3=signed_change[:,k].tolist()) for k,u in enumerate(uids)]
    tag=f'{mode}_l{level}'
    np.savez_compressed(CACHE/(tag+'.npz'),vertices=v,triangles=tri,H0_km=h0,source_uid=uid,known=known,
        uids=uids,V0_km3=v0,points=np.asarray(snapshots),J=np.asarray(j_history),times_ma=times,
        source_volumes_km3=volumes,crossed_ever=crossed_ever)
    peak=getattr(__import__('psutil').Process().memory_info(),'peak_wset',None)
    q=dict(checkpoint='U36-S12',method_class='procedural_constrained',mapping_class='interpolated',mode=mode,level=level,
        triangle_count=len(tri),vertex_count=nv,dt_myr=.25,time_count=len(times),all_points_supported_at_sampled_times=True,
        precision_fragment_ledger=precision_ledger,precision_area_floor_km2=1e-9,
        precision_fragment_known_volume_km3=sum(z['known_volume_km3'] or 0. for z in precision_ledger),
        input_width_km=200.,timeline=timeline,source_summary=summaries,
        ever_crossed_cell_count=int(crossed_ever.sum()),
        crossing_volume_time_fraction=float(exposure[1].sum()/(v0.sum()*140)),
        crossing_step_absolute_volume_change_km3=float(abs_change[1].sum()),
        single_face_step_absolute_volume_change_km3=float(abs_change[0].sum()),
        signed_increment_telescoping_error_km3=float(np.max(abs(signed_change.sum(axis=0)-(volumes[-1]-v0)))),
        elapsed_s=time.perf_counter()-start,peak_working_set_mib=peak/2**20 if peak else None,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        framework_versions=dict(pygplates=pg.__version__,numpy=np.__version__),
        units=dict(length='km',area='km2',volume='km3',thickness='km',time='Ma/Myr'),
        limitations=['保持S11同一分片运动场与原200km假设，不是物理带宽或古接触恢复',
            'crossing仅指有限直边三角形在当前时点越过中心所选分片，不能等同真实断层',
            '一步独立有限面积乘中心厚度的体积差按起点跨面状态分组，仅为关联诊断',
            '初始对齐不保证演化后持续对齐；不每步切分或重映射，不能称完整守恒耦合',
            '不对逐UID体积归一化，非陆壳/无覆盖保留缺值；未接全球/B5'])
    write(DATA/f'u36_s12_{tag}.json',q)
    print('FINISHED',tag,q['triangle_count'],q['elapsed_s'],timeline[-1],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--mode',choices=['original','aligned'],default='aligned');p.add_argument('--level',type=int,choices=[0,1,2],default=0);args=p.parse_args()
    if args.prepare:prepare()
    else:run(args.mode,args.level)