"""U36-S10：固定来源域中检验S9候选的柱连续性；不写回正式引擎。"""
from __future__ import annotations
from pathlib import Path
import argparse,json,pickle,time,subprocess
import numpy as np
import pygplates as pg
from mythic.tectonics.geometry import unit
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.source_clipping import triangulate_sources
from mythic.tectonics.rotation_interpolation import BoundaryRotationMap,rotation_matrix
from mythic.tectonics.rotation_transport import IndexedRotationMap
from mythic.tectonics.crust import CONT
from design_material_audit import write_json

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/u36_s10'
DATA=ROOT/'data/worlds/design_v2'

def run_trial(pieces,projection,edge=30.,radius=1.,dt=.25,anchor=0,tag=None,scope='fixed_inner_domain'):
    started=time.perf_counter()
    tag=tag or f'edge{edge:g}_patch{radius:g}_dt{dt:g}_a{anchor}'
    v,tri,source,fragments=triangulate_sources(pieces,projection,max_edge_km=edge,return_diagnostics=True)
    h0=np.array([pieces[i]['T_km'] if pieces[i]['ctype']==CONT else np.nan for i in source])
    uid=np.array([pieces[i]['uid'] for i in source]);known=np.isfinite(h0)
    centers=unit(v[tri].sum(axis=1))
    pp,pt=small_material_patches(centers,radius)
    points0=np.r_[v,pp];nv=len(v)
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    features=pg.FeatureCollection(str(CACHE/'network.gpml'))
    template=pg.TopologicalModel(features,rot).topological_snapshot(35.).get_resolved_topologies()[0]
    archived=json.loads((DATA/'u36_s9_mesh_audit.json').read_text(encoding='utf8'))
    reference=next(z for z in archived['rows'] if z['case']=='real')['reference_connectivity']
    connectivity=[[n.replace('S9_real','S10_real') for n in face] for face in reference]
    # 同一候选必须使用归档的来源连接，不能从换机重建三角网再选一次。
    BoundaryRotationMap(template,connectivity)
    model=pg.TopologicalModel(features,rot,anchor_plate_id=anchor,topological_snapshot_cache_size=2)
    points=points0@rotation_matrix(rot.get_rotation(35.,anchor))
    a0=abs(spherical_triangle_area(v,tri))
    signed0=spherical_triangle_area(v,tri)
    p0=spherical_triangle_area(pp,pt).reshape(-1,4).sum(axis=1)
    uids,inverse=np.unique(uid[known],return_inverse=True)
    v0=np.bincount(inverse,weights=a0[known]*h0[known],minlength=len(uids))
    timeline=[];volumes=[];saved={};failure=None;min_det=float('inf')
    times=35.-np.arange(round(35/dt)+1)*dt
    for j,t in enumerate(times):
        net=model.topological_snapshot(float(t)).get_resolved_topologies()[0]
        mapper=IndexedRotationMap(net,connectivity)
        restore=rotation_matrix(rot.get_rotation(float(t),anchor))
        q=points@restore.T
        signed=spherical_triangle_area(q[:nv],tri);area=abs(signed)
        jac=spherical_triangle_area(q[nv:],pt).reshape(-1,4).sum(axis=1)/p0
        finite=signed/signed0
        if np.any(finite<=0) or np.any(jac<=0):
            failure=dict(t_ma=float(t),reason='材料有限单元或微斑块面积转负',inverted_cells=int(np.sum(finite<=0)),inverted_patches=int(np.sum(jac<=0)))
            break
        h=h0/jac
        vs=np.bincount(inverse,weights=area[known]*h[known],minlength=len(uids))
        volumes.append(vs)
        timeline.append(dict(t_ma=float(t),volume_km3=float(vs.sum()),volume_relative_residual=float(vs.sum()/v0.sum()-1),
            weighted_source_absolute_relative_error=float(np.sum(abs(vs-v0))/v0.sum()),
            maximum_source_absolute_relative_error=float(np.max(abs(vs/v0-1))),
            worst_source_uid=int(uids[np.argmax(abs(vs/v0-1))]),
            known_thickness_quantiles_km=np.quantile(h[known],[0,.1,.5,.9,1]).tolist(),
            inverted_cells=0,inverted_patches=0))
        if t in [35.,20.,0.]:
            saved.update({f'points_{int(t)}Ma':q,f'H_{int(t)}Ma':h,f'J_{int(t)}Ma':jac})
        try:
            if t==0.:
                mapper.weights(points)
            else:
                rr={p:rot.get_rotation(float(t-dt),p,float(t),anchor_plate_id=anchor) for p in [301,401]}
                points=mapper.rotation_map(points,rr)
            min_det=min(min_det,mapper.minimum_selected_triangle_det)
        except ValueError as error:
            _,flags=mapper.support_diagnostics(points)
            bad=np.logical_or.reduce(list(flags.values()))
            badcells=bad[:nv][tri].any(axis=1)|bad[nv:nv+len(tri)]|bad[nv+len(tri):].reshape(-1,4).any(axis=1)
            failure=dict(t_ma=float(t),reason=str(error),query_counts={k:int(v.sum()) for k,v in flags.items()},
                affected_known_source_uids=np.unique(uid[badcells&known]).tolist(),affected_known_initial_volume_km3=float(np.sum(a0[badcells&known]*h0[badcells&known])))
            np.savez_compressed(CACHE/(tag+'_failure.npz'),points=points,bad=bad)
            break
        if j%40==0:print(tag,'time',t,'net%',100*timeline[-1]['volume_relative_residual'],'seconds',time.perf_counter()-started,flush=True)
    volumes=np.asarray(volumes)
    np.savez_compressed(CACHE/(tag+'.npz'),vertices=v,triangles=tri,patch_triangles=pt,
        H0_km=h0,source_uid=uid,source_index=source,known=known,uids=uids,V0_km3=v0,
        source_volumes_km3=volumes,times_ma=times[:len(volumes)],**saved)
    import psutil
    peak=getattr(psutil.Process().memory_info(),'peak_wset',None)
    report=dict(checkpoint='U36-S10',tag=tag,scope=scope,method_class='procedural_constrained',
        mapping_class='interpolated',mapping='S9固定35Ma来源连接+LAEA重心权重+两侧有限旋转SLERP',
        framework_version=pg.__version__,maximum_edge_km=edge,patch_radius_km=radius,dt_myr=dt,anchor=anchor,
        triangle_count=len(tri),point_count=len(points),initial_known_volume_km3=float(v0.sum()),
        numerical_fragments=fragments,complete_35_to_0Ma=failure is None,
        all_points_preserved=True,failure=failure,minimum_selected_projected_det=min_det,
        elapsed_s=time.perf_counter()-started,process_peak_working_set_mib=peak/2**20 if peak else None,
        timeline=timeline,source_trajectories=[dict(uid=int(u),initial_volume_km3=float(a),volume_km3=volumes[:,i].tolist()) for i,(u,a) in enumerate(zip(uids,v0))],
        units={'length':'km','area':'km2','volume':'km3','thickness':'km','time':'Ma/Myr'},
        assumptions=['H=H0/微斑块J，有限来源三角形面积独立验算，不用H0/有限单元J自证闭合',
        '2km固定内域为S8相同审计保护余量；200km物理带宽仍为user_override',
        '原始非陆壳和无覆盖保留缺值；不把来源UID当成分混合比',
        '141/281离散时点检查，不声称连续时间完全支持；不关闭S9整网边界阻断'])
    write_json(DATA/f'u36_s10_{tag}.json',report)
    print('FINISHED',tag,report['complete_35_to_0Ma'],report['elapsed_s'],timeline[-1] if timeline else failure,flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge',type=float,default=30.)
    parser.add_argument('--radius',type=float,default=1.)
    parser.add_argument('--dt',type=float,default=.25)
    parser.add_argument('--anchor',type=int,default=0)
    parser.add_argument('--outer',action='store_true')
    parser.add_argument('--uid',type=int)
    args=parser.parse_args()
    if args.dt<=0 or not np.isclose(round(1/args.dt)*args.dt,1):raise ValueError('步长须为正且整除1Myr')
    with (CACHE/'sources.pkl').open('rb') as stream:data=pickle.load(stream)
    pieces=data['pieces'];tag=None;scope='fixed_inner_domain'
    if args.uid is not None:
        if args.outer:raise ValueError('单来源诊断与外圈尝试分开运行')
        pieces=[z for z in pieces if z['uid']==args.uid]
        if not pieces:raise ValueError('来源不在原固定域中')
        tag=f'uid{args.uid}_edge{args.edge:g}_patch{args.radius:g}'
        scope='single_source_diagnostic'
    if args.outer:
        pieces=[]
        for p in data['original_pieces']:
            hit=p['geometry'].difference(data['domain'])
            if not hit.is_empty:pieces.append(dict(p,geometry=hit))
        tag='outer_attempt';scope='outer_boundary_attempt'
    run_trial(pieces,data['projection'],args.edge,args.radius,args.dt,args.anchor,tag=tag,scope=scope)
if __name__=='__main__':main()
