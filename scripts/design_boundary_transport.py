"""U36-S11：原始完整来源域的边界与解析面积导数小试，不接全球/B5。"""
from pathlib import Path
import argparse,json,pickle,time,hashlib,subprocess
import numpy as np
import pygplates as pg
import shapely
from mythic.tectonics.geometry import unit,R_KM
from mythic.tectonics.material_band import reference_strip,spherical_triangle_area,point_array
from mythic.tectonics.source_clipping import triangulate_sources,spherical_geometry_area
from mythic.tectonics.boundary_transport import GnomonicRotationMap,aligned_network,constrained_connectivity
from mythic.tectonics.rotation_interpolation import rotation_matrix
from mythic.tectonics.crust import CONT
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s11'
DATA=ROOT/'data/worlds/design_v2'
def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    s=json.loads((DATA/'u36_s4_material_band.json').read_text(encoding='utf8'))
    strip=reference_strip(*np.asarray(s['registered_endpoints_xyz']),200.,30.)
    features=aligned_network(strip,rot,tag='S11_aligned')
    pg.FeatureCollection(features).write(str(CACHE/'network.gpml'))
    net=pg.TopologicalModel(features,rot).topological_snapshot(35.).get_resolved_topologies()[0]
    record_path=DATA/'u36_s11_reference.json'
    archived=json.loads(record_path.read_text(encoding='utf8')) if record_path.exists() else None
    graph=archived['reference_connectivity'] if archived else constrained_connectivity(net)
    m=GnomonicRotationMap(net,graph)
    with (ROOT/'.cache/u36_s10/sources.pkl').open('rb') as stream:d=pickle.load(stream)
    if archived and hashlib.sha256(d['target'].wkb).hexdigest()!=archived['source_target_wkb_sha256']:
        raise ValueError('归档来源域改变，不能静默重选插值连接')
    old=pg.TopologicalModel(str(ROOT/'.cache/u36_s10/network.gpml'),rot).topological_snapshot(35.).get_resolved_topologies()[0]
    a=shapely.Polygon(d['projection'].forward(point_array(old.get_resolved_boundary().get_points())))
    b=shapely.Polygon(d['projection'].forward(point_array(net.get_resolved_boundary().get_points())))
    target=d['target'];bd=target.boundary
    row=dict(checkpoint='U36-S11',method_class='procedural_constrained',mapping_class='interpolated',
        reference_connectivity=graph,reference_connectivity_reuses_archive=archived is not None,
        reference_connectivity_sha256=hashlib.sha256(json.dumps(graph,separators=(',',':')).encode()).hexdigest(),
        network_boundary_nodes=len(m.xyz),triangle_count=len(graph),
        physical_input_width_km=200.,end_guard_km=400.,
        old_new_boundary_hausdorff_m=float(a.hausdorff_distance(b)*1000),
        new_network_minus_old_area_km2=float(spherical_geometry_area(b,d['projection'])-spherical_geometry_area(a,d['projection'])),
        original_target_outside_new_network_area_km2=float(spherical_geometry_area(target.difference(b),d['projection'])),
        source_target_wkb_sha256=hashlib.sha256(target.wkb).hexdigest(),
        geometry_backend=f'Shapely {shapely.__version__}/GEOS {shapely.geos_version_string}',
        assumptions=['材料范围内采用原材料轨道节点；只在范围外保留400km端部余量采样',
            '指定边界的折线表示和插值连接确实改变，不宣称同一S9映射',
            '不移动来源边界，不丢外圈，不增壳，不修改原200km假设宽度'])
    write(DATA/'u36_s11_reference.json',row)
    return row
def mesh_audit(m):
    xy=m.project(m.xyz);p=shapely.Polygon(m.project(point_array(m.network.get_resolved_boundary().get_points())))
    faces=shapely.polygons(xy[m.triangles]);u=shapely.union_all(faces)
    return dict(nonpositive_faces=int((m.determinants<=0).sum()),
        missing_relative_area=float(p.difference(u).area/p.area),
        excess_relative_area=float(u.difference(p).area/p.area),
        overlap_relative_area=float((shapely.area(faces).sum()-u.area)/p.area))
def run(args):
    start=time.perf_counter()
    if not (DATA/'u36_s11_reference.json').exists():prepare()
    ref=json.loads((DATA/'u36_s11_reference.json').read_text(encoding='utf8'))
    with (ROOT/'.cache/u36_s10/sources.pkl').open('rb') as stream:d=pickle.load(stream)
    pieces=d['original_pieces']
    if args.scope=='outer':
        pieces=[dict(p,geometry=p['geometry'].difference(d['domain'])) for p in pieces]
        pieces=[p for p in pieces if not p['geometry'].is_empty]
    if args.scope=='inner':pieces=d['pieces']
    if args.uid is not None:pieces=[p for p in pieces if p['uid']==args.uid]
    v,tri,src,fragments=triangulate_sources(pieces,d['projection'],args.edge,True)
    for _ in range(args.levels):
        abc=v[tri];a,b,c=abc.transpose(1,0,2);ab,bc,ca=unit(a+b),unit(b+c),unit(c+a)
        refined=np.concatenate([np.stack(z,axis=1) for z in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
        keep=len(v)
        v,inv=np.unique(np.r_[v,refined.reshape(-1,3)],axis=0,return_inverse=True)
        tri=inv[keep:].reshape(-1,3);src=np.tile(src,4)
    h0=np.array([pieces[i]['T_km'] if pieces[i]['ctype']==CONT else np.nan for i in src])
    uid=np.array([pieces[i]['uid'] for i in src]);known=np.isfinite(h0)
    signed0=spherical_triangle_area(v,tri);a0=abs(signed0)
    centers=unit(v[tri].sum(axis=1));nvert=len(v);points0=np.r_[v,centers]
    uids,group=np.unique(uid[known],return_inverse=True)
    v0=np.bincount(group,weights=a0[known]*h0[known],minlength=len(uids))
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    model=pg.TopologicalModel(str(CACHE/'network.gpml'),rot,anchor_plate_id=args.anchor,topological_snapshot_cache_size=2)
    points=points0@rotation_matrix(rot.get_rotation(35.,args.anchor))
    cumulative=np.ones(len(tri));timeline=[];volumes=[];snapshots=[];j_history=[];step_history=[]
    failure=None;times=35.-np.arange(round(35/args.dt)+1)*args.dt
    tag=f'{args.scope}_e{args.edge:g}_l{args.levels}_dt{args.dt:g}_a{args.anchor}'+(f'_uid{args.uid}' if args.uid is not None else '')
    for i,t in enumerate(times):
        net=model.topological_snapshot(float(t)).get_resolved_topologies()[0]
        try:
            m=GnomonicRotationMap(net,ref['reference_connectivity'])
            mesh=mesh_audit(m)
            if mesh['nonpositive_faces'] or max(mesh[k] for k in ['missing_relative_area','excess_relative_area','overlap_relative_area'])>1e-9:
                raise ValueError('完整支撑网未形成无翻转、无缺口和无重叠的区域分割')
            _,flags=m.support_diagnostics(points)
            if any(z.any() for z in flags.values()):m.weights(points)
            q=points@rotation_matrix(rot.get_rotation(float(t),args.anchor)).T
            signed=spherical_triangle_area(q[:nvert],tri)
            if np.any(signed/signed0<=0):raise ValueError('来源有限三角形翻转')
            h=h0/cumulative
            vs=np.bincount(group,weights=abs(signed)[known]*h[known],minlength=len(uids))
            volumes.append(vs);snapshots.append(q);j_history.append(cumulative.copy())
            timeline.append(dict(t_ma=float(t),volume_relative_residual=float(vs.sum()/v0.sum()-1),
                weighted_source_absolute_relative_error=float(np.sum(abs(vs-v0))/v0.sum()),
                worst_source_uid=int(uids[np.argmax(abs(vs/v0-1))]),
                worst_source_relative_residual=float((vs/v0-1)[np.argmax(abs(vs/v0-1))]),
                thickness_km_quantiles=np.quantile(h[known],[0,.1,.5,.9,1]).tolist(),
                mesh=mesh,query_counts={k:int(z.sum()) for k,z in flags.items()}))
            if t==0:break
            rr={p:rot.get_rotation(float(t-args.dt),p,float(t),anchor_plate_id=args.anchor) for p in [301,401]}
            points,step=m.step_with_jacobian(points,rr)
            cumulative*=step[nvert:];step_history.append(step[nvert:])
            if i%40==0:print(tag,'t',t,'net%',timeline[-1]['volume_relative_residual']*100,flush=True)
        except ValueError as error:
            failure=dict(t_ma=float(t),reason=str(error))
            try:
                _,flags=m.support_diagnostics(points)
                failure['query_counts']={k:int(z.sum()) for k,z in flags.items()}
            except Exception:pass
            np.savez_compressed(CACHE/(tag+'_failure.npz'),points=points)
            break
    volumes=np.asarray(volumes);snapshots=np.asarray(snapshots);j_history=np.asarray(j_history)
    np.savez_compressed(CACHE/(tag+'.npz'),vertices=v,triangles=tri,source_uid=uid,
        known=known,H0_km=h0,uids=uids,V0_km3=v0,points=snapshots,
        J=j_history,step_J=np.asarray(step_history),times_ma=times[:len(timeline)],
        source_volumes_km3=volumes)
    peak=getattr(__import__('psutil').Process().memory_info(),'peak_wset',None)
    summaries=[]
    if len(volumes):
        for k,u in enumerate(uids):
            residual=volumes[:,k]/v0[k]-1
            summaries.append(dict(uid=int(u),initial_volume_km3=float(v0[k]),final_volume_km3=float(volumes[-1,k]),
                final_relative_residual=float(residual[-1]),maximum_absolute_relative_residual=float(abs(residual).max())))
    row=dict(checkpoint='U36-S11',method_class='procedural_constrained',mapping_class='interpolated',
        mapping='显式边界采样+心射约束来源连接+两板块SLERP，逐步解析球面面积导数',
        tag=tag,scope=args.scope,edge_km=args.edge,uniform_refinement_levels=args.levels,dt_myr=args.dt,anchor=args.anchor,
        triangle_count=len(tri),vertex_count=nvert,all_auditable_cells_and_vertices_preserved=True,
        initial_known_volume_km3=float(v0.sum()),complete_35_to_0Ma=failure is None,failure=failure,
        fragments=fragments,elapsed_s=time.perf_counter()-start,process_peak_working_set_mib=peak/2**20 if peak else None,
        timeline=timeline,source_summary=summaries,framework_version=pg.__version__,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        units={'length':'km','area':'km2','volume':'km3','thickness':'km','time':'Ma/Myr'},
        limitations=['所有141/281离散时点检查，不证明连续时间始终支持',
            'J是所选分片插值场的解析导数；片边可有单侧不同导数，不是物理真值',
            '来源体积仍由有限来源三角形面积乘中心厚度独立求积，不用同一有限面积比归一化',
            '重采样改变插值场；200km物理带宽和古接触仍未由地学证据确定',
            '非陆壳/无覆盖保留缺值，未接全球/B5，未改变默认世界'])
    write(DATA/f'u36_s11_{tag}.json',row)
    print('FINISHED',tag,'complete',failure is None,'seconds',row['elapsed_s'],'failure',failure,'last',timeline[-1] if timeline else None,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--scope',choices=['full','outer','inner'],default='full')
    p.add_argument('--edge',type=float,default=30.);p.add_argument('--levels',type=int,default=0)
    p.add_argument('--dt',type=float,default=.25);p.add_argument('--anchor',type=int,default=0)
    p.add_argument('--uid',type=int);a=p.parse_args()
    if a.dt<=0 or not np.isclose(round(1/a.dt)*a.dt,1):raise ValueError('步长须为正并整除1Myr')
    if a.levels<0 or a.levels>3:raise ValueError('本小试只允许0至3层均匀细分')
    if a.prepare:print(json.dumps(prepare(),ensure_ascii=False),flush=True)
    else:run(a)
