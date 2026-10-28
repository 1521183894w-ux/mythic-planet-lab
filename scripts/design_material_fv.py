"""U36-S14：零通量移动材料体积账与独立内部厚度误差；不接正式引擎。"""
from pathlib import Path
import argparse,json,time,hashlib,subprocess
import numpy as np
import pygplates as pg
import scipy
from mythic.tectonics.geometry import unit,R_KM
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.boundary_transport import GnomonicRotationMap
from mythic.tectonics.curved_quadrature import triangle_quadrature,material_boundary
from mythic.tectonics.material_fv import boundary_areas,update_material_mean,thickness_error_integrals
from mythic.tectonics.piecewise_audit import uniform_refine
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/worlds/design_v2";CACHE=ROOT/".cache/u36_s14"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def write(p,q):p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
def main(scope,level,segments,orders):
    start=time.perf_counter();CACHE.mkdir(exist_ok=True)
    a=np.load(ROOT/".cache/u36_s12/original_l0.npz")
    original_v=a["vertices"];original_tri=a["triangles"];h_original=a["H0_km"];uid_original=a["source_uid"]
    parent=np.array([z["cell_index"] for z in read(DATA/"u36_s13_curved_quadrature.json")["selection"]]) if scope=="selected" else np.arange(len(original_tri))
    tri=original_tri[parent].copy();vertices=original_v.copy()
    signed=spherical_triangle_area(vertices,tri);tri[signed<0]=tri[signed<0][:,[0,2,1]]
    for _ in range(level):vertices,tri,parent=uniform_refine(vertices,tri,parent)
    area0=abs(spherical_triangle_area(vertices,tri));h0=h_original[parent];uid=uid_original[parent];known=np.isfinite(h0)
    source_uids,source_group=np.unique(uid[known],return_inverse=True)
    source_v0=np.bincount(source_group,weights=area0[known]*h0[known],minlength=len(source_uids))
    def sums(values):
        return np.bincount(source_group,weights=np.asarray(values)[known],minlength=len(source_uids))
    init_inventory_difference={}
    for u in source_uids:
        before=np.sum(abs(spherical_triangle_area(original_v,original_tri[(uid_original==u)&np.isin(np.arange(len(original_tri)),parent)]))*h_original[(uid_original==u)&np.isin(np.arange(len(original_tri)),parent)])
        init_inventory_difference[int(u)]=float(source_v0[np.searchsorted(source_uids,u)]-before)
    nc=len(tri);parts=[];centers=[];bounds=[];quad={n:[] for n in orders};weights={n:[] for n in orders};cursor=0
    def add(p):
        nonlocal cursor
        index=np.arange(cursor,cursor+len(p));cursor+=len(p);parts.append(p);return index
    for abc in vertices[tri]:
        centers.append(int(add(unit(abc.sum(axis=0))[None])[0]))
        bounds.append(add(material_boundary(abc,segments)))
        for n in orders:
            p,w=triangle_quadrature(abc,n);quad[n].append(add(p));weights[n].append(w)
    raw=np.concatenate(parts);pos,inv=np.unique(raw,axis=0,return_inverse=True)
    ci=inv[np.array(centers)];bi=inv[np.array(bounds)]
    qi={n:inv[np.array(quad[n])] for n in orders};ww={n:np.array(weights[n]) for n in orders}
    geometry_indices=np.unique(np.r_[ci,bi.ravel()])
    ci_geometry=np.searchsorted(geometry_indices,ci);bi_geometry=np.searchsorted(geometry_indices,bi)
    jac=np.ones(len(pos));means=np.repeat(h0[None],2,axis=0);prev_area=np.repeat(area0[None],2,axis=0)
    rotation=pg.RotationModel(str(DATA/"history_v2.rot"))
    model=pg.TopologicalModel(str(ROOT/".cache/u36_s11/network.gpml"),rotation,topological_snapshot_cache_size=2)
    reference=read(DATA/"u36_s11_reference.json");times=35-np.arange(141)*.25
    hist_geometry=[];hist_h=[];hist_a=[];hist_chord=[];hist_center_j=[];hist_center_face=[]
    hist_j={n:[] for n in orders};hist_face={n:[] for n in orders};snapshots=[];timeline=[]
    minimum_fan=float("inf");max_position_m=0.;max_center_j=0.
    archive_positions=a["points"] if level==0 else None
    archive_j=a["J"] if level==0 else None
    for step,t in enumerate(times):
        mapper=GnomonicRotationMap(model.topological_snapshot(float(t)).get_resolved_topologies()[0],reference["reference_connectivity"])
        face,flags=mapper.support_diagnostics(pos)
        if any(x.any() for x in flags.values()):mapper.weights(pos)
        boundary=pos[bi]
        fine,minimum=boundary_areas(boundary,pos[ci]);minimum_fan=min(minimum_fan,minimum)
        coarse,_=boundary_areas(boundary.reshape(nc,3,segments,3)[:,:,::2].reshape(nc,-1,3),pos[ci])
        chord,_=boundary_areas(boundary[:,np.arange(3)*segments],pos[ci])
        areas=np.stack([coarse,fine])
        used=areas if step else np.repeat(area0[None],2,axis=0)
        if step:means=update_material_mean(means,prev_area,used)
        law=sums(means[1]*used[1]);legacy=sums(chord*h0/jac[ci])
        row=dict(t_ma=float(t),scheme_source_weighted_absolute_drift=float(abs(law-source_v0).sum()/source_v0.sum()),
                 old_endpoint_center_source_weighted_absolute_drift=float(abs(legacy-source_v0).sum()/source_v0.sum()))
        for n in orders:
            j=jac[qi[n]]
            values=thickness_error_integrals(means[1],h0,ww[n],j)
            low=thickness_error_integrals(means[0],h0,ww[n],j)
            center=thickness_error_integrals(h0/jac[ci],h0,ww[n],j)
            volume=sums(values["candidate_volume_km3"])
            row[f"q{n}"]=dict(weighted_L1_thickness_error=float(np.nansum(values["l1_volume_error_km3"])/source_v0.sum()),
                 old_center_weighted_L1_thickness_error=float(np.nansum(center["l1_volume_error_km3"])/source_v0.sum()),
                 reference_volume_source_weighted_absolute_difference=float(abs(volume-source_v0).sum()/source_v0.sum()),
                 maximum_source_reference_volume_relative_difference=float(np.max(abs(volume/source_v0-1))),
                 coarse_boundary_weighted_L1_thickness_error=float(np.nansum(low["l1_volume_error_km3"])/source_v0.sum()))
            hist_j[n].append(j.copy());hist_face[n].append(face[qi[n]].astype(np.int32))
        timeline.append(row)
        hist_geometry.append(pos[geometry_indices].copy());hist_h.append(means.copy());hist_a.append(used.copy())
        hist_chord.append(chord);hist_center_j.append(jac[ci].copy());hist_center_face.append(face[ci].astype(np.int32))
        if step in [0,70,140]:snapshots.append(pos.copy())
        if level==0:
            old_centers=archive_positions[step,len(original_v)+parent]
            max_position_m=max(max_position_m,float(np.linalg.norm(old_centers-pos[ci],axis=1).max()*R_KM*1000))
            max_center_j=max(max_center_j,float(abs(archive_j[step,parent]-jac[ci]).max()))
        prev_area=used
        if step%20==0:print(scope,level,"Ma",t,"points",len(pos),"seconds",time.perf_counter()-start,flush=True)
        if t==0:break
        rr={p:rotation.get_rotation(float(t-.25),p,float(t)) for p in [301,401]}
        pos,dj=mapper.step_with_jacobian(pos,rr);jac*=dj
    tag=f"{scope}_l{level}_b{segments}_q{max(orders)}"
    payload=dict(times_ma=times,geometry_points=np.asarray(hist_geometry),boundary_indices=bi_geometry,center_indices=ci_geometry,
                 mean_thickness_km=np.asarray(hist_h),scheme_areas_km2=np.asarray(hist_a),
                 chord_areas_km2=np.asarray(hist_chord),center_J=np.asarray(hist_center_j),center_face_id=np.asarray(hist_center_face),
                 snapshots=np.asarray(snapshots),snapshot_steps=np.array([0,70,140]),pool_boundary_indices=bi,pool_center_indices=ci,
                 initial_vertices=vertices,initial_triangles=tri,initial_area_km2=area0,H0_km=h0,source_uid=uid,parent_cell_index=parent,
                 known=known,uids=source_uids,V0_km3=source_v0,geometry_pool_indices=geometry_indices)
    for n in orders:
        payload[f"q{n}_J"]=np.asarray(hist_j[n]);payload[f"q{n}_weights_km2"]=ww[n]
        payload[f"q{n}_face_id"]=np.asarray(hist_face[n]);payload[f"q{n}_pool_indices"]=qi[n]
    np.savez_compressed(CACHE/(tag+".npz"),**payload)
    final=[]
    for k,u in enumerate(source_uids):
        mask=(uid==u)&known
        n=max(orders);j=jac[qi[n]]
        values=thickness_error_integrals(means[1],h0,ww[n],j)
        candidate=float(values["candidate_volume_km3"][mask].sum())
        final.append(dict(uid=int(u),initial_volume_km3=float(source_v0[k]),
            source_known_cells=int(mask.sum()),scheme_relative_drift=float(law[k]/source_v0[k]-1),
            old_endpoint_center_relative_difference=float(legacy[k]/source_v0[k]-1),
            reference_volume_relative_difference=float(candidate/source_v0[k]-1),
            L1_thickness_error=float(values["l1_volume_error_km3"][mask].sum()/source_v0[k])))
    peak=getattr(__import__("psutil").Process().memory_info(),"peak_wset",None)
    protected={}
    paths=list(read(DATA/"u36_s13_verification.json")["protected"])+["packages/tectonics/curved_quadrature.py","scripts/design_curved_quadrature.py","tests/tectonics/test_curved_quadrature.py"]
    for path in paths:
        protected[path]=subprocess.check_output(["git","hash-object",path],text=True).strip()==subprocess.check_output(["git","rev-parse","6be2d4d:"+path],text=True).strip()
    assert all(protected.values())
    q=dict(checkpoint="U36-S14",method_class="procedural_constrained",mapping_class="interpolated",scope=scope,level=level,tag=tag,
        cell_count=nc,known_cell_count=int(known.sum()),source_count=len(source_uids),unique_sample_count=len(pos),
        parent_cells=np.unique(parent).tolist(),initial_volume_difference_by_uid_km3=init_inventory_difference,
        boundary_segments_per_initial_child_edge=segments,coarse_boundary_segments=segments//2,quadrature_orders=orders,
        time_count=141,dt_myr=.25,width_km=200.,physical_input_resolution_km=60.,all_sampled_points_supported=True,
        minimum_sampled_fan_area_km2=minimum_fan,maximum_archived_center_difference_m=max_position_m if level==0 else None,
        maximum_archived_center_J_difference=max_center_j if level==0 else None,timeline=timeline,source_summary=final,protected=protected,
        base_commit="6be2d4d",reference_connectivity_sha256=reference["reference_connectivity_sha256"],
        input_sha256=hashlib.sha256((ROOT/".cache/u36_s12/original_l0.npz").read_bytes()).hexdigest(),
        elapsed_s=time.perf_counter()-start,peak_working_set_mib=peak/2**20 if peak else None,
        peak_scope="截至原始NPZ保存后的本进程峰值，不含其他并发进程或整个工作单元",
        versions=dict(pygplates=pg.__version__,numpy=np.__version__,scipy=scipy.__version__),
        units=dict(length="km",area="km²",volume="km³",time="Ma/Myr",thickness="km"),
        method_name="S14_zero_relative_boundary_flux_material_mean_v1",
        references=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html",
                    "https://faculty.washington.edu/rjl/classes/am574w2011/slides/am574lecture5.pdf"],
        assumptions=["曲边控制体与材料一起移动，边界零通量；恒密度柱体积账，不是三维流变求解",
                     "平均厚度逐步按H_new=H_old*A_old/A_new更新，未逐UID归一化；账面守恒是方案性质",
                     "独立内部H0/J是同一指定离散映射的连续性参照，不是物理真值",
                     "非陆壳/无覆盖H0保留NaN，known仅作已知库存审计，不把未知厚度设零"],
        limitations=["没有材料到原生宿主网格的保守重映射、相邻通量、碰撞、热沉降、沉积或水位",
                     "平均厚度分片常数，可守恒而抹平场内变化；L1误差和求积差另列",
                     "全部141时点有限采样支撑/扇形正向不证明全域连续时间单射",
                     "未接全球/B5，未改默认C2或质量指标，未验证物理带宽和古接触"])
    if scope=="full" and level==0:
        assert np.array_equal(source_uids,a["uids"])
        q["maximum_archived_initial_UID_volume_difference_km3"]=float(abs(source_v0-a["V0_km3"]).max())
    write(DATA/f"u36_s14_{tag}.json",q)
    print("FINISHED",tag,q["elapsed_s"],q["peak_working_set_mib"],timeline[-1],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scope",choices=["selected","full"],default="selected")
    p.add_argument("--level",type=int,choices=[0,1,2],default=0);p.add_argument("--segments",type=int,default=128)
    p.add_argument("--orders",default="8,16,32");args=p.parse_args()
    assert args.segments>=2 and args.segments%2==0
    main(args.scope,args.level,args.segments,[int(v) for v in args.orders.split(",")])
