"""U36-S13：同一S11运动场的曲边面积、中心J与内部求积对照；不接全球/B5。"""
from pathlib import Path
import json, time, subprocess, hashlib
import numpy as np
import pygplates as pg
import scipy
from mythic.tectonics.geometry import unit, R_KM
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.boundary_transport import GnomonicRotationMap
from mythic.tectonics.piecewise_audit import crossing_reference_faces
from mythic.tectonics.curved_quadrature import triangle_quadrature, material_boundary, polygon_fan_area
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/worlds/design_v2"; CACHE=ROOT/".cache/u36_s13"
ORDERS=[2,4,8,16,32,64]; SEGMENTS=[1,2,4,8,16,32,64,128,256,512]
BOUNDARY_MAX=max(SEGMENTS); ORDER_MAX=max(ORDERS)


def read(p): return json.loads(p.read_text(encoding="utf8"))
def write(p,q): p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def main():
    started=time.perf_counter(); CACHE.mkdir(exist_ok=True)
    old=np.load(ROOT/".cache/u36_s12/original_l0.npz")
    vertices=old["vertices"]; triangles=old["triangles"]; nv=len(vertices)
    previous=old["points"]; old_j=old["J"]; source=old["source_uid"]; h0=old["H0_km"]
    area=abs(spherical_triangle_area(vertices,triangles))
    afinal=abs(spherical_triangle_area(previous[-1,:nv],triangles))
    error=afinal/(area*old_j[-1])-1
    eligible=np.isfinite(h0)&(area>1)
    crossed=old["crossed_ever"]
    worst=lambda mask:int(np.flatnonzero(mask)[np.argmax(abs(error[mask]))])
    a=int(np.flatnonzero(source==448261)[0])
    b=worst(eligible&crossed)
    c=worst(eligible&~crossed&(np.arange(len(triangles))!=a))
    nonmono=read(DATA/"u36_s12_verification.json")["nonmonotonic_UIDs"]["original"]
    d=worst(eligible&np.isin(source,nonmono)&~np.isin(np.arange(len(triangles)),[a,b,c]))
    chosen=[a,b,c,d]
    reasons=["UID448261，既有不跨片反例","曾跨片组末态单元误差最大（面积>1km²）",
             "其余未跨片组末态单元误差最大（面积>1km²）",
             "原15非单调来源中其余末态单元误差最大（面积>1km²）"]
    selection=[]; parts=[]; specs=[]; cursor=0
    def add(p):
        nonlocal cursor
        index=np.arange(cursor,cursor+len(p)); cursor+=len(p); parts.append(p); return index
    for k,reason in zip(chosen,reasons):
        initial_order=triangles[k].copy()
        if spherical_triangle_area(vertices,initial_order[None])[0]<0: initial_order=initial_order[[0,2,1]]
        abc=vertices[initial_order]
        center=unit(abc.sum(axis=0))
        center_index=int(add(center[None])[0])
        boundary_index=add(material_boundary(abc,BOUNDARY_MAX))
        quadratures={}
        for order in ORDERS:
            p,w=triangle_quadrature(abc,order); index=add(p)
            quadratures[order]=(index,w)
        selection.append(dict(cell_index=k,uid=int(source[k]),reason=reason,
                              initial_area_km2=float(area[k]),H0_km=float(h0[k]),
                              initial_vertex_order=initial_order.tolist(),archived_endpoint_residual=float(error[k]),archived_ever_crossed=bool(crossed[k])))
        specs.append((center_index,boundary_index,quadratures))
    positions=np.concatenate(parts); jacobian=np.ones(len(positions))
    rotation=pg.RotationModel(str(DATA/"history_v2.rot"))
    model=pg.TopologicalModel(str(ROOT/".cache/u36_s11/network.gpml"),rotation,topological_snapshot_cache_size=2)
    ref=read(DATA/"u36_s11_reference.json")
    graph=ref["reference_connectivity"]
    timeline=[]; hist_pos=[]; hist_j=[]
    max_reference_position_m=0.; max_reference_j=0.
    for step,t in enumerate(old["times_ma"]):
        mapper=GnomonicRotationMap(model.topological_snapshot(float(t)).get_resolved_topologies()[0],graph)
        faces,flags=mapper.support_diagnostics(positions)
        if any(v.any() for v in flags.values()): mapper.weights(positions)
        hist_pos.append(positions.copy()); hist_j.append(jacobian.copy())
        rr=[]
        for record,(ci,bi,qs) in zip(selection,specs):
            k=record["cell_index"]; center_j=jacobian[ci]
            boundary=positions[bi]
            endpoint=boundary[np.arange(3)*BOUNDARY_MAX]
            chord,minimum=polygon_fan_area(endpoint,positions[ci])
            boundaries={str(n):polygon_fan_area(boundary.reshape(3,BOUNDARY_MAX,3)[:,::BOUNDARY_MAX//n].reshape(-1,3),positions[ci])[0]
                        for n in SEGMENTS}
            integrals={str(n):float(w@jacobian[indices]) for n,(indices,w) in qs.items()}
            initial_weights={str(n):float(w.sum()) for n,(indices,w) in qs.items()}
            indices,weights=qs[ORDER_MAX]
            face_disagreement=float(weights[faces[indices]!=faces[ci]].sum()/weights.sum())
            # 保留旧直边定义，另列真正跟随材料的内部采样点分片差异。
            chord_cross=bool(crossing_reference_faces(mapper,endpoint,np.array([[0,1,2]]))[0])
            reference=np.r_[previous[step,record["initial_vertex_order"]],previous[step,nv+k][None]]
            here=np.r_[endpoint,positions[ci][None]]
            max_reference_position_m=max(max_reference_position_m,float(np.linalg.norm(reference-here,axis=1).max()*R_KM*1000))
            max_reference_j=max(max_reference_j,abs(float(center_j-old_j[step,k])))
            denominator=record["initial_area_km2"]*center_j
            curved=boundaries[str(BOUNDARY_MAX)]
            rr.append(dict(cell_index=k,uid=record["uid"],center_J=float(center_j),endpoint_area_km2=chord,
                           boundary_area_km2=boundaries,interior_integral_J_km2=integrals,
                           quadrature_initial_area_km2=initial_weights,
                           total_endpoint_center_residual=chord/denominator-1,
                           chord_edge_contribution=(chord-curved)/denominator,
                           center_vs_curved_contribution=(curved-denominator)/denominator,
                           boundary_interior_area_difference_relative=(curved-integrals[str(ORDER_MAX)])/record["initial_area_km2"],
                           quadrature_material_other_face_weight_fraction=face_disagreement,
                           chord_crosses_reference_face=chord_cross,minimum_endpoint_fan_area_km2=minimum))
        timeline.append(dict(t_ma=float(t),cells=rr))
        if t==0: break
        rotations={p:rotation.get_rotation(float(t-.25),p,float(t)) for p in [301,401]}
        positions,delta=mapper.step_with_jacobian(positions,rotations)
        jacobian*=delta
        if step%35==0: print("S13",t,positions.shape,flush=True)
    peak=getattr(__import__("psutil").Process().memory_info(),"peak_wset",None)
    protected={}
    for path in read(DATA/"u36_s12_verification.json")["protected"]:
        base=subprocess.check_output(["git","rev-parse","7cc61a8:"+path],text=True).strip()
        current=subprocess.check_output(["git","hash-object",path],text=True).strip()
        protected[path]=base==current
    assert all(protected.values())
    # 同运动场复现核对与分解算术核对分别命名，不冒充独立物理证明。
    arithmetic=max(abs(z["total_endpoint_center_residual"]-z["chord_edge_contribution"]-z["center_vs_curved_contribution"])
                   for row in timeline for z in row["cells"])
    assert max_reference_position_m<1e-4 and max_reference_j<1e-10 and arithmetic<1e-12
    sample_meta=[dict(center_index=ci,boundary_indices=bi.tolist(),
                     quadratures={str(n):dict(indices=index.tolist(),weights_km2=w.tolist()) for n,(index,w) in qs.items()})
                 for ci,bi,qs in specs]
    np.savez_compressed(CACHE/"material_samples.npz",points=np.asarray(hist_pos),J=np.asarray(hist_j),
                        times_ma=old["times_ma"])
    q=dict(checkpoint="U36-S13",method_class="procedural_constrained",mapping_class="interpolated",
           selection=selection,selection_is_representative=False,timeline=timeline,sampling=sample_meta,
           time_count=len(timeline),sample_count=len(positions),dt_myr=.25,boundary_segments_per_initial_edge=SEGMENTS,
           quadrature_orders_per_Duffy_axis=ORDERS,all_sampled_points_supported=True,
           maximum_archived_vertex_center_position_difference_m=max_reference_position_m,
           maximum_archived_center_J_difference=max_reference_j,maximum_decomposition_arithmetic_error=arithmetic,
           protected=protected,base_commit="7cc61a8",width_km=200.,
           reference_connectivity_sha256=ref["reference_connectivity_sha256"],
           network_sha256=hashlib.sha256((ROOT/".cache/u36_s11/network.gpml").read_bytes()).hexdigest(),
           input_archive_sha256=hashlib.sha256((ROOT/".cache/u36_s12/original_l0.npz").read_bytes()).hexdigest(),
           elapsed_s=time.perf_counter()-started,peak_working_set_mib=peak/2**20 if peak else None,
           peak_measurement_scope="截至材料循环结束的进程峰值；不含后续NPZ序列化及整个工作单元",
           resolution=dict(physical_input_km=60,boundary_segments_per_edge=BOUNDARY_MAX,interior_Gauss_order_per_axis=ORDER_MAX),
           references=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html",
                       "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html"],
           versions=dict(pygplates=pg.__version__,scipy=scipy.__version__,numpy=np.__version__),
           units=dict(length="km",area="km²",time="Ma/Myr",thickness="km"),
           method=dict(boundary="材料边界点独立经过140次S11映射，正向球面扇形求面积",
                       quadrature="局部心射域Duffy变换×SciPy Gauss-Legendre；球面权重不归一化，逐点累计解析J",
                       decomposition="总差=直边与曲边差+曲边平均J与中心J差；这是算术分解，不是物理守恒证明"),
           limitations=["四单元按既有误差选取，是反例/诊断，不代表全部77来源或完整域",
                        "曲边面积采用有限512段，内部最高64×64求积；跨片时空间积分未必收敛",
                        "只查141离散时点样本，非连续时间、全材料域无折叠证明",
                        "未逐来源归一化、未以H/J×J恒等式当守恒证据；没有重算全球/B5",
                        "200km是假设带宽；未恢复古接触或求解应力、热沉降、沉积、水位"])
    write(DATA/"u36_s13_curved_quadrature.json",q)
    print("FINISHED",q["elapsed_s"],q["peak_working_set_mib"],flush=True)
    for z in timeline[-1]["cells"]: print("FINAL",z,flush=True)
if __name__=="__main__": main()
