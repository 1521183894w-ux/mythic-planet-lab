"""U36-S15：同一S14材料到真实原生宿主的只读球面交叠与有限重建。"""
from pathlib import Path
import argparse,json,pickle,time,hashlib,subprocess
import numpy as np
import shapely,scipy
from shapely.geometry import Polygon,GeometryCollection
from mythic.tectonics.geometry import R_KM,unit,km_from_chord
from mythic.tectonics.source_clipping import LocalGnomonic,voronoi_vertices
from mythic.tectonics.native_remap import spherical_moments,same_source_neighbors,limited_reconstruction,overlap_integrals
from mythic.tectonics.sampling import native_cover
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s15';DATA=ROOT/'data/worlds/design_v2'


def clean(q):
    if isinstance(q,dict):return {str(k):clean(v) for k,v in q.items()}
    if isinstance(q,(list,tuple)):return [clean(v) for v in q]
    if isinstance(q,np.ndarray):return clean(q.tolist())
    if isinstance(q,(float,np.floating)):return float(q) if np.isfinite(q) else None
    if isinstance(q,(np.bool_,)):return bool(q)
    if isinstance(q,(np.integer,)):return int(q)
    return q


def write(path,q):path.write_text(json.dumps(clean(q),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def pack_geometry(polygons,prefix):
    raw=[g.wkb for g in polygons]
    return {prefix+'_wkb_bytes':np.frombuffer(b''.join(raw),np.uint8),prefix+'_wkb_offsets':np.r_[0,np.cumsum([len(v) for v in raw])]}


def native_partition(eng,domain,projection,padding):
    """沿用native_cover宿主优先级，原生大圆Voronoi边；不填未覆盖。"""
    position=eng.positions();c=eng.c;boundary=projection.inverse(np.array(domain.envelope.exterior.coords)[:-1])
    radius=km_from_chord(np.linalg.norm(boundary-projection.center,axis=1)).max()
    distance=km_from_chord(np.linalg.norm(position-projection.center,axis=1))
    candidates=np.flatnonzero(eng.alive&(distance<radius+padding))
    cells=voronoi_vertices(eng.m,c.node[candidates])
    score=c.ctype[candidates]*1e6+c.T[candidates]*1e3-c.age[candidates]
    order=np.lexsort((c.uid[candidates],c.plate[candidates],-score))
    claimed=GeometryCollection();polys=[];uids=[];types=[];full=[]
    for index in candidates[order]:
        xyz=cells[int(c.node[index])]@eng.R[int(c.plate[index])].T
        raw=Polygon(projection.forward(xyz));hit=raw.intersection(domain)
        if hit.is_empty:continue
        owned=hit.difference(claimed);claimed=claimed.union(hit)
        if owned.is_empty:continue
        polys.append(owned);uids.append(int(c.uid[index]));types.append(int(c.ctype[index]));full.append(spherical_moments(raw,projection)[0])
    gap=domain.difference(claimed)
    if not gap.is_empty:polys.append(gap);uids.append(-1);types.append(-1);full.append(np.nan)
    faces=eng.m.simplices[np.any(np.isin(eng.m.simplices,c.node[candidates]),axis=1)]
    plates=np.array(sorted(eng.R))
    inputs=dict(native_candidate_node=c.node[candidates],native_candidate_uid=c.uid[candidates],
                native_candidate_plate=c.plate[candidates],native_candidate_type=c.ctype[candidates],
                native_candidate_thickness_km=c.T[candidates],native_candidate_age_myr=c.age[candidates],
                native_face_nodes=faces,native_face_xyz=eng.m.xyz[faces],
                native_plate_ids=plates,native_rotation_matrices=np.array([eng.R[int(v)] for v in plates]))
    return polys,np.array(uids),np.array(types),np.array(full),len(candidates),inputs


def assign_points(points,geometries):
    q=shapely.points(points);pairs=shapely.STRtree(geometries).query(q,predicate='covered_by')
    assigned=np.full(len(points),-2,np.int32)
    # 几何边界共享时保留原优先级；不以最近邻填网外点。
    order=np.lexsort((pairs[1],pairs[0]));a,b=pairs[:,order]
    unique,first=np.unique(a,return_index=True);assigned[unique]=b[first]
    count=np.bincount(a,minlength=len(points))
    return assigned,int(np.sum(count>1))


def main(step,boundary_stride):
    start=time.perf_counter();CACHE.mkdir(exist_ok=True)
    source=ROOT/'.cache/u36_s14/full_l0_b64_q8.npz';a=np.load(source)
    geometry=a['geometry_points'][step];bounds=geometry[a['boundary_indices']][:,::boundary_stride];centers=geometry[a['center_indices']]
    projection=LocalGnomonic(unit(centers.sum(axis=0)));source_polys=[Polygon(projection.forward(b)) for b in bounds]
    assert all(g.is_valid for g in source_polys)
    domain=shapely.union_all(source_polys)
    source_moments=[spherical_moments(g,projection) for g in source_polys]
    area=np.array([v[0] for v in source_moments]);moments=np.array([v[1] for v in source_moments]);centroids=moments/area[:,None]
    h=a['mean_thickness_km'][step,1 if boundary_stride==1 else 0];h0=a['H0_km'];known=a['known'];uid=a['source_uid'];uids=a['uids'];v0=a['V0_km3']
    neighbors=same_source_neighbors(a['initial_triangles'],uid,known)
    gradients,limiter,ranks=limited_reconstruction(h,centroids,bounds,neighbors)
    pkl=ROOT/'.cache/u36_s4/control/reference_35Ma.pkl' if step==0 else CACHE/'reference_0Ma.pkl'
    with pkl.open('rb') as stream:eng=pickle.load(stream)
    before=hashlib.sha256(pickle.dumps((eng.c.T,eng.area,eng.alive,eng.R),protocol=5)).hexdigest()
    target,target_uid,target_type,full_host_area,candidates,native_inputs=native_partition(eng,domain,projection,180.)
    broad,broad_uid,_,_,broad_count,_=native_partition(eng,domain,projection,360.)
    broad_difference=max([spherical_moments(g.symmetric_difference(broad[np.flatnonzero(broad_uid==u)[0]]),projection)[0] if u in broad_uid else spherical_moments(g,projection)[0] for g,u in zip(target,target_uid)]+[0.])
    assert np.array_equal(target_uid,broad_uid)
    overlap=overlap_integrals(target,source_polys,projection,h,centroids,gradients)
    src=overlap['source_cell'];dst=overlap['target_cell'];oa=overlap['area_km2'];nt=len(target)
    source_area_sum=np.bincount(src,weights=oa,minlength=len(h));target_area=np.array([spherical_moments(g,projection)[0] for g in target])
    ka=np.bincount(dst[known[src]],weights=oa[known[src]],minlength=nt);ua=np.bincount(dst[~known[src]],weights=oa[~known[src]],minlength=nt)
    transfer={};target_means={};source_rows=[]
    for method in ['constant','linear']:
        value=overlap[method+'_volume_km3'];vol=np.bincount(dst[known[src]],weights=value[known[src]],minlength=nt)
        target_means[method]=np.divide(vol,ka,out=np.full(nt,np.nan),where=ka>0);target_means[method][target_uid<0]=np.nan;transfer[method]=vol
    for u,v in zip(uids,v0):
        row=dict(uid=int(u),initial_volume_km3=float(v),material_polygon_volume_km3=float(np.sum(area[(uid==u)&known]*h[(uid==u)&known])))
        for method in ['constant','linear']:
            mask=(uid[src]==u)&known[src];values=overlap[method+'_volume_km3']
            mapped=float(np.sum(values[mask&(target_uid[dst]>=0)]));unmapped=float(np.sum(values[mask&(target_uid[dst]<0)]))
            row[method]=dict(mapped_volume_km3=mapped,unmapped_volume_km3=unmapped,total_minus_material_volume_km3=mapped+unmapped-row['material_polygon_volume_km3'],total_minus_initial_volume_km3=mapped+unmapped-v)
        source_rows.append(row)
    snap=0 if step==0 else 2;pool=a['snapshots'][snap]
    raw=dict(source_area_km2=area,source_moment_xyz_km2=moments,source_centroid_xyz=centroids,source_mean_thickness_km=h,
             source_H0_km=h0,source_uid=uid,known=known,source_gradients_km=gradients,limiter=limiter,gradient_rank=ranks,
             source_boundary_xyz=bounds,source_centers_xyz=centers,target_uid=target_uid,target_type=target_type,
             target_area_km2=target_area,target_full_raw_host_area_km2=full_host_area,target_known_area_km2=ka,target_unknown_area_km2=ua,
             target_constant_mean_km=target_means['constant'],target_linear_mean_km=target_means['linear'],
             projection_center_xyz=projection.center,projection_east_xyz=projection.east,projection_north_xyz=projection.north,
             initial_vertices=a['initial_vertices'],initial_triangles=a['initial_triangles'],
             reconstruction_neighbor_offsets=np.r_[0,np.cumsum([len(v) for v in neighbors])],
             reconstruction_neighbor_indices=np.array([i for ns in neighbors for i in ns]),
             uids=uids,V0_km3=v0,**native_inputs,**overlap,**pack_geometry(source_polys,'source'),**pack_geometry(target,'target'))
    groups=np.searchsorted(uids,np.maximum(uid,uids[0]));slot=dst*len(uids)+groups[src]
    component_area=np.bincount(slot[known[src]],weights=oa[known[src]],minlength=nt*len(uids))
    component_means={}
    for method in ['constant','linear']:
        component_volume=np.bincount(slot[known[src]],weights=overlap[method+'_volume_km3'][known[src]],minlength=nt*len(uids))
        means=np.divide(component_volume,component_area,out=np.full_like(component_volume,np.nan),where=component_area>0).reshape(nt,len(uids))
        means[target_uid<0]=np.nan;component_means[method]=means;raw['target_source_'+method+'_mean_km']=means
    raw['target_source_known_area_km2']=component_area.reshape(nt,len(uids))
    audit=[]
    for n in [4,8]:
        points=pool[a[f'q{n}_pool_indices']];w=a[f'q{n}_weights_km2'];j=a[f'q{n}_J'][step];weight=w*j
        reference=h0[:,None]/j;flat=points.reshape(-1,3);assigned,duplicates=assign_points(projection.forward(flat),target)
        assigned=assigned.reshape(j.shape);safe=np.maximum(assigned,0)
        native=native_cover(eng,flat);actual=np.where(native>=0,eng.c.uid[np.maximum(native,0)],-1).reshape(j.shape)
        expected=np.where(assigned>=0,target_uid[safe],-2);mismatch=(actual!=expected)&(assigned>=0)
        inside=np.array([shapely.covers(g,shapely.points(projection.forward(p))) for g,p in zip(source_polys,points)])
        reconstructed=h[:,None]+np.einsum('cni,ci->cn',points-centroids[:,None],gradients)
        row=dict(order=n,point_count=int(j.size),duplicate_target_boundary_count=duplicates,
                 outside_target_count=int(np.sum(assigned<0)),outside_own_source_polygon_count=int(np.sum(~inside)),
                 native_query_mismatch_count=int(mismatch.sum()),
                 known_reference_volume_outside_target_km3=float(np.nansum(np.where(assigned<0,w*h0[:,None],0.))),
                 minimum_sampled_reconstructed_thickness_km=float(np.nanmin(reconstructed)))
        covered=known[:,None]&(assigned>=0)&(target_uid[safe]>=0)
        covered_v0=float(np.nansum(np.where(covered,w*h0[:,None],0.)))
        row['host_covered_reference_volume_km3']=covered_v0
        row['known_reference_volume_without_native_host_km3']=float(np.nansum(np.where(known[:,None]&(assigned>=0)&(target_uid[safe]<0),w*h0[:,None],0.)))
        fields=[('material_constant',np.broadcast_to(h[:,None],j.shape)),('material_linear',reconstructed),
                ('host_constant',target_means['constant'][safe]),('host_linear',target_means['linear'][safe]),
                ('host_source_constant',component_means['constant'][safe,groups[:,None]]),
                ('host_source_linear',component_means['linear'][safe,groups[:,None]])]
        for method,field in fields:
            row[method+'_missing_host_covered_count']=int(np.sum(covered&~np.isfinite(field)))
            valid=known[:,None]&(assigned>=0)&np.isfinite(field)
            error=np.where(valid,weight*abs(field-reference),0.)
            if method.startswith('material_'):row[method+'_full_L1']=float(np.nansum(error)/v0.sum())
            covered_error=np.where(covered&np.isfinite(field),weight*abs(field-reference),0.)
            row[method+'_host_covered_L1']=float(np.nansum(covered_error)/covered_v0)
            raw[f'q{n}_{method}_covered_L1_volume_by_source_cell_km3']=np.nansum(covered_error,axis=1)
        raw.update({f'q{n}_points_xyz':points,f'q{n}_weights_km2':w,f'q{n}_J':j,f'q{n}_target_cell':assigned,f'q{n}_source_contains':inside,f'q{n}_native_query_uid':actual})
        audit.append(row)
    after=hashlib.sha256(pickle.dumps((eng.c.T,eng.area,eng.alive,eng.R),protocol=5)).hexdigest();assert before==after
    tag=f'{35 if step==0 else 0}Ma_b{64//boundary_stride}'
    np.savez_compressed(CACHE/(tag+'.npz'),**raw)
    peak=getattr(__import__('psutil').Process().memory_info(),'peak_wset',None)
    summary=dict(checkpoint='U36-S15',method_class='procedural_constrained',mapping_class='interpolated',tag=tag,
      method_name='S15_spherical_overlap_native_host_stable_moment_positive_cubature_v2',
      units=dict(length='km',area='km²',moment='km²×单位xyz',volume='km³',thickness='km',time='Ma'),
      physical_input_resolution_km=60.,t_ma=35 if step==0 else 0,source_cell_count=len(h),known_cell_count=int(known.sum()),source_count=len(uids),
      target_count=nt,native_target_count=int(np.sum(target_uid>=0)),candidate_count=candidates,broad_candidate_count=broad_count,
      broad_search_maximum_symmetric_difference_km2=float(broad_difference),
      domain_area_km2=spherical_moments(domain,projection)[0],source_area_sum_km2=float(area.sum()),
      source_partition_area_residual_km2=float(area.sum()-spherical_moments(domain,projection)[0]),
      maximum_source_overlap_area_residual_km2=float(abs(source_area_sum-area).max()),
      maximum_target_area_partition_residual_km2=float(abs(ka+ua-target_area).max()),
      known_area_km2=float(ka.sum()),unknown_area_km2=float(ua.sum()),uncovered_native_area_km2=float(target_area[target_uid<0].sum()),
      known_volume_to_uncovered_native_km3={m:float(transfer[m][target_uid<0].sum()) for m in transfer},
      hosts_with_both_known_and_unknown=int(np.sum((ka>0)&(ua>0))),host_mean_domain='仅已知材料覆盖面积；无宿主不输出宿主均值，带外与未知未补值；混合计数是所有正浮点面积，含精度碎片',
      overlap_moment_cubature_entry_count=int(np.sum(overlap['moment_cubature_polygons']>0)),
      negative_linear_fragment_count=int(np.sum(overlap['linear_volume_km3'][known[src]]<0)),
      moment_cubature_area_4_8_sensitivity_km2=float(overlap['moment_cubature_area_4_8_difference_km2'].sum()),
      moment_cubature_vector_4_8_sensitivity_km2=float(overlap['moment_cubature_moment_4_8_difference_km2'].sum()),
      gradient_rank2_count=int(np.sum(ranks==2)),nonzero_limited_gradient_count=int(np.sum(np.linalg.norm(gradients,axis=1)>0)),
      audit=audit,source_summary=source_rows,engine_selected_state_unchanged=before==after,engine_selected_state_sha256=before,
      versions=dict(shapely=shapely.__version__,geos=shapely.geos_version_string,numpy=np.__version__,scipy=scipy.__version__),
      base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      code_sha256={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['packages/tectonics/native_remap.py','scripts/design_native_remap.py']},
      input_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
      elapsed_s=time.perf_counter()-start,peak_working_set_mib=peak/2**20 if peak else None,
      peak_scope='本进程截至NPZ保存，不包含其他并发进程',
      references=['https://earthsystemmodeling.org/regrid/','https://shapely.readthedocs.io/en/2.2.0/reference/shapely.intersection.html'],
      assumptions=['使用冻结控制世界对应年代的真实原生Voronoi宿主及native_cover优先级，只读重映射，不写回地壳',
      '曲边以S14保存边界点之间的大圆段近似，布尔交叠不吸附、不归一化、不删除正面积碎片',
      '闭式矩首项用中心差分；质心超出单位球/外环球冠128eps时用4/8阶正权球面cubature核准并采用8阶，保存敏感性，不裁零体积',
      '线性梯度只从同源已知邻居均值拟合，球面一阶矩维持本单元均值，限幅是边界采样约束',
      '未知和原生未覆盖分别存面积；已知部分均值不能作为整个宿主厚度'],
      limitations=['仅35/0Ma的局部投影窗口，不是完整时序或全球/B5耦合','内部H0/J为指定映射参照，不是独立物理真值',
      '原生宿主内混合会抹平场；按来源保留广延体积不等于保留来源边界或完整原生属性','不是应力、热沉降、沉积、水位或海岸修复'])
    write(DATA/f'u36_s15_{tag}.json',summary);print('FINISHED',tag,summary['elapsed_s'],summary['peak_working_set_mib'],audit,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[0,140],default=140);p.add_argument('--boundary-stride',type=int,choices=[1,2],default=1);a=p.parse_args();main(a.step,a.boundary_stride)
