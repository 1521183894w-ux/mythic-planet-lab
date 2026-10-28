"""S10支持性复算：同一S9材料轨迹与原S8单来源细分，不改变物理引擎。"""
def legacy():
    from pathlib import Path
    import pickle,json,time,numpy as np,pygplates as pg
    from mythic.tectonics.source_clipping import triangulate_sources
    from mythic.tectonics.geometry import unit
    from mythic.tectonics.material_band import spherical_triangle_area,point_array
    from mythic.tectonics.material_audit import small_material_patches
    from design_material_audit import write_json
    root=Path(__file__).resolve().parents[1];cache=root/'.cache/u36_s10';data=root/'data/worlds/design_v2'
    with (cache/'sources.pkl').open('rb') as f:d=pickle.load(f)
    pieces=[z for z in d['pieces'] if z['uid']==448261]
    rot=pg.RotationModel(str(data/'history_v2.rot'));model=pg.TopologicalModel(str(cache/'network.gpml'),rot,topological_snapshot_cache_size=2)
    rows=[];start=time.perf_counter()
    for edge,radius in [(7.5,1.)]+[(e,r) for r in [.25,.0625] for e in [7.5,1.875,.9375,.46875]]:
        v,tri,src,diag=triangulate_sources(pieces,d['projection'],max_edge_km=edge,return_diagnostics=True)
        centers=unit(v[tri].sum(axis=1));pp,pt=small_material_patches(centers,radius)
        p0=spherical_triangle_area(pp,pt).reshape(-1,4).sum(axis=1);a0=abs(spherical_triangle_area(v,tri))
        span=model.reconstruct_geometry([tuple(z) for z in np.r_[v,pp]],35,time_increment=.25,deactivate_points=None)
        final=point_array(span.get_geometry_points(0.));area=abs(spherical_triangle_area(final[:len(v)],tri))
        jac=spherical_triangle_area(final[len(v):],pt).reshape(-1,4).sum(axis=1)/p0
        row=dict(edge_km=edge,patch_km=radius,triangle_count=len(tri),initial_volume_km3=float(a0.sum()*33),final_residual=float(np.sum(area*33/jac)/(a0.sum()*33)-1));rows.append(row)
        tag=f'legacy_uid448261_edge{edge:g}_patch{radius:g}'
        np.savez_compressed(cache/(tag+'.npz'),vertices=v,triangles=tri,patch_triangles=pt,points0=np.r_[v,pp],points_final=final,initial_thickness_km=33.)
    old=json.loads((data/'u36_s8_edge7.5_patch1.json').read_text(encoding='utf8'))
    c=next(z for z in old['source_trajectories'] if z['uid']==448261);oldvalue=c['volume_km3'][-1]/c['initial_volume_km3']-1
    report=dict(checkpoint='U36-S10',scope='single_source_diagnostic',method_class='procedural_constrained',mapping_class='interpolated',rows=rows,archived_S8_final_residual=oldvalue,reconstructed_minus_archived_residual=rows[0]['final_residual']-oldvalue,elapsed_s=time.perf_counter()-start,note='复用S8同一API与同一来源多边形；网络由同一端点重建、非S4 GPML字节相同；此组只独立核对末态几何，不能代替全域支撑')
    write_json(data/'u36_s10_uid_legacy.json',report)
    print('legacy UID diagnostic',report,flush=True)

def parity():
    from pathlib import Path
    import json,numpy as np,pygplates as pg
    from mythic.tectonics.material_band import reference_strip,spherical_triangle_area
    from mythic.tectonics.material_audit import small_material_patches
    from mythic.tectonics.geometry import unit,R_KM
    from mythic.tectonics.rotation_transport import IndexedRotationMap
    root=Path.cwd();data=root/'data/worlds/design_v2';cache=root/'.cache/u36_s10'
    source=json.loads((data/'u36_s4_material_band.json').read_text(encoding='utf8'))
    strip=reference_strip(*np.asarray(source['registered_endpoints_xyz']),200.,30.)
    centers=unit(strip['xyz'][strip['triangles']].sum(axis=1));points,tri=small_material_patches(centers,1.)
    a0=spherical_triangle_area(points,tri).reshape(-1,4).sum(axis=1)
    graph=next(z for z in json.loads((data/'u36_s9_mesh_audit.json').read_text(encoding='utf8'))['rows'] if z['case']=='real')['reference_connectivity']
    graph=[[n.replace('S9_real','S10_real') for n in row] for row in graph]
    rot=pg.RotationModel(str(data/'history_v2.rot'));model=pg.TopologicalModel(str(cache/'network.gpml'),rot,topological_snapshot_cache_size=2)
    old=np.load(root/'.cache/u36_s9/real_0_0.25.npz');pm=0.;jm=0.;initial=float(np.max(abs(points[:len(centers)]-old['candidate_xyz'][0])))
    for j,t in enumerate(old['times_ma']):
        pm=max(pm,float(np.max(np.linalg.norm(points[:len(centers)]-old['candidate_xyz'][j],axis=1))*R_KM))
        jac=spherical_triangle_area(points,tri).reshape(-1,4).sum(axis=1)/a0
        jm=max(jm,float(np.max(abs(jac/old['candidate_J'][j]-1))))
        net=model.topological_snapshot(float(t)).get_resolved_topologies()[0];mapper=IndexedRotationMap(net,graph)
        if t==0.:mapper.weights(points)
        else:points=mapper.rotation_map(points,{p:rot.get_rotation(float(t-.25),p,float(t)) for p in [301,401]})
    report=dict(checkpoint='U36-S10',method_class='procedural_constrained',source='.cache/u36_s9/real_0_0.25.npz',times_checked=len(old['times_ma']),material_centers=len(centers),initial_coordinate_component_difference=initial,maximum_center_position_difference_km=pm,maximum_J_relative_difference=jm,reference_connectivity_from_archived_S9=True,note='同一432中心及微斑块、同一步长、所有141时点核对；不是物理真值')
    (data/'u36_s10_S9_parity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(report)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['all','legacy','parity'],default='all')
    args=parser.parse_args()
    if args.part in ['all','legacy']:legacy()
    if args.part in ['all','parity']:parity()
