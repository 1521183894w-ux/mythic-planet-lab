"""S16 exploratory geometry audit: shared boundary versus S15 edge sampling."""
from pathlib import Path
import json, time, hashlib
from collections import Counter
import numpy as np
import shapely
import pygplates as pg
from shapely.geometry import Polygon
from mythic.tectonics.geometry import unit
from mythic.tectonics.source_clipping import LocalGnomonic
from mythic.tectonics.native_remap import spherical_moments
from mythic.tectonics.conforming_edges import shared_boundary
from mythic.tectonics.curved_quadrature import material_boundary
from mythic.tectonics.boundary_transport import GnomonicRotationMap

root=Path(__file__).resolve().parents[1]
cache=root/'.cache/u36_s16'; cache.mkdir(exist_ok=True)
source=root/'.cache/u36_s11/full_e30_l0_dt0.25_a0.npz'
a=np.load(source); vertices=a['vertices']; tri=a['triangles']
projection=LocalGnomonic(unit(vertices[tri].reshape(-1,3).mean(axis=0)))
pool, bounds, incidence, splits=shared_boundary(vertices,tri,projection.forward,segments=32)
old=np.array([material_boundary(vertices[t],32) for t in tri])
allpoints=np.concatenate([pool,old.reshape(-1,3)])
points,inverse=np.unique(allpoints,axis=0,return_inverse=True)
new_indices=[inverse[b] for b in bounds]
old_indices=inverse[len(pool):].reshape(len(tri),-1)
rotation=pg.RotationModel(str(root/'data/worlds/design_v2/history_v2.rot'))
model=pg.TopologicalModel(str(root/'.cache/u36_s11/network.gpml'),rotation,topological_snapshot_cache_size=2)
ref=json.loads((root/'data/worlds/design_v2/u36_s11_reference.json').read_text(encoding='utf8'))
start=time.perf_counter()
results=[];snapshots={}
def measure(name,index,t):
    polys=[Polygon(projection.forward(points[ix])) for ix in index]
    invalid=[i for i,p in enumerate(polys) if not p.is_valid or p.is_empty]
    if invalid: raise ValueError(f'{name}: invalid polygons {invalid[:20]}')
    merged=shapely.union_all(polys)
    areas=np.array([spherical_moments(p,projection)[0] for p in polys])
    union_area=spherical_moments(merged,projection)[0]
    pairs=shapely.STRtree(polys).query(polys,predicate='intersects')
    pairs=pairs[:,pairs[0]<pairs[1]]
    overlap=shapely.intersection(np.array(polys,dtype=object)[pairs[0]],np.array(polys,dtype=object)[pairs[1]])
    positive=np.flatnonzero(shapely.area(overlap)>0)
    pair_area=sum(spherical_moments(overlap[i],projection)[0] for i in positive)
    return polys,dict(time_ma=t,method=name,source_area_sum_km2=float(areas.sum()),
        union_area_km2=float(union_area),sum_minus_union_km2=float(areas.sum()-union_area),
        positive_pair_count=int(len(positive)),pair_overlap_km2=float(pair_area))
for step in range(141):
    t=35-step*.25
    mapper=GnomonicRotationMap(model.topological_snapshot(float(t)).get_resolved_topologies()[0],ref['reference_connectivity'])
    face,flags=mapper.support_diagnostics(points)
    counts={k:int(v.sum()) for k,v in flags.items()}
    if any(counts.values()): raise ValueError(f'unsupported at {t}: {counts}')
    if step in (0,60,120,140):
        old_polys,oldrow=measure('independent',old_indices,t)
        new_polys,newrow=measure('shared',new_indices,t)
        old_union=shapely.union_all(old_polys); new_union=shapely.union_all(new_polys)
        oldrow['union_symmetric_difference_km2']=float(spherical_moments(old_union.symmetric_difference(new_union),projection)[0])
        newrow['union_symmetric_difference_km2']=oldrow['union_symmetric_difference_km2']
        results.extend([oldrow,newrow]);snapshots[str(step)]=points.copy()
        print('MEASURE',oldrow,newrow,flush=True)
    if step==140:break
    rr={p:rotation.get_rotation(float(t-.25),p,float(t)) for p in [301,401]}
    points,_=mapper.step_with_jacobian(points,rr)
    if step%20==0:print('step',step,'points',len(points),'seconds',round(time.perf_counter()-start,1),flush=True)
report=dict(method_class='procedural_constrained',mapping_class='interpolated',
    method_version='1', base_commit='2e2b0f3',
    versions=dict(numpy=np.__version__,pygplates=pg.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string),
    references=['https://proj.org/en/stable/operations/projections/gnom.html','https://shapely.readthedocs.io/en/2.2.0/reference/shapely.STRtree.html'],
    method_name='shared_atomic_edge_local_diagnostic_v1',physical_input_resolution_km=60,
    units={'distance':'km','area':'km²','time':'Ma'},input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    sampling_segments_per_atomic_edge=32,recognition_tolerance_km=1e-9,
    split_count=len(splits),maximum_split_departure_km=max((s[-1] for s in splits),default=0),
    incidence_counts=dict(Counter(map(len,incidence.values()))),sample_count=len(points),
    snapshots=results,elapsed_s=time.perf_counter()-start,
    assumptions=['reference strip 200 km is user_override','same archived two-plate interpolated field and 0.25 Myr steps',
    'existing nodes within tolerance split initial great-circle edges; this changes the edge by the measured departure',
    'both representations are finite great-circle chord approximations, not physical truth'],
    limitations=['four sampled times do not prove continuous-time validity','does not resolve missing native hosts or crust type mismatch',
    'does not update global engine or B5','no physical width or ancient contact established'])
(cache/'geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
np.savez_compressed(cache/'snapshots.npz',vertices=vertices,triangles=tri,old_indices=old_indices,
    shared_offsets=np.r_[0,np.cumsum([len(x) for x in new_indices])],shared_indices=np.concatenate(new_indices),
    **snapshots)
print('FINISHED',report['elapsed_s'],flush=True)
