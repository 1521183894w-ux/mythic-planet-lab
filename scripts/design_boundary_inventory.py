"""S11保留S8有符号账差，并分别核对真实几何和有限单元库存。"""
from pathlib import Path
import json,pickle,hashlib
import numpy as np
import pygplates as pg
import shapely
from mythic.tectonics.crust import CONT
from mythic.tectonics.material_band import point_array
from mythic.tectonics.source_clipping import spherical_geometry_area
from mythic.tectonics.boundary_transport import GnomonicRotationMap
from mythic.tectonics.geometry import R_KM
r=Path.cwd();data=r/'data/worlds/design_v2';cache=r/'.cache/u36_s11'
with (r/'.cache/u36_s10/sources.pkl').open('rb') as f:d=pickle.load(f)
old=json.loads((data/'u36_s8_fixed_domain.json').read_text(encoding='utf8'))['domain']
full=np.load(cache/'full_e30_l0_dt0.25_a0.npz');outer=np.load(cache/'outer_e30_l0_dt0.25_a0.npz')
fv=dict(zip(full['uids'].tolist(),full['V0_km3'].tolist()))
ov=dict(zip(outer['uids'].tolist(),outer['V0_km3'].tolist()))
ledger={p['uid']:p for p in old['outer_source_inventory']};rows=[]
for p in d['original_pieces']:
    if p['ctype']!=CONT:continue
    geom=p['geometry'];area=spherical_geometry_area(geom,d['projection'])
    tail=geom.difference(d['domain']);outer_area=spherical_geometry_area(tail,d['projection'])
    rows.append(dict(uid=p['uid'],full_geometry_volume_km3=area*p['T_km'],
        full_finite_volume_km3=fv.get(p['uid'],0.),
        outer_geometry_volume_km3=outer_area*p['T_km'],outer_finite_volume_km3=ov.get(p['uid'],0.),
        original_S8_signed_outer_inventory_km3=ledger[p['uid']]['reserved_outer_volume_km3']))
rot=pg.RotationModel(str(data/'history_v2.rot'))
net=pg.TopologicalModel(str(cache/'network.gpml'),rot).topological_snapshot(35.).get_resolved_topologies()[0]
m=GnomonicRotationMap(net)
target=shapely.Polygon(m.project(d['projection'].inverse(np.asarray(d['target'].exterior.coords)[:-1]))*R_KM)
boundary=shapely.Polygon(m.project(point_array(net.get_resolved_boundary().get_points()))*R_KM)
points=shapely.points(m.project(full['vertices'])*R_KM)
record=dict(checkpoint='U36-S11',method_class='procedural_constrained',rows=rows,
    source_uid_count=len(fv),original_S8_negative_signed_inventory_count=old['negative_reserved_uid_count'],
    original_S8_minimum_signed_inventory_km3=old['minimum_reserved_signed_difference_km3'],
    original_S8_signed_inventory_preserved=True,gnomonic_planar_target_outside_area_km2=float(target.difference(boundary).area),
    maximum_original_vertex_distance_outside_network_m=float(shapely.distance(points,boundary).max()*1000),
    full_geometry_minus_finite_initial_km3=sum(z['full_geometry_volume_km3']-z['full_finite_volume_km3'] for z in rows),
    outer_geometry_minus_finite_initial_km3=sum(z['outer_geometry_volume_km3']-z['outer_finite_volume_km3'] for z in rows),
    notes=['S8原库存有符号差逐UID原样保留，不归零或当作物理负材料；本轮正几何外圈另列，不冒充相同逐项账值',
        '有限单元中低于既有面积精度底的碎片继续单列，不改变旧裁切账',
        '心射平面面积只诊断几何差，不替代球面材料面积；极薄片球面面积可能受浮点误差底影响'])
(data/'u36_s11_source_inventory.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k not in ['rows','notes']},ensure_ascii=False))
