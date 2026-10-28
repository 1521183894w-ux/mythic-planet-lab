"""U36-S11 起始完整外圈支撑诊断；先出图，不修改S10数据。"""
from pathlib import Path
import json,pickle,time
import numpy as np
import pygplates as pg
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.tectonics.geometry import unit,R_KM
from mythic.tectonics.source_clipping import triangulate_sources
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.rotation_transport import IndexedRotationMap
from mythic.tectonics.boundary_transport import GnomonicRotationMap

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/u36_s11';CACHE.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/worlds/design_v2'
FIG=ROOT/'figures/design/v2/forward/u36_s11';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    start=time.perf_counter()
    with (ROOT/'.cache/u36_s10/sources.pkl').open('rb') as stream:d=pickle.load(stream)
    pieces=[]
    for p in d['original_pieces']:
        hit=p['geometry'].difference(d['domain'])
        if not hit.is_empty:pieces.append(dict(p,geometry=hit))
    v,tri,src,fragments=triangulate_sources(pieces,d['projection'],30.,True)
    centers=unit(v[tri].sum(axis=1));pp,pt=small_material_patches(centers,1.)
    points=np.r_[v,pp]
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    model=pg.TopologicalModel(str(ROOT/'.cache/u36_s10/network.gpml'),rot)
    net=model.topological_snapshot(35.).get_resolved_topologies()[0]
    old=json.loads((DATA/'u36_s9_mesh_audit.json').read_text(encoding='utf8'))
    graph=[[n.replace('S9_real','S10_real') for n in row]
        for row in next(z for z in old['rows'] if z['case']=='real')['reference_connectivity']]
    kinds=np.r_[np.zeros(len(v),int),np.ones(len(tri),int),np.full(4*len(tri),2,int)]
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,cls,name in zip(axes,[IndexedRotationMap,GnomonicRotationMap],['S10 LAEA','S11 心射，同一来源连接']):
        mapper=cls(net,graph);_,flags=mapper.support_diagnostics(points)
        xy=mapper.project(points)*R_KM
        mesh=mapper.project(mapper.xyz)*R_KM
        ax.triplot(mesh[:,0],mesh[:,1],mapper.triangles,color='.7',lw=.35)
        ax.scatter(xy[:,0],xy[:,1],s=3,c='.25',alpha=.2)
        bad=np.logical_or.reduce(list(flags.values()))
        ax.scatter(xy[bad,0],xy[bad,1],s=14,c='crimson',label='被拒绝查询')
        row=dict(projection=name,point_count=len(points),query_counts={k:int(z.sum()) for k,z in flags.items()},
            by_role={n:{k:int((z&(kinds==role)).sum()) for k,z in flags.items()}
                for role,n in enumerate(['source_vertices','source_centers','patch_ring'])},
            nonpositive_projected_triangles=int((mapper.determinants<=0).sum()))
        rows.append(row)
        ax.set_title(name+'\n'+str(row['query_counts']),fontsize=10)
        ax.set_xlabel('局部东向 / km');ax.set_ylabel('局部北向 / km');ax.set_aspect('equal');ax.legend()
    fig.suptitle('完整外圈起始35Ma：同一网络、同一来源、同一查询点')
    fig.text(.015,.015,'7cb67e9+U36-S11 · design_v2 · 原L7约60km；method_class: interpolated / procedural_constrained；图 visual_only。200km假设宽度未改。',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(FIG/'112_initial_boundary_support.png',dpi=160);plt.close(fig)
    report=dict(checkpoint='U36-S11',method_class='procedural_constrained',mapping_class='interpolated',
        same_S9_connectivity=True,edge_km=30,patch_km=1,rows=rows,fragments=fragments,elapsed_s=time.perf_counter()-start)
    write(DATA/'u36_s11_initial_support.json',report)
    np.savez_compressed(CACHE/'initial_support.npz',points=points,vertices=v,triangles=tri,source_index=src,kinds=kinds)
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
