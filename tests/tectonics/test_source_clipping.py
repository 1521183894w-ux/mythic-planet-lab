"""几何裁切须与独立原生查询一致，含重叠、非陆壳和未覆盖区。"""
import numpy as np
import shapely
from mythic.tectonics.crust import Crust,CONT,OCEAN,Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.geometry import lonlat_to_xyz,omega_rotmat
from mythic.tectonics.material_band import reference_strip,spherical_triangle_area
from mythic.tectonics.sampling import native_cover
from mythic.tectonics.source_clipping import clip_native_sources,triangulate_sources


def test_priority_partition_and_exact_cell_areas_match_native_cover():
    mesh=load_mesh(3);n=mesh.n;c=Crust(2*n)
    c.node[:]=np.tile(np.arange(n),2);c.plate[:n]=1;c.plate[n:]=2
    c.ctype[:]=CONT;c.T[:n]=35;c.T[n:]=42
    # 同地块的两份宿主允许重叠；明确删去部分范围以产生空隙。
    eng=Engine(mesh,c,Params(),35,{1:np.zeros(3),2:np.zeros(3)})
    eng.R[2]=omega_rotmat(np.array([.2,-.3,.1]),.1)
    eng.alive[n:]=mesh.lon>0
    eng.alive[:n]&=~((abs(mesh.lon)<8)&(mesh.lat>5)&(mesh.lat<12))
    eng.c.ctype[n:][mesh.lat<-5]=OCEAN
    eng._rebuild()
    s=reference_strip(lonlat_to_xyz(0,-15),lonlat_to_xyz(0,15),1500,80)
    pieces,projection,report=clip_native_sources(eng,s['xyz'][s['boundary']],search_padding_km=2*mesh.h_km)
    assert report['max_full_host_area_error_km2']<1e-4
    assert abs(report['area_partition_residual_km2'])<1e-3
    rng=np.random.default_rng(3605)
    b=projection.forward(s['xyz'][s['boundary']]);p=rng.uniform(b.min(0),b.max(0),(5000,2))
    target=shapely.Polygon(b);p=p[shapely.contains_xy(target,p[:,0],p[:,1])]
    index=native_cover(eng,projection.inverse(p));expected=np.where(index>=0,eng.c.uid[np.maximum(index,0)],-1)
    actual=np.full(len(p),-2)
    for piece in pieces:actual[shapely.contains_xy(piece['geometry'],p[:,0],p[:,1])]=piece['uid']
    np.testing.assert_array_equal(actual,expected)
    v,tri,source=triangulate_sources(pieces,projection,max_edge_km=100)
    area=abs(spherical_triangle_area(v,tri))
    assert abs(area.sum()-report['target_area_km2'])<1e-3
    known=np.array([pieces[k]['ctype']==CONT for k in source])
    h=np.array([pieces[k]['T_km'] for k in source])
    assert abs(np.sum(area[known]*h[known])-report['known_cont_volume_km3'])<.1
