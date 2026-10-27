"""原生球面Voronoi宿主与参考带的面积裁切；只读材料来源审计。

局部心射投影把大圆边映成直线，用Shapely/GEOS布尔运算与约束三角剖分。
所有物理面积回球面计算，绝不使用投影平面面积冒充km²。
"""
from __future__ import annotations
import numpy as np
import pygplates as pg
import shapely
from shapely.geometry import Polygon,GeometryCollection
from .geometry import R_KM,unit,km_from_chord
from .material_band import spherical_triangle_area
from .crust import CONT


class LocalGnomonic:
    def __init__(self,center):
        self.center=unit(center)
        self.east=unit(np.cross([0.,0.,1.],self.center));self.north=np.cross(self.center,self.east)
    def forward(self,xyz):
        x=np.asarray(xyz);z=x@self.center
        if np.any(z<=0):raise ValueError("裁切窗口须位于同一可见半球")
        return np.column_stack([x@self.east/z,x@self.north/z])*R_KM
    def inverse(self,xy):
        xy=np.asarray(xy)/R_KM
        return unit(self.center+xy[:,0,None]*self.east+xy[:,1,None]*self.north)


def polygons(geometry):
    if geometry.is_empty:return []
    if geometry.geom_type=="Polygon":return [geometry]
    return [p for g in geometry.geoms for p in polygons(g)] if hasattr(geometry,"geoms") else []


def spherical_geometry_area(geometry,projection):
    result=0.
    for poly in polygons(geometry):
        result+=pg.PolygonOnSphere([tuple(p) for p in projection.inverse(np.asarray(poly.exterior.coords)[:-1])]).get_area()*R_KM**2
        for ring in poly.interiors:
            result-=pg.PolygonOnSphere([tuple(p) for p in projection.inverse(np.asarray(ring.coords)[:-1])]).get_area()*R_KM**2
    return result


def voronoi_vertices(mesh,nodes):
    """从Delaunay三角形的球面外心取得原生Voronoi边，不用半径圆代替宿主。"""
    nodes=np.asarray(nodes,np.int64)
    faces=mesh.simplices[np.any(np.isin(mesh.simplices,nodes),axis=1)]
    abc=mesh.xyz[faces]
    centers=unit(np.cross(abc[:,1]-abc[:,0],abc[:,2]-abc[:,0]))
    centers*=np.where(np.sum(centers*abc[:,0],axis=1)>=0,1.,-1.)[:,None]
    result={}
    for node in np.unique(nodes):
        v=centers[np.any(faces==node,axis=1)]
        x=mesh.xyz[node];e=unit(np.cross([0.,0.,1.],x));n=np.cross(x,e)
        order=np.argsort(np.arctan2(v@n,v@e))
        result[int(node)]=v[order]
    return result


def clip_native_sources(eng,boundary_xyz,search_padding_km=180.):
    """按native_cover相同优先级划分重叠宿主，保留无覆盖和非陆壳。

    返回每个相交宿主的实际球面面积与体积；不改变宿主属性或引擎面积。
    """
    boundary=np.asarray(boundary_xyz)
    projection=LocalGnomonic(unit(boundary.sum(0)))
    target=Polygon(projection.forward(boundary))
    if not target.is_valid:raise ValueError("参考带多边形无效")
    pos=eng.positions();c=eng.c
    radius=max(km_from_chord(np.linalg.norm(boundary-projection.center,axis=1)))
    distance=km_from_chord(np.linalg.norm(pos-projection.center,axis=1))
    candidates=np.flatnonzero(eng.alive&(distance<radius+search_padding_km))
    cells=voronoi_vertices(eng.m,c.node[candidates])
    score=c.ctype[candidates]*1e6+c.T[candidates]*1e3-c.age[candidates]
    order=np.lexsort((c.uid[candidates],c.plate[candidates],-score))
    claimed=GeometryCollection();pieces=[];full_errors=[]
    for index in candidates[order]:
        vertices=cells[int(c.node[index])]@eng.R[int(c.plate[index])].T
        full=Polygon(projection.forward(vertices))
        hit=full.intersection(target)
        if hit.is_empty:continue
        full_area=spherical_geometry_area(full,projection)
        full_errors.append(full_area-eng.area[index])
        owned=hit.difference(claimed)
        claimed=claimed.union(hit)
        if owned.is_empty:continue
        area=spherical_geometry_area(owned,projection)
        pieces.append(dict(uid=int(c.uid[index]),plate=int(c.plate[index]),ctype=int(c.ctype[index]),origin=int(c.origin[index]),
            T_km=float(c.T[index]),area_km2=float(area),volume_km3=float(area*c.T[index]) if c.ctype[index]==CONT else None,geometry=owned))
    gap=target.difference(claimed)
    if not gap.is_empty:pieces.append(dict(uid=-1,plate=-1,ctype=-1,origin=-1,T_km=np.nan,
        area_km2=float(spherical_geometry_area(gap,projection)),volume_km3=None,geometry=gap))
    area=spherical_geometry_area(target,projection)
    summary=dict(target_area_km2=float(area),source_area_sum_km2=sum(p["area_km2"] for p in pieces),
        uncovered_area_km2=sum(p["area_km2"] for p in pieces if p["uid"]<0),
        noncont_area_km2=sum(p["area_km2"] for p in pieces if p["uid"]>=0 and p["ctype"]!=CONT),
        known_cont_area_km2=sum(p["area_km2"] for p in pieces if p["ctype"]==CONT),
        known_cont_volume_km3=sum(p["volume_km3"] for p in pieces if p["ctype"]==CONT),
        native_host_count=sum(p["uid"]>=0 for p in pieces),candidate_count=len(candidates),
        max_full_host_area_error_km2=max(map(abs,full_errors),default=0.),
        method_class="procedural_constrained",geometry_backend=f"Shapely {shapely.__version__}/GEOS {shapely.geos_version_string}",
        boundary_note="参考带本身仍是假设；精确为指定离散大圆多边形的浮点布尔裁切，非真实断层恢复")
    summary["area_partition_residual_km2"]=summary["source_area_sum_km2"]-area
    return pieces,projection,summary


def triangulate_sources(pieces,projection,max_edge_km=15.,return_diagnostics=False):
    """约束三角剖分及大圆中点细分，源UID跟随三角形，不跨源平均厚度。"""
    triangles=[];sources=[]
    for k,piece in enumerate(pieces):
        for poly in polygons(piece["geometry"]):
            for t in shapely.constrained_delaunay_triangles(poly).geoms:
                triangles.append(projection.inverse(np.asarray(t.exterior.coords)[:3]));sources.append(k)
    xyz=np.asarray(triangles);source=np.asarray(sources,np.int64)
    for _ in range(10):
        edge=km_from_chord(np.linalg.norm(xyz-np.roll(xyz,1,axis=1),axis=2)).max(1)
        split=edge>max_edge_km
        if not split.any():break
        a,b,c=xyz[split].transpose(1,0,2);ab,bc,ca=unit(a+b),unit(b+c),unit(c+a)
        refined=np.concatenate([np.stack(v,axis=1) for v in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
        xyz=np.r_[xyz[~split],refined];source=np.r_[source[~split],np.tile(source[split],4)]
    else:raise RuntimeError("裁切单元细分未收敛")
    vertices,inverse=np.unique(xyz.reshape(-1,3),axis=0,return_inverse=True)
    tri=inverse.reshape(-1,3)
    area=abs(spherical_triangle_area(vertices,tri))
    # GEOS布尔交点可产生近重合顶点；低于0.001m²的片不能稳定求J。
    # 不修改来源多边形或其面积账，明确另报未作柱变形审计的面积。
    tiny=area<=1e-9
    diagnostics=dict(raw_triangle_count=len(tri),numerically_degenerate_count=int(tiny.sum()),
        degenerate_area_km2=float(area[tiny].sum()),area_precision_floor_km2=1e-9,
        note="浮点碎片只从J比值审计中排除，原裁切面积/体积账保持；非质量指标阈值")
    tri,source=tri[~tiny],source[~tiny]
    if return_diagnostics:return vertices,tri,source,diagnostics
    return vertices,tri,source
