"""U36-S12：指定分片运动场的求积诊断，不接正式地壳引擎。"""
from __future__ import annotations
import numpy as np
import shapely
from .geometry import unit
from .material_band import spherical_triangle_area


def crossing_reference_faces(mapper, vertices, triangles):
    """球面三角形是否越过其中心所处的分片；边上容差沿用原查询。"""
    abc=np.asarray(vertices)[triangles]
    centers=unit(abc.sum(axis=1))
    face,flags=mapper.support_diagnostics(centers)
    if any(z.any() for z in flags.values()):
        mapper.weights(centers)
    xy=mapper.project(abc.reshape(-1,3)).reshape(-1,3,2)
    w2=np.einsum('tij,tkj->tki',mapper.inverse[face],xy-mapper.origin[face,None,:])
    weights=np.concatenate([w2,1-w2.sum(axis=-1,keepdims=True)],axis=-1)
    return weights.min(axis=(1,2)) < -1e-9


def split_on_reference_faces(mapper, vertices, triangles):
    """仅改变初始求积分区：来源有限单元按原场的面裁切，不改运动场。

    GEOS约束剖分保留交片边界；所有物理面积回球面计算。低于既有
    1e-9 km²面积精度底的子片只留在有符号库存差账，不求不稳定J。
    """
    xy=mapper.project(mapper.xyz)[mapper.triangles]
    faces=shapely.polygons(xy)
    tree=shapely.STRtree(faces)
    result=[];parents=[];discard=[];area_before=abs(spherical_triangle_area(vertices,triangles))
    for k,abc in enumerate(np.asarray(vertices)[triangles]):
        cell=shapely.Polygon(mapper.project(abc))
        hits=tree.query(cell,predicate='intersects')
        covered=shapely.union_all([cell.intersection(faces[i]) for i in hits])
        if cell.difference(covered).area > max(cell.area*1e-9,1e-18):
            raise ValueError('初始材料单元没有完整参考面支撑，不外插')
        for i in hits:
            cut=cell.intersection(faces[i])
            for poly in shapely.constrained_delaunay_triangles(cut).geoms:
                q=mapper.unproject(np.asarray(poly.exterior.coords)[:3])
                area=float(abs(spherical_triangle_area(q,np.array([[0,1,2]])))[0])
                if area<=1e-9:
                    discard.append((k,area));continue
                result.append(q);parents.append(k)
    result=np.asarray(result)
    if not len(result):raise ValueError('没有可审计的初始交片')
    v,inv=np.unique(result.reshape(-1,3),axis=0,return_inverse=True)
    tri=inv.reshape(-1,3);parents=np.asarray(parents,dtype=int)
    signed=spherical_triangle_area(v,tri)
    tri[signed<0]=tri[signed<0][:,[0,2,1]]
    area_after=np.bincount(parents,weights=abs(signed),minlength=len(triangles))
    return v,tri,parents,dict(initial_area_difference_by_parent_km2=(area_after-area_before).tolist(),
        maximum_absolute_parent_area_difference_km2=float(abs(area_after-area_before).max()),
        excluded_tiny_subfragments=[dict(parent=int(k),area_km2=float(a)) for k,a in discard],
        tiny_subfragment_area_km2=float(sum(a for _,a in discard)),
        precision_area_floor_km2=1e-9)


def uniform_refine(vertices,triangles,parent):
    abc=vertices[triangles];a,b,c=abc.transpose(1,0,2)
    ab,bc,ca=unit(a+b),unit(b+c),unit(c+a)
    refined=np.concatenate([np.stack(z,axis=1) for z in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
    keep=len(vertices)
    v,inv=np.unique(np.r_[vertices,refined.reshape(-1,3)],axis=0,return_inverse=True)
    return v,inv[keep:].reshape(-1,3),np.tile(parent,4)