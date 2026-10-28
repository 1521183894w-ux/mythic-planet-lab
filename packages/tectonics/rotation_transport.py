"""S10局部审计加速：复用S9权重，空间索引只减少候选面，不改变接受条件。"""
from __future__ import annotations
import numpy as np
import shapely
from .rotation_interpolation import BoundaryRotationMap
from .geometry import unit

class IndexedRotationMap(BoundaryRotationMap):
    """同一固定来源三角形的索引查询；不向边界外延拓，不接全球引擎。"""
    def __init__(self,network,connectivity=None):
        super().__init__(network,connectivity)
        xy=self.project(self.xyz)[self.triangles]
        # 包围盒只作候选筛选。余量大于原权重1e-9容差的最大外延。
        pad=1e-8
        lo=xy.min(axis=1)-pad;hi=xy.max(axis=1)+pad
        self.tree=shapely.STRtree(shapely.box(lo[:,0],lo[:,1],hi[:,0],hi[:,1]))
        self.determinants=1/np.linalg.det(self.inverse)
    def support_diagnostics(self,points):
        x=unit(np.asarray(points));xy=self.project(x)
        pi,ti=self.tree.query(shapely.points(xy))
        w2=np.einsum('pij,pj->pi',self.inverse[ti],xy[pi]-self.origin[ti])
        w=np.c_[w2,1-w2.sum(axis=1)]
        inside=w.min(axis=1)>=-1e-9
        face=np.full(len(x),len(self.triangles),dtype=np.int64)
        np.minimum.at(face,pi[inside],ti[inside])
        supported=face<len(self.triangles)
        strict=np.bincount(pi[w.min(axis=1)>1e-9],minlength=len(x))
        safe=np.minimum(face,len(self.triangles)-1)
        outside=np.array([not self.network.get_resolved_boundary().is_point_in_polygon(tuple(p)) for p in x])
        inverted=supported&(self.determinants[safe]<=0)
        return face,dict(outside=outside,unsupported=~supported,inverted=inverted,overlapping=strict>1)
    def weights(self,points):
        x=unit(np.asarray(points));face,flags=self.support_diagnostics(x)
        for key,message in [('outside','查询位于网络之外，不补值'),('unsupported','查询没有受支持的官方三角形'),('inverted','查询选中了翻转的参考三角形'),('overlapping','查询处有多个三角形内部重叠')]:
            if flags[key].any():raise ValueError(message)
        xy=self.project(x)
        w2=np.einsum('pij,pj->pi',self.inverse[face],xy-self.origin[face])
        weights=np.c_[w2,1-w2.sum(axis=1)]
        weights=np.maximum(weights,0.);weights/=weights.sum(axis=1,keepdims=True)
        self.minimum_selected_triangle_det=float(self.determinants[face].min())
        return self.triangles[face],weights
