"""U36-S11 心射坐标的两板块有限旋转插值小试，不接正式引擎。

method_class=interpolated。复用pyGPlates边界和来源身份、SciPy SLERP；
解析面积因子是指定插值场的导数，不是应力/流变求解或物理带宽反演。
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import unit
from .rotation_transport import IndexedRotationMap
from .rotation_interpolation import rotation_matrix


class GnomonicRotationMap(IndexedRotationMap):
    """大圆边映成直线；保持原连接，不补网络外的查询。"""

    def project(self, points):
        x = unit(np.asarray(points))
        z = x @ self.center
        if np.any(z <= 1e-10):
            raise ValueError("心射投影仅支持中心可见半球")
        return np.stack([x @ self.east, x @ self.north], axis=-1)/z[..., None]

    def unproject(self, xy):
        q = np.asarray(xy)
        return unit(self.center + q[..., :1]*self.east + q[..., 1:]*self.north)

    def step_with_jacobian(self, points, rotations):
        """在选中面的单侧导数上返回F(x)与球面面积因子。

        f为第二板块的重心权重总和，F=R0 exp(f[omega]x)x；
        切平面的矩阵行列式引理给J=1+grad_S(f) dot (omega cross x)。
        面边处导数分侧；有限来源体积仍用独立面积检查，不能由此自证闭合。
        """
        x = unit(np.asarray(points))
        tri, weights = self.weights(x)
        face, _ = self.support_diagnostics(x)
        flags = (self.plate[tri] == self.plates[1]).astype(float)
        fraction = np.sum(weights*flags, axis=1)
        gxy = np.einsum('pi,pij->pj',
            flags[:, :2]-flags[:, 2, None], self.inverse[face])
        xy = self.project(x)
        z = x @ self.center
        gradient = (gxy[:, :1]*self.east + gxy[:, 1:]*self.north
            - np.sum(gxy*xy, axis=1)[:, None]*self.center)/z[:, None]
        matrices = np.stack([rotation_matrix(rotations[int(p)]) for p in self.plates])
        relative = Rotation.from_matrix(matrices[0].T @ matrices[1]).as_rotvec()
        jac = 1 + np.einsum('ij,ij->i', gradient, np.cross(relative, x))
        if np.any(~np.isfinite(jac)) or np.any(jac <= 0):
            raise ValueError("候选插值场出现非正面积导数")
        moved = Slerp([0., 1.], Rotation.from_matrix(matrices))(
            np.clip(fraction, 0., 1.)).apply(x)
        return moved, jac


def aligned_network(strip, rotation_model, initial_time=35., plates=(301,401), tag="S11"):
    """原材料范围内逐边采用同一轨道节点，保留原400km端部余量。

    所有节点仍在给定的200km宽度两侧；材料范围内替换原网络采样、连接确实改变，
    因此这是新插值场，不冒称与S9/S10同一场，也不移动来源多边形。
    """
    import pygplates as pg
    from .material_band import build_network, point_array
    _, old = build_network(strip, rotation_model, initial_time, plates=plates, tag=tag+"_base")
    sides = []
    for side, plate in enumerate(plates):
        old_points = [np.asarray((rotation_model.get_rotation(initial_time,plate)*f.get_geometry()).to_xyz())
            for f in old[:-1] if f.get_reconstruction_plate_id()==plate]
        rail = strip["xyz"][strip["grid"][:,0 if side==0 else -1]]
        a = strip["xyz"][strip["grid"][0,strip["grid"].shape[1]//2]]
        b = strip["xyz"][strip["grid"][-1,strip["grid"].shape[1]//2]]
        tangent = unit(b-a*(a@b))
        old_points = np.asarray(old_points)
        along = np.arctan2(old_points@tangent,old_points@a)
        length_angle = np.arctan2(b@tangent,b@a)
        # 原材料范围内必须逐边相同：插入中间小圆点会改变大圆折线。
        guard = old_points[(along < -1e-12)|(along > length_angle+1e-12)]
        points = np.r_[guard,rail]
        order = np.argsort(np.arctan2(points@tangent,points@a))
        sides.append(points[order if side==0 else order[::-1]])
    features,sections=[],[]
    for side,points in enumerate(sides):
        plate=plates[side]
        for j,p in enumerate(points):
            present=rotation_model.get_rotation(0.,plate,initial_time)*pg.PointOnSphere(tuple(p))
            f=pg.Feature.create_reconstructable_feature(
                pg.FeatureType.gpml_unclassified_feature,present,
                name=f"{tag}_side{side}_{j:05d}",reconstruction_plate_id=plate,
                valid_time=(initial_time+1.,0.))
            features.append(f)
            sections.append(pg.GpmlTopologicalSection.create(f,topological_geometry_type=pg.GpmlTopologicalNetwork))
    network=pg.Feature.create_topological_network_feature(
        pg.GpmlTopologicalNetwork(sections),name=tag,valid_time=(initial_time+1.,0.))
    return features+[network]


def constrained_connectivity(network):
    """由GEOS约束剖分给定大圆多边形；固定节点身份，不继承LAEA的越界面。"""
    import shapely
    from scipy.spatial import cKDTree
    from .material_band import point_array
    mapper=GnomonicRotationMap(network)
    xy=mapper.project(mapper.xyz)
    boundary=mapper.project(point_array(network.get_resolved_boundary().get_points()))
    polygon=shapely.Polygon(boundary)
    if not polygon.is_valid:
        raise ValueError("边界多边形无效")
    lookup=cKDTree(xy)
    graph=[]
    for poly in shapely.constrained_delaunay_triangles(polygon).geoms:
        q=np.asarray(poly.exterior.coords)[:3]
        distance,index=lookup.query(q)
        if distance.max()>1e-11:
            raise ValueError("约束剖分产生未知来源顶点")
        if (q[1,0]-q[0,0])*(q[2,1]-q[0,1])-(q[1,1]-q[0,1])*(q[2,0]-q[0,0])<0:
            index=index[[0,2,1]]
        graph.append([mapper.names[i] for i in index])
    GnomonicRotationMap(network,graph)
    return graph
