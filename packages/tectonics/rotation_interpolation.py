"""U36-S9两侧刚体边界的点插值小试，不接入正式地壳引擎。

复用pyGPlates官方网络三角形、板块阶段旋转及SciPy SLERP。
method_class=interpolated；只改变给定点运动的插值，不求应力/流变。
仅支持两板块、无内部刚块、每个三角顶点能唯一匹配源边界的局部网络。
"""
from __future__ import annotations
import numpy as np
import pygplates as pg
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from .geometry import unit
from .material_band import point_array


def rotation_matrix(rotation):
    """直接旋转单位基，保留被角度/恒等判断截断的小旋转。"""
    return np.array([(rotation*pg.PointOnSphere(tuple(e))).to_xyz() for e in np.eye(3)]).T


class BoundaryRotationMap:
    """以官方连接关系及同一LAEA重心权重比较投影位置/有限旋转插值。"""
    def __init__(self, network, connectivity=None):
        if network.get_rigid_blocks():
            raise ValueError("本小试不支持内部刚体块")
        self.network = network
        boundary = network.get_resolved_boundary()
        self.center = np.asarray(boundary.get_boundary_centroid().to_xyz())
        pole = np.array([0., 0., 1.])
        if abs(self.center @ pole) > .9:
            pole = np.array([0., 1., 0.])
        self.east = unit(np.cross(pole, self.center))
        self.north = np.cross(self.center, self.east)
        triangulation = network.get_network_triangulation()
        vertices = list(triangulation.get_vertices())
        index = {v: i for i, v in enumerate(vertices)}
        self.xyz = point_array([v.position for v in vertices])
        self.triangles = np.asarray([
            [index[t.get_vertex(i)] for i in range(3)]
            for t in triangulation.get_triangles() if t.is_in_deforming_region])
        source = point_array(network.get_resolved_geometry_points())
        features = network.get_resolved_geometry_point_features()
        plate = np.array([f.get_reconstruction_plate_id() for f in features])
        distance, owner = cKDTree(source).query(self.xyz)
        if np.max(distance) > 1e-10:
            raise ValueError("存在不能匹配原边界身份的网络顶点；本小试拒绝填补")
        for x, p in zip(self.xyz, plate[owner]):
            coincident = np.linalg.norm(source-x, axis=1) < 1e-10
            if len(np.unique(plate[coincident])) != 1:
                raise ValueError("重合边界点的板块身份不唯一")
        self.names = [features[i].get_name() for i in owner]
        if len(set(self.names)) != len(self.names):
            raise ValueError("边界来源名称不唯一，不能保持拓扑身份")
        if connectivity is not None:
            lookup = {name: i for i, name in enumerate(self.names)}
            if set(lookup) != {name for row in connectivity for name in row}:
                raise ValueError("固定连接关系的来源顶点集合改变")
            self.triangles = np.array([[lookup[name] for name in row] for row in connectivity])
        self.connectivity = [[self.names[i] for i in row] for row in self.triangles]
        self.plate = plate[owner]
        self.plates = np.unique(self.plate)
        if len(self.plates) != 2:
            raise ValueError("本小试仅支持两个刚体板块")
        xy = self.project(self.xyz)[self.triangles]
        self.origin = xy[:, 2]
        self.inverse = np.linalg.inv(np.stack([
            xy[:, 0]-xy[:, 2], xy[:, 1]-xy[:, 2]], axis=-1))
        self.source_match_max_chord = float(np.max(distance))

    def project(self, points):
        x = unit(np.asarray(points))
        dot = x @ self.center
        if np.any(dot <= -1+1e-10):
            raise ValueError("LAEA对跖点不支持")
        return np.sqrt(2/(1+dot))[..., None]*np.stack([
            x @ self.east, x @ self.north], axis=-1)

    def unproject(self, xy):
        q = np.asarray(xy)
        r2 = np.sum(q*q, axis=-1)
        if np.any(r2 >= 4):
            raise ValueError("投影后的点越出LAEA定义域")
        return unit((1-r2/2)[..., None]*self.center
            + np.sqrt(1-r2/4)[..., None]*(
                q[..., :1]*self.east+q[..., 1:]*self.north))

    def weights(self, points):
        x = unit(np.asarray(points))
        if any(not self.network.get_resolved_boundary().is_point_in_polygon(tuple(p)) for p in x):
            raise ValueError("查询位于网络之外，不补值")
        xy = self.project(x)
        ww = np.einsum('tij,ptj->pti', self.inverse, xy[:, None, :]-self.origin)
        ww = np.concatenate([ww, 1-ww.sum(axis=-1, keepdims=True)], axis=-1)
        inside = (ww.min(axis=-1) >= -1e-9)
        if np.any(~inside.any(axis=1)):
            raise ValueError("查询没有受支持的官方三角形")
        face = inside.argmax(axis=1)
        selected_det = 1/np.linalg.det(self.inverse[face])
        if np.any(selected_det<=0):
            raise ValueError("查询选中了翻转的参考三角形")
        if np.any(np.sum(ww.min(axis=-1)>1e-9,axis=1)>1):
            raise ValueError("查询处有多个三角形内部重叠")
        self.minimum_selected_triangle_det = float(selected_det.min())
        weights = ww[np.arange(len(x)), face]
        # 仅消除边界舍入负权；不向网络外延拓。
        weights = np.maximum(weights, 0.)
        weights /= weights.sum(axis=1, keepdims=True)
        return self.triangles[face], weights

    def projected_map(self, points, rotations):
        """复现官方重心分支：旋转顶点→同一LAEA投影→插值→反投影。"""
        tri, weights = self.weights(points)
        moved = np.empty_like(self.xyz)
        for plate in self.plates:
            use = self.plate == plate
            moved[use] = self.xyz[use] @ rotation_matrix(rotations[int(plate)]).T
        xy = self.project(moved)
        return self.unproject(np.sum(xy[tri]*weights[..., None], axis=1))

    def rotation_map(self, points, rotations):
        """相同权重只插值两侧阶段旋转，再作用于原查询点。"""
        tri, weights = self.weights(points)
        fraction = np.sum(weights*(self.plate[tri] == self.plates[1]), axis=1)
        matrices = np.stack([rotation_matrix(rotations[int(p)]) for p in self.plates])
        slerp = Slerp([0., 1.], Rotation.from_matrix(matrices))
        return slerp(np.clip(fraction, 0., 1.)).apply(unit(points))
