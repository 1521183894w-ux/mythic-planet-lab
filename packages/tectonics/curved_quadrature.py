"""U36-S13：用材料曲边与面内求积分开有限单元误差；不接正式引擎。"""
from __future__ import annotations
import numpy as np
from scipy.special import roots_legendre
from .geometry import unit, R_KM


def triangle_quadrature(abc, order):
    """局部心射三角域上的Duffy×Gauss积分，权重为球面km²，不归一化。"""
    abc = unit(np.asarray(abc))
    center = unit(abc.sum(axis=0))
    east = unit(abc[1] - center * (abc[1] @ center))
    north = np.cross(center, east)
    xy = np.stack([abc @ east, abc @ north], axis=1) / (abc @ center)[:, None]
    nodes, weights = roots_legendre(order)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    w = np.outer(weights / 2, weights / 2)
    a, b, c = xy
    pos = a + u[..., None] * (b-a) + ((1-u)*v)[..., None] * (c-a)
    determinant = abs(np.linalg.det(np.stack([b-a, c-a])))
    area_weights = w * (1-u) * determinant * R_KM**2 / (1+(pos*pos).sum(axis=-1))**1.5
    points = unit(center + pos[..., :1]*east + pos[..., 1:]*north)
    return points.reshape(-1, 3), area_weights.ravel()


def material_boundary(abc, segments):
    """初始大圆边按归一化线性混合参数取样；不重复末端，之后随材料前进。"""
    abc = unit(np.asarray(abc))
    fraction = np.arange(segments) / segments
    return np.concatenate([unit((1-fraction[:, None])*a + fraction[:, None]*b)
                           for a, b in zip(abc, np.roll(abc, -1, axis=0))])


def polygon_fan_area(boundary, interior):
    """有向球面扇形面积；差向量行列式减小小单元的消减误差。"""
    p = unit(np.asarray(boundary))
    center = unit(np.asarray(interior))
    q = np.roll(p, -1, axis=0)
    numerator = np.einsum("i,ji->j", center, np.cross(p-center, q-center))
    denominator = 1 + p @ center + q @ center + np.einsum("ij,ij->i", p, q)
    signed = 2 * np.arctan2(numerator, denominator) * R_KM**2
    if np.any(signed <= 0):
        raise ValueError("材料曲边出现非正扇形；不以绝对值掩盖")
    return float(signed.sum()), float(signed.min())
