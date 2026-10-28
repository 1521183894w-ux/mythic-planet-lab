"""独立的积分几何/共同刚体负对照；不放宽既有失败断言。"""
import numpy as np
from scipy.spatial.transform import Rotation
from mythic.tectonics.geometry import lonlat_to_xyz, unit, R_KM
from mythic.tectonics.curved_quadrature import triangle_quadrature, material_boundary, polygon_fan_area


def triangle():
    return lonlat_to_xyz(np.array([-0.6, 0.7, 0.1]), np.array([-1.1, -1.0, 1.2]))


def test_spherical_octant_area_without_weight_normalization():
    # 三条互相正交的大圆围成八分球，面积有独立闭式值。
    _, weights = triangle_quadrature(np.eye(3), 32)
    np.testing.assert_allclose(weights.sum(), np.pi*R_KM**2/2, rtol=2e-12)


def test_linear_cartesian_moment_of_octant():
    points, weights = triangle_quadrature(np.eye(3), 32)
    # 球面八分区的x/y/z一阶矩均为pi*R²/4。
    np.testing.assert_allclose(weights @ points, np.full(3, np.pi*R_KM**2/4), rtol=2e-12)


def test_dense_boundary_and_quadrature_under_arbitrary_common_rotation():
    abc = triangle()
    rot = Rotation.from_rotvec([0.7, -0.3, 1.1])
    center = unit(abc.sum(axis=0))
    _, weights = triangle_quadrature(abc, 8)
    for count in [1, 4, 32, 128]:
        area, minimum = polygon_fan_area(rot.apply(material_boundary(abc, count)), rot.apply(center))
        np.testing.assert_allclose(area, weights.sum(), rtol=2e-12)
        assert minimum > 0
