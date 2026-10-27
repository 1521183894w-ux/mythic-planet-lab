"""解析运动学独立几何检验；面积变化与走滑剪切分别验证。"""
import numpy as np
import pytest
from mythic.tectonics.geometry import R_KM, omega_rotmat
from mythic.tectonics.deformation_kinematics import affine_band, boundary_jump, integrate_band


@pytest.mark.parametrize("width", [100., 200., 400.])
@pytest.mark.parametrize("slip,angle", [(0., 0.), (200., 0.), (200., 1.3)])
def test_deformed_polygon_area_and_column_volume(width, slip, angle):
    r = affine_band(width, 200., slip, angle)
    polygon = np.array([[-width/2, 0.], [width/2, 0.], [width/2, 1000.], [-width/2, 1000.]])
    p = polygon @ r["F"].T
    area = abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1) - p[:, 1]*np.roll(p[:, 0], -1)))/2
    assert area == pytest.approx((width+200)*1000)
    assert area * (35/r["J"]) == pytest.approx(width*1000*35)


@pytest.mark.parametrize("dt", [.5, 1., 3., 5.])
def test_incremental_stretch_uses_current_width(dt):
    r = integrate_band(200., 35., 10., 20., dt)
    assert r["width_final_km"] == pytest.approx(400.)
    assert r["thickness_final_km"] == pytest.approx(17.5)
    assert r["volume_per_length_km2"] == pytest.approx(7000.)


def test_pure_slip_has_shear_but_no_area_dilation():
    r = affine_band(200., 0., 200.)
    assert r["J"] == pytest.approx(1.)
    assert not np.allclose(r["F"].T @ r["F"], np.eye(2))


def test_rigid_rotation_has_neither_area_nor_material_strain():
    r = affine_band(200., rotation_rad=1.2)
    assert r["J"] == pytest.approx(1.)
    np.testing.assert_allclose(r["F"].T @ r["F"], np.eye(2), atol=1e-14)


def test_boundary_normal_rejects_slip_and_is_frame_invariant():
    x, n = np.array([0., 1., 0.]), np.array([-1., 0., 0.])
    a, b = np.array([0., 0., 0.]), np.array([1., 0., 0.])/R_KM
    vn, vt = boundary_jump(x, n, a, b)
    assert vn == pytest.approx(0.)
    assert abs(vt) == pytest.approx(1.)
    # 同一纯走滑，最近点错开时旧连线判据却为正。
    separation = np.array([-1., 0., 1.])
    assert np.cross(b-a, x) @ separation > 0.
    q = omega_rotmat(np.array([.3, -.2, .1]), 2.)
    common = np.array([.02, .01, -.03])
    rotated = boundary_jump(q@x, q@n, q@a+common, q@b+common)
    np.testing.assert_allclose(rotated, [vn, vt], atol=1e-12)


def test_compression_is_signed_and_invalid_width_is_rejected():
    vn, _ = boundary_jump([0.,1.,0.], [-1.,0.,0.], [0.,0.,0.], [0.,0.,-1/R_KM])
    assert vn == pytest.approx(-1.)
    assert affine_band(200., -50.)["J"] == pytest.approx(.75)
    for width, opening in [(0., 1.), (100., -100.)]:
        with pytest.raises(ValueError):
            affine_band(width, opening)
