"""运动学框架的独立几何检验；不把H=H0/J的定义性闭合当物理验证。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, omega_rotmat
from mythic.tectonics.material_band import (build_network, reconstruct_band,
    reference_strip, spherical_triangle_area)
from mythic.tectonics.reverse import PlateHistory


@pytest.mark.parametrize("width", [100.,200.,400.])
@pytest.mark.parametrize("spacing", [60.,30.,15.])
def test_spherical_strip_area_matches_independent_small_circle_integral(width,spacing):
    a,b=lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5)
    strip=reference_strip(a,b,width,spacing)
    areas=spherical_triangle_area(strip["xyz"],strip["triangles"])
    exact=2*R_KM**2*np.sin(width/2/R_KM)*np.radians(7)
    # 离散边缘以大圆弧近似小圆，独立解析积分限定其几何误差。
    assert abs(areas).sum()==pytest.approx(exact,rel=(strip["actual_dy_km"]/R_KM)**2/2)
    assert np.all(np.sign(areas)==np.sign(areas[0]))


def test_common_rotation_stays_within_spherical_sampling_tolerance(tmp_path):
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    w=np.array([.4,-.2,.7])/R_KM
    history=PlateHistory()
    for plate in (1,2):
        history.add(plate,0,str(plate),[(0,np.eye(3)),(10,omega_rotmat(w,-10))])
    model,_=build_network(strip,pg.RotationModel(str(history.write_rot(tmp_path/"rigid.rot"))),10,plates=(1,2))
    rows,snapshots=reconstruct_band(strip,model,10,.5,[10,5,0])
    for r in rows:
        # 网络是离散球面速度插值/积分，并非直接对全部内点施加解析刚体矩阵。
        # 单独把解析位置和框架标量与真值核对，面积容差为0.1 ppm；
        # 不是质量指标或物理上的允许应变，长时/高速偏差另在负对照报表公开。
        assert r["area_ratio"]==pytest.approx(1.,abs=1e-7)
        assert r["inverted_cells"]==0
        assert r["outside_center_count"]==0
    np.testing.assert_allclose(snapshots[0.]["xyz"],strip["xyz"]@omega_rotmat(w,10).T,atol=2e-7)
    np.testing.assert_allclose(snapshots[0.]["framework_beta"],1.,atol=1e-9)


def test_orientation_reversal_is_visible_and_not_hidden_by_absolute_area():
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    triangles=strip["triangles"]
    np.testing.assert_allclose(spherical_triangle_area(strip["xyz"],triangles[:,::-1]),
                               -spherical_triangle_area(strip["xyz"],triangles),atol=1e-7)


@pytest.mark.parametrize("width,spacing", [(0,30),(-1,30),(200,0),(np.nan,30)])
def test_nonphysical_reference_inputs_rejected(width,spacing):
    with pytest.raises(ValueError):
        reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),width,spacing)
