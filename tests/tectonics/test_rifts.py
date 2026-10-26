"""U30 陆内裂谷（packages/tectonics/rifts.py）：地堑减薄、两肩挠曲抬升与轴部隆起。"""

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, ORO_RIFT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.rifts import apply_graben, rift_relief_m, progressive_beta

DEG_KM = np.pi / 180 * R_KM


def test_rift_relief_has_flanks_outside_graben_and_axial_swell():
    line = lonlat_to_xyz(np.array([0.0, 0.0]), np.array([-10.0, 10.0]))
    lon = np.array([0.0, 30 / DEG_KM, 41 / DEG_KM, 160 / DEG_KM, 600 / DEG_KM])
    x = lonlat_to_xyz(lon, np.zeros_like(lon))
    h = rift_relief_m(x, line, 40.0, 1000.0, 60.0)
    assert h[0] == 0.0 and h[1] == 0.0                         # 地堑里没有肩部抬升
    assert h[2] == pytest.approx(1000.0, rel=0.03)             # 边界断层外侧最高
    assert h[3] == pytest.approx(1000.0 * np.exp(-2.0), rel=0.05) and h[4] < 1.0
    s = rift_relief_m(x, line, 40.0, 0.0, 60.0, 600.0, 400.0)
    assert s[0] == pytest.approx(600.0, rel=0.01) and s[4] == 0.0   # 轴部隆起（轴线按 15 km 加密）


def test_graben_thins_only_continental_crust_near_axis():
    mesh = load_mesh(5)
    c = Crust(mesh.n)
    c.plate[:] = 1
    land = mesh.lon < 20
    c.ctype[land], c.T[land] = CONT, 35.0
    eng = Engine(mesh, c, Params(), 30.0, {1: np.zeros(3)})
    T0 = eng.c.T.copy()
    line = lonlat_to_xyz(np.array([10.0, 10.0]), np.array([-30.0, 30.0]))
    r = apply_graben(eng, line, 300.0, 1.2)                    # L5 网格间距约 220 km，地堑放宽
    lon = mesh.lon[eng.c.node]
    near = (np.abs(lon - 10) < 1.5) & (np.abs(mesh.lat[eng.c.node]) < 25)
    assert r["n"] > 0 and np.allclose(eng.c.T[near], 35.0 / 1.2)
    assert (eng.c.oro[near] == ORO_RIFT).all()
    assert (eng.c.T[lon > 25] == T0[lon > 25]).all() and (eng.c.T[np.abs(lon - 10) > 8] == T0[np.abs(lon - 10) > 8]).all()
    assert r["volume_km3"] == pytest.approx(float(((T0 - eng.c.T) * eng.area).sum()))


def test_progressive_interval_integrates_only_active_time_and_is_partition_invariant():
    # 起点/终点跨在半步内，也必须积分满5Myr且只积分一次。
    for dt in (1., .5, .25):
        factors = [progressive_beta(1.5, 5., 8.25, t, dt) for t in np.arange(0., 11., dt)]
        assert np.prod(factors) == pytest.approx(1.5, rel=1e-12)
    assert progressive_beta(1.5, 5., 8.25, 9.) == 1.
    with pytest.raises(ValueError):
        progressive_beta(1.5, 0., 8., 7.)


def test_repeated_log_thinning_matches_single_total_beta_in_core_and_taper():
    import copy
    mesh = load_mesh(4)
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.ctype[:] = CONT
    c.T[:] = 45.
    eng = Engine(mesh, c, Params(), 35., {1: np.zeros(3)})
    control = copy.deepcopy(eng)
    line = lonlat_to_xyz(np.array([10., 10.]), np.array([-30., 30.]))
    apply_graben(control, line, 500., 1.5, logarithmic=True)
    for t in range(34, -1, -1):
        apply_graben(eng, line, 500., progressive_beta(1.5, 35., 35., t), logarithmic=True)
    np.testing.assert_allclose(eng.c.T, control.c.T, rtol=1e-12, atol=1e-12)
    assert eng.budget["rift_thinning"] == pytest.approx(control.budget["rift_thinning"], rel=1e-12)
    changed = eng.c.T < 45. - 1e-8
    assert changed.any() and (~changed).any()
