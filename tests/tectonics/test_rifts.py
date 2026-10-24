"""U30 陆内裂谷（packages/tectonics/rifts.py）：地堑减薄、两肩挠曲抬升与轴部隆起。"""

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, ORO_RIFT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.rifts import apply_graben, rift_relief_m

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
