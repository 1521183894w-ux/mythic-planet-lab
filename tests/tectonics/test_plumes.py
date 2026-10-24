"""U29 地幔柱：动力地形穹隆与大火成岩省（packages/tectonics/plumes.py）。"""

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.plumes import apply_lip, distance_km, swell_m, taper


def test_taper_is_flat_then_raised_cosine():
    d = np.array([0.0, 50.0, 100.0, 175.0, 250.0, 400.0])
    w = taper(d, 250.0, 100.0)
    assert w[0] == w[1] == w[2] == 1.0
    assert w[3] == pytest.approx(0.5)                       # 半程处为一半
    assert w[4] == pytest.approx(0.0, abs=1e-12) and w[5] == 0.0
    assert (np.diff(taper(np.linspace(0, 300, 31), 250.0)) <= 1e-12).all()


def test_distance_to_point_and_polyline():
    line = lonlat_to_xyz(np.array([0.0, 10.0]), np.array([0.0, 0.0]))
    x = lonlat_to_xyz(np.array([5.0, 5.0, 15.0]), np.array([0.0, 1.0, 0.0]))
    d = distance_km(x, line)
    deg = np.pi / 180 * R_KM
    assert d[0] < 10.0                                       # 在线上（加密间距 15 km）
    assert d[1] == pytest.approx(deg, rel=0.02)              # 离中段 1°
    assert d[2] == pytest.approx(5 * deg, rel=0.01)          # 离端点 5°
    assert distance_km(x[:1], line[:1])[0] == pytest.approx(5 * deg, rel=1e-6)


def test_swell_only_while_active_and_within_radius():
    sw = [{"center": lonlat_to_xyz(0.0, 0.0), "amp_m": 900.0, "radius_km": 1000.0, "active": (30.0, 0.0)}]
    x = lonlat_to_xyz(np.array([0.0, 0.0, 0.0]), np.array([0.0, 4.5, 20.0]))   # 中心、约 500 km、约 2200 km
    h = swell_m(x, 10.0, sw)
    assert h[0] == pytest.approx(900.0) and h[1] == pytest.approx(450.0, rel=0.03) and h[2] == 0.0
    assert (swell_m(x, 40.0, sw) == 0.0).all()              # 还没开始


def test_lip_thickens_continent_and_ocean_inside_footprint_only():
    mesh = load_mesh(5)
    c = Crust(mesh.n)
    c.plate[:] = 1
    land = mesh.lon < 0
    c.ctype[land], c.T[land] = CONT, 35.0
    eng = Engine(mesh, c, Params(), 50.0, {1: np.zeros(3)})
    T0 = eng.c.T.copy()
    fp = lonlat_to_xyz(np.array([0.0]), np.array([0.0]))    # 跨在海岸线上
    r = apply_lip(eng, fp, 900.0, 10.0, 450.0)              # L5 网格间距约 220 km
    d = distance_km(eng.positions(), fp)
    dT = eng.c.T - T0
    assert r["n"] > 0 and (dT[d > 900.0] == 0).all()
    assert np.allclose(dT[d < 450.0], 10.0)
    assert (dT[(d < 450.0) & land] > 0).any() and (dT[(d < 450.0) & ~land] > 0).any()   # 陆壳与洋壳都加厚
    assert r["volume_km3"] == pytest.approx(float((dT * eng.area).sum()))
    assert eng.budget["lip_added"] == pytest.approx(r["volume_km3"])
