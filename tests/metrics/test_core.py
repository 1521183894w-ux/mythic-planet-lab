"""底层模块测试：格网、拓扑（环绕/跨极点）、球面距离与形态学。"""

import numpy as np
import pytest
from _synth import cap, cap_area_km2

from mythic.metrics.grid import LatLonGrid, chord_to_km, km_to_chord
from mythic.metrics.spherical import closing, dilate, distance_to_set_km, erode, opening
from mythic.metrics.topology import any_neighbor, label, roll_to_avoid_seam

G512 = LatLonGrid(512, 256)
G1024 = LatLonGrid(1024, 512)


# ---- 格网 -----------------------------------------------------------------------
def test_grid_total_area_equals_sphere():
    for g in (G512, G1024):
        assert g.cell_area_km2.sum() == pytest.approx(4 * np.pi * g.radius_km**2, rel=1e-12)


def test_grid_rows_symmetric_and_rowcol():
    a = G512.row_area_km2
    assert np.allclose(a, a[::-1])
    i, j = G512.rowcol(0.1, 179.9)
    assert (i, j) == (127, 511)
    i, j = G512.rowcol(-89.99, -180.0 + 1e-9)
    assert (i, j) == (255, 0)
    # 经度环绕：190°E 等同于 170°W
    assert G512.rowcol(10, 190)[1] == G512.rowcol(10, -170)[1]


def test_chord_km_roundtrip():
    km = np.array([0.0, 10.0, 1000.0, 10000.0, 20000.0])
    assert np.allclose(chord_to_km(km_to_chord(km)), km)


def test_cap_area_matches_analytic():
    m = cap(G1024, 40, 30, 20)
    assert (G1024.cell_area_km2 * m).sum() == pytest.approx(cap_area_km2(G1024, 20), rel=0.01)


# ---- 拓扑 -----------------------------------------------------------------------
def test_label_merges_across_dateline():
    m = np.zeros(G512.shape, bool)
    m[100:110, :3] = True
    m[100:110, -3:] = True
    _, n = label(m)
    assert n == 1


def test_label_diagonal_across_dateline():
    m = np.zeros(G512.shape, bool)
    m[100, 0] = True
    m[101, -1] = True  # 对角相邻且跨接缝
    _, n = label(m)
    assert n == 1


def test_label_across_pole():
    m = np.zeros(G512.shape, bool)
    m[0, 10] = True
    m[0, 10 + G512.nx // 2] = True
    _, n = label(m)
    assert n == 1


def test_label_keeps_separate_blobs():
    m = np.zeros(G512.shape, bool)
    m[50:60, 50:60] = True
    m[50:60, 70:80] = True
    lab, n = label(m)
    assert n == 2 and set(np.unique(lab)) == {0, 1, 2}


def test_any_neighbor_wraps():
    m = np.zeros(G512.shape, bool)
    m[100, 0] = True
    nb = any_neighbor(m)
    assert nb[100, -1] and nb[99, -1] and nb[101, 1] and not nb[100, 0]


def test_roll_to_avoid_seam():
    m = np.zeros(G512.shape, bool)
    m[10:20, -5:] = True
    m[10:20, :5] = True
    s = roll_to_avoid_seam(m)
    r = np.roll(m, s, axis=1)
    assert not (r[:, 0].any() and r[:, -1].any())


# ---- 球面距离与形态学 --------------------------------------------------------------
def test_distance_from_pole_to_cap_edge():
    land = cap(G512, 90, 0, 30)  # 北极冠，边界在 60°N
    d = distance_to_set_km(G512, ~land, query=land)
    expected = np.radians(30) * G512.radius_km  # 3336 km
    pix = np.pi * G512.radius_km / G512.ny  # 一个格点的南北跨度
    assert np.nanmax(d) == pytest.approx(expected, abs=pix)


def test_distance_symmetric_across_dateline():
    land = cap(G1024, 0, 180, 10)  # 以日界线为中心的球冠
    d = distance_to_set_km(G1024, ~land, query=land)
    i, j = G1024.rowcol(0, 179.9)
    i2, j2 = G1024.rowcol(0, -179.9)
    assert d[i, j] == pytest.approx(d[i2, j2], rel=1e-6)


def test_opening_removes_thin_peninsula_keeps_body():
    g = G1024  # 格点约 39 km
    body = cap(g, 20, 0, 20)
    pen = np.zeros(g.shape, bool)
    i0, _ = g.rowcol(40.5, 0)  # 球冠北缘约 40°N
    i1, _ = g.rowcol(60, 0)
    _, j = g.rowcol(0, 0)
    pen[i1:i0, j] = True  # 1 列宽（约 20–30 km）、约 20° 长的半岛
    land = body | pen
    opened = opening(g, land, 50.0)
    area = g.cell_area_km2
    removed = (land & ~opened)
    # 半岛被削掉，主体几乎不变
    assert (removed & pen & ~body).sum() >= 0.9 * (pen & ~body).sum()
    assert (area * (body & opened)).sum() >= 0.99 * (area * body).sum()
    assert not (opened & ~land).any()


def test_closing_fills_narrow_bay():
    g = G1024
    body = cap(g, 20, 0, 20)
    bay = np.zeros(g.shape, bool)
    i0, _ = g.rowcol(36, 0)
    i1, _ = g.rowcol(25, 0)
    _, j = g.rowcol(0, 0)
    bay[i0:i1, j] = True  # 从北岸切入的 1 列宽窄海湾
    land = body & ~bay
    closed = closing(g, land, 50.0)
    assert closed[bay & body].all()
    assert (closed | ~land).all()  # 闭运算只增不减


def test_erode_dilate_monotone():
    land = cap(G512, -10, 100, 25)
    e = erode(G512, land, 300)
    d = dilate(G512, land, 300)
    assert not (e & ~land).any() and not (land & ~d).any()
    assert e.sum() < land.sum() < d.sum()
