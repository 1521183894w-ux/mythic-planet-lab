"""A / B / C 层指标测试：合成世界，答案可解析或可预期。"""

import numpy as np
import pytest
from _synth import angular_distance_deg, cap, cap_area_km2

from mythic.metrics import coastal, earthlike, hard
from mythic.metrics.compute import compute_all, load_config
from mythic.metrics.grid import LatLonGrid

G1024 = LatLonGrid(1024, 512)  # 赤道格点约 39 km
G2048 = LatLonGrid(2048, 1024)  # 约 19.5 km
CFG = load_config()


def two_caps_with_bridge(g, width_cols):
    a = cap(g, 0, -30, 15)
    b = cap(g, 0, 30, 10)
    bridge = np.zeros(g.shape, bool)
    i, _ = g.rowcol(0, 0)
    _, j0 = g.rowcol(0, -16)
    _, j1 = g.rowcol(0, 21)
    bridge[i - width_cols // 2 : i - width_cols // 2 + width_cols, j0:j1] = True
    return a, b, a | b | bridge


# ---- A 层 ------------------------------------------------------------------------
def test_thin_bridge_does_not_connect():
    a, b, land = two_caps_with_bridge(G2048, 1)  # 1 行（约 19.5 km）宽的陆桥
    u = hard.landmass_units(G2048, land, 50)
    assert u["n_without_thin_link_rule"] == 1  # 普通连通算法认为只有一块
    big = sorted(u["areas_km2"][1:], reverse=True)
    assert big[0] == pytest.approx(cap_area_km2(G2048, 15), rel=0.05)
    assert big[1] == pytest.approx(cap_area_km2(G2048, 10), rel=0.08)


def test_wide_bridge_connects():
    _, _, land = two_caps_with_bridge(G2048, 6)  # 约 117 km 宽
    u = hard.landmass_units(G2048, land, 50)
    assert u["areas_km2"][u["main_id"]] / (G2048.cell_area_km2 * land).sum() > 0.99


def test_subcontinents_split_at_neck():
    _, _, land = two_caps_with_bridge(G1024, 5)  # 约 200 km 宽的颈部
    main = hard.landmass_units(G1024, land, 50)["main_mask"]
    assert hard.subcontinents(G1024, main, 250, 1e6)["count"] == 2
    assert hard.subcontinents(G1024, main, 50, 1e6)["count"] == 1


def test_inland_sea_with_one_strait():
    g = G1024
    cont = cap(g, 30, 20, 35)
    sea = cap(g, 30, 20, 10)  # 内海面积约 387 万 km²
    land = cont & ~sea
    # 一条从内海通向外洋、1 列宽（约 30 km）的海峡
    _, j = g.rowcol(30, 20)
    i_top, _ = g.rowcol(66, 20)
    i_sea, _ = g.rowcol(39, 20)
    land[i_top:i_sea, j] = False
    main = hard.landmass_units(g, land, 50)["main_mask"]
    seas = hard.inland_seas(g, land, main, 150, 0.5)
    chk = hard.main_inland_sea_check(seas["seas"], CFG["A_hard_constraints"]["main_inland_sea"])
    assert chk["exists"]
    assert chk["main"]["area_km2"] == pytest.approx(cap_area_km2(g, 10), rel=0.1)
    assert chk["main"]["straits_to_world_ocean"] == 1
    assert chk["passes"]


def _inland_sea_geometry(g, land):
    """跑一遍 A 层主内海判定（含 U18 的形状与接壤条件），返回判定结果。"""
    cfg = CFG["A_hard_constraints"]
    main = hard.landmass_units(g, land, 50)["main_mask"]
    sub = hard.subcontinents(g, main, cfg["subcontinents"]["neck_radius_km"], cfg["subcontinents"]["min_area_km2"])
    seas = hard.inland_seas(g, land, main, 150, 0.5)
    cis = cfg["main_inland_sea"]
    chk = hard.main_inland_sea_check(seas["seas"], cis)
    geo = hard.inland_sea_geometry(g, land, seas["region_labels"] == chk["main"]["region_id"], sub["labels"],
                                   sub["big_ids"], cis["bordering_subcontinents"]["min_border_share"])
    return sub, hard.apply_inland_sea_geometry(chk, geo, cis)


def test_inland_sea_between_two_subcontinents_is_elongated_and_bordered_twice():
    """赤道上一块大陆，中间一条东西向的狭长海（约 7100 × 670 km）几乎把它切成南北两半，
    两端各留约 330 km 宽的颈部（亚大陆在此切开），东端颈部开一条 1 格宽的海峡通外洋。"""
    g = G1024
    cont = cap(g, 0, 20, 35)
    sea = (np.abs(g.lat)[:, None] < 3) & (angular_distance_deg(g, 0, 20) < 32)
    land = cont & ~sea
    i, _ = g.rowcol(0.1, 52)
    _, j0 = g.rowcol(0, 51)
    _, j1 = g.rowcol(0, 57)
    land[i, j0:j1] = False
    sub, chk = _inland_sea_geometry(g, land)
    assert sub["count"] == 2
    geo = chk["geometry"]
    assert geo["n_bordering_subcontinents"] == 2
    assert all(s["border_share"] > 0.3 for s in geo["subcontinent_border_shares"])
    assert geo["aspect_ratio"] > 5
    assert 80 < geo["orientation_deg"] < 100          # 东西走向
    assert chk["checks"]["bordering_subcontinents_ok"] and chk["checks"]["elongation_ok"]


def test_round_sea_inside_one_subcontinent_fails_shape_and_border_checks():
    """圆形内海挖在一块大陆中间（圆润世界、噪声海岸世界的情形）：旧的三项通过，新的两项不通过。"""
    g = G1024
    land = cap(g, 30, 20, 35) & ~cap(g, 30, 20, 10)
    _, j = g.rowcol(30, 20)
    i_top, _ = g.rowcol(66, 20)
    i_sea, _ = g.rowcol(39, 20)
    land[i_top:i_sea, j] = False
    sub, chk = _inland_sea_geometry(g, land)
    assert sub["count"] == 1
    assert chk["geometry"]["aspect_ratio"] < 1.3
    assert chk["geometry"]["n_bordering_subcontinents"] == 1
    assert chk["checks"]["area_in_range"] and chk["checks"]["straits_ok"]
    assert not chk["checks"]["bordering_subcontinents_ok"] and not chk["checks"]["elongation_ok"]
    assert not chk["passes"]


def test_open_ocean_emptiness_and_polar_continent():
    g = G1024
    land = cap(g, -90, 0, 18)  # 南极大陆，约 1250 万 km²
    oo = hard.open_ocean_emptiness(g, land)
    assert oo["max_distance_to_land_km"] == pytest.approx(np.radians(180 - 18) * g.radius_km, abs=60)
    u = hard.landmass_units(g, land, 50)
    lat = hard.unit_centroid_latitudes(g, u["labels"], u["n"])
    assert lat[u["main_id"]] < -85


# ---- B1 / B2 ---------------------------------------------------------------------
def test_disc_has_near_zero_peninsula_and_bay_index():
    land = cap(G1024, 20, 40, 30)
    for r in (50, 100, 200):
        assert coastal.peninsula_index(G1024, land, r)[0] < 0.01
        assert coastal.bay_index(G1024, land, r)[0] < 0.01


def test_peninsulas_raise_peninsula_index():
    g = G2048
    land = cap(g, 0, 0, 25)
    for lon in (-15, -5, 5, 15):  # 4 条约 2 格点宽、约 12° 长的半岛
        _, j = g.rowcol(0, lon)
        i0, _ = g.rowcol(37, lon)
        i1, _ = g.rowcol(24, lon)
        land[i0:i1, j : j + 2] = True
    v50, _ = coastal.peninsula_index(g, land, 50)
    assert v50 > 0.005


def test_island_count():
    g = G1024
    land = np.zeros(g.shape, bool)
    for k in range(7):
        land |= cap(g, -30 + 10 * k, 10 + 25 * k, 2.0)
    st = coastal.island_stats(g, land, 5e6, 4)
    assert st["n_islands"] == 7 and st["n_continents"] == 0


def test_shelf_fraction():
    g = G1024
    land = cap(g, 0, 0, 20)
    elev = np.where(land, 500.0, -4000.0)
    elev[(angular_distance_deg(g, 0, 0) < 25) & ~land] = -100.0  # 陆地周围一圈浅海
    expected = (cap_area_km2(g, 25) - cap_area_km2(g, 20)) / (4 * np.pi * g.radius_km**2 - cap_area_km2(g, 20))
    assert coastal.shelf_fraction(g, land, elev, 200) == pytest.approx(expected, rel=0.03)


def test_distance_to_sea_distribution():
    g = G1024
    land = cap(g, 90, 0, 30)
    st, _ = coastal.distance_to_sea(g, land, [1000])
    assert st["max_km"] == pytest.approx(np.radians(30) * g.radius_km, abs=60)
    # 距海 > 1000 km 的陆地 = 半径 (30° − 1000 km) 的内球冠
    inner = 30 - np.degrees(1000 / g.radius_km)
    assert st["land_fraction_beyond_1000km"] == pytest.approx(cap_area_km2(g, inner) / cap_area_km2(g, 30), abs=0.03)


# ---- C 层 ------------------------------------------------------------------------
def test_slope_of_latitude_ramp():
    g = G1024
    k = 10.0  # 每度纬度 10 m
    elev = k * g.lat[:, None] * np.ones(g.shape)
    s = earthlike.slope_degrees(g, elev)
    expected = np.degrees(np.arctan(k / (np.radians(1) * g.radius_km * 1000)))
    assert np.median(s[100:400]) == pytest.approx(expected, rel=0.01)


def test_hypsometry_quantiles():
    g = G1024
    land = cap(g, 0, 0, 40)
    elev = np.where(land, 500.0, -4000.0)
    h = earthlike.hypsometry(g, elev, land, 100)
    assert h["quantiles_land_m"]["0.5"] == pytest.approx(500.0)
    assert h["mean_ocean_m"] == pytest.approx(-4000.0)


def test_mountain_ridge_length():
    g = G2048
    land = (np.abs(g.lat)[:, None] < 20) & (g.lon[None, :] > -15) & (g.lon[None, :] < 45)
    d_km = np.radians(np.abs(g.lat))[:, None] * g.radius_km * np.ones(g.shape)
    along = (g.lon[None, :] >= 0) & (g.lon[None, :] <= 30)
    elev = np.where(land, 200.0, -3000.0) + np.where(along & land, 3000.0 * np.exp(-(d_km / 80.0) ** 2), 0.0)
    res, _ = earthlike.mountain_systems(g, elev, land, CFG["C_earthlike"]["mountains"])
    assert res["n_systems"] == 1
    assert res["systems"][0]["length_km"] == pytest.approx(np.radians(30) * g.radius_km, rel=0.15)


def test_compute_all_on_synthetic_world():
    g = G1024
    land = cap(g, 20, 0, 50)
    elev = np.where(land, 400.0, -4000.0)
    res = compute_all(elev, land)
    assert res["A"]["ocean_fraction"]["value"] == pytest.approx(1 - cap_area_km2(g, 50) / (4 * np.pi * g.radius_km**2), abs=0.005)
    assert res["by_region"]["main_continent"]["fractal_dimension"] == pytest.approx(1.0, abs=0.1)
    assert res["B1"]["peninsula_index"]["100"] < 0.01
    # 主大陆纬度范围：球冠中心 20°N、半径 50° -> 北端约 70°N、南端约 30°S
    lat = res["A"]["main_continent_latitude"]
    assert lat["north_deg"] == pytest.approx(70.0, abs=0.5)
    assert lat["south_deg"] == pytest.approx(-30.0, abs=0.5)
    assert res["resolution_warnings"]  # 1024x512 上 25 km 的细线判定应给出警告
