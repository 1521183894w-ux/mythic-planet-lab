"""区域放大图的自动选取（packages/figures/select_regions.py）与世界文件读取。

合成世界（1024x512，主内海判定需要这个分辨率，见 docs/metrics_definitions.md 的分辨率限制）：
- 主大陆 A：(20°N, 0°) 半径 35° 的球冠，中间挖出 (25°N, 0°) 半径 10° 的内海，内海经一条约 1 个格点宽的海峡通外洋；
- 亚大陆 B：(20°N, 90°E) 半径 25° 的球冠，经 20°N 附近约 110 km 宽的陆桥与 A 相连（宽于 50 km 的细线门槛、窄于 500 km 的颈部门槛）；
- 岛群：(-10°, 120°E) 附近 5×5 个半径 0.8° 的小岛；
- 极地大陆：南极点半径 18° 的球冠（约 12.5 百万 km²），最北的海岸在约 72°S。
"""

import numpy as np
import pytest
import xarray as xr

from mythic.figures.select_regions import CoastAnalysis, select_regions
from mythic.figures.world import load_world_nc
from mythic.metrics.compute import compute_all, load_config
from mythic.metrics.grid import LatLonGrid


def _ang(grid, lat0, lon0):
    c = np.array([np.cos(np.radians(lat0)) * np.cos(np.radians(lon0)),
                  np.cos(np.radians(lat0)) * np.sin(np.radians(lon0)), np.sin(np.radians(lat0))])
    return np.degrees(np.arccos(np.clip(grid.xyz @ c, -1, 1)))


def _gc_deg(a, b):
    """两点 (纬度, 经度) 之间的大圆角距离（度）。"""
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    c = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


@pytest.fixture(scope="module")
def synth():
    g = LatLonGrid(1024, 512)
    lat2 = np.broadcast_to(g.lat[:, None], g.shape)
    lon2 = np.broadcast_to(g.lon[None, :], g.shape)
    land = _ang(g, 20, 0) < 35
    land &= ~(_ang(g, 25, 0) < 10)
    strait = (np.abs(lat2 - 25) < 0.2) & (lon2 < -9) & (lon2 > -40)  # 内海向西通外洋的窄海峡（1 个格点）
    land &= ~strait
    land |= _ang(g, 20, 90) < 25
    land |= (np.abs(lat2 - 20) < 0.5) & (lon2 > 30) & (lon2 < 70)   # 陆桥
    for i in range(5):
        for j in range(5):
            land |= _ang(g, -16 + 3 * i, 114 + 3 * j) < 0.8
    land |= lat2 < -72
    elev = np.where(land, 500.0, -4000.0)
    cfg = load_config()
    m, f = compute_all(elev, land, cfg=cfg, return_fields=True)
    picks, ca = select_regions("synthetic_test", g, land, m, f, cfg, {"mediterranean": 4000.0})
    return g, land, m, f, cfg, {p.key: p for p in picks}, ca


def test_main_inland_sea_found_and_centered(synth):
    *_, m, f, cfg, picks, ca = synth
    assert m["A"]["main_inland_sea"]["exists"]
    p = picks["r1_main_inland_sea"]
    assert p.missing is None
    assert _gc_deg(p.center, (25, 0)) < 3
    assert p.size_km >= 4000  # 取 max(内海所需, 地中海对照边长)


def test_inland_sea_complex_coast_is_on_the_sea(synth):
    p = synth[5]["r2_inland_sea_complex_coast"]
    assert p.missing is None
    # 在内海海岸上（内海半径 10°）；海峡两岸的海岸点也属于内海沿岸，所以允许略远于 10°
    assert 8 < _gc_deg(p.center, (25, 0)) < 20
    sea = synth[3]["sea_regions"]
    i, j = synth[0].rowcol(*p.center)
    near = sea[max(0, i - 3):i + 4, max(0, j - 3):j + 4]
    assert (near == synth[2]["A"]["main_inland_sea"]["main"]["region_id"]).any()


def test_north_coast_and_polar_coast(synth):
    picks = synth[5]
    assert abs(picks["r3_north_coast"].center[0] - 55.0) < 1.5   # 主大陆最北：20 + 35 = 55°N
    assert picks["r4_polar_coast"].missing is None
    assert abs(picks["r4_polar_coast"].center[0] + 72.0) < 0.6


def test_isthmus_on_land_bridge(synth):
    p = synth[5]["r5_isthmus"]
    assert p.missing is None
    assert abs(p.center[0] - 20) < 1.5 and 30 < p.center[1] < 70
    assert p.score < 500  # 颈部宽度估计 < 500 km


def test_archipelago_and_smooth_coast(synth):
    picks = synth[5]
    assert _gc_deg(picks["r7_archipelago"].center, (-10, 120)) < 8
    assert picks["r7_archipelago"].score >= 20
    p6 = picks["r6_smoothest_coast"]
    assert p6.missing is None and p6.score < 1.3  # 光滑球冠海岸：复杂度约为 1


def test_coast_windows_shared_with_metrics(synth):
    """U18(a)：区域选取复用 compute_all 算好的海岸窗口，结果与重新计算完全相同；定义与 config 一致。"""
    g, land, m, f, cfg, picks, ca = synth
    assert "coast_windows" in f
    fresh = CoastAnalysis(g, land)
    np.testing.assert_array_equal(ca.complexity, fresh.complexity)
    np.testing.assert_array_equal(ca.cand_sea_cell, fresh.cand_sea_cell)
    cs = cfg["B1_coast"]["complexity_spread"]
    from mythic.figures import select_regions as sr
    assert (cs["window_km"], cs["smooth_radius_km"], cs["candidate_step_km"]) == (
        sr.DEFAULT_SIZE_KM, sr.SMOOTH_RADIUS_KM, sr.CANDIDATE_STEP_KM)
    spread = m["B1"]["coast_complexity_spread"]
    # 合成世界的大陆都是光滑球冠：成形海岸窗口几乎都平直（小岛群在平滑时被抹掉，不算成形海岸）
    assert spread["n_windows"] > 100
    assert spread["p50"] < 1.3 and spread["straight_fraction"] > 0.8


def test_random_coast_is_reproducible(synth):
    g, land, m, f, cfg, picks, ca = synth
    again, _ = select_regions("synthetic_test", g, land, m, f, cfg, {}, ca=ca)
    other, _ = select_regions("another_world", g, land, m, f, cfg, {}, ca=ca)
    r8 = picks["r8_random_coast"].center
    assert again[-1].center == r8
    assert other[-1].center != r8
    d = np.degrees(np.arccos(np.clip(ca.pts @ np.array([np.cos(np.radians(r8[0])) * np.cos(np.radians(r8[1])),
                                                         np.cos(np.radians(r8[0])) * np.sin(np.radians(r8[1])),
                                                         np.sin(np.radians(r8[0]))]), -1, 1)))
    assert d.min() < 1e-6  # 随机点就是一个海岸采样点


def test_missing_rules_reported():
    """没有内海、没有极地大陆、只有一块圆形大陆时，相应规则报告"未找到"而不是乱选。"""
    g = LatLonGrid(512, 256)
    land = _ang(g, 0, 0) < 40
    cfg = load_config()
    m, f = compute_all(np.where(land, 500.0, -4000.0), land, cfg=cfg, return_fields=True)
    picks = {p.key: p for p in select_regions("disc", g, land, m, f, cfg, {})[0]}
    for k in ("r1_main_inland_sea", "r2_inland_sea_complex_coast", "r4_polar_coast", "r5_isthmus", "r7_archipelago"):
        assert picks[k].missing, k
    assert picks["r6_smoothest_coast"].missing is None
    assert picks["r8_random_coast"].missing is None


def test_mirrored_world_keeps_metrics():
    """东西镜像 + 经度平移是保距变换：A 层与距海距离等指标不变（盲评用，见 mythic.figures.world.mirrored）。"""
    from mythic.figures.world import World, blinded, mirrored

    g = LatLonGrid(512, 256)
    land = (_ang(g, 30, -40) < 30) | (_ang(g, -20, 60) < 18) | (np.broadcast_to(g.lat[:, None], g.shape) < -72)
    w = World(world_id="t", display_name="t", elevation=np.where(land, 500.0, -4000.0), land=land,
              method_class="visual_only", source="test")
    w2 = blinded(mirrored(w, 100.0), "世界 X", "calib_x")
    assert w2.method_label != "visual_only" and w2.method_class == "visual_only" and w2.display_name == "世界 X"
    cfg = load_config()
    m1 = compute_all(w.elevation, w.land, cfg=cfg)
    m2 = compute_all(w2.elevation, w2.land, cfg=cfg)
    assert m1["A"]["ocean_fraction"]["value"] == pytest.approx(m2["A"]["ocean_fraction"]["value"], rel=1e-12)
    assert m1["A"]["main_continent"]["share_of_land"] == pytest.approx(m2["A"]["main_continent"]["share_of_land"])
    assert m1["B2"]["median_km"] == pytest.approx(m2["B2"]["median_km"], rel=1e-9)
    assert m1["B1"]["peninsula_index"]["100"] == pytest.approx(m2["B1"]["peninsula_index"]["100"], rel=1e-9)


def test_load_world_nc(tmp_path):
    from mythic.figures.netcdf import write_netcdf
    g = LatLonGrid(64, 32)
    elev = np.where(_ang(g, 0, 0) < 30, 800.0, -3000.0)
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": g.lat, "lon": g.lon},
                    attrs={"world_id": "w1", "seed": "42", "method_class_elevation": "procedural_constrained"})
    p = tmp_path / "w1.nc"
    write_netcdf(ds, p)
    w = load_world_nc(p)
    assert w.world_id == "w1" and w.seed == "42" and w.method_class == "procedural_constrained"
    assert w.land.sum() == (elev > 0).sum()
    ds.attrs.pop("method_class_elevation")
    p2 = tmp_path / "w2.nc"
    write_netcdf(ds, p2)
    with pytest.raises(ValueError, match="method_class"):
        load_world_nc(p2)
