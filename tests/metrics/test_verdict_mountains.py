"""U11 判定（verdict.py）与 U16 山系形态量（earthlike.shape_axes 等）的单元测试。"""

import numpy as np
import pytest
from _synth import cap

from mythic.metrics import earthlike
from mythic.metrics.compute import load_config
from mythic.metrics.grid import LatLonGrid
from mythic.metrics.verdict import b_verdict_flat, earth_unit_reference

G = LatLonGrid(1024, 512)
CFG = load_config()


def test_shape_axes_strip_vs_disc():
    lat2 = np.broadcast_to(G.lat[:, None], G.shape)
    lon2 = np.broadcast_to(G.lon[None, :], G.shape)
    strip_ew = (np.abs(lat2 - 10) < 1.5) & (lon2 > 0) & (lon2 < 40)        # 东西向条带，约 330 km × 4400 km
    strip_ns = (np.abs(lon2 - 20) < 1.5) & (lat2 > -20) & (lat2 < 20)      # 南北向条带
    disc = cap(G, 10, 20, 8)
    a_ew, a_ns, a_d = (earthlike.shape_axes(G, m) for m in (strip_ew, strip_ns, disc))
    assert a_ew["aspect_ratio"] > 8 and a_ns["aspect_ratio"] > 8
    assert a_d["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
    assert abs(a_ew["orientation_deg"] - 90) < 5                           # 东西走向
    assert min(a_ns["orientation_deg"], 180 - a_ns["orientation_deg"]) < 5  # 南北走向


def test_mountain_summary_separates_belt_from_blobs():
    lat2 = np.broadcast_to(G.lat[:, None], G.shape)
    lon2 = np.broadcast_to(G.lon[None, :], G.shape)
    land = cap(G, 0, 30, 45)
    belt = np.exp(-((lat2 - 5) / 1.2) ** 2) * ((lon2 > 0) & (lon2 < 50))
    blobs = sum(np.exp(-((lat2 - la) ** 2 + (lon2 - lo) ** 2) / 4.0) for la, lo in ((-20, 10), (-25, 45), (25, 40)))
    cfg = CFG["C_earthlike"]["mountains"]
    r_belt, _ = earthlike.mountain_systems(G, np.where(land, 300 + 3500 * belt, -4000.0), land, cfg)
    r_blob, _ = earthlike.mountain_systems(G, np.where(land, 300 + 3500 * blobs, -4000.0), land, cfg)
    assert r_belt["longest_km"] > 4000 and r_belt["elongated_area_fraction"] > 0.9
    assert r_blob["longest_km"] < 1500 and r_blob["elongated_area_fraction"] < 0.1
    assert r_belt["area_weighted_aspect_ratio"] > 3 * r_blob["area_weighted_aspect_ratio"]
    assert "low_tier" in r_belt


def _region(D, dev, pen, bay, far, med, area=10e6):
    return {"area_km2": area, "fractal_dimension": D, "coastline_development": dev,
            "peninsula_index": {"50": pen, "100": pen, "200": pen}, "bay_index": {"50": bay, "100": bay, "200": bay},
            "land_fraction_beyond_1000km": far, "distance_to_sea_median_km": med}


def test_verdict_lines_and_directions():
    earth = {"main_continent": _region(1.12, 4.6, 0.02, 0.03, 0.25, 530),
             "unit_2": _region(1.05, 1.8, 0.01, 0.02, 0.00, 320),
             "unit_9": _region(1.30, 9.0, 0.05, 0.05, 0.10, 400, area=1e6)}   # 面积不够，不参与参照
    ref = earth_unit_reference(earth, 5e6, {"main_continent": "亚非欧（本体）"})
    assert set(ref["units"]) == {"main_continent", "unit_2"}
    assert ref["pass_line"]["fractal_dimension"]["value"] == 1.05
    assert ref["pass_line"]["distance_to_sea_median_km"] == {"value": 530, "unit": "main_continent", "direction": "max"}

    from mythic.metrics.verdict import flat
    world = {"main_continent": {"area_km2": 9e6, **flat(_region(0.98, 2.7, 0.0, 0.0, 0.40, 790))},
             "unit_5": {"area_km2": 6e6, **flat(_region(1.10, 1.0, 0.03, 0.04, 0.05, 300))}}
    v = b_verdict_flat(world, ref, CFG)
    main, other = v["units"]["main_continent"], v["units"]["unit_5"]
    assert main["n_pass"] == 0 and main["n_judged"] == 9          # 海岸发育系数只作参考，不计入
    assert main["metrics"]["coastline_development"]["pass"] is None
    assert other["all_pass"]                                       # 距海距离类"越小越好"
    assert not v["main_continent_all_pass"] and not v["all_units_pass"]
