"""探索性诊断：能否从高程数据里识别"有复杂度、无成因"的海岸（阶段 0.7 第二个对照样本）。

只做比较，不是正式指标；若要纳入 config/metrics.yaml，须单独提出、经用户同意（CLAUDE.md §2.5）。
对地球、圆润世界、噪声海岸世界在同一分辨率上计算：

1. 海岸复杂度的空间异质性：2000 km 窗口（与区域放大图同一套 CoastAnalysis）的复杂度分布，
   只取中心落在成形海岸上、平滑后窗口内仍有 ≥ 1000 km 海岸的窗口；看 p10、p50、p90、p90/p10 与变异系数。
   地球的复杂海岸集中在少数地方（峡湾、岛弧、多岛海），其余很平直；噪声应处处差不多。
2. 复杂度与纬度：窗口复杂度与 |纬度| 的 Spearman 相关（地球高纬有冰川海岸，应为正）。
3. 复杂度与地形的耦合：窗口复杂度与窗口内陆地起伏（高程 90% 分位减 10% 分位）的 Spearman 相关。
   地球上破碎的海岸常紧挨高山、断块；独立生成的噪声海岸与噪声地形应不相关。
4. 远洋岛屿：达到面积下限的岛中，离最近的大陆级陆块（≥ 500 万 km²）超过 1000 km 的比例与个数。
5. 山系形态（U16）：最长山系、面积加权主轴长宽比、成带山系面积占比。
6. 大尺度半岛、海湾指数（世界 C 评审员的建议）：最大陆块在 r = 100、200、300、500 km 的半岛指数与海湾指数，
   看能不能用更大的开、闭运算半径量出"有没有几百公里以上的大半岛"。

用法：powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\coast_origin_diagnostics.py [分辨率]
输出：reviews/stage0_noise_diagnostics_<分辨率>.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from mythic.figures.select_regions import CoastAnalysis
from mythic.figures.world import ROOT, code_version, load_earth, load_world_nc, to_jsonable
from mythic.metrics.coastal import bay_index, peninsula_index
from mythic.metrics.compute import load_config
from mythic.metrics.earthlike import mountain_systems
from mythic.metrics.grid import km_to_chord
from mythic.metrics.spherical import distance_to_set_km
from mythic.metrics.topology import component_sums, label


def window_relief(grid, elev, land, centers, radius_km):
    """每个窗口内陆地高程的 90% 分位减 10% 分位（m）。"""
    xyz = grid.xyz.reshape(-1, 3)
    idx_land = np.flatnonzero(land.ravel())
    tree = cKDTree(xyz[idx_land])
    e = elev.ravel()[idx_land]
    out = np.full(len(centers), np.nan)
    chord = float(km_to_chord(radius_km, grid.radius_km))
    for i, nb in enumerate(tree.query_ball_point(centers, chord, workers=-1)):
        if len(nb) > 20:
            v = e[nb]
            out[i] = np.percentile(v, 90) - np.percentile(v, 10)
    return out


def diagnose(world, cfg) -> dict:
    g, land, elev = world.grid, world.land, world.elevation
    ca = CoastAnalysis(g, land)
    ok = (ca.L_smooth >= ca.radius_km) & ca.on_smooth
    c = ca.complexity[ok]
    lat = np.degrees(np.arcsin(np.clip(ca.cand[ok, 2], -1, 1)))
    relief = window_relief(g, elev, land, ca.cand[ok], ca.radius_km)
    fin = np.isfinite(relief)
    q = np.percentile(c, [10, 50, 90])
    # 远洋岛屿
    area = g.cell_area_km2
    lab, n = label(land)
    comp = component_sums(lab, n, area)
    a_min = cfg["B1_coast"]["islands"]["min_area_pixels"] * (np.pi * g.radius_km / g.ny) ** 2
    big = np.flatnonzero(comp[1:] >= cfg["B1_coast"]["continent_min_area_km2"]) + 1
    isl = np.flatnonzero((comp[1:] < cfg["B1_coast"]["continent_min_area_km2"]) & (comp[1:] >= a_min)) + 1
    far_frac, n_far = float("nan"), 0
    if isl.size and big.size:
        isl_mask = np.isin(lab, isl)
        d_cells = distance_to_set_km(g, np.isin(lab, big), query=isl_mask)  # 每个岛屿格点到最近大陆级陆块的距离
        dist = ndi.minimum(np.nan_to_num(d_cells, nan=np.inf), labels=lab, index=isl)  # 每个岛取最近的格点
        dist = np.asarray(dist, float)
        far_frac, n_far = float((dist > 1000).mean()), int((dist > 1000).sum())
    largest = lab == (int(np.argmax(comp[1:])) + 1)
    large_scale = {str(r): {"peninsula": peninsula_index(g, largest, r)[0], "bay": bay_index(g, largest, r)[0]}
                   for r in (100, 200, 300, 500)}
    mt, _ = mountain_systems(g, elev, land, cfg["C_earthlike"]["mountains"], exclude=world.mountain_exclude)
    return {
        "n_windows": int(ok.sum()),
        "complexity_p10_p50_p90": [float(x) for x in q], "complexity_p90_over_p10": float(q[2] / q[0]),
        "complexity_cv": float(c.std() / c.mean()),
        "spearman_complexity_vs_abslat": float(spearmanr(c, np.abs(lat)).correlation),
        "spearman_complexity_vs_relief": float(spearmanr(c[fin], relief[fin]).correlation),
        "largest_landmass_Mkm2": float(comp[1:].max() / 1e6), "largest_landmass_peninsula_bay_by_radius_km": large_scale,
        "islands_counted": int(isl.size), "oceanic_islands_beyond_1000km": n_far, "oceanic_island_fraction": far_frac,
        "mountains": {k: mt[k] for k in ("n_systems", "n_longer_than_min_length", "longest_km",
                                         "area_weighted_aspect_ratio", "elongated_area_fraction")},
    }


def main() -> None:
    res = sys.argv[1] if len(sys.argv) > 1 else "2048x1024"
    cfg = load_config()
    worlds = [("earth", load_earth(res)),
              ("rounded", load_world_nc(ROOT / f"data/worlds/rounded_control/rounded_control_{res}.nc")),
              ("noise", load_world_nc(ROOT / f"data/worlds/noise_control/noise_control_{res}.nc"))]
    out = {"description": "探索性诊断：识别'有复杂度、无成因'的海岸（不是正式指标）", "resolution": res,
           "code_version": code_version(), "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "worlds": {}}
    for name, w in worlds:
        t0 = time.perf_counter()
        d = diagnose(w, cfg)
        out["worlds"][name] = d
        ls = d["largest_landmass_peninsula_bay_by_radius_km"]
        print(f"{name:<8} 窗口 {d['n_windows']}  复杂度 p10/p50/p90 = " + "/".join(f"{x:.2f}" for x in d["complexity_p10_p50_p90"])
              + f"  p90/p10 {d['complexity_p90_over_p10']:.2f}  CV {d['complexity_cv']:.2f}"
              f"  ρ(|纬度|) {d['spearman_complexity_vs_abslat']:+.2f}  ρ(起伏) {d['spearman_complexity_vs_relief']:+.2f}"
              f"  岛 {d['islands_counted']}，远洋 {d['oceanic_islands_beyond_1000km']}（{d['oceanic_island_fraction']:.2f}）"
              f"  山系最长 {d['mountains']['longest_km']:.0f} km，成带占比 {d['mountains']['elongated_area_fraction']:.2f}"
              f"  最大陆块半岛/海湾 r300 {ls['300']['peninsula']:.3f}/{ls['300']['bay']:.3f} r500 {ls['500']['peninsula']:.3f}/{ls['500']['bay']:.3f}"
              f"  （{time.perf_counter() - t0:.0f}s）", flush=True)
    p = ROOT / "reviews" / f"stage0_noise_diagnostics_{res}.json"
    p.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"写出 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
