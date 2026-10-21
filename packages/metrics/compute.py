"""指标总调度：输入一个世界（高程 + 陆地掩膜 + 区域标签），输出 A / B1 / B2 / C 层全部"只需高程"的指标。

地球基准与虚构世界调用同一个 compute_all：
- 地球：land = Natural Earth 陆地比例 ≥ 0.5；regions = 大洲编号；
- 虚构世界：land = 高程 > 0；regions = 陆块单元（面积 ≥ continent_min_area_km2 的单元）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from . import coast_windows, coastal, earthlike, hard
from .coast import coastline_development
from .distributions import weighted_quantiles
from .grid import LatLonGrid

ROOT = Path(__file__).resolve().parents[1].parent
DEFAULT_CONFIG = ROOT / "config" / "metrics.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _in_range(v, lo, hi) -> bool:
    return v is not None and np.isfinite(v) and lo <= v <= hi


def compute_all(
    elevation: np.ndarray,
    land: np.ndarray,
    regions: np.ndarray | None = None,
    region_names: dict[int, str] | None = None,
    region_groups: dict[str, list[int]] | None = None,
    cfg: dict | None = None,
    mountain_exclude: np.ndarray | None = None,
    radius_km: float = 6371.0,
    return_fields: bool = False,
) -> dict | tuple[dict, dict]:
    cfg = cfg or load_config()
    grid = LatLonGrid.from_shape(elevation.shape, radius_km)
    land = land.astype(bool)
    area = grid.cell_area_km2
    timings: dict[str, float] = {}
    fields: dict[str, np.ndarray] = {}

    def tick(name, t0):
        timings[name] = round(time.perf_counter() - t0, 2)

    # ---- A 层 ------------------------------------------------------------------
    A = {}
    ca = cfg["A_hard_constraints"]
    t0 = time.perf_counter()
    ocean_frac = 1.0 - hard.area_fraction(grid, land)
    A["ocean_fraction"] = {"value": ocean_frac, "target": ca["ocean_fraction"]["target"],
                           "pass": _in_range(ocean_frac, *ca["ocean_fraction"]["target"])}
    units = hard.landmass_units(grid, land, ca["main_continent"]["thin_link_width_km"])
    land_area = float((area * land).sum())
    main_area = float(units["areas_km2"][units["main_id"]]) if units["n"] else 0.0
    share = main_area / land_area if land_area else 0.0
    A["main_continent"] = {"area_km2": main_area, "share_of_land": share,
                           "target": ca["main_continent"]["share_of_land"]["target"],
                           "pass": _in_range(share, *ca["main_continent"]["share_of_land"]["target"]),
                           "n_units": units["n"], "n_components_ignoring_thin_link_rule": units["n_without_thin_link_rule"]}
    fields["units"] = units["labels"]
    main_mask = units["main_mask"]
    tick("A_units", t0)

    t0 = time.perf_counter()
    cs = ca["subcontinents"]
    sub = hard.subcontinents(grid, main_mask, cs["neck_radius_km"], cs["min_area_km2"])
    A["subcontinents"] = {"count": sub["count"], "areas_km2": sub["areas_km2"],
                          "target": cs["count"]["target"], "pass": _in_range(sub["count"], *cs["count"]["target"])}
    fields["subcontinents"] = sub["labels"]
    tick("A_subcontinents", t0)

    t0 = time.perf_counter()
    cis = ca["main_inland_sea"]
    seas = hard.inland_seas(grid, land, main_mask, cis["strait_max_width_km"], cis["min_main_border_fraction"],
                            cis["min_candidate_area_km2"])
    check = hard.main_inland_sea_check(seas["seas"], cis)
    if check.get("exists"):
        geo = hard.inland_sea_geometry(grid, land, seas["region_labels"] == check["main"]["region_id"], sub["labels"],
                                       sub["big_ids"], cis.get("bordering_subcontinents", {}).get("min_border_share", 0.0))
        check = hard.apply_inland_sea_geometry(check, geo, cis)
    A["main_inland_sea"] = {**check, "pass": bool(check.get("passes", False)),
                            "candidate_seas": seas["seas"][:12]}
    fields["sea_regions"] = seas.get("region_labels")
    tick("A_inland_seas", t0)

    rows = np.flatnonzero(main_mask.any(axis=1))
    north = float(grid.lat[rows.min()]) if rows.size else float("nan")
    south = float(grid.lat[rows.max()]) if rows.size else float("nan")
    cl = ca["main_continent_latitude"]
    A["main_continent_latitude"] = {"north_deg": north, "south_deg": south,
                                    "target": {"north_min": cl["north_min_deg"], "south_max": cl["south_max_deg"]},
                                    "pass": north >= cl["north_min_deg"] and south <= cl["south_max_deg"]}

    cp = ca["polar_continent"]
    clat = hard.unit_centroid_latitudes(grid, units["labels"], units["n"])
    polar = [{"area_km2": float(units["areas_km2"][k]), "centroid_lat_deg": float(clat[k])}
             for k in range(1, units["n"] + 1)
             if abs(clat[k]) >= cp["min_abs_centroid_lat_deg"] and cp["area_km2"][0] <= units["areas_km2"][k] <= cp["area_km2"][1]]
    A["polar_continent"] = {"found": polar, "pass": bool(polar), "criteria": cp}

    soft = (hard.soft_limits(grid, units, clat, south, cfg["soft_limits"], cp["min_abs_centroid_lat_deg"],
                             cfg["B1_coast"]["continent_min_area_km2"]) if "soft_limits" in cfg else None)

    t0 = time.perf_counter()
    oo = hard.open_ocean_emptiness(grid, land)
    A["open_ocean"] = {**oo, "target_min_diameter_km": ca["open_ocean"]["min_land_free_diameter_km"],
                       "pass": oo["land_free_circle_diameter_km"] >= ca["open_ocean"]["min_land_free_diameter_km"]}
    tick("A_open_ocean", t0)

    # ---- 区域划分 -----------------------------------------------------------------
    cb = cfg["B1_coast"]
    if regions is None:
        # 虚构世界：区域 = 陆块单元；只给面积 ≥ continent_min_area_km2 的单元单独命名统计
        big = [k for k in range(1, units["n"] + 1) if units["areas_km2"][k] >= cb["continent_min_area_km2"]]
        regions = units["labels"]
        region_names = {k: ("main_continent" if k == units["main_id"] else f"unit_{k}") for k in big}
    region_ids = sorted(int(r) for r in region_names)

    # ---- B1 海岸复杂度 ---------------------------------------------------------------
    B1 = {}
    t0 = time.perf_counter()
    coast = coastal.coastline_stats(grid, land, regions, cb["fractal"])
    B1["coastline_length_km"] = coast["total_length_km"]
    B1["fractal_dimension"] = coast["fractal_global"]
    B1["fractal_box_sizes_km"] = coast["box_sizes_km"]
    tick("B1_coastline", t0)

    t0 = time.perf_counter()
    pen, bay, pen_attr, bay_attr = {}, {}, {}, {}
    for r_km in cb["peninsula_radii_km"]:
        v, removed = coastal.peninsula_index(grid, land, r_km)
        pen[str(r_km)] = v
        pen_attr[str(r_km)] = coastal.attribute_cells(grid, land, regions, removed, False)
        fields[f"peninsula_removed_{r_km}"] = removed
    for r_km in cb["bay_radii_km"]:
        v, filled = coastal.bay_index(grid, land, r_km)
        bay[str(r_km)] = v
        bay_attr[str(r_km)] = coastal.attribute_cells(grid, land, regions, filled, True)
        fields[f"bay_filled_{r_km}"] = filled
    B1["peninsula_index"] = pen
    B1["bay_index"] = bay
    tick("B1_morphology", t0)

    if "complexity_spread" in cb:
        # U18(a)：海岸复杂度的空间分布（只作统计，不设合格线）；窗口留给区域放大图的选取复用
        t0 = time.perf_counter()
        cs_cfg = cb["complexity_spread"]
        cw = coast_windows.CoastWindows(grid, land, cs_cfg["window_km"], cs_cfg["smooth_radius_km"],
                                        cs_cfg["candidate_step_km"])
        B1["coast_complexity_spread"] = coast_windows.complexity_spread(cw)
        fields["coast_windows"] = cw
        tick("B1_complexity_spread", t0)

    t0 = time.perf_counter()
    B1["islands"] = coastal.island_stats(grid, land, cb["continent_min_area_km2"], cb["islands"]["min_area_pixels"])
    B1["shelf_fraction_of_ocean"] = coastal.shelf_fraction(grid, land, elevation, cb["shelf_depth_m"])
    tick("B1_islands_shelf", t0)

    # ---- B2 距海距离 -----------------------------------------------------------------
    t0 = time.perf_counter()
    B2, dist = coastal.distance_to_sea(grid, land, cfg["B2_dryness"]["distance_to_sea_thresholds_km"])
    fields["distance_to_sea_km"] = dist
    tick("B2_distance", t0)

    # ---- 分区域（单个区域 + 组合区域，例如"亚非欧"）--------------------------------------------
    t0 = time.perf_counter()
    groups = {region_names[r]: [r] for r in region_ids}
    groups.update(region_groups or {})
    groups["all_land"] = [int(i) for i in np.unique(regions[land]) if i > 0]
    by_region = {}
    for gname, ids in groups.items():
        m = land & np.isin(regions, ids)
        g_area = float((area * m).sum())
        cg = coastal.coastline_for_group(coast, ids)
        q = weighted_quantiles(dist[m], area[m], [0.5, 0.9]) if m.any() else [float("nan")] * 2
        by_region[gname] = {
            "region_ids": [int(i) for i in ids],
            "area_km2": g_area,
            "coastline_length_km": cg["length_km"],
            "coastline_development": coastline_development(cg["length_km"], g_area, grid.radius_km),
            "fractal_dimension": cg["fractal"]["D"],
            "fractal_r2": cg["fractal"].get("r2"),
            "peninsula_index": {k: coastal.group_ratio(grid, land, regions, a, ids) for k, a in pen_attr.items()},
            "bay_index": {k: coastal.group_ratio(grid, land, regions, a, ids) for k, a in bay_attr.items()},
            "distance_to_sea_median_km": float(q[0]),
            "distance_to_sea_p90_km": float(q[1]),
            "land_fraction_beyond_1000km": float((area * m * (dist > 1000)).sum() / g_area) if g_area else float("nan"),
        }
    tick("by_region", t0)

    # ---- C 层 -------------------------------------------------------------------------
    cc = cfg["C_earthlike"]
    t0 = time.perf_counter()
    C = {"hypsometry": earthlike.hypsometry(grid, elevation, land, cc["hypsometry_bin_m"])}
    slope_stats, slope = earthlike.slope_distribution(grid, elevation, land)
    C["slope"] = slope_stats
    fields["slope_deg"] = slope
    tick("C_hypsometry_slope", t0)
    t0 = time.perf_counter()
    mtn, mtn_mask = earthlike.mountain_systems(grid, elevation, land, cc["mountains"], exclude=mountain_exclude)
    C["mountains"] = mtn
    fields["mountain_systems"] = mtn_mask
    tick("C_mountains", t0)

    result = {
        "grid": grid.name,
        "pixel_km_equator": grid.pixel_km_equator,
        "A": A,
        "soft_limits": soft,
        "B1": B1,
        "B2": B2,
        "C": C,
        "by_region": by_region,
        "resolution_warnings": resolution_warnings(grid, cfg),
        "timings_s": timings,
        "config_version": cfg.get("version"),
    }
    return (result, fields) if return_fields else result


def resolution_warnings(grid: LatLonGrid, cfg: dict) -> list[str]:
    """形态学判定以格点中心距离定义"半径 r 的圆盘"。当格点间距（南北向）大于 r 时，该判定在此分辨率下无法分辨，
    结果不可靠（阶段 0.5 实测：512x256 格网上一个格点宽的海峡不会被切断，地中海与大西洋连成一片）。"""
    pix = np.pi * grid.radius_km / grid.ny
    out = []
    ca, cb = cfg["A_hard_constraints"], cfg["B1_coast"]
    checks = [("主大陆细线连接（半径）", ca["main_continent"]["thin_link_width_km"] / 2),
              ("主内海海峡（半径）", ca["main_inland_sea"]["strait_max_width_km"] / 2)]
    checks += [(f"半岛指数 r={r} km", r) for r in cb["peninsula_radii_km"]]
    checks += [(f"海湾指数 r={r} km", r) for r in cb["bay_radii_km"]]
    for name, r in checks:
        if pix > r:
            out.append(f"{name}：判定半径 {r:g} km 小于格点间距 {pix:.0f} km，此分辨率下不可靠")
    return out
