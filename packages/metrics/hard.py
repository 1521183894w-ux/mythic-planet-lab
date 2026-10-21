"""A 层硬约束（docs/quality_metrics.md §A）。只依赖陆地掩膜（和格网）。"""

from __future__ import annotations

import numpy as np

from .earthlike import shape_axes
from .grid import LatLonGrid
from .spherical import distance_to_set_km, erode, nearest_index, opening
from .topology import any_neighbor, component_sums, label


def area_fraction(grid: LatLonGrid, mask: np.ndarray) -> float:
    a = grid.cell_area_km2
    return float((a * mask).sum() / a.sum())


def landmass_units(grid: LatLonGrid, land: np.ndarray, thin_link_width_km: float) -> dict:
    """把陆地划分为"陆块单元"：宽度小于 thin_link_width_km 的细线连接不算相连。

    做法：先对陆地做半径 w/2 的球面开运算，开运算后仍存在的连通区域为"核心"；
    被削掉的格点归属到同一原始连通区域内最近的核心；没有核心的小陆块（小岛）各自成为一个单元。
    返回 labels（0 = 海洋）、各单元面积（km²，下标 0 为海洋）、主大陆编号与掩膜。
    """
    area = grid.cell_area_km2
    opened = opening(grid, land, thin_link_width_km / 2.0)
    core_lab, n_core = label(opened)
    orig_lab, n_orig = label(land)

    has_core = np.zeros(n_orig + 1, bool)
    has_core[np.unique(orig_lab[opened])] = True
    has_core[0] = False
    rest = land & ~opened
    r_with = rest & has_core[orig_lab]
    r_without = rest & ~has_core[orig_lab]

    unit = np.zeros(grid.shape, np.int64)
    unit[opened] = core_lab[opened]
    unit[r_without] = n_core + orig_lab[r_without]
    if r_with.any():
        idx = nearest_index(grid, opened, r_with)
        vals = core_lab.ravel()[idx]
        same = orig_lab.ravel()[idx] == orig_lab[r_with]
        if not same.all():
            # 最近的核心落在另一个原始陆块上（隔着窄海峡），改为只在本陆块的核心中找
            qflat = np.flatnonzero(r_with.ravel())
            bad = np.flatnonzero(~same)
            for o in np.unique(orig_lab.ravel()[qflat[bad]]):
                sel = bad[orig_lab.ravel()[qflat[bad]] == o]
                qmask = np.zeros(land.size, bool)
                qmask[qflat[sel]] = True
                idx2 = nearest_index(grid, opened & (orig_lab == o), qmask.reshape(grid.shape))
                vals[sel] = core_lab.ravel()[idx2]
        unit[r_with] = vals

    # 重新编号为 1..n
    ids = np.unique(unit[land])
    remap = np.zeros(unit.max() + 1, np.int64)
    remap[ids] = np.arange(1, ids.size + 1)
    unit = remap[unit]
    n = int(ids.size)
    areas = component_sums(unit, n, area)
    main_id = int(np.argmax(areas[1:]) + 1) if n else 0
    return {
        "labels": unit,
        "n": n,
        "areas_km2": areas,
        "main_id": main_id,
        "main_mask": unit == main_id if n else np.zeros(grid.shape, bool),
        "n_without_thin_link_rule": int(n_orig),
    }


def unit_centroid_latitudes(grid: LatLonGrid, labels: np.ndarray, n: int) -> np.ndarray:
    """各单元面积加权质心的纬度（度），下标 0 无意义。"""
    w = grid.cell_area_km2
    sx = component_sums(labels, n, w * grid.xyz[..., 0])
    sy = component_sums(labels, n, w * grid.xyz[..., 1])
    sz = component_sums(labels, n, w * grid.xyz[..., 2])
    norm = np.sqrt(sx**2 + sy**2 + sz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arcsin(np.where(norm > 0, sz / norm, 0.0)))


def subcontinents(grid: LatLonGrid, main_mask: np.ndarray, neck_radius_km: float, min_area_km2: float) -> dict:
    """亚大陆：把主大陆按"瓶颈"切分。对主大陆做半径 neck_radius_km 的腐蚀，宽度小于 2r 的地峡/颈部被切断，
    剩余的核心各自代表一个亚大陆；主大陆其余格点归属最近的核心。面积 ≥ min_area_km2 的才计数。
    labels 中每个"瓣"一个编号（含面积不足下限的小瓣）；big_ids 是计数的亚大陆编号，按面积从大到小。"""
    area = grid.cell_area_km2
    core = erode(grid, main_mask, neck_radius_km)
    lab, n = label(core)
    lobes = np.zeros(grid.shape, np.int64)
    if n == 0:
        return {"count": 1 if main_mask.any() else 0, "areas_km2": [float((area * main_mask).sum())],
                "labels": main_mask.astype(int), "big_ids": [1] if main_mask.any() else []}
    lobes[core] = lab[core]
    rest = main_mask & ~core
    if rest.any():
        lobes[rest] = lab.ravel()[nearest_index(grid, core, rest)]
    areas = component_sums(lobes, n, area)[1:]
    order = [int(k) for k in np.argsort(areas)[::-1] if areas[k] >= min_area_km2]
    return {"count": len(order), "areas_km2": [float(areas[k]) for k in order], "labels": lobes,
            "big_ids": [k + 1 for k in order]}


def inland_seas(
    grid: LatLonGrid,
    land: np.ndarray,
    main_mask: np.ndarray,
    strait_max_width_km: float,
    min_main_border_fraction: float,
    min_candidate_area_km2: float = 5.0e4,
) -> dict:
    """半封闭海域：宽度小于 strait_max_width_km 的水道视为海峡。

    做法：对海洋做半径 w/2 的开运算，窄海峡被切断；开运算后最大的海域为"外洋"，其余为候选海域；
    其余海洋格点归属最近的海域。每个候选海域统计：面积、沿岸陆地中属于主大陆的比例、
    与外洋相接的海峡数（与外洋区域相邻的格点所组成的连通段数）、与其他海域相接的海峡数。"""
    area = grid.cell_area_km2
    ocean = ~land
    open_oc = opening(grid, ocean, strait_max_width_km / 2.0)
    lab, n = label(open_oc)
    if n == 0:
        return {"seas": [], "world_ocean_area_km2": 0.0}
    region = np.zeros(grid.shape, np.int64)
    region[open_oc] = lab[open_oc]
    rest = ocean & ~open_oc
    if rest.any():
        region[rest] = lab.ravel()[nearest_index(grid, open_oc, rest)]
    areas = component_sums(region, n, area)
    world = int(np.argmax(areas[1:]) + 1)

    seas = []
    for k in np.argsort(areas[1:])[::-1] + 1:
        if k == world or areas[k] < min_candidate_area_km2:
            continue
        reg = region == k
        border = land & any_neighbor(reg)
        bw = area * border
        main_frac = float((bw * main_mask).sum() / bw.sum()) if bw.sum() > 0 else 0.0
        iface = reg & any_neighbor(ocean & ~reg)
        ilab, ni = label(iface)
        to_ocean, to_seas = 0, 0
        for c in range(1, ni + 1):
            nb = any_neighbor(ilab == c) & ocean & ~reg
            if (region[nb] == world).any():
                to_ocean += 1
            else:
                to_seas += 1
        rows = np.flatnonzero(reg.any(axis=1))
        seas.append({
            "region_id": int(k),
            "area_km2": float(areas[k]),
            "main_continent_border_fraction": round(main_frac, 3),
            "straits_to_world_ocean": to_ocean,
            "straits_to_other_seas": to_seas,
            "lat_range": [float(grid.lat[rows.max()]), float(grid.lat[rows.min()])],
            "sample_lat_lon": [float(v) for v in _sample_point(grid, reg)],
        })
    return {"seas": seas, "world_ocean_area_km2": float(areas[world]), "region_labels": region, "world_id": world}


def _sample_point(grid: LatLonGrid, mask: np.ndarray) -> tuple[float, float]:
    """区域内离边界最远的格点附近的代表点（用面积加权质心最近的格点）。"""
    w = grid.cell_area_km2 * mask
    v = (grid.xyz * w[..., None]).sum(axis=(0, 1))
    v = v / np.linalg.norm(v)
    flat = np.flatnonzero(mask.ravel())
    k = flat[np.argmax(grid.xyz.reshape(-1, 3)[flat] @ v)]
    i, j = divmod(int(k), grid.nx)
    return grid.lat[i], grid.lon[j]


def main_inland_sea_check(seas: list[dict], cfg: dict) -> dict:
    """按 quality_metrics.md：主大陆内部最大的半封闭海域，面积 1–5 Mkm²，至少是第二大内部水体的 2 倍，1–3 条海峡连通外洋。"""
    inland = [s for s in seas if s["main_continent_border_fraction"] >= cfg["min_main_border_fraction"]]
    inland.sort(key=lambda s: s["area_km2"], reverse=True)
    if not inland:
        return {"exists": False, "reason": "主大陆内部没有半封闭海域"}
    main = inland[0]
    second = inland[1]["area_km2"] if len(inland) > 1 else 0.0
    lo, hi = cfg["area_km2"]["target"]
    s_lo, s_hi = cfg["straits_to_ocean"]["target"]
    checks = {
        "area_in_range": lo <= main["area_km2"] <= hi,
        "ratio_to_second_ok": second == 0.0 or main["area_km2"] / second >= cfg["min_ratio_to_second"],
        "straits_ok": s_lo <= main["straits_to_world_ocean"] <= s_hi,
    }
    return {
        "exists": True,
        "main": main,
        "second_area_km2": second,
        "ratio_to_second": (main["area_km2"] / second) if second else None,
        "checks": checks,
        "passes": all(checks.values()),
        "n_inland_candidates": len(inland),
    }


def inland_sea_geometry(grid: LatLonGrid, land: np.ndarray, sea: np.ndarray, lobes: np.ndarray, big_ids: list[int],
                        min_border_share: float) -> dict:
    """主内海的形状与接壤关系（U18(c)，2026-10-21 用户同意）。

    - 接壤亚大陆：沿岸陆地格点（与海域相邻的陆地，按面积加权）中属于各亚大陆的比例；
      比例 ≥ min_border_share 的亚大陆才算"接壤"（只在尖角处碰到的不算）。
    - 形状：海域的主轴长宽比与走向（与山系同一个 shape_axes：切平面上面积加权协方差的 sqrt(λ1/λ2)）。
      近圆的海约为 1，地中海约 3.3。海岸毛边几乎不影响它，所以噪声边缘的圆海骗不过。"""
    area = grid.cell_area_km2
    bw = area * (land & any_neighbor(sea))
    tot = float(bw.sum())
    shares = sorted(({"subcontinent_id": int(k), "border_share": float((bw * (lobes == k)).sum() / tot) if tot else 0.0}
                     for k in big_ids), key=lambda s: s["border_share"], reverse=True)
    sh = shape_axes(grid, sea)
    return {"aspect_ratio": sh["aspect_ratio"], "orientation_deg": sh["orientation_deg"],
            "subcontinent_border_shares": shares, "min_border_share": min_border_share,
            "n_bordering_subcontinents": int(sum(s["border_share"] >= min_border_share for s in shares))}


def apply_inland_sea_geometry(check: dict, geometry: dict, cfg: dict) -> dict:
    """把形状与接壤条件并入主内海判定。config 中没有对应键时只记录、不判定（便于按旧版规则复现对照样本）。"""
    if not check.get("exists"):
        return check
    checks = dict(check["checks"])
    bc = cfg.get("bordering_subcontinents")
    if bc:
        checks["bordering_subcontinents_ok"] = geometry["n_bordering_subcontinents"] >= bc["min_count"]
    if "min_aspect_ratio" in cfg:
        checks["elongation_ok"] = geometry["aspect_ratio"] >= cfg["min_aspect_ratio"]
    return {**check, "geometry": geometry, "checks": checks, "passes": all(checks.values())}


def soft_limits(grid: LatLonGrid, units: dict, centroid_lat: np.ndarray, south_deg: float, cfg: dict,
                polar_min_abs_lat_deg: float, continent_min_area_km2: float) -> dict:
    """软限制（U17，2026-10-21 用户同意）：不淘汰种子，只在阶段 1 起的种子评分中扣分（docs/quality_metrics.md §E）。

    - 北极圈以北的主大陆面积占比（U17(b)）：不限最北点（伸向高纬的窄半岛正是简报要的峡湾海岸），
      只限大片陆地深入北极圈。地球亚非欧本体为 4.6%（北端 77.5°N 只是泰梅尔半岛的尖角）。
    - 主大陆南端（U17(b)）：简报约到 40°S，超过 main_continent_south_limit_deg 扣分。
    - 次大陆（U17(c)）：主大陆之外、质心不在极区（|纬度| < polar_min_abs_lat_deg）、面积 ≥ continent_min_area_km2 的陆块，
      简报允许最多一块。"""
    area = grid.cell_area_km2
    main = units["main_mask"]
    m_area = float((area * main).sum())
    north_frac = (float((area * main * (grid.lat[:, None] > cfg["arctic_circle_deg"])).sum() / m_area)
                  if m_area else float("nan"))
    secondary = []
    for k in range(1, units["n"] + 1):
        a = float(units["areas_km2"][k])
        if k != units["main_id"] and a >= continent_min_area_km2 and abs(centroid_lat[k]) < polar_min_abs_lat_deg:
            secondary.append({"unit_id": k, "area_km2": a, "centroid_lat_deg": float(centroid_lat[k])})
    secondary.sort(key=lambda s: s["area_km2"], reverse=True)
    return {
        "note": "软限制：只扣分、不淘汰（U17）",
        "north_arctic_area": {"value": north_frac, "max": cfg["max_main_area_fraction_north_of_arctic_circle"],
                              "arctic_circle_deg": cfg["arctic_circle_deg"],
                              "ok": bool(north_frac <= cfg["max_main_area_fraction_north_of_arctic_circle"])},
        "south_tip": {"value_deg": south_deg, "limit_deg": cfg["main_continent_south_limit_deg"],
                      "ok": bool(south_deg >= cfg["main_continent_south_limit_deg"])},
        "secondary_continents": {"count": len(secondary), "max": cfg["max_secondary_continents"], "found": secondary,
                                 "ok": len(secondary) <= cfg["max_secondary_continents"]},
    }


def open_ocean_emptiness(grid: LatLonGrid, land: np.ndarray) -> dict:
    """远洋空旷度：海洋中离陆地最远的点（"海洋难抵极"）。最大无陆地圆的直径 = 2 × 该距离。"""
    d = distance_to_set_km(grid, land, query=~land)
    k = int(np.nanargmax(d))
    i, j = divmod(k, grid.nx)
    return {"max_distance_to_land_km": float(d.ravel()[k]),
            "land_free_circle_diameter_km": float(2 * d.ravel()[k]),
            "at_lat_lon": [float(grid.lat[i]), float(grid.lon[j])]}
