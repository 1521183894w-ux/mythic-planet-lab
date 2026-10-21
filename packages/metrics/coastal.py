"""B1 海岸复杂度与 B2 距海距离（docs/quality_metrics.md §B）。

所有"按区域"的统计接受一个区域标签图（地球用大洲编号，虚构世界用陆块单元），
海岸线点、被削掉/被填充的格点都归属到最近陆地格点所在的区域。
"""

from __future__ import annotations

import numpy as np

from .coast import (
    box_counting_dimension,
    coastline_development,
    coastline_segments,
    default_box_sizes_km,
    densify_km,
    polyline_lengths_km,
    to_xyz,
)
from .distributions import powerlaw_ccdf_slope, weighted_quantiles
from .grid import LatLonGrid
from .spherical import closing, distance_to_set_km, nearest_index, opening
from .topology import component_sums, label


def _region_of_points(grid: LatLonGrid, land: np.ndarray, regions: np.ndarray, xyz_unit: np.ndarray) -> np.ndarray:
    """三维单位向量点 -> 最近陆地格点的区域编号。"""
    from scipy.spatial import cKDTree

    flat = np.flatnonzero(land.ravel())
    tree = cKDTree(grid.xyz.reshape(-1, 3)[flat])
    _, k = tree.query(xyz_unit, k=1, workers=-1)
    return regions.ravel()[flat[k]]


def coastline_stats(grid: LatLonGrid, land: np.ndarray, regions: np.ndarray, cfg: dict) -> dict:
    """海岸线：全球长度与分形维数，以及按区域分组所需的原始数据
    （每段海岸的长度与所属区域、加密采样点与所属区域），分组统计见 coastline_for_group。"""
    segs = [s for s in coastline_segments(grid, land) if len(s) > 1]
    lens, mids = [], []
    for s in segs:
        lens.append(polyline_lengths_km(grid, s))
        m = to_xyz(s[:, 0], s[:, 1])
        m = (m[1:] + m[:-1]) / 2.0
        mids.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    seg_len = np.concatenate(lens) if lens else np.zeros(0)
    seg_reg = _region_of_points(grid, land, regions, np.concatenate(mids)) if mids else np.zeros(0, int)

    sizes = default_box_sizes_km(grid, cfg["min_size_km"], cfg["max_size_km"], cfg["n_sizes"])
    step = min(sizes[0] / 4.0, np.pi * grid.radius_km / grid.ny / 2.0)
    pts = densify_km(grid, segs, step)
    pts_reg = _region_of_points(grid, land, regions, pts / grid.radius_km) if len(pts) else np.zeros(0, int)
    return {"total_length_km": float(seg_len.sum()), "fractal_global": box_counting_dimension(pts, sizes),
            "box_sizes_km": sizes.tolist(), "n_segments": len(segs),
            "_seg_len": seg_len, "_seg_reg": seg_reg, "_pts": pts, "_pts_reg": pts_reg, "_sizes": sizes}


def coastline_for_group(coast: dict, ids: list[int]) -> dict:
    """某个区域组（一个或多个区域编号）的海岸线长度与分形维数。"""
    L = float(coast["_seg_len"][np.isin(coast["_seg_reg"], ids)].sum())
    sub = coast["_pts"][np.isin(coast["_pts_reg"], ids)]
    D = box_counting_dimension(sub, coast["_sizes"]) if len(sub) > 50 else {"D": float("nan"), "r2": float("nan")}
    return {"length_km": L, "fractal": D}


def peninsula_index(grid: LatLonGrid, land: np.ndarray, r_km: float) -> tuple[float, np.ndarray]:
    """半岛指数：半径 r 的开运算削掉的陆地面积 ÷ 陆地面积。返回 (全球值, 被削掉的格点)。"""
    area = grid.cell_area_km2
    removed = land & ~opening(grid, land, r_km)
    return float((area * removed).sum() / (area * land).sum()), removed


def bay_index(grid: LatLonGrid, land: np.ndarray, r_km: float) -> tuple[float, np.ndarray]:
    """海湾指数：半径 r 的闭运算填充的海域面积 ÷ 陆地面积。返回 (全球值, 被填充的格点)。"""
    area = grid.cell_area_km2
    filled = closing(grid, land, r_km) & ~land
    return float((area * filled).sum() / (area * land).sum()), filled


def attribute_cells(grid: LatLonGrid, land: np.ndarray, regions: np.ndarray, cells: np.ndarray,
                    assign_by_nearest: bool) -> np.ndarray:
    """给一组格点（被削掉的陆地或被填充的海）标上区域编号：陆地格点直接取所在区域，
    海洋格点（assign_by_nearest）取最近陆地格点的区域。返回标签图，0 表示不在 cells 中。"""
    lab = np.zeros(grid.shape, np.int64)
    if not cells.any():
        return lab
    if assign_by_nearest:
        lab[cells] = regions.ravel()[nearest_index(grid, land, cells)]
    else:
        lab[cells] = regions[cells]
    return lab


def group_ratio(grid: LatLonGrid, land: np.ndarray, regions: np.ndarray, attributed: np.ndarray, ids: list[int]) -> float:
    """区域组内：被归属格点的面积 ÷ 该组陆地面积。"""
    area = grid.cell_area_km2
    denom = (area * (land & np.isin(regions, ids))).sum()
    return float((area * np.isin(attributed, ids) * (attributed > 0)).sum() / denom) if denom > 0 else float("nan")


def island_stats(grid: LatLonGrid, land: np.ndarray, continent_min_area_km2: float, min_area_pixels: float) -> dict:
    """岛屿：面积小于 continent_min_area_km2 的连通陆块（8 连通、经度环绕）。
    统计各数量级的个数，并在 [分辨率下限, 最大岛] 内拟合 Korčak 累积分布 N(≥A) ∝ A^(−b)。"""
    lab, n = label(land)
    areas = component_sums(lab, n, grid.cell_area_km2)[1:]
    islands = areas[areas < continent_min_area_km2]
    pix_eq = (np.pi * grid.radius_km / grid.ny) ** 2  # 赤道附近一个格点的面积
    a_min = min_area_pixels * pix_eq
    bins = [1e2, 1e3, 1e4, 1e5, 1e6, continent_min_area_km2]
    counts = {f"{bins[i]:.0e}-{bins[i + 1]:.0e}": int(((islands >= bins[i]) & (islands < bins[i + 1])).sum())
              for i in range(len(bins) - 1)}
    fit = powerlaw_ccdf_slope(islands, s_min=a_min, s_max=float(islands.max()) if islands.size else a_min)
    return {"n_landmasses": int(n), "n_islands": int(islands.size), "n_continents": int((areas >= continent_min_area_km2).sum()),
            "continent_areas_km2": sorted([float(a) for a in areas[areas >= continent_min_area_km2]], reverse=True),
            "largest_islands_km2": [float(a) for a in np.sort(islands)[::-1][:10]],
            "count_by_area_km2": counts, "korcak": fit, "resolution_min_area_km2": float(a_min)}


def shelf_fraction(grid: LatLonGrid, land: np.ndarray, elevation: np.ndarray, depth_m: float) -> float:
    """大陆架：水深 0–depth_m 的海域面积 ÷ 海洋面积（海洋中高程 ≥ 0 的格点，例如冰架，不计入大陆架）。"""
    area = grid.cell_area_km2
    ocean = ~land
    shelf = ocean & (elevation < 0) & (elevation >= -depth_m)
    return float((area * shelf).sum() / (area * ocean).sum())


def distance_to_sea(grid: LatLonGrid, land: np.ndarray, thresholds_km: list[float]) -> tuple[dict, np.ndarray]:
    """B2：陆地各格点到最近海洋格点的大圆距离分布（面积加权）。"""
    d = distance_to_set_km(grid, ~land, query=land)
    w = grid.cell_area_km2 * land
    q = weighted_quantiles(d[land], grid.cell_area_km2[land], [0.5, 0.9, 0.99])
    k = int(np.nanargmax(np.where(land, d, -1)))
    i, j = divmod(k, grid.nx)
    out = {"median_km": float(q[0]), "p90_km": float(q[1]), "p99_km": float(q[2]),
           "max_km": float(d.ravel()[k]), "max_at_lat_lon": [float(grid.lat[i]), float(grid.lon[j])]}
    for t in thresholds_km:
        out[f"land_fraction_beyond_{int(t)}km"] = float((w * (d > t)).sum() / w.sum())
    return out, d
