"""C 层地球对照分布中只需高程的部分：高程曲线、坡度分布、山系统计（v1 定义）。"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from .distributions import weighted_histogram, weighted_quantiles
from .grid import LatLonGrid, chord_to_km
from .spherical import closing
from .topology import component_sums, label, roll_to_avoid_seam

QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def hypsometry(grid: LatLonGrid, elevation: np.ndarray, land: np.ndarray, bin_m: float) -> dict:
    """面积加权的高程分布（全球与陆地）。地球呈双峰：大陆台地与深海平原。"""
    w = grid.cell_area_km2
    edges = np.arange(-11000, 9000 + bin_m, bin_m)
    return {
        "quantiles_global_m": dict(zip([str(q) for q in QUANTILES],
                                       np.round(weighted_quantiles(elevation, w, QUANTILES), 1).tolist())),
        "quantiles_land_m": dict(zip([str(q) for q in QUANTILES],
                                     np.round(weighted_quantiles(elevation[land], w[land], QUANTILES), 1).tolist())),
        "histogram_global": weighted_histogram(elevation, w, edges),
        "mean_land_m": float((elevation * w * land).sum() / (w * land).sum()),
        "mean_ocean_m": float((elevation * w * ~land).sum() / (w * ~land).sum()),
    }


def slope_degrees(grid: LatLonGrid, elevation: np.ndarray) -> np.ndarray:
    """球面上的坡度（度）：东西向间距 R·cosφ·Δλ，南北向 R·Δφ，经度环绕。"""
    dy_m = np.pi * grid.radius_km * 1000.0 / grid.ny
    dx_m = 2 * np.pi * grid.radius_km * 1000.0 / grid.nx * np.cos(np.radians(grid.lat))[:, None]
    dzdx = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / (2 * dx_m)
    dzdy = np.zeros_like(elevation, dtype=float)
    dzdy[1:-1] = (elevation[:-2] - elevation[2:]) / (2 * dy_m)
    dzdy[0] = (elevation[0] - elevation[1]) / dy_m
    dzdy[-1] = (elevation[-2] - elevation[-1]) / dy_m
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def slope_distribution(grid: LatLonGrid, elevation: np.ndarray, land: np.ndarray) -> tuple[dict, np.ndarray]:
    s = slope_degrees(grid, elevation)
    w = grid.cell_area_km2
    edges = np.concatenate([np.arange(0, 2, 0.1), np.arange(2, 10, 0.5), [10, 15, 20, 30, 45, 90]])
    return {
        "quantiles_land_deg": dict(zip([str(q) for q in QUANTILES],
                                       np.round(weighted_quantiles(s[land], w[land], QUANTILES), 4).tolist())),
        "histogram_land": weighted_histogram(s[land], w[land], edges),
    }, s


def local_relief(grid: LatLonGrid, elevation: np.ndarray, radius_km: float) -> np.ndarray:
    """局部起伏：以每个格点为中心、东西与南北各 ±radius_km 的球面"方框"内的最高与最低高程之差。
    东西方向的格点数随纬度变化（高纬同样距离跨更多列）。"""
    dy_km = np.pi * grid.radius_km / grid.ny
    ry = max(1, int(round(radius_km / dy_km)))
    mx = np.empty_like(elevation, dtype=float)
    mn = np.empty_like(elevation, dtype=float)
    for i in range(grid.ny):
        dx_km = 2 * np.pi * grid.radius_km / grid.nx * max(np.cos(np.radians(grid.lat[i])), 1e-6)
        rx = min(grid.nx // 2, max(1, int(round(radius_km / dx_km))))
        row = elevation[i]
        mx[i] = ndi.maximum_filter1d(row, 2 * rx + 1, mode="wrap")
        mn[i] = ndi.minimum_filter1d(row, 2 * rx + 1, mode="wrap")
    mx = ndi.maximum_filter1d(mx, 2 * ry + 1, axis=0, mode="nearest")
    mn = ndi.minimum_filter1d(mn, 2 * ry + 1, axis=0, mode="nearest")
    return mx - mn


def _skeleton_longest_path(grid: LatLonGrid, comp: np.ndarray) -> dict:
    """山系骨架上的最长路径（大圆长度）与弯曲度。在像元空间骨架化（先把区域滚动到不跨接缝的位置）。"""
    shift = roll_to_avoid_seam(comp)
    c = np.roll(comp, shift, axis=1)
    rows = np.flatnonzero(c.any(axis=1))
    cols = np.flatnonzero(c.any(axis=0))
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = c[r0:r1, c0:c1]
    sk = skeletonize(np.pad(sub, 1))[1:-1, 1:-1]
    yy, xx = np.nonzero(sk)
    if yy.size < 2:
        return {"length_km": 0.0, "sinuosity": float("nan")}
    gi = yy + r0
    gj = (xx + c0 - shift) % grid.nx
    xyz = grid.xyz[gi, gj]
    index = -np.ones(sk.shape, np.int64)
    index[yy, xx] = np.arange(yy.size)
    a_list, b_list = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y2, x2 = yy + dy, xx + dx
        ok = (y2 >= 0) & (y2 < sk.shape[0]) & (x2 >= 0) & (x2 < sk.shape[1])
        ok[ok] = sk[y2[ok], x2[ok]]
        a_list.append(index[yy[ok], xx[ok]])
        b_list.append(index[y2[ok], x2[ok]])
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    w = chord_to_km(np.linalg.norm(xyz[a] - xyz[b], axis=1), grid.radius_km)
    g = coo_matrix((w, (a, b)), shape=(yy.size, yy.size))
    d0 = dijkstra(g, directed=False, indices=0)
    p = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    dp = dijkstra(g, directed=False, indices=p)
    q = int(np.nanargmax(np.where(np.isfinite(dp), dp, -1)))
    length = float(dp[q])
    straight = float(chord_to_km(np.linalg.norm(xyz[p] - xyz[q]), grid.radius_km))
    return {"length_km": length, "sinuosity": length / straight if straight > 0 else float("nan")}


def mountain_systems(grid: LatLonGrid, elevation: np.ndarray, land: np.ndarray, cfg: dict,
                     exclude: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """山系（v1 定义，参数见 config/metrics.yaml）：
    山地格点 = 陆地 & 高程 ≥ min_elevation_m & 局部起伏（±relief_radius_km）≥ min_relief_m；
    以半径 merge_gap_km/2 的闭运算把间断的山地连成山系；每个山系量骨架最长路径长度、平均宽度（面积/长度）、弯曲度。
    exclude 用于剔除冰盖（地球的 ETOPO 是冰面高程，南极与格陵兰冰穹不是山）。"""
    relief = local_relief(grid, elevation, cfg["relief_radius_km"])
    mtn = land & (elevation >= cfg["min_elevation_m"]) & (relief >= cfg["min_relief_m"])
    if exclude is not None:
        mtn &= ~exclude
    merged = closing(grid, mtn, cfg["merge_gap_km"] / 2.0) & land
    if exclude is not None:
        merged &= ~exclude
    lab, n = label(merged)
    areas = component_sums(lab, n, grid.cell_area_km2)
    systems = []
    for k in np.argsort(areas[1:])[::-1] + 1:
        if areas[k] < cfg["min_area_km2"]:
            break
        sk = _skeleton_longest_path(grid, lab == k)
        rows = np.flatnonzero((lab == k).any(axis=1))
        width = areas[k] / sk["length_km"] if sk["length_km"] > 0 else float("nan")
        systems.append({"area_km2": float(areas[k]), "length_km": round(sk["length_km"], 1),
                        "mean_width_km": round(float(width), 1), "sinuosity": round(sk["sinuosity"], 3),
                        "lat_range": [float(grid.lat[rows.max()]), float(grid.lat[rows.min()])]})
    long_ones = [s for s in systems if s["length_km"] >= cfg["min_length_km"]]
    return {"n_systems": len(systems), "n_longer_than_min_length": len(long_ones),
            "mountain_land_fraction": float((grid.cell_area_km2 * mtn).sum() / (grid.cell_area_km2 * land).sum()),
            "systems": systems[:20]}, merged
