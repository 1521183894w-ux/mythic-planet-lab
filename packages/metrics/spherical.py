"""球面距离与球面形态学（以格点中心的大圆距离定义"半径为 r 的圆盘"）。

定义（离散、自洽，地球基准与虚构世界同一套）：
- 距离：格点中心之间的大圆距离；
- 腐蚀 erode(A, r)  = { p ∈ A : 到 A 之外最近格点中心的距离 > r }；
- 膨胀 dilate(A, r) = { p : 到 A 中最近格点中心的距离 ≤ r }；
- 开运算 = 先腐蚀后膨胀（去掉宽度 < 2r 的凸出部分，例如半岛、细地峡）；
- 闭运算 = 先膨胀后腐蚀（填掉宽度 < 2r 的凹入部分，例如海湾、窄海峡）。
注意：r 小于格点间距时运算基本不起作用，这是格网分辨率的真实限制，不做修正。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .grid import LatLonGrid, chord_to_km, km_to_chord
from .topology import inner_boundary


def distance_to_set_km(
    grid: LatLonGrid,
    target: np.ndarray,
    query: np.ndarray | None = None,
    max_km: float | None = None,
) -> np.ndarray:
    """query 中各格点到 target 中最近格点的大圆距离（km）。
    target 内部的格点距离为 0；query 之外为 NaN；超过 max_km 的记为 +inf。"""
    if query is None:
        query = np.ones(grid.shape, dtype=bool)
    out = np.full(grid.shape, np.nan, dtype="float64")
    if not target.any():
        out[query] = np.inf
        return out
    xyz = grid.xyz
    # 对 target 之外的点，最近的 target 格点一定在 target 的边界上（与非 target 相邻）
    seeds = inner_boundary(target)
    if not seeds.any():  # target 覆盖全球
        out[query] = 0.0
        return out
    tree = cKDTree(xyz[seeds])
    q = query & ~target
    ub = np.inf if max_km is None else float(km_to_chord(max_km, grid.radius_km)) * (1 + 1e-9)
    chord, _ = tree.query(xyz[q], k=1, distance_upper_bound=ub, workers=-1)
    d = np.where(np.isfinite(chord), chord_to_km(np.where(np.isfinite(chord), chord, 0.0), grid.radius_km), np.inf)
    out[q] = d
    out[query & target] = 0.0
    return out


def nearest_index(grid: LatLonGrid, source: np.ndarray, query: np.ndarray) -> np.ndarray:
    """query 中各格点最近的 source 格点的扁平索引（用于把格点归属到最近的区域）。"""
    xyz = grid.xyz
    src_idx = np.flatnonzero(source.ravel())
    tree = cKDTree(xyz.reshape(-1, 3)[src_idx])
    _, k = tree.query(xyz[query], k=1, workers=-1)
    return src_idx[k]


def erode(grid: LatLonGrid, mask: np.ndarray, r_km: float) -> np.ndarray:
    d = distance_to_set_km(grid, ~mask, query=mask, max_km=r_km)
    return mask & (d > r_km)


def dilate(grid: LatLonGrid, mask: np.ndarray, r_km: float) -> np.ndarray:
    d = distance_to_set_km(grid, mask, query=~mask, max_km=r_km)
    return mask | ((~mask) & (d <= r_km))


def opening(grid: LatLonGrid, mask: np.ndarray, r_km: float) -> np.ndarray:
    return dilate(grid, erode(grid, mask, r_km), r_km) & mask


def closing(grid: LatLonGrid, mask: np.ndarray, r_km: float) -> np.ndarray:
    return erode(grid, dilate(grid, mask, r_km), r_km) | mask
