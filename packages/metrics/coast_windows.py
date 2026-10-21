"""海岸窗口复杂度：区域放大图的选取（packages/figures/select_regions.py）与 U18(a)"海岸复杂度的空间分布"共用这一套定义。

- 海岸线采样点：陆地掩膜 0.5 等值线按约 20 km 加密，每点代表一小段海岸长度；
- 平滑海岸：对陆地做半径 100 km 的开运算再闭运算，去掉宽度 < 200 km 的半岛、海湾和小岛后的海岸线；
- 窗口：中心沿海岸每 100 km 取一个，窗口为边长 2000 km 的方框，复杂度在其内切圆（半径 1000 km）上计算；
- 窗口内的海岸复杂度 = 内切圆内的海岸线长度 ÷ max(平滑海岸线长度, 圆的半径)。平直海岸约为 1，越破碎越大；
- "成形海岸上的窗口"：平滑后窗口内仍有 ≥ 半径长度的海岸、且中心离平滑海岸不超过平滑半径（排除只有小岛的窗口）。

U18(a)（2026-10-21 用户同意）：0.7 的噪声海岸世界 B1 全部达标，但它的海岸处处一样碎（复杂度变异系数 0.24，地球 0.52；
最平直的一段 1.45，地球 1.01）。complexity_spread 汇总成形窗口的复杂度分布，只作统计，不设合格线。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .coast import coastline_segments, densify_km, polyline_lengths_km
from .grid import LatLonGrid, chord_to_km, km_to_chord
from .spherical import closing, opening

DEFAULT_WINDOW_KM = 2000.0
SMOOTH_RADIUS_KM = 100.0
COAST_STEP_KM = 20.0
CANDIDATE_STEP_KM = 100.0
STRAIGHT_MAX = 1.5      # 复杂度 < 1.5 的窗口记为"较平直"（地球的被动陆缘、沙漠海岸多在 1.0–1.4）


def coast_points(grid: LatLonGrid, land: np.ndarray, step_km: float = COAST_STEP_KM) -> tuple[np.ndarray, np.ndarray]:
    """海岸线加密采样点（单位向量）与每点代表的海岸长度（km）。"""
    pts, wts = [], []
    for s in coastline_segments(grid, land):
        if len(s) < 2:
            continue
        p = densify_km(grid, [s], step_km) / grid.radius_km
        pts.append(p)
        wts.append(np.full(len(p), polyline_lengths_km(grid, s).sum() / len(p)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(wts)


def ball_sums(tree: cKDTree, weights: np.ndarray, centers: np.ndarray, chord: float, chunk: int = 400) -> np.ndarray:
    out = np.zeros(len(centers))
    for a in range(0, len(centers), chunk):
        for k, idx in enumerate(tree.query_ball_point(centers[a:a + chunk], chord, workers=-1)):
            out[a + k] = weights[idx].sum() if idx else 0.0
    return out


class CoastWindows:
    """一个世界的海岸采样、候选窗口与每个窗口的海岸复杂度。"""

    def __init__(self, grid: LatLonGrid, land: np.ndarray, window_km: float = DEFAULT_WINDOW_KM,
                 smooth_radius_km: float = SMOOTH_RADIUS_KM, candidate_step_km: float = CANDIDATE_STEP_KM):
        self.grid, self.land = grid, land
        self.size_km, self.radius_km, self.smooth_radius_km = window_km, window_km / 2.0, smooth_radius_km
        self.pts, self.w = coast_points(grid, land)
        smooth = closing(grid, opening(grid, land, smooth_radius_km), smooth_radius_km)
        self.spts, self.sw = coast_points(grid, smooth)
        cand, _ = coast_points(grid, land, candidate_step_km)
        self.cand = cand
        chord = float(km_to_chord(self.radius_km, grid.radius_km))
        self.L_raw = ball_sums(cKDTree(self.pts), self.w, cand, chord) if len(self.pts) else np.zeros(len(cand))
        self.L_smooth = ball_sums(cKDTree(self.spts), self.sw, cand, chord) if len(self.spts) else np.zeros(len(cand))
        # 分母下限取窗口半径：平滑海岸很短的窗口（例如只有小岛）比值不会发散
        self.complexity = self.L_raw / np.maximum(self.L_smooth, self.radius_km)
        # 候选点是否位于平滑后仍存在的海岸上（距平滑海岸 ≤ 平滑半径）：保证窗口中心真的落在一段成形的海岸上
        if len(self.spts) and len(cand):
            d, _ = cKDTree(self.spts).query(cand, workers=-1)
            self.on_smooth = chord_to_km(d, grid.radius_km) <= smooth_radius_km
        else:
            self.on_smooth = np.zeros(len(cand), bool)

    @property
    def formed(self) -> np.ndarray:
        """成形海岸上的窗口：平滑后窗口内仍有 ≥ 半径长度的海岸，且中心落在平滑海岸附近。"""
        return (self.L_smooth >= self.radius_km) & self.on_smooth


def complexity_spread(cw: CoastWindows) -> dict:
    """成形窗口的海岸复杂度分布（U18(a)，只作统计）。窗口中心沿海岸等距，所以分布按海岸长度加权。"""
    c = cw.complexity[cw.formed]
    out = {"window_km": cw.size_km, "smooth_radius_km": cw.smooth_radius_km, "n_windows": int(c.size)}
    if c.size < 10:
        return {**out, "p10": None, "p50": None, "p90": None, "p90_over_p10": None, "cv": None, "min": None,
                "straight_fraction": None, "straight_max": STRAIGHT_MAX, "note": "成形海岸窗口不足 10 个"}
    q10, q50, q90 = (float(x) for x in np.percentile(c, [10, 50, 90]))
    return {**out, "p10": q10, "p50": q50, "p90": q90, "p90_over_p10": q90 / q10, "cv": float(c.std() / c.mean()),
            "min": float(c.min()), "straight_fraction": float((c < STRAIGHT_MAX).mean()), "straight_max": STRAIGHT_MAX}
