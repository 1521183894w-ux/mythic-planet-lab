"""海岸线：提取、球面长度、三维盒计数分形维数、海岸发育系数。

海岸线 = 陆地掩膜的 0.5 等值线（marching squares），经度接缝单独处理。
分形维数用"三维盒计数"：把海岸线点放到以 km 为单位的三维空间里，数被占据的立方格子数。
在格子远小于星球半径时，球面局部近似平面，这等价于平面盒计数，但不受任何地图投影畸变影响。
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from .grid import LatLonGrid, chord_to_km


def coastline_segments(grid: LatLonGrid, land: np.ndarray) -> list[np.ndarray]:
    """返回海岸线折线列表，每条为 (k, 2) 的 [纬度, 经度]（度）。
    在右侧补一列（=第 0 列）以捕获跨接缝的海岸；接缝处的海岸线会被切成两段，总长度不受影响。"""
    padded = np.concatenate([land, land[:, :1]], axis=1).astype(np.float32)
    segs = []
    for c in measure.find_contours(padded, 0.5):
        lat = 90.0 - (c[:, 0] + 0.5) * grid.dlat_deg
        lon = -180.0 + (c[:, 1] + 0.5) * grid.dlon_deg
        segs.append(np.column_stack([lat, lon]))
    return segs


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def polyline_lengths_km(grid: LatLonGrid, seg: np.ndarray) -> np.ndarray:
    """折线每一段的大圆长度（km）。"""
    p = to_xyz(seg[:, 0], seg[:, 1])
    return chord_to_km(np.linalg.norm(np.diff(p, axis=0), axis=1), grid.radius_km)


def coastline_length_km(grid: LatLonGrid, segments: list[np.ndarray]) -> float:
    return float(sum(polyline_lengths_km(grid, s).sum() for s in segments if len(s) > 1))


def densify_km(grid: LatLonGrid, segments: list[np.ndarray], step_km: float) -> np.ndarray:
    """沿海岸线按约 step_km 的间距采样，返回三维坐标（km），形状 (N, 3)。"""
    out = []
    for s in segments:
        if len(s) < 2:
            continue
        p = to_xyz(s[:, 0], s[:, 1])
        seglen = chord_to_km(np.linalg.norm(np.diff(p, axis=0), axis=1), grid.radius_km)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        n = max(int(np.ceil(cum[-1] / step_km)) + 1, 2)
        t = np.linspace(0.0, cum[-1], n)
        q = np.column_stack([np.interp(t, cum, p[:, k]) for k in range(3)])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        out.append(q)
    if not out:
        return np.zeros((0, 3))
    return np.concatenate(out) * grid.radius_km


def default_box_sizes_km(grid: LatLonGrid, s_min_floor: float = 10.0, s_max: float = 500.0, n: int = 8) -> np.ndarray:
    """盒子尺度：下限取 max(10 km, 2 个格点南北跨度)，上限 500 km；不足 1 个数量级时把上限放宽到下限的 8 倍。"""
    pix = np.pi * grid.radius_km / grid.ny
    lo = max(s_min_floor, 2.0 * pix)
    hi = max(s_max, 8.0 * lo)
    return np.geomspace(lo, hi, n)


def box_count(points_km: np.ndarray, size_km: float, n_offsets: int = 4, seed: int = 0) -> float:
    """三维盒计数：被占据的边长 size_km 的立方格子数（对若干随机网格偏移取平均）。"""
    if len(points_km) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_offsets):
        off = rng.uniform(0.0, size_km, 3)
        k = np.floor((points_km + off) / size_km).astype(np.int64)
        k -= k.min(axis=0)
        span = k.max(axis=0) + 1
        key = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
        counts.append(np.unique(key).size)
    return float(np.mean(counts))


def box_counting_dimension(points_km: np.ndarray, sizes_km: np.ndarray) -> dict:
    """D = log N(s) 对 log(1/s) 的拟合斜率。返回维数、各尺度计数与拟合 R²。"""
    counts = np.array([box_count(points_km, s) for s in sizes_km])
    ok = counts > 0
    if ok.sum() < 3:
        return {"D": float("nan"), "sizes_km": sizes_km.tolist(), "counts": counts.tolist(), "r2": float("nan")}
    x, y = np.log(1.0 / sizes_km[ok]), np.log(counts[ok])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return {"D": float(slope), "sizes_km": [round(float(s), 1) for s in sizes_km],
            "counts": [round(float(c), 1) for c in counts], "r2": float(r2)}


def equal_area_circle_perimeter_km(area_km2: float, radius_km: float) -> float:
    """面积为 A 的球冠的周长（球面上的"等面积圆"）。"""
    cos_t = 1.0 - area_km2 / (2.0 * np.pi * radius_km**2)
    return float(2.0 * np.pi * radius_km * np.sqrt(max(0.0, 1.0 - cos_t**2)))


def coastline_development(length_km: float, area_km2: float, radius_km: float) -> float:
    """海岸发育系数 = 海岸线长度 ÷ 等面积（球面）圆的周长。圆形陆地约为 1。"""
    per = equal_area_circle_perimeter_km(area_km2, radius_km)
    return float(length_km / per) if per > 0 else float("nan")
