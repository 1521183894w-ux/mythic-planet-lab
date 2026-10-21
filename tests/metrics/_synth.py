"""单元测试用的合成形状（答案可解析计算）。"""

from __future__ import annotations

import numpy as np

from mythic.metrics.grid import LatLonGrid


def angular_distance_deg(grid: LatLonGrid, lat0: float, lon0: float) -> np.ndarray:
    """各格点中心到 (lat0, lon0) 的角距离（度）。"""
    c = np.array([np.cos(np.radians(lat0)) * np.cos(np.radians(lon0)),
                  np.cos(np.radians(lat0)) * np.sin(np.radians(lon0)),
                  np.sin(np.radians(lat0))])
    return np.degrees(np.arccos(np.clip(grid.xyz @ c, -1.0, 1.0)))


def cap(grid: LatLonGrid, lat0: float, lon0: float, radius_deg: float) -> np.ndarray:
    """球冠：到中心角距离 < radius_deg 的格点。"""
    return angular_distance_deg(grid, lat0, lon0) < radius_deg


def cap_area_km2(grid: LatLonGrid, radius_deg: float) -> float:
    return 2 * np.pi * grid.radius_km**2 * (1 - np.cos(np.radians(radius_deg)))
