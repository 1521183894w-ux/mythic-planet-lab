"""全球等经纬度格网（等距圆柱，2:1）。

约定（地球基准与虚构世界共用）：
- 数组形状 (ny, nx)，nx = 2·ny；第 0 行在最北，第 0 列从 180°W 开始，经度向东增大。
- 格点代表一个"面元"（像元配准），中心在 lat = 90 - (i+0.5)·180/ny，lon = -180 + (j+0.5)·360/nx。
- 面元的球面面积：A = R² · Δλ · (sin φ_北 − sin φ_南)，全部面元之和 = 4πR²。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class LatLonGrid:
    nx: int
    ny: int
    radius_km: float = EARTH_RADIUS_KM

    def __post_init__(self):
        if self.nx != 2 * self.ny:
            raise ValueError(f"格网必须是 2:1（nx = 2·ny），得到 {self.nx}x{self.ny}")

    @classmethod
    def from_shape(cls, shape: tuple[int, int], radius_km: float = EARTH_RADIUS_KM) -> "LatLonGrid":
        ny, nx = shape
        return cls(nx=nx, ny=ny, radius_km=radius_km)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def name(self) -> str:
        return f"{self.nx}x{self.ny}"

    @property
    def dlat_deg(self) -> float:
        return 180.0 / self.ny

    @property
    def dlon_deg(self) -> float:
        return 360.0 / self.nx

    @cached_property
    def lat(self) -> np.ndarray:
        """格点中心纬度（度），自北向南。"""
        return 90.0 - (np.arange(self.ny) + 0.5) * self.dlat_deg

    @cached_property
    def lon(self) -> np.ndarray:
        """格点中心经度（度），自 -180 向东。"""
        return -180.0 + (np.arange(self.nx) + 0.5) * self.dlon_deg

    @cached_property
    def lat_edges(self) -> np.ndarray:
        return 90.0 - np.arange(self.ny + 1) * self.dlat_deg

    @cached_property
    def row_area_km2(self) -> np.ndarray:
        """每一行单个面元的面积（km²），形状 (ny,)。"""
        s = np.sin(np.radians(self.lat_edges))
        return self.radius_km**2 * np.radians(self.dlon_deg) * (s[:-1] - s[1:])

    @cached_property
    def cell_area_km2(self) -> np.ndarray:
        """面元面积（km²），形状 (ny, nx)。"""
        return np.broadcast_to(self.row_area_km2[:, None], self.shape).copy()

    @property
    def total_area_km2(self) -> float:
        return 4.0 * np.pi * self.radius_km**2

    @property
    def pixel_km_equator(self) -> float:
        """赤道处一个格点的东西向宽度（km）。"""
        return 2.0 * np.pi * self.radius_km / self.nx

    @cached_property
    def xyz(self) -> np.ndarray:
        """格点中心的单位向量，形状 (ny, nx, 3)。"""
        lat = np.radians(self.lat)[:, None]
        lon = np.radians(self.lon)[None, :]
        return np.stack(
            np.broadcast_arrays(np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
        ).astype("float64")

    def transform(self):
        """rasterio 仿射变换（EPSG:4326，像元配准）。"""
        from rasterio.transform import from_origin

        return from_origin(-180.0, 90.0, self.dlon_deg, self.dlat_deg)

    def rowcol(self, lat_deg, lon_deg):
        """经纬度 -> (行, 列) 索引（含经度环绕）。"""
        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = (np.asarray(lon_deg, dtype=float) + 180.0) % 360.0 - 180.0
        i = np.clip(((90.0 - lat_deg) / self.dlat_deg).astype(int), 0, self.ny - 1)
        j = np.clip(((lon_deg + 180.0) / self.dlon_deg).astype(int), 0, self.nx - 1)
        return i, j


def chord_to_km(chord, radius_km: float = EARTH_RADIUS_KM):
    """单位球上的弦长 -> 大圆距离（km）。"""
    return 2.0 * radius_km * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0.0, 1.0))


def km_to_chord(km, radius_km: float = EARTH_RADIUS_KM):
    """大圆距离（km）-> 单位球上的弦长。"""
    return 2.0 * np.sin(np.clip(np.asarray(km) / radius_km, 0.0, np.pi) / 2.0)
