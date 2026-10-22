"""球面网格（stripy 二十面体网格，带缓存）与栅格化。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

from .geometry import R_KM, km_from_chord, unit, xyz_to_lonlat

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "tectonics"
# KD 树查询线程数；并行跑多个种子时设环境变量 MYTHIC_KD_WORKERS 为较小值
KD_WORKERS = int(os.environ.get("MYTHIC_KD_WORKERS", "-1"))


@dataclass
class Mesh:
    level: int
    xyz: np.ndarray          # (N, 3) 单位球面
    lon: np.ndarray          # 度
    lat: np.ndarray
    simplices: np.ndarray
    indptr: np.ndarray       # 邻接（CSR）
    nbrs: np.ndarray
    edges: np.ndarray        # (E, 2) 无向边，i < j
    degree: np.ndarray       # 每个节点的邻居数（5 或 6）
    area_km2: np.ndarray     # 每个节点的球面 Voronoi 面积
    h_km: float              # 平均边长
    tree: cKDTree

    @property
    def n(self) -> int:
        return self.xyz.shape[0]


def _adjacency(simplices: np.ndarray, n: int):
    e = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    both = np.concatenate([e, e[:, ::-1]])
    both = both[np.lexsort((both[:, 1], both[:, 0]))]
    indptr = np.searchsorted(both[:, 0], np.arange(n + 1))
    return indptr, both[:, 1].copy(), e


def load_mesh(level: int) -> Mesh:
    """L5 约 10,242 点（约 220 km），L6 约 40,962（约 110 km），L7 约 163,842（约 55 km），L8 约 655,362（约 28 km）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"mesh_L{level}.npz"
    if path.exists():
        d = np.load(path)
        xyz, simplices, area = d["xyz"], d["simplices"], d["area_km2"]
    else:
        import stripy

        m = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=level, tree=False)
        xyz = unit(np.column_stack([m.x, m.y, m.z]))
        simplices = np.asarray(m.simplices, np.int64)
        area = SphericalVoronoi(xyz, radius=1.0).calculate_areas() * R_KM**2
        np.savez_compressed(path, xyz=xyz, simplices=simplices, area_km2=area)
    n = xyz.shape[0]
    indptr, nbrs, edges = _adjacency(simplices, n)
    lon, lat = xyz_to_lonlat(xyz)
    h = float(km_from_chord(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)).mean())
    return Mesh(level, xyz, lon, lat, simplices, indptr, nbrs, edges, np.diff(indptr), area, h, cKDTree(xyz))


class Raster:
    """网格节点 -> 经纬度格网（与 mythic.metrics 的 LatLonGrid 同一约定：第 0 行在北，第 0 列从 180°W 起），每个像元取最近节点。"""

    def __init__(self, mesh: Mesh, ny: int = 512):
        from mythic.metrics.grid import LatLonGrid

        self.grid = LatLonGrid.from_shape((ny, 2 * ny))
        _, idx = mesh.tree.query(self.grid.xyz.reshape(-1, 3), workers=KD_WORKERS)
        self.idx = idx.reshape(self.grid.shape)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field)[self.idx]
