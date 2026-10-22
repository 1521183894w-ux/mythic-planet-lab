"""阶段 1.1 构造方案原型（spike）的公共部分：路径、球面几何、旋转、网格缓存、栅格化。

运行方式见本目录 README.md。图写到 figures/stage1/methods/，数值结果写到 figures/stage1/methods/results/，
中间数据（网格缓存、快照）写到 .cache/spikes/stage1/。这些都不进 Git，可随时重跑生成。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

ROOT = Path(os.environ.get("MYTHIC_ROOT", Path(__file__).resolve().parents[2])).resolve()
FIG_DIR = ROOT / "figures" / "stage1" / "methods"
RES_DIR = FIG_DIR / "results"
ROT_DIR = FIG_DIR / "rotation_files"
CACHE = ROOT / ".cache" / "spikes" / "stage1"
for _d in (FIG_DIR, RES_DIR, ROT_DIR, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# 让 spike 脚本能 import 本目录的模块与项目包（项目包已以可编辑模式装入环境，这里只补本目录）
sys.path.insert(0, str(Path(__file__).resolve().parent))

R_KM = 6371.0
DEG = np.pi / 180.0
# KD 树查询的线程数。并行跑多个种子时由 run_all.py 设为较小值，避免线程过度争用
KD_WORKERS = int(os.environ.get("MYTHIC_KD_WORKERS", "-1"))


def code_version() -> str:
    """Git 短提交号；项目代码或本目录有未提交改动时加 -dirty（项目的 code_version 不看 spikes/，这里补上）。"""
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        paths = ["packages", "scripts", "config", "spikes/stage1_methods", "environment.yml", "pyproject.toml"]
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal", "--", *paths], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# ---- 球面几何 -------------------------------------------------------------------------
def lonlat_to_xyz(lon_deg, lat_deg) -> np.ndarray:
    lon = np.asarray(lon_deg, float) * DEG
    lat = np.asarray(lat_deg, float) * DEG
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, float)
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    return lon, lat


def chord_from_km(km):
    return 2.0 * np.sin(np.asarray(km, float) / (2.0 * R_KM))


def km_from_chord(c):
    return 2.0 * R_KM * np.arcsin(np.clip(np.asarray(c, float) / 2.0, 0.0, 1.0))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---- 旋转 --------------------------------------------------------------------------------
def rotmat(axis, angle_rad: float) -> np.ndarray:
    """绕单位轴 axis 转 angle_rad（右手），Rodrigues 公式。作用方式：x_new = R @ x。"""
    a = unit(np.asarray(axis, float))
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def axis_angle(rm: np.ndarray) -> tuple[np.ndarray, float]:
    """旋转矩阵 -> (单位轴, 角度 rad)，角度在 [0, π]。单位矩阵返回 (北极, 0)。"""
    c = np.clip((np.trace(rm) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    v = np.array([rm[2, 1] - rm[1, 2], rm[0, 2] - rm[2, 0], rm[1, 0] - rm[0, 1]])
    s = np.linalg.norm(v)
    if ang < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if s < 1e-9:  # 接近 180°：取 (R + I)/2 的最大列
        b = (rm + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(b)))
        return unit(b[:, k]), ang
    return v / s, ang


def omega_rotmat(omega: np.ndarray, dt: float) -> np.ndarray:
    """角速度矢量 omega（rad/Myr，方向 = 欧拉极）在 dt Myr 内的有限旋转。"""
    w = np.linalg.norm(omega)
    if w < 1e-15:
        return np.eye(3)
    return rotmat(omega / w, w * dt)


def omega_from_rotmat(rm: np.ndarray, dt: float) -> np.ndarray:
    a, ang = axis_angle(rm)
    return a * ang / dt


def pole_vector(lat_deg: float, lon_deg: float, rate_deg_per_myr: float) -> np.ndarray:
    """欧拉极（纬度、经度）与角速度（°/Myr，正值 = 绕该极右手旋转） -> 角速度矢量（rad/Myr）。"""
    return lonlat_to_xyz(lon_deg, lat_deg) * rate_deg_per_myr * DEG


def surface_speed_cm_yr(omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """角速度矢量 omega（rad/Myr）在单位球面点 x 处的线速度大小（cm/yr；1 cm/yr = 10 km/Myr）。"""
    return np.linalg.norm(np.cross(omega, x), axis=-1) * R_KM / 10.0


# ---- 网格 --------------------------------------------------------------------------------
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
    """stripy 二十面体网格（L6 约 40,962 点、平均间距约 110 km；L7 约 163,842 点、约 55 km），带缓存。"""
    path = CACHE / f"mesh_L{level}.npz"
    if path.exists():
        d = np.load(path)
        xyz, simplices, area = d["xyz"], d["simplices"], d["area_km2"]
    else:
        import stripy

        m = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=level, tree=False)
        xyz = unit(np.column_stack([m.x, m.y, m.z]))
        simplices = np.asarray(m.simplices, np.int64)
        sv = SphericalVoronoi(xyz, radius=1.0)
        area = sv.calculate_areas() * R_KM**2
        np.savez_compressed(path, xyz=xyz, simplices=simplices, area_km2=area)
    indptr, nbrs, edges = _adjacency(simplices, xyz.shape[0])
    lon, lat = xyz_to_lonlat(xyz)
    h = float(km_from_chord(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)).mean())
    return Mesh(level, xyz, lon, lat, simplices, indptr, nbrs, edges, area, h, cKDTree(xyz))


def neighbors_any(mesh: Mesh, mask: np.ndarray) -> np.ndarray:
    """与 mask 中任一节点相邻的节点（不含 mask 本身）。"""
    e = mesh.edges
    out = np.zeros(mesh.n, bool)
    out[e[mask[e[:, 0]], 1]] = True
    out[e[mask[e[:, 1]], 0]] = True
    return out & ~mask


# ---- 栅格化（每个像元取最近的网格节点）--------------------------------------------------
class Raster:
    """网格节点 -> 经纬度格网（与 mythic.metrics 的 LatLonGrid 同一约定：第 0 行在北，第 0 列从 180°W 起）。"""

    def __init__(self, mesh: Mesh, ny: int = 512):
        from mythic.metrics.grid import LatLonGrid

        self.grid = LatLonGrid.from_shape((ny, 2 * ny))
        _, idx = mesh.tree.query(self.grid.xyz.reshape(-1, 3), workers=-1)
        self.idx = idx.reshape(self.grid.shape)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        return np.asarray(field)[self.idx]
