"""球面几何与旋转。约定：单位球面上的点用三维单位矢量表示；旋转矩阵作用方式 x_new = R @ x（行矢量写作 X @ R.T）。"""

from __future__ import annotations

import numpy as np

R_KM = 6371.0
DEG = np.pi / 180.0


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


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotmat(axis, angle_rad: float) -> np.ndarray:
    """绕单位轴 axis 转 angle_rad（右手），Rodrigues 公式。"""
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
    if s < 1e-9:  # 接近 180°
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
    """角速度矢量（rad/Myr）在单位球面点 x 处的线速度（cm/yr；1 cm/yr = 10 km/Myr）。"""
    return np.linalg.norm(np.cross(omega, x), axis=-1) * R_KM / 10.0


def orthonormalize(rm: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rm)
    return u @ vt
