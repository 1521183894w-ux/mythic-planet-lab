"""U36-S3 运动学诊断；不接入演化，也不从网格空隙反演物理宽度。

method_class=procedural_constrained：给定几何/速度的运动学恒等式。
单位：长度 km，时间 Myr，角速度 rad/Myr；平面验证不冒充全球求解。
"""
from __future__ import annotations
import numpy as np
from .geometry import R_KM


def boundary_jump(points, normal_a_to_b, omega_a, omega_b):
    """在同一边界位置分解相对速度；法向必须由独立边界几何给定。

    normal_a_to_b 是球面切平面内由 A 指向 B 的单位法向。
    返回正张开速度、沿边界有符号走滑速度（km/Myr）。
    纯走滑仍可有剪切应变，但面积伸展为零。
    """
    x = np.asarray(points, float)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.asarray(normal_a_to_b, float)
    n = n - np.sum(n * x, axis=-1, keepdims=True) * x
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError("边界法向不能垂直于球面切平面")
    n = n / length
    tangent = np.cross(x, n)
    dv = R_KM * np.cross(np.asarray(omega_b) - np.asarray(omega_a), x)
    return np.sum(dv * n, axis=-1), np.sum(dv * tangent, axis=-1)


def affine_band(width0_km, opening_km=0., slip_km=0., rotation_rad=0.):
    """已给定参考宽度的解析有限变形，不替用户或力学模型选择宽度。

    x 为跨带方向，y 为沿带方向。F = Q [[1+opening/W0,0],[slip/W0,1]]。
    J = Wt/W0，体积守恒的整柱厚度 Ht = H0/J；不是断层基底位移。
    """
    if not np.isfinite([width0_km, opening_km, slip_km, rotation_rad]).all():
        raise ValueError("输入必须为有限值")
    if width0_km <= 0 or width0_km + opening_km <= 0:
        raise ValueError("参考宽度和最终宽度必须为正")
    c, s = np.cos(rotation_rad), np.sin(rotation_rad)
    q = np.array([[c, -s], [s, c]])
    f = q @ np.array([[1. + opening_km / width0_km, 0.],
                      [slip_km / width0_km, 1.]])
    return dict(F=f, J=float(np.linalg.det(f)), width0_km=float(width0_km),
                width_final_km=float(width0_km + opening_km))


def integrate_band(width0_km, thickness0_km, opening_rate_km_myr, duration_myr, dt_myr):
    """面积因子按当前宽度逐步相乘，避免把每次开口都除以初始宽度。"""
    if dt_myr <= 0 or duration_myr < 0 or thickness0_km <= 0:
        raise ValueError("时间步、时长或厚度无效")
    width, thickness, elapsed = float(width0_km), float(thickness0_km), 0.
    affine_band(width)
    while elapsed < duration_myr - 1e-12:
        step = min(dt_myr, duration_myr - elapsed)
        opening = opening_rate_km_myr * step
        beta = affine_band(width, opening)["J"]
        width += opening
        thickness /= beta
        elapsed += step
    return dict(width_final_km=width, thickness_final_km=thickness,
                area_stretch=width / width0_km,
                volume_per_length_km2=width * thickness)
