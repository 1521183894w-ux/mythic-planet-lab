"""只读球面剖面工具；密集观察不会增加地壳模型的物理分辨率。"""
from __future__ import annotations

import numpy as np

from .geometry import R_KM, lonlat_to_xyz, unit


def spherical_sections(line_lonlat, along_km, cross_km):
    """沿折线的大圆分段定位，再沿正交大圆取横剖面。

    along 可延长到首尾之外；cross 正值指向行进方向左侧。折点使用后一个
    分段的切线，不对弯角平滑，避免引入未定义的几何形状。返回 (s,n,3)、线长。
    """
    line = np.asarray(line_lonlat, float)
    along, cross = np.asarray(along_km, float), np.asarray(cross_km, float)
    if (line.ndim != 2 or line.shape[1] != 2 or len(line) < 2
            or not np.isfinite(line).all() or (abs(line[:, 1]) > 90).any()):
        raise ValueError("折线须为至少两个有限经纬度点")
    if along.ndim != 1 or cross.ndim != 1 or not np.isfinite(along).all() or not np.isfinite(cross).all():
        raise ValueError("剖面距离须为有限的一维数组")
    p = lonlat_to_xyz(line[:, 0], line[:, 1])
    angle = np.arctan2(np.linalg.norm(np.cross(p[:-1], p[1:]), axis=1), np.sum(p[:-1] * p[1:], axis=1))
    if (angle < 1e-10).any() or (angle > np.pi - 1e-10).any():
        raise ValueError("相邻点重合或互为对跖点，无法定义分段切线")
    s = np.r_[0., np.cumsum(angle * R_KM)]
    segment = np.clip(np.searchsorted(s, along, side="right") - 1, 0, len(angle) - 1)
    a, b = p[segment], p[segment + 1]
    normal = unit(np.cross(a, b))
    tangent = np.cross(normal, a)
    theta = (along - s[segment]) / R_KM
    center = np.cos(theta)[:, None] * a + np.sin(theta)[:, None] * tangent
    phi = cross / R_KM
    points = np.cos(phi)[None, :, None] * center[:, None, :] + np.sin(phi)[None, :, None] * normal[:, None, :]
    return points, float(s[-1])


def axis_connected_span(offset_km, height_m, threshold_m, valid=None):
    """仅量穿过轴线的连续超阈带；空隙/低地会截断，远处另一山体不并入。

    给出采样内界与外侧括号。触及窗口边缘时不给有限宽度上界；它不是物理
    山脉宽度的验收定义，也不把查询步距当成模型精度。
    """
    x, h = np.asarray(offset_km, float), np.asarray(height_m, float)
    if x.ndim != 1 or h.shape != x.shape or len(x) < 3 or not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        raise ValueError("横剖面坐标须严格递增，并与高程一一对应")
    zero = np.flatnonzero(np.isclose(x, 0., rtol=0, atol=1e-8))
    if len(zero) != 1:
        raise ValueError("横剖面须恰好含一个轴线零点")
    ok = np.isfinite(h)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    above = ok & (h >= threshold_m)
    c = int(zero[0])
    if not above[c]:
        return {"axis_above": False, "axis_covered": bool(ok[c]), "threshold_m": float(threshold_m)}
    lo = hi = c
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < len(x) and above[hi + 1]:
        hi += 1
    return {"axis_above": True, "axis_covered": True, "threshold_m": float(threshold_m),
            "inside_km": [float(x[lo]), float(x[hi])], "width_inside_km": float(x[hi] - x[lo]),
            "left_bracket_km": None if lo == 0 else [float(x[lo - 1]), float(x[lo])],
            "right_bracket_km": None if hi == len(x) - 1 else [float(x[hi]), float(x[hi + 1])],
            "left_stop": "window" if lo == 0 else "low" if ok[lo - 1] else "uncovered",
            "right_stop": "window" if hi == len(x) - 1 else "low" if ok[hi + 1] else "uncovered"}
