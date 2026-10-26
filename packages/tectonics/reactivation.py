"""板内挤压复活（1.2d U33）：大陆碰撞的应力传到远离碰撞带的内陆，沿老缝合带、老裂谷这些薄弱带挤压缩短、地壳增厚
（天山、阿尔泰、蒙古高原式：印度—欧亚碰撞约 50 Ma 开始，天山约 25–20 Ma 起复活，离碰撞前缘 1500–2000 km；
天山现今吸收印度—欧亚汇聚的约一半、约 20 mm/yr，Abdrakhmatov 等 1996；凭已有知识，原文待核对）。

做法：范围由设计给出（终图山脊线两侧或高原轮廓，现今坐标、锚在所在板块上，随板块走——与大火成岩省、陆内裂谷同一约定），
起止时刻与总增厚比由设计给出；期间每 Myr 按纯剪切增厚：T ← T · exp(rate · w · dt)，rate = ln(1 + strain) / 持续时间，
w 是范围权重（0–1）。剥蚀照常（packages/tectonics/crust.py 的 erosion_law），所以最后的高度是增厚与剥蚀相抵的结果。
method_class：procedural_constrained（位置、时间、幅度由设计给出，量级按地球）。
"""

from __future__ import annotations

import numpy as np

from .crust import CONT, ORO_REACT


def apply_shortening(eng, w: np.ndarray, rate_per_myr: float, dt: float, plate: int | None = None, origin: int | None = None) -> dict:
    """存活陆壳单元按权重 w（0–1，长度 = 单元数）纯剪切增厚 exp(rate·w·dt) 倍；plate 不为 None 时只增厚这块板块；origin 可进一步选择宿主的来源域（不是混合岩石成分）。"""
    sel = eng.alive & (eng.c.ctype == CONT)
    if plate is not None:
        sel &= eng.c.plate == plate
    if origin is not None:
        sel &= eng.c.origin == origin
    idx = np.flatnonzero(sel)
    wa = np.asarray(w, float)[idx]
    hit = idx[wa > 1e-3]
    f = np.exp(rate_per_myr * wa[wa > 1e-3] * dt)
    before = eng.c.T[hit].copy()
    eng.c.T[hit] = before * f
    strong = hit[wa[wa > 1e-3] > 0.3]
    eng.c.oro[strong] = ORO_REACT          # 不记成碰撞：那里没有板块边界（评审 stage1_2026-10-25_3 第 2 条）
    eng.c.oro_t[strong] = eng.t
    vol = float(((eng.c.T[hit] - before) * eng.area[hit]).sum())
    eng.budget["reactivation_added"] = eng.budget.get("reactivation_added", 0.0) + vol
    return {"n": int(hit.size), "volume_km3": vol}


def line_weight(d_km: np.ndarray, half_width_km: float) -> np.ndarray:
    """山脊线两侧的权重：半宽内为 1，到两倍半宽升余弦减到 0。"""
    d = np.asarray(d_km, float)
    out = np.where(d <= half_width_km, 1.0, 0.0)
    mid = (d > half_width_km) & (d < 2.0 * half_width_km)
    out = np.where(mid, 0.5 * (1.0 + np.cos(np.pi * (d - half_width_km) / half_width_km)), out)
    return out
