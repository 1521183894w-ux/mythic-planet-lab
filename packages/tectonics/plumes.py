"""地幔柱（1.2d U29）：动力地形穹隆与大火成岩省。

- **动力地形穹隆**（swell）：地幔柱头部的热浮力把上方岩石圈整体顶起，形成宽 1000–2000 km、高约 1 km 的宽缓穹隆
  （夏威夷海隆宽约 1200 km、高约 1 km；南部非洲高原比一般克拉通的 400–500 m 高出 1 km 以上，Lithgow-Bertelloni & Silver 1998）。
  只改高程、不改地壳厚度；位置固定在地幔（热点）参照系里，活动期间叠加在上方的任何板块上，地幔柱停了就消失。
  剖面取升余弦：中心为 amp_m，半径 radius_km 处为 0（半高宽 = radius_km）。
- **大火成岩省**（lip）：短时间（1 Myr 量级）的大量岩浆加到地壳上——溢流玄武岩 1–2 km 加下地壳底侵 8–12 km
  （埃塞俄比亚高原，约 30 Ma 在 1 Myr 内喷出、面积约 60 万 km²，高原下地壳约 40 km）；洋底高原 = 洋壳加厚到 20–25 km
  （凯尔盖朗高原 21–25 km）。陆壳与洋壳都按剖面加厚 add_km；之后随板块漂移、按引擎的规则剥蚀与流动。
  范围是一个点或一条折线外 radius_km 内（flat_km 内满额、之外升余弦减到 0）。

method_class：procedural_constrained（地幔柱的位置、时间与规模由设计给出，量级按上面的地球实例）。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import R_KM, km_from_chord, unit


def taper(d_km, radius_km: float, flat_km: float = 0.0) -> np.ndarray:
    """flat_km 内为 1，之外升余弦减到 radius_km 处为 0。"""
    d = np.asarray(d_km, float)
    if radius_km <= flat_km:
        return (d <= flat_km).astype(float)
    s = np.clip((d - flat_km) / (radius_km - flat_km), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * s))


def densify_line(pts: np.ndarray, step_km: float = 15.0) -> np.ndarray:
    """球面折线（n×3 单位矢量）按大圆弧加密到间距不超过 step_km。"""
    pts = unit(np.atleast_2d(np.asarray(pts, float)))
    if len(pts) == 1:
        return pts
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        ang = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        n = max(1, int(np.ceil(ang * R_KM / step_km)))
        f = np.linspace(0.0, 1.0, n + 1)[1:]
        if ang < 1e-12:
            seg = np.repeat(a[None], f.size, 0)
        else:   # 球面线性插值
            seg = (np.sin((1 - f) * ang)[:, None] * a + np.sin(f * ang)[:, None] * b) / np.sin(ang)
        out.append(seg)
    return unit(np.concatenate(out))


def distance_km(x: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """x（n×3）到一个点或一条折线（m×3）的大圆距离（km）。"""
    line = densify_line(footprint)
    d, _ = cKDTree(line).query(np.atleast_2d(x))
    return km_from_chord(d)


def swell_m(x: np.ndarray, t: float, swells: list) -> np.ndarray:
    """t 时刻固定网格（地幔参照系）上的动力地形（m）。swells：[{center: 单位矢量, amp_m, radius_km, active: (开始, 停止)}]。"""
    out = np.zeros(len(x))
    for s in swells:
        start, stop = s["active"]
        if not (stop - 1e-9 <= t <= start + 1e-9):
            continue
        d = km_from_chord(np.linalg.norm(x - s["center"], axis=1))
        out += float(s["amp_m"]) * taper(d, float(s["radius_km"]))
    return out


def apply_lip(eng, footprint: np.ndarray, radius_km: float, add_km: float, flat_km: float = 0.0) -> dict:
    """大火成岩省：当前位置落在范围里的存活地块单元（陆壳与洋壳）按剖面加厚 add_km。返回加上的体积与单元数。"""
    alive = np.flatnonzero(eng.alive)
    w = np.zeros(len(eng.c))
    w[alive] = taper(distance_km(eng.positions()[alive], footprint), radius_km, flat_km)
    return apply_lip_weights(eng, w, add_km)


def apply_lip_weights(eng, w: np.ndarray, add_km: float) -> dict:
    """大火成岩省（范围已算好）：每个存活单元按权重 w（0–1，长度 = 单元数）加厚 add_km × w。"""
    alive = np.flatnonzero(eng.alive)
    wa = np.asarray(w, float)[alive]
    hit = alive[wa > 1e-3]
    dT = add_km * wa[wa > 1e-3]
    eng.c.T[hit] += dT
    vol = float((dT * eng.area[hit]).sum())
    eng.budget["lip_added"] = eng.budget.get("lip_added", 0.0) + vol
    return {"n": int(hit.size), "volume_km3": vol}
