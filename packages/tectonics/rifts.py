"""陆内裂谷（1.2d U30）：不把板块裂开的大陆裂谷（东非式）。

- **地堑**：裂谷开始时，轴线 half_km 内的陆壳按拉伸系数 β 减薄（T / β，向外 half_km 内渐变到不减薄），局部 Airy 均衡下谷底下陷。
- **裂谷肩**：拉张卸载后下盘挠曲回弹抬起两肩（Weissel & Karner 1989）。引擎只有局部 Airy 均衡，挠曲回弹用只改高程的叠加：
  地堑外侧 flank_amp_m × exp(−(d − half_km) / flank_decay_km)，地堑里为 0（边界断层处成陡崖）。
  量级：阿法尔与主埃塞俄比亚裂谷两肩的挠曲抬升约 500–1200 m（由北往南增大）。
- **轴部隆起**：裂谷下方的地幔上涌（东非高原式），只改高程的升余弦，中心 swell_amp_m、离轴 swell_radius_km 处为 0。
裂谷线用现今坐标、锚在所在板块上，随板块走（与大火成岩省同一约定）。method_class：procedural_constrained。
"""

from __future__ import annotations

import numpy as np

from .crust import CONT, ORO_RIFT
from .plumes import distance_km, taper


def rift_relief_m(x: np.ndarray, line_now: np.ndarray, half_km: float, flank_amp_m: float, flank_decay_km: float,
                  swell_amp_m: float = 0.0, swell_radius_km: float = 1.0) -> np.ndarray:
    """固定网格 x（n×3）上裂谷两肩的挠曲抬升与轴部隆起（m），line_now 是此刻的裂谷轴线。"""
    d = distance_km(x, line_now)
    flank = np.where(d > half_km, flank_amp_m * np.exp(-(d - half_km) / max(flank_decay_km, 1e-9)), 0.0)
    return flank + swell_amp_m * taper(d, swell_radius_km)


def progressive_beta(beta: float, duration_myr: float, t_start: float, t_now: float, dt: float = 1.) -> float:
    """持续伸展在当前时间步内的倍率；只积分与活动区间相交的时间。

    beta是规定的总伸展倍率，不由目标高程反算。需与apply_graben的对数权重
    配合，才能在固定宿主/权重下保持时间步细分不改变总减薄。
    """
    if not np.isfinite([beta, duration_myr, t_start, t_now, dt]).all() or beta < 1 or duration_myr <= 0 or dt <= 0:
        raise ValueError("持续伸展要求beta>=1，正持续时间和正时间步")
    active = max(0., min(t_now + dt, t_start) - max(t_now, t_start - duration_myr))
    return float(np.exp(np.log(beta) * active / duration_myr))


def apply_graben(eng, line_now: np.ndarray, half_km: float, beta: float, *, logarithmic: bool = False) -> dict:
    """地堑：存活陆壳单元按离轴距离减薄（轴线 half_km 内 T / β，到 2 × half_km 渐变到不减薄）。返回减掉的体积与单元数。"""
    if not np.isfinite(beta) or beta < 1:
        raise ValueError("地堑伸展倍率beta须为有限数且>=1")
    alive = np.flatnonzero(eng.alive & (eng.c.ctype == CONT))
    w = taper(distance_km(eng.positions()[alive], line_now), 2.0 * half_km, half_km)
    hit = alive[w > 1e-3]
    f = np.exp(np.log(beta) * w[w > 1e-3]) if logarithmic else 1.0 + (beta - 1.0) * w[w > 1e-3]
    before = eng.c.T[hit].copy()
    eng.c.T[hit] = before / f
    eng.c.oro[hit] = ORO_RIFT
    eng.c.oro_t[hit] = eng.t
    vol = float(((before - eng.c.T[hit]) * eng.area[hit]).sum())
    eng.budget["rift_thinning"] = eng.budget.get("rift_thinning", 0.0) + vol
    return {"n": int(hit.size), "volume_km3": vol}
