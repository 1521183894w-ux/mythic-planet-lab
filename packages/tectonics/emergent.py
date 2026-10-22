"""涌现型运动学（1.1 路线 A，Cortial 2019 式）。只用于 300 Ma 之前的"前史"：前史不需要命中特定目标，
要的是多样、合理的构造痕迹（缝合带、增生地体、裂谷陆缘）。300 Ma -> 0 的正式演化用剧本（kinematics.py）。

规则：
- 板片拉力：正在俯冲的板块，角速度向"把板块质心拉向俯冲前缘"的方向松弛（时间尺度 tau），
  目标速度 v = v_max · L / (L + L0)，L 为海沟长度；另保留初始运动作为背景（地幔拖曳、洋脊推力的代理）。
- 碰撞：一对板块累计被拼贴的陆壳面积达到 a_trigger 后，下行板块上与上盘相接的整块大陆改属上盘板块（地体转移），
  之后随上盘运动——大陆由此焊合。
- 裂解：陆壳面积 ≥ rift_min_cont 的板块按泊松过程裂开（平均等待 rift_wait），由 rift_fn 决定裂谷位置。

method_class：procedural_constrained。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crust import CONT
from .engine import Engine
from .geometry import DEG, R_KM, omega_rotmat, unit


@dataclass
class EmergentParams:
    v_max_cm_yr: float = 8.0
    tau_myr: float = 10.0
    L0_km: float = 3000.0
    a_trigger_km2: float = 3.0e5
    rift_wait_myr: float = 150.0
    rift_min_cont_km2: float = 1.5e7
    rift_rate_deg_myr: tuple = (0.2, 0.35)


@dataclass
class EmergentDriver:
    eng: Engine
    omega: dict
    rng: np.random.Generator
    params: EmergentParams = field(default_factory=EmergentParams)
    rift_fn: object = None          # rift_fn(eng, plate, rng) -> (s1, s2, axis) 或 None；默认随机
    next_id: int = 0

    def __post_init__(self):
        self.bg = {q: np.asarray(w, float).copy() for q, w in self.omega.items()}
        self.omega = {q: np.asarray(w, float).copy() for q, w in self.omega.items()}
        self.transferred: set = set()
        self.next_id = max(self.next_id, max(self.eng.plates) + 1)
        self._centroids()

    def _centroids(self):
        self.cen = self.eng.plate_centroids()
        self.R_at = {q: self.eng.R[q].copy() for q in self.eng.plates}

    def step(self) -> bool:
        eng, p = self.eng, self.params
        dR = {q: omega_rotmat(self.omega[q], eng.p.dt) for q in eng.plates}
        committed = eng.step(dR, {q: self.omega[q] for q in eng.plates})
        fr = eng.recent_fronts()
        for q in eng.plates:
            target = self.bg[q].copy()
            if fr is not None and q in self.cen:
                X, S, _, V = fr
                sel = S == q
                if sel.any():
                    cq = eng.R[q] @ self.R_at[q].T @ self.cen[q]
                    n = unit(unit(np.cross(cq, X[sel])).sum(0))
                    vbar = max(float(V[sel].mean()) * 10.0, 5.0)                     # km/Myr
                    L = sel.sum() * eng.m.h_km**2 / (vbar * eng.p.front_memory_myr)  # 海沟长度估计
                    target = target + n * (p.v_max_cm_yr * 10.0 * L / (L + p.L0_km)) / R_KM
            self.omega[q] = self.omega[q] + (target - self.omega[q]) * (eng.p.dt / p.tau_myr)
        if committed:
            self._on_commit()
        return committed

    def _on_commit(self):
        eng, p = self.eng, self.params
        for (down, over), a in list(eng.coll_area.items()):
            if a >= p.a_trigger_km2 and (down, over) not in self.transferred and down in eng.plates and over in eng.plates:
                eng.transfer_terrane(down, over)
                self.transferred.add((down, over))
        for q in list(eng.plates):
            if eng.cont_area.get(q, 0.0) < p.rift_min_cont_km2:
                continue
            if self.rng.random() >= 1.0 - np.exp(-eng.p.commit_every * eng.p.dt / p.rift_wait_myr):
                continue
            spec = (self.rift_fn or random_rift)(eng, q, self.rng)
            if spec is None:
                continue
            s1, s2, axis = spec
            rate = float(self.rng.uniform(*p.rift_rate_deg_myr))
            w_new = self.omega[q] + unit(axis) * rate * DEG
            if eng.split_plate(q, s1, s2, axis, self.next_id, self.rng, omega_new=w_new) is not None:
                self.omega[self.next_id] = w_new
                self.bg[self.next_id] = self.bg[q] + unit(axis) * rate * DEG
                self.next_id += 1
        for q in list(self.omega):
            if q not in eng.R:
                self.omega.pop(q)
                self.bg.pop(q, None)
        self._centroids()


def random_rift(eng: Engine, q: int, rng: np.random.Generator):
    """随机裂谷：板块陆壳上随机两点，裂谷为其垂直平分线（再经阶梯化），相对运动轴斜交 20–50°。"""
    pos = eng.positions()
    cont = np.flatnonzero(eng.alive & (eng.c.plate == q) & (eng.c.ctype == CONT))
    if cont.size < 10:
        return None
    s1 = pos[rng.choice(cont)]
    far = cont[np.linalg.norm(pos[cont] - s1, axis=1) > 2 * np.sin(1500.0 / (2 * R_KM))]
    if far.size == 0:
        return None
    s2 = pos[rng.choice(far)]
    axis0 = unit(np.cross(s1, s2))
    tilt = unit(np.cross(axis0, unit(s1 + s2)))
    ob = np.radians(rng.uniform(20, 50)) * rng.choice([-1, 1])
    return s1, s2, unit(np.cos(ob) * axis0 + np.sin(ob) * tilt)
