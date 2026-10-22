"""地壳状态（地块单元表）、参数与构造尺度高程。

method_class：洋壳水深 GDH1、陆壳 Airy 均衡 = parameterized_physics；其余经验形状函数 = procedural_constrained。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import smoothstep

OCEAN, CONT = 0, 1
ORO_NONE, ORO_ARC, ORO_COLL, ORO_RIFT = 0, 1, 2, 3
ORIGIN_OCEAN, ORIGIN_ARC = 0, 90


@dataclass
class Params:
    dt: float = 1.0                    # Myr
    commit_every: int = 5              # 每 5 步提交一次（删除消耗单元、填空隙）
    max_sweeps: int = 6                # 每步相互作用检测的最多轮数（快速汇聚时一步可能推进几行单元）
    T_ocean: float = 7.0               # 新洋壳厚度 km
    T_cont: float = 35.0               # 正常陆壳厚度 km
    e_cont_ref_m: float = 400.0        # 35 km 陆壳的高程（大陆"干舷"）
    iso_m_per_km: float = 151.5        # Airy：每增厚 1 km 抬升 (1 - ρc/ρm) km，ρc 2800、ρm 3300
    blend_T0: float = 7.0              # 高程在洋壳与陆壳公式之间按厚度过渡（7–28 km）
    blend_T1: float = 28.0
    cont_lith_age: float = 100.0       # 减薄陆缘下方岩石圈按 100 Myr 洋壳的热沉降处理
    arc_rate_km_myr: float = 0.8       # 俯冲上盘最大增厚速率（相对速度 10 cm/yr 时）
    v0_cm_yr: float = 10.0
    arc_peak_km: float = 150.0         # 增厚在距俯冲前缘 150 km 处最大（火山弧位置）
    arc_max_km: float = 600.0          # 600 km 外为 0
    arc_to_cont_T_km: float = 18.0     # 洋内弧增厚到 18 km 后视为浮力地壳，不再俯冲
    front_memory_myr: float = 3.0      # 俯冲前缘取最近 3 Myr 内被消耗的单元位置
    collision_radius_km: float = 400.0 # 碰撞时被拼贴地壳的体积分配到上盘 400 km 半径内
    tau_oro_myr: float = 120.0         # 超出 35 km 的地壳厚度按 120 Myr 时间尺度衰减（剥蚀 + 垮塌）
    kappa_km2_myr: float = 300.0       # 增厚地壳的侧向流动（厚度扩散系数）
    flow_T_km: float = 40.0            # 只在相邻单元中至少一个厚于 40 km 时流动
    T_max_km: float = 65.0             # 地壳厚度上限：超出部分摊到同板块外围陆壳上（重力垮塌 / 下地壳流动的代理）。
                                       # 地球上最厚的陆壳约 70 km（青藏高原），高原面约 5 km；65 km 对应高程约 4.9 km
    trench_depth_m: float = 2500.0     # 俯冲板块在前缘处的附加下沉（显示用）
    trench_width_km: float = 70.0
    near_continent_km: float = 400.0   # 洋—洋首次接触时，400 km 内有大陆的一方作上盘


def gdh1_depth_m(age_myr):
    """GDH1 板块冷却模型的洋底水深（m，正值向下；Stein & Stein 1992, Nature 359:123）。"""
    a = np.maximum(np.asarray(age_myr, float), 0.0)
    return np.where(a < 20.0, 2600.0 + 365.0 * np.sqrt(a), 5651.0 - 2473.0 * np.exp(-0.0278 * a))


class Crust:
    """地块单元表（每行一个单元）。node 是该单元在所属板块参照系中的网格节点编号；uid 是单元的永久编号。
    Engine.view() 返回的"固定网格视图"也用这个类，此时 node = 固定网格节点编号。"""
    FIELDS = ("uid", "plate", "node", "ctype", "T", "age", "origin", "oro", "oro_t")

    def __init__(self, n: int):
        self.uid = np.arange(n, dtype=np.int64)
        self.plate = np.zeros(n, np.int32)
        self.node = np.arange(n, dtype=np.int64)
        self.ctype = np.zeros(n, np.int8)
        self.T = np.full(n, 7.0)
        self.age = np.zeros(n)
        self.origin = np.zeros(n, np.int16)   # 地体来源：初始陆块编号；90 = 洋内弧增生；0 = 洋壳
        self.oro = np.zeros(n, np.int8)       # 最近一次构造改造的类型
        self.oro_t = np.full(n, np.nan)       # 最近一次构造改造的时间（Ma）

    def __len__(self):
        return self.plate.size

    def gather(self, idx: np.ndarray) -> "Crust":
        out = Crust(0)
        for f in self.FIELDS:
            setattr(out, f, getattr(self, f)[idx].copy())
        return out

    def concat(self, other: "Crust") -> "Crust":
        out = Crust(0)
        for f in self.FIELDS:
            setattr(out, f, np.concatenate([getattr(self, f), getattr(other, f)]))
        return out


def elevation(c: Crust, p: Params, trench: np.ndarray | None = None) -> np.ndarray:
    """构造尺度高程（m）：洋壳按 GDH1，陆壳按 Airy 均衡，两者之间按地壳厚度平滑过渡。"""
    age = np.where(c.ctype == OCEAN, c.age, p.cont_lith_age)
    e_oc = -gdh1_depth_m(age)
    w = smoothstep((c.T - p.blend_T0) / (p.blend_T1 - p.blend_T0))
    e_c = p.e_cont_ref_m + (c.T - p.T_cont) * p.iso_m_per_km
    e = (1.0 - w) * e_oc + w * e_c
    if trench is not None:
        e = e - trench
    return e
