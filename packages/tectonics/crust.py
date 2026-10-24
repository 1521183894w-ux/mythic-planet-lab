"""地壳状态（地块单元表）、参数与构造尺度高程。

method_class：洋壳水深 GDH1、陆壳 Airy 均衡 = parameterized_physics；其余经验形状函数 = procedural_constrained。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import smoothstep

OCEAN, CONT = 0, 1
ORO_NONE, ORO_ARC, ORO_COLL, ORO_RIFT = 0, 1, 2, 3
ORIGIN_OCEAN, ORIGIN_ARC, ORIGIN_RIFT = 0, 90, 91   # 91 = 两块陆块之间张开时填的拉伸减薄陆壳（U30 拉分盆地与裂谷陆缘）


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
    arc_shortening_km_myr: float | None = None  # U28：上盘朝海沟前进造成的构造缩短增厚（上盘绝对速度在指向海沟方向上
                                       # 10 cm/yr 时的最大增厚速率）；None = 不分（1.2c 及以前：全部按相对速度）。
                                       # 设了它时 arc_rate_km_myr 只代表岩浆增生。依据：安第斯的缩短与南美西漂同步
                                       # （Oncken 等 2006），上盘不动或后撤的岛弧（日本、马里亚纳）只有岩浆增生、山低
    v0_cm_yr: float = 10.0
    arc_peak_km: float = 150.0         # 陆缘弧（安第斯型）：增厚在距俯冲前缘 150 km 处最大（火山弧位置）
    arc_max_km: float = 600.0          # 600 km 外为 0（含弧后褶皱冲断带）
    arc_peak_ocean_km: float = 70.0    # 洋内弧：上盘是洋壳时增厚带窄得多（伊豆—小笠原、阿留申弧宽 100–200 km）
    arc_max_ocean_km: float = 200.0    # （1.1 原型中洋内弧也用 600 km，增生的陆壳偏多）
    arc_ocean_flux_factor: float | None = None  # U28 第二轮：洋壳单元上岩浆增生速率的倍数。同样的岩浆通量（每公里弧长每 Myr
                                       # 的体积）摊在 200 km 宽的洋内弧带上，比摊在 600 km 宽的陆缘弧带上增厚快 3 倍（两条形状函数的
                                       # 积分之比 arc_max_km / arc_max_ocean_km）；None = 1（同一峰值速率，洋内弧的通量只有陆缘弧的 1/3）。
                                       # 取 3 时伊豆—小笠原式（汇聚约 5 cm/yr、50 Myr）弧轴地壳约 22 km，与实测约 20 km 相当
    arc_to_cont_T_km: float = 18.0     # 洋内弧增厚到 18 km 后视为浮力地壳，不再俯冲
    arc_accrete_T_km: float | None = None  # U28 弧—陆碰撞：洋壳类型的弧地壳厚到这么多、被大陆撞上时转成陆壳拼贴，不被俯冲
                                       # （台湾、新几内亚式）；None = 不拼贴（被俯冲掉）
    arc_accrete_far_km: float = 300.0  # 只拼贴洋内弧：离所在板块自己的陆壳这么远以外（陆缘外的弧前地壳仍被俯冲掉，
                                       # 否则设计中的下行陆块会先撞上它、被多消耗一截，见 U26）
    arc_ocean_T_max_km: float | None = None  # 洋内弧（洋壳与岛弧增生地壳）的俯冲增厚上限；None = 不限（1.2c 及以前）。
                                       # 成熟洋内弧地壳 20–35 km（伊豆—小笠原约 20 km，阿留申约 35 km），整体是水下脊、
                                       # 只有火山顶露出海面。不限时洋内弧一直增厚到 65 km、长成条带状平顶"陆地"
                                       # （1.2c 世界 6 的梳状条带、U25 第一次运行的 14 百万 km² 洋内弧陆地）
    arc_compress_T_max_km: float | None = None  # U28 第三轮：挤压型洋内弧的上限。岩浆增生仍封顶于 arc_ocean_T_max_km，
                                       # 上盘朝海沟推进造成的构造缩短（arc_shortening_km_myr 那一项）可以继续增厚到这里。
                                       # 上盘后撤或不动的岛弧（马里亚纳、伊豆—小笠原）只有岩浆增生，弧地壳约 20 km、整体在水下；
                                       # 被两侧挤压的岛弧（吕宋）地壳 20–46 km（Geoscience Letters 2025 的接收函数与噪声自相关），
                                       # 阿留申中段 35–37 km，露出成大岛。None = 与岩浆增生同一上限（U28 第二轮及以前）
    front_memory_myr: float = 3.0      # 俯冲前缘取最近 3 Myr 内被消耗的单元位置
    collision_radius_km: float = 400.0 # 碰撞时被拼贴地壳的体积分配到上盘 400 km 半径内
    tau_oro_myr: float = 120.0         # 超出 35 km 的地壳厚度按 120 Myr 时间尺度衰减（剥蚀 + 垮塌）；erosion_law = "linear" 时用
    erosion_law: str = "linear"        # U31："linear" = 超出 35 km 的厚度按 tau_oro_myr 指数衰减（1.2a–U30，高山与低矮老山一样快）；
                                       # "relief" = 剥蚀速率随起伏非线性增长 E = ero_K·(H_r / 1 km)^ero_n + ero_bg·h_ex（km/Myr 岩石），
                                       # 只剥超出 35 km 的部分。H_r = 本单元高程 − ero_relief_km 半径内（同板块）最低处（不低于海平面），
                                       # h_ex = 高出 35 km 陆壳的高程。低矮的老山剥得很慢，几亿年后仍有约 1 km（古生代造山带的持久性）
    ero_K_km_myr: float = 0.025        # 起伏 1 km 时 25 m/Myr：阿巴拉契亚（大烟山）流域 ¹⁰Be 剥蚀速率 25–30 m/Myr（Matmon 等 2003）
    ero_n: float = 2.0                 # 剥蚀随起伏非线性增长（Montgomery & Brandon 2002：起伏 1 km 以下近线性、更高时急增）；
                                       # n = 2 时起伏 4 km 约 0.4 mm/yr、5 km 约 0.6 mm/yr（造山带流域平均的量级）
    ero_relief_km: float = 165.0       # 起伏按 165 km 半径算（L7 三圈邻居；约为山脊到山麓的距离）
    ero_bg_per_myr: float = 0.005      # 背景剥蚀：高原内部起伏小，但仍以每 km 超高 5 m/Myr 剥蚀（青藏高原内部 10–20 m/Myr 量级）
    kappa_km2_myr: float = 300.0       # 增厚地壳的侧向流动（厚度扩散系数）
    flow_T_km: float = 40.0            # 只在相邻单元中至少一个厚于 40 km 时流动
    T_max_km: float = 65.0             # 地壳厚度上限：超出部分摊到同板块外围陆壳上（重力垮塌 / 下地壳流动的代理）。
                                       # 地球上最厚的陆壳约 70 km（青藏高原），高原面约 5 km；65 km 对应高程约 4.9 km
    trench_depth_m: float = 2500.0     # 俯冲板块在前缘处的附加下沉（显示用）
    trench_width_km: float = 70.0
    near_continent_km: float = 400.0   # 洋—洋首次接触时，400 km 内有大陆的一方作上盘
    gap_prefer_opening_only: bool = False  # U28 第二轮：Engine.gap_prefer 只用于两块真正张开的地方，汇聚边界上网格对不齐
                                       # 留下的小洞归既定的下行板块（见 Engine.commit）；False = 指定板块对之间的所有空隙按指定、
                                       # 其余按最近单元（U26–U28 第一轮）


def gdh1_depth_m(age_myr):
    """GDH1 板块冷却模型的洋底水深（m，正值向下；Stein & Stein 1992, Nature 359:123）。"""
    a = np.maximum(np.asarray(age_myr, float), 0.0)
    return np.where(a < 20.0, 2600.0 + 365.0 * np.sqrt(a), 5651.0 - 2473.0 * np.exp(-0.0278 * a))


class Crust:
    """地块单元表（每行一个单元）。node 是该单元在所属板块参照系中的网格节点编号；uid 是单元的永久编号。
    Engine.view() 返回的"固定网格视图"也用这个类，此时 node = 固定网格节点编号。"""
    FIELDS = ("uid", "plate", "node", "ctype", "T", "age", "origin", "prov", "t_form", "oro", "oro_t")

    def __init__(self, n: int):
        self.uid = np.arange(n, dtype=np.int64)
        self.plate = np.zeros(n, np.int32)
        self.node = np.arange(n, dtype=np.int64)
        self.ctype = np.zeros(n, np.int8)
        self.T = np.full(n, 7.0)
        self.age = np.zeros(n)
        self.origin = np.zeros(n, np.int16)   # 地体来源：初始陆块编号；90 = 洋内弧增生；0 = 洋壳
        self.prov = np.zeros(n, np.int32)     # 所属古老地壳省份（前史生成，见 prehistory.py）；0 = 无
        self.t_form = np.full(n, np.nan)      # 地壳形成时间（Ma）：洋壳 = 洋脊生成时刻，弧地壳 = 转为陆壳时刻，省份 = 省份年龄
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
