"""两条路线共用的情景："残留洋盆"试验（阶段 1 情景族第 2 步的缩小版，100 Ma → 0 Ma，3 块板块起步）。

初始条件（100 Ma）：
- 北大陆 C1（约 4100 万 km²）带一个西南岬角（类似伊比利亚），南大陆 C2（约 2000 万 km²）带一个东北岬角（类似阿拉伯）。
  两者之间是一片东西向的古洋盆（类似特提斯洋），它属于南大陆的板块，向北俯冲到北大陆之下。
- 板块 1 = C1 及其外 100 km 内的洋壳；板块 2 = C2 及大片洋壳（含古洋盆）；板块 3 = 另半球的大洋。
- 大陆形状故意用几个球冠拼成、边界光滑，**没有噪声**：结果里的海岸复杂度只能来自构造过程，便于判断两条路线各自产生了什么。
- 大陆边缘 250 km 内地壳由 35 km 减薄到 22 km（被动陆缘的陆架、陆坡）。

目标（情景想要的结果）：南大陆向北汇聚，东端岬角先与北大陆碰撞并锁住，西端岬角之间只留下一条窄海峡，
中间留下一个东西向伸长、两岸分属两块大陆的残留洋盆——也就是简报 §4 要的主内海的成因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from common import Mesh, chord_from_km, km_from_chord, lonlat_to_xyz, pole_vector, smoothstep, unit
from engine import CONT, OCEAN, Crust
from scipy.spatial import cKDTree

T0 = 100.0   # Ma

# 几何（球冠：纬度、经度、半径°）。两端缺口（陆壳边缘之间）约 1400–1500 km，中段约 2300 km，
# 汇聚约 1500 km 后两端闭合、中段留下约 1000 km 宽的洋盆。
C1_CAPS = [(48.0, 10.0, 27.0), (22.0, -13.0, 9.0)]                         # 北大陆主体 + 西南岬角
C2_CAPS = [(-22.0, 12.0, 22.0), (-8.0, -8.0, 9.0), (-5.0, 30.0, 12.0), (10.0, 38.0, 8.0)]   # 南大陆主体 + 西北凸出 + 东北凸出 + 东北岬角
MARGIN_KM = 250.0
TETHYS_LON = (-25.0, 60.0)

# 基准运动（°/Myr）：板块 2 相对板块 1 绕西侧赤道上的欧拉极旋转向北汇聚（两端闭合速度相近），
# 板块 3 整体向西运动（在北大陆东岸俯冲、在两块大陆西侧形成洋脊），板块 1 缓慢漂移。
BASE = {
    "pole21": (0.0, -80.0), "rate21": 0.28,
    "pole3": (-90.0, 0.0), "rate3": 0.25,
    "pole1": (30.0, 100.0), "rate1": 0.05,
}


def caps_mask(mesh: Mesh, caps) -> np.ndarray:
    m = np.zeros(mesh.n, bool)
    for lat, lon, r in caps:
        m |= mesh.xyz @ lonlat_to_xyz(lon, lat) >= np.cos(np.radians(r))
    return m


def distance_to(mesh: Mesh, mask: np.ndarray, query: np.ndarray | None = None) -> np.ndarray:
    q = mesh.xyz if query is None else query
    d, _ = cKDTree(mesh.xyz[mask]).query(q, workers=-1)
    return km_from_chord(d)


def initial_crust(mesh: Mesh) -> Crust:
    c = Crust(mesh.n)
    c1, c2 = caps_mask(mesh, C1_CAPS), caps_mask(mesh, C2_CAPS)
    c2 &= ~c1
    cont = c1 | c2
    d1, d2 = distance_to(mesh, c1), distance_to(mesh, c2)
    # 板块 3 的"源"：另半球赤道上两点（加权 Voronoi：d1/0.07、d2/1、d3/1 取最小）
    s3 = lonlat_to_xyz(np.array([-110.0, 120.0]), np.array([0.0, 0.0]))
    d3 = km_from_chord(cKDTree(s3).query(mesh.xyz)[0])
    score = np.stack([d1 / 0.07, d2 / 1.0, d3 / 1.0])
    c.plate = (np.argmin(score, 0) + 1).astype(np.int32)
    c.plate[c1] = 1
    c.plate[c2] = 2

    c.ctype[cont] = CONT
    ocean = ~cont
    d_in = distance_to(mesh, ocean)                 # 陆地节点到海岸的距离
    c.T[cont] = 22.0 + 13.0 * smoothstep(d_in[cont] / MARGIN_KM)
    c.T[ocean] = 7.0
    c.origin[c1] = 1
    c.origin[c2] = 2
    # 洋壳年龄（初始条件的简化）：靠近大陆的老（被动陆缘），远离的新；古洋盆最老
    dc = np.minimum(d1, d2)
    c.age[ocean] = 120.0 - 60.0 * smoothstep(dc[ocean] / 2500.0)
    lon = mesh.lon
    tethys = ocean & (c.plate == 2) & (lon > TETHYS_LON[0]) & (lon < TETHYS_LON[1]) & (mesh.lat > -5) & (mesh.lat < 40)
    c.age[tethys] = 140.0
    return c


@dataclass
class Kinematics:
    """一个种子的运动参数（角速度矢量，rad/Myr）与裂解剧本。"""
    omega1: np.ndarray
    omega21: np.ndarray                  # 板块 2 相对板块 1
    omega3: np.ndarray
    pole21: tuple[float, float]
    rate21: float
    rift: dict = field(default_factory=dict)
    jitter: dict = field(default_factory=dict)

    @property
    def omega2(self) -> np.ndarray:
        return self.omega1 + self.omega21


def jitter_pole(rng, lat, lon, sigma_deg):
    return (float(np.clip(lat + rng.normal(0, sigma_deg), -89, 89)), float(lon + rng.normal(0, sigma_deg)))


def seed_kinematics(seed: int) -> Kinematics:
    """按种子扰动基准运动（两条路线用同一个扰动，所以起点完全相同）。"""
    rng = np.random.default_rng(1000 + seed)
    b = BASE
    p21 = jitter_pole(rng, *b["pole21"], 6.0)
    r21 = b["rate21"] * rng.uniform(0.85, 1.15)
    p3 = jitter_pole(rng, *b["pole3"], 8.0)
    r3 = b["rate3"] * rng.uniform(0.8, 1.2)
    p1 = (rng.uniform(-60, 60), rng.uniform(-180, 180))
    r1 = rng.uniform(0.02, 0.08)
    # 裂解剧本（路线 B 照此执行；路线 A 的裂解是随机触发的，不用它）：北大陆西北角裂离，形成新洋盆
    s1 = lonlat_to_xyz(-5.0 + rng.normal(0, 3), 42.0 + rng.normal(0, 3))
    s2 = lonlat_to_xyz(-38.0 + rng.normal(0, 3), 66.0 + rng.normal(0, 3))
    obliq = np.radians(rng.uniform(20, 50))
    axis0 = unit(np.cross(s1, s2))
    tilt = unit(np.cross(axis0, unit(s1 + s2)))
    axis = unit(np.cos(obliq) * axis0 + np.sin(obliq) * tilt * rng.choice([-1, 1]))
    rift = {"t": float(rng.uniform(40, 55)), "parent": 1, "s1": s1, "s2": s2, "axis": axis,
            "rate": float(rng.uniform(0.2, 0.35)), "obliquity_deg": float(np.degrees(obliq)),
            "rng_seed": int(rng.integers(1 << 31))}
    return Kinematics(
        omega1=pole_vector(p1[0], p1[1], r1), omega21=pole_vector(p21[0], p21[1], r21),
        omega3=pole_vector(p3[0], p3[1], r3), pole21=p21, rate21=r21, rift=rift,
        jitter={"pole21": p21, "rate21": r21, "pole3": p3, "rate3": r3, "pole1": p1, "rate1": r1})


def land_mask_initial(c: Crust) -> np.ndarray:
    """初始高程 > 0 的陆地（35 km 陆壳 +400 m，减薄陆缘约在边缘内 170 km 处入海）。"""
    from engine import Params, elevation
    return elevation(c, Params()) > 0


__all__ = ["T0", "initial_crust", "seed_kinematics", "Kinematics", "caps_mask", "distance_to", "chord_from_km",
           "land_mask_initial", "BASE"]
