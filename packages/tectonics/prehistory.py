"""前史（阶段 1.2b，U20 修订）：生成 300 Ma 的初始大陆。

为什么要有前史：300 Ma 的大陆不是凭空出现的。地球上 3 亿年前的各大陆（劳伦、波罗的、冈瓦纳……）的外形，
主要来自更老的超大陆**沿古老薄弱带裂解**留下的裂谷陆缘；内部是不同年龄的地壳省份拼贴（太古宙克拉通、元古宙活动带、
年轻的增生带），之间是古缝合带。1.2b 试验过"从克拉通核自由演化"：200 Myr 内核的形状基本保留，大陆仍是几团圆疙瘩，
所以改用下面的三步。

1. 古老省份拼贴（初始条件，procedural_constrained，见 belt_mosaic）：约 320 个大小、伸长、走向各不相同的地壳省份
   （各向异性球面 Voronoi），8–12 个等轴偏大的太古宙克拉通；靠近克拉通的省份伸长 1.8–4 倍、长轴环绕克拉通——
   地球上的元古宙活动带也是环绕克拉通的狭长弧形带。超大陆按省份逐个长出，外缘由省份边界拼成。
   离克拉通越远的省份越年轻（大陆从核向外增生），外缘是古生代增生带；不同克拉通的活动带相遇处是碰撞缝合带。
   每个省份记录年龄、类型、地壳厚度，以及矿产需要的继承属性（绿岩带、条带状铁建造、金伯利岩省；
   docs/mineral_resources.md §3）。另有几个椭圆碟状的克拉通内盆地。
   （前两版：各向同性 Voronoi 像蜂窝，裂开后的陆缘缺少窄长的凸出与凹入；克拉通外套同心圆环，外缘成了大圆弧，更假。）
   **省份的几何是程序生成的初始条件，海岸线不是**：海岸由下面两步的过程产生。
2. 沿薄弱带裂解（过程）：省份之间的边界按两侧年龄与类型给出"薄弱度"（年轻缝合带弱，克拉通之间强）。
   把超大陆分成 8–12 块时，让分界线优先走薄弱的缝合带、绕开克拉通（地球上的裂谷也这样，例如东非裂谷绕过坦桑尼亚克拉通）。
   **分界要能真正张开**（U20）：先定各陆块的漂移（每对相邻陆块在分界上都在分开、不以走滑为主），再把每条分界整理成
   相对运动极下的"洋脊段 + 转换断层"：沿极的每条纬度带只穿过一次，陆块沿纬度圈分开时处处张开。整理时尽量沿原来的
   薄弱带走（动态规划，偏离一个单元、错开一次都有代价），只在咬合处局部截弯。
   1.2b 初版的分界逐段沿省份边界，参差咬合，任何刚体漂移都分不开，到 300 Ma 多数陆块对仍只隔一个网格。
   没被用作分界的强薄弱带，挑几条从陆缘伸进内陆的减薄成"夭折裂谷"（例如贝努埃槽），以后会成为深入内陆的海湾、低地。
   新裂开的陆缘按距离减薄（陆架、陆坡）。
3. 漂移（过程，引擎 + 剧本 + pyGPlates）：t_start -> t_end（默认 400 -> 300 Ma），各陆块按上一步定的角速度散开，
   之间生出新洋壳，外缘向外推进处发生俯冲、长出岛弧。

**设计与网格分开**（U20）：所有随机抽样与省份级的决定（省份、超大陆、分块、漂移、分界整理、夭折裂谷）都在固定的
设计网格（L6）上做一次；目标网格（L5/L6/L7……）只按同一份设计生成节点上的地壳，再跑引擎。所以同一种子在不同网格上
是同一个世界，L6 上的筛选对 L7 有效（1.2b 初版在各自网格上抽样，L6、L7 的结果完全不同）。

输出 InitialWorld：t_end 时刻的固定网格地壳（含板块划分、省份、形成年龄）、各板块角速度、省份表、事件日志、前史快照。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .crust import CONT, ORO_RIFT, Crust, Params
from .engine import Engine, graph_components
from .geometry import (DEG, R_KM, chord_from_km, km_from_chord, omega_rotmat, rotmat, smoothstep, unit, wrap,
                       xyz_to_lonlat)
from .kinematics import RotationFileKinematics, StageScript
from .mesh import CACHE, KD_WORKERS, Mesh, load_mesh

PROV_TYPES = {1: "太古宙克拉通", 2: "古元古代活动带", 3: "中—新元古代活动带", 4: "古生代增生带"}
AGE_RANGE_MA = {1: (2500.0, 3600.0), 2: (1600.0, 2500.0), 3: (540.0, 1600.0), 4: (420.0, 540.0)}
# 400 Ma 时的地壳厚度（km）：古生代增生带是年轻造山带，仍有 1 km 上下的起伏；克拉通内盆地另设
T_RANGE_KM = {1: (35.5, 36.5), 2: (35.0, 36.0), 3: (34.5, 35.5), 4: (37.0, 40.0)}


@dataclass
class PrehistoryParams:
    t_start: int = 400                   # 老超大陆裂解的时刻（Ma）
    t_end: int = 300
    cont_area_km2: float = 1.75e8        # 老超大陆面积 ≈ 300 Ma 陆壳总量（0 Ma 陆地约 29% 的目标再加陆架）
    n_seeds: int = 320                   # 全球省份种子数（平均间距约 1400 km；大小、伸长各不相同）
    n_cratons: tuple = (8, 12)           # 太古宙克拉通个数
    n_blocks: tuple = (8, 12)            # U20：1.2b 初版 5–7 块，主大陆只由两三块拼成（亚非欧由十多块拼成）
    block_size_sigma: float = 0.35       # 分块大小权重的对数标准差：陆块大小有大有小
    n_basins: tuple = (3, 5)
    n_failed_rifts: tuple = (2, 4)
    n_ocean_plates: int = 3
    halo_km: float = 700.0               # 陆块周围这一范围内的洋壳与陆块同属一块板块
    margin_km: float = 250.0             # 超大陆外缘陆缘减薄宽度
    rift_margin_km: float = 200.0        # 裂谷陆缘减薄宽度
    failed_rift_km: float = 120.0        # 夭折裂谷减薄带半宽
    tangential_cost: float = 1.5         # 分块生长时绕超大陆中心切向生长的额外代价倍数（外圈陆块成楔形，见 breakup_blocks）
    nested_share: float = 0.5            # 与某个邻块的共享边界超过自身周长的这一比例，就并入那块（嵌套的分不开）
    touch_km: float = 250.0              # 漂移评分：300 Ma 时相距小于这个就算连在一起（与剧本规划器的编组标准相同）
    # 漂移目标：以超大陆中心为心的膨胀，离开中心的角速度 = 该值 × 到中心的角距（60° 处 0.3°/Myr ≈ 3.3 cm/yr）；
    # 100 Myr 后外缘（约 72°）到约 108°。取 0.0075 时外缘到 126°，在球面另一侧互相撞上
    expansion_per_myr: float = 0.005
    max_drift_deg: float = 0.7           # 单个陆块漂移角速度上限（°/Myr，约 7.8 cm/yr；地球上快的板块 10 cm/yr 以上）
    sep_rate_deg: float = 0.08           # 相邻陆块在分界上至少以这个相对角速度分开（°/Myr，约 0.9 cm/yr，100 Myr 约 900 km）
    max_obliquity: float = 1.2           # 分界中点处相对运动的走滑分量 / 张开分量上限（约 50°）：不以走滑为主
    min_gap_km: float = 800.0            # 300 Ma 时陆块两两最小间距的目标（刚体估计）
    drift_tries: int = 30
    cut_window_km: float = 600.0         # 分界整理只动原分界两侧这一范围内的单元
    cut_jump_km: float = 120.0           # 分界整理时转换断层每错开这么长（约一个设计网格间距），代价相当于改划一个单元：
                                         # 错开太长的转换断层两侧一直滑过、到 300 Ma 仍挨着
    min_block_km2: float = 3.0e6
    design_level: int = 6                # 设计网格（所有抽样与省份级决定都在这上面做）


@dataclass
class InitialWorld:
    level: int
    seed: int
    t: float
    crust: Crust                         # 固定网格视图（node = 固定节点编号）
    omega: dict                          # 各板块在 t 时刻的角速度矢量（rad/Myr）
    provinces: list
    blocks: list
    events: list
    boundaries: list = field(default_factory=list)   # 省份边界：两侧省份编号、薄弱度、是否成了陆块分界、是否夭折裂谷
    snapshots: list = field(default_factory=list)
    rot_path: str = ""
    params: dict = field(default_factory=dict)


# ---- 1. 省份拼贴 ------------------------------------------------------------------------------
@dataclass
class Mosaic:
    cell: np.ndarray          # 每个设计网格节点所属省份（-1 = 洋）
    ptype: dict               # 省份 -> 类型 1–4
    age: dict                 # 省份 -> 年龄（Ma）
    craton: dict              # 省份 -> 所属克拉通（环带围绕的那一个）
    ring: dict                # 省份 -> 环序号（0 = 克拉通本身）
    craton_cells: list
    basins: list              # 克拉通内盆地：(中心, 长轴, 半径 km, 伸长, 最大减薄 km)
    gen: tuple = ()           # 省份生成器（种子、长轴、尺度、伸长度）：任何网格上都能生成同样的省份
    lut: np.ndarray | None = None   # 种子编号 -> 省份编号（-1 = 洋）


def _poisson(rng, n, min_sep_rad, fixed=None):
    pts = [] if fixed is None else list(fixed)
    tries = 0
    while len(pts) < n and tries < 300 * n:
        x = unit(rng.normal(size=3))
        tries += 1
        if not pts or (np.asarray(pts) @ x).max() < np.cos(min_sep_rad):
            pts.append(x)
    return np.asarray(pts)


def _hops(nbrs, members, sources):
    """省份图上到最近源的跳数，以及最近的源。"""
    dist = {s: 0 for s in sources}
    src = {s: s for s in sources}
    frontier = list(sources)
    while frontier:
        nxt = []
        for i in frontier:
            for j in nbrs[i]:
                if j in members and j not in dist:
                    dist[j] = dist[i] + 1
                    src[j] = src[i]
                    nxt.append(j)
        frontier = nxt
    return dist, src


def keep_largest(mesh: Mesh, label: np.ndarray, mask: np.ndarray | None = None, rounds: int = 5) -> np.ndarray:
    """每个标签只保留面积最大的连通块，其余节点由相邻的保留块逐圈吸收（mask 之外的节点不参与、不改动）。"""
    label = label.copy()
    m = np.ones(mesh.n, bool) if mask is None else mask
    e = mesh.edges
    e = e[m[e[:, 0]] & m[e[:, 1]]]
    for _ in range(rounds):
        same = e[label[e[:, 0]] == label[e[:, 1]]]
        lab = graph_components(mesh.n, same)
        lab = np.where(m, lab, -1)
        idx = np.flatnonzero(m)
        _, inv = np.unique(lab[idx], return_inverse=True)
        n_lab = inv.max() + 1
        size = np.bincount(inv, weights=mesh.area_km2[idx], minlength=n_lab)
        lab_of = np.zeros(n_lab, np.int64)
        lab_of[inv] = label[idx]
        best = {}
        for l_ in np.argsort(size)[::-1]:
            best.setdefault(int(lab_of[l_]), int(l_))
        keep = np.zeros(n_lab, bool)
        keep[list(best.values())] = True
        bad = np.zeros(mesh.n, bool)
        bad[idx] = ~keep[inv]
        if not bad.any():
            break
        for _ in range(200):
            m1 = bad[e[:, 0]] & ~bad[e[:, 1]]
            m2 = bad[e[:, 1]] & ~bad[e[:, 0]]
            if not (m1.any() or m2.any()):
                break
            label[e[m1, 0]] = label[e[m1, 1]]
            bad[e[m1, 0]] = False
            label[e[m2, 1]] = label[e[m2, 0]]
            bad[e[m2, 1]] = False
    return label


def anisotropic_cells(mesh: Mesh, seeds, along, scale_km, aspect, chunk: int = 20000) -> np.ndarray:
    """各向异性球面 Voronoi：每个种子有自己的大小、伸长度和长轴方向；距离在种子处的对数映射坐标里按椭圆度量。
    之后每个省份只保留最大的连通块，零碎部分并给相邻省份。返回每个节点的种子编号（与网格无关，便于跨网格对应）。"""
    S, U = seeds, along
    V = np.cross(S, U)
    L = scale_km * np.sqrt(aspect) / R_KM
    W = scale_km / np.sqrt(aspect) / R_KM
    cell = np.empty(mesh.n, np.int64)
    for s in range(0, mesh.n, chunk):
        X = mesh.xyz[s:s + chunk]
        c = np.clip(X @ S.T, -1, 1)
        th = np.arccos(c)
        f = np.where(th > 1e-9, th / np.maximum(np.sqrt(1 - c * c), 1e-12), 1.0)
        pu, pv = (X @ U.T) * f, (X @ V.T) * f
        d = (pu / L) ** 2 + (pv / W) ** 2
        d[c < -0.2] = np.inf                       # 远于约 100° 的种子不参与
        cell[s:s + chunk] = np.argmin(d, 1)
    return keep_largest(mesh, cell)


def belt_mosaic(mesh: Mesh, rng, p: "PrehistoryParams") -> Mosaic:
    """各向异性省份拼贴（1.2b 第三版）。
    第一版各向同性 Voronoi 像蜂窝；第二版克拉通外套同心圆环，超大陆外缘成了大圆弧，裂开后是"圆弧 + 锯齿"，更假。
    这一版：
    - 克拉通种子放在超大陆所在区域内，等轴、偏大；
    - 其余种子大小不一（对数正态），靠近克拉通的伸长 1.8–4 倍、长轴沿环绕最近克拉通的方向（±25°），即狭长弧形活动带；
    - 超大陆从中心所在的省份逐个长出（共享边界多者优先，加随机扰动），外缘由省份边界拼成；
    - 年龄按到克拉通的省份跳数递减，外缘年轻；另有几个椭圆碟状的克拉通内盆地。"""
    target = p.cont_area_km2
    rho = np.arccos(1.0 - 1.4 * target / (2 * np.pi * R_KM**2))      # 超大陆所在球冠（面积为目标的 1.4 倍）
    C = unit(rng.normal(size=3))
    nc = int(rng.integers(*p.n_cratons, endpoint=True))
    crat_seeds = []
    tries = 0
    while len(crat_seeds) < nc and tries < 5000:
        tries += 1
        x = unit(rng.normal(size=3))
        if np.arccos(np.clip(x @ C, -1, 1)) > 0.8 * rho:
            continue
        if crat_seeds and (np.asarray(crat_seeds) @ x).max() > np.cos(1400.0 / R_KM):
            continue
        crat_seeds.append(x)
    nc = len(crat_seeds)
    spacing = np.sqrt(4 * np.pi / p.n_seeds)
    seeds = _poisson(rng, p.n_seeds, 0.65 * spacing, fixed=crat_seeds)
    ns = len(seeds)
    cs = np.asarray(crat_seeds)
    near = np.argmax(seeds @ cs.T, 1)
    t = cs[near] - (cs[near] * seeds).sum(1, keepdims=True) * seeds             # 指向最近克拉通的切向
    rnd = rng.normal(size=(ns, 3))
    rnd -= (rnd * seeds).sum(1, keepdims=True) * seeds
    t = unit(np.where(np.linalg.norm(t, axis=1, keepdims=True) > 1e-6, t, rnd))  # 克拉通种子自己：随机切向
    along = unit(np.cross(seeds, t))                                             # 环绕方向
    rot = np.radians(rng.uniform(-25, 25, ns))
    along = unit(np.cos(rot)[:, None] * along + np.sin(rot)[:, None] * t)
    scale = spacing * R_KM * 0.5 * rng.lognormal(0.0, 0.3, ns)
    in_region = np.arccos(np.clip(seeds @ C, -1, 1)) < 1.2 * rho
    aspect = np.where(in_region, rng.uniform(1.8, 4.0, ns), rng.uniform(1.0, 1.6, ns))
    aspect[:nc] = rng.uniform(1.0, 1.4, nc)
    scale[:nc] *= 1.3
    cell = anisotropic_cells(mesh, seeds, along, scale, aspect)
    cell_area = np.bincount(cell, weights=mesh.area_km2, minlength=ns)
    shared, nbrs, perim = cell_graph(mesh, cell, ns)
    # 超大陆：从中心所在省份长出
    start = int(cell[np.argmax(mesh.xyz @ C)])
    inside = np.zeros(ns, bool)
    inside[start] = True
    area = cell_area[start]
    while area < target:
        front = sorted({j for i in np.flatnonzero(inside) for j in nbrs[i] if not inside[j]})
        best, score = None, -np.inf
        for j in front:
            s = sum(_sh(shared, i, j) for i in nbrs[j] if inside[i]) / max(perim[j], 1) + 0.4 * rng.random()
            if s > score:
                best, score = j, s
        inside[best] = True
        area += cell_area[best]
    present = cell_area > 0
    out = np.flatnonzero(~inside & present)
    loc = {c_: k_ for k_, c_ in enumerate(out)}
    oe = np.asarray([(loc[a], loc[b]) for (a, b) in shared if a in loc and b in loc], np.int64).reshape(-1, 2)
    lab = graph_components(len(out), oe)
    big = np.bincount(lab, weights=cell_area[out]).argmax()
    inside[out[lab != big]] = True                       # 被包围的洋区并入
    members = set(np.flatnonzero(inside & present).tolist())
    # 克拉通 = 含克拉通种子的省份
    craton_cells = sorted({int(cell[np.argmax(mesh.xyz @ x)]) for x in crat_seeds} & members)
    hops, src = _hops(nbrs, members, craton_cells)
    periphery = {i for i in members if any(not inside[j] for j in nbrs[i])}
    ptype, age, crat, ring = {}, {}, {}, {}
    for i in sorted(members):
        h = hops.get(i, 9)
        if h == 0:
            a_ = rng.uniform(2500, 3600)
        elif i in periphery and h >= 2:
            a_ = rng.uniform(420, 540)
        elif h == 1:
            a_ = rng.uniform(1600, 2500)
        else:
            a_ = rng.uniform(560, 1600)
        age[i], ring[i], crat[i] = float(a_), int(h), int(src.get(i, -1))
        ptype[i] = 1 if a_ >= 2500 else 2 if a_ >= 1600 else 3 if a_ >= 540 else 4
    # 重新编号：超大陆省份 0..m-1，洋 = -1
    order = sorted(members)
    remap = {c_: k_ for k_, c_ in enumerate(order)}
    lut = np.full(ns, -1, np.int64)
    lut[order] = np.arange(len(order))
    out_cell = lut[cell]
    basins = []
    hosts = [i for i in order if ptype[i] in (1, 2)]
    for i in rng.choice(hosts, size=min(int(rng.integers(*p.n_basins, endpoint=True)), len(hosts)), replace=False):
        sel = np.flatnonzero(cell == i)
        c0 = mesh.xyz[rng.choice(sel)]
        ax_ = unit(np.cross(c0, rng.normal(size=3)))
        basins.append((c0, ax_, float(rng.uniform(250, 500)), float(rng.uniform(1.0, 2.5)), float(rng.uniform(3.0, 5.0))))
    return Mosaic(out_cell, {remap[i]: ptype[i] for i in order}, {remap[i]: age[i] for i in order},
                  {remap[i]: crat[i] for i in order}, {remap[i]: ring[i] for i in order},
                  [remap[i] for i in craton_cells], basins, (seeds, along, scale, aspect), lut)


def cell_graph(mesh: Mesh, cell: np.ndarray, n: int):
    """省份邻接：共享网格边数（≈ 共享边界长度 / 网格间距）。"""
    e = mesh.edges
    ci, cj = cell[e[:, 0]], cell[e[:, 1]]
    d = ci != cj
    pairs = np.sort(np.stack([ci[d], cj[d]], 1), axis=1)
    up, cnt = np.unique(pairs, axis=0, return_counts=True)
    shared = {(int(a), int(b)): int(k) for (a, b), k in zip(up, cnt)}
    nbrs = {i: set() for i in range(n)}
    for a, b in shared:
        nbrs[a].add(b)
        nbrs[b].add(a)
    perim = np.zeros(n)
    for (a, b), k in shared.items():
        perim[a] += k
        perim[b] += k
    return shared, nbrs, perim


def _sh(shared, a, b):
    return shared.get((a, b) if a < b else (b, a), 0)


def _farthest(rng, cand, centers, k):
    chosen = [int(rng.choice(cand))]
    while len(chosen) < min(k, len(cand)):
        rest = np.array([c_ for c_ in cand if c_ not in chosen])
        d = np.min(np.arccos(np.clip(centers[rest] @ centers[chosen].T, -1, 1)), axis=1)
        top = rest[np.argsort(d)[::-1][:3]]
        chosen.append(int(rng.choice(top)))
    return chosen


# ---- 2. 薄弱带与裂解分块 -----------------------------------------------------------------------
def boundary_weakness(rng, ptype, craton, pairs):
    """省份边界的薄弱度（越大越容易裂开）：取决于两侧中较年轻的一方；不同类型之间更弱；
    属于不同克拉通的环带相遇处是碰撞缝合带，再弱一些；克拉通之间最强。"""
    base = {4: 1.0, 3: 0.7, 2: 0.45, 1: 0.25}
    w = {}
    for a, b in pairs:
        v = base[max(ptype[a], ptype[b])]
        if ptype[a] != ptype[b]:
            v += 0.25
        if craton[a] != craton[b]:
            v += 0.3
        w[(a, b)] = v + 0.3 * rng.random()
    return w


def sector_seeds(rng, members, centers, c_sc, k, cratons, rho):
    """分块种子：超大陆中央一块，其余 k-1 块按方位角均匀分布在外圈（离中心 0.55–0.75 倍超大陆半径），
    有克拉通在附近就用克拉通。超大陆像橘子瓣一样裂开：中央一块基本不动，外圈各瓣向外散开（盘古大陆裂解时非洲居中、
    美洲、欧亚、南极、澳大利亚、印度向四周散开）。"""
    mem = np.array(sorted(members))
    e1 = unit(np.cross(c_sc, [0.0, 0.0, 1.0]) if abs(c_sc[2]) < 0.9 else np.cross(c_sc, [1.0, 0.0, 0.0]))
    e2 = np.cross(c_sc, e1)
    crat = np.array(sorted(cratons)) if cratons else np.zeros(0, np.int64)

    def nearest(x, avoid):
        cand_c = [c_ for c_ in crat if c_ not in avoid and np.arccos(np.clip(centers[c_] @ x, -1, 1)) < np.radians(8)]
        if cand_c:
            return int(min(cand_c, key=lambda c_: -centers[c_] @ x))
        free = [m_ for m_ in mem if m_ not in avoid]
        return int(free[int(np.argmax(centers[free] @ x))])

    chosen = [nearest(c_sc, set())]
    az0 = rng.uniform(0, 2 * np.pi)
    for i in range(k - 1):
        az = az0 + 2 * np.pi * i / (k - 1) + rng.uniform(-0.25, 0.25) * 2 * np.pi / (k - 1)
        r = rng.uniform(0.55, 0.75) * rho
        x = np.cos(r) * c_sc + np.sin(r) * (np.cos(az) * e1 + np.sin(az) * e2)
        chosen.append(nearest(x, set(chosen)))
    return chosen


def breakup_blocks(members, seeds, nbrs, weak, cell_area, size_w, centers, c_sc, tangential_cost):
    """轮流生长分块：每个陆块从自己的种子按路径代价由低到高吞并相邻省份，越过薄弱边界代价高，前锋在薄弱带停住，
    所以分界线优先落在薄弱带上；每一步由"面积 / 大小权重"最小的陆块生长，大小有差别但不会一块独大。
    每一步的代价按两省份中心的距离计（不按跳数），绕超大陆中心的切向生长再贵 tangential_cost 倍：
    活动带省份狭长、环绕克拉通，按跳数时外圈陆块沿活动带长成跨 120° 的月牙、内圈陆块被包在中间，任何漂移都分不开；
    径向生长便宜时外圈陆块成为楔形（橘子瓣），向外散开时彼此分开。
    （1.2b 初版是多源最短路、各块生长速度不同：8 块里一块吞了 113 个省份中的 82 个。）"""
    owner = {}
    heaps = [[(0.0, int(s))] for s in seeds]
    area = np.zeros(len(seeds))
    active = set(range(len(seeds)))
    while active:
        b = min(active, key=lambda x: (area[x] / size_w[x], x))
        h = heaps[b]
        while h and h[0][1] in owner:
            heapq.heappop(h)
        if not h:
            active.discard(b)
            continue
        d, i = heapq.heappop(h)
        owner[i] = b
        area[b] += cell_area[i]
        ri = centers[i] - (centers[i] @ c_sc) * c_sc
        ri = unit(ri) if np.linalg.norm(ri) > 1e-6 else None
        for j in sorted(nbrs[i]):
            if j in members and j not in owner:
                key = (i, j) if i < j else (j, i)
                dist = np.arccos(np.clip(centers[i] @ centers[j], -1, 1)) * R_KM / 1000.0
                aniso = 1.0
                if ri is not None:
                    v = centers[j] - centers[i]
                    v = v - (v @ centers[i]) * centers[i]
                    if np.linalg.norm(v) > 1e-9:
                        tf = np.linalg.norm(v - (v @ ri) * ri) / np.linalg.norm(v)
                        aniso = 1.0 + tangential_cost * tf
                heapq.heappush(h, (d + dist * aniso * (0.3 + 3.0 * weak[key]), j))
    return owner


def _boundary_distance(mesh, mask_a, mask_b, query, max_km):
    """query 节点到 mask_a 与 mask_b 交界处的距离（km，超过 max_km 为 inf）。"""
    e = mesh.edges
    hit = (mask_a[e[:, 0]] & mask_b[e[:, 1]]) | (mask_b[e[:, 0]] & mask_a[e[:, 1]])
    bn = np.unique(e[hit].ravel())
    if bn.size == 0:
        return np.full(len(query), np.inf)
    d, _ = cKDTree(mesh.xyz[bn]).query(mesh.xyz[query], distance_upper_bound=chord_from_km(max_km), workers=KD_WORKERS)
    return km_from_chord(d)


# ---- 2b. 能张开的分界（U20） ----------------------------------------------------------------------
@dataclass
class RiftCut:
    """一对相邻陆块的分界整理结果。在相对运动极 axis 的坐标里（θ = 到极的角距，φ = 绕极的方位角），
    分界是每条 θ 带上的一个 φ（相对 phi_ref）：φ 大的一侧归 b。b 相对 a 绕 axis 正向转动时 φ 增大，所以处处张开；
    相邻两带的 φ 不同处就是转换断层（沿纬度圈，只滑动不张开也不挤压）。"""
    a: int
    b: int
    axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    phi_ref: float
    th0: float
    dth: float
    cut: np.ndarray
    window_km: float

    def polar(self, X):
        th = np.arccos(np.clip(X @ self.axis, -1, 1))
        d = wrap(np.arctan2(X @ self.e2, X @ self.e1) - self.phi_ref)
        k = np.floor((th - self.th0) / self.dth).astype(np.int64)
        return th, d, k

    def assign_b(self, X):
        """返回 (在整理范围内, 归 b)。"""
        _, d, k = self.polar(X)
        ok = (k >= 0) & (k < len(self.cut))
        return ok, d > self.cut[np.clip(k, 0, len(self.cut) - 1)]


def monotone_cut(X, is_b, axis, a, b, bin_km, jump_km, window_km) -> RiftCut:
    """把 X（原分界两侧 window_km 内、属于 a 或 b 的单元）上的分界整理成每条 θ 带只有一个 φ：
    动态规划，代价 = 改划的单元数 + 相邻带分界错开的弧长 / jump_km。尽量沿原分界（薄弱带）走，只在咬合处截弯。"""
    w = unit(axis)
    e1 = unit(np.cross(w, [0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0.0, 0.0]))
    e2 = np.cross(w, e1)
    th = np.arccos(np.clip(X @ w, -1, 1))
    ph = np.arctan2(X @ e2, X @ e1)
    phi_ref = float(np.angle(np.exp(1j * ph).mean()))
    d = wrap(ph - phi_ref)
    dth = bin_km / R_KM
    th0 = float(th.min())
    k = np.floor((th - th0) / dth).astype(np.int64)
    nbin = int(k.max()) + 1
    span_km = (d.max() - d.min()) * R_KM
    M = int(np.clip(span_km / 20.0, 60, 400))
    G = np.linspace(d.min() - 1e-6, d.max() + 1e-6, M)
    cost = np.zeros((nbin, M))
    for kk in range(nbin):
        sel = k == kk
        da = np.sort(d[sel & ~is_b])
        db = np.sort(d[sel & is_b])
        cost[kk] = (len(da) - np.searchsorted(da, G, "right")) + np.searchsorted(db, G, "left")
    th_mid = th0 + (np.arange(nbin) + 0.5) * dth
    tot = cost[0].copy()
    back = np.zeros((nbin, M), np.int64)
    dG = np.abs(G[:, None] - G[None, :])
    for kk in range(1, nbin):
        trans = tot[None, :] + max(np.sin(th_mid[kk]), 0.05) * R_KM / jump_km * dG
        back[kk] = np.argmin(trans, 1)
        tot = cost[kk] + trans[np.arange(M), back[kk]]
    g = int(np.argmin(tot))
    path = [g]
    for kk in range(nbin - 1, 0, -1):
        g = int(back[kk, g])
        path.append(g)
    return RiftCut(a, b, w, e1, e2, phi_ref, th0, dth, G[np.array(path[::-1])], window_km)


def _pair_list(mesh: Mesh, blk: np.ndarray):
    e = mesh.edges
    ea, eb = blk[e[:, 0]], blk[e[:, 1]]
    sel = (ea > 0) & (eb > 0) & (ea != eb)
    return sorted({(int(min(x, y)), int(max(x, y))) for x, y in zip(ea[sel], eb[sel])})


def _pair_samples(mesh: Mesh, blk: np.ndarray, a: int, b: int, local_km: float = 400.0,
                  qs: tuple = (0.0, 0.15, 0.5, 0.85, 1.0)):
    """a–b 分界上的取样点（沿分界主方向的两端、15%、50%、85% 处）与该处从 a 指向 b 的法向、分界切向。
    两端也要取：只取 15–85% 时相对运动极可以落在分界的末端，像剪刀一样张开，末端到 300 Ma 仍贴着。"""
    e = mesh.edges
    ea, eb = blk[e[:, 0]], blk[e[:, 1]]
    es = e[((ea == a) & (eb == b)) | ((ea == b) & (eb == a))]
    if len(es) == 0:
        return []
    mid = unit(mesh.xyz[es[:, 0]] + mesh.xyz[es[:, 1]])
    c = unit(mid.mean(0))
    u = mid - c
    w_, v_ = np.linalg.eigh(u.T @ u)
    s = u @ v_[:, -1]
    ta, tb_ = cKDTree(mesh.xyz[blk == a]), cKDTree(mesh.xyz[blk == b])
    Xa, Xb = mesh.xyz[blk == a], mesh.xyz[blk == b]
    r = chord_from_km(local_km)
    out = []
    for q in qs:
        x = mid[np.argmin(np.abs(s - np.quantile(s, q)))]
        ia, ib = ta.query_ball_point(x, r), tb_.query_ball_point(x, r)
        if not ia or not ib:
            continue
        n = Xb[ib].mean(0) - Xa[ia].mean(0)
        n = n - (n @ x) * x
        if np.linalg.norm(n) < 1e-9:
            continue
        n = unit(n)
        out.append((x, n, np.cross(x, n)))
    return out


def apply_cuts(mesh: Mesh, blk: np.ndarray, cuts: list, passes: int = 2) -> np.ndarray:
    """按顺序把各分界整理结果用到某个网格上（设计网格与目标网格用同一函数，结果一致）。
    走两遍：三块陆块交汇处，后整理的分界会改动先整理的分界附近的单元，第二遍把它们改回来（只走一遍时每对陆块
    有 5–30 个单元留在错的一侧）。"""
    blk = blk.copy()
    cont = blk > 0
    for _ in range(passes):
        for ct in cuts:
            ma, mb = blk == ct.a, blk == ct.b
            q = np.flatnonzero(ma | mb)
            d = _boundary_distance(mesh, ma, mb, q, ct.window_km)
            q = q[np.isfinite(d)]
            if q.size == 0:
                continue
            ok, to_b = ct.assign_b(mesh.xyz[q])
            blk[q[ok]] = np.where(to_b[ok], ct.b, ct.a)
    return np.where(cont, keep_largest(mesh, blk, cont), 0)


def choose_drift(mesh: Mesh, blk: np.ndarray, nb: int, c_sc: np.ndarray, p: "PrehistoryParams", rng):
    """各陆块的漂移角速度（rad/Myr）与各条分界的整理结果。每次抽样：
    1. 目标漂移 ω̃：以老超大陆中心为心的膨胀——离开中心，速率与到中心的角距成正比（expansion_per_myr），
       方向偏转 ±15°、速率 ±15%，加少量自转。分块是"中央一块 + 外圈楔形"（sector_seeds、breakup_blocks），
       膨胀时楔形之间、楔形与中央一块之间都在分开。
       （1.2b 初版各块随机速率离开中心，同一方向上内圈快于外圈的就贴在一起。试过把各块当圆盘推开到互不重叠的目标：
       当时的陆块是沿外圈的长条月牙、内圈的被包在中间，圆盘模型根本放不下，什么漂移都分不开——问题在分块形状。）
    2. 二次规划求最接近 ω̃ 的一组 ω：每对相邻陆块在分界中点处以 ≥ sep_rate_deg 分开、走滑分量不超过张开分量的
       max_obliquity 倍，单块速率 ≤ max_drift_deg（约束带松弛量重罚，总有解。三处取样都要求张开时，弯曲的分界两端
       法向相反，约束互相矛盾）；
    3. 按相对运动极整理每条分界（monotone_cut），刚体旋转 100 Myr 估计各陆块两两间距
       （中途不相邻的陆块不得穿插），取最小间距最大的一次，达到 min_gap_km 即停。"""
    from scipy.optimize import minimize

    pairs = _pair_list(mesh, blk)
    cbs = {b: unit((mesh.xyz[blk == b] * mesh.area_km2[blk == b, None]).sum(0)) for b in range(1, nb + 1)}
    nz = 3 * nb
    rows, kinds = [], []
    for a, b in pairs:
        smp = _pair_samples(mesh, blk, a, b)
        if not smp:
            continue
        xm, n, t = smp[len(smp) // 2]
        for j, (x, _, _) in enumerate(smp):
            # 两端与中点都按中点的法向张开：相对运动极不会落在分界上（否则像剪刀一样一端张开、另一端一直贴着）；
            # 用同一个法向，弯曲分界两端的约束也不会互相矛盾（弯曲处靠 monotone_cut 截弯）
            nx = unit(n - (n @ x) * x)
            g = np.zeros(nz)
            g[3 * (b - 1):3 * b] = np.cross(x, nx)
            g[3 * (a - 1):3 * a] = -np.cross(x, nx)
            rows.append((g, 0.0, p.sep_rate_deg * (1.0 if j == len(smp) // 2 else 0.7)))
        g = np.zeros(nz)
        g[3 * (b - 1):3 * b] = np.cross(xm, n)
        g[3 * (a - 1):3 * a] = -np.cross(xm, n)
        h = np.zeros(nz)
        h[3 * (b - 1):3 * b] = np.cross(xm, t)
        h[3 * (a - 1):3 * a] = -np.cross(xm, t)
        rows.append((p.max_obliquity * g - h, 0.0, 0.0))             # 中点处走滑 ≤ k × 张开（两个方向）
        rows.append((p.max_obliquity * g + h, 0.0, 0.0))
    A = np.array([r[0] for r in rows]) if rows else np.zeros((0, nz))
    lo = np.array([r[2] for r in rows])
    m_ = len(A)
    lam = 50.0
    cons = [{"type": "ineq", "fun": lambda y: A @ y[:nz] + y[nz:] - lo, "jac": lambda y: np.hstack([A, np.eye(m_)])},
            {"type": "ineq", "fun": lambda y: np.array([p.max_drift_deg ** 2 - y[3 * i:3 * i + 3] @ y[3 * i:3 * i + 3]
                                                        for i in range(nb)])}]
    bounds = [(None, None)] * nz + [(0.0, None)] * m_
    dur = float(p.t_start - p.t_end)
    best = None
    for k in range(p.drift_tries):
        target = np.zeros(nz)
        for b in range(1, nb + 1):
            cb = cbs[b]
            r_deg = float(np.degrees(np.arccos(np.clip(cb @ c_sc, -1, 1))))
            d = cb - (cb @ c_sc) * c_sc
            if np.linalg.norm(d) < 0.05:                    # 中央一块：基本不动
                d = rng.normal(size=3)
                d = d - (d @ cb) * cb
            d = rotmat(cb, np.radians(rng.uniform(-15.0, 15.0))) @ unit(d)
            rate = p.expansion_per_myr * r_deg * rng.uniform(0.85, 1.15)
            target[3 * (b - 1):3 * b] = unit(np.cross(cb, d)) * rate + cb * rng.uniform(-0.03, 0.03)
        if m_:
            y0 = np.concatenate([target, np.maximum(lo - A @ target, 0.0)])
            res = minimize(lambda y: 0.5 * float((y[:nz] - target) @ (y[:nz] - target)) + 0.5 * lam * float(y[nz:] @ y[nz:]),
                           y0, jac=lambda y: np.concatenate([y[:nz] - target, lam * y[nz:]]),
                           constraints=cons, bounds=bounds, method="SLSQP", options={"maxiter": 300})
            z = res.x[:nz]
            slack = float(res.x[nz:].max()) if m_ else 0.0
        else:
            z, slack = target, 0.0
        om = {b: z[3 * (b - 1):3 * b] * DEG for b in range(1, nb + 1)}
        cuts = []
        cur = blk.copy()
        for a, b in pairs:
            w = om[b] - om[a]
            if np.linalg.norm(w) < 1e-12:
                continue
            ma, mb = cur == a, cur == b
            q = np.flatnonzero(ma | mb)
            dd = _boundary_distance(mesh, ma, mb, q, p.cut_window_km)
            q = q[np.isfinite(dd)]
            if q.size < 4:
                continue
            ct = monotone_cut(mesh.xyz[q], cur[q] == b, w, a, b, mesh.h_km, p.cut_jump_km, p.cut_window_km)
            ok, to_b = ct.assign_b(mesh.xyz[q])
            cur[q[ok]] = np.where(to_b[ok], b, a)
            cuts.append(ct)
        cur = apply_cuts(mesh, blk, cuts)
        moved = int(((cur != blk) & (blk > 0)).sum())
        moved_mkm2 = float(mesh.area_km2[(cur != blk) & (blk > 0)].sum() / 1e6)
        # 刚体估计：t_end 时两两间距；中途（25/50/75/100%）任何两块不得互相压上（相邻陆块在分界处已分开，
        # 远处的部分也可能转过去撞上）。按重叠面积判断，不按距离：转换断层两侧只是滑过、一直挨着，不算穿插。
        X = {b: mesh.xyz[np.flatnonzero(cur == b)] for b in range(1, nb + 1)}
        gaps, crossings = {}, 0
        for f in (0.25, 0.5, 0.75, 1.0):
            P = {b: X[b] @ omega_rotmat(om[b], f * dur).T for b in X if len(X[b])}
            nodes_b = np.concatenate([np.stack([mesh.tree.query(P[b])[1], np.full(len(P[b]), b)], 1) for b in P])
            nodes_b = np.unique(nodes_b, axis=0)
            nn, cnt = np.unique(nodes_b[:, 0], return_counts=True)
            multi = set(nn[cnt > 1].tolist())
            if multi:
                sel = nodes_b[np.isin(nodes_b[:, 0], list(multi))]
                ov = {}
                for node in multi:
                    bs = sorted(sel[sel[:, 0] == node, 1].tolist())
                    for i_ in range(len(bs)):
                        for j_ in range(i_ + 1, len(bs)):
                            ov[(bs[i_], bs[j_])] = ov.get((bs[i_], bs[j_]), 0) + 1
                crossings += sum(1 for v in ov.values() if v > 3)
            if f == 1.0:
                trees = {b: cKDTree(P[b][::2]) for b in P}
                for a in P:
                    for b in P:
                        if b > a:
                            gaps[(a, b)] = float(km_from_chord(trees[b].query(P[a][::2])[0].min()))
        gmin = min(gaps.values()) if gaps else np.inf
        # 评分：300 Ma 时互不相连（间距 ≥ touch_km）的陆块组数最多（剧本规划器按同一标准编组），其次间距大；
        # 中途互相压上重罚；整理分界改划的面积也扣分（每 1 Mkm² 抵 50 分），免得分界离薄弱带太远
        root = list(range(nb + 1))

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for (a, b), g in gaps.items():
            if g < p.touch_km:
                root[find(a)] = find(b)
        groups = len({find(b) for b in range(1, nb + 1)})
        apart = [g for g in gaps.values() if g >= p.touch_km]
        score = (1000.0 * groups - 1500.0 * crossings - 50.0 * moved_mkm2
                 + 0.5 * min(min(apart) if apart else 0.0, 1500.0))
        if best is None or score > best[0]:
            best = (score, om, cuts, gaps, crossings, k + 1, slack, moved, groups)
        if groups == nb and crossings == 0 and gmin >= p.min_gap_km:
            break
    score, om, cuts, gaps, crossings, tries, slack, moved, groups = best
    close = sorted(gaps.items(), key=lambda x: x[1])[:4]
    info = {"groups_rigid": groups, "min_gap_km_rigid": float(min(gaps.values())) if gaps else None,
            "target_gap_km": p.min_gap_km,
            "crossings": crossings, "best_try": tries, "qp_max_slack_deg": slack, "n_adjacent_pairs": len(pairs),
            "cells_reassigned_by_cuts": moved,
            "closest_pairs_km": [[a, b, round(v)] for (a, b), v in close],
            "rates_deg": {b: float(np.linalg.norm(w) / DEG) for b, w in om.items()}}
    return om, cuts, info


# ---- 设计（固定网格）与生成（目标网格） ------------------------------------------------------------
@dataclass
class Design:
    seed: int
    level: int
    mos: Mosaic
    provinces: dict
    weak: dict
    in_pairs: list
    owner: dict               # 省份 -> 分块（整理分界之前）
    block_of_prov: dict       # 省份 -> 多数单元所属陆块（整理之后）
    cuts: list
    omega: dict
    drift: dict
    failed: list
    oseeds: np.ndarray
    o_omega: dict
    nb: int
    n_cratons: int
    h_km: float


def design_prehistory(dm: Mesh, seed: int, p: PrehistoryParams) -> Design:
    rng = np.random.default_rng(seed)
    mos = belt_mosaic(dm, rng, p)
    n = int(mos.cell.max()) + 1
    OC = n                                            # 洋区在省份图里的编号
    cell = np.where(mos.cell >= 0, mos.cell, OC)
    cell_area = np.bincount(cell, weights=dm.area_km2, minlength=n + 1)
    shared, nbrs, perim = cell_graph(dm, cell, n + 1)
    inside = np.ones(n + 1, bool)
    inside[OC] = False
    members = set(range(n))
    centers = np.zeros((n + 1, 3))
    np.add.at(centers, cell, dm.xyz * dm.area_km2[:, None])
    centers = unit(centers)
    cratons = list(mos.craton_cells)
    ptype = mos.ptype
    provinces = {}
    for i in range(n):
        t = ptype[i]
        flags = {"greenstone": bool(t == 1 and rng.random() < 0.7),
                 "banded_iron": bool((t == 1 and rng.random() < 0.5) or (t == 2 and rng.random() < 0.3)),
                 "kimberlite": bool(t == 1 and rng.random() < 0.35)}
        lon, lat = xyz_to_lonlat(centers[i])
        provinces[i] = {"id": i + 1, "type": t, "type_name": PROV_TYPES[t], "age_ma": mos.age[i],
                        "craton": mos.craton[i] + 1, "ring": mos.ring[i],
                        "T_km": float(rng.uniform(*T_RANGE_KM[t])), "area_km2": float(cell_area[i]),
                        "lat_lon_at_start": [float(lat), float(lon)], **flags}
    # 薄弱带与分块
    in_pairs = sorted((a, b) for (a, b) in shared if inside[a] and inside[b])
    weak = boundary_weakness(rng, ptype, mos.craton, in_pairs)
    k = int(rng.integers(*p.n_blocks, endpoint=True))
    c_sc = unit((centers[:n] * cell_area[:n, None]).sum(0))
    rho = np.arccos(1.0 - cell_area[:n].sum() / (2 * np.pi * R_KM ** 2))
    bseeds = sector_seeds(rng, members, centers, c_sc, k, cratons, rho)
    size_w = rng.lognormal(0.0, p.block_size_sigma, len(bseeds))
    owner = breakup_blocks(members, bseeds, nbrs, weak, cell_area, size_w, centers, c_sc, p.tangential_cost)
    while True:                                       # 太小的块并入共享边界最长的相邻块
        areas = {b: sum(cell_area[i] for i in owner if owner[i] == b) for b in set(owner.values())}
        small = sorted(b for b, a in areas.items() if a < p.min_block_km2)
        if not small or len(areas) <= 2:
            break
        b = small[0]
        nb_len = {}
        for i in [i for i in owner if owner[i] == b]:
            for j in nbrs[i]:
                if j in owner and owner[j] != b:
                    nb_len[owner[j]] = nb_len.get(owner[j], 0) + _sh(shared, i, j)
        tgt = max(sorted(nb_len), key=nb_len.get)
        for i in owner:
            if owner[i] == b:
                owner[i] = tgt
    # 嵌套的陆块并入外面那块：与某个邻块的共享边界超过自身周长的 nested_share，裂开后只能沿"海湾"的两壁滑出去，
    # 两壁一直挨着（世界种子 3：一块 1100 万 km² 的陆块嵌在邻块凹处，100 Myr 后侧壁仍只隔 10 km）。相当于这条裂谷夭折
    while True:
        bset = sorted(set(owner.values()))
        perim_b = {b: 0 for b in bset}
        share_b = {}
        for (i, j), cnt in shared.items():
            bi = owner.get(i) if i in owner else None
            bj = owner.get(j) if j in owner else None
            if bi is not None:
                if bj != bi:
                    perim_b[bi] += cnt
            if bj is not None:
                if bi != bj:
                    perim_b[bj] += cnt
            if bi is not None and bj is not None and bi != bj:
                key = (min(bi, bj), max(bi, bj))
                share_b[key] = share_b.get(key, 0) + cnt
        nested = None
        for (x, y), cnt in sorted(share_b.items()):
            ax_, ay_ = sum(cell_area[i] for i in owner if owner[i] == x), sum(cell_area[i] for i in owner if owner[i] == y)
            small_, big_ = (x, y) if ax_ <= ay_ else (y, x)
            if cnt > p.nested_share * max(perim_b[small_], 1):
                nested = (small_, big_)
                break
        if nested is None or len(bset) <= 2:
            break
        for i in owner:
            if owner[i] == nested[0]:
                owner[i] = nested[1]
    block_ids = {b: kk + 1 for kk, b in enumerate(sorted(set(owner.values())))}
    owner = {i: block_ids[b] for i, b in owner.items()}
    nb = len(block_ids)
    # 漂移与能张开的分界
    cont = mos.cell >= 0
    blk0 = np.zeros(dm.n, np.int64)
    blk0[cont] = [owner[i] for i in mos.cell[cont]]
    omega, cuts, drift = choose_drift(dm, blk0, nb, c_sc, p, np.random.default_rng([seed, 4]))
    blk = apply_cuts(dm, blk0, cuts)
    block_of_prov = {}
    for i in range(n):
        m = mos.cell == i
        block_of_prov[i] = int(np.bincount(blk[m]).argmax()) if m.any() else owner[i]
    # 夭折裂谷：陆块内部、至少一侧挨着陆缘的强薄弱带
    bop = block_of_prov
    edge_cells = {i for i in members if any((not inside[j]) or bop.get(j) != bop[i] for j in nbrs[i])}
    cands = sorted([(weak[(a, b)] + 0.3 * rng.random(), a, b) for (a, b) in in_pairs
                    if bop[a] == bop[b] and (a in edge_cells or b in edge_cells)], reverse=True)
    used, failed = set(), []
    n_failed = int(rng.integers(*p.n_failed_rifts, endpoint=True))
    for _, a, b in cands:
        if len(failed) >= n_failed:
            break
        if a in used or b in used:
            continue
        used |= {a, b}
        failed.append((a, b))
    oseeds = unit(rng.normal(size=(p.n_ocean_plates, 3)))
    o_omega = {q: unit(rng.normal(size=3)) * rng.uniform(0.03, 0.12) * DEG for q in range(p.n_ocean_plates)}
    return Design(seed, dm.level, mos, provinces, weak, in_pairs, owner, block_of_prov, cuts, omega, drift, failed,
                  oseeds, o_omega, nb, len(cratons), dm.h_km)


def realize(mesh: Mesh, ds: Design, p: PrehistoryParams):
    """按设计在目标网格上生成 t_start 时刻的地壳与板块。"""
    seeds, along, scale, aspect = ds.mos.gen
    prov = ds.mos.lut[anisotropic_cells(mesh, seeds, along, scale, aspect)]
    cont = prov >= 0
    c = Crust(mesh.n)
    c.ctype[cont] = CONT
    c.prov[cont] = prov[cont] + 1
    c.t_form[cont] = [ds.provinces[i]["age_ma"] for i in prov[cont]]
    blk0 = np.zeros(mesh.n, np.int64)
    blk0[cont] = [ds.owner[i] for i in prov[cont]]
    blk = apply_cuts(mesh, blk0, ds.cuts)
    c.origin[cont] = blk[cont]
    T = np.zeros(mesh.n)
    T[cont] = [ds.provinces[i]["T_km"] for i in prov[cont]]
    e = mesh.edges
    ce = e[cont[e[:, 0]] & cont[e[:, 1]]]
    n_iter = max(1, int(round(4 * (ds.h_km / mesh.h_km) ** 2)))   # 省份之间的厚度过渡约 150–250 km（与网格无关）
    for _ in range(n_iter):
        s = np.bincount(ce[:, 0], T[ce[:, 1]], mesh.n) + np.bincount(ce[:, 1], T[ce[:, 0]], mesh.n)
        k_ = np.bincount(ce[:, 0], minlength=mesh.n) + np.bincount(ce[:, 1], minlength=mesh.n)
        T = np.where(cont & (k_ > 0), 0.5 * T + 0.5 * s / np.maximum(k_, 1), T)
    for c0, ax_, R, asp, dT in ds.mos.basins:   # 克拉通内盆地：椭圆碟状沉陷（地壳减薄后热沉降的代理）
        pu = mesh.xyz @ ax_ * R_KM
        pv = mesh.xyz @ np.cross(c0, ax_) * R_KM
        r2 = (pu / (R * np.sqrt(asp))) ** 2 + (pv / (R / np.sqrt(asp))) ** 2
        sel = cont & (r2 < 1.0) & (mesh.xyz @ c0 > 0.5)
        T[sel] -= dT * (1.0 - r2[sel])
    d_out, _ = cKDTree(mesh.xyz[~cont]).query(mesh.xyz[cont], workers=KD_WORKERS)
    T[cont] *= 0.63 + 0.37 * smoothstep(km_from_chord(d_out) / p.margin_km)      # 超大陆外缘陆缘
    rifted = np.zeros(mesh.n, bool)                    # 裂谷陆缘：各陆块之间的分界附近减薄
    for b in range(1, ds.nb + 1):
        mb = blk == b
        if not mb.any():
            continue
        d = _boundary_distance(mesh, mb, cont & ~mb, np.flatnonzero(mb), p.rift_margin_km)
        ok = np.isfinite(d)
        idx = np.flatnonzero(mb)[ok]
        T[idx] *= 0.35 + 0.65 * smoothstep(d[ok] / p.rift_margin_km)
        rifted[idx] = True
    c.oro[rifted] = ORO_RIFT
    c.oro_t[rifted] = float(p.t_start)
    failed = []
    for a, b in ds.failed:                             # 夭折裂谷
        ma, mb_ = prov == a, prov == b
        q = np.flatnonzero(ma | mb_)
        d = _boundary_distance(mesh, ma, mb_, q, p.failed_rift_km)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        T[q[ok]] *= 0.78 + 0.22 * smoothstep(d[ok] / p.failed_rift_km)
        c.oro[q[ok]] = ORO_RIFT
        c.oro_t[q[ok]] = float(p.t_start)
        lon, lat = xyz_to_lonlat(unit(mesh.xyz[q[ok]].mean(0)))
        nhit = int(((ma[e[:, 0]] & mb_[e[:, 1]]) | (mb_[e[:, 0]] & ma[e[:, 1]])).sum())
        failed.append({"provinces": [a + 1, b + 1], "block": ds.block_of_prov[a], "lat": float(lat), "lon": float(lon),
                       "length_km": float(nhit * mesh.h_km)})
    c.T[cont] = T[cont]
    d_c, _ = cKDTree(mesh.xyz[cont]).query(mesh.xyz[~cont], workers=KD_WORKERS)   # 洋壳：离超大陆越远越年轻
    c.age[~cont] = 150.0 - 110.0 * smoothstep(km_from_chord(d_c) / 6000.0)
    c.t_form[~cont] = p.t_start + c.age[~cont]
    # 板块：每个陆块连同周围 halo_km 内的洋壳一块板块；其余洋壳分成 n_ocean_plates 块
    plate = np.zeros(mesh.n, np.int32)
    plate[cont] = blk[cont]
    oc = np.flatnonzero(~cont)
    d_b, k_b = cKDTree(mesh.xyz[cont]).query(mesh.xyz[oc], workers=KD_WORKERS)
    halo = km_from_chord(d_b) < p.halo_km
    plate[oc[halo]] = blk[np.flatnonzero(cont)[k_b[halo]]]
    far = oc[~halo]
    plate[far] = ds.nb + 1 + np.argmax(mesh.xyz[far] @ ds.oseeds.T, 1)
    c.plate = plate
    return c, blk, failed


# ---- 主流程 ---------------------------------------------------------------------------------------
def build_initial_world(mesh: Mesh, seed: int, params: PrehistoryParams | None = None, snap_every: int = 20,
                        work_dir: Path | None = None) -> InitialWorld:
    p = params or PrehistoryParams()
    dm = mesh if mesh.level == p.design_level else load_mesh(p.design_level)
    ds = design_prehistory(dm, seed, p)
    c, blk, failed = realize(mesh, ds, p)
    nb = ds.nb
    omega0 = dict(ds.omega)
    for k, w in ds.o_omega.items():
        omega0[nb + 1 + k] = w
    script = StageScript(p.t_start)
    for q, w in omega0.items():
        script.add_plate(q, 0, f"block {q}" if q <= nb else f"ocean {q}")
        script.set_absolute(q, w, p.t_start, p.t_end if p.t_end > 0 else 0)
        if p.t_end > 0:
            script.set_absolute(q, w, p.t_end, 0)        # 旋转文件要求写到 0 Ma；t_end 之后不使用
    work = Path(work_dir or (CACHE / "prehistory"))
    rot = script.write_rot(work / f"prehistory_L{mesh.level}_seed{seed}.rot")
    kin = RotationFileKinematics(rot)
    events = [{"t": float(p.t_start), "kind": "supercontinent_breakup", "n_blocks": nb,
               "n_provinces": len(ds.provinces), "n_cratons": ds.n_cratons, "failed_rifts": failed,
               "drift": ds.drift, "design_level": ds.level}]
    eng = Engine(mesh, c, Params(), float(p.t_start), omega0)
    snaps = [eng.snapshot()]
    t = p.t_start
    while t > p.t_end:
        dR, om = kin.step_rotations(eng.plates, t)
        committed = eng.step(dR, om)
        t -= 1
        if committed and (p.t_start - t) % snap_every == 0:
            snaps.append(eng.snapshot())
    view = eng.view()
    provinces = {i: {**r, "block": ds.block_of_prov[i]} for i, r in ds.provinces.items()}
    blocks = []
    for b in range(1, nb + 1):
        mb = (view.origin == b) & (view.ctype == CONT)
        blocks.append({"id": b, "area_km2": float(mesh.area_km2[mb].sum()),
                       "n_provinces": int(sum(1 for i in provinces if provinces[i]["block"] == b))})
    failed_pairs = set(ds.failed)
    boundaries = [{"a": a + 1, "b": b + 1, "weakness": float(ds.weak[(a, b)]),
                   "block_boundary": ds.block_of_prov[a] != ds.block_of_prov[b],
                   "failed_rift": (a, b) in failed_pairs} for (a, b) in ds.in_pairs]
    return InitialWorld(level=mesh.level, seed=seed, t=float(p.t_end), crust=view,
                        omega={q: eng.omega[q].copy() for q in eng.plates},
                        provinces=[provinces[i] for i in sorted(provinces)], blocks=blocks,
                        events=events + eng.events, boundaries=boundaries, snapshots=snaps, rot_path=str(rot),
                        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items()})
