"""前史（阶段 1.2b）：生成 300 Ma 的初始大陆。

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
   把超大陆分成 5–7 块时，让分界线优先走薄弱的缝合带、绕开克拉通（地球上的裂谷也这样，例如东非裂谷绕过坦桑尼亚克拉通）。
   没被用作分界的强薄弱带，挑几条从陆缘伸进内陆的减薄成"夭折裂谷"（例如贝努埃槽），以后会成为深入内陆的海湾、低地。
   新裂开的陆缘按距离减薄（陆架、陆坡）。
3. 漂移（过程，引擎 + 剧本 + pyGPlates）：t_start -> t_end（默认 400 -> 300 Ma），各陆块离开超大陆中心散开，
   之间生出新洋壳，外缘向外推进处发生俯冲、长出岛弧。

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
from .geometry import DEG, R_KM, chord_from_km, km_from_chord, smoothstep, unit, xyz_to_lonlat
from .kinematics import RotationFileKinematics, StageScript
from .mesh import CACHE, KD_WORKERS, Mesh

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
    n_blocks: tuple = (5, 7)
    n_basins: tuple = (3, 5)
    n_failed_rifts: tuple = (2, 4)
    n_ocean_plates: int = 3
    halo_km: float = 700.0               # 陆块周围这一范围内的洋壳与陆块同属一块板块
    margin_km: float = 250.0             # 超大陆外缘陆缘减薄宽度
    rift_margin_km: float = 200.0        # 裂谷陆缘减薄宽度
    failed_rift_km: float = 120.0        # 夭折裂谷减薄带半宽
    drift_rate: tuple = (0.25, 0.45)     # 陆块离开超大陆中心的角速度（°/Myr），约 2.8–5 cm/yr（0.15–0.32 时到 300 Ma 只隔着窄海道）
    min_block_km2: float = 4.0e6


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
    cell: np.ndarray          # 每个固定节点所属省份（-1 = 洋）
    ptype: dict               # 省份 -> 类型 1–4
    age: dict                 # 省份 -> 年龄（Ma）
    craton: dict              # 省份 -> 所属克拉通（环带围绕的那一个）
    ring: dict                # 省份 -> 环序号（0 = 克拉通本身）
    craton_cells: list
    basins: list              # 克拉通内盆地：(中心, 半径 km, 最大减薄 km)


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


def anisotropic_cells(mesh: Mesh, seeds, along, scale_km, aspect, chunk: int = 20000) -> np.ndarray:
    """各向异性球面 Voronoi：每个种子有自己的大小、伸长度和长轴方向；距离在种子处的对数映射坐标里按椭圆度量。
    之后每个省份只保留最大的连通块，零碎部分并给相邻省份。"""
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
    # 连通化：每个省份只保留面积最大的连通块，其余节点由相邻的保留块逐圈吸收
    e = mesh.edges
    for _ in range(5):
        lab = graph_components(mesh.n, e[cell[e[:, 0]] == cell[e[:, 1]]])
        n_lab = lab.max() + 1
        size = np.bincount(lab, weights=mesh.area_km2, minlength=n_lab)
        cl_of_lab = np.zeros(n_lab, np.int64)
        cl_of_lab[lab] = cell
        best = {}
        for l_ in np.argsort(size)[::-1]:
            best.setdefault(int(cl_of_lab[l_]), int(l_))
        keep = np.zeros(n_lab, bool)
        keep[list(best.values())] = True
        bad = ~keep[lab]
        if not bad.any():
            break
        for _ in range(200):
            m1 = bad[e[:, 0]] & ~bad[e[:, 1]]
            m2 = bad[e[:, 1]] & ~bad[e[:, 0]]
            if not (m1.any() or m2.any()):
                break
            cell[e[m1, 0]] = cell[e[m1, 1]]
            bad[e[m1, 0]] = False
            cell[e[m2, 1]] = cell[e[m2, 0]]
            bad[e[m2, 1]] = False
    _, cell = np.unique(cell, return_inverse=True)
    return cell.astype(np.int64)


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
    n = int(cell.max()) + 1
    cell_area = np.bincount(cell, weights=mesh.area_km2, minlength=n)
    shared, nbrs, perim = cell_graph(mesh, cell, n)
    # 超大陆：从中心所在省份长出
    start = int(cell[np.argmax(mesh.xyz @ C)])
    inside = np.zeros(n, bool)
    inside[start] = True
    area = cell_area[start]
    while area < target:
        front = {j for i in np.flatnonzero(inside) for j in nbrs[i] if not inside[j]}
        best, score = None, -np.inf
        for j in front:
            s = sum(_sh(shared, i, j) for i in nbrs[j] if inside[i]) / max(perim[j], 1) + 0.4 * rng.random()
            if s > score:
                best, score = j, s
        inside[best] = True
        area += cell_area[best]
    out = np.flatnonzero(~inside)
    loc = {c_: k_ for k_, c_ in enumerate(out)}
    oe = np.asarray([(loc[a], loc[b]) for (a, b) in shared if not inside[a] and not inside[b]], np.int64).reshape(-1, 2)
    lab = graph_components(len(out), oe)
    big = np.bincount(lab, weights=cell_area[out]).argmax()
    inside[out[lab != big]] = True                       # 被包围的洋区并入
    members = set(np.flatnonzero(inside).tolist())
    # 克拉通 = 含克拉通种子的省份
    craton_cells = sorted({int(cell[np.argmax(mesh.xyz @ x)]) for x in crat_seeds} & members)
    hops, src = _hops(nbrs, members, craton_cells)
    periphery = {i for i in members if any(not inside[j] for j in nbrs[i])}
    ptype, age, crat, ring = {}, {}, {}, {}
    for i in members:
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
    out_cell = np.array([remap.get(int(c_), -1) for c_ in cell], np.int64)
    basins = []
    hosts = [i for i in order if ptype[i] in (1, 2)]
    for i in rng.choice(hosts, size=min(int(rng.integers(*p.n_basins, endpoint=True)), len(hosts)), replace=False):
        sel = np.flatnonzero(cell == i)
        c0 = mesh.xyz[rng.choice(sel)]
        ax_ = unit(np.cross(c0, rng.normal(size=3)))
        basins.append((c0, ax_, float(rng.uniform(250, 500)), float(rng.uniform(1.0, 2.5)), float(rng.uniform(3.0, 5.0))))
    return Mosaic(out_cell, {remap[i]: ptype[i] for i in order}, {remap[i]: age[i] for i in order},
                  {remap[i]: crat[i] for i in order}, {remap[i]: ring[i] for i in order},
                  [remap[i] for i in craton_cells], basins)


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
        d = np.min(np.arccos(np.clip(centers[cand] @ centers[chosen].T, -1, 1)), axis=1)
        top = cand[np.argsort(d)[::-1][:3]]
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


def breakup_blocks(rng, members, seeds, nbrs, shared, weak, speed):
    """多源最短路分块：越过薄弱边界代价高，生长前锋在薄弱带停住，所以分界线优先落在薄弱带上。"""
    owner = {}
    heap = [(0.0, int(s), b) for b, s in enumerate(seeds)]
    heapq.heapify(heap)
    best = {int(s): 0.0 for s in seeds}
    while heap:
        d, i, b = heapq.heappop(heap)
        if i in owner:
            continue
        owner[i] = b
        for j in nbrs[i]:
            if j not in members or j in owner:
                continue
            key = (i, j) if i < j else (j, i)
            nd = d + (0.3 + 3.0 * weak[key]) / speed[b]
            if nd < best.get(j, np.inf):
                best[j] = nd
                heapq.heappush(heap, (nd, j, b))
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


# ---- 主流程 ---------------------------------------------------------------------------------------
def build_initial_world(mesh: Mesh, seed: int, params: PrehistoryParams | None = None, snap_every: int = 20,
                        work_dir: Path | None = None) -> InitialWorld:
    p = params or PrehistoryParams()
    rng = np.random.default_rng(seed)
    events = []

    # 1. 省份拼贴与老超大陆（克拉通 + 同心增生环带）
    mos = belt_mosaic(mesh, rng, p)
    n = int(mos.cell.max()) + 1
    OC = n                                            # 洋区在省份图里的编号
    cell = np.where(mos.cell >= 0, mos.cell, OC)
    cell_area = np.bincount(cell, weights=mesh.area_km2, minlength=n + 1)
    shared, nbrs, perim = cell_graph(mesh, cell, n + 1)
    inside = np.ones(n + 1, bool)
    inside[OC] = False
    members = set(range(n))
    centers = np.zeros((n + 1, 3))
    np.add.at(centers, cell, mesh.xyz * mesh.area_km2[:, None])
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

    # 2. 薄弱带、分块、夭折裂谷
    in_pairs = [(a, b) for (a, b) in shared if inside[a] and inside[b]]
    weak = boundary_weakness(rng, ptype, mos.craton, in_pairs)
    k = int(rng.integers(*p.n_blocks, endpoint=True))
    bseeds = _farthest(rng, np.array(cratons), centers, k)
    speed = rng.lognormal(0.0, 0.3, len(bseeds))
    owner = breakup_blocks(rng, members, bseeds, nbrs, shared, weak, speed)
    # 太小的块并入共享边界最长的相邻块
    while True:
        areas = {b: sum(cell_area[i] for i in owner if owner[i] == b) for b in set(owner.values())}
        small = [b for b, a in areas.items() if a < p.min_block_km2]
        if not small or len(areas) <= 2:
            break
        b = small[0]
        nb_len = {}
        for i in [i for i in owner if owner[i] == b]:
            for j in nbrs[i]:
                if j in owner and owner[j] != b:
                    nb_len[owner[j]] = nb_len.get(owner[j], 0) + _sh(shared, i, j)
        tgt = max(nb_len, key=nb_len.get)
        for i in owner:
            if owner[i] == b:
                owner[i] = tgt
    block_ids = {b: kk + 1 for kk, b in enumerate(sorted(set(owner.values())))}
    owner = {i: block_ids[b] for i, b in owner.items()}
    for i in provinces:
        provinces[i]["block"] = owner[i]

    # 3. 300 Ma 之前的地壳：固定网格
    c = Crust(mesh.n)
    cont = inside[cell]
    c.ctype[cont] = CONT
    c.prov[cont] = cell[cont] + 1
    c.origin[cont] = [owner[i] for i in cell[cont]]
    c.t_form[cont] = [provinces[i]["age_ma"] for i in cell[cont]]
    T = np.zeros(mesh.n)
    T[cont] = [provinces[i]["T_km"] for i in cell[cont]]
    e = mesh.edges
    ce = e[cont[e[:, 0]] & cont[e[:, 1]]]
    for _ in range(4):        # 省份之间的厚度过渡（约 150–250 km）
        s = np.bincount(ce[:, 0], T[ce[:, 1]], mesh.n) + np.bincount(ce[:, 1], T[ce[:, 0]], mesh.n)
        k_ = np.bincount(ce[:, 0], minlength=mesh.n) + np.bincount(ce[:, 1], minlength=mesh.n)
        T = np.where(cont & (k_ > 0), 0.5 * T + 0.5 * s / np.maximum(k_, 1), T)
    for c0, ax_, R, asp, dT in mos.basins:      # 克拉通内盆地：椭圆碟状沉陷（地壳减薄后热沉降的代理）
        pu = mesh.xyz @ ax_ * R_KM
        pv = mesh.xyz @ np.cross(c0, ax_) * R_KM
        r2 = (pu / (R * np.sqrt(asp))) ** 2 + (pv / (R / np.sqrt(asp))) ** 2
        sel = cont & (r2 < 1.0) & (mesh.xyz @ c0 > 0.5)
        T[sel] -= dT * (1.0 - r2[sel])
    d_out, _ = cKDTree(mesh.xyz[~cont]).query(mesh.xyz[cont], workers=KD_WORKERS)
    T[cont] *= 0.63 + 0.37 * smoothstep(km_from_chord(d_out) / p.margin_km)      # 超大陆外缘陆缘
    blk = np.zeros(mesh.n, np.int32)
    blk[cont] = c.origin[cont]
    # 裂谷陆缘：各陆块之间的分界附近减薄
    rifted = np.zeros(mesh.n, bool)
    for b in sorted(set(owner.values())):
        mb = blk == b
        d = _boundary_distance(mesh, mb, cont & ~mb, np.flatnonzero(mb), p.rift_margin_km)
        ok = np.isfinite(d)
        idx = np.flatnonzero(mb)[ok]
        T[idx] *= 0.35 + 0.65 * smoothstep(d[ok] / p.rift_margin_km)
        rifted[idx] = True
    c.oro[rifted] = ORO_RIFT
    c.oro_t[rifted] = float(p.t_start)
    # 夭折裂谷：陆块内部、至少一侧挨着陆缘的强薄弱带
    edge_cells = {i for i in members if any((not inside[j]) or owner.get(j) != owner[i] for j in nbrs[i])}
    cand = sorted([(weak[(a, b)] + 0.3 * rng.random(), a, b) for (a, b) in in_pairs
                   if owner[a] == owner[b] and (a in edge_cells or b in edge_cells)], reverse=True)
    used, failed = set(), []
    for _, a, b in cand:
        if len(failed) >= int(rng.integers(*p.n_failed_rifts, endpoint=True)):
            break
        if a in used or b in used:
            continue
        ma, mb = cell == a, cell == b
        q = np.flatnonzero(ma | mb)
        d = _boundary_distance(mesh, ma, mb, q, p.failed_rift_km)
        ok = np.isfinite(d)
        T[q[ok]] *= 0.78 + 0.22 * smoothstep(d[ok] / p.failed_rift_km)
        c.oro[q[ok]] = ORO_RIFT
        c.oro_t[q[ok]] = float(p.t_start)
        used |= {a, b}
        mid = unit(mesh.xyz[q[ok]].mean(0))
        lon, lat = xyz_to_lonlat(mid)
        failed.append({"provinces": [a + 1, b + 1], "block": owner[a], "lat": float(lat), "lon": float(lon),
                       "length_km": float(_sh(shared, a, b) * mesh.h_km)})
    c.T[cont] = T[cont]
    # 洋壳：离超大陆越远越年轻（外大洋的洋脊在远处）
    d_c, _ = cKDTree(mesh.xyz[cont]).query(mesh.xyz[~cont], workers=KD_WORKERS)
    c.age[~cont] = 150.0 - 110.0 * smoothstep(km_from_chord(d_c) / 6000.0)
    c.t_form[~cont] = p.t_start + c.age[~cont]

    # 板块：每个陆块连同周围 halo_km 内的洋壳一块板块；其余洋壳分成 n_ocean_plates 块
    nb = len(set(owner.values()))
    plate = np.zeros(mesh.n, np.int32)
    plate[cont] = blk[cont]
    oc = np.flatnonzero(~cont)
    d_b, k_b = cKDTree(mesh.xyz[cont]).query(mesh.xyz[oc], workers=KD_WORKERS)
    halo = km_from_chord(d_b) < p.halo_km
    plate[oc[halo]] = blk[np.flatnonzero(cont)[k_b[halo]]]
    far = oc[~halo]
    oseeds = unit(rng.normal(size=(p.n_ocean_plates, 3)))
    plate[far] = nb + 1 + np.argmax(mesh.xyz[far] @ oseeds.T, 1)
    c.plate = plate

    # 4. 剧本：各陆块离开超大陆中心，洋壳板块缓慢运动
    w_area = mesh.area_km2[cont]
    c_sc = unit((mesh.xyz[cont] * w_area[:, None]).sum(0))
    script = StageScript(p.t_start)
    omega0 = {}
    for b in range(1, nb + 1):
        mb = blk == b
        cb = unit((mesh.xyz[mb] * mesh.area_km2[mb, None]).sum(0))
        ax = np.cross(c_sc, cb)
        rate = rng.uniform(*p.drift_rate)
        w = (unit(ax) * rate if np.linalg.norm(ax) > 0.15 else unit(rng.normal(size=3)) * 0.05) * DEG
        w = w + cb * rng.uniform(-0.12, 0.12) * DEG            # 绕自身中心的少量自转
        omega0[b] = w
    for q in range(nb + 1, nb + 1 + p.n_ocean_plates):
        omega0[q] = unit(rng.normal(size=3)) * rng.uniform(0.03, 0.12) * DEG
    for q, w in omega0.items():
        script.add_plate(q, 0, f"block {q}" if q <= nb else f"ocean {q}")
        script.set_absolute(q, w, p.t_start, p.t_end if p.t_end > 0 else 0)
        if p.t_end > 0:
            script.set_absolute(q, w, p.t_end, 0)        # 旋转文件要求写到 0 Ma；t_end 之后不使用
    work = Path(work_dir or (CACHE / "prehistory"))
    rot = script.write_rot(work / f"prehistory_L{mesh.level}_seed{seed}.rot")
    kin = RotationFileKinematics(rot)

    events.append({"t": float(p.t_start), "kind": "supercontinent_breakup", "n_blocks": nb,
                   "n_provinces": len(members), "n_cratons": len(cratons), "failed_rifts": failed})
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
    blocks = []
    for b in range(1, nb + 1):
        mb = (view.origin == b) & (view.ctype == CONT)
        blocks.append({"id": b, "area_km2": float(mesh.area_km2[mb].sum()),
                       "n_provinces": int(sum(1 for i in provinces if provinces[i]["block"] == b))})
    failed_pairs = {tuple(sorted((f["provinces"][0] - 1, f["provinces"][1] - 1))) for f in failed}
    boundaries = [{"a": a + 1, "b": b + 1, "weakness": float(weak[(a, b)]), "block_boundary": owner[a] != owner[b],
                   "failed_rift": (a, b) in failed_pairs, "length_km": float(shared[(a, b)] * mesh.h_km)}
                  for (a, b) in in_pairs]
    return InitialWorld(level=mesh.level, seed=seed, t=float(p.t_end), crust=view,
                        omega={q: eng.omega[q].copy() for q in eng.plates},
                        provinces=[provinces[i] for i in sorted(provinces)], blocks=blocks,
                        events=events + eng.events, boundaries=boundaries, snapshots=snaps, rot_path=str(rot),
                        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items()})
