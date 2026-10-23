"""剧本生成器 v1（阶段 1.2c）：从 300 Ma 初始大陆（prehistory.py）出发，按情景族写 300 -> 0 Ma 的板块运动剧本。

情景族（tasks/stage1_tectonics.md）：300 Ma 分散的陆块 -> 汇聚成相连的超大陆、留下一个残留洋盆（主内海） -> 局部裂解；
一块陆块漂到极地成为极地大陆；最多一块次大陆留在远洋另一侧。

规划都只做刚体旋转（不跑地壳规则），在"锚定陆块"（最大的主陆块）的 300 Ma 参照系里进行：
1. 分角色：极地大陆、次大陆（可选）、主大陆各块；锚定陆块不动。
2. 依次拼合：每块从它所在一侧向已拼好的大陆滑过去，接触后再楔入 penetration_km 为目标位置；
   从 300 Ma 位置到目标位置用一个固定欧拉极匀速旋转，到 dock 时刻抵达，之后保留 residual 比例的汇聚 residual_myr，
   然后锁定。检查途中不会提前撞上别的陆块、速度不超过 max_rate_deg；直接路线会碰撞时换成分段路线（背离邻居、绕行、接近，
   终点不变）。
3. 主内海：第二大的主陆块最后规划、最后拼上，两段式拼合圈出残留洋盆；先用网格上的快速代理筛选，再用阶段 0 的
   A 层主内海代码（mythic.metrics.hard）在 1024x512 上核对。完整模拟后按 0 Ma 的 A 层闭环修正远端海峡宽度（run_scenario_loop）。
4. 整体定位：把拼好的主大陆整体转到目标纬度（北端、南端、北极圈内面积、主内海纬度），锚定陆块在 300 Myr 内匀速完成这个旋转。
5. 极地大陆、次大陆：相对锚定陆块分段匀速运动，0 Ma 抵达目标位置（整体定位时已让它们要走的路尽量短）；全程不碰撞。
6. 局部裂解（按概率抽样）：沿拼合缝合带"剪刀式"张开（一端仍相连，形成狭长裂谷海），或从陆块边缘裂出微大陆。都避开主内海。
最后写成 PLATES4 旋转文件，引擎完整模拟；0 Ma 用 A 层、B1 代码评估。

method_class：运动 = procedural_constrained（剧本设定，速度受地球范围约束）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .crust import CONT, Crust, Params
from .engine import Engine, graph_components
from .geometry import DEG, R_KM, axis_angle, chord_from_km, km_from_chord, lonlat_to_xyz, rotmat, unit, xyz_to_lonlat
from .kinematics import RotationFileKinematics, StageScript
from .mesh import CACHE, KD_WORKERS, Mesh, Raster
from .prehistory import InitialWorld


@dataclass
class ScenarioParams:
    t0: int = 300
    dock_window: tuple = (240, 125)        # 各主陆块拼上的时刻（Ma）
    # 拼合按陆地（300 Ma 海面以上）接触判定，接触后再楔入 penetration_km 锁定。按陆壳接触时（1.2c 初版），两侧约 200 km 宽、
    # 沉在水下的被动陆缘先碰上，楔入 150 km 后陆地之间仍隔约 250 km 的薄陆壳，碰撞增厚抬不出海面，0 Ma 时拼上的陆块之间
    # 留下一圈窄海峡，主大陆只占陆地 21–45%（新前史 4 个世界中 3 个）。地球上大陆碰撞要挤到两侧厚陆壳相撞
    # （印度初次接触后又向北推进了上千公里），被动陆缘俯冲或卷进造山带
    dock_on: str = "land"
    penetration_km: float = 60.0
    margin_km: float = 200.0               # 前史裂谷陆缘（水下减薄陆壳）的宽度，见 prehistory.rift_margin_km
    residual: float = 0.10                 # 锁定后保留的汇聚比例（青藏式持续楔入）
    residual_myr: float = 20.0             # （1.2c 初版 250 km + 15% × 25 Myr，碰撞带长出好几个青藏级高原）
    main_share_max: float = 0.85           # 主大陆占陆地比例上限（简报 §3）；超过就必须设次大陆
    max_rate_deg: float = 0.75             # 相对运动最大角速度（°/Myr；赤道约 8.3 cm/yr）
    polar_area_km2: tuple = (5.0e6, 1.6e7)
    secondary_area_km2: tuple = (8.0e6, 2.6e7)
    p_secondary: float = 0.6
    max_block_km2: float = 1.1e8           # 前史种子筛选：最大陆块（编组后）超过这个就不用（1.2b 局限）
    min_groups: int = 4                    # 前史种子筛选：编组后至少这么多个陆块组
    inner_sea_area_km2: tuple = (1.0e6, 5.0e6)
    # 内洋：主大陆圈住的水体（规划网格上按 150 km 海峡与外洋隔开）超过这个面积就拒绝。A 层把最大的内部水体当主内海
    # （要求 1–5 Mkm²），圈出更大的水体时主内海必然不合格（1.2c 第三步世界 2 圈出 2750 万 km²）；取上限的 1.6 倍，给规划
    # 代理的误差留余地
    inner_ocean_km2: float = 8.0e6
    # 主内海那块的出发位置当其他陆块障碍的时长（见 Planner.plan_assembly）。1.2c 第三步试过 20 Myr：世界 1、2、4、8 主大陆占比
    # 69/43/42/75%（不加 80/43/44/75%），主内海那块照样途中碰撞（碰撞推迟到 268–280 Ma），没有改善，默认不用
    inner_obstacle_myr: float = 0.0
    breakup_window: tuple = (110, 25)
    n_breakup: tuple = (1, 3)
    target_north: tuple = (62.0, 72.0)     # 主大陆北端（简报 §3：65–70°N）
    target_south: tuple = (-45.0, -30.0)   # 主大陆南端（简报：约 40°S，软限制 50°S）
    inner_sea_lat: tuple = (30.0, 42.0)    # 主内海纬度（地中海气候）
    polar_lat: float = -82.0


OWN_OFFSETS = (-40, -20, 0, 20, 40)        # 主内海那块：以它自己所在方位为基准的接近方向（度），见 Planner._plan_dock


@dataclass
class Dock:
    """一块主陆块的拼合运动（锚定参照系）。segs 按时间先后：(起始时刻, 结束时刻, 欧拉极, 转角 rad)，每段绕固定欧拉极匀速转动。
    一般的拼合只有一段；主内海那次是两段：先接近到第一处接触，再绕接触点转动到另一端只剩窄海峡。"""
    block: int
    t_dock: float
    segs: list = field(default_factory=list)
    t_contact: float | None = None         # 计划中的第一次接触时刻（途中检查到此为止）；None = t_dock + 15
    inner_sea: bool = False
    res_seg: tuple | None = None           # 锁定后残余汇聚沿用的那一段；None = 最后一段（分段路线沿用原直接路线的，终点不变）

    def residual_seg(self) -> tuple:
        return self.res_seg if self.res_seg is not None else self.segs[-1]

    @property
    def angle(self) -> float:
        return float(sum(abs(s[3]) for s in self.segs))


def dock_pose(d: "Dock | None", t: float, residual: float, residual_myr: float) -> np.ndarray:
    """锚定参照系中 t 时刻的旋转：已完成的段全转，进行中的段按比例；dock 之后按最后一段的速率保留 residual 的汇聚。"""
    if d is None or not d.segs:
        return np.eye(3)
    R = np.eye(3)
    for ta, tb, ax, ang in d.segs:
        if t >= ta:
            break
        f = min(1.0, (ta - t) / (ta - tb))
        R = rotmat(ax, ang * f) @ R
    if t < d.t_dock:
        ta, tb, ax, ang = d.residual_seg()
        R = rotmat(ax, residual * ang / (ta - tb) * min(d.t_dock - t, residual_myr)) @ R
    return R


@dataclass
class Plan:
    seed: int
    anchor: int
    roles: dict
    docks: dict
    G_axis: np.ndarray
    G_angle: float
    moves: dict                            # 极地大陆、次大陆、孤立陆块：分段 [(起, 止, 欧拉极, 转角)]（锚定参照系，相对锚定陆块）
    breakups: list
    inner_sea: dict
    script: StageScript | None = None
    notes: list = field(default_factory=list)
    score: float = 0.0
    placement: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    attempt: int = 0                       # 第几次随机抽样（Planner.plan），闭环修正时只重做这一次


# ---- 几何小工具 -------------------------------------------------------------------------------
def move_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3)
    return rotmat(ax / s, np.arctan2(s, float(a @ b)))


def _centroid(X, w=None):
    return unit((X * (1.0 if w is None else w[:, None])).sum(0))


def _ang(a, b):
    return float(np.arccos(np.clip(a @ b, -1, 1)))


class Contact:
    """已拼好部分的陆壳点集：判断一组点是否与之接触（距离 < 0.6 网格间距）。"""

    def __init__(self, pts: np.ndarray, h_km: float):
        self.tree = cKDTree(pts) if len(pts) else None
        self.r = chord_from_km(0.6 * h_km)

    # 查询点都只有几百到几千个：单线程（多线程查询每次都要起线程，规划中几十万次查询的线程开销占了总时间的大半）
    def hits(self, Y: np.ndarray) -> bool:
        if self.tree is None:
            return False
        return bool(np.isfinite(self.tree.query(Y, distance_upper_bound=self.r)[0]).any())

    def min_dist(self, Y: np.ndarray, cap: float = 2 * np.sin(np.radians(2.5))) -> float:
        """Y 到已拼好部分的最近距离（弦长，单位球），最多算到 cap（默认 5°）：远处的点不用细查，查询快一个数量级。"""
        if self.tree is None:
            return cap
        d = self.tree.query(Y, distance_upper_bound=cap)[0].min()
        return float(d) if np.isfinite(d) else cap


def slide_to_contact(Y, n_axis, total, contact: Contact, min_step=np.radians(0.1)):
    """把 Y 绕 n_axis 旋转，返回第一次接触的角度（rad；二分到约 0.01°），total 内没有接触返回 None。
    保守推进：转过角度 a 时任何一点的位移（弦长）不超过 a，所以每步可以放心转过"当前最近距离 - 接触半径"。"""
    a = 0.0
    prev = None
    while a <= total:
        d = contact.min_dist(Y @ rotmat(n_axis, a).T)
        if d < contact.r:
            if prev is None:
                return 0.0
            lo, hi = prev, a
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                if contact.hits(Y @ rotmat(n_axis, mid).T):
                    hi = mid
                else:
                    lo = mid
            return hi
        prev = a
        a += max(d - contact.r, min_step)
    return None


# ---- 陆块 --------------------------------------------------------------------------------------
@dataclass
class Block:
    id: int
    cont: np.ndarray        # 陆壳节点（固定网格编号，300 Ma）
    land: np.ndarray        # 其中高出海面的节点
    area_km2: float
    centroid: np.ndarray
    members: tuple = ()     # 编组的各板块（见 group_adjacent）；空 = 只有 id 本身


def group_adjacent(mesh: Mesh, blocks: dict, merge_km: float = 250.0, links: list | None = None,
                   safety_km: float = 100.0) -> dict:
    """300 Ma 时相距不到 merge_km 的陆块编成一组，整组一起运动（组里最大的一块领头，其余跟随）。
    前史裂解后不少陆块对到 300 Ma 仍然贴在一起（8 个世界种子里 7 个有相距只一个网格的陆块对：
    裂解沿省份边界走，分界参差咬合，各块又大体同向漂离老超大陆中心），剧本里只要其中一块一动就撞上另一块；
    1.2c 早期把它们当独立陆块规划，8 个种子里有 5 个在 288 Ma 前后就出现计划外接触。
    物理上相当于这条裂谷停滞（夭折裂谷），两侧之后一起运动。
    （试过改前史的漂移方向让各块彼此分开，参差咬合的分界使刚体漂移无法处处分开，没有采用。）
    U20 起前史的分界能张开，并在设计网格上判定哪些陆块对到 300 Ma 仍连在一起（InitialWorld.links）。给出 links 时按它编组，
    另外只把引擎结果里主体真正贴上（< safety_km）的陆块对补并进来：在各自网格上按 merge_km 量间距编组时，
    贴近 250 km 的陆块对会随网格翻转（L6/L7 差 ±50 km），L6 上的筛查就不能代表 L7。"""
    ids = sorted(blocks, key=lambda b: -blocks[b].area_km2)
    root = {b: b for b in ids}

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        big, small = (ra, rb) if blocks[ra].area_km2 >= blocks[rb].area_km2 else (rb, ra)
        root[small] = big

    if links is not None:
        for a, b in links:
            if a in root and b in root:
                union(a, b)
    r = chord_from_km(merge_km if links is None else safety_km)
    trees = {b: cKDTree(mesh.xyz[blocks[b].cont]) for b in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if find(a) == find(b):
                continue
            d = trees[b].query(mesh.xyz[blocks[a].cont], distance_upper_bound=r)[0]
            if np.isfinite(d).any():
                union(a, b)
    out = {}
    for lead in ids:
        if find(lead) != lead:
            continue
        mem = [b for b in ids if find(b) == lead]
        if len(mem) == 1:
            out[lead] = blocks[lead]
            continue
        cont = np.concatenate([blocks[b].cont for b in mem])
        land = np.concatenate([blocks[b].land for b in mem])
        a = float(sum(blocks[b].area_km2 for b in mem))
        out[lead] = Block(lead, cont, land, a, _centroid(mesh.xyz[cont], mesh.area_km2[cont]), tuple(mem))
    return out


def blocks_from_world(mesh: Mesh, w: InitialWorld, min_km2: float = 3.0e6, body_km2: float = 1.0e6,
                      strip_km: float = 300.0) -> dict:
    """各板块上的陆块主体，规划拼合与接触都只用主体：
    - 去掉窄于 strip_km 的细带（形态学开运算：先去掉离海不到 strip_km/2 的部分，再向外扩回）——板块洋壳上的岛弧带
      只有一个网格宽、却可以连着主体延伸数千公里，1.2c 初版把它们算进去，陆块碰到这些细带就停下；
    - 再只取面积 ≥ body_km2 的连通部分。
    细带仍在板块上，物理模拟里照常参与碰撞。"""
    v = w.crust
    elev = w.snapshots[-1]["elev"]
    out = {}
    e = mesh.edges
    cont_all = v.ctype == CONT
    d_sea = km_from_chord(cKDTree(mesh.xyz[~cont_all]).query(mesh.xyz, workers=KD_WORKERS)[0])
    core = cont_all & (d_sea >= strip_km / 2.0)
    d_core = km_from_chord(cKDTree(mesh.xyz[core]).query(mesh.xyz, workers=KD_WORKERS)[0]) if core.any() else np.full(mesh.n, np.inf)
    body = cont_all & (d_core <= strip_km / 2.0 + mesh.h_km)
    for q in np.unique(v.plate):
        m = (v.plate == q) & body
        if not m.any():
            continue
        lab = graph_components(mesh.n, e[m[e[:, 0]] & m[e[:, 1]]])
        areas = np.bincount(lab[m], weights=mesh.area_km2[m])
        big = np.flatnonzero(areas >= body_km2)
        cont = np.flatnonzero(m & np.isin(lab, big))
        a = float(mesh.area_km2[cont].sum())
        if a < min_km2:
            continue
        land = cont[elev[cont] > 0]
        out[int(q)] = Block(int(q), cont, land, a, _centroid(mesh.xyz[cont], mesh.area_km2[cont]))
    return out


def world_usable(blocks: dict, p: "ScenarioParams") -> tuple[bool, str]:
    """前史种子筛选：编组后至少 min_groups 个陆块组（主大陆至少要两块拼合，外加极地、次大陆），最大一组不超过 max_block_km2
    （一组就占了大半陆壳时，"分散的陆块汇聚"这个情景无从谈起）。"""
    n = len(blocks)
    amax = max((b.area_km2 for b in blocks.values()), default=0.0)
    if n < p.min_groups:
        return False, f"编组后只有 {n} 个陆块组（至少 {p.min_groups}）"
    if amax > p.max_block_km2:
        return False, f"最大陆块组 {amax / 1e6:.0f} Mkm² 超过 {p.max_block_km2 / 1e6:.0f}"
    return True, ""


# ---- 主内海代理（网格上快速估计） --------------------------------------------------------------
def basin_proxy(mesh: Mesh, land_pts: list, strait_km: float = 150.0, min_km2: float = 3.0e5):
    """land_pts：各部分（陆块）的陆地点集（已摆好位置）。返回被窄于 strait_km 的水道与外洋隔开的水体列表：
    面积、长宽比、两岸属于各部分的比例、质心。"""
    pts = np.concatenate(land_pts)
    part = np.concatenate([np.full(len(p), k) for k, p in enumerate(land_pts)])
    d, k = cKDTree(pts).query(mesh.xyz, workers=KD_WORKERS)
    dkm = km_from_chord(d)
    land = dkm < 0.6 * mesh.h_km
    deep = ~land & (dkm > strait_km / 2.0)
    e = mesh.edges
    lab = graph_components(mesh.n, e[deep[e[:, 0]] & deep[e[:, 1]]])
    a_all = np.bincount(lab[deep], weights=mesh.area_km2[deep])
    ids = np.flatnonzero(a_all > 0)
    areas = a_all[ids]
    if ids.size < 2:
        return []
    world = ids[np.argmax(areas)]
    out = []
    oc = np.flatnonzero(~land)
    dn = np.flatnonzero(deep)
    _, kk = cKDTree(mesh.xyz[dn]).query(mesh.xyz[oc], workers=KD_WORKERS)
    region = np.full(mesh.n, -1, np.int64)
    region[oc] = lab[dn[kk]]
    for i, a in zip(ids, areas):
        if i == world or a < min_km2 * 0.5:
            continue
        reg = region == i
        A = float(mesh.area_km2[reg].sum())
        if A < min_km2:
            continue
        # 沿岸陆地按部分统计
        nb = np.zeros(mesh.n, bool)
        nb[e[reg[e[:, 0]], 1]] = True
        nb[e[reg[e[:, 1]], 0]] = True
        coast = nb & land
        cnt = np.bincount(part[k[coast]], weights=mesh.area_km2[coast], minlength=len(land_pts))
        shares = cnt / max(cnt.sum(), 1e-9)
        X = mesh.xyz[reg]
        c = _centroid(X, mesh.area_km2[reg])
        e1 = unit(np.cross(c, [0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.cross(c, [1.0, 0.0, 0.0]))
        e2 = np.cross(c, e1)
        uv = np.stack([X @ e1, X @ e2], 1)
        wgt = mesh.area_km2[reg]
        mu = (uv * wgt[:, None]).sum(0) / wgt.sum()
        cov = ((uv - mu).T * wgt) @ (uv - mu) / wgt.sum()
        ev = np.sort(np.linalg.eigvalsh(cov))
        asp = float(np.sqrt(ev[1] / max(ev[0], 1e-12)))
        out.append({"area_km2": A, "aspect": asp, "shares": shares.tolist(), "centroid": c, "nodes": np.flatnonzero(reg)})
    return out


def official_inland_sea(mesh: Mesh, raster: Raster, land_nodes: np.ndarray, part_of_node: np.ndarray) -> dict:
    """用阶段 0 的 A 层代码核对（mesh 节点上的陆地掩膜栅格化到 1024x512），与 metrics.compute.compute_all 同一串调用：
    主大陆、亚大陆切分（颈宽 < 500 km 处切开）、主内海（面积、与第二大水体之比、海峡、接壤亚大陆、长宽比）。
    part_of_node：每个陆地节点属于哪一块陆块，只作诊断（1.2c 初版按陆块算接壤，两侧陆块沿宽阔碰撞带焊成一个亚大陆时
    规划以为接壤 2 块、模拟后 A 层只认 1 个）。"""
    from mythic.metrics import hard
    from mythic.metrics.compute import load_config
    from mythic.metrics.topology import any_neighbor

    ca = load_config()["A_hard_constraints"]
    g = raster.grid
    land = raster(land_nodes)
    part = raster(part_of_node)
    units = hard.landmass_units(g, land, ca["main_continent"]["thin_link_width_km"])
    land_area = float((g.cell_area_km2 * land).sum())
    main_share = float(units["areas_km2"][units["main_id"]]) / land_area if units["n"] and land_area else 0.0
    cs = ca["subcontinents"]
    sub = hard.subcontinents(g, units["main_mask"], cs["neck_radius_km"], cs["min_area_km2"])
    cis = ca["main_inland_sea"]
    seas = hard.inland_seas(g, land, units["main_mask"], cis["strait_max_width_km"], cis["min_main_border_fraction"],
                            cis["min_candidate_area_km2"])
    chk = hard.main_inland_sea_check(seas["seas"], cis)
    if not chk.get("exists"):
        return {"pass": False, "reason": chk.get("reason"), "main_share_pred": main_share, "n_sub": sub["count"]}
    sea = seas["region_labels"] == chk["main"]["region_id"]
    geo = hard.inland_sea_geometry(g, land, sea, sub["labels"], sub["big_ids"],
                                   cis.get("bordering_subcontinents", {}).get("min_border_share", 0.0))
    chk = hard.apply_inland_sea_geometry(chk, geo, cis)
    bw = g.cell_area_km2 * (land & any_neighbor(sea))
    shares = {int(p): float((bw * (part == p)).sum() / bw.sum()) for p in np.unique(part[land & any_neighbor(sea)])}
    # 海峡：规划网格（L6，约 110 km）分辨不出 150 km 以下的海峡，计划留的 60–200 km 缺口栅格化后常被陆地封死（海峡数 0）。
    # 规划阶段把"0 条海峡"视为待定（unresolved），由 0 Ma 的完整模拟按 A 层判定（闭环修正）。
    straits = chk["main"]["straits_to_world_ocean"]
    checks = dict(chk["checks"])
    unresolved = straits == 0
    if unresolved:
        checks["straits_ok"] = True
    ok = bool(all(checks.values()))
    lat, lon = chk["main"]["sample_lat_lon"]
    return {"pass": ok, "area_km2": chk["main"]["area_km2"], "straits": straits, "strait_unresolved": unresolved,
            "aspect": float(geo["aspect_ratio"]), "n_bordering_sub": geo["n_bordering_subcontinents"],
            "sub_shares": [round(s["border_share"], 2) for s in geo["subcontinent_border_shares"][:3]],
            "shares": shares, "checks": checks, "lat": lat, "lon": lon, "main_share_pred": main_share,
            "n_sub": sub["count"]}


# ---- 规划 ------------------------------------------------------------------------------------------
@dataclass
class _PlanWorld:
    """规划用的粗网格世界：只要板块、地壳类型、省份与高程。"""
    crust: Crust
    snapshots: list
    seed: int


def coarsen_world(mesh_run: Mesh, world: InitialWorld, mesh_plan: Mesh) -> _PlanWorld:
    """把运行网格上的 300 Ma 世界映射到规划网格（每个规划节点取最近的运行节点）。"""
    _, idx = mesh_run.tree.query(mesh_plan.xyz, workers=KD_WORKERS)
    c = world.crust.gather(idx)
    c.node = np.arange(mesh_plan.n)
    return _PlanWorld(c, [{"elev": world.snapshots[-1]["elev"][idx]}], world.seed)


class Planner:
    """规划都在粗网格（默认 L6，约 110 km）上做：只是刚体旋转，与分辨率无关，写出的剧本直接用于细网格（L7/L8）的模拟。"""

    def __init__(self, mesh: Mesh, world: InitialWorld, seed: int, params: ScenarioParams | None = None,
                 plan_level: int = 6):
        from .mesh import load_mesh
        self.w = world
        if mesh.level > plan_level:
            self.m = load_mesh(plan_level)
            self.pw = coarsen_world(mesh, world, self.m)
        else:
            self.m = mesh
            self.pw = world
        self.p = params or ScenarioParams()
        self.rng = np.random.default_rng(10_000 + seed)
        self.seed = seed
        self.blocks = group_adjacent(self.m, blocks_from_world(self.m, self.pw), links=getattr(world, "links", None))
        self._attach_small_plates(world)
        self.follower_of = {q: b for b, B in self.blocks.items() for q in B.members if q != b}
        self.usable, self.unusable_reason = world_usable(self.blocks, self.p)
        self.notes: list[str] = []

    def _attach_small_plates(self, world: InitialWorld, max_km: float = 1500.0):
        """前史的微大陆和其他太小、不单独规划的陆块板块（规划陆块主体小于 3 Mkm²）：跟随 300 Ma 时最近的规划陆块一起运动
        （离岸地体，像马达加斯加随非洲），并计入那块的点集，规划拼合、途中检查、主内海预测都看得到它们。
        1.2c 初版让它们按洋板块随机慢漂，挡在别的陆块路上：世界 1 的极地陆块途中撞上两块微大陆和次大陆，9 Mkm² 被吃到 2 Mkm²。
        只处理前史记录的陆块板块（InitialWorld.blocks）；洋板块上的岛弧照旧随洋板块运动。"""
        v, m = self.pw.crust, self.m
        cont = v.ctype == CONT
        elev = self.pw.snapshots[-1]["elev"]
        grouped = {q for B in self.blocks.values() for q in (B.members or (B.id,))}
        trees = {b: cKDTree(m.xyz[B.cont]) for b, B in self.blocks.items()}
        self.attached = {}
        for q in sorted(int(x["id"]) for x in world.blocks):
            if q in grouped:
                continue
            idx = np.flatnonzero(cont & (v.plate == q))
            if idx.size < 3 or not trees:
                continue
            dist = {b: float(km_from_chord(tr.query(m.xyz[idx])[0].min())) for b, tr in trees.items()}
            lead = min(dist, key=dist.get)
            if dist[lead] > max_km:
                continue
            B = self.blocks[lead]
            cont_n = np.concatenate([B.cont, idx])
            land_n = np.concatenate([B.land, idx[elev[idx] > 0]])
            self.blocks[lead] = Block(lead, cont_n, land_n, B.area_km2 + float(m.area_km2[idx].sum()), B.centroid,
                                      tuple(B.members or (lead,)) + (q,))
            self.attached[q] = {"lead": lead, "dist_km": round(dist[lead]), "area_km2": float(m.area_km2[idx].sum())}

    # 锚定参照系中第 b 块在时刻 t 的旋转（拼合规划用；锁定后的残余汇聚计入最终位置）
    def pose(self, b: int, t: float) -> np.ndarray:
        return dock_pose(self.docks.get(b), t, self.p.residual, self.p.residual_myr)

    def posed(self, b: int, t: float, which: str = "cont", sub: int = 1) -> np.ndarray:
        if which == "dock":
            which = self._dock_which(b)
        idx = getattr(self.blocks[b], which)[::sub]
        return self.m.xyz[idx] @ self.pose(b, t).T

    def _dock_which(self, b: int) -> str:
        """拼合接触、0 Ma 海岸预测用的点集：陆地（被动陆缘在水下，见 ScenarioParams.dock_on）；陆地太少时退回全部陆壳。"""
        B = self.blocks[b]
        return "land" if self.p.dock_on == "land" and len(B.land) >= max(20, 0.2 * len(B.cont)) else "cont"

    def assign_roles(self):
        p, rng = self.p, self.rng
        bl = sorted(self.blocks.values(), key=lambda b: -b.area_km2)
        roles = {}
        polar = [b for b in bl[1:] if p.polar_area_km2[0] <= b.area_km2 <= p.polar_area_km2[1]]
        if polar:
            # 离最大陆块（锚定陆块）最远的那块：主大陆在它周围拼合，离得远的不容易被拼合中的陆块碾过；
            # 整体定位再把南极放到它附近（plan_placement）。候选多时随机在前两名里挑。
            polar.sort(key=lambda b: -_ang(b.centroid, bl[0].centroid))
            roles[polar[int(rng.integers(min(2, len(polar))))].id] = "polar"
        else:
            self.notes.append("没有面积合适的陆块做极地大陆")
        sec = [b for b in bl[1:] if b.id not in roles and p.secondary_area_km2[0] <= b.area_km2 <= p.secondary_area_km2[1]]
        main_left = [b for b in bl if b.id not in roles]
        total = sum(b.area_km2 for b in bl)
        share_if_none = sum(b.area_km2 for b in main_left) / total
        need = share_if_none > p.main_share_max          # 不设次大陆主大陆就会超过 85%
        if sec and len(main_left) >= 3 and (need or rng.random() < p.p_secondary):
            # 需要时挑能把主大陆占比压到上限以下、又最接近 78% 的那块
            pick = min(sec, key=lambda b: abs((sum(x.area_km2 for x in main_left) - b.area_km2) / total - 0.78))
            roles[pick.id if need else sec[int(rng.integers(len(sec)))].id] = "secondary"
        for b in bl:
            roles.setdefault(b.id, "main")
        self.roles = roles
        self.main = [b.id for b in bl if roles[b.id] == "main"]
        self.anchor = self.main[0]
        return roles

    def _axis_dir(self, b: int, C: np.ndarray) -> np.ndarray:
        """拼合主轴在 C 处的切向，朝向第 b 块所在的那一端（让主大陆拼成长条，见 plan_assembly）。"""
        u = unit(self.axis_u - (self.axis_u @ C) * C)
        cb = self.blocks[b].centroid
        return u if (cb @ u) >= 0 else -u

    def _plan_dock(self, b: int, t_dock: float, az_offsets, spins, pens, check_transit=True, staged=True,
                   own_offsets=()):
        """返回满足速度（与途中无碰撞，check_transit 时）的候选（Dock 列表），按接近方向偏离程度排序。
        接近方向以拼合主轴为基准（偏离 az_offsets 度）；own_offsets 另加以陆块自身所在方位（从已拼部分中心看）为基准的
        接近方向——陆块坐在主轴侧面时，从主轴端点接近要绕过半个已拼部分（1.2c 第三步：世界 4 的主内海那块要转 119°）。
        直接路线都在途中碰撞时，按偏离程度试前几个终点的分段路线（_staged_routes）。"""
        p, m = self.p, self.m
        docked = [k for k in self.docks]
        cl_pts = np.concatenate([self.posed(k, 0.0) for k in docked])
        contact = Contact(np.concatenate([self.posed(k, 0.0, "dock") for k in docked]), m.h_km)
        C = _centroid(cl_pts)
        rad_cl = max(_ang(C, x) for x in cl_pts[::7])
        X = m.xyz[getattr(self.blocks[b], self._dock_which(b))]          # 按陆地接触（dock_on）
        cb = self.blocks[b].centroid
        rad_b = max(_ang(cb, x) for x in X[::7])
        t = self._axis_dir(b, C)
        dirs = [(float(az), abs(az)) for az in az_offsets]                 # (相对主轴的方位角, 排序用的偏离量)
        ub = cb - (cb @ C) * C
        if len(own_offsets) and np.linalg.norm(ub) > 1e-6:
            ub = unit(ub)
            az_b = float(np.degrees(np.arctan2(C @ np.cross(t, ub), t @ ub)))
            dirs += [(az_b + o, abs(o)) for o in own_offsets]
        out, fails = [], []
        for az, dev in dirs:
            dir_ = rotmat(C, np.radians(az)) @ t
            far = min(rad_cl + rad_b + np.radians(10), np.pi - 0.05)
            F = np.cos(far) * C + np.sin(far) * dir_
            for sp in spins:
                P0 = move_to(cb, F) @ rotmat(cb, np.radians(sp))
                Y = X @ P0.T
                n_ax = unit(np.cross(F, C))
                a_c = slide_to_contact(Y, n_ax, far, contact)
                if a_c is None or a_c == 0.0:
                    continue
                for pen in pens:
                    R = rotmat(n_ax, a_c + pen / R_KM) @ P0
                    ax, ang = axis_angle(R)
                    if np.degrees(ang) / (p.t0 - t_dock) > p.max_rate_deg:
                        continue
                    d = Dock(b, t_dock, [(float(p.t0), float(t_dock), ax, ang)])
                    d._R1, d._contact_pts = R, None
                    d._t_stop = t_dock + self._margin_myr(ax, ang / (p.t0 - t_dock), cb, pen)
                    d._approach_az = float(az)
                    if not check_transit or self._transit_ok(b, d, docked):
                        out.append((dev + 0.3 * abs(sp), d))
                    else:
                        fails.append((dev + 0.3 * abs(sp), d))
        if check_transit and staged and not out:
            for sc, d in sorted(fails, key=lambda x: x[0])[:6]:
                v = next((v for v in self._staged_routes(b, d, docked, n=8) if self._transit_ok(b, v, docked)), None)
                if v is not None:
                    out.append((sc, v))
        out.sort(key=lambda x: x[0])
        return [d for _, d in out]

    def _margin_myr(self, axis: np.ndarray, rate_rad: float, x: np.ndarray, pen_km: float) -> float:
        """按陆地接触锁定时，两侧水下陆缘（各宽约 margin_km）比陆地早多久碰上（Myr）：陆缘总宽加楔入量除以接近速度，再加 5 Myr。
        途中检查到这个时刻为止；此后与目标大陆的接触是计划内的（1.2c 初版固定为拼合前 15 Myr，按陆地拼合后合理的路线被判失败）。"""
        v = rate_rad * R_KM * max(np.linalg.norm(np.cross(unit(axis), x)), 0.2)
        return float(min(60.0, (2.0 * self.p.margin_km + pen_km) / max(v, 1e-6) + 5.0))

    @staticmethod
    def _t_allow(d: Dock) -> float:
        """从这个时刻起，拼合陆块与目标大陆的接触是计划内的（途中检查、剧本核对都用它）。"""
        t = getattr(d, "_t_stop", None)
        if t is not None:
            return float(t)
        return (d.t_contact + 10.0) if d.t_contact is not None else (d.t_dock + 15.0)

    def _transit_ok(self, b, d: Dock, docked) -> bool:
        """途中不提前撞上已规划的陆块：检查到计划的第一次接触前 10 Myr（一段式拼合取 dock 前 15 Myr）。"""
        p = self.p
        X = self.m.xyz[self.blocks[b].cont[::3]]
        # 极地陆块、次大陆陆块在头 40 Myr 按 300 Ma 位置当障碍：拼合路线不得一开始就碾过它们
        # （世界种子 3：极地陆块夹在锚定陆块和另一块主陆块之间，那块主陆块的拼合路线正好扫过它，极地陆块无路可走）。
        # 只管头 40 Myr，给它们留出离开的时间；它们之后的路线在 plan_movers 里对照全部拼合路线检查
        # （全程当障碍太严：最后一块主陆块就近拼合的路线几乎都要经过次大陆陆块的出发位置）。
        # 主内海那块（最后规划）只在头 inner_obstacle_myr 当障碍（_obst_until），给它留出发的空间
        obst = tuple(k for k in getattr(self, "_obstacles", ()) if k != b)
        until = getattr(self, "_obst_until", {})
        ckey = (tuple(docked), obst, len(getattr(self, "_placed_movers", [])))
        if not hasattr(self, "_tcache") or self._tcache[0] != ckey:
            self._tcache = (ckey, {})
        cache = self._tcache[1]
        t_stop = self._t_allow(d)
        for t in np.arange(p.t0 - 2.0, t_stop, -3.0):
            Y = X @ dock_pose(d, t, p.residual, p.residual_myr).T
            key = round(float(t), 1)
            if key not in cache:
                pts = [self.posed(k, t, sub=3) for k in docked]
                pts += [self.m.xyz[self.blocks[k].cont[::3]] for k in obst if t >= until.get(k, p.t0 - 40.0)]
                # 已排好路线的极地、次大陆陆块（补拼进主大陆的陆块要避开它们，1.2c 初版只查已拼合的主陆块）
                for Xo, so in getattr(self, "_placed_movers", []):
                    pts.append(Xo[::2] @ dock_pose(Dock(0, so[-1][1], so), t, 0.0, 0.0).T)
                cache[key] = Contact(np.concatenate(pts), self.m.h_km * 1.2)
            if cache[key].hits(Y):
                return False
        return True

    def plan_assembly(self):
        p, rng = self.p, self.rng
        self.docks: dict[int, Dock] = {}
        self.docks[self.anchor] = Dock(self.anchor, float(p.t0), [])
        self.inner_sea = {"found": False}
        self._obstacles = tuple(b for b, r in self.roles.items() if r in ("polar", "secondary"))
        self._obst_until = {}
        rest = [b for b in self.main if b != self.anchor]
        if not rest:
            self.notes.append("只有一块主陆块，无法拼合")
            return
        if len(rest) > 1 and p.inner_obstacle_myr > 0:
            # 主内海那块（rest[0]，最后规划）的出发位置在头 inner_obstacle_myr 内当其他陆块的障碍：先规划的邻近陆块的路线
            # 常在出发后 4–12 Myr 扫过它的出发位置，它之后怎么走都会撞上（1.2c 第三步世界 1：陆块 7 与主内海那块 1，
            # 300 Ma 相距 331 km，296 Ma 只剩 46 km）。第二轮试过 40 Myr（当时没有分段路线，其他陆块无路可绕，整体变差）
            self._obstacles += (rest[0],)
            self._obst_until[rest[0]] = p.t0 - p.inner_obstacle_myr
        # 拼合主轴：各主陆块从主轴两端接上去，主大陆拼成长条（1.2c 初版各块从四面拼上，成了大圆团，
        # 约 1.3 亿 km² 的圆团纬度跨度 140°，放不进 40°S–70°N；亚非欧也是东西向长条）。
        # 主轴取其余主陆块质心在锚定陆块切平面上的主方向（少走路），再随机偏转 ±20°。
        ca = self.blocks[self.anchor].centroid
        e1 = unit(np.cross(ca, [0.0, 0.0, 1.0]) if abs(ca[2]) < 0.9 else np.cross(ca, [1.0, 0.0, 0.0]))
        e2 = np.cross(ca, e1)
        uv = np.array([[self.blocks[b].centroid @ e1, self.blocks[b].centroid @ e2] for b in rest])
        if len(rest) >= 2:
            w_, v_ = np.linalg.eigh(uv.T @ uv)
            a0 = np.arctan2(v_[1, -1], v_[0, -1])
        else:
            a0 = np.arctan2(uv[0, 1], uv[0, 0])
        a0 += np.radians(rng.uniform(-20, 20))
        self.axis_u = np.cos(a0) * e1 + np.sin(a0) * e2
        # 第二大的主陆块最后规划、最后拼上，圈出主内海：它的拼合按其余各块拼好后的真实轮廓选，闭环修正也只改它一块。
        # （1.2c 第三步试过让它最先规划、其余陆块避开这片海：4 个世界主大陆占比 51/61/35/56%（0/4 达标，第二轮 3/4），
        # 每个世界多出 1–2 块拼不上的孤岛；闭环里主内海一变，后面各块的拼合跟着变，修正不收敛。主内海那块选不出来的
        # 真正原因是路线只有一段（一个欧拉极直接转到终点，会扫过锚定陆块或邻近陆块），改用分段路线解决，见 _staged_routes。
        # 也试过把它的 300 Ma 位置当作其他块头 40 Myr 的障碍，其他块拼合受限，整体变差。）
        inner = rest[0]
        others = sorted(rest[1:], key=lambda b: _ang(self.blocks[b].centroid, self.blocks[self.anchor].centroid))
        times = np.linspace(p.dock_window[0], p.dock_window[1], len(rest) + 1)[:-1] if len(rest) > 1 else []
        times = [float(5 * round((tt + rng.uniform(-8, 8)) / 5)) for tt in times]
        for b, tdock in zip(others, times):
            cands = self._plan_dock(b, tdock, [0, -20, 20, -40, 40], [0, -30, 30], [p.penetration_km])
            if not cands:
                cands = self._plan_dock(b, max(tdock - 30, p.dock_window[1]), [0, -30, 30, -60, 60, -90, 90],
                                        [0, -45, 45], [p.penetration_km])
            if not cands:
                # 第三轮：从主轴两侧甚至反方向接近、早一些拼上（陆块多了以后，只从主轴两端接近常常无路可走）
                cands = self._plan_dock(b, min(tdock + 30, p.dock_window[0]), [-120, 120, -150, 150, 180, -90, 90],
                                        [0, -60, 60], [p.penetration_km])
            if not cands:
                self.notes.append(f"陆块 {b} 找不到无碰撞的拼合路线，改为留在原处")
                self.roles[b] = "island"
                continue
            # 不圈出内洋：优先选拼上后已拼部分圈住的水体不超过 inner_ocean_km2 的候选
            pick = None
            for d in cands[:8]:
                self.docks[b] = d
                if self._max_enclosed_km2() <= p.inner_ocean_km2:
                    pick = d
                    break
            if pick is None:
                pick = cands[0]
                self.notes.append(f"陆块 {b} 的拼合候选都圈出内洋（> {p.inner_ocean_km2 / 1e6:.0f} Mkm²），取第一个")
            self.docks[b] = pick
        self.inner_sea = self._plan_inner_sea(inner, float(5 * round(p.dock_window[1] / 5)))

    def _max_enclosed_final(self) -> float:
        """按写好的剧本，所有规划陆块（含极地、次大陆、留在原处的陆块）0 Ma 的陆地圈住的最大水体面积。
        只看拼合陆块不够：1.2c 第三步世界 2 规划时拼合陆块只圈住 0.7 Mkm²，模拟里一块留在原处的陆块把 2400 万 km² 的
        海围了起来。"""
        s = self.script
        parts = [self.m.xyz[getattr(B, self._dock_which(b))] @ s.forward_total(b, 0).T
                 for b, B in self.blocks.items() if b in s.stages]
        if len(parts) < 2:
            return 0.0
        return max((x["area_km2"] for x in basin_proxy(self.m, parts)), default=0.0)

    def _max_enclosed_km2(self) -> float:
        """已拼合各块（0 Ma 位置、按陆地）圈住的最大水体面积（与外洋之间只有窄于 150 km 的水道），用于拒绝内洋。"""
        parts = [self.posed(k, 0.0, "dock") for k in self.docks]
        if len(parts) < 2:
            return 0.0
        return max((s["area_km2"] for s in basin_proxy(self.m, parts)), default=0.0)

    def _staged_routes(self, b: int, d: Dock, docked: list, n: int = 10) -> list:
        """同一个终点（d 第一段转到的位置 R1，之后各段不变），换成分段路线：可选的原地等待 -> 背离 300 Ma 时最近的邻居
        3–15° -> 可选的径向退到已拼部分外面、绕已拼部分的中心转到终点所在方位（短路或长路）-> 接近终点。
        一个欧拉极直接转到终点的路线会扫过挡在中间的陆块（1.2c 第三步诊断：世界 4 主内海那块 163 个候选里 157 个途中
        撞上锚定陆块本身——它在锚定陆块的另一侧；世界 2 的 91/94 撞上旁边 386 km 处极地陆块的出发位置）。
        地球上陆块也不走单一欧拉极：印度先向北东、后向北，澳大利亚先南后北。终点不变，所以主内海的评分与 A 层核对仍然有效。
        随机数按 (种子, 陆块, 抵达时刻) 单独取，不影响规划其他部分的随机抽样，闭环重做同一次抽样时路线相同。"""
        p, m = self.p, self.m
        ta0, t1, ax1, ang1 = d.segs[0]
        R1 = rotmat(ax1, ang1)
        X = m.xyz[self.blocks[b].cont]
        cb = self.blocks[b].centroid
        rad_b = max(_ang(cb, x) for x in X[::7])
        cl = np.concatenate([self.posed(k, 0.0, sub=3) for k in docked]) if docked else m.xyz[self.blocks[self.anchor].cont]
        C = _centroid(cl)
        rad_cl = max(_ang(C, x) for x in cl[::5])
        others = np.concatenate([m.xyz[B.cont[::2]] for k, B in self.blocks.items() if k != b])
        dn, jn = cKDTree(others).query(X[::2])
        nb = others[jn[np.argmin(dn)]]
        away = np.cross(nb, cb)
        away = unit(away) if np.linalg.norm(away) > 1e-6 else unit(np.cross(C, cb))
        cf = R1 @ cb
        rng = np.random.default_rng([int(self.seed), int(b), int(round(t1)), 5])
        out = []
        # 走法由轻到重：只先背离邻居（多数碰撞发生在出发后 10–20 Myr，与同样在移动的邻近陆块相撞）、原地等一阵再走、
        # 背离后绕已拼部分中心转到终点方位、先退到已拼部分外面再绕（转角大，常超速，只占少数）
        kinds = ["escape", "wait", "orbit", "around"]
        for _ in range(n):
            kind = kinds[int(rng.choice(4, p=[0.35, 0.25, 0.25, 0.15]))]
            delay = float(rng.choice([10.0, 20.0, 30.0])) if kind == "wait" else 0.0
            t_start = p.t0 - delay
            # 背离方向：一半在"背离最近邻居" ±60° 内，一半任意（撞上的常是正在移动的另一块邻居，世界 1 诊断）
            spread = 60.0 if rng.random() < 0.5 else 180.0
            Rs = [rotmat(rotmat(cb, np.radians(rng.uniform(-spread, spread))) @ away, np.radians(rng.uniform(3, 15)))]
            c1 = Rs[0] @ cb
            if kind == "around":
                need = rad_cl + rad_b + np.radians(3.0) - _ang(C, c1)       # 绕行前先退到已拼部分外面
                if need > 0 and np.linalg.norm(np.cross(C, c1)) > 1e-6:
                    Rs.append(rotmat(unit(np.cross(C, c1)), need))
                    c1 = Rs[-1] @ c1
            u1, uF = c1 - (c1 @ C) * C, cf - (cf @ C) * C
            if kind in ("orbit", "around") and np.linalg.norm(u1) > 1e-6 and np.linalg.norm(uF) > 1e-6:
                u1, uF = unit(u1), unit(uF)
                dphi = float(np.arctan2(np.cross(u1, uF) @ C, u1 @ uF))
                if rng.random() < 0.3:                                       # 走另一边绕过去
                    dphi -= np.sign(dphi) * 2 * np.pi
                if abs(dphi) > np.radians(5.0):
                    Rs.append(rotmat(C, dphi))
            Rcur = np.eye(3)
            for R in Rs:
                Rcur = R @ Rcur
            Rs.append(R1 @ Rcur.T)                                           # 接近终点
            angs = [axis_angle(R) for R in Rs]
            wts = np.array([max(a, np.radians(3.0)) for _, a in angs])
            cuts = t_start - np.concatenate([[0.0], np.cumsum(wts / wts.sum())]) * (t_start - t1)
            cuts = [float(round(c)) for c in cuts]
            cuts[-1] = float(t1)
            if any(cuts[i] <= cuts[i + 1] for i in range(len(Rs))):
                continue
            segs = [(cuts[i], cuts[i + 1], ax, a) for i, (ax, a) in enumerate(angs)]
            if any(np.degrees(a) / (ta - tb) > p.max_rate_deg for ta, tb, _, a in segs):
                continue
            if delay:
                segs = [(float(p.t0), t_start, np.array([0.0, 0.0, 1.0]), 0.0)] + segs
            # 残余汇聚沿用原路线的（一段式拼合是那一段；两段式是绕 P1 的那段，仍在最后、闭环会改它，不能存副本）
            res = d.res_seg if d.res_seg is not None else (d.segs[-1] if len(d.segs) == 1 else None)
            v = Dock(b, d.t_dock, segs + list(d.segs[1:]), t_contact=d.t_contact, inner_sea=d.inner_sea, res_seg=res)
            for k, val in vars(d).items():
                if k.startswith("_"):
                    setattr(v, k, val)
            ta, tb, axl, al = segs[-1]
            v._t_stop = t1 + self._margin_myr(axl, al / (ta - tb), Rcur @ cb, p.penetration_km)
            v._staged = {"kind": kind, "delay": delay, "n_legs": len(Rs)}
            out.append(v)
        return out

    def _pivot_candidates(self, b: int, t_dock: float) -> list:
        """两段式拼合候选：先接近到第一处接触 P1（并楔入），再绕 P1 转动，直到另一端几乎碰上，只留 gap 宽的海峡。
        一次滑动只会在一点接触，两块凸形大陆之间留下两头敞开的海湾，圈不出内海（1.2c 初版的失败）；
        两处接触之间才可能夹出残留洋盆（阿拉伯、印度的转动碰撞也是这样）。"""
        p, m = self.p, self.m
        # 接近方向：拼合主轴 ±60°，另加它自己所在方位 ±40°（在已拼部分的侧面圈海，像非洲在欧亚南侧圈出地中海）
        firsts = self._plan_dock(b, t_dock, list(range(-60, 61, 20)), [-30, 0, 30],
                                 [p.penetration_km, p.penetration_km + 250.0], check_transit=False,
                                 own_offsets=OWN_OFFSETS)
        docked = [k for k in self.docks if k != b]
        # 两端都按陆地：第一处接触是陆地碰陆地；另一端的海峡宽度是陆地到陆地的距离。两侧的被动陆缘（水下陆壳，宽 100–300 km）
        # 在 0 Ma 仍在水下（1.2c 初版已拼部分按全部陆壳算，模拟里口子总比规划宽，4/5 次运行因此圈不住海）
        cl = np.concatenate([self.posed(k, 0.0, "dock") for k in docked])
        tree_cl = cKDTree(cl)
        X = m.xyz[getattr(self.blocks[b], self._dock_which(b))]
        Xl = X
        out = []
        for d1 in firsts:
            R1 = d1._R1
            Z = X @ R1.T
            dd = km_from_chord(tree_cl.query(Z)[0])
            cpts = Z[dd < m.h_km]
            if len(cpts) < 3:
                continue
            P1 = _centroid(cpts)
            far_cl = cl[km_from_chord(np.linalg.norm(cl - P1, axis=1)) > 700.0]
            Zl = Xl @ R1.T
            Zf = Zl[km_from_chord(np.linalg.norm(Zl - P1, axis=1)) > 700.0]
            if len(far_cl) == 0 or len(Zf) == 0:
                continue
            cf = Contact(far_cl, m.h_km)
            tf = cKDTree(far_cl)
            for sgn in (1.0, -1.0):
                phi2 = None
                for phd in range(1, 71):
                    if cf.hits(Zf @ rotmat(P1, sgn * np.radians(phd)).T):
                        phi2 = np.radians(phd)
                        break
                if phi2 is None:
                    continue
                Zc = Zf @ rotmat(P1, sgn * phi2).T
                c2 = Zc[np.argmin(tf.query(Zc)[0])]
                L2 = km_from_chord(np.linalg.norm(c2 - P1))
                ax1, ang1 = axis_angle(R1)
                for gap in (60.0, 120.0, 200.0):
                    phi = phi2 - gap / max(L2, 1.0)
                    if phi < np.radians(2):
                        continue
                    t_c = float(5 * round((t_dock + max(15.0, np.degrees(phi) / 0.5)) / 5))
                    if t_c >= p.t0 - 20:
                        continue
                    if np.degrees(ang1) / (p.t0 - t_c) > p.max_rate_deg:
                        continue
                    segs = [(float(p.t0), t_c, ax1, ang1), (t_c, float(t_dock), P1, sgn * phi)]
                    dk = Dock(b, float(t_dock), segs, t_contact=t_c, inner_sea=True)
                    dk._t_stop = t_c + self._margin_myr(ax1, ang1 / (p.t0 - t_c), self.blocks[b].centroid, p.penetration_km)
                    dk._L2, dk._phi, dk._sgn, dk._gap = float(L2), float(phi), float(sgn), float(gap)
                    dk._c2 = c2                                  # 远端海峡的位置（锚定参照系，0 Ma）
                    out.append(dk)
        return out

    def _plan_inner_sea(self, b: int, t_dock: float) -> dict:
        """选出主内海那块的拼合（_select_inner_sea），再按闭环修正量调远端海峡宽度：两段式拼合的第二段绕第一处接触点 P1
        多转 dgap / L2（L2 = P1 到远端接触点的距离），远端陆地到陆地的缺口就收窄约 dgap km。"""
        info = self._select_inner_sea(b, t_dock)
        dg = getattr(self, "inner_dgap_km", 0.0)
        d = self.docks.get(b)
        if dg and d is not None and getattr(d, "_L2", None):
            phi = max(d._phi + dg / d._L2, np.radians(1.0))
            ta, tb, P1, _ = d.segs[-1]
            d.segs[-1] = (ta, tb, P1, d._sgn * phi)
            info["dgap_km"] = float(dg)
            info["gap_planned_km"] = float(d._gap - dg)
        return info

    def _select_inner_sea(self, b: int, t_dock: float) -> dict:
        """最后一块：在两段式与一段式候选里找让主大陆圈出主内海的（代理评分 -> 途中检查 -> A 层核对）。
        主内海不一定要挨着最后这块：前面几块拼合时可能已经夹出残留洋盆，最后这块只要把它收窄到合适大小、或不破坏它即可
        （1.2c 早期只认挨着最后一块的水体，世界种子 3 上前面已经夹出的 5.7 Mkm² 残留洋盆被忽略）。
        评分按 A 层的判定方式：取最大的内部水体，面积在目标范围、至少是第二大的 2 倍、至少两块陆块各占沿岸 15%、伸长。"""
        p, m = self.p, self.m
        cands = self._pivot_candidates(b, t_dock)
        n_pivot = len(cands)
        cands += self._plan_dock(b, t_dock, list(range(-90, 91, 30)), [-30, 0, 30],
                                 [p.penetration_km, p.penetration_km + 250.0], check_transit=False,
                                 own_offsets=OWN_OFFSETS)
        if not cands:
            self.notes.append(f"主内海陆块 {b} 找不到拼合路线")
            self.roles[b] = "island"
            return {"found": False}
        lo, hi = p.inner_sea_area_km2
        # 长宽比要求：A 层是 ≥ 2；闭环发现模拟里海比规划的圆（碰撞增厚、岛弧把海挤短挤宽），会按缩水比例提高（inner_min_aspect）
        need_asp = max(2.2, getattr(self, "inner_min_aspect", 0.0))
        scored = []
        for d in cands:
            self.docks[b] = d
            # 0 Ma 海岸预测：各块都按海面以上的陆地（被动陆缘仍在水下；1.2c 初版已拼部分按全部陆壳，口子在模拟里变宽）
            parts = [self.posed(k, 0.0, "dock") for k in self.docks]
            seas = sorted(basin_proxy(m, parts), key=lambda s: -s["area_km2"])
            if not seas:
                continue
            s = seas[0]
            if not (lo * 0.6 <= s["area_km2"] <= hi * 1.2):
                continue
            sh = sorted(s["shares"], reverse=True)
            second = seas[1]["area_km2"] if len(seas) > 1 else 0.0
            score = (abs(np.log(s["area_km2"] / 2.5e6)) + max(0.0, need_asp + 0.8 - s["aspect"]) * 0.5
                     + max(0.0, need_asp - s["aspect"]) * 3.0     # A 层要求长宽比 ≥ 2
                     + max(0.0, 0.2 - sh[1]) * 5 + (1.0 if second and s["area_km2"] < 2.2 * second else 0.0)
                     + (0.0 if getattr(d, "_L2", None) else 0.6))   # 优先两段式：闭环要调它第二段的转角
            scored.append((score, d, s))
        scored.sort(key=lambda x: x[0])
        n_in_range = len(scored)
        docked_before = [k for k in self.docks if k != b]
        # 途中检查：直接路线碰撞就试分段路线（终点不变，评分不变）
        kept, n_staged = [], 0
        for sc, d, s in scored[:20]:
            if not self._transit_ok(b, d, docked_before):
                d = next((v for v in self._staged_routes(b, d, docked_before, n=10)
                          if self._transit_ok(b, v, docked_before)), None)
                if d is None:
                    continue
                n_staged += 1
            kept.append((sc, d, s))
        scored = kept
        raster = Raster(m, 512)
        info = {"n_candidates": len(cands), "n_pivot": n_pivot, "n_in_range": n_in_range, "n_scored": len(scored),
                "n_staged": n_staged, "official_fails": []}
        for score, d, s in scored[:10]:
            self.docks[b] = d
            land_nodes, part = self._land_nodes_final(incoming=b)
            chk = official_inland_sea(m, raster, land_nodes, part)
            if chk["pass"] and chk.get("aspect", 0.0) < getattr(self, "inner_min_aspect", 0.0):
                chk = {**chk, "pass": False, "checks": {**chk["checks"], "loop_aspect_ok": False}}
            if chk["pass"]:
                self.docks[b].inner_sea = True
                return {"found": True, "proxy": {k: v for k, v in s.items() if k not in ("centroid", "nodes")}, "_nodes": s["nodes"],
                        "centroid": s["centroid"], "official": chk, **info}
            info["official_fails"].append(
                {"reason": chk.get("reason"), "area_Mkm2": round(chk.get("area_km2", 0) / 1e6, 2),
                 "aspect": round(chk.get("aspect", 0), 2), "straits": chk.get("straits"),
                 "failed": [k for k, v in chk.get("checks", {}).items() if not v],
                 "shares": {k: round(v, 2) for k, v in chk.get("shares", {}).items()}})
        if scored:
            score, d, s = scored[0]
            self.docks[b] = d
            self.docks[b].inner_sea = True
            self.notes.append("主内海：没有候选通过 A 层核对，采用代理评分最好的")
            return {"found": False, "proxy": {k: v for k, v in s.items() if k not in ("centroid", "nodes")}, "_nodes": s["nodes"], "centroid": s["centroid"],
                    **info}
        cands = sorted(cands, key=lambda d: d.angle)                      # 退路：路线短的优先，少碰撞
        ok = [d for d in cands if self._transit_ok(b, d, docked_before)]
        if not ok:
            # 退而求其次时也先试分段路线，免得直接取一条途中碰撞的（世界 1、4 因此在 288 Ma 就有计划外接触）
            for d in cands[:12]:
                v = next((v for v in self._staged_routes(b, d, docked_before, n=8)
                          if self._transit_ok(b, v, docked_before)), None)
                if v is not None:
                    ok = [v]
                    break
        self.docks[b] = (ok or cands)[0]
        self.notes.append("主内海：没有候选能圈出大小合适、途中不碰撞的水体")
        return {"found": False, **info}

    def _land_nodes_final(self, incoming: int | None = None):
        """所有已拼合主陆块在最终位置的陆地（0 Ma 海岸预测，同 basin_proxy），映射回固定网格节点（锚定参照系）。"""
        m = self.m
        pts, part = [], []
        for k in self.docks:
            P = self.posed(k, 0.0, "dock")
            pts.append(P)
            part.append(np.full(len(P), k))
        pts, part = np.concatenate(pts), np.concatenate(part)
        d, kk = cKDTree(pts).query(m.xyz, workers=KD_WORKERS)
        land = km_from_chord(d) < 0.6 * m.h_km
        return land, np.where(land, part[kk], 0)

    def _sec_targets(self, G: np.ndarray, C: np.ndarray, lats=(-20.0, 0.0, 20.0), dlons=(150.0, 180.0, 210.0)):
        """次大陆目标（锚定参照系）：0 Ma 时在主大陆质心经度 +140°–220°（远洋另一侧）、低纬。"""
        lon_c, _ = xyz_to_lonlat(G @ C)
        return [G.T @ lonlat_to_xyz(float(lon_c) + dl, la) for la in lats for dl in dlons]

    def plan_placement(self):
        """整体定位：把拼好的主大陆（锚定参照系）转到目标纬度。
        纬度满足要求的定位里，再挑极地大陆、次大陆（相对主大陆）要走的路最短的（1.2c 早期南极随便放，
        极地陆块要横穿正在拼合的主大陆去南极，常常找不到不碰撞的路线）。"""
        p, m, rng = self.p, self.m, self.rng
        pts = np.concatenate([self.posed(k, 0.0, "land", sub=2) for k in self.docks])
        wts = np.concatenate([m.area_km2[self.blocks[k].land[::2]] for k in self.docks])
        C = _centroid(pts, wts)
        sea_c = self.inner_sea.get("centroid") if self.inner_sea else None
        polar = [b for b, r in self.roles.items() if r == "polar" and b not in self.docks]
        sec = [b for b, r in self.roles.items() if r == "secondary" and b not in self.docks]
        S = np.array([0.0, 0.0, -1.0])
        best = None
        for lat_c in (5.0, 10.0, 15.0, 20.0, 25.0):
            T = lonlat_to_xyz(rng.uniform(-180, 180), lat_c)
            M = move_to(C, T)
            for sp in range(0, 360, 10):
                G = rotmat(T, np.radians(sp)) @ M
                Y = pts @ G.T
                lat = np.degrees(np.arcsin(np.clip(Y[:, 2], -1, 1)))
                north, south = np.percentile(lat, 99.7), np.percentile(lat, 0.3)
                arctic = wts[lat > 66.56].sum() / wts.sum()
                pen = (max(0, p.target_north[0] - north) + max(0, north - p.target_north[1])
                       + max(0, p.target_south[0] - south) + max(0, south - p.target_south[1]) + 300 * max(0, arctic - 0.05))
                if sea_c is not None:
                    sl = np.degrees(np.arcsin((G @ sea_c)[2]))
                    pen += max(0, p.inner_sea_lat[0] - sl) + max(0, sl - p.inner_sea_lat[1])
                travel = 0.0
                for b in polar:
                    travel += max(0.0, np.degrees(_ang(self.blocks[b].centroid, G.T @ S)) - (90.0 + p.polar_lat))
                for b in sec:
                    travel += min(np.degrees(_ang(self.blocks[b].centroid, x)) for x in self._sec_targets(G, C))
                score = pen + 0.02 * travel
                if best is None or score < best[0]:
                    best = (score, pen, travel, G, north, south, arctic)
        score, pen, travel, G, north, south, arctic = best
        self.G = G
        self.G_axis, self.G_angle = axis_angle(G)
        self.main_C = C
        self.placement = {"penalty": float(pen), "travel_deg": float(travel), "north_deg": float(north),
                          "south_deg": float(south), "arctic_fraction": float(arctic)}

    def abs_main(self, t: float, sub: int = 4) -> np.ndarray:
        """主大陆各块在 t 时刻的绝对位置（锚定陆块的整体定位旋转 × 锚定参照系中的位置）。"""
        Ra = rotmat(self.G_axis, self.G_angle * (self.p.t0 - t) / self.p.t0)
        return np.concatenate([self.posed(k, t, sub=sub) for k in self.docks]) @ Ra.T

    def plan_movers(self):
        """极地大陆、次大陆、没拼进主大陆的孤立陆块：相对锚定陆块运动（锚定参照系），0 Ma 时抵达整体定位后的目标位置；
        全程（300 -> 0 Ma）不得碰到主大陆各块或彼此。路线可以是直接一段，也可以先背离主大陆绕出去一段再去目标（两段）；
        出发时刻可以晚一些。目标位置按路程从近到远优先试。
        （1.2c 初版只检查了终点，极地陆块南下途中撞穿一块主陆块，被吞掉的陆壳把那块整个变成了高原；
        之后在绝对参照系里规划，主大陆整体定位的旋转会扫过大半个地球，极地陆块常常找不到路。
        碰撞只取决于相对运动，所以改在锚定参照系里规划，并让整体定位把南极放到极地陆块附近。）"""
        p, m, rng = self.p, self.m, self.rng
        self._obstacles = ()
        G = self.G
        main0 = np.concatenate([self.posed(k, 0.0, sub=4) for k in self.docks])
        main_c = _centroid(main0)
        main_t0 = np.concatenate([self.posed(k, float(p.t0), sub=2) for k in self.docks])
        tree_t0 = cKDTree(main_t0)
        rad_main = max(_ang(main_c, x) for x in main0[::5])
        times = np.arange(p.t0 - 12.0, -0.1, -4.0)        # 开头 12 Myr 不查：300 Ma 时相邻的陆块会先分开
        main_ct = {t: Contact(np.concatenate([self.posed(k, t, sub=2) for k in self.docks]), m.h_km * 1.3)
                   for t in times}
        self.moves = {}
        placed = []                                       # (点集, 分段)
        self._placed_movers = placed                      # _transit_ok 也查它们（_late_dock）
        self._tcache = ((), {})
        order = {"polar": 0, "secondary": 1, "island": 2}
        movers = sorted([b for b, r in self.roles.items() if r in order and b not in self.docks],
                        key=lambda b: order[self.roles[b]])
        self.mover_log = {}
        for b in movers:
            role = self.roles[b]
            X = m.xyz[self.blocks[b].cont]
            Xs = X[::2]
            cb = self.blocks[b].centroid
            if role == "polar":
                targets = [G.T @ lonlat_to_xyz(lo, p.polar_lat) for lo in rng.uniform(-180, 180, 60)]
            elif role == "secondary":
                targets = self._sec_targets(G, self.main_C, lats=rng.uniform(-25, 25, 8), dlons=rng.uniform(140, 220, 8))
            else:                                         # 孤立陆块：远离主大陆的任意位置
                targets = [T for T in (unit(x) for x in rng.normal(size=(200, 3)))
                           if _ang(T, main_c) >= rad_main + np.radians(25)]
            if not targets:
                targets = [cb]
            targets.sort(key=lambda T: _ang(cb, T))
            near = targets[:max(8, len(targets) // 4)]
            # 300 Ma 时离它最近的主陆块点：绕行路线的第一段背离它（前史留下的相邻陆块可能只隔 100 km，
            # 1.2c 早期背离主大陆质心，方向不对就在出发时撞上邻居）
            dn, jn = tree_t0.query(Xs)
            nb_pt = main_t0[jn[np.argmin(dn)]]
            ok = False
            first_clash = []
            away = np.cross(nb_pt, cb)
            away = unit(away) if np.linalg.norm(away) > 1e-6 else unit(np.cross(main_c, cb))
            for _ in range(120):
                T = near[int(rng.integers(len(near)))]
                t_start = float(rng.choice([p.t0, p.t0, p.t0 - 20, p.t0 - 40, p.t0 - 80]))
                t_arr = float(5 * round(rng.uniform(60, 250) / 5))
                if t_arr >= t_start - 30:
                    continue
                spin = 0.0 if rng.random() < 0.4 else rng.uniform(-np.pi, np.pi)
                kind = rng.choice(["direct", "detour", "orbit"], p=[0.3, 0.3, 0.4])
                esc = rotmat(cb, np.radians(rng.uniform(-60, 60))) @ away     # 背离方向随机偏转 ±60°
                if kind == "direct":
                    Rs = [rotmat(T, spin) @ move_to(cb, T)]
                elif kind == "detour":                    # 先背离最近的主陆块绕出去 10–40°，再去目标
                    R1 = rotmat(esc, np.radians(rng.uniform(10, 40)))
                    Rs = [R1, rotmat(T, spin) @ move_to(R1 @ cb, T)]
                else:
                    # 背离邻居 -> 绕主大陆（最终位置的质心）转到目标所在的方位 -> 去目标：
                    # 目标常在锚定陆块的另一侧（前史的锚定陆块常是一大片），直线过去要穿过它
                    R1 = rotmat(esc, np.radians(rng.uniform(5, 25)))
                    c1 = R1 @ cb
                    u1, uT = c1 - (c1 @ main_c) * main_c, T - (T @ main_c) * main_c
                    if np.linalg.norm(u1) < 1e-6 or np.linalg.norm(uT) < 1e-6:
                        continue
                    u1, uT = unit(u1), unit(uT)
                    dphi = np.arctan2(np.cross(u1, uT) @ main_c, u1 @ uT)
                    R2 = rotmat(main_c, dphi)
                    Rs = [R1, R2, rotmat(T, spin) @ move_to(R2 @ c1, T)]
                angs = [axis_angle(R) for R in Rs]
                wts_ = np.array([max(a, np.radians(3.0)) for _, a in angs])
                cuts = t_start - np.concatenate([[0.0], np.cumsum(wts_ / wts_.sum())]) * (t_start - t_arr)
                cuts = [float(round(c)) for c in cuts]
                if any(cuts[i] <= cuts[i + 1] for i in range(len(Rs))):
                    continue
                segs = [(cuts[i], cuts[i + 1], ax, a) for i, (ax, a) in enumerate(angs)]
                if any(np.degrees(a) / (ta - tb) > p.max_rate_deg for ta, tb, _, a in segs):
                    continue
                clash = None
                for t in times:
                    Y = Xs @ dock_pose(Dock(b, t_arr, segs), t, 0.0, 0.0).T
                    if main_ct[t].hits(Y):
                        clash = t
                        break
                    for Xo, so in placed:
                        if Contact(Xo @ dock_pose(Dock(0, so[-1][1], so), t, 0.0, 0.0).T, m.h_km * 1.3).hits(Y):
                            clash = t
                            break
                    if clash is not None:
                        break
                if clash is None:
                    self.moves[b] = segs
                    placed.append((Xs, segs))
                    ok = True
                    break
                first_clash.append(float(clash))
            self.mover_log[b] = {"role": role, "ok": ok, "travel_deg": float(np.degrees(_ang(cb, near[0]))),
                                 "first_clash_Ma": sorted(first_clash, reverse=True)[:10]}
            if not ok:
                # 找不到无碰撞路线：改为有计划地拼进主大陆（增生地体，接触后锁定），免得被拼合中的主陆块碾过
                d = self._late_dock(b)
                if d is not None:
                    self.docks[b] = d
                    self.roles[b] = "main"
                    self.notes.append(f"{role} 陆块 {b} 找不到全程不碰撞的路线，改为在 {d.t_dock:.0f} Ma 拼进主大陆")
                else:
                    self.notes.append(f"{role} 陆块 {b} 找不到全程不碰撞的路线，也拼不进主大陆，留在原处缓慢漂移")
                    self.roles[b] = "island_static"

    def _late_dock(self, b: int) -> "Dock | None":
        """把一块陆块有计划地拼进主大陆（远离主内海），途中不碰别的陆块。"""
        p = self.p
        sea_c = self.inner_sea.get("centroid") if self.inner_sea else None
        for t_dock in (170.0, 150.0, 200.0, 130.0):
            cands = self._plan_dock(b, t_dock, [0, -30, 30, -60, 60, -90, 90, 120, -120, 180], [0, -40, 40],
                                    [p.penetration_km])
            for d in cands:
                if sea_c is not None:
                    Y = self.m.xyz[self.blocks[b].cont[::5]] @ dock_pose(d, 0.0, p.residual, p.residual_myr).T
                    if min(_ang(y, sea_c) for y in Y) * R_KM < 1200:
                        continue
                return d
        return None

    def validate_script(self, step: float = 4.0) -> list:
        """按写好的剧本（绝对参照系）检查所有陆块之间的接触，列出计划外的接触（陆块对、首次时刻）。
        计划内：主陆块之间在各自 dock 前 15 Myr 以后的接触、300 Ma 起 12 Myr 内的初始相邻。"""
        p, m, s = self.p, self.m, self.script
        ids = [b for b in self.blocks if b in s.stages]
        pts = {b: m.xyz[self.blocks[b].cont[::2]] for b in ids}
        found = {}
        for t in np.arange(p.t0 - 12.0, -0.1, -step):
            tt = int(round(t))
            P = {b: pts[b] @ s.forward_total(b, tt).T for b in ids}
            allp = np.concatenate([P[b] for b in ids])
            lab = np.concatenate([np.full(len(P[b]), b) for b in ids])
            tree = cKDTree(allp)
            # 陆壳相距不到 0.8 个网格间距就算接触：引擎里陆壳一重叠就按碰撞处理、吃掉下行一方的陆壳
            # （1.2c 初版只算深度穿插 0.45 格，又隔 4 点抽 1 点，途中擦碰的陆缘碰撞都漏掉了）
            r = chord_from_km(0.8 * m.h_km)
            for b in ids:
                nb = tree.query_ball_point(P[b], r)
                hit = {int(lab[j]) for lst in nb for j in lst} - {b}
                for o in hit:
                    pair = tuple(sorted((b, o)))
                    if pair in found:
                        continue
                    both_main = all(self.roles.get(x) == "main" and x in self.docks for x in pair)
                    if both_main:
                        def t_allow(x):
                            return self._t_allow(self.docks[x])
                        nonanchor = [x for x in pair if x != self.anchor]
                        t_ok = max(t_allow(x) for x in nonanchor) if nonanchor else p.t0
                        if t <= t_ok:
                            continue
                    found[pair] = float(t)
        return [{"pair": list(k), "t": v} for k, v in sorted(found.items(), key=lambda x: -x[1])]

    def plan_breakup(self):
        """局部裂解：剪刀式张开拼合缝合带（一端仍相连）或裂出微大陆。都避开主内海。"""
        p, m, rng = self.p, self.m, self.rng
        self.breakups = []
        n = int(rng.integers(p.n_breakup[0], p.n_breakup[1] + 1))
        sea_c = self.inner_sea.get("centroid") if self.inner_sea else None
        cands = [b for b in self.docks if b != self.anchor and not self.docks[b].inner_sea]
        rng.shuffle(cands)
        next_id = max(int(q) for q in np.unique(self.w.crust.plate)) + 1
        used = set()
        for _ in range(n):
            kind = "reopen" if (cands and rng.random() < 0.6) else "micro"
            t_r = float(5 * round(rng.uniform(min(p.breakup_window), max(p.breakup_window)) / 5))
            dur = float(int(rng.uniform(25, 50)))
            if kind == "reopen":
                ev = self._plan_reopen(cands, used, sea_c, t_r, dur)
            else:
                ev = self._plan_micro(used, sea_c, t_r, dur, next_id)
                if ev:
                    next_id += 1
            if ev:
                self.breakups.append(ev)

    def _plan_reopen(self, cands, used, sea_c, t_r, dur):
        m, rng = self.m, self.rng
        for b in cands:
            if b in used:
                continue
            Xb = self.posed(b, 0.0)
            others = np.concatenate([self.posed(k, 0.0) for k in self.docks if k != b])
            d, _ = cKDTree(others).query(Xb)
            cn = Xb[km_from_chord(d) < 2.0 * m.h_km]
            if len(cn) < 10:
                continue
            c = _centroid(cn)
            e1 = unit(np.cross(c, [0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.cross(c, [1.0, 0.0, 0.0]))
            e2 = np.cross(c, e1)
            uv = np.stack([cn @ e1, cn @ e2], 1)
            w_, v_ = np.linalg.eigh(np.cov(uv.T))
            s = uv @ v_[:, -1]
            ends = [cn[np.argmin(s)], cn[np.argmax(s)]]
            L = km_from_chord(np.linalg.norm(ends[0] - ends[1]))
            if L < 800:
                continue
            if sea_c is not None:
                ends.sort(key=lambda x: _ang(x, sea_c))        # 极点放在离主内海近的一端，张开处远离主内海
                if _ang(ends[1], sea_c) * R_KM < 1200:
                    continue
            P = unit(ends[0])
            dmax = km_from_chord(np.linalg.norm(Xb - P, axis=1)).max()
            theta = rng.uniform(300, 800) / dmax                 # 陆块上离极点最远处的位移 300–800 km（红海式）
            far = []
            for sgn in (1, -1):
                Y = Xb @ rotmat(P, sgn * theta).T
                dd = km_from_chord(cKDTree(others).query(Y)[0])
                far.append(np.percentile(dd, 10))
            sgn = 1 if far[0] >= far[1] else -1
            # 张开后主大陆在陆地上仍要连通：挂在这块上、只靠它连着主体的陆块会被一起拉断
            # （世界 1：陆块 1 只靠陆块 2 连着主体，陆块 2 沿缝合带张开后陆块 1 成了孤岛）
            if not self._still_connected(b, rotmat(P, sgn * theta)):
                continue
            used.add(b)
            return {"kind": "reopen", "block": b, "t": t_r, "dur": dur, "pole": P, "angle": sgn * theta,
                    "contact_km": float(L)}
        return None

    def _still_connected(self, b: int, Rb: np.ndarray) -> bool:
        """0 Ma 时第 b 块再转 Rb（锚定参照系）后，已拼合各块的陆地是否仍连成一片（陆地相距 < 1.5 格算相连）。"""
        m = self.m
        ids = list(self.docks)
        P = {k: self.posed(k, 0.0, "dock") @ (Rb.T if k == b else np.eye(3)) for k in ids}
        trees = {k: cKDTree(P[k]) for k in ids}
        r = chord_from_km(1.5 * m.h_km)
        adj = {k: set() for k in ids}
        for i, a in enumerate(ids):
            for c in ids[i + 1:]:
                if np.isfinite(trees[c].query(P[a], distance_upper_bound=r)[0]).any():
                    adj[a].add(c)
                    adj[c].add(a)
        seen, stack = {ids[0]}, [ids[0]]
        while stack:
            for c in adj[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return len(seen) == len(ids)

    def _plan_micro(self, used, sea_c, t_r, dur, new_id):
        m, rng, w = self.m, self.rng, self.pw
        v = w.crust
        provs = []
        for b in self.docks:
            if b in used:
                continue
            R = self.pose(b, 0.0)
            others = np.concatenate([self.posed(k, 0.0) for k in self.docks if k != b])
            tree_o = cKDTree(others)
            cl = self.blocks[b].cont
            cl = cl[v.plate[cl] == b]                         # 编组陆块只从领头板块上裂（引擎按板块拆分）
            for pid in np.unique(v.prov[cl]):
                if pid <= 0:
                    continue
                idx = cl[v.prov[cl] == pid]
                a = m.area_km2[idx].sum()
                if not (3e5 <= a <= 3e6):
                    continue
                Y = m.xyz[idx] @ R.T
                if km_from_chord(tree_o.query(Y)[0]).min() < 600:
                    continue                                  # 挨着别的陆块（在大陆内部）
                c = _centroid(Y)
                if sea_c is not None and _ang(c, sea_c) * R_KM < 1500:
                    continue
                # 必须在本陆块的外缘：省份到本陆块以外海域的距离
                own = m.xyz[self.blocks[b].cont] @ R.T
                cb = _centroid(own)
                if _ang(c, cb) < 0.5 * max(_ang(cb, x) for x in own[::9]):
                    continue
                provs.append((b, int(pid), c, cb))
        if not provs:
            return None
        b, pid, c, cb = provs[int(rng.integers(len(provs)))]
        used.add(b)
        scissor = bool(rng.random() < 0.5)
        axis = unit(np.cross(cb, c))
        rate = float(rng.uniform(0.25, 0.5))
        return {"kind": "micro", "block": b, "province": pid, "new_plate": new_id, "t": t_r, "dur": dur,
                "axis": axis, "rate_deg": rate, "scissor": scissor}

    # ---- 剧本 -----------------------------------------------------------------------------------
    def build_script(self) -> StageScript:
        p = self.p
        t0 = p.t0
        s = StageScript(t0)
        plates = sorted(int(q) for q in np.unique(self.w.crust.plate))
        role = {q: self.roles.get(q, "ocean") for q in plates}
        s.add_plate(self.anchor, 0, f"anchor block {self.anchor}")
        s.set_absolute(self.anchor, self.G_axis * self.G_angle / t0, t0, 0)
        for q in plates:
            if q == self.anchor or q in self.follower_of:
                continue
            r = role[q]
            if r == "main" and q in self.docks:
                s.add_plate(q, self.anchor, f"main block {q}")
                d = self.docks[q]
                segs = [(ta, tb, ax * ang / (ta - tb)) for ta, tb, ax, ang in d.segs]
                tr = max(d.t_dock - p.residual_myr, 0)
                if d.t_dock > tr:
                    ra, rb, rax, rang = d.residual_seg()
                    segs.append((d.t_dock, tr, rax * rang / (ra - rb) * p.residual))
                ev = [e for e in self.breakups if e["kind"] == "reopen" and e["block"] == q]
                cur = tr
                for e in sorted(ev, key=lambda e: -e["t"]):
                    tb, te = min(e["t"], cur), max(min(e["t"], cur) - e["dur"], 0)
                    if cur > tb:
                        segs.append((cur, tb, np.zeros(3)))
                    segs.append((tb, te, e["pole"] * e["angle"] / max(tb - te, 1)))
                    cur = te
                if cur > 0:
                    segs.append((cur, 0, np.zeros(3)))
                for a, b_, wv in segs:
                    a, b_ = int(round(a)), int(round(b_))
                    if a > b_:
                        s.set_relative(q, self.anchor, wv, a, b_, frame_t=t0)
            elif q in self.moves:
                s.add_plate(q, self.anchor, f"{r} block {q}")
                cur = t0
                for ta, tb, ax, ang in self.moves[q]:
                    ta, tb = int(round(ta)), int(round(tb))
                    if cur > ta:
                        s.set_relative(q, self.anchor, np.zeros(3), cur, ta, frame_t=t0)
                    s.set_relative(q, self.anchor, ax * ang / (ta - tb), ta, tb, frame_t=t0)
                    cur = tb
                if cur > 0:
                    s.set_relative(q, self.anchor, np.zeros(3), cur, 0, frame_t=t0)
            elif q in self.blocks:
                # 没有安排去处的陆块：跟着锚定陆块，只有很慢的相对漂移（绝对参照系里不动的话，整体定位中的主大陆会碾过它）
                s.add_plate(q, self.anchor, f"{r} block {q}")
                s.set_relative(q, self.anchor, unit(self.rng.normal(size=3)) * self.rng.uniform(0.02, 0.06) * DEG, t0, 0,
                               frame_t=t0)
            else:
                s.add_plate(q, 0, f"{r} plate {q}")
                s.set_absolute(q, unit(self.rng.normal(size=3)) * self.rng.uniform(0.03, 0.12) * DEG, t0, 0)
        for q, lead in sorted(self.follower_of.items()):          # 编组陆块：与领头陆块一起运动
            s.add_plate(q, lead, f"block {q} (moves with block {lead})")
            s.set_relative(q, lead, np.zeros(3), t0, 0, frame_t=t0)
        for e in self.breakups:
            if e["kind"] != "micro":
                continue
            q, par = e["new_plate"], e["block"]
            s.add_plate(q, par, f"microcontinent from block {par}")
            tr, te = int(round(e["t"])), int(round(max(e["t"] - e["dur"], 0)))
            s.follow(q, par, t0, tr)
            # 轴在锚定参照系（= 父陆块锁定后的参照系）中给出，换成父陆块 300 Ma 参照系
            Rpar = self.pose(par, 0.0)
            w = Rpar.T @ e["axis"] * e["rate_deg"] * DEG
            s.set_relative(q, par, w, tr, te, frame_t=t0)
            if te > 0:
                s.set_relative(q, par, np.zeros(3), te, 0, frame_t=t0)
        self.script = s
        return s

    def plan(self, attempts: int = 4, only_attempt: int | None = None, inner_dgap_km: float = 0.0,
             inner_min_aspect: float = 0.0) -> Plan:
        """规划；有计划外接触、没找到主内海或极地大陆、圈出内洋时换一组随机抽样重来，取评分最好的一次。
        闭环修正（run_scenario_loop）时只重做选中的那一次（only_attempt，随机抽样相同），并把主内海那块远端海峡收窄
        inner_dgap_km（负值放宽），或按 inner_min_aspect 提高主内海的长宽比要求、重选它的拼法。"""
        best = None
        self.inner_dgap_km = inner_dgap_km
        self.inner_min_aspect = inner_min_aspect
        for k in ([only_attempt] if only_attempt is not None else range(attempts)):
            self.rng = np.random.default_rng(10_000 + 97 * self.seed + k)
            self.notes = []
            self._tcache = ((), {})
            self._placed_movers = []
            plan = self._plan_once()
            plan.attempt = k
            ocean = self.max_enclosed > self.p.inner_ocean_km2
            score = (10 * len(self.conflicts) + (0 if self.inner_sea.get("found") else 3)
                     + (0 if "polar" in self.roles.values() else 2) + round(self.placement["penalty"] / 5.0, 1)
                     + (5 if ocean else 0))
            plan.notes.append(f"第 {k + 1} 次规划，评分 {score}（计划外接触 ×10、无主内海 +3、无极地大陆 +2、"
                              f"纬度偏差每 5° +1、圈出内洋 +5）")
            plan.score = score
            if best is None or score < best[0]:
                best = (score, plan)
            if score == 0:
                break
        return best[1]

    def _plan_once(self) -> Plan:
        self.assign_roles()
        self.plan_assembly()
        self.plan_placement()
        self.plan_movers()
        self.plan_breakup()
        self.build_script()
        self.conflicts = self.validate_script()
        for c in self.conflicts:
            self.notes.append(f"计划外接触：陆块 {c['pair'][0]} 与 {c['pair'][1]}，约 {c['t']:.0f} Ma 起")
        self.max_enclosed = self._max_enclosed_final()
        if self.max_enclosed > self.p.inner_ocean_km2:
            self.notes.append(f"主大陆圈出内洋 {self.max_enclosed / 1e6:.1f} Mkm²（规划预测）")
        self.placement["max_enclosed_km2"] = float(self.max_enclosed)
        return Plan(self.seed, self.anchor, dict(self.roles), dict(self.docks), self.G_axis, self.G_angle,
                    dict(self.moves), list(self.breakups), dict(self.inner_sea), self.script, list(self.notes),
                    placement=dict(self.placement), conflicts=list(self.conflicts))


# ---- 运行 ------------------------------------------------------------------------------------------
@dataclass
class ScenarioRun:
    plan: Plan
    engine: Engine
    snapshots: list
    rot_path: str
    wall_s: float
    planner: "Planner | None" = None
    loop: list = field(default_factory=list)          # 主内海闭环每一轮的记录（run_scenario_loop）
    metrics: dict | None = None                       # 0 Ma 指标（闭环时已算好）


def run_scenario(mesh: Mesh, world: InitialWorld, seed: int, params: ScenarioParams | None = None,
                 snap_every: int = 10, work_dir: Path | None = None) -> ScenarioRun:
    """规划一次、完整模拟一次（不做主内海闭环修正，见 run_scenario_loop）。"""
    import time

    t_start = time.perf_counter()
    planner = Planner(mesh, world, seed, params)
    plan = planner.plan()
    run = simulate_plan(mesh, world, planner, plan, seed, snap_every, work_dir)
    run.wall_s = time.perf_counter() - t_start
    return run


def evaluate_0ma(mesh: Mesh, run: ScenarioRun):
    """0 Ma 的 A 层与字段（1024x512，阶段 0 的指标代码）。"""
    from mythic.metrics.compute import compute_all, load_config

    raster = Raster(mesh, 512)
    E = raster(run.snapshots[-1]["elev"])
    m, fields = compute_all(E, E > 0, cfg=load_config(), return_fields=True)
    return m, fields, raster, E


def locate_planned_sea(plan: Plan, A: dict, fields: dict, raster: Raster, E: np.ndarray, radius_km: float = 400.0) -> dict:
    """在 0 Ma 的 A 层海域划分里找规划的残留洋盆（规划质心 radius_km 内海洋格点最多的海域），判断它的状态：
    open = 并进了外洋或海峡多于 3 条（口子太宽）；closed = 0 条海峡或已被陆地填满（口子封死）；
    found = 是独立的半封闭海域（是否通过 A 层另看）。"""
    c = plan.inner_sea.get("centroid") if plan.inner_sea else None
    if c is None:
        return {"state": "none"}
    g = raster.grid
    x = rotmat(plan.G_axis, plan.G_angle) @ np.asarray(c, float)
    near = (g.xyz @ x) > np.cos(radius_km / R_KM)
    ocean = near & (E <= 0)
    if not ocean.any():
        return {"state": "closed", "why": "规划位置附近已无海洋"}
    reg = fields["sea_regions"][ocean]
    vals, cnt = np.unique(reg, return_counts=True)
    rid = int(vals[np.argmax(cnt)])
    cands = {s["region_id"]: s for s in A["main_inland_sea"].get("candidate_seas", [])}
    main_id = A["main_inland_sea"].get("main", {}).get("region_id")
    if rid not in cands:
        return {"state": "open", "why": "规划的海并进了外洋", "region_id": rid}
    s = cands[rid]
    st = s["straits_to_world_ocean"]
    state = "closed" if st == 0 else "open" if st > 3 else "found"
    return {"state": state, "region_id": rid, "is_main": rid == main_id, "straits": st,
            "area_km2": s["area_km2"], "main_border": s["main_continent_border_fraction"]}


def run_scenario_loop(mesh: Mesh, world: InitialWorld, seed: int, params: ScenarioParams | None = None,
                      max_rounds: int = 4, snap_every: int = 10, work_dir: Path | None = None) -> ScenarioRun:
    """主内海闭环：规划 -> 完整模拟 -> 0 Ma 按 A 层核对 -> 修正 -> 重新规划、模拟（确定性），最多 max_rounds 次模拟。
    两种修正：
    - 海峡：口子太宽（并进外洋、海峡多于 3 条、面积超上限）就收窄主内海那块的远端海峡，被封死（0 条海峡、面积不到下限）就放宽，
      二分夹逼（首次收窄 120 km、放宽 80 km）；
    - 形状：海峡与面积都合格、规划的海就是 A 层主内海、只差长宽比时，按模拟里长宽比缩水的比例提高规划时的要求
      （inner_min_aspect = 2.1 × 规划长宽比 / 模拟长宽比），重选主内海那块的拼法，海峡宽度从头调。
      （1.2c 第三步世界 8：规划时 2.23，模拟后 1.99，碰撞带增厚与岛弧把海挤短挤宽；调海峡改不了形状。）
    其他原因（接壤、第二大水体太大等）停止。返回主内海最好的一次（通过 A 层的优先），run.loop 记录每一轮。"""
    import time

    t_start = time.perf_counter()
    planner = Planner(mesh, world, seed, params)
    plan = planner.plan()
    dgap, too_open, too_closed, min_asp = 0.0, None, None, 0.0
    history, best = [], None
    for rnd in range(max_rounds):
        run = simulate_plan(mesh, world, planner, plan, seed, snap_every, work_dir)
        m, fields, raster, E = evaluate_0ma(mesh, run)
        A = m["A"]
        isea = A["main_inland_sea"]
        loc = locate_planned_sea(plan, A, fields, raster, E)
        n_fail = sum(1 for v in A.values() if not v["pass"])
        rec = {"round": rnd, "dgap_km": dgap, "min_aspect": min_asp, "planned_sea": loc,
               "plan_aspect": (plan.inner_sea.get("official") or {}).get("aspect"),
               "sim_aspect": (isea.get("geometry") or {}).get("aspect_ratio"), "inner_sea_pass": bool(isea["pass"]),
               "inner_sea_checks": isea.get("checks"), "A_fail": [k for k, v in A.items() if not v["pass"]],
               "plan_score": plan.score}
        history.append(rec)
        key = (0 if isea["pass"] else 1, n_fail)
        if best is None or key < best[0]:
            best = (key, run, m)
        if isea["pass"] or rnd == max_rounds - 1:
            break
        d = plan.docks.get(next((b for b, dk in plan.docks.items() if dk.inner_sea), -1))
        # 海找到了但面积不在目标范围：太小就放宽远端海峡（少转一点，两岸张得更开），太大就收窄
        lo_a, hi_a = planner.p.inner_sea_area_km2
        direction = loc["state"]
        if direction == "found" and loc.get("area_km2", 0.0) < lo_a:
            direction = "closed"
        elif direction == "found" and loc.get("area_km2", 0.0) > hi_a:
            direction = "open"
        chk = isea.get("checks") or {}
        if direction == "found" and loc.get("is_main") and not chk.get("elongation_ok", True) \
                and chk.get("straits_ok") and chk.get("area_in_range") and rec["plan_aspect"] and rec["sim_aspect"]:
            direction = "reshape"
        rec["direction"] = direction
        if direction == "reshape":
            new = max(min_asp + 0.3, 2.1 * rec["plan_aspect"] / max(rec["sim_aspect"], 0.5))
            if new > 6.0:
                rec["stop"] = f"要求的规划长宽比 {new:.1f} 太高"
                break
            min_asp, dgap, too_open, too_closed = float(new), 0.0, None, None
            planner = Planner(mesh, world, seed, params)
            plan = planner.plan(only_attempt=plan.attempt, inner_dgap_km=0.0, inner_min_aspect=min_asp)
            if not plan.inner_sea.get("found"):
                rec["stop"] = f"长宽比要求提高到 {min_asp:.2f} 后选不出主内海"
                break
            continue
        if d is None or not getattr(d, "_L2", None) or direction not in ("open", "closed"):
            rec["stop"] = "主内海那块不是两段式拼合" if d is None or not getattr(d, "_L2", None) \
                else f"规划的海状态 {loc['state']}：调海峡解决不了"
            break
        if direction == "open":
            too_open = dgap if too_open is None else max(too_open, dgap)
            dgap = 0.5 * (dgap + too_closed) if too_closed is not None else dgap + 120.0
        else:
            too_closed = dgap if too_closed is None else min(too_closed, dgap)
            dgap = 0.5 * (dgap + too_open) if too_open is not None else dgap - 80.0
        planner = Planner(mesh, world, seed, params)
        plan = planner.plan(only_attempt=plan.attempt, inner_dgap_km=dgap, inner_min_aspect=min_asp)
    _, run, m = best
    run.loop = history
    run.metrics = m
    run.wall_s = time.perf_counter() - t_start
    return run


def simulate_plan(mesh: Mesh, world: InitialWorld, planner: Planner, plan: Plan, seed: int, snap_every: int = 10,
                  work_dir: Path | None = None) -> ScenarioRun:
    """按规划写旋转文件，引擎完整模拟 300 -> 0 Ma。"""
    import time

    t_start = time.perf_counter()
    p = planner.p
    work = Path(work_dir or (CACHE / "scenario"))
    rot = plan.script.write_rot(work / f"scenario_L{mesh.level}_w{world.seed}_s{seed}.rot")
    kin = RotationFileKinematics(rot)
    c = world.crust.gather(np.arange(len(world.crust)))
    c.uid = np.arange(len(c), dtype=np.int64)
    c.node = np.arange(len(c), dtype=np.int64)
    plates0 = sorted(int(q) for q in np.unique(c.plate))
    _, om = kin.step_rotations(plates0, p.t0)
    eng = Engine(mesh, c, Params(), float(p.t0), om)
    snaps = [eng.snapshot()]
    micro = {int(round(e["t"])): e for e in plan.breakups if e["kind"] == "micro"}
    rng = np.random.default_rng(20_000 + seed)
    t = p.t0
    while t > 0:
        dR, om = kin.step_rotations(eng.plates, t)
        committed = eng.step(dR, om)
        t -= 1
        if committed:
            e = micro.get(t)
            if e and e["block"] in eng.plates:
                mask = (eng.c.prov == e["province"]) & (eng.c.plate == e["block"]) & eng.alive
                if mask.sum() > 20:
                    om4 = kin.step_rotations([e["new_plate"]], t)[1][e["new_plate"]]
                    ax_abs = unit(om4 - om[e["block"]]) if np.linalg.norm(om4 - om[e["block"]]) > 0 else np.array([0, 0, 1.0])
                    eng.split_plate(e["block"], None, None, ax_abs, e["new_plate"], rng, new_mask=mask, omega_new=om4,
                                    margin_km=150.0)
            if t % snap_every == 0:
                snaps.append(eng.snapshot())
    return ScenarioRun(plan, eng, snaps, str(rot), time.perf_counter() - t_start, planner)
