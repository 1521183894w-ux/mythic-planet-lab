"""阶段 1.1 两条路线共用的地壳引擎（第二版表示）。

两条路线（route_a.py 涌现型、route_b.py pyGPlates 剧本型）共用这里的网格、地壳状态与相互作用规则，
区别只在每一步的板块旋转从哪里来。这样对比图上的差别只来自"运动学怎么决定"，而不是两套规则写得不一样。

表示方式：
- 一套固定的二十面体网格（stripy）。每块板块都有一份"自己的"这套网格（板块参照系），地壳以"地块单元"（parcel）存放：
  每个单元属于一块板块、对应该板块参照系中的一个网格节点，当前位置 = R_p · 节点坐标，R_p 是该板块自诞生以来的累计旋转。
- **板块内部永远不重新取样**，所以没有数值扩散，慢速运动也不会被取整吞掉。
  （第一版每 5 Myr 把所有板块重新取样到固定网格，位移小于半个网格间距的运动会被整个吞掉——L6 上低于约 1.2 cm/yr
  的相对运动永远不动，见 README "表示方式的教训"。）
- 判断"单元 u 是否进入了板块 q 的范围"：把 u 的当前位置用 R_q 转回 q 的参照系，看最近的节点上是否有 q 的存活单元。
- 每 5 Myr "提交"一次：删除已俯冲消耗的单元；固定网格上没有任何板块覆盖的节点（扩张造成的空隙）生成新洋壳，
  归给最近的板块，放在该板块参照系里最近的节点上（只有这里和地体转移时有一次不超过半个网格间距的取整），
  洋壳年龄按到两侧板块边缘的距离插值。

规则与 method_class（CLAUDE.md 第 4 节）：
- 洋壳水深—年龄：GDH1（Stein & Stein 1992），parameterized_physics；
- 地壳均衡：Airy（ρc 2800、ρm 3300 kg/m³），parameterized_physics；
- 俯冲带上盘增厚（岛弧、安第斯型山脉）、碰撞增厚核函数、造山带衰减、地壳流动扩散、海沟深度：经验形状函数，
  procedural_constrained；
- 裂谷：洋脊—转换断层阶梯（洋脊段沿相对欧拉极的经线、转换断层沿纬线），段长与错距取随机值，procedural_constrained。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from common import KD_WORKERS, R_KM, Mesh, chord_from_km, km_from_chord, smoothstep, unit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

OCEAN, CONT = 0, 1
ORO_NONE, ORO_ARC, ORO_COLL, ORO_RIFT = 0, 1, 2, 3
ORIGIN_OCEAN, ORIGIN_ARC = 0, 90


@dataclass
class Params:
    dt: float = 1.0                    # Myr
    commit_every: int = 5              # 每 5 步提交一次（删除消耗单元、填空隙）
    T_ocean: float = 7.0               # 新洋壳厚度 km
    T_cont: float = 35.0               # 正常陆壳厚度 km
    e_cont_ref_m: float = 400.0        # 35 km 陆壳的高程（大陆"干舷"）
    iso_m_per_km: float = 151.5        # Airy：每增厚 1 km 抬升 (1 - ρc/ρm) km
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
    trench_depth_m: float = 2500.0     # 俯冲板块在前缘处的附加下沉（显示用）
    trench_width_km: float = 70.0


def gdh1_depth_m(age_myr):
    """GDH1 板块冷却模型的洋底水深（Stein & Stein 1992, Nature 359:123）。"""
    a = np.maximum(np.asarray(age_myr, float), 0.0)
    return np.where(a < 20.0, 2600.0 + 365.0 * np.sqrt(a), 5651.0 - 2473.0 * np.exp(-0.0278 * a))


class Crust:
    """地块单元表（每行一个单元）。node 是该单元在所属板块参照系中的网格节点编号。
    snapshot()/view() 返回的"固定网格视图"也用这个类，此时 node = 固定网格节点编号。"""
    FIELDS = ("plate", "node", "ctype", "T", "age", "origin", "oro", "oro_t")

    def __init__(self, n: int):
        self.plate = np.zeros(n, np.int32)
        self.node = np.arange(n, dtype=np.int64)
        self.ctype = np.zeros(n, np.int8)
        self.T = np.full(n, 7.0)
        self.age = np.zeros(n)
        self.origin = np.zeros(n, np.int16)   # 地体来源：初始大陆编号；90 = 洋内弧增生；0 = 洋壳
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


def arc_profile(d_km, peak_km, max_km):
    up = smoothstep(d_km / peak_km)
    down = 1.0 - smoothstep((d_km - peak_km) / (max_km - peak_km))
    return np.where(d_km < peak_km, up, down)


def graph_components(n: int, edges: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def orthonormalize(rm: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rm)
    return u @ vt


class Engine:
    def __init__(self, mesh: Mesh, crust: Crust, params: Params, t0: float, omega: dict):
        self.m, self.c, self.p = mesh, crust, params
        self.t = float(t0)
        self.alive = np.ones(len(crust), bool)
        self.R = {int(q): np.eye(3) for q in np.unique(crust.plate)}
        self.omega = {q: np.asarray(omega[q], float) for q in self.R}
        self.steps_since_commit = 0
        self.polarity: dict[int, int] = {}     # 板块对 -> 俯冲（下行）板块
        self.fronts: deque = deque()
        self.coll_area: dict[tuple[int, int], float] = {}
        self.events: list[dict] = []
        self.budget = {"arc_added": 0.0, "eroded": 0.0, "rift_thinning": 0.0, "arc_converted": 0.0}
        self.step_log: list[dict] = []
        self.commit_log: list[dict] = []
        self._rebuild()
        self._log_commit(n_gap=0, n_new=0, wall=0.0)

    # ---- 基本量 ------------------------------------------------------------------------
    @property
    def plates(self) -> list[int]:
        return sorted(self.R)

    def _rebuild(self):
        """单元增删或改属后重建：各板块"节点 -> 单元"索引、板块内单元邻接、单元面积。"""
        m, c = self.m, self.c
        self.area = m.area_km2[c.node]
        self.owner = {}
        pe = []
        for q in self.plates:
            o = np.full(m.n, -1, np.int64)
            idx = np.flatnonzero((c.plate == q) & self.alive)
            o[c.node[idx]] = idx
            self.owner[q] = o
            a, b = o[m.edges[:, 0]], o[m.edges[:, 1]]
            ok = (a >= 0) & (b >= 0)
            pe.append(np.stack([a[ok], b[ok]], 1))
        self.pedges = np.concatenate(pe) if pe else np.zeros((0, 2), np.int64)
        cont = (c.ctype == CONT) & self.alive
        self.cont_area = {q: float(self.area[cont & (c.plate == q)].sum()) for q in self.plates}

    def positions(self) -> np.ndarray:
        stack = np.zeros((max(self.R) + 1, 3, 3))
        for q, rm in self.R.items():
            stack[q] = rm
        return np.einsum("nij,nj->ni", stack[self.c.plate], self.m.xyz[self.c.node])

    def covering(self, q: int, pts: np.ndarray) -> np.ndarray:
        """pts（当前绝对位置）落在板块 q 的哪个存活单元里；不在 q 范围内为 -1。"""
        _, w = self.m.tree.query(pts @ self.R[q], workers=KD_WORKERS)
        par = self.owner[q][w]
        ok = par >= 0
        ok[ok] = self.alive[par[ok]]
        return np.where(ok, par, -1)

    def plate_centroids(self) -> dict:
        pos = self.positions()
        out = {}
        for q in self.plates:
            s = self.alive & (self.c.plate == q)
            out[q] = unit((pos[s] * self.area[s, None]).sum(0))
        return out

    def _event(self, kind: str, **info):
        self.events.append({"t": round(self.t, 3), "kind": kind, **info})

    def cont_volume(self) -> float:
        cont = self.alive & (self.c.ctype == CONT)
        return float((self.c.T[cont] * self.area[cont]).sum())

    # ---- 一步 --------------------------------------------------------------------------
    def step(self, dR: dict, omega: dict) -> bool:
        """推进 dt。dR：每块板块在这一步的有限旋转（3x3）；omega：这一步的角速度矢量（rad/Myr，用于相对速度）。
        返回这一步之后是否做了提交。"""
        t0 = time.perf_counter()
        p = self.p
        for q in self.plates:
            self.R[q] = dR[q] @ self.R[q]
        self.omega = {q: np.asarray(omega[q], float) for q in self.plates}
        self.t -= p.dt
        pos = self.positions()
        n_sub, n_coll = self._interact(pos)
        self._arc(pos)
        self._surface(p.dt)
        ocean = self.alive & (self.c.ctype == OCEAN)
        self.c.age[ocean] += p.dt
        self.steps_since_commit += 1
        committed = False
        if self.steps_since_commit >= p.commit_every:
            self.commit()
            committed = True
        self.step_log.append({"t": self.t, "n_subducted": n_sub, "n_collision": n_coll,
                              "wall_s": time.perf_counter() - t0})
        return committed

    # ---- 板块相互作用 --------------------------------------------------------------------
    def _interact(self, pos: np.ndarray) -> tuple[int, int]:
        c, p = self.c, self.p
        M = len(c)
        alive_idx = np.flatnonzero(self.alive)
        inside = np.full(M, -1, np.int32)
        partner = np.full(M, -1, np.int64)
        for q in self.plates:
            U = alive_idx[c.plate[alive_idx] != q]
            if U.size == 0:
                continue
            par = self.covering(q, pos[U])
            hit = (par >= 0) & (inside[U] < 0)
            inside[U[hit]] = q
            partner[U[hit]] = par[hit]
        U = np.flatnonzero(inside >= 0)
        if U.size == 0:
            return 0, 0
        P, Q, B = c.plate[U], inside[U], partner[U]
        cu, cb = c.ctype[U], c.ctype[B]
        key = np.minimum(P, Q) * 1000 + np.maximum(P, Q)

        # 俯冲极性：每个板块对首次接触时决定一次，之后保持（避免沿同一边界来回翻转）
        for k in np.unique(key):
            if int(k) in self.polarity:
                continue
            s = key == k
            oc_under = (cu[s] == OCEAN) & (cb[s] == CONT)
            co_over = (cu[s] == CONT) & (cb[s] == OCEAN)
            oo = (cu[s] == OCEAN) & (cb[s] == OCEAN)
            older = np.where(c.age[U[s]] >= c.age[B[s]], P[s], Q[s])
            lo, hi = int(k) // 1000, int(k) % 1000
            xm = unit(pos[U[s]].mean(0))
            # 洋—洋接触：若只有一方在接触处 400 km 内有大陆（被动陆缘外窄洋壳），那一方作上盘（俯冲在陆缘处开始），
            # 否则较老的一方俯冲
            if oo.any():
                near = {}
                for q in (lo, hi):
                    cq = np.flatnonzero(self.alive & (c.plate == q) & (c.ctype == CONT))
                    near[q] = bool(cq.size) and bool(
                        (km_from_chord(np.linalg.norm(pos[cq] - xm, axis=1)) < 400.0).any())
                if near[lo] != near[hi]:
                    older = np.where(oo, hi if near[lo] else lo, older)
            votes = np.concatenate([P[s][oc_under], Q[s][co_over], older[oo]])
            if votes.size:
                vals, cnt = np.unique(votes, return_counts=True)
                sub = int(vals[np.argmax(cnt)])
                kind = "subduction_start"
            else:  # 两块大陆直接相遇：大陆面积小的一方下行（"印度"）
                sub = lo if self.cont_area.get(lo, 0) <= self.cont_area.get(hi, 0) else hi
                kind = "continent_contact"
            self.polarity[int(k)] = sub
            self._event(kind, plates=[lo, hi], subducting=sub,
                        lat=round(float(np.degrees(np.arcsin(xm[2]))), 2),
                        lon=round(float(np.degrees(np.arctan2(xm[1], xm[0]))), 2))
        sub_plate = np.array([self.polarity[int(k)] for k in key])

        consume = ((cu == OCEAN) & (cb == CONT)) | ((cu == OCEAN) & (cb == OCEAN) & (sub_plate == P))
        coll = (cu == CONT) & (cb == CONT) & (sub_plate == P)

        if consume.any():
            idx = U[consume]
            x = pos[idx]
            op = np.stack([self.omega[int(q)] for q in P[consume]])
            oq = np.stack([self.omega[int(q)] for q in Q[consume]])
            v = np.linalg.norm(np.cross(op - oq, x), axis=1) * R_KM / 10.0
            self.fronts.append({"t": self.t, "x": x, "sub": P[consume], "over": Q[consume], "v": v})
            self.alive[idx] = False

        if coll.any():
            r = p.collision_radius_km
            add = np.zeros(M)
            for o in np.unique(Q[coll]):
                sel = coll & (Q == o)
                src = U[sel]
                tgt = np.flatnonzero(self.alive & (c.plate == o) & (c.ctype == CONT))
                if tgt.size:
                    tree = cKDTree(pos[tgt])
                    for s_node, lst in zip(src, tree.query_ball_point(pos[src], chord_from_km(r), workers=KD_WORKERS)):
                        if not lst:
                            continue
                        j = tgt[lst]
                        d = km_from_chord(np.linalg.norm(pos[j] - pos[s_node], axis=1))
                        w = (1.0 - (d / r) ** 2) ** 2
                        add[j] += c.T[s_node] * self.area[s_node] * w / (w * self.area[j]).sum()
                for d_plate in np.unique(P[sel]):
                    pair = (int(d_plate), int(o))
                    first = pair not in self.coll_area
                    a = float(self.area[src[P[sel] == d_plate]].sum())
                    self.coll_area[pair] = self.coll_area.get(pair, 0.0) + a
                    if first:
                        xm = unit(pos[src[P[sel] == d_plate]].mean(0))
                        self._event("collision_onset", downgoing=pair[0], overriding=pair[1],
                                    lat=round(float(np.degrees(np.arcsin(xm[2]))), 2),
                                    lon=round(float(np.degrees(np.arctan2(xm[1], xm[0]))), 2))
                self.alive[src] = False
            c.T += add
            mark = add > 0.05
            c.oro[mark] = ORO_COLL
            c.oro_t[mark] = self.t
        return int(consume.sum()), int(coll.sum())

    # ---- 俯冲上盘增厚（岛弧、安第斯型山脉）与海沟 ---------------------------------------
    def _recent_fronts(self):
        p = self.p
        while self.fronts and self.fronts[0]["t"] - self.t > p.front_memory_myr + 1e-9:
            self.fronts.popleft()
        if not self.fronts:
            return None
        f = self.fronts
        return (np.concatenate([a["x"] for a in f]), np.concatenate([a["sub"] for a in f]),
                np.concatenate([a["over"] for a in f]), np.concatenate([a["v"] for a in f]))

    def _arc(self, pos: np.ndarray):
        c, p = self.c, self.p
        fr = self._recent_fronts()
        if fr is None:
            return
        X, S, O, V = fr
        for o in np.unique(O):
            if o not in self.R:
                continue
            sel = O == o
            tgt = np.flatnonzero(self.alive & (c.plate == o))
            d, k = cKDTree(X[sel]).query(pos[tgt], distance_upper_bound=chord_from_km(p.arc_max_km),
                                         workers=KD_WORKERS)
            ok = np.isfinite(d)
            j = tgt[ok]
            f = arc_profile(km_from_chord(d[ok]), p.arc_peak_km, p.arc_max_km)
            dT = p.arc_rate_km_myr * f * np.clip(V[sel][k[ok]] / p.v0_cm_yr, 0.0, 1.5) * p.dt
            c.T[j] += dT
            self.budget["arc_added"] += float((dT * self.area[j])[c.ctype[j] == CONT].sum())
            strong = j[f > 0.3]
            c.oro[strong] = np.where(c.oro[strong] == ORO_COLL, ORO_COLL, ORO_ARC)
            c.oro_t[strong] = self.t
            conv = j[(c.ctype[j] == OCEAN) & (c.T[j] >= p.arc_to_cont_T_km)]
            if conv.size:
                c.ctype[conv] = CONT
                c.origin[conv] = ORIGIN_ARC
                self.budget["arc_converted"] += float((c.T[conv] * self.area[conv]).sum())

    def trench_now(self, pos: np.ndarray | None = None) -> np.ndarray:
        """当前位置下的海沟附加深度（m，显示用，每个单元一个值）：俯冲板块上距最近 3 Myr 俯冲前缘越近越深。"""
        c, p = self.c, self.p
        pos = self.positions() if pos is None else pos
        out = np.zeros(len(c))
        fr = self._recent_fronts()
        if fr is None:
            return out
        X, S, _, _ = fr
        for s in np.unique(S):
            tgt = np.flatnonzero(self.alive & (c.plate == s) & (c.ctype == OCEAN))
            if tgt.size == 0:
                continue
            d, _ = cKDTree(X[S == s]).query(pos[tgt], distance_upper_bound=chord_from_km(4 * p.trench_width_km),
                                            workers=KD_WORKERS)
            ok = np.isfinite(d)
            out[tgt[ok]] = p.trench_depth_m * np.exp(-km_from_chord(d[ok]) / p.trench_width_km)
        return out

    # ---- 造山带衰减与地壳流动 -------------------------------------------------------------
    def _surface(self, dt: float):
        c, p = self.c, self.p
        cont = self.alive & (c.ctype == CONT)
        ex = np.where(cont, np.maximum(c.T - p.T_cont, 0.0), 0.0)
        loss = ex * dt / p.tau_oro_myr
        c.T -= loss
        self.budget["eroded"] += float((loss * self.area).sum())
        i, j = self.pedges[:, 0], self.pedges[:, 1]
        ok = cont[i] & cont[j] & (np.maximum(c.T[i], c.T[j]) > p.flow_T_km)
        if ok.any():
            i, j = i[ok], j[ok]
            coef = p.kappa_km2_myr * dt * 2.0 / (3.0 * self.m.h_km**2)
            flux = coef * (c.T[j] - c.T[i]) * 0.5 * (self.area[i] + self.area[j])   # 体积通量，守恒
            dv = np.bincount(i, flux, minlength=len(c)) - np.bincount(j, flux, minlength=len(c))
            c.T += dv / self.area

    # ---- 提交：删除消耗单元，空隙生成新洋壳 ------------------------------------------------
    def commit(self):
        t0 = time.perf_counter()
        m, c, p = self.m, self.c, self.p
        pos = self.positions()
        covered = np.zeros(m.n, bool)
        for q in self.plates:
            covered |= self.covering(q, m.xyz) >= 0
        gap = np.flatnonzero(~covered)
        keep = np.flatnonzero(self.alive)
        new = c.gather(keep)
        n_new = 0
        if gap.size and keep.size:
            kq = min(12, keep.size)
            d, k = cKDTree(pos[keep]).query(m.xyz[gap], k=kq, workers=KD_WORKERS)
            pl = new.plate[k]
            owner = pl[:, 0]
            other = pl != owner[:, None]
            has = other.any(1)
            first = np.argmax(other, 1)
            d1 = d[:, 0]
            d2 = d[np.arange(gap.size), first]
            frac = np.where(has, (d2 - d1) / np.maximum(d1 + d2, 1e-12), 1.0)
            age = self.steps_since_commit * p.dt * np.clip(frac, 0.0, 1.0)
            bn = np.empty(gap.size, np.int64)
            for q in np.unique(owner):
                s = owner == q
                bn[s] = m.tree.query(m.xyz[gap[s]] @ self.R[int(q)], workers=KD_WORKERS)[1]
            _, uniq = np.unique(owner.astype(np.int64) * m.n + bn, return_index=True)
            g = Crust(uniq.size)
            g.plate = owner[uniq].astype(np.int32)
            g.node = bn[uniq]
            g.T[:] = p.T_ocean
            g.age = age[uniq]
            new = new.concat(g)
            n_new = int(uniq.size)
        self.c = new
        self.alive = np.ones(len(new), bool)
        for q in self.plates:
            self.R[q] = orthonormalize(self.R[q])
        present = set(np.unique(new.plate).tolist())
        for q in self.plates:
            if q not in present:
                del self.R[q]
                self.omega.pop(q, None)
                self._event("plate_death", plate=q)
        self.steps_since_commit = 0
        self._rebuild()
        self._log_commit(n_gap=int(gap.size), n_new=n_new, wall=time.perf_counter() - t0)

    def _log_commit(self, n_gap: int, n_new: int, wall: float):
        c = self.c
        v = self.view()
        self.commit_log.append({
            "t": self.t,
            "plate_area_km2": {q: float(self.area[self.alive & (c.plate == q)].sum()) for q in self.plates},
            "cont_area_km2": float(self.area[self.alive & (c.ctype == CONT)].sum()),
            "land_area_km2": float(self.m.area_km2[elevation(v, self.p) > 0].sum()),
            "covered_area_km2": float(self.m.area_km2[v.plate >= 0].sum()),
            "cont_volume_km3": self.cont_volume(),
            "n_parcels": int(self.alive.sum()),
            "budget": dict(self.budget),
            "n_gap_nodes": n_gap,
            "n_new_parcels": n_new,
            "wall_s": wall,
        })

    # ---- 固定网格视图与快照 -----------------------------------------------------------------
    def view(self, with_trench: bool = False):
        """把各板块的单元映射到固定网格上（每个固定节点取覆盖它的单元；重叠时陆壳优先，其次较厚、较年轻）。
        没有任何板块覆盖的节点（只在两次提交之间出现）plate = -1，按零龄洋壳显示。"""
        m, c = self.m, self.c
        best = np.full(m.n, -1, np.int64)
        score = np.full(m.n, -np.inf)
        for q in self.plates:
            par = self.covering(q, m.xyz)
            ok = par >= 0
            s = np.where(ok, c.ctype[np.maximum(par, 0)] * 1e6 + c.T[np.maximum(par, 0)] * 1e3
                         - c.age[np.maximum(par, 0)], -np.inf)
            better = ok & (s > score)
            best[better] = par[better]
            score[better] = s[better]
        v = c.gather(np.maximum(best, 0))
        v.node = np.arange(m.n)
        gap = best < 0
        v.plate[gap] = -1
        v.ctype[gap] = OCEAN
        v.T[gap] = self.p.T_ocean
        v.age[gap] = 0.0
        v.origin[gap] = ORIGIN_OCEAN
        v.oro[gap] = ORO_NONE
        if with_trench:
            tr = self.trench_now()
            return v, np.where(gap, 0.0, tr[np.maximum(best, 0)])
        return v

    def snapshot(self) -> dict:
        v, tr = self.view(with_trench=True)
        return {"t": self.t, "plate": v.plate.copy(), "ctype": v.ctype.copy(), "T": v.T.copy(), "age": v.age.copy(),
                "origin": v.origin.copy(), "oro": v.oro.copy(), "oro_t": v.oro_t.copy(),
                "elev": elevation(v, self.p, tr), "omega": {q: self.omega[q].copy() for q in self.plates}}

    # ---- 离散事件：裂解与地体转移（只在提交后进行）------------------------------------------
    def split_plate(self, parent: int, s1: np.ndarray, s2: np.ndarray, axis: np.ndarray, new_id: int,
                    rng: np.random.Generator, seg_km=(200.0, 600.0), offset_km=(30.0, 150.0),
                    p_extra_offset: float = 0.3, margin_km: float = 200.0, min_frac: float = 0.05,
                    omega_new: np.ndarray | None = None) -> np.ndarray | None:
        """把板块 parent 沿 s1、s2 的垂直平分线附近裂开，s2 一侧成为新板块 new_id（新板块继承 parent 的参照系）。

        裂谷线不直接用平分线，而是改成洋脊—转换断层阶梯：在以相对运动欧拉极 axis 为极的坐标里，
        洋脊段沿经线（垂直于扩张方向），段与段之间用沿纬线的转换断层错开。阶梯跟随裂谷的总体走向
        （斜向裂谷自然形成同向错开的阶梯），只有 p_extra_offset 比例的段再加一个随机错距 offset_km。
        （第一次每段都随机左右错开，结果是锯齿形，不像真实的洋脊—转换断层系统。）段长 seg_km 随机。
        新裂开的陆缘在 margin_km 内按距离减薄（陆架、陆坡）。成功返回新板块的单元，失败（太小）返回 None。"""
        assert self.steps_since_commit == 0
        c = self.c
        pos = self.positions()
        nodes = np.flatnonzero(self.alive & (c.plate == parent))
        x = pos[nodes]
        base_new = (x @ (np.asarray(s2) - np.asarray(s1))) > 0
        w = unit(axis)
        e1 = unit(np.cross(w, [0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0.0, 0.0]))
        e2 = np.cross(w, e1)
        theta = np.arccos(np.clip(x @ w, -1, 1))
        phi = np.arctan2(x @ e2, x @ e1)
        local = np.full(len(c), -1, np.int64)
        local[nodes] = np.arange(nodes.size)
        pe = self.pedges[(c.plate[self.pedges[:, 0]] == parent) & (c.plate[self.pedges[:, 1]] == parent)]
        ee = local[pe]
        ee = ee[(ee >= 0).all(1)]
        bnodes = np.unique(ee[base_new[ee[:, 0]] != base_new[ee[:, 1]]].ravel())
        if bnodes.size == 0:
            return None
        new = base_new.copy()
        th = theta[bnodes].min() - 1e-6
        th_end = theta[bnodes].max() + 1e-6
        n_seg = 0
        while th < th_end:
            th2 = th + rng.uniform(*seg_km) / R_KM
            band = (theta >= th) & (theta < th2)
            bb = bnodes[(theta[bnodes] >= th) & (theta[bnodes] < th2)]
            if bb.size and band.any():
                phi_b = float(np.angle(np.exp(1j * phi[bb]).mean()))
                s_mid = max(np.sin(0.5 * (th + th2)), 0.1)
                phi_band = phi_b
                if rng.random() < p_extra_offset:
                    phi_band += rng.choice([-1.0, 1.0]) * rng.uniform(*offset_km) / (R_KM * s_mid)
                bi = np.flatnonzero(band)
                d = wrap(phi[bi] - phi_band)
                near = np.abs(wrap(phi[bi] - phi_b)) < 0.35
                orient = np.sign(np.sum(np.where(base_new[bi][near], 1.0, -1.0) * np.sign(d[near]))) or 1.0
                far = np.abs(d) > np.radians(60)
                new[bi] = np.where(far, base_new[bi], d * orient > 0)
                n_seg += 1
            th = th2
        if not new.any():
            return None
        # 只保留包含 s2 的连通块
        sub_e = ee[new[ee[:, 0]] & new[ee[:, 1]]]
        lab = graph_components(nodes.size, sub_e)
        seed = np.flatnonzero(new)[np.argmax(x[new] @ unit(s2))]
        keep_local = np.flatnonzero(new & (lab == lab[seed]))
        if keep_local.size < min_frac * nodes.size or keep_local.size > (1 - min_frac) * nodes.size:
            return None
        is_new = np.zeros(nodes.size, bool)
        is_new[keep_local] = True
        bnd_local = np.unique(ee[is_new[ee[:, 0]] != is_new[ee[:, 1]]].ravel())
        keep = nodes[keep_local]
        c.plate[keep] = new_id
        self.R[new_id] = self.R[parent].copy()
        self.omega[new_id] = self.omega[parent].copy() if omega_new is None else np.asarray(omega_new, float)
        cont = nodes[c.ctype[nodes] == CONT]
        if cont.size and bnd_local.size:
            d, _ = cKDTree(x[bnd_local]).query(pos[cont], distance_upper_bound=chord_from_km(margin_km),
                                               workers=KD_WORKERS)
            ok = np.isfinite(d)
            j = cont[ok]
            before = c.T[j].copy()
            c.T[j] *= 0.35 + 0.65 * smoothstep(km_from_chord(d[ok]) / margin_km)
            self.budget["rift_thinning"] += float(((before - c.T[j]) * self.area[j]).sum())
            c.oro[j] = ORO_RIFT
            c.oro_t[j] = self.t
        self._rebuild()
        xm = unit(x[bnd_local].mean(0)) if bnd_local.size else unit(s1 + s2)
        self._event("rift", parent=parent, new_plate=new_id, n_parcels=int(keep.size), n_segments=n_seg,
                    lat=round(float(np.degrees(np.arcsin(xm[2]))), 2),
                    lon=round(float(np.degrees(np.arctan2(xm[1], xm[0]))), 2))
        return keep

    def transfer_terrane(self, down: int, over: int) -> int:
        """Cortial 式地体转移：下行板块上与上盘相接的连通陆壳整体改属上盘板块。
        地体要换到上盘的参照系（另一套互相旋转的格网），用"收集式"重采样：上盘格网上每个落在地体范围内的节点，
        取覆盖它的地体单元的属性（一次不超过半个网格间距的取整）；该节点上已有上盘单元的，地体厚度叠加上去。
        （第一次写成"散射式"——把地体单元逐个塞进最近的上盘节点——两套格网互相旋转，有的节点收到两个单元、
        有的一个也没有，空洞随后被当成空隙填成洋壳，出现莫尔条纹。）返回转移的单元数。"""
        assert self.steps_since_commit == 0
        m, c = self.m, self.c
        pos = self.positions()
        cont = np.flatnonzero(self.alive & (c.plate == down) & (c.ctype == CONT))
        over_idx = np.flatnonzero(self.alive & (c.plate == over))
        if cont.size == 0 or over_idx.size == 0:
            return 0
        local = np.full(len(c), -1, np.int64)
        local[cont] = np.arange(cont.size)
        ee = local[self.pedges]
        ee = ee[(ee >= 0).all(1)]
        lab = graph_components(cont.size, ee)
        d, _ = cKDTree(pos[over_idx]).query(pos[cont], workers=KD_WORKERS)
        touch = km_from_chord(d) < 1.5 * m.h_km
        hit = np.unique(lab[touch])
        moved = cont[np.isin(lab, hit)]
        if moved.size == 0:
            return 0
        Ro = self.R[over]
        _, cand = m.tree.query(pos[moved] @ Ro, k=7, workers=KD_WORKERS)
        cand = np.unique(cand.ravel())
        src = self.covering(down, m.xyz[cand] @ Ro.T)       # 上盘格网节点的绝对位置落在下行板块的哪个单元里
        ok = np.isin(src, moved)
        w, src = cand[ok], src[ok]
        occ = self.owner[over][w]
        has = occ >= 0
        has[has] = self.alive[occ[has]]
        # 已有上盘单元：叠加地体厚度（洋壳单元被陆壳覆盖时改为陆壳）
        o, s = occ[has], src[has]
        c.T[o] += c.T[s]
        to_cont = c.ctype[o] == OCEAN
        for f in ("ctype", "origin", "oro", "oro_t"):
            getattr(c, f)[o[to_cont]] = getattr(c, f)[s[to_cont]]
        # 空节点：新建上盘单元
        add = c.gather(src[~has])
        add.plate[:] = over
        add.node = w[~has]
        vol_before = float((c.T[moved] * self.area[moved]).sum())
        self.alive[moved] = False
        self.c = c.concat(add)
        self.alive = np.concatenate([self.alive, np.ones(len(add), bool)])
        self._rebuild()
        vol_after = float((c.T[s] * m.area_km2[c.node[o]]).sum() + (add.T * m.area_km2[add.node]).sum())
        self._event("terrane_transfer", downgoing=down, overriding=over, n_parcels=int(moved.size),
                    n_new=int(len(add)), n_stacked=int(has.sum()), area_km2=float(m.area_km2[w].sum()),
                    volume_ratio_after_before=round(vol_after / max(vol_before, 1e-9), 4))
        return int(moved.size)
