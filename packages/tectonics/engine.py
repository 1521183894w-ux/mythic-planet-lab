"""地壳引擎：板块参照系中的地块单元与板块相互作用（阶段 1.1 原型 `spikes/stage1_methods/engine.py` 整理入库并加速）。

表示方式（docs/tectonics_method.md §2）：
- 每块板块有一份自己的参照格网（同一套二十面体网格），地壳以"地块单元"存放：单元属于一块板块、对应该格网的一个节点，
  当前位置 = R_p · 节点坐标，R_p 是该板块的累计旋转。**板块内部永不重新取样。**
- "单元 u 是否进入板块 q 的范围"：把 u 的当前位置用 R_q 转回 q 的参照系，看最近的节点上是否有 q 的存活单元。
- 每 commit_every 步提交一次：删除已俯冲消耗的单元；固定网格上没有板块覆盖的节点（扩张空隙）生成新洋壳，
  放在最近板块参照系里最近的节点上（取整一次，不超过半个网格间距）。

加速（相对 1.1 原型）：
- 每步只检测"边缘单元"（参照格网上至少一个相邻节点没有本板块存活单元的单元），并且只对包围球冠覆盖它的板块检测；
  消耗一批单元后，新暴露的边缘单元在同一步内再检测（最多 max_sweeps 轮），快速汇聚一步推进几行时也不会漏。
- 提交与固定网格视图：用一次全局最近邻查询确定每个固定节点的候选板块，只对板块交界处的节点做精确检测。

规则与 method_class 见 docs/tectonics_method.md §3。
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .crust import (CONT, OCEAN, ORIGIN_ARC, ORIGIN_OCEAN, ORIGIN_RIFT, ORO_ARC, ORO_COLL, ORO_NONE, ORO_RIFT, Crust,
                    Params, elevation)
from .geometry import R_KM, chord_from_km, km_from_chord, orthonormalize, smoothstep, unit, wrap
from .mesh import KD_WORKERS, Mesh


def arc_profile(d_km, peak_km, max_km):
    up = smoothstep(d_km / peak_km)
    down = 1.0 - smoothstep((d_km - peak_km) / (max_km - peak_km))
    return np.where(d_km < peak_km, up, down)


HOLE_BORROW_T_KM = 34.0    # U32：走滑带小洞补陆壳时，只从厚于这个的陆壳匀体积（见 Engine.commit）


def gap_profile(spec) -> tuple[np.ndarray, np.ndarray, bool]:
    """Engine.cont_gap_pairs 的一项 -> (宽度节点 km, 厚度节点 km, 是否补小洞)。
    兼容 U30 的 (T_narrow, w1, T_wide, w_max) 四元组；U32 的写法是 {"knots": [(宽度, 厚度), ...], "holes": bool}。"""
    if isinstance(spec, dict):
        kn = np.asarray(spec["knots"], float)
        return kn[:, 0], kn[:, 1], bool(spec.get("holes", False))
    T_n, w1, T_w, w_max = spec
    return np.array([0.0, w1, w_max], float), np.array([T_n, T_n, T_w], float), False


def graph_components(n: int, edges: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _lat_lon(x: np.ndarray) -> dict:
    x = unit(x)
    return {"lat": round(float(np.degrees(np.arcsin(x[2]))), 2), "lon": round(float(np.degrees(np.arctan2(x[1], x[0]))), 2)}


class Engine:
    def __init__(self, mesh: Mesh, crust: Crust, params: Params, t0: float, omega: dict):
        self.m, self.c, self.p = mesh, crust, params
        self.t = float(t0)
        self.alive = np.ones(len(crust), bool)
        self.R = {int(q): np.eye(3) for q in np.unique(crust.plate)}
        self.omega = {q: np.asarray(omega[q], float) for q in self.R}
        self.next_uid = int(crust.uid.max()) + 1 if len(crust) else 0
        self.steps_since_commit = 0
        self.polarity: dict[int, int] = {}     # 板块对（小*1000+大）-> 俯冲（下行）板块
        self.gap_prefer: dict[int, int] = {}   # 板块对 -> 两者之间张开的新洋壳全部归这块（U26；默认按最近单元归属）
        self.hole_down_pairs: set | None = None  # U28 第二轮：汇聚边界上的网格小洞归下行板块的板块对；None = 所有板块对（见 commit）
        self.arc_flux_pairs: set | None = None   # U28 第二轮：Params.arc_ocean_flux_factor 只用于这些板块对的俯冲带；None = 所有
        self.arc_accrete_pairs: set | None = None  # U28 第二轮：弧—陆拼贴只在这些板块对之间；None = 所有
        self.arc_compress_pairs: set | None = None  # U28 第三轮：Params.arc_compress_T_max_km 只用于这些板块对的俯冲带；None = 所有
        # U30：两块陆块之间张开（走滑拉分、裂谷）时，空隙填拉伸减薄的陆壳而不是洋壳：板块对键 -> (T_narrow, w1, T_wide, w_max)，
        # 或（U32）{"knots": [(宽度 km, 厚度 km), ...], "holes": bool}。
        # 累计张开宽度（两侧原有陆壳——不含之前填的拉伸陆壳——之间的距离）不到 w1 时厚 T_narrow（沉积物充填的拉分盆地、走滑谷），
        # w1 到 w_max 之间线性减薄到 T_wide（拉伸陆缘），超过 w_max 才长洋壳（大陆裂开、海底扩张）。knots 是同一件事的分段线性写法
        # （U32 裂开的被动陆缘：陆架 -> 陆坡），最后一个节点的宽度就是 w_max。
        # 只用于两块真正张开处、且在 cont_gap_since 的时刻以后（第一次运行从 180 Ma 洲 C、D 碰撞起，碰撞带上网格对不齐的小缝也被填成陆壳、
        # 再被碰撞增厚，缝合带上堆出 1.1 百万 km²、平均 49 km 的陆壳）。
        # holes（U32）：两块陆壳之间不在张开的小洞（走滑带上两块板块的网格对不齐留下的）补成相邻的陆壳（厚度取两侧较薄的一侧、
        # 但不超过剖面的第一个厚度，即拉分盆地的厚度；属性抄最近的单元），不补成洋壳：U31 的 0 Ma 上洲 C、D 之间的走滑带有一串 0–20 Myr 的洋壳深水坑（评审 stage1_2026-10-25_1 第 4 条）
        self.cont_gap_pairs: dict = {}
        self.cont_gap_since: dict = {}             # 板块对键 -> 从哪个时刻（Ma）起才填陆壳；不写 = 一直
        # U32：陆壳上盘的弧增厚（岩浆增生与构造缩短）只用于这些板块对的俯冲带；None = 所有。终图倒推时外大洋板块固定在地幔参照系里，
        # 大陆一漂移，前方就与它慢速俯冲（U28 第二轮查明），终图在那里是被动陆缘；这种俯冲照旧消耗洋壳，但不再增厚陆壳
        # （否则补在被动陆缘外的陆架、陆表海被抬出海面，U30 的浅海湾有 29–44% 的格点这样露出）
        self.arc_cont_pairs: set | None = None
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
        """单元增删或改属后重建：各板块"节点 -> 单元"索引、板块内单元邻接、单元面积、包围球冠。"""
        m, c = self.m, self.c
        self.area = m.area_km2[c.node]
        self.full_deg = m.degree[c.node]
        self.owner, self.cap = {}, {}
        pe = []
        margin = 3.0 * m.h_km / R_KM
        for q in self.plates:
            idx = np.flatnonzero((c.plate == q) & self.alive)
            o = np.full(m.n, -1, np.int64)
            o[c.node[idx]] = idx
            self.owner[q] = o
            a, b = o[m.edges[:, 0]], o[m.edges[:, 1]]
            ok = (a >= 0) & (b >= 0)
            pe.append(np.stack([a[ok], b[ok]], 1))
            xb = m.xyz[c.node[idx]]
            s = xb.sum(0)
            if idx.size == 0 or np.linalg.norm(s) < 1e-3 * idx.size:
                self.cap[q] = (np.array([0.0, 0.0, 1.0]), -2.0)
            else:
                cen = unit(s)
                r = np.arccos(np.clip((xb @ cen).min(), -1, 1)) + margin
                self.cap[q] = (cen, float(np.cos(min(r, np.pi))) if r < np.pi else -2.0)
        self.pedges = np.concatenate(pe) if pe else np.zeros((0, 2), np.int64)
        cont = (c.ctype == CONT) & self.alive
        self.cont_area = {q: float(self.area[cont & (c.plate == q)].sum()) for q in self.plates}

    def _alive_nb(self) -> np.ndarray:
        """每个单元在本板块参照格网上的存活相邻单元数。"""
        i, j = self.pedges[:, 0], self.pedges[:, 1]
        ok = self.alive[i] & self.alive[j]
        M = len(self.c)
        return np.bincount(i[ok], minlength=M) + np.bincount(j[ok], minlength=M)

    def positions(self) -> np.ndarray:
        stack = np.zeros((max(self.R) + 1, 3, 3))
        for q, rm in self.R.items():
            stack[q] = rm
        return np.einsum("nij,nj->ni", stack[self.c.plate], self.m.xyz[self.c.node])

    def covering(self, q: int, pts: np.ndarray) -> np.ndarray:
        """pts（当前绝对位置）落在板块 q 的哪个存活单元里；不在 q 范围内为 -1。"""
        if len(pts) == 0:
            return np.zeros(0, np.int64)
        _, w = self.m.tree.query(pts @ self.R[q], workers=KD_WORKERS)
        par = self.owner[q][w]
        ok = par >= 0
        ok[ok] = self.alive[par[ok]]
        return np.where(ok, par, -1)

    def _in_cap(self, q: int, pts: np.ndarray) -> np.ndarray:
        cen, cosr = self.cap[q]
        return pts @ (self.R[q] @ cen) >= cosr

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

    def _new_uids(self, n: int) -> np.ndarray:
        u = np.arange(self.next_uid, self.next_uid + n, dtype=np.int64)
        self.next_uid += n
        return u

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
        n_sub, n_coll, n_tested = self._interact(pos)
        self._arc(pos)
        self._surface(p.dt, pos)
        ocean = self.alive & (self.c.ctype == OCEAN)
        self.c.age[ocean] += p.dt
        self.steps_since_commit += 1
        committed = False
        if self.steps_since_commit >= p.commit_every:
            self.commit()
            committed = True
        self.step_log.append({"t": self.t, "n_subducted": n_sub, "n_collision": n_coll, "n_tested": n_tested,
                              "wall_s": time.perf_counter() - t0})
        return committed

    # ---- 板块相互作用 --------------------------------------------------------------------
    def _interact(self, pos: np.ndarray) -> tuple[int, int, int]:
        c, p = self.c, self.p
        tested = np.zeros(len(c), bool)
        n_sub = n_coll = n_tested = 0
        for _ in range(p.max_sweeps):
            U0 = np.flatnonzero(self.alive & (self._alive_nb() < self.full_deg) & ~tested)
            if U0.size == 0:
                break
            tested[U0] = True
            n_tested += U0.size
            inside = np.full(U0.size, -1, np.int32)
            partner = np.full(U0.size, -1, np.int64)
            for q in self.plates:
                loc = np.flatnonzero((c.plate[U0] != q) & (inside < 0))
                if loc.size:
                    loc = loc[self._in_cap(q, pos[U0[loc]])]
                if loc.size == 0:
                    continue
                par = self.covering(q, pos[U0[loc]])
                hit = par >= 0
                inside[loc[hit]] = q
                partner[loc[hit]] = par[hit]
            h = inside >= 0
            if not h.any():
                break
            s, k = self._resolve(pos, U0[h], inside[h], partner[h])
            n_sub += s
            n_coll += k
            if s + k == 0:
                break
        return n_sub, n_coll, n_tested

    def _resolve(self, pos, U, Q, B) -> tuple[int, int]:
        c, p = self.c, self.p
        P = c.plate[U]
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
            # 洋—洋接触：若只有一方在接触处附近有大陆（被动陆缘外窄洋壳），那一方作上盘；否则较老的一方俯冲
            if oo.any():
                near = {}
                for q in (lo, hi):
                    cq = np.flatnonzero(self.alive & (c.plate == q) & (c.ctype == CONT))
                    near[q] = bool(cq.size) and bool(
                        (km_from_chord(np.linalg.norm(pos[cq] - xm, axis=1)) < p.near_continent_km).any())
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
            self._event(kind, plates=[lo, hi], subducting=sub, **_lat_lon(xm))
        sub_plate = np.array([self.polarity[int(k)] for k in key])

        if p.arc_accrete_T_km is not None:
            # 弧—陆碰撞（U28）：厚的洋内弧地壳被大陆撞上时转成陆壳拼贴（浮力大，不俯冲），之后按陆—陆碰撞处理
            cand = np.flatnonzero((cu == OCEAN) & (cb == CONT) & (c.T[U] >= p.arc_accrete_T_km))
            if self.arc_accrete_pairs is not None and cand.size:   # 终图倒推：只在设计的弧—陆碰撞处（U28 第二轮）
                cand = cand[np.isin(key[cand], list(self.arc_accrete_pairs))]
            if cand.size:
                acc = np.zeros(len(U), bool)
                for q in np.unique(P[cand]):
                    sel = cand[P[cand] == q]
                    cq = np.flatnonzero(self.alive & (c.plate == q) & (c.ctype == CONT))
                    if cq.size:
                        d, _ = cKDTree(pos[cq]).query(pos[U[sel]], workers=KD_WORKERS)
                        acc[sel[km_from_chord(d) > p.arc_accrete_far_km]] = True
                    else:
                        acc[sel] = True
                if acc.any():
                    j = U[acc]
                    c.ctype[j] = CONT
                    c.origin[j] = ORIGIN_ARC
                    c.t_form[j] = self.t
                    c.oro[j] = ORO_ARC
                    c.oro_t[j] = self.t
                    cu = c.ctype[U]
                    self.budget["arc_accreted"] = self.budget.get("arc_accreted", 0.0) + float(
                        (c.T[j] * self.area[j]).sum())

        consume = ((cu == OCEAN) & (cb == CONT)) | ((cu == OCEAN) & (cb == OCEAN) & (sub_plate == P))
        coll = (cu == CONT) & (cb == CONT) & (sub_plate == P)
        # 只有符合既定极性的消耗才是俯冲、记录前缘（驱动上盘增厚）。与极性相反的"洋壳在陆壳下"是上盘自己的弧前洋壳
        # 被到达的下行大陆掩覆（碰撞初期），只删除。（1.2a 合成试验中曾因此给下行大陆加了岛弧增厚。）
        front = consume & (sub_plate == P)

        if consume.any():
            idx = U[front]
            x = pos[idx]
            if idx.size:
                op = np.stack([self.omega[int(q)] for q in P[front]])
                oq = np.stack([self.omega[int(q)] for q in Q[front]])
                v = np.linalg.norm(np.cross(op - oq, x), axis=1) * R_KM / 10.0
                self.fronts.append({"t": self.t, "x": x, "sub": P[front], "over": Q[front], "v": v})
            self.alive[U[consume]] = False

        if coll.any():
            r = p.collision_radius_km
            add = np.zeros(len(c))
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
                    a_src = src[P[sel] == d_plate]
                    self.coll_area[pair] = self.coll_area.get(pair, 0.0) + float(self.area[a_src].sum())
                    if first:
                        self._event("collision_onset", downgoing=pair[0], overriding=pair[1], **_lat_lon(pos[a_src].mean(0)))
                self.alive[src] = False
            c.T += add
            mark = add > 0.05
            c.oro[mark] = ORO_COLL
            c.oro_t[mark] = self.t
        return int(consume.sum()), int(coll.sum())

    # ---- 俯冲上盘增厚（岛弧、安第斯型山脉）与海沟 ---------------------------------------
    def recent_fronts(self):
        """最近 front_memory_myr 内的俯冲前缘：(位置, 俯冲板块, 上盘板块, 相对速度 cm/yr)；没有则 None。"""
        while self.fronts and self.fronts[0]["t"] - self.t > self.p.front_memory_myr + 1e-9:
            self.fronts.popleft()
        if not self.fronts:
            return None
        f = self.fronts
        return (np.concatenate([a["x"] for a in f]), np.concatenate([a["sub"] for a in f]),
                np.concatenate([a["over"] for a in f]), np.concatenate([a["v"] for a in f]))

    def _arc(self, pos: np.ndarray):
        c, p = self.c, self.p
        fr = self.recent_fronts()
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
            dk = km_from_chord(d[ok])
            oceanic = c.ctype[j] == OCEAN          # 洋壳上盘：窄的洋内弧；陆壳上盘：宽的安第斯型带
            if p.arc_compress_T_max_km is not None:
                # U28 第三轮：洋内弧来源的地壳厚过 arc_to_cont_T_km 转成陆壳类型后，仍按窄的洋内弧带增厚。否则一过 31 km 增厚形状
                # 就换成安第斯式宽带（峰值离海沟 150 km），同一处的速率掉一半、也不再用洋内弧的通量系数，挤压型弧卡在 29–32 km
                oceanic = oceanic | (c.origin[j] == ORIGIN_ARC)
            f = np.where(oceanic, arc_profile(dk, p.arc_peak_ocean_km, p.arc_max_ocean_km),
                         arc_profile(dk, p.arc_peak_km, p.arc_max_km))
            dT = p.arc_rate_km_myr * f * np.clip(V[sel][k[ok]] / p.v0_cm_yr, 0.0, 1.5) * p.dt
            if p.arc_ocean_flux_factor is not None:   # 洋内弧带窄，同样的岩浆通量增厚更快（见 Params.arc_ocean_flux_factor）
                fac = oceanic
                if self.arc_flux_pairs is not None:   # 终图倒推：只用于设计的俯冲带（固定外大洋在大陆围裙下的慢速俯冲是简化带来的）
                    sub = S[sel][k[ok]]
                    fac = fac & np.isin(np.minimum(sub, o) * 1000 + np.maximum(sub, o), list(self.arc_flux_pairs))
                dT = np.where(fac, dT * p.arc_ocean_flux_factor, dT)
            dT_s = np.zeros_like(dT)
            if p.arc_shortening_km_myr is not None:
                # 上盘朝海沟前进造成的构造缩短（见 Params.arc_shortening_km_myr）：上盘的绝对速度在指向最近前缘方向上的分量
                Xk = X[sel][k[ok]]
                xj = pos[j]
                u = Xk - xj
                u = u - np.sum(u * xj, axis=1, keepdims=True) * xj
                u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                v_adv = np.sum(np.cross(self.omega[int(o)], xj) * u, axis=1) * R_KM / 10.0
                dT_s = p.arc_shortening_km_myr * f * np.clip(v_adv / p.v0_cm_yr, 0.0, 1.5) * p.dt
            if p.arc_ocean_T_max_km is not None:    # 洋内弧增厚上限（见 Params.arc_ocean_T_max_km）
                juv = (c.ctype[j] == OCEAN) | (c.origin[j] == ORIGIN_ARC)
                if p.arc_compress_T_max_km is None:
                    dT = dT + dT_s
                    dT = np.where(juv, np.minimum(dT, np.maximum(p.arc_ocean_T_max_km - c.T[j], 0.0)), dT)
                else:
                    # 挤压型洋内弧（见 Params.arc_compress_T_max_km）：岩浆增生封顶于 arc_ocean_T_max_km，构造缩短可到更高的上限；
                    # 不在 arc_compress_pairs 里的俯冲带两项合起来仍封顶于 arc_ocean_T_max_km（与不设时相同）
                    comp = juv
                    if self.arc_compress_pairs is not None:
                        sub = S[sel][k[ok]]
                        comp = comp & np.isin(np.minimum(sub, o) * 1000 + np.maximum(sub, o), list(self.arc_compress_pairs))
                    if p.arc_ocean_flux_factor is not None:
                        # 挤压型洋内弧的构造缩短与岩浆增生一样集中在窄的洋内弧带上：arc_shortening_km_myr 是按 600 km 宽的安第斯带
                        # 标定的，同样的水平缩短摊在 200 km 宽的带上增厚快 arc_ocean_flux_factor 倍（见 Params.arc_ocean_flux_factor）
                        dT_s = np.where(comp & oceanic, dT_s * p.arc_ocean_flux_factor, dT_s)
                    dT_m = np.where(juv, np.minimum(dT, np.maximum(p.arc_ocean_T_max_km - c.T[j], 0.0)), dT)
                    lim = np.where(comp, p.arc_compress_T_max_km, p.arc_ocean_T_max_km)
                    dT_s = np.where(juv, np.minimum(dT_s, np.maximum(lim - c.T[j] - dT_m, 0.0)), dT_s)
                    dT = dT_m + dT_s
            else:
                dT = dT + dT_s
            blocked = np.zeros(j.size, bool)
            cont_j = (c.ctype[j] == CONT) & (c.origin[j] != ORIGIN_ARC)
            if self.arc_cont_pairs is not None:     # U32：陆壳上的弧增厚只用于设计的俯冲带（见 arc_cont_pairs）
                sub = S[sel][k[ok]]
                key = np.minimum(sub, o) * 1000 + np.maximum(sub, o)
                blocked |= cont_j & ~np.isin(key, list(self.arc_cont_pairs))
            if p.arc_cont_min_T_km is not None:     # U32：海平面以下的减薄陆缘不接受弧增厚（见 Params.arc_cont_min_T_km）
                blocked |= cont_j & (c.T[j] < p.arc_cont_min_T_km)
            if blocked.any():
                self.budget["arc_suppressed"] = (self.budget.get("arc_suppressed", 0.0)
                                                 + float((dT * self.area[j])[blocked].sum()))
                dT = np.where(blocked, 0.0, dT)
            c.T[j] += dT
            self.budget["arc_added"] += float((dT * self.area[j])[c.ctype[j] == CONT].sum())
            strong = j[(f > 0.3) & ~blocked]
            c.oro[strong] = np.where(c.oro[strong] == ORO_COLL, ORO_COLL, ORO_ARC)
            c.oro_t[strong] = self.t
            conv = j[(c.ctype[j] == OCEAN) & (c.T[j] >= p.arc_to_cont_T_km)]
            if conv.size:
                c.ctype[conv] = CONT
                c.origin[conv] = ORIGIN_ARC
                c.t_form[conv] = self.t
                self.budget["arc_converted"] += float((c.T[conv] * self.area[conv]).sum())

    def trench_now(self, pos: np.ndarray | None = None) -> np.ndarray:
        """当前位置下的海沟附加深度（m，显示用，每个单元一个值）：俯冲板块上距最近俯冲前缘越近越深。"""
        c, p = self.c, self.p
        pos = self.positions() if pos is None else pos
        out = np.zeros(len(c))
        fr = self.recent_fronts()
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
    def _surface(self, dt: float, pos: np.ndarray | None = None):
        c, p = self.c, self.p
        cont = self.alive & (c.ctype == CONT)
        ex = np.where(cont, np.maximum(c.T - p.T_cont, 0.0), 0.0)
        if p.erosion_law == "relief":
            loss = np.minimum(self.erosion_rate(pos) * dt, ex)
        elif p.erosion_law == "linear":
            loss = ex * dt / p.tau_oro_myr
        else:
            raise ValueError(f"未知的 erosion_law：{p.erosion_law}")
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
        self._collapse(cont)

    def relief_km(self, pos: np.ndarray | None = None) -> np.ndarray:
        """每个单元的起伏（km）：本单元高程 − ero_relief_km 半径内最低处的高程（海面以下按海平面算）。
        各单元按当前位置落到固定网格的最近节点，在固定网格上一圈一圈取最小（L7 一圈约 60 km），所以跨板块边界的落差也算在内
        （第一版只在同一板块内找，压在板块交界上的缝合带山脊看不到对面的低地，几乎不剥蚀）。"""
        c, p = self.c, self.p
        if pos is None:
            pos = self.positions()
        a = np.flatnonzero(self.alive)
        z = np.maximum(elevation(c.gather(a), p), 0.0) / 1000.0
        _, node = self.m.tree.query(pos[a], workers=KD_WORKERS)
        zf = np.full(self.m.n, np.inf)
        np.minimum.at(zf, node, z)
        i, j = self.m.edges[:, 0], self.m.edges[:, 1]
        for _ in range(max(1, int(round(p.ero_relief_km / self.m.h_km)))):
            prev = zf.copy()
            np.minimum.at(zf, i, prev[j])
            np.minimum.at(zf, j, prev[i])
        out = np.zeros(len(c))
        out[a] = np.maximum(z - zf[node], 0.0)
        return out

    def erosion_rate(self, pos: np.ndarray | None = None) -> np.ndarray:
        """erosion_law = "relief" 的岩石剥蚀速率（km/Myr），见 Params.erosion_law。"""
        c, p = self.c, self.p
        h_ex = np.maximum(c.T - p.T_cont, 0.0) * p.iso_m_per_km / 1000.0
        return p.ero_K_km_myr * self.relief_km(pos) ** p.ero_n + p.ero_bg_per_myr * h_ex

    def _collapse(self, cont: np.ndarray):
        """超过 T_max 的地壳"摊平"：每个超厚连通区的多余体积，一圈一圈填到同板块外围陆壳上（每圈按剩余容量比例填），
        直到填完，体积守恒，结果是平顶高原向外扩展。这让碰撞带长成高原而不是无限长高
        （1.2a 合成试验没有这一条时地壳厚到 107 km、高程 11 km）。
        （第一次写成逐轮向邻居平分，高原内部单元互相传递，多余厚度只能像扩散一样慢慢挪出，L7 上仍超过上限。）"""
        c, p = self.c, self.p
        over = cont & (c.T > p.T_max_km)
        if not over.any():
            return
        from scipy.sparse import csr_matrix
        M = len(c)
        e = self.pedges
        ce = e[cont[e[:, 0]] & cont[e[:, 1]]]
        adj = csr_matrix((np.ones(2 * len(ce)), (np.r_[ce[:, 0], ce[:, 1]], np.r_[ce[:, 1], ce[:, 0]])), shape=(M, M))
        ae = e[self.alive[e[:, 0]] & self.alive[e[:, 1]]]
        adj_all = csr_matrix((np.ones(2 * len(ae)), (np.r_[ae[:, 0], ae[:, 1]], np.r_[ae[:, 1], ae[:, 0]])), shape=(M, M))
        widened = []
        near = None                             # 按当前位置找相邻单元（只在本板块上没有邻居时用，见下）
        oi = np.flatnonzero(over)
        local = np.full(M, -1, np.int64)
        local[oi] = np.arange(oi.size)
        oe = local[ce]
        oe = oe[(oe >= 0).all(1)]
        lab = graph_components(oi.size, oe)
        for k in np.unique(lab):
            members = oi[lab == k]
            E = float(((c.T[members] - p.T_max_km) * self.area[members]).sum())
            c.T[members] = p.T_max_km
            visited = np.zeros(M, bool)
            visited[members] = True
            frontier = members
            while E > 1e-9:
                ring = np.unique(adj[frontier].indices)
                ring = ring[~visited[ring]]
                if ring.size == 0:
                    # 外围没有陆壳了（孤立的弧岛、小地块）：向同板块相邻的洋壳摊，岛向外长大。
                    # 1.2c 之前这里把多余体积留在原处，洋内弧岛被岛弧增厚反复加厚到 600 km（高程 80 km）。
                    ring = np.unique(adj_all[frontier].indices)
                    ring = ring[~visited[ring]]
                    if ring.size == 0:
                        # 本板块参照格网上连一个存活的相邻单元都没有（碰撞前缘残留在别的板块中间的一两个单元）：
                        # 按当前位置向任何板块的相邻单元摊。U20 后的 L5 剧本试验里这样一个单元被碰撞拼贴反复加厚到 149 km
                        if near is None:
                            ai = np.flatnonzero(self.alive)
                            pos = self.positions()
                            near = (ai, pos, cKDTree(pos[ai]))
                        ai, pos, tr = near
                        hits = tr.query_ball_point(pos[frontier], r=chord_from_km(1.5 * self.m.h_km))
                        ring = np.unique(ai[np.concatenate([np.asarray(h, np.int64) for h in hits])])
                        ring = ring[~visited[ring]]
                    widened.append(ring)
                if ring.size == 0:          # 周围什么都没有（实际不会发生）：多余体积留在原处
                    c.T[members] += E / self.area[members].sum()
                    break
                cap = np.maximum(p.T_max_km - c.T[ring], 0.0) * self.area[ring]
                C = float(cap.sum())
                if C >= E:
                    c.T[ring] += cap * (E / C) / self.area[ring]
                    E = 0.0
                else:
                    c.T[ring] += cap / self.area[ring]
                    E -= C
                visited[ring] = True
                frontier = ring
        if widened:                         # 摊到的洋壳够厚就成为陆壳（与岛弧转化同一标准）
            w = np.unique(np.concatenate(widened))
            conv = w[(c.ctype[w] == OCEAN) & (c.T[w] >= p.arc_to_cont_T_km)]
            if conv.size:
                c.ctype[conv] = CONT
                c.origin[conv] = ORIGIN_ARC
                c.t_form[conv] = self.t
                c.oro[conv] = np.where(c.oro[conv] == ORO_COLL, ORO_COLL, ORO_ARC)
                c.oro_t[conv] = self.t
                self.budget["arc_converted"] += float((c.T[conv] * self.area[conv]).sum())

    # ---- 固定网格覆盖 -----------------------------------------------------------------------
    def cover_fixed(self, pos: np.ndarray | None = None) -> np.ndarray:
        """每个固定网格节点由哪个存活单元覆盖（-1 = 空隙）。重叠时陆壳优先，其次较厚、较年轻。
        先用一次全局最近邻：最近的单元是本板块内部单元、距离 < 0.75 个网格间距、且最近 6 个单元同属一块板块时直接取它；
        其余（板块交界、空隙边缘）对候选板块逐个做精确检测。"""
        m, c = self.m, self.c
        pos = self.positions() if pos is None else pos
        ai = np.flatnonzero(self.alive)
        best = np.full(m.n, -1, np.int64)
        if ai.size == 0:
            return best
        interior = self._alive_nb() == self.full_deg
        d, k = cKDTree(pos[ai]).query(m.xyz, k=min(6, ai.size), workers=KD_WORKERS)
        nn = ai[k]
        pl = c.plate[nn]
        easy = (pl == pl[:, :1]).all(1) & interior[nn[:, 0]] & (km_from_chord(d[:, 0]) < 0.75 * m.h_km)
        best[easy] = nn[easy, 0]
        hard = np.flatnonzero(~easy)
        if hard.size:
            score = np.full(hard.size, -np.inf)
            for q in np.unique(pl[hard]):
                if int(q) not in self.R:
                    continue
                rows = np.flatnonzero((pl[hard] == q).any(1))
                par = self.covering(int(q), m.xyz[hard[rows]])
                ok = par >= 0
                pp = np.maximum(par, 0)
                s = np.where(ok, c.ctype[pp] * 1e6 + c.T[pp] * 1e3 - c.age[pp], -np.inf)
                better = ok & (s > score[rows])
                best[hard[rows[better]]] = par[better]
                score[rows[better]] = s[better]
        return best

    # ---- 提交：删除消耗单元，空隙生成新洋壳 ------------------------------------------------
    def commit(self):
        t0 = time.perf_counter()
        m, c, p = self.m, self.c, self.p
        pos = self.positions()
        gap = np.flatnonzero(self.cover_fixed(pos) < 0)
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
            if self.gap_prefer and has.any():     # 指定的板块对之间张开的新洋壳全部归指定的一块（U26）
                owner = owner.copy()
                p2 = pl[np.arange(gap.size), first]
                key = np.minimum(owner, p2).astype(np.int64) * 1000 + np.maximum(owner, p2)
                opening = np.ones(gap.size, bool)
                if p.gap_prefer_opening_only:
                    # 只有两块在这里张开（相对速度朝两侧最近单元的连线方向分开）才按指定归属；汇聚边界上两块板块的网格
                    # 对不齐留下的小洞不算（见下）（U28 第二轮：否则指定的一块每次提交都往汇聚的另一块里长一点，
                    # 菲律宾海式板块的东缘 50 Myr 往外长了约 600 km）
                    om = np.zeros((max(self.R) + 1, 3))
                    for q, w in self.omega.items():
                        om[q] = w
                    u = pos[keep[k[:, 0]]] - pos[keep[k[np.arange(gap.size), first]]]
                    v = np.cross(om[owner] - om[p2], m.xyz[gap])
                    opening = np.sum(v * u, axis=1) > 0.0
                for kk, q in self.gap_prefer.items():
                    sel = has & (key == kk) & opening
                    owner[sel] = q
                if p.gap_prefer_opening_only:
                    # 汇聚边界上的小洞归既定的下行板块（下一步就被俯冲掉）：按最近单元归属时一半补给上盘，上盘的弧前每次提交
                    # 往海里长一点，海沟离海岸越来越远，岸上的弧增厚变弱（第二轮第三次运行：东岸山系 1864 -> 796 m）
                    # 只限两侧挨着的都是洋壳的海沟处，且只限 hole_down_pairs 里的板块对（终图倒推：大洋板块下行的、岛弧系统设计的）；
                    # 碰撞带、陆缘仍按最近单元（第二轮第四至七次运行对所有板块对都这样做：印度式碰撞开始得晚，补回陆缘留在高原范围里的
                    # 少了，东方大高原 4945 -> 2394 m；按实测重定保留率后高原回来，北缘山系仍只有 1175 m）
                    i2 = k[np.arange(gap.size), first]
                    ct_o, ct_2 = new.ctype[k[:, 0]], new.ctype[i2]
                    conv = np.flatnonzero(has & ~opening)
                    for kk in np.unique(key[conv]):
                        sub = self.polarity.get(int(kk))
                        if sub is None or (self.hole_down_pairs is not None and int(kk) not in self.hole_down_pairs):
                            continue
                        r = conv[key[conv] == kk]
                        owner[r[(ct_o[r] == OCEAN) & (ct_2[r] == OCEAN)]] = sub
            cont_T = np.full(gap.size, np.nan)        # U30：两块陆块之间的窄缺口填拉伸减薄陆壳（见 cont_gap_pairs）
            hole_src = np.full(gap.size, -1, np.int64)   # U32：补成相邻陆壳的小洞，抄哪个单元的属性（new 里的序号）
            hole_T = np.full(gap.size, np.nan)           # 小洞的厚度：两侧较薄的一侧，但不超过剖面的第一个厚度（拉分盆地厚度）
            if self.cont_gap_pairs and has.any():
                p2 = pl[np.arange(gap.size), first]
                key2 = np.minimum(pl[:, 0], p2).astype(np.int64) * 1000 + np.maximum(pl[:, 0], p2)
                i1, i2 = k[:, 0], k[np.arange(gap.size), first]
                both = has & (new.ctype[i1] == CONT) & (new.ctype[i2] == CONT)
                om2 = np.zeros((max(self.R) + 1, 3))
                for q, wv in self.omega.items():
                    om2[q] = wv
                apart = np.sum(np.cross(om2[pl[:, 0]] - om2[p2], m.xyz[gap]) * (pos[keep[i1]] - pos[keep[i2]]), axis=1) > 0.0
                for kk, spec in self.cont_gap_pairs.items():
                    if self.t > self.cont_gap_since.get(kk, np.inf) + 1e-9:
                        continue
                    kw, kT, holes = gap_profile(spec)
                    if holes:
                        hs = np.flatnonzero(both & ~apart & (key2 == kk))
                        hole_src[hs] = np.where(new.T[i1[hs]] <= new.T[i2[hs]], i1[hs], i2[hs])
                        hole_T[hs] = np.minimum(new.T[hole_src[hs]], kT[0])
                    sel = np.flatnonzero(both & apart & (key2 == kk))
                    if sel.size == 0:
                        continue
                    width = np.zeros(sel.size)
                    for q in (int(kk) // 1000, int(kk) % 1000):
                        orig = np.flatnonzero((new.plate == q) & (new.ctype == CONT) & (new.origin != ORIGIN_RIFT))
                        if orig.size == 0:
                            width[:] = np.inf
                            continue
                        dq, _ = cKDTree(pos[keep[orig]]).query(m.xyz[gap[sel]], workers=KD_WORKERS)
                        width += km_from_chord(dq)
                    cont_T[sel] = np.where(width <= kw[-1], np.interp(width, kw, kT), np.nan)
            bn = np.empty(gap.size, np.int64)
            for q in np.unique(owner):
                s = owner == q
                bn[s] = m.tree.query(m.xyz[gap[s]] @ self.R[int(q)], workers=KD_WORKERS)[1]
            _, uniq = np.unique(owner.astype(np.int64) * m.n + bn, return_index=True)
            g = Crust(uniq.size)
            g.uid = self._new_uids(uniq.size)
            g.plate = owner[uniq].astype(np.int32)
            g.node = bn[uniq]
            g.T[:] = p.T_ocean
            g.age = age[uniq]
            g.t_form = self.t + g.age
            cg = np.isfinite(cont_T[uniq])
            if cg.any():
                g.ctype[cg] = CONT
                g.T[cg] = cont_T[uniq][cg]
                g.origin[cg] = ORIGIN_RIFT
                g.oro[cg] = ORO_RIFT
                g.oro_t[cg] = self.t
                g.t_form[cg] = self.t
                self.budget["rift_fill"] = self.budget.get("rift_fill", 0.0) + float((g.T[cg] * m.area_km2[g.node[cg]]).sum())
            hg = hole_src[uniq] >= 0
            if hg.any():                          # U32：走滑带上的小洞补成相邻陆壳（见 cont_gap_pairs 的 holes）
                src = hole_src[uniq][hg]
                for f in ("ctype", "origin", "prov", "t_form", "oro", "oro_t"):
                    getattr(g, f)[hg] = getattr(new, f)[src]
                # 厚度封顶在拉分盆地厚度；体积从最近的 12 个单元里增厚的陆壳匀出来。
                # （只从两侧两个单元匀时每个一下子薄约 16 km，又成了深坑）
                # 第一次运行抄邻居厚度、凭空加体积：走滑带受阻段一直在汇聚，小洞每次提交反复出现，补上的陆壳下一步又被碰撞并进山里，
                # 累计多出 8.4×10⁶ km³，南方缝合带山系 3003 -> 3899 m；只封顶不守恒时仍多出 8.1×10⁶ km³、3909 m
                g.T[hg] = hole_T[uniq][hg]
                g.age[hg] = new.age[src]
                V = g.T[hg] * m.area_km2[g.node[hg]]
                nb = k[uniq][hg]                                   # 小洞最近的 12 个单元（两侧都有）
                a_nb = m.area_km2[new.node[nb]]
                # 只从增厚的陆壳（厚于 hole_borrow_T_km）匀体积：受阻挤压段由旁边的山脊供给（补上的陆壳下一步又被碰撞并回山里，
                # 体积守恒）；纯走滑段两侧是正常陆壳，匀不出来的部分当作拉分盆地自己的填充（那里不汇聚，不会被并进山里）。
                # 第一版从最近的 12 个陆壳单元一律匀，同一地带每次提交都匀，累计把走滑带东侧的平原压成一串低于海平面的洼地
                avail = np.where(new.ctype[nb] == CONT, np.maximum(new.T[nb] - HOLE_BORROW_T_KM, 0.0), 0.0) * a_nb
                tot = avail.sum(1)
                take = np.minimum(V, tot)
                share = np.where(tot[:, None] > 0, avail * (take / np.maximum(tot, 1e-12))[:, None], 0.0)
                np.subtract.at(new.T, nb.ravel(), (share / a_nb).ravel())
                self.budget["hole_created"] = self.budget.get("hole_created", 0.0) + float((V - take).sum())
                self.budget["hole_fill"] = self.budget.get("hole_fill", 0.0) + float(V.sum())
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
        """把各板块的单元映射到固定网格上。没有板块覆盖的节点（只在两次提交之间出现）plate = -1，按零龄洋壳显示。"""
        m, c = self.m, self.c
        pos = self.positions()
        best = self.cover_fixed(pos)
        v = c.gather(np.maximum(best, 0))
        v.node = np.arange(m.n)
        gap = best < 0
        v.uid[gap] = -1
        v.plate[gap] = -1
        v.ctype[gap] = OCEAN
        v.T[gap] = self.p.T_ocean
        v.age[gap] = 0.0
        v.origin[gap] = ORIGIN_OCEAN
        v.oro[gap] = ORO_NONE
        if with_trench:
            tr = self.trench_now(pos)
            return v, np.where(gap, 0.0, tr[np.maximum(best, 0)])
        return v

    def snapshot(self) -> dict:
        v, tr = self.view(with_trench=True)
        return {"t": self.t, "plate": v.plate.copy(), "ctype": v.ctype.copy(), "T": v.T.copy(), "age": v.age.copy(),
                "origin": v.origin.copy(), "prov": v.prov.copy(), "t_form": v.t_form.copy(),
                "oro": v.oro.copy(), "oro_t": v.oro_t.copy(),
                "elev": elevation(v, self.p, tr), "omega": {q: self.omega[q].copy() for q in self.plates}}

    # ---- 离散事件：裂解与地体转移（只在提交后进行）------------------------------------------
    def split_plate(self, parent: int, s1: np.ndarray | None, s2: np.ndarray | None, axis: np.ndarray, new_id: int,
                    rng: np.random.Generator, seg_km=(200.0, 600.0), offset_km=(30.0, 150.0),
                    p_extra_offset: float = 0.3, margin_km: float = 200.0, min_frac: float = 0.05,
                    omega_new: np.ndarray | None = None, new_mask: np.ndarray | None = None,
                    staircase: bool = True) -> np.ndarray | None:
        """把板块 parent 裂开，一侧成为新板块 new_id（继承 parent 的参照系，不重新取样）。

        裂开的位置：给了 new_mask（按单元的布尔掩膜）就沿它的边界（例如古老省份之间的缝合带，见 prehistory.py），
        否则沿 s1、s2 的垂直平分线，s2 一侧为新板块。
        staircase=True 时把裂谷线改成洋脊—转换断层阶梯：以相对运动欧拉极 axis 为极，洋脊段沿经线、转换断层沿纬线；
        阶梯跟随裂谷总体走向，只有 p_extra_offset 比例的段再加随机错距 offset_km。段长 seg_km 随机。
        新陆缘在 margin_km 内按距离减薄。成功返回新板块的单元，失败（新板块太小或太大）返回 None。"""
        assert self.steps_since_commit == 0, "裂解只在提交后进行"
        c = self.c
        pos = self.positions()
        nodes = np.flatnonzero(self.alive & (c.plate == parent))
        x = pos[nodes]
        if new_mask is not None:
            base_new = np.asarray(new_mask, bool)[nodes]
        else:
            base_new = (x @ (np.asarray(s2) - np.asarray(s1))) > 0
        w = unit(axis)
        e1 = unit(np.cross(w, [0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0.0, 0.0]))
        e2 = np.cross(w, e1)
        theta = np.arccos(np.clip(x @ w, -1, 1))
        phi = np.arctan2(x @ e2, x @ e1)
        local = np.full(len(c), -1, np.int64)
        local[nodes] = np.arange(nodes.size)
        ee = local[self.pedges]
        ee = ee[(ee >= 0).all(1)]
        bnodes = np.unique(ee[base_new[ee[:, 0]] != base_new[ee[:, 1]]].ravel())
        if bnodes.size == 0:
            return None
        new = base_new.copy()
        th = theta[bnodes].min() - 1e-6
        th_end = theta[bnodes].max() + 1e-6 if staircase else th
        n_seg = 0
        while th < th_end:
            th2 = th + rng.uniform(*seg_km) / R_KM
            band = (theta >= th) & (theta < th2)
            bb = bnodes[(theta[bnodes] >= th) & (theta[bnodes] < th2)]
            if bb.size and band.any():
                phi_b = float(np.angle(np.exp(1j * phi[bb]).mean()))
                phi_band = phi_b
                if rng.random() < p_extra_offset:
                    s_mid = max(np.sin(0.5 * (th + th2)), 0.1)
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
        lab = graph_components(nodes.size, ee[new[ee[:, 0]] & new[ee[:, 1]]])
        if s2 is not None:
            seed = np.flatnonzero(new)[np.argmax(x[new] @ unit(s2))]
        else:                                   # 按掩膜裂开：取新侧最大的连通块
            vals, cnt = np.unique(lab[new], return_counts=True)
            seed = np.flatnonzero(new & (lab == vals[np.argmax(cnt)]))[0]
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
        xm = x[bnd_local].mean(0) if bnd_local.size else x.mean(0)
        self._event("rift", parent=parent, new_plate=new_id, n_parcels=int(keep.size), n_segments=n_seg,
                    **_lat_lon(xm))
        return keep

    def add_plate(self, q: int, omega) -> bool:
        """新增一块还没有单元的板块（U28 第二轮：俯冲起始时从大洋里分出的新板块，例如菲律宾海式板块）。
        参照系取当前的地幔参照系（R = 单位阵），之后按旋转文件运动；单元由 transfer_units 转进来。已存在时不动，返回 False。"""
        assert self.steps_since_commit == 0, "新增板块只在提交后进行"
        q = int(q)
        if q in self.R:
            return False
        self.R[q] = np.eye(3)
        self.omega[q] = np.asarray(omega, float)
        self._rebuild()
        self._event("plate_added", plate=q)
        return True

    def transfer_units(self, src: int, dst: int, moved: np.ndarray, kind: str = "unit_transfer") -> int:
        """把板块 src 的一批单元 moved 整体改属板块 dst（U26：上盘洋壳"围裙"转给下行板块，被动陆缘转为俯冲带）。
        与 transfer_terrane 一样用"收集式"重采样：dst 参照格网上落在这批单元范围内、还没有 dst 单元的节点，
        取覆盖它的 src 单元的属性；已有 dst 单元的节点保持不变。重采样后的小空洞在下一次提交时按新洋壳补上。
        返回 dst 新增的单元数。"""
        assert self.steps_since_commit == 0, "单元转移只在提交后进行"
        moved = np.asarray(moved, np.int64)
        if moved.size == 0 or src not in self.R or dst not in self.R:
            return 0
        m, c = self.m, self.c
        pos = self.positions()
        Rd = self.R[dst]
        _, cand = m.tree.query(pos[moved] @ Rd, k=7, workers=KD_WORKERS)
        cand = np.unique(cand.ravel())
        s = self.covering(src, m.xyz[cand] @ Rd.T)
        ok = np.isin(s, moved)
        w, s = cand[ok], s[ok]
        occ = self.owner[dst][w]
        has = occ >= 0
        has[has] = self.alive[occ[has]]
        w, s = w[~has], s[~has]
        add = c.gather(s)
        add.uid = self._new_uids(len(add))
        add.plate[:] = dst
        add.node = w
        self.alive[moved] = False
        self.c = c.concat(add)
        self.alive = np.concatenate([self.alive, np.ones(len(add), bool)])
        self._rebuild()
        self._event(kind, src=int(src), dst=int(dst), n_parcels=int(moved.size), n_new=int(len(add)),
                    area_km2=round(float(m.area_km2[w].sum())))
        return int(len(add))

    def transfer_terrane(self, down: int, over: int) -> int:
        """Cortial 式地体转移：下行板块上与上盘相接的连通陆壳整体改属上盘板块。
        换到上盘的参照格网用"收集式"重采样：上盘格网上每个落在地体范围内的节点，取覆盖它的地体单元的属性；
        该节点上已有上盘单元的，地体厚度叠加上去。（"散射式"会造成重复与空洞，见 spikes/stage1_methods/README.md。）
        返回转移的单元数。"""
        assert self.steps_since_commit == 0, "地体转移只在提交后进行"
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
        hit = np.unique(lab[km_from_chord(d) < 1.5 * m.h_km])
        moved = cont[np.isin(lab, hit)]
        if moved.size == 0:
            return 0
        Ro = self.R[over]
        _, cand = m.tree.query(pos[moved] @ Ro, k=7, workers=KD_WORKERS)
        cand = np.unique(cand.ravel())
        src = self.covering(down, m.xyz[cand] @ Ro.T)
        ok = np.isin(src, moved)
        w, src = cand[ok], src[ok]
        occ = self.owner[over][w]
        has = occ >= 0
        has[has] = self.alive[occ[has]]
        o, s = occ[has], src[has]
        c.T[o] += c.T[s]
        to_cont = c.ctype[o] == OCEAN
        for f in ("ctype", "origin", "oro", "oro_t"):
            getattr(c, f)[o[to_cont]] = getattr(c, f)[s[to_cont]]
        add = c.gather(src[~has])
        add.uid = self._new_uids(len(add))
        add.plate[:] = over
        add.node = w[~has]
        vol_before = float((c.T[moved] * self.area[moved]).sum())
        vol_after = float((c.T[s] * m.area_km2[c.node[o]]).sum() + (add.T * m.area_km2[add.node]).sum())
        self.alive[moved] = False
        self.c = c.concat(add)
        self.alive = np.concatenate([self.alive, np.ones(len(add), bool)])
        self._rebuild()
        self._event("terrane_transfer", downgoing=down, overriding=over, n_parcels=int(moved.size),
                    n_new=int(len(add)), n_stacked=int(has.sum()), area_km2=float(m.area_km2[w].sum()),
                    volume_ratio_after_before=round(vol_after / max(vol_before, 1e-9), 4))
        return int(moved.size)
