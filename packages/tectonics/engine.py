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

from .crust import (CONT, OCEAN, ORIGIN_ARC, ORIGIN_OCEAN, ORO_ARC, ORO_COLL, ORO_NONE, ORO_RIFT, Crust, Params,
                    elevation)
from .geometry import R_KM, chord_from_km, km_from_chord, orthonormalize, smoothstep, unit, wrap
from .mesh import KD_WORKERS, Mesh


def arc_profile(d_km, peak_km, max_km):
    up = smoothstep(d_km / peak_km)
    down = 1.0 - smoothstep((d_km - peak_km) / (max_km - peak_km))
    return np.where(d_km < peak_km, up, down)


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
        self._surface(p.dt)
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
        self._collapse(cont)

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
                if ring.size == 0:          # 没有可填的外围（孤立地块）：多余体积留在原处
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
                "origin": v.origin.copy(), "oro": v.oro.copy(), "oro_t": v.oro_t.copy(),
                "elev": elevation(v, self.p, tr), "omega": {q: self.omega[q].copy() for q in self.plates}}

    # ---- 离散事件：裂解与地体转移（只在提交后进行）------------------------------------------
    def split_plate(self, parent: int, s1: np.ndarray, s2: np.ndarray, axis: np.ndarray, new_id: int,
                    rng: np.random.Generator, seg_km=(200.0, 600.0), offset_km=(30.0, 150.0),
                    p_extra_offset: float = 0.3, margin_km: float = 200.0, min_frac: float = 0.05,
                    omega_new: np.ndarray | None = None) -> np.ndarray | None:
        """把板块 parent 沿 s1、s2 的垂直平分线附近裂开，s2 一侧成为新板块 new_id（继承 parent 的参照系，不重新取样）。

        裂谷线改成洋脊—转换断层阶梯：以相对运动欧拉极 axis 为极，洋脊段沿经线、转换断层沿纬线；阶梯跟随裂谷总体走向，
        只有 p_extra_offset 比例的段再加随机错距 offset_km。段长 seg_km 随机。新陆缘在 margin_km 内按距离减薄。
        成功返回新板块的单元，失败（新板块太小或太大）返回 None。"""
        assert self.steps_since_commit == 0, "裂解只在提交后进行"
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
        ee = local[self.pedges]
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
        xm = x[bnd_local].mean(0) if bnd_local.size else s1 + s2
        self._event("rift", parent=parent, new_plate=new_id, n_parcels=int(keep.size), n_segments=n_seg,
                    **_lat_lon(xm))
        return keep

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
