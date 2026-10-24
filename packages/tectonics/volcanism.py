"""火山建造（1.2d U27）：地幔参照系里固定的热点，在上方的板块上长出火山；火山随板块漂走，露出海面的部分被剥蚀，
海底随板块变老而沉降，老火山沉成海山，热带的沉到海面时由珊瑚礁顶在海面上成为环礁（达尔文的环礁成因）。

表示：火山是点要素，挂在所在板块的参照系上（xref = R_p^T · 当前位置），不占网格（真实火山岛大多十几到几十公里宽，
小于 L7 格点）。所在的地块单元被俯冲消耗时火山随之消失；所在洋壳整块转给别的板块（U26 的围裙转移）时火山跟着换板块。

高度：火山体按圆锥（水下平均坡度 slope_deg）由体积换算高度 H = (3V·tan²α/π)^(1/3)；高度有上限：高出周围海底不超过
H_cap_km，顶部高出海面不超过热点强弱对应的 summit_max_m（强 4000 m 夏威夷式，中 2000 m 加拉帕戈斯、留尼汪式，
弱 800 m 圣赫勒拿、阿森松式）。火山长满后热点在旁边另起一座卫星火山（慢速板块上的岛群：佛得角、加那利、加拉帕戈斯式）。
顶部高程 = 所在单元的高程（洋底按板块年龄的 GDH1 水深）+ H − 累计剥蚀。正在被热点喂的火山不剥蚀（新熔岩不断覆盖），
高度上限跟着当前海底深度走；离开热点后冻结，露出海面的部分按 tau_erosion_myr 剥蚀到海面（地球上夏威夷群岛几百万年里
由 4 km 降到几百米，其中一部分是挠曲沉降，这里没有挠曲，合在剥蚀里）。
海面处的岛半径 = 顶部高出海面的高度 / tan(水上坡度 air_slope_deg)；大于约半个格点的进海陆掩膜，其余只作符号
（与终图的约定相同）。冰岛那样的洋中脊上的厚地壳台地不是单座火山，属地幔柱高原（U29）。

热点在上方是陆壳时不长火山岛（大陆上的溢流玄武岩、动力地形属 U29），只记录经过的时刻。
method_class：procedural_constrained（热点位置与活动时段由设计给出，火山大小按参数化的岩浆通量）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crust import CONT, elevation
from .geometry import R_KM, km_from_chord, unit, xyz_to_lonlat

SEAMOUNT, ISLAND, ATOLL = 0, 1, 2


@dataclass
class VolcanoParams:
    slope_deg: float = 5.0             # 火山体水下平均坡度（夏威夷式盾状火山 5–10°，取平缓一侧）
    air_slope_deg: float = 6.0         # 水上部分的平均坡度，决定海面处的岛半径（夏威夷岛约 6°：顶高 4 km、半径约 40 km）
    H_cap_km: float = 10.0             # 火山体高出周围海底的上限（夏威夷岛约 9–10 km）
    spacing_km: float = 70.0           # 板块挪开这么远后热点另起一座新火山（夏威夷火山间距 50–100 km）
    tau_erosion_myr: float = 3.0       # 露出海面部分的剥蚀时间常数
    atoll_max_abs_lat: float = 28.0    # 珊瑚礁大致只在热带与副热带长成环礁（与终图出图的约定相同）
    # 岩浆通量（km³/Myr）与顶部高出海面的上限（m），按热点强弱：夏威夷近几百万年约 2×10⁵ km³/Myr、顶 4.2 km；
    # 留尼汪顶 3.1 km、加拉帕戈斯 1.7 km；圣赫勒拿 0.8 km、阿森松 0.9 km、布韦 0.8 km
    flux_km3_myr: dict = field(default_factory=lambda: {"strong": 1.2e5, "medium": 2.0e4, "weak": 5.0e3})
    summit_max_m: dict = field(default_factory=lambda: {"strong": 4000.0, "medium": 2000.0, "weak": 800.0})
    satellite_frac: float = 0.5        # 火山长满后，卫星火山离它 satellite_frac × spacing_km
    # U28 岛弧火山：俯冲带上盘洋壳上、离俯冲前缘 arc_offset_km 的一条带（岛弧—海沟间距一般 100–200 km）
    arcs: bool = False
    arc_offset_km: float = 120.0
    arc_band_km: float = 50.0
    arc_spacing_km: float = 70.0
    arc_flux_per_cm_yr: float = 600.0  # 每座弧火山每 1 cm/yr 俯冲速度的岩浆通量（km³/Myr）；地球岛弧的岩浆增生约 30–90 km³/km/Myr
    arc_summit_max_m: float = 1500.0   # 伊豆、马里亚纳、千岛、小安的列斯、瓦努阿图的弧火山岛大多高出海面 0.3–2 km
    arc_min_T_km: float = 20.0         # 只在已增厚到这么厚的弧地壳上长火山（成熟洋内弧 20–35 km）。不限时大洋里所有
                                       # 洋壳对洋壳的板块边界都长弧火山，第一次试算长出 1 万多座、多出 3.6 百万 km² 陆地
    arc_min_v_cm_yr: float = 1.5       # 俯冲速度太慢不长弧火山
    arc_min_cont_km: float = 250.0     # 只长洋内弧：离上盘自己的陆壳这么远以外。大陆边缘的火山弧（安第斯、喀斯喀特）长在
                                       # 陆上，海岸外没有一串火山岛；不限时远方大陆西岸、洲 B 东岸、洲 C 西岸外都排出一条火山岛


@dataclass
class Hotspot:
    id: str
    name: str
    xyz: np.ndarray                    # 地幔参照系（绝对参照系）里的位置
    t_on: float                        # 开始活动（Ma）
    t_off: float                       # 停止活动（Ma，0 = 至今）
    flux: float                        # km³/Myr
    summit_max_m: float = 4000.0       # 顶部高出海面的上限（m）


class VolcanoField:
    """所有火山的状态（每座一行）。"""

    def __init__(self, hotspots: list[Hotspot], p: VolcanoParams | None = None):
        self.hs = hotspots
        self.p = p or VolcanoParams()
        n0 = 0
        self.plate = np.zeros(n0, np.int64)
        self.xref = np.zeros((n0, 3))
        self.t_birth = np.zeros(n0)
        self.vol = np.zeros(n0)
        self.eroded = np.zeros(n0)
        self.reef = np.zeros(n0, bool)
        self.was_island = np.zeros(n0, bool)
        self.hs_idx = np.zeros(n0, np.int64)
        self.alive = np.zeros(n0, bool)
        self.depth0 = np.zeros(n0)         # 诞生时所在海底的深度（km，正值向下），决定高度上限
        self.smax = np.zeros(n0)           # 顶部高出海面的上限（m）
        self.src = np.zeros(n0, np.int8)   # 0 = 热点，1 = 岛弧
        self.arc_active = np.zeros(n0, bool)  # 这一步正在被俯冲带喂的弧火山
        self.feeding = {}                  # 热点序号 -> 正在喂的火山序号
        self.events: list[dict] = []
        self.continental: list[dict] = []  # 热点在陆壳下方的时段（U29 用）
        # 岛弧火山只长在设计的俯冲带上（U28 第二轮，终图倒推用；None = 不限）：
        # arc_pairs = 允许的（下行, 上盘）板块对键；上盘是 arc_cont_plates 里的陆块时，还要求弧所在洋壳在 arc_protected_uids 里
        # （岛弧系统划给陆块的那片洋壳），或下行板块也是陆块（主内海式）
        self.arc_pairs: set | None = None
        self.arc_cont_plates: set = set()
        self.arc_protected_uids = np.zeros(0, np.int64)

    # ---- 几何 -------------------------------------------------------------------------
    def positions(self, eng) -> np.ndarray:
        out = np.zeros((len(self.plate), 3))
        for q in np.unique(self.plate):
            s = self.plate == q
            if int(q) in eng.R:
                out[s] = self.xref[s] @ eng.R[int(q)].T
        return out

    def cap_km(self) -> np.ndarray:
        return np.minimum(self.p.H_cap_km, np.maximum(self.depth0, 0.0) + self.smax / 1000.0)

    def height_km(self) -> np.ndarray:
        tan2 = np.tan(np.radians(self.p.slope_deg)) ** 2
        return np.minimum((3.0 * self.vol * tan2 / np.pi) ** (1.0 / 3.0), self.cap_km())

    def cone_volume_km3(self, H: np.ndarray) -> np.ndarray:
        return np.pi * H ** 3 / (3.0 * np.tan(np.radians(self.p.slope_deg)) ** 2)

    # ---- 所在地块单元 ---------------------------------------------------------------------
    @staticmethod
    def _cover(eng, pts: np.ndarray):
        """pts 处由哪块板块的哪个单元覆盖（与 Engine.cover_fixed 相同的优先：陆壳、较厚、较年轻）。返回 (单元, 板块)。"""
        best = np.full(len(pts), -1, np.int64)
        score = np.full(len(pts), -np.inf)
        c = eng.c
        for q in eng.plates:
            par = eng.covering(q, pts)
            ok = par >= 0
            pp = np.maximum(par, 0)
            s = np.where(ok, c.ctype[pp] * 1e6 + c.T[pp] * 1e3 - c.age[pp], -np.inf)
            better = ok & (s > score)
            best[better], score[better] = par[better], s[better]
        return best, np.where(best >= 0, c.plate[np.maximum(best, 0)], -1)

    def _rehome(self, eng):
        """所在板块不再覆盖的火山：所在洋壳整块换了板块（转移）就跟着换，否则（俯冲消耗）消失。"""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return
        pos = self.positions(eng)
        lost = []
        for q in np.unique(self.plate[idx]):
            sel = idx[self.plate[idx] == q]
            if int(q) not in eng.R:
                lost.append(sel)
                continue
            par = eng.covering(int(q), pos[sel])
            lost.append(sel[par < 0])
        lost = np.concatenate(lost) if lost else np.zeros(0, np.int64)
        if lost.size == 0:
            return
        par, pl = self._cover(eng, pos[lost])
        c = eng.c
        pp = np.maximum(par, 0)
        # 换板块的条件：被同样老（形成时间不晚于火山）的洋壳覆盖——说明是这片洋壳整块换了板块，而不是火山被俯冲、上面盖了别的板块
        moved = (par >= 0) & (c.ctype[pp] != CONT) & (c.t_form[pp] >= self.t_birth[lost] - 1e-6)
        for k in np.flatnonzero(moved):
            i, q = lost[k], int(pl[k])
            self.plate[i] = q
            self.xref[i] = pos[i] @ eng.R[q]
        gone = lost[~moved]
        if gone.size:
            self.alive[gone] = False
            for i in gone:
                self.events.append({"t": round(eng.t, 2), "kind": "volcano_subducted", "hotspot": self._label(i)})

    # ---- 一步 ---------------------------------------------------------------------------
    def step(self, eng, dt: float = 1.0):
        """在 eng.step 之后调用（eng.t 已是这一步结束的时刻）。"""
        t = eng.t
        self._rehome(eng)
        pts = np.array([h.xyz for h in self.hs])
        par, pl = self._cover(eng, pts)
        pos = self.positions(eng)
        for k, h in enumerate(self.hs):
            if not (h.t_off - 1e-9 <= t <= h.t_on + 1e-9) or par[k] < 0:
                self.feeding.pop(k, None)
                continue
            if eng.c.ctype[par[k]] == CONT:
                self.feeding.pop(k, None)
                self.continental.append({"t": round(t, 2), "hotspot": h.id, "plate": int(pl[k])})
                continue
            q = int(pl[k])
            cur = self.feeding.get(k)
            if cur is not None and self.alive[cur] and self.plate[cur] == q \
                    and km_from_chord(np.linalg.norm(pos[cur] - h.xyz)) < self.p.spacing_km:
                self.eroded[cur] = 0.0         # 正在喂的火山：新熔岩不断覆盖，不剥蚀
                # 高度上限跟着当前海底深度走（活动火山一直长到高出海面 summit_max_m）；离开热点后冻结，随板块下沉
                self.depth0[cur] = -float(self._base_elev(eng, pos[cur][None, :])[0]) / 1000.0
                full = float(self.cone_volume_km3(self.cap_km()[cur:cur + 1])[0])
                # 长到上限的 95% 就算满（板块变老、海底变深时上限每步略升，不留余量会永远"差一点点没满"）
                if self.vol[cur] < 0.95 * full:
                    self.vol[cur] = min(self.vol[cur] + h.flux * dt, full)
                    continue
                # 长满了：在旁边另起一座卫星火山（方向按黄金角轮换，岛群均匀散开）
                n = int((self.hs_idx == k).sum())
                ang = np.radians(137.5 * n)
                e1 = unit(np.cross([0.0, 0.0, 1.0], h.xyz)) if abs(h.xyz[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                e2 = np.cross(h.xyz, e1)
                d = self.p.satellite_frac * self.p.spacing_km / R_KM
                x = unit(np.cos(d) * h.xyz + np.sin(d) * (np.cos(ang) * e1 + np.sin(ang) * e2))
                d0 = -float(self._base_elev(eng, x[None, :])[0]) / 1000.0
                self._spawn(k, q, x @ eng.R[q], t, h.flux * dt, d0, h.summit_max_m)
                pos = np.vstack([pos, x])
                continue
            d0 = -float(self._base_elev(eng, h.xyz[None, :])[0]) / 1000.0
            self._spawn(k, q, h.xyz @ eng.R[q], t, h.flux * dt, d0, h.summit_max_m)
            pos = np.vstack([pos, h.xyz])
        if self.p.arcs:
            self._arc_step(eng, dt)
        self._surface(eng, dt)

    def _label(self, i: int) -> str:
        return "ARC" if self.src[i] == 1 else self.hs[self.hs_idx[i]].id

    def _arc_step(self, eng, dt: float):
        """岛弧火山（U28）：最近 3 Myr 的俯冲前缘后方 arc_offset_km 的上盘洋壳带上，按间距补火山；带内的火山按俯冲速度长。"""
        from scipy.spatial import cKDTree

        from .crust import OCEAN
        from .geometry import chord_from_km

        self.arc_active[:] = False
        fr = eng.recent_fronts()
        if fr is None:
            return
        X, S, O, V = fr
        c = eng.c
        pos_all = eng.positions()
        lo_km, hi_km = self.p.arc_offset_km - 0.5 * self.p.arc_band_km, self.p.arc_offset_km + 0.5 * self.p.arc_band_km
        t = eng.t
        for o in np.unique(O):
            o = int(o)
            if o not in eng.R:
                continue
            sel = O == o
            tree = cKDTree(X[sel])
            # 带内的上盘洋壳单元
            u = np.flatnonzero(eng.alive & (c.plate == o) & (c.ctype == OCEAN) & (c.T >= self.p.arc_min_T_km))
            if u.size == 0:
                continue
            d, kk = tree.query(pos_all[u], distance_upper_bound=chord_from_km(hi_km), workers=-1)
            ok = np.isfinite(d)
            dk = np.where(ok, km_from_chord(np.where(ok, d, 0.0)), np.inf)
            band = ok & (dk >= lo_km) & (dk <= hi_km)
            band[ok] &= V[sel][kk[ok]] >= self.p.arc_min_v_cm_yr
            if self.arc_pairs is not None and band.any():     # 只长设计的俯冲带上的弧（见 __init__）
                b_idx = np.flatnonzero(band)
                sub = S[sel][kk[b_idx]]
                okp = np.isin(np.minimum(sub, o) * 1000 + np.maximum(sub, o), list(self.arc_pairs))
                if o in self.arc_cont_plates:
                    okp &= np.isin(c.uid[u[b_idx]], self.arc_protected_uids) | np.isin(sub, list(self.arc_cont_plates))
                band[b_idx[~okp]] = False
            cont_o = np.flatnonzero(eng.alive & (c.plate == o) & (c.ctype != OCEAN))
            if cont_o.size and band.any():     # 只长洋内弧：离上盘自己的陆壳 arc_min_cont_km 以外
                dc, _ = cKDTree(pos_all[cont_o]).query(pos_all[u[band]], workers=-1)
                b_idx = np.flatnonzero(band)
                band[b_idx[km_from_chord(dc) < self.p.arc_min_cont_km]] = False
            if not band.any():
                continue
            cand, vcand = pos_all[u[band]], V[sel][kk[band]]
            # 已有的、在这块板块上的活弧火山
            mine = np.flatnonzero(self.alive & (self.src == 1) & (self.plate == o))
            mpos = self.positions(eng)[mine] if mine.size else np.zeros((0, 3))
            # 带内已有的火山：按俯冲速度长
            if mine.size:
                dm, km_ = tree.query(mpos, distance_upper_bound=chord_from_km(hi_km), workers=-1)
                okm = np.isfinite(dm)
                dmk = np.where(okm, km_from_chord(np.where(okm, dm, 0.0)), np.inf)
                inb = okm & (dmk >= lo_km - 20.0) & (dmk <= hi_km + 20.0)
                grow = mine[inb]
                v = V[sel][km_[inb]]
                full = self.cone_volume_km3(self.cap_km()[grow])
                self.vol[grow] = np.minimum(self.vol[grow] + self.p.arc_flux_per_cm_yr * np.clip(v, 0, 15) * dt, full)
                self.eroded[grow] = 0.0
                self.arc_active[grow] = True
            # 补新火山：离已有火山（及本步新补的）至少 arc_spacing_km
            sp = chord_from_km(self.p.arc_spacing_km)
            if mine.size:                      # 先剔掉已有火山附近的候选（向量化），剩下的才逐个放
                dn, _ = cKDTree(mpos).query(cand, workers=-1)
                keep = dn >= sp
                cand, vcand = cand[keep], vcand[keep]
            placed = []
            order = np.argsort(-vcand)
            for i in order:
                x = cand[i]
                if placed and np.min(np.linalg.norm(np.asarray(placed) - x, axis=1)) < sp:
                    continue
                d0 = -float(self._base_elev(eng, x[None, :])[0]) / 1000.0
                self._spawn(-1, o, x @ eng.R[o], t, self.p.arc_flux_per_cm_yr * float(np.clip(vcand[i], 0, 15)) * dt,
                            d0, self.p.arc_summit_max_m, src=1)
                self.arc_active[-1] = True
                placed.append(x)

    def _spawn(self, k: int, q: int, xref: np.ndarray, t: float, vol: float, depth0: float = 5.0,
               smax: float = 4000.0, src: int = 0):
        self.plate = np.append(self.plate, q)
        self.xref = np.vstack([self.xref, xref])
        self.t_birth = np.append(self.t_birth, t)
        self.vol = np.append(self.vol, vol)
        self.eroded = np.append(self.eroded, 0.0)
        self.reef = np.append(self.reef, False)
        self.was_island = np.append(self.was_island, False)
        self.hs_idx = np.append(self.hs_idx, k)
        self.alive = np.append(self.alive, True)
        self.depth0 = np.append(self.depth0, depth0)
        self.smax = np.append(self.smax, smax)
        self.src = np.append(self.src, np.int8(src))
        if len(self.arc_active) < len(self.plate):
            self.arc_active = np.append(self.arc_active, False)
        if k >= 0:
            self.feeding[k] = len(self.plate) - 1

    def _base_elev(self, eng, pos: np.ndarray) -> np.ndarray:
        par, _ = self._cover(eng, pos)
        e = np.full(len(pos), -5000.0)
        ok = par >= 0
        if ok.any():
            e[ok] = elevation(eng.c.gather(par[ok]), eng.p)
        return e

    def summit_m(self, eng) -> np.ndarray:
        """顶部高程（m，未计珊瑚礁）。"""
        z = np.full(len(self.plate), np.nan)
        a = np.flatnonzero(self.alive)
        if a.size:
            z[a] = self._base_elev(eng, self.positions(eng)[a]) + 1000.0 * (self.height_km()[a] - self.eroded[a])
        return z

    def _surface(self, eng, dt: float):
        a = np.flatnonzero(self.alive)
        if a.size == 0:
            return
        z = self.summit_m(eng)[a]
        above = z > 0
        self.was_island[a[above]] = True
        # 离开热点后，露出海面的部分剥蚀（只削海面以上，削到海面为止）；正在喂的不剥蚀
        er = above & ~np.isin(a, list(self.feeding.values())) & ~self.arc_active[a]
        self.eroded[a[er]] += (z[er] / 1000.0) * min(1.0, dt / self.p.tau_erosion_myr)
        lat = np.degrees(np.arcsin(np.clip(self.positions(eng)[a][:, 2], -1, 1)))
        tropic = np.abs(lat) < self.p.atoll_max_abs_lat
        # 沉到海面以下：在热带由珊瑚礁顶在海面上（环礁）；离开热带的环礁淹没成平顶海山
        self.reef[a[~above & tropic & self.was_island[a]]] = True
        self.reef[a[~tropic]] = False
        self.reef[a[above]] = False

    # ---- 输出 ---------------------------------------------------------------------------
    def state(self, eng) -> dict:
        a = np.flatnonzero(self.alive)
        pos = self.positions(eng)[a]
        lon, lat = xyz_to_lonlat(pos)
        z = self.summit_m(eng)[a]
        kind = np.where(z > 0, ISLAND, np.where(self.reef[a], ATOLL, SEAMOUNT))
        r = np.where(z > 0, z / 1000.0 / np.tan(np.radians(self.p.air_slope_deg)), 0.0)
        return {"lon": np.asarray(lon, float), "lat": np.asarray(lat, float), "plate": self.plate[a].copy(),
                "hotspot": np.array([self._label(i) for i in a], dtype=object),
                "t_birth": self.t_birth[a].copy(), "height_km": self.height_km()[a], "summit_m": z,
                "kind": kind.astype(np.int8), "radius_km": r}


def island_mask(grid, state: dict, min_radius_km: float) -> np.ndarray:
    """火山岛（半径 ≥ min_radius_km）画进经纬度格网的海陆掩膜（圆）。"""
    out = np.zeros(grid.shape, bool)
    sel = (state["kind"] == ISLAND) & (state["radius_km"] >= min_radius_km)
    for lo, la, r in zip(state["lon"][sel], state["lat"][sel], state["radius_km"][sel]):
        c = unit(np.array([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                           np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))]))
        out |= (grid.xyz @ c) >= np.cos(r / R_KM)
    return out
