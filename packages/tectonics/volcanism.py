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
        self.feeding = {}                  # 热点序号 -> 正在喂的火山序号
        self.events: list[dict] = []
        self.continental: list[dict] = []  # 热点在陆壳下方的时段（U29 用）

    # ---- 几何 -------------------------------------------------------------------------
    def positions(self, eng) -> np.ndarray:
        out = np.zeros((len(self.plate), 3))
        for q in np.unique(self.plate):
            s = self.plate == q
            if int(q) in eng.R:
                out[s] = self.xref[s] @ eng.R[int(q)].T
        return out

    def cap_km(self) -> np.ndarray:
        smax = np.array([self.hs[i].summit_max_m for i in self.hs_idx]) / 1000.0
        return np.minimum(self.p.H_cap_km, np.maximum(self.depth0, 0.0) + smax)

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
                self.events.append({"t": round(eng.t, 2), "kind": "volcano_subducted", "hotspot": self.hs[self.hs_idx[i]].id})

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
                self._spawn(k, q, x @ eng.R[q], t, h.flux * dt, d0)
                pos = np.vstack([pos, x])
                continue
            d0 = -float(self._base_elev(eng, h.xyz[None, :])[0]) / 1000.0
            self._spawn(k, q, h.xyz @ eng.R[q], t, h.flux * dt, d0)
            pos = np.vstack([pos, h.xyz])
        self._surface(eng, dt)

    def _spawn(self, k: int, q: int, xref: np.ndarray, t: float, vol: float, depth0: float = 5.0):
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
        er = above & ~np.isin(a, list(self.feeding.values()))
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
                "hotspot": np.array([self.hs[i].id for i in self.hs_idx[a]], dtype=object),
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
