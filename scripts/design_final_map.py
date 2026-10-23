"""终图设计 -> 世界格网、A 层核对与设计图（PROGRESS U22，CLAUDE.md §4）。

读 config/design/final_map_<版本>.yaml（开发者按用户构想画的终图：只含几百公里以上的要素），
栅格化成 2048x1024 的海陆掩膜与示意高程，按 A 层核对，出设计图给用户看。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\design_final_map.py --design v1

输出：
- data/worlds/design_<版本>/design_<版本>_<分辨率>.nc（不进 Git，可重建）与 params.json（面积与核对结果，进 Git）；
- figures/design/<版本>/（不进 Git）：01 终图、02 要素与计划成因、03 A 层结构、04 与同样概括程度的地球对照、05 南极区。

method_class：海陆与要素 = user_override（设计）；示意高程 = visual_only（只为看图，不是模拟结果）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from mythic.figures.world import code_version  # noqa: E402
from mythic.metrics.grid import LatLonGrid  # noqa: E402
from mythic.metrics.spherical import distance_to_set_km  # noqa: E402

R_KM = 6371.0
KM_PER_DEG = np.pi * R_KM / 180.0
NE_LAND = ROOT / "data" / "earth_reference" / "raw" / "natural_earth" / "ne_10m_land.zip"
ATOLL_MAX_ABS_LAT = 28.0     # 珊瑚礁大致只在热带与副热带长成环礁
MIN_ISLAND_RADIUS_KM = 9.0   # 小于约半个格点的岛不进海陆掩膜（只作符号）


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 球面几何小工具
# ---------------------------------------------------------------------------
def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def to_lonlat(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])), np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


def densify(line, step_km=10.0):
    """折线（[经, 纬] 列表）沿大圆加密，返回单位向量 (n, 3) 与累计距离 (n,)。"""
    pts = [unit(*line[0])]
    for a, b in zip(line[:-1], line[1:]):
        pa, pb = unit(*a), unit(*b)
        om = np.arccos(np.clip(pa @ pb, -1, 1))
        n = max(2, int(om * R_KM / step_km) + 1)
        t = np.linspace(0, 1, n)[1:, None]
        pts.append((np.sin((1 - t) * om) * pa + np.sin(t * om) * pb) / np.sin(om))
    p = np.vstack([np.atleast_2d(x) for x in pts])
    seg = np.arccos(np.clip(np.sum(p[1:] * p[:-1], axis=1), -1, 1)) * R_KM
    return p, np.concatenate([[0.0], np.cumsum(seg)])


def chord_km(c):
    return 2 * R_KM * np.arcsin(np.clip(c / 2, 0, 1))


def lonlat_mask(grid: LatLonGrid, poly) -> np.ndarray:
    """经纬度多边形（不跨 180° 经线、不含极点）的栅格掩膜。边在经纬度平面上取直线，设计精度足够。"""
    poly = np.asarray(poly, float)
    lo0, lo1 = poly[:, 0].min(), poly[:, 0].max()
    la0, la1 = poly[:, 1].min(), poly[:, 1].max()
    rows = np.flatnonzero((grid.lat >= la0 - 1) & (grid.lat <= la1 + 1))
    cols = np.flatnonzero((grid.lon >= lo0 - 1) & (grid.lon <= lo1 + 1))
    m = np.zeros(grid.shape, bool)
    if rows.size == 0 or cols.size == 0:
        return m
    LO, LA = np.meshgrid(grid.lon[cols], grid.lat[rows])
    inside = MPath(poly).contains_points(np.c_[LO.ravel(), LA.ravel()]).reshape(LO.shape)
    m[np.ix_(rows, cols)] = inside
    return m


def line_distance_km(grid: LatLonGrid, line, query: np.ndarray, max_km: float) -> np.ndarray:
    """query 格点到折线的大圆距离（km），超过 max_km 记 inf。"""
    p, _ = densify(line)
    out = np.full(grid.shape, np.inf)
    d, _ = cKDTree(p).query(grid.xyz[query], distance_upper_bound=2 * np.sin(max_km / R_KM / 2) * 1.0001,
                            workers=-1)
    out[query] = np.where(np.isfinite(d), chord_km(np.where(np.isfinite(d), d, 0)), np.inf)
    return out


def ellipse_mask(grid: LatLonGrid, center_v, along_v, a_km, b_km) -> np.ndarray:
    """以 center_v 为中心、长轴沿 along_v 方向的椭圆（局部切平面上），处理经度环绕与极区。"""
    lon_c, lat_c = to_lonlat(center_v)
    r_deg = a_km / KM_PER_DEG + 0.5
    rows = np.flatnonzero(np.abs(grid.lat - lat_c) <= r_deg)
    m = np.zeros(grid.shape, bool)
    if rows.size == 0:
        return m
    xyz = grid.xyz[rows]                                      # (nr, nx, 3)
    e1 = along_v - (along_v @ center_v) * center_v
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center_v, e1)
    d = xyz - center_v
    near = (xyz @ center_v) > np.cos((a_km + 50) / R_KM)
    x = (d @ e1) * R_KM
    y = (d @ e2) * R_KM
    m[rows] = near & ((x / a_km) ** 2 + (y / b_km) ** 2 <= 1.0)
    return m


def smooth_noise(n: int, step_km: float, amps, sigmas, rng: np.random.Generator) -> np.ndarray:
    """环形的平滑随机起伏（白噪声做高斯平滑后按标准差归一，再乘幅度）。不用周期正弦：等距的波纹和直线一样假。"""
    freq = np.fft.rfftfreq(n, d=step_km)
    off = np.zeros(n)
    for a, sigma in zip(amps, sigmas):
        x = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) * np.exp(-0.5 * (2 * np.pi * freq * sigma) ** 2), n)
        off += a * x / (x.std() + 1e-12)
    return off


def polar_mask(grid: LatLonGrid, coast, rng: np.random.Generator | None = None, amps=(0.0, 0.0)) -> np.ndarray:
    """按经度给出海岸纬度的极地大陆（围绕南极一圈，周期插值）。
    rng 给出时海岸纬度加平滑随机起伏（amps 为 km，两档尺度约 220 与 80 km），去掉按 10° 插值的光滑弧线。"""
    c = np.asarray(sorted(coast), float)
    lon = np.concatenate([c[:, 0] - 360, c[:, 0], c[:, 0] + 360])
    lat = np.tile(c[:, 1], 3)
    coast_lat = np.interp(grid.lon, lon, lat)
    if rng is not None and max(amps) > 0:
        km_per_col = KM_PER_DEG * np.cos(np.radians(np.abs(coast_lat).mean())) * 360.0 / grid.nx
        coast_lat = coast_lat + smooth_noise(grid.nx, km_per_col, amps, (220.0, 80.0), rng) / KM_PER_DEG
    return grid.lat[:, None] <= coast_lat[None, :]


def feature_rng(fid: str) -> np.random.Generator:
    """每个要素自己的随机数（按 id 固定），改一个要素不会牵动别的要素。"""
    return np.random.default_rng(zlib.crc32(fid.encode("utf-8")))


def width_profile(widths, t, L):
    """宽度沿轴分段线性：widths 为标量，或 ≥ 2 个数（沿轴等距分布，第一个在起点、最后一个在终点）。"""
    w = np.atleast_1d(np.asarray(widths, float))
    if w.size == 1:
        return np.full_like(t, w[0], dtype=float)
    return np.interp(t / L, np.linspace(0, 1, w.size), w)


def meander(line, amp_km: float, rng: np.random.Generator):
    """让折线在局部切平面里左右缓慢摆动（两档波长 700–1300 km 与 250–450 km），去掉尺子画的直线感。"""
    p, s = densify(line, 20.0)
    if len(p) < 3:
        return line
    l1, l2 = rng.uniform(700, 1300), rng.uniform(250, 450)
    f1, f2 = rng.uniform(0, 2 * np.pi, 2)
    off = amp_km * (0.7 * np.sin(2 * np.pi * s / l1 + f1) + 0.3 * np.sin(2 * np.pi * s / l2 + f2))
    off *= np.sin(np.pi * np.clip(s / s[-1], 0, 1)) ** 0.5     # 两端不动
    tang = np.gradient(p, axis=0)
    nrm = np.cross(p, tang)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    q = p + nrm * (off / R_KM)[:, None]
    lon, lat = to_lonlat(q)
    return np.c_[lon, lat].tolist()


def spine_mask(grid: LatLonGrid, line, widths, rng: np.random.Generator | None = None,
               flat_start: bool = False) -> np.ndarray:
    """沿中轴线、宽度沿轴分段线性渐变的形状（端部为圆头）。
    rng 给出时宽度沿轴有两档缓慢起伏（波长 500–900 km 与 180–320 km，幅度约 11% 与 7%），避免完全光滑。
    flat_start：起点一端平切（不要圆头），用于冰架的前缘冰崖。"""
    p, s = densify(line, 8.0)
    L = max(s[-1], 1.0)
    half_max = float(np.max(widths)) * 0.6
    lon, lat = to_lonlat(p)
    pad = half_max / KM_PER_DEG + 0.5
    rows = np.flatnonzero((grid.lat >= lat.min() - pad) & (grid.lat <= lat.max() + pad))
    m = np.zeros(grid.shape, bool)
    if rows.size == 0:
        return m
    xyz = grid.xyz[rows].reshape(-1, 3)
    d, idx = cKDTree(p).query(xyz, distance_upper_bound=2 * np.sin(half_max / R_KM / 2), workers=-1)
    ok = np.isfinite(d)
    t = s[idx[ok]]
    w = width_profile(widths, t, L)
    if rng is not None:
        l1, l2 = rng.uniform(500, 900), rng.uniform(180, 320)
        f1, f2 = rng.uniform(0, 2 * np.pi, 2)
        w = w * (1 + 0.11 * np.sin(2 * np.pi * t / l1 + f1) + 0.07 * np.sin(2 * np.pi * t / l2 + f2))
    inside = np.zeros(xyz.shape[0], bool)
    inside[ok] = chord_km(d[ok]) <= w / 2
    if flat_start:
        inside[ok] &= idx[ok] > 0
    m[rows] = inside.reshape(rows.size, grid.nx)
    return m


def meander_polygon(poly, rng: np.random.Generator, amps=(70.0, 30.0), protect=()):
    """大陆粗轮廓的边沿法向缓慢起伏（两档波长 500–1100 km 与 180–350 km），去掉几百公里长的直边。
    protect：[(经, 纬, 半径 km), ...] 附近不动（海峡、地峡的宽度由设计决定，不能被起伏改掉）。"""
    loop = list(poly) + [poly[0]]
    p, s = densify(loop, 25.0)
    p, s = p[:-1], s[:-1]
    # 沿周长的平滑随机起伏
    n = len(p)
    off = smooth_noise(n, (s[-1] + 25.0) / n, amps, (220.0, 80.0), rng)
    for lon, lat, r in protect:
        d = np.arccos(np.clip(p @ unit(lon, lat), -1, 1)) * R_KM
        x = np.clip(d / r, 0, 1)
        off *= x * x * (3 - 2 * x)
    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    nrm = np.cross(p, tang)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    lon, lat = to_lonlat(p + nrm * (off / R_KM)[:, None])
    return np.c_[lon, lat]


# ---------------------------------------------------------------------------
# 栅格化
# ---------------------------------------------------------------------------
def build(design: dict, grid: LatLonGrid, seed: int = 1):
    rng = np.random.default_rng(seed)
    shape = grid.shape
    land = np.zeros(shape, bool)
    cont = np.full(shape, "", dtype=object)      # 只用于统计各洲面积与标注
    seeds = {c: unit(*np.asarray(v, float).T) for c, v in design.get("continent_seeds", {}).items()}

    def assign(m, c):
        if len(c) == 2 and all(k in seeds for k in c):   # 例如 "AB"：按最近的种子点分到 A 或 B
            xyz = grid.xyz[m]
            best = np.stack([(xyz @ seeds[k].T).max(axis=1) for k in c], axis=1).argmax(axis=1)
            cont[m] = np.array(list(c), dtype=object)[best]
        else:
            cont[m] = c

    protect = [tuple(x) for x in design.get("meander_protect", [])]
    for b in design.get("masses", []):
        amps = b.get("meander_km", [70, 30])
        poly = meander_polygon(b["poly"], feature_rng(b["id"]), amps, protect) if max(amps) > 0 else b["poly"]
        m = lonlat_mask(grid, poly)
        land |= m
        assign(m, b["continent"])
    pol = polar_mask(grid, design["polar"]["coast"], feature_rng("polar"),
                     design["polar"].get("meander_km", [0, 0]))
    land |= pol
    cont[pol] = "polar"
    for pn in design.get("peninsulas", []):
        m = spine_mask(grid, pn["line"], pn["width_km"], feature_rng(pn["id"])) & ~land
        land |= m
        assign(m, pn["continent"])
        if pn["continent"] == "polar":     # 极地大陆的半岛同样盖着冰
            pol |= m

    sea_masks = {}
    for g in design.get("gulfs", []):
        m = spine_mask(grid, g["line"], g["width_km"], feature_rng(g["id"]))
        sea_masks[g["id"]] = m
        land &= ~m
    for s in design.get("seas", []):
        m = lonlat_mask(grid, s["poly"])
        sea_masks[s["id"]] = m
        land &= ~m
    for t in design.get("straits", []):
        d = line_distance_km(grid, t["line"], land, t["half_width_km"] * 2)
        m = d <= t["half_width_km"]
        sea_masks[t["id"]] = m
        land &= ~m
    cont[~land] = ""

    # 湖：不从陆地掩膜里挖掉（与地球数据口径一致：Natural Earth 的陆地含坦噶尼喀湖这类湖），只在图上画成水面
    lake = np.zeros(shape, bool)
    salt = np.zeros(shape, bool)
    for lk in design.get("lakes", []):
        m = spine_mask(grid, lk["line"], lk["width_km"], feature_rng(lk["id"])) & land
        lake |= m
        if lk.get("salt"):
            salt |= m
    # 冰架：海湾里浮在海上的冰（掩膜里是海，图上画成冰）
    shelf = np.zeros(shape, bool)
    for sh in design.get("ice_shelves", []):
        shelf |= spine_mask(grid, sh["line"], sh["width_km"], feature_rng(sh["id"]), flat_start=True) & ~land

    island_peak = np.zeros(shape)
    symbols = []        # 不进掩膜的环礁、海山（只画符号）
    for isl in design.get("islands", []):
        kind = isl["type"]
        if kind in ("continental", "spine"):
            if kind == "continental":
                m = lonlat_mask(grid, isl["poly"]) & ~land
            else:
                m = spine_mask(grid, isl["line"], isl["width_km"], feature_rng(isl["id"])) & ~land
            land |= m
            island_peak[m] = np.maximum(island_peak[m], 900.0)
            cont[m] = "island"
            if isl.get("ice"):          # 冰盖覆盖的岛：图上画成冰
                pol |= m
            continue
        p, s = densify(isl["line"])
        L = s[-1]
        if kind == "arc":
            # 真实岛弧不是等距等大的虚线：岛的大小取对数正态（偶有大岛）、间距随机并偶有缺口（沉在水下的弧段）、
            # 岛位与长轴方向稍偏离弧线。每条岛弧用自己的随机数，改一条不牵动别的。
            frng = feature_rng(isl["id"])
            size = float(isl["size_km"])
            step = size * (1.0 + float(isl.get("gap", 0.3)))
            t = frng.uniform(0, step * 0.5)
            while t < L:
                i = int(np.searchsorted(s, t).clip(1, len(s) - 1))
                c, along = p[i], p[i] - p[i - 1]
                nrm = np.cross(c, along)
                nrm /= np.linalg.norm(nrm)
                c = c + nrm * (frng.normal(0.0, 0.3) * size / R_KM)
                c /= np.linalg.norm(c)
                along = along + nrm * np.linalg.norm(along) * frng.normal(0.0, 0.3)
                f = float(np.clip(np.exp(frng.normal(0.0, 0.5)), 0.35, 2.6))
                a = size * f * frng.uniform(0.6, 1.0) / 2
                b = max(a * frng.uniform(0.35, 0.7), 8.0)
                if a >= MIN_ISLAND_RADIUS_KM:
                    m = ellipse_mask(grid, c, along, a, b) & ~land
                    land |= m
                    island_peak[m] = np.maximum(island_peak[m], 1200.0 + 1400.0 * min(1.0, a / 50))
                    cont[m] = "island"
                t += step * frng.uniform(0.6, 1.4) * (frng.uniform(2.0, 3.5) if frng.random() < 0.12 else 1.0)
        elif kind == "hotspot":
            young = float(isl.get("young_km", 500))
            t = 0.0
            while t < L:
                i = int(np.searchsorted(s, t).clip(1, len(s) - 1))
                c, along = p[i], p[i] - p[i - 1]
                r = 55.0 * np.exp(-t / young) * rng.uniform(0.8, 1.15)
                lon_c, lat_c = to_lonlat(c)
                if r >= MIN_ISLAND_RADIUS_KM:
                    m = ellipse_mask(grid, c, along, r * 1.25, r * 0.85) & ~land
                    land |= m
                    island_peak[m] = np.maximum(island_peak[m], 600.0 + 3600.0 * r / 55.0)
                    cont[m] = "island"
                    symbols.append({"lon": float(lon_c), "lat": float(lat_c), "kind": "volcanic", "id": isl["id"]})
                elif abs(lat_c) <= ATOLL_MAX_ABS_LAT:
                    symbols.append({"lon": float(lon_c), "lat": float(lat_c), "kind": "atoll", "id": isl["id"]})
                else:
                    symbols.append({"lon": float(lon_c), "lat": float(lat_c), "kind": "seamount", "id": isl["id"]})
                t += (150.0 if t < young else 260.0) * rng.uniform(0.8, 1.2)
    return {"land": land, "cont": cont, "polar": pol & land, "island_peak": island_peak,
            "symbols": symbols, "sea_masks": sea_masks, "lake": lake, "salt": salt, "shelf": shelf & ~land}


def apply_options(design: dict, options: set[str]) -> dict:
    """按选项过滤要素：写了 option: X 的只在 X 打开时保留；写了 unless: X 的在 X 打开时去掉（被替换）。"""
    out = {}
    for k, v in design.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            v = [it for it in v if (it.get("option") is None or it["option"] in options)
                 and (it.get("unless") is None or it["unless"] not in options)]
        out[k] = v
    return out


def circle_lonlat(center, radius_km: float, n: int = 120) -> np.ndarray:
    """球面上以 center 为中心、半径 radius_km 的小圆（经纬度，用于画撞击构造的环）。"""
    c = unit(*center)
    a = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(c, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    t = np.linspace(0, 2 * np.pi, n)
    ang = radius_km / R_KM
    p = np.cos(ang) * c + np.sin(ang) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    lon, lat = to_lonlat(p)
    return np.c_[lon, lat]


def elevation(design: dict, grid: LatLonGrid, b: dict) -> np.ndarray:
    """示意高程（visual_only）：只为看图和让指标代码能跑，不是模拟结果。"""
    land = b["land"]
    d_land = np.nan_to_num(distance_to_set_km(grid, ~land, query=land))
    d_sea = np.nan_to_num(distance_to_set_km(grid, land, query=~land))
    elev = np.zeros(grid.shape)
    base = 120.0 + 380.0 * (1 - np.exp(-d_land / 500.0))
    elev[land] = base[land]
    def region(p, default_amps):
        """高原、盆地的范围（轮廓加平滑起伏）与离边缘的平滑渐变（0 -> 1，ramp_km 内，smoothstep）。"""
        amps = p.get("meander_km", default_amps)
        m = lonlat_mask(grid, meander_polygon(p["poly"], feature_rng(p["id"]), amps)) & land
        if not m.any():
            return m, None
        d_in = np.nan_to_num(distance_to_set_km(grid, ~m, query=m))
        x = np.clip(d_in / float(p.get("ramp_km", 150.0)), 0, 1)
        return m, x * x * (3 - 2 * x)

    # 高原（ramp_km 小 = 陡崖围着的平顶，大 = 动力地形那样的宽缓穹隆）
    for p in design.get("plateaus", []):
        m, ramp = region(p, [120.0, 45.0])
        if ramp is not None:
            elev[m] = np.maximum(elev[m], base[m] + (p["height_m"] - base[m]) * ramp[m])
    # 盆地（克拉通内部沉降、前陆盆地：比周围低平）；height_m < 0 的是低于海平面的陆地洼地（死海式）
    depr = np.zeros(grid.shape, bool)
    for p in design.get("basins", []):
        m, ramp = region(p, [150.0, 50.0])
        if ramp is not None:
            low = np.minimum(elev[m], p["height_m"])
            elev[m] = elev[m] + (low - elev[m]) * ramp[m]
            if p["height_m"] < 0:
                depr |= m
    # 山脉（高斯形横剖面，取最大值叠加）；profile: rift 的裂谷是两侧肩部隆起、中间谷底下陷
    ridge = np.zeros(grid.shape)
    trough = np.zeros(grid.shape)
    for mt in design.get("mountains", []):
        rng = feature_rng(mt["id"])
        wmax = float(np.max(mt["width_km"]))
        # 裂谷、断层谷只轻微摆动：谷底要对准设计线上的湖与洼地（摆 0.3 倍宽时谷肩会压到洼地上）
        line = meander(mt["line"], (0.06 if mt.get("profile") == "rift" else 0.3) * wmax, rng)
        p, s = densify(line, 8.0)
        L = max(s[-1], 1.0)
        q = land
        dc, idx = cKDTree(p).query(grid.xyz[q], distance_upper_bound=2 * np.sin(wmax / R_KM), workers=-1)
        ok = np.isfinite(dc)
        t = s[idx[ok]]
        # 宽度与山脊高度沿走向起伏，两端降低：山系分段、有鞍部，不是一根均匀的"牙膏"
        w = width_profile(mt["width_km"], t, L)
        l1, l2 = rng.uniform(400, 800), rng.uniform(150, 300)
        f = rng.uniform(0, 2 * np.pi, 4)
        w = w * (1 + 0.18 * np.sin(2 * np.pi * t / l1 + f[0]) + 0.10 * np.sin(2 * np.pi * t / l2 + f[1]))
        h = mt["height_m"] * (0.78 + 0.16 * np.sin(2 * np.pi * t / (l1 * 1.3) + f[2])
                              + 0.08 * np.sin(2 * np.pi * t / (l2 * 0.8) + f[3]))
        h *= 0.55 + 0.45 * np.sin(np.pi * np.clip(t / L, 0, 1)) ** 0.35
        r = np.zeros(int(q.sum()))
        dk = chord_km(dc[ok])
        if mt.get("profile") == "rift":
            r[ok] = h * np.exp(-((dk - 0.30 * w) / (0.15 * w)) ** 2)
            g = np.zeros(int(q.sum()))
            g[ok] = np.exp(-(dk / (0.11 * w)) ** 2)
            trough[q] = np.maximum(trough[q], g)
        else:
            r[ok] = h * np.exp(-(dk / (0.45 * w)) ** 2)
        cur = ridge[q]
        ridge[q] = np.maximum(cur, r)
    # 山脊从当前地面与基准面中较低者算起：高原上取两者最大（山不会叠在高原上翻倍），
    # 盆地里从盆底升起（v1 用 max(地面, 基准面 + 山脊)，把盆地整片抬回了基准面）
    surf = elev[land]
    elev[land] = np.maximum(surf, np.minimum(base[land], surf) + ridge[land])
    sub = land & (elev > 0)
    elev[sub] *= 1 - 0.5 * trough[sub]            # 裂谷谷底约为两肩外侧地面的一半高（海平面以下的洼地不动）
    # 撞击构造（示意）：环形山岭 + 中央峰（老的剥蚀得只剩低缓环岭）
    for x in design.get("impacts", []):
        c = unit(*x["center"])
        D = float(x["diameter_km"])
        d = chord_km(np.linalg.norm(grid.xyz - c, axis=-1))
        old = x.get("age_ma", 0) > 500
        rim, peak = (150.0, 120.0) if old else (300.0, 450.0)
        bump = rim * np.exp(-((d - 0.45 * D) / (0.07 * D)) ** 2) + peak * np.exp(-(d / (0.08 * D)) ** 2)
        elev[land] += bump[land]
    # 湖面比周围略低（示意），图上画成水面；洼地里的湖面在海平面以下
    lake = b.get("lake")
    if lake is not None and lake.any():
        elev[lake] = np.where(elev[lake] > 0, np.maximum(5.0, 0.8 * elev[lake]), elev[lake] - 20.0)
    # 极地冰盖：冰面高程随离岸距离升高（冰下山脉略微抬升冰面）
    pol = b["polar"]
    ice = 3300.0 * (1 - np.exp(-d_land / 450.0)) + 0.3 * ridge
    elev[pol] = np.maximum(elev[pol], ice[pol])
    # 岛屿
    isl = b["island_peak"] > 0
    elev[isl] = np.maximum(30.0, b["island_peak"][isl] * (1 - np.exp(-d_land[isl] / 25.0)))
    # 海洋：陆架 -> 陆坡 -> 深海
    shelf = -50.0 - 150.0 * np.clip(d_sea / 140.0, 0, 1)
    deep = -200.0 - 4300.0 * (1 - np.exp(-np.clip(d_sea - 140.0, 0, None) / 220.0))
    elev[~land] = np.where(d_sea <= 140.0, shelf, deep)[~land]
    keep = land & ~depr                             # 洼地可以低于海平面（海陆由掩膜决定，与地球数据同）
    elev[keep] = np.maximum(elev[keep], 5.0)
    elev[~land] = np.minimum(elev[~land], -20.0)
    return elev


def downsample(field: np.ndarray, f: int) -> np.ndarray:
    ny, nx = field.shape
    g = LatLonGrid.from_shape(field.shape)
    w = g.row_area_km2[:, None]
    num = (field * w).reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    den = np.broadcast_to(w, field.shape).reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    return num / den


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def label_points(design: dict):
    """每个要素的标注位置与文字。"""
    out = []
    for pn in design.get("peninsulas", []):
        p = np.asarray(pn["line"], float)
        out.append(("peninsula", pn["id"], pn["name"], *p[-1]))
    for g in design.get("gulfs", []):
        p = np.asarray(g["line"], float)
        out.append(("sea", g["id"], g["name"], *p[len(p) // 2]))
    for g in design.get("lakes", []):
        p = np.asarray(g["line"], float)
        out.append(("lake", g["id"], g["name"], *p[len(p) // 2]))
    for g in design.get("ice_shelves", []):
        p = np.asarray(g["line"], float)
        out.append(("shelf", g["id"], g["name"], *p[len(p) // 2]))
    for s in design.get("seas", []):
        p = np.asarray(s["poly"], float)
        out.append(("sea", s["id"], s["name"], *p.mean(axis=0)))
    for t in design.get("straits", []):
        p = np.asarray(t["line"], float)
        out.append(("strait", t["id"], t["name"], *p[len(p) // 2]))
    for m in design.get("mountains", []):
        p = np.asarray(m["line"], float)
        out.append(("mountain", m["id"], m["name"], *p[len(p) // 2]))
    for p_ in design.get("plateaus", []):
        p = np.asarray(p_["poly"], float)
        out.append(("plateau", p_["id"], p_["name"], *p.mean(axis=0)))
    for p_ in design.get("basins", []):
        p = np.asarray(p_["poly"], float)
        out.append(("basin", p_["id"], p_["name"], *p.mean(axis=0)))
    for i in design.get("islands", []):
        p = np.asarray(i.get("poly", i.get("line")), float)
        out.append((i["type"], i["id"], i["name"], *(p.mean(axis=0) if "poly" in i else p[len(p) // 2])))
    # yaml 里写了 label_at 的要素用指定位置（避开拥挤处）
    at = {}
    for key in ("peninsulas", "gulfs", "lakes", "ice_shelves", "seas", "straits", "mountains", "plateaus",
                "basins", "islands"):
        for it in design.get(key, []):
            if "label_at" in it:
                at[it["id"]] = it["label_at"]
    return [(k, f, n, *at[f]) if f in at else (k, f, n, lo, la) for k, f, n, lo, la in out]


def draw_specials(ax, design: dict, halo, PC, GEO, labels: bool = True):
    """"神怒"候选海湾（湾顶红色标记）与撞击构造（环）。"""
    for g in design.get("gulfs", []):
        if g.get("shenu"):
            lo, la = g["line"][-1]
            ax.plot(lo, la, marker="^", ms=11, color="#c0392b", mec="white", mew=1.2, transform=PC, zorder=6)
            if labels:
                ax.text(lo, la - 2.2, f"{g['name']}（神怒）", transform=PC, fontsize=9, weight="bold", color="#c0392b",
                        ha="center", va="top", path_effects=halo, zorder=6)
    for x in design.get("impacts", []):
        ring = circle_lonlat(x["center"], x["diameter_km"] / 2)
        ax.plot(ring[:, 0], ring[:, 1], color="#6c3483", lw=1.3, transform=GEO, zorder=6)
        if labels:
            ax.text(x["center"][0], x["center"][1] - x["diameter_km"] / 2 / KM_PER_DEG - 0.8, x["name"],
                    transform=PC, fontsize=8, color="#6c3483", ha="center", va="top", path_effects=halo, zorder=6)


def feature_points(design: dict) -> dict:
    """要素编号 -> 代表位置（经, 纬），06 图标原型位置用。"""
    pts = {fid: (lo, la) for _, fid, _, lo, la in label_points(design)}
    for m in design.get("masses", []):
        pts[m["id"]] = tuple(np.asarray(m["poly"], float).mean(axis=0))
    for lt in design.get("lithology", []):
        pts[lt["id"]] = tuple(np.asarray(lt["poly"], float).mean(axis=0))
    for x in design.get("impacts", []):
        pts[x["id"]] = tuple(x["center"])
    return pts


def figures(design: dict, grid: LatLonGrid, b: dict, elev: np.ndarray, metrics: dict, fields: dict, out: Path,
            ver: str, areas: dict):
    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from mythic.figures.style import setup_fonts, shaded_elevation

    setup_fonts()
    PC = ccrs.PlateCarree()
    lon0 = float(design.get("map_central_longitude", 30))
    proj = ccrs.EqualEarth(central_longitude=lon0)
    land = b["land"]
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    opt = f"（选项 {design['_options']}）" if design.get("_options") else ""

    def stamp(fig, extra=""):
        fig.text(0.01, 0.006, f"终图设计 {design['name']}{opt} · config/design/final_map_{design['_ver']}.yaml · "
                 f"scripts/design_final_map.py · 代码版本 {ver} · method_class：海陆与要素 user_override（设计），"
                 f"示意高程 visual_only（不是模拟结果）{extra}", fontsize=7.5, color="#555555")

    def base_rgb():
        rgb = shaded_elevation(elev, grid.pixel_km_equator, vert_exag=3.0, land=land)[..., :3]
        pol = b["polar"]
        shade = rgb[pol].mean(axis=1, keepdims=True)
        rgb[pol] = 0.25 * rgb[pol] + 0.75 * np.clip(0.55 + 0.6 * shade, 0, 1)   # 冰盖：偏白
        rgb[b["shelf"]] = np.array([0.88, 0.93, 0.97])                          # 冰架：浮在海上的冰
        rgb[b["lake"]] = np.array([0.36, 0.60, 0.82])                           # 湖面
        rgb[b["salt"]] = np.array([0.55, 0.80, 0.78])                           # 咸湖
        return rgb

    rgb = base_rgb()

    # ---- 01 终图 ----
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.84], projection=proj)
    ax.set_global()
    ax.imshow(rgb, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(3000, 1500),
              interpolation="nearest")
    ax.contour(grid.lon, grid.lat, land.astype(float), levels=[0.5], colors="#333333", linewidths=0.45,
               transform=PC)
    if b["lake"].any():
        ax.contour(grid.lon, grid.lat, b["lake"].astype(float), levels=[0.5], colors="#24507a", linewidths=0.4,
                   transform=PC)
    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#777777", lw=0.3, alpha=0.6)
    sym = b["symbols"]
    for kind, mk, col, ms in (("atoll", "o", "#1f8f8f", 4.5), ("seamount", ".", "#6c8fb3", 3.0)):
        pts = [(s["lon"], s["lat"]) for s in sym if s["kind"] == kind]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, mk, mfc="none" if kind == "atoll" else col, mec=col, ms=ms, mew=1.0, transform=PC,
                    linestyle="none")
    big = {"A": "洲 A", "B": "洲 B", "C": "洲 C", "D": "洲 D", "far": "远方大陆"}
    cont_at = design.get("continent_label_at", {})
    for c, name in big.items():
        m = b["cont"] == c
        v = (grid.xyz * (grid.cell_area_km2 * m)[..., None]).sum(axis=(0, 1))
        lo, la = cont_at[c] if c in cont_at else to_lonlat(v)
        ax.text(lo, la, f"{name}\n{areas['continents_km2'].get(c, 0) / 1e6:.0f} 百万 km²", transform=PC,
                fontsize=15, weight="bold", ha="center", va="center", color="#3a2a10", path_effects=halo)
    style = {"sea": ("#0b3d6e", 9.5), "strait": ("#0b3d6e", 8.5), "plateau": ("#5a3d1e", 9),
             "lake": ("#0b3d6e", 8), "shelf": ("#2c4a6b", 8.5), "basin": ("#2f5a1e", 8.5),
             "mountain": ("#6b3a12", 8), "peninsula": ("#3d4a1e", 8), "continental": ("#2a4a2a", 8),
             "spine": ("#2a4a2a", 7.5), "arc": ("#5b2a6e", 7.5), "hotspot": ("#8a3b12", 7.5)}
    heights = {m["id"]: m["height_m"] for m in design.get("mountains", [])}
    hide = set(design.get("map_hide_labels", []))
    for kind, fid, name, lo, la in label_points(design):
        if fid in hide or (kind == "mountain" and heights.get(fid, 0) < 3000):   # 低矮山地只在 02 图里标
            continue
        col, fs = style.get(kind, ("#333333", 8))
        ax.text(lo, la, name, transform=PC, fontsize=fs, color=col, ha="center", va="center", path_effects=halo)
    draw_specials(ax, design, halo, PC, ccrs.Geodetic())
    pol_lab = (0, -80)
    ax.text(*pol_lab, f"{design['polar']['name']}（冰盖）", transform=PC, fontsize=12, weight="bold",
            ha="center", color="#2c4a6b", path_effects=halo)
    A = metrics["A"]
    fig.text(0.5, 0.975, f"终图 {design['name']}{opt}（设计图：只画几百公里以上的要素，海岸细节、河谷、峡湾等由之后的模拟产生）",
             fontsize=16, ha="center", va="top")
    fig.text(0.5, 0.945, f"海洋 {A['ocean_fraction']['value'] * 100:.1f}% · 主大陆 {A['main_continent']['area_km2'] / 1e6:.0f} 百万 km²"
             f"（占陆地 {A['main_continent']['share_of_land'] * 100:.0f}%；亚非欧约 85）· 亚大陆 {A['subcontinents']['count']} 个 · "
             f"等面积投影，中央经线 {lon0:g}°E · 示意高程为分层设色 + 阴影（垂直夸张 3×）· ○ 环礁 · 海山",
             fontsize=11, ha="center", va="top", color="#333333")
    stamp(fig)
    fig.savefig(out / "01_final_map.png", dpi=150)
    plt.close(fig)

    # ---- 02 要素与计划成因 ----
    fig = plt.figure(figsize=(20, 12.5))
    ax = fig.add_axes([0.02, 0.24, 0.96, 0.70], projection=proj)
    ax.set_global()
    faded = 0.45 * rgb + 0.55
    ax.imshow(faded, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(3000, 1500),
              interpolation="nearest")
    ax.contour(grid.lon, grid.lat, land.astype(float), levels=[0.5], colors="#555555", linewidths=0.4, transform=PC)
    kind_col = {"collision": "#c0392b", "suture_old": "#d68910", "margin": "#7d3c98", "rift": "#e67e22",
                "other": "#6e4b2a"}

    def mt_kind(m):
        c = m["cause"]
        if m.get("profile") == "rift":
            return "rift"
        if "碰撞" in c and ("今" in c or "仍在" in c):
            return "collision"
        if "碰撞" in c or "缝合" in c or "古老" in c or "褶皱" in c:
            return "suture_old"
        if "俯冲" in c:
            return "margin"
        if "裂谷" in c:
            return "rift"
        return "other"

    rows = []
    for m in design.get("mountains", []):
        p = np.asarray(m["line"], float)
        k = mt_kind(m)
        ax.plot(p[:, 0], p[:, 1], color=kind_col[k], lw=3.2, transform=ccrs.Geodetic(), solid_capstyle="round")
        mid = p[len(p) // 2]
        ax.text(mid[0], mid[1], m["id"], transform=PC, fontsize=9, weight="bold", color=kind_col[k],
                path_effects=halo, ha="center", va="bottom")
        rows.append(f"{m['id']} {m['name']}（{m['height_m']} m）：{m['cause']}")
    for i in design.get("islands", []):
        if i["type"] in ("arc", "hotspot"):
            p = np.asarray(i["line"], float)
            col = "#7d3c98" if i["type"] == "arc" else "#a04000"
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.4, ls="--" if i["type"] == "hotspot" else "-",
                    transform=ccrs.Geodetic())
            if i["type"] == "hotspot":
                ax.plot(p[0, 0], p[0, 1], marker="*", ms=11, color=col, mec="white", transform=PC)
            mid = p[len(p) // 2]
            ax.text(mid[0], mid[1], i["id"], transform=PC, fontsize=8.5, color=col, path_effects=halo)
            rows.append(f"{i['id']} {i['name']}：{i['cause']}")
    for s in design.get("seas", []) + design.get("straits", []):
        p = np.asarray(s.get("poly", s.get("line")), float)
        c = p.mean(axis=0) if "poly" in s else p[len(p) // 2]
        ax.text(c[0], c[1], s["id"], transform=PC, fontsize=8.5, color="#0b3d6e", weight="bold", path_effects=halo)
        rows.append(f"{s['id']} {s['name']}：{s['cause']}")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=kind_col["collision"], lw=3, label="碰撞造山带（年轻，仍在隆升）"),
               Line2D([], [], color=kind_col["suture_old"], lw=3, label="老缝合带与褶皱带（已剥蚀）"),
               Line2D([], [], color=kind_col["margin"], lw=3, label="俯冲带上的山系与岛弧"),
               Line2D([], [], color=kind_col["rift"], lw=3, label="裂谷"),
               Line2D([], [], color="#a04000", lw=1.4, ls="--", marker="*", ms=9, label="热点岛链（★ 活动端）")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ncol = 3
    per = int(np.ceil(len(rows) / ncol))
    for c in range(ncol):
        fig.text(0.02 + c * 0.33, 0.215, "\n".join(rows[c * per:(c + 1) * per]), fontsize=7.2, va="top",
                 ha="left", wrap=True)
    fig.text(0.5, 0.975, f"终图初稿 {design['name']} · 要素与计划成因（倒推的起点，倒推时再细化时间与过程）",
             fontsize=16, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "02_features_and_causes.png", dpi=150)
    plt.close(fig)

    # ---- 04 与同样概括程度的地球对照 ----
    tol = float(design.get("earth_compare_tolerance_deg", 1.2))
    with zipfile.ZipFile(NE_LAND) as zf:
        shp = [n for n in zf.namelist() if n.endswith(".shp")][0]
    geoms = gpd.read_file(f"zip://{NE_LAND.as_posix()}!{shp}").geometry
    geoms = geoms[geoms.area > 0.3]  # 去掉小于约 0.3 平方度的小岛（设计图同样只画大岛）
    simp = geoms.simplify(tol, preserve_topology=True)
    earth = rasterize(((g, 1) for g in simp if not g.is_empty), out_shape=grid.shape,
                      transform=from_origin(-180, 90, grid.dlon_deg, grid.dlat_deg), fill=0, dtype="uint8").astype(bool)
    fig = plt.figure(figsize=(16, 17))
    for k, (msk, title, c0) in enumerate([
            (land, f"终图初稿 {design['name']}（中央经线 {lon0:g}°E）", lon0),
            (earth, f"地球：海岸线按约 {tol * 111:.0f} km 容差概括、去掉小岛后（中央经线 30°E）", 30.0)]):
        a = fig.add_axes([0.03, 0.52 - k * 0.47, 0.94, 0.42], projection=ccrs.EqualEarth(central_longitude=c0))
        a.set_global()
        img = np.where(msk[..., None], np.array([0.80, 0.74, 0.58]), np.array([0.86, 0.92, 0.96]))
        a.imshow(img, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(2400, 1200),
                 interpolation="nearest")
        a.contour(grid.lon, grid.lat, msk.astype(float), levels=[0.5], colors="#333333", linewidths=0.5, transform=PC)
        a.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
        a.set_title(title, fontsize=13)
    fig.text(0.5, 0.985, "同样的概括程度下比较：设计图只定几百公里以上的形状；地球的细碎海岸在这个尺度下同样看不到",
             fontsize=14, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "04_vs_earth_same_detail.png", dpi=130)
    plt.close(fig)

    # ---- 06 奇景舞台：原型库 × 终图核对（config/design/archetype_check_<版本>.yaml）----
    chk_path = ROOT / "config" / "design" / f"archetype_check_{design['_ver']}.yaml"
    if chk_path.exists():
        chk = yaml.safe_load(chk_path.read_text(encoding="utf-8"))
        entries = chk["entries"]
        st_col = {"placed": "#1e8449", "added": "#c0392b", "candidate": "#2e86c1", "independent": "#909497"}
        st_name = chk["statuses"]
        lith_col = {"carbonate": "#5dade2", "sandstone": "#e59866", "redbed": "#cb4335", "evaporite": "#f4d03f",
                    "craton": "#9a7d0a", "granite": "#c39bd3", "ophiolite": "#196f3d"}
        lith_name = {"carbonate": "碳酸盐岩（喀斯特）", "sandstone": "古老砂岩（平顶山）", "redbed": "红层（丹霞）",
                     "evaporite": "蒸发岩（盐丘）", "craton": "古陆核", "granite": "花岗岩基", "ophiolite": "蛇绿岩"}
        fig = plt.figure(figsize=(20, 18))
        ax = fig.add_axes([0.02, 0.37, 0.96, 0.57], projection=proj)
        ax.set_global()
        ax.imshow(0.35 * rgb + 0.65, extent=(-180, 180, -90, 90), transform=PC, origin="upper",
                  regrid_shape=(3000, 1500), interpolation="nearest")
        ax.contour(grid.lon, grid.lat, land.astype(float), levels=[0.5], colors="#666666", linewidths=0.4,
                   transform=PC)
        for lt in design.get("lithology", []):
            p = np.asarray(lt["poly"], float)
            c = lith_col.get(lt["kind"], "#888888")
            ax.fill(p[:, 0], p[:, 1], facecolor=c, edgecolor=c, alpha=0.45, lw=1.0, transform=PC, zorder=3)
        draw_specials(ax, design, halo, PC, ccrs.Geodetic(), labels=False)
        pts = feature_points(design)
        for i, e in enumerate(entries, 1):
            if e["status"] == "independent":
                continue
            locs = [pts[f] for f in e.get("features", []) if f in pts] + [tuple(x) for x in e.get("at", [])]
            for lo, la in locs:
                ax.text(lo, la, str(i), transform=PC, fontsize=6.5, color="white", ha="center", va="center",
                        zorder=7, bbox=dict(boxstyle="circle,pad=0.18", fc=st_col[e["status"]], ec="white", lw=0.6))
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        cnt = {s: sum(e["status"] == s for e in entries) for s in st_col}
        h1 = [Line2D([], [], marker="o", ls="none", ms=9, mfc=st_col[s], mec="white", label=f"{st_name[s]}（{cnt[s]}）")
              for s in st_col]
        h2 = [Patch(fc=lith_col[k], ec=lith_col[k], alpha=0.45, label=v) for k, v in lith_name.items()]
        h3 = [Line2D([], [], marker="^", ls="none", ms=9, mfc="#c0392b", mec="white", label="“神怒”候选海湾"),
              Line2D([], [], color="#6c3483", lw=1.3, label="撞击构造")]
        leg = ax.legend(handles=h1, loc="lower left", fontsize=8.5, framealpha=0.92, title="原型状态（圆内为编号）",
                        title_fontsize=9)
        ax.add_artist(leg)
        ax.legend(handles=h2 + h3, loc="lower right", fontsize=8.5, framealpha=0.92, title="岩性区与其他",
                  title_fontsize=9)
        ncol = 5
        per = int(np.ceil(len(entries) / ncol))
        for c in range(ncol):
            for r, (i, e) in enumerate(list(enumerate(entries, 1))[c * per:(c + 1) * per]):
                fig.text(0.02 + c * 0.196, 0.345 - r * 0.0148, f"{i:>3} {e['archetype']}", fontsize=7.6,
                         color=st_col[e["status"]], va="top", ha="left")
        fig.text(0.5, 0.975, f"终图 {design['name']} · 奇景舞台：原型库 {len(entries)} 个原型在终图上的位置"
                 f"（核对表 docs/design_archetype_check_{design['_ver']}.md）", fontsize=16, ha="center", va="top")
        fig.text(0.5, 0.95, "绿 = 终图已安排；红 = 本版新增；蓝 = 候选地点（终图给了条件，形态由后续阶段的过程确认）；"
                 "灰 = 不依赖终图（尺度太小、随处可生、随机事件或行星级）", fontsize=11, ha="center", va="top",
                 color="#333333")
        stamp(fig)
        fig.savefig(out / "06_archetype_stage.png", dpi=150)
        plt.close(fig)

    # ---- 05 南极区 ----
    fig = plt.figure(figsize=(11, 12))
    pp = ccrs.LambertAzimuthalEqualArea(central_latitude=-90)
    a = fig.add_axes([0.05, 0.07, 0.9, 0.84], projection=pp)
    a.set_extent([-180, 180, -90, -38], PC)
    a.imshow(rgb, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(1800, 1800),
             interpolation="nearest")
    a.contour(grid.lon, grid.lat, land.astype(float), levels=[0.5], colors="#333333", linewidths=0.5, transform=PC)
    a.gridlines(xlocs=range(-180, 181, 30), ylocs=[-80, -70, -60, -50, -40], color="#777777", lw=0.3)
    for kind, fid, name, lo, la in label_points(design):
        if la < -45 and kind not in ("arc", "hotspot") and not (kind == "mountain" and heights.get(fid, 0) < 3000):
            col, fs = style.get(kind, ("#333333", 8))
            a.text(lo, la, name, transform=PC, fontsize=fs + 1.5, color=col, ha="center", va="center",
                   path_effects=halo)
    pa = areas["polar_km2"]
    a.set_title(f"南极区：{design['polar']['name']} {pa / 1e6:.1f} 百万 km²（南极洲约 14）· 等面积方位投影", fontsize=13)
    stamp(fig)
    fig.savefig(out / "05_south_pole.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="v1")
    ap.add_argument("--options", default="", help="逗号分隔的设计选项，例如 bigA（洲 A 扩大）")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    path = ROOT / "config" / "design" / f"final_map_{args.design}.yaml"
    options = {o.strip() for o in args.options.split(",") if o.strip()}
    design = apply_options(yaml.safe_load(path.read_text(encoding="utf-8")), options)
    design["_ver"] = args.design
    design["_options"] = ",".join(sorted(options))
    tag = args.design + "".join(f"_{o}" for o in sorted(options))
    ver = code_version()
    grid = LatLonGrid(2048, 1024)
    log(f"栅格化 {path.name}")
    b = build(design, grid)
    land = b["land"]
    area = grid.cell_area_km2
    tot_land = float((area * land).sum())
    conts = {}
    for c in ("A", "B", "C", "D", "polar", "far"):
        conts[c] = float((area * (b["cont"] == c)).sum())
    log(f"海洋占比 {1 - tot_land / grid.total_area_km2:.4f}；陆地 {tot_land / 1e6:.1f} 百万 km²；"
        + "，".join(f"{k} {v / 1e6:.1f}" for k, v in conts.items()))
    log("示意高程")
    elev = elevation(design, grid, b)

    world_id = f"design_{tag}"
    out_w = ROOT / "data" / "worlds" / world_id
    out_w.mkdir(parents=True, exist_ok=True)
    attrs = {"world_id": world_id, "display_name": f"终图设计 {design['name']}", "seed": "design",
             "method_class_elevation": "visual_only", "code_version": ver,
             "source": f"scripts/design_final_map.py：config/design/{path.name}（海陆 user_override，高程为示意）"}
    files = {}
    for f in (1, 2):
        g = LatLonGrid(2048 // f, 1024 // f)
        e = elev if f == 1 else downsample(elev, f)
        lf = land.astype(float) if f == 1 else downsample(land.astype(float), f)
        ds = xr.Dataset({"elevation": (("lat", "lon"), e.astype("float32"), {"units": "m"}),
                         "land": (("lat", "lon"), lf.astype("float32"), {"description": "陆地比例（≥ 0.5 为陆地）"})},
                        coords={"lat": g.lat, "lon": g.lon}, attrs={**attrs, "grid": g.name})
        p = out_w / f"{world_id}_{g.name}.nc"
        ds.to_netcdf(p)
        files[g.name] = p.relative_to(ROOT).as_posix()

    log("A 层核对（2048x1024）")
    from mythic.metrics.compute import compute_all, load_config
    cfg = load_config()
    metrics, fields = compute_all(elev, land, cfg=cfg, return_fields=True)
    A = metrics["A"]
    for k, v in A.items():
        log(f"  {k}: {'达标' if v.get('pass') else '未达标'}")
    sl = metrics.get("soft_limits", {})
    areas = {"land_km2": tot_land, "continents_km2": conts, "polar_km2": conts["polar"]}
    summary = {
        "description": "终图设计的面积与 A 层核对（重建：scripts/design_final_map.py）",
        "design_file": path.relative_to(ROOT).as_posix(), "options": sorted(options), "files": files,
        "code_version": ver,
        "ocean_fraction": A["ocean_fraction"]["value"], "areas_km2": areas,
        "A": {k: {kk: vv for kk, vv in v.items() if kk in ("pass", "value", "share_of_land", "area_km2", "count",
                                                         "areas_km2", "north_deg", "south_deg", "found")}
              for k, v in A.items()},
        "main_inland_sea": {k: A["main_inland_sea"].get(k) for k in ("exists", "main", "geometry", "passes")},
        "soft_limits": sl,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    from mythic.figures.world import to_jsonable
    (out_w / "params.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=1),
                                       encoding="utf-8", newline="\n")
    if not args.no_figures:
        out_f = ROOT / "figures" / "design" / tag
        out_f.mkdir(parents=True, exist_ok=True)
        log("出图")
        figures(design, grid, b, elev, metrics, fields, out_f, ver, areas)
        from mythic.figures import maps
        from mythic.figures.world import load
        world = load(str(out_w / f"{world_id}_2048x1024.nc"))
        maps.landmass_units(world, metrics, fields, cfg, out_f / "03_A_layer.png")
    log(f"完成，用时 {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
