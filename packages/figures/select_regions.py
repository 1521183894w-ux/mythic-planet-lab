"""区域放大图的自动选取（docs/review_protocol.md §2.2 的 8 条规则）。

每条规则给出一个窗口（中心 + 边长，默认 2000 km 见方），并指定一个性质相近的地球对照区域
（config/earth_regions.yaml 中的 id；随机海岸则在地球上同样随机抽一段）。

用到的量（全部在球面上计算，与指标代码同一套格网和形态学定义）：
- 海岸线采样点：陆地掩膜 0.5 等值线按约 20 km 加密，每点代表一小段海岸长度；
- 平滑海岸：对陆地做半径 100 km 的开运算再闭运算，去掉宽度 < 200 km 的半岛、海湾和小岛后的海岸线；
- 窗口内的海岸复杂度 = 窗口内切圆（半径 = 边长/2）内的海岸线长度 ÷ max(平滑海岸线长度, 圆的半径)。
  平直海岸约为 1，海岸越破碎、半岛海湾岛屿越多，比值越大。
- 群岛：面积小于"大陆"门槛（config/metrics.yaml 的 continent_min_area_km2）的连通陆块，按质心计数。
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from mythic.metrics.coast import coastline_segments, densify_km, polyline_lengths_km
from mythic.metrics.grid import LatLonGrid, chord_to_km, km_to_chord
from mythic.metrics.hard import unit_centroid_latitudes
from mythic.metrics.spherical import closing, opening
from mythic.metrics.topology import component_sums, label

DEFAULT_SIZE_KM = 2000.0
SMOOTH_RADIUS_KM = 100.0
COAST_STEP_KM = 20.0
CANDIDATE_STEP_KM = 100.0

# (键, 标题, 规则说明, 地球对照区域 id)
ROLES = [
    ("r1_main_inland_sea", "主内海全貌", "A 层识别出的主内海；窗口边长按内海范围放大", "mediterranean"),
    ("r2_inland_sea_complex_coast", "主内海中海岸最复杂的一段", "主内海沿岸的候选窗口中，海岸复杂度最高者", "aegean"),
    ("r3_north_coast", "主大陆北端高纬海岸", "主大陆最北的海岸点", "norway"),
    ("r4_polar_coast", "极地大陆的一段海岸", "极地大陆（A 层判据）伸向低纬最远的海岸点", "antarctic_peninsula"),
    ("r5_isthmus", "亚大陆之间的陆桥或地峡", "主大陆各亚大陆（面积 ≥ 下限）之间最窄的分界处", "panama_isthmus"),
    ("r6_smoothest_coast", "海岸复杂度最低的一段", "平滑后窗口内仍有 ≥ 1000 km 海岸的候选窗口中，海岸复杂度最低者（专门暴露问题）",
     "west_africa_coast"),
    ("r7_archipelago", "最大的群岛区", "窗口内岛屿个数最多者（只数达到岛屿统计面积下限的岛）", "se_asia"),
    ("r8_random_coast", "随机抽取的一段海岸", "按海岸线长度加权随机抽取（种子由世界编号确定，可复现），防止只挑好看的", None),
]


@dataclass
class RegionPick:
    key: str
    title: str
    rule: str
    earth_ref: str | None
    center: tuple[float, float] | None = None
    size_km: float = DEFAULT_SIZE_KM
    reason: str = ""
    score: float | None = None
    missing: str | None = None

    def as_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "rule": self.rule, "earth_ref": self.earth_ref,
                "center_lat_lon": None if self.center is None else [round(self.center[0], 3), round(self.center[1], 3)],
                "size_km": self.size_km, "reason": self.reason, "score": self.score, "missing": self.missing}


def _xyz_to_latlon(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), float(np.degrees(np.arctan2(v[1], v[0])))


def coast_points(grid: LatLonGrid, land: np.ndarray, step_km: float = COAST_STEP_KM) -> tuple[np.ndarray, np.ndarray]:
    """海岸线加密采样点（单位向量）与每点代表的海岸长度（km）。"""
    pts, wts = [], []
    for s in coastline_segments(grid, land):
        if len(s) < 2:
            continue
        p = densify_km(grid, [s], step_km) / grid.radius_km
        pts.append(p)
        wts.append(np.full(len(p), polyline_lengths_km(grid, s).sum() / len(p)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(wts)


def _ball_sums(tree: cKDTree, weights: np.ndarray, centers: np.ndarray, chord: float, chunk: int = 400) -> np.ndarray:
    out = np.zeros(len(centers))
    for a in range(0, len(centers), chunk):
        for k, idx in enumerate(tree.query_ball_point(centers[a:a + chunk], chord, workers=-1)):
            out[a + k] = weights[idx].sum() if idx else 0.0
    return out


class CoastAnalysis:
    """一个世界的海岸采样、候选窗口与复杂度。"""

    def __init__(self, grid: LatLonGrid, land: np.ndarray, size_km: float = DEFAULT_SIZE_KM):
        self.grid, self.land, self.size_km = grid, land, size_km
        self.radius_km = size_km / 2.0
        self.pts, self.w = coast_points(grid, land)
        smooth = closing(grid, opening(grid, land, SMOOTH_RADIUS_KM), SMOOTH_RADIUS_KM)
        self.spts, self.sw = coast_points(grid, smooth)
        cand, _ = coast_points(grid, land, CANDIDATE_STEP_KM)
        self.cand = cand
        chord = float(km_to_chord(self.radius_km, grid.radius_km))
        self.L_raw = _ball_sums(cKDTree(self.pts), self.w, cand, chord) if len(self.pts) else np.zeros(len(cand))
        self.L_smooth = (_ball_sums(cKDTree(self.spts), self.sw, cand, chord) if len(self.spts)
                         else np.zeros(len(cand)))
        # 分母下限取窗口半径：平滑海岸很短的窗口（例如只有小岛）比值不会发散
        self.complexity = self.L_raw / np.maximum(self.L_smooth, self.radius_km)
        # 候选点是否位于平滑后仍存在的海岸上（距平滑海岸 ≤ 平滑半径）：保证窗口中心真的落在一段成形的海岸上
        if len(self.spts) and len(cand):
            d, _ = cKDTree(self.spts).query(cand, workers=-1)
            self.on_smooth = chord_to_km(d, grid.radius_km) <= SMOOTH_RADIUS_KM
        else:
            self.on_smooth = np.zeros(len(cand), bool)
        # 每个候选点 -> 最近的陆地格点、最近的海洋格点（扁平索引），用于判断属于哪块陆地、哪片海
        xyz = grid.xyz.reshape(-1, 3)
        land_idx = np.flatnonzero(land.ravel())
        sea_idx = np.flatnonzero(~land.ravel())
        self.cand_land_cell = land_idx[cKDTree(xyz[land_idx]).query(cand, workers=-1)[1]] if len(cand) else np.zeros(0, int)
        self.cand_sea_cell = sea_idx[cKDTree(xyz[sea_idx]).query(cand, workers=-1)[1]] if len(cand) else np.zeros(0, int)

    def reason(self, i: int) -> str:
        return (f"窗口内海岸线 {self.L_raw[i]:.0f} km，平滑后 {self.L_smooth[i]:.0f} km，"
                f"海岸复杂度 {self.complexity[i]:.2f}")

    def center(self, i: int) -> tuple[float, float]:
        return _xyz_to_latlon(self.cand[i])


def _seed(world_id: str, salt: str) -> int:
    return zlib.crc32(f"{world_id}|{salt}".encode("utf-8"))


def random_coast_center(ca: CoastAnalysis, world_id: str, salt: str = "r8") -> tuple[tuple[float, float], str]:
    rng = np.random.default_rng(_seed(world_id, salt))
    i = int(rng.choice(len(ca.pts), p=ca.w / ca.w.sum()))
    return _xyz_to_latlon(ca.pts[i]), f"随机种子 crc32('{world_id}|{salt}')，按海岸线长度加权抽取"


def select_regions(world_id: str, grid: LatLonGrid, land: np.ndarray, metrics: dict, fields: dict, cfg: dict,
                   earth_sizes: dict[str, float], ca: CoastAnalysis | None = None) -> tuple[list[RegionPick], CoastAnalysis]:
    """按 8 条规则选窗口。metrics/fields 是 compute_all(..., return_fields=True) 的输出。"""
    ca = ca or CoastAnalysis(grid, land)
    picks = {k: RegionPick(k, t, r, e) for k, t, r, e in ROLES}
    big_coast = ca.L_raw >= ca.radius_km  # 窗口内至少有 1000 km 海岸，才算"一段海岸"
    area = grid.cell_area_km2

    # r1 主内海全貌
    p = picks["r1_main_inland_sea"]
    mis = metrics["A"]["main_inland_sea"]
    sea_regions = fields.get("sea_regions")
    sea_id, note = None, ""
    if mis.get("exists"):
        sea_id = mis["main"]["region_id"]
    else:
        cands = [s for s in mis.get("candidate_seas", [])
                 if s["main_continent_border_fraction"] >= cfg["A_hard_constraints"]["main_inland_sea"]["min_main_border_fraction"]]
        if cands:
            sea_id, note = cands[0]["region_id"], "（A 层未识别出主内海，改用主大陆内部最大的半封闭海域）"
    if sea_id is None or sea_regions is None:
        p.missing = "主大陆内部没有半封闭海域"
    else:
        sea = sea_regions == sea_id
        v = (grid.xyz * (area * sea)[..., None]).sum(axis=(0, 1))
        v /= np.linalg.norm(v)
        far = float(chord_to_km(np.linalg.norm(grid.xyz[sea] - v, axis=1).max(), grid.radius_km))
        need = max(DEFAULT_SIZE_KM, np.ceil(2.1 * far / 500.0) * 500.0)
        p.center = _xyz_to_latlon(v)
        p.size_km = float(max(need, earth_sizes.get(p.earth_ref, DEFAULT_SIZE_KM)))
        p.score = float((area * sea).sum())
        p.reason = (f"海域面积 {p.score / 1e6:.2f} 百万 km²，从中心到最远海域格点 {far:.0f} km{note}；"
                    f"窗口边长取 max(内海所需 {need:.0f} km, 地中海对照 {earth_sizes.get(p.earth_ref, 0):.0f} km)，两边同比例尺")

    # r2 主内海中海岸最复杂的一段
    p = picks["r2_inland_sea_complex_coast"]
    if sea_id is None or sea_regions is None:
        p.missing = "没有主内海"
    else:
        on_sea = (sea_regions.ravel()[ca.cand_sea_cell] == sea_id) & big_coast
        if not on_sea.any():
            p.missing = "主内海沿岸没有足够长的海岸（窗口内 < 1000 km）"
        else:
            i = int(np.flatnonzero(on_sea)[np.argmax(ca.complexity[on_sea])])
            p.center, p.score, p.reason = ca.center(i), float(ca.complexity[i]), ca.reason(i)

    # r3 主大陆北端高纬海岸
    p = picks["r3_north_coast"]
    units = fields["units"]
    main_id = int(np.bincount(units[land].ravel(), weights=area[land]).argmax()) if land.any() else 0
    on_main = units.ravel()[ca.cand_land_cell] == main_id
    if not on_main.any():
        p.missing = "没有主大陆"
    else:
        lat = np.degrees(np.arcsin(np.clip(ca.cand[:, 2], -1, 1)))
        i = int(np.flatnonzero(on_main)[np.argmax(lat[on_main])])
        p.center, p.score = ca.center(i), float(lat[i])
        p.reason = f"主大陆最北的海岸点 {lat[i]:.1f}°；" + ca.reason(i)

    # r4 极地大陆的一段海岸
    p = picks["r4_polar_coast"]
    cp = cfg["A_hard_constraints"]["polar_continent"]
    n_units = int(units.max())
    u_area = component_sums(units, n_units, area)
    u_lat = unit_centroid_latitudes(grid, units, n_units)
    polar_ids = [k for k in range(1, n_units + 1) if abs(u_lat[k]) >= cp["min_abs_centroid_lat_deg"]
                 and cp["area_km2"][0] <= u_area[k] <= cp["area_km2"][1]]
    on_polar = np.isin(units.ravel()[ca.cand_land_cell], polar_ids)
    if not polar_ids or not on_polar.any():
        p.missing = "没有符合 A 层判据的极地大陆"
    else:
        lat = np.degrees(np.arcsin(np.clip(ca.cand[:, 2], -1, 1)))
        i = int(np.flatnonzero(on_polar)[np.argmin(np.abs(lat[on_polar]))])
        p.center, p.score = ca.center(i), float(lat[i])
        p.reason = f"极地大陆伸向低纬最远的海岸点 {lat[i]:.1f}°；" + ca.reason(i)

    # r5 亚大陆之间的陆桥或地峡
    p = picks["r5_isthmus"]
    lobes = fields["subcontinents"]
    dist = fields["distance_to_sea_km"]
    min_area = cfg["A_hard_constraints"]["subcontinents"]["min_area_km2"]
    n_lobes = int(lobes.max())
    l_area = component_sums(lobes, n_lobes, area)
    big = np.zeros(n_lobes + 1, bool)
    big[1:] = l_area[1:] >= min_area
    best = None
    if big.sum() >= 2:
        pairs: dict[tuple[int, int], list[int]] = {}
        flat_l = lobes.ravel()
        idx = np.arange(lobes.size).reshape(lobes.shape)
        for a_idx, b_idx in ((idx, np.roll(idx, -1, axis=1)), (idx[:-1], idx[1:])):
            a, b = flat_l[a_idx.ravel()], flat_l[b_idx.ravel()]
            m = (a != b) & big[a] & big[b]
            for ia, ib, ca_, cb_ in zip(a[m], b[m], a_idx.ravel()[m], b_idx.ravel()[m]):
                pairs.setdefault((min(ia, ib), max(ia, ib)), []).extend([int(ca_), int(cb_)])
        for pair, cells in pairs.items():
            cells = np.unique(cells)
            d = dist.ravel()[cells]
            k = int(np.argmax(d))
            if best is None or d[k] < best[0]:
                best = (float(d[k]), int(cells[k]), pair)
    if best is None:
        p.missing = "主大陆没有被颈部分开的两个以上亚大陆"
    else:
        hw, cell, pair = best
        i, j = divmod(cell, grid.nx)
        p.center, p.score = (float(grid.lat[i]), float(grid.lon[j])), 2 * hw
        rank = {int(k): n + 1 for n, k in enumerate(np.argsort(l_area[1:])[::-1] + 1)}  # 与陆块图一致：按面积从大到小编号
        p.reason = (f"亚大陆 {rank[pair[0]]} 与 {rank[pair[1]]} 的分界线，是所有亚大陆分界中最窄的；"
                    f"分界线上离海最远 {hw:.0f} km，颈部宽度约 {2 * hw:.0f} km")

    # r6 海岸复杂度最低的一段
    # 只考虑中心在成形海岸上、且平滑后窗口内仍有 ≥ 1000 km 海岸的窗口：
    # 否则小岛群（平滑后海岸为 0）或只在边缘擦到海岸的窗口会得到假的"低复杂度"
    p = picks["r6_smoothest_coast"]
    real_coast = (ca.L_smooth >= ca.radius_km) & ca.on_smooth
    if not real_coast.any():
        p.missing = "没有平滑后窗口内海岸线仍 ≥ 1000 km 的位置"
    else:
        i = int(np.flatnonzero(real_coast)[np.argmin(ca.complexity[real_coast])])
        p.center, p.score, p.reason = ca.center(i), float(ca.complexity[i]), ca.reason(i)

    # r7 最大的群岛区
    # 只数面积 ≥ 岛屿统计下限（4 个赤道格点，与 Korčak 拟合同一下限）的岛：等经纬度格网在高纬像元更小，
    # 能分辨出更多单像元碎岛，不设面积下限会让高纬碎岛区（例如法兰士约瑟夫地）因格网假象胜出。
    p = picks["r7_archipelago"]
    lab, n = label(land)
    comp_area = component_sums(lab, n, area)
    a_min = cfg["B1_coast"]["islands"]["min_area_pixels"] * (np.pi * grid.radius_km / grid.ny) ** 2
    a_max = cfg["B1_coast"]["continent_min_area_km2"]
    isl = np.flatnonzero((comp_area[1:] < a_max) & (comp_area[1:] >= a_min)) + 1
    if isl.size == 0:
        p.missing = "没有达到面积下限的岛屿"
    else:
        cx = np.stack([component_sums(lab, n, area * grid.xyz[..., k])[isl] for k in range(3)], axis=1)
        cx /= np.linalg.norm(cx, axis=1, keepdims=True)
        chord = float(km_to_chord(ca.radius_km, grid.radius_km))
        counts = cKDTree(cx).query_ball_point(ca.cand, chord, return_length=True, workers=-1)
        i = int(np.argmax(counts + 1e-6 * ca.L_raw))  # 个数相同时取海岸线更长者
        p.center, p.score = ca.center(i), float(counts[i])
        p.reason = (f"窗口内岛屿 {int(counts[i])} 个（面积 {a_min:.0f} km² 至 {a_max / 1e6:g} 百万 km² 的陆块，按质心计）；"
                    + ca.reason(i))

    # r8 随机抽取的一段海岸
    p = picks["r8_random_coast"]
    if len(ca.pts) == 0:
        p.missing = "没有海岸"
    else:
        p.center, p.reason = random_coast_center(ca, world_id)

    return [picks[k] for k, *_ in ROLES], ca
