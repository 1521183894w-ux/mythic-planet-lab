"""阶段 0.7 尺子校准用的"圆润世界"对照样本（故意做坏的世界）。

模仿上一轮（Codex）的典型产出：海陆比 71/29，一块陆路相连的主大陆约占陆地 75–80%，有一个主内海，
能通过 A 层全部硬约束；但海岸光滑圆润、没有半岛和群岛，山脉是随机噪声斑块，没有任何地质过程。
它的用途只有一个：检验质量指标和 world-reviewer 能不能识别出这种世界。它不是生成器的原型。

几何（全部是光滑的解析形状，海岸线只有格网锯齿）：
- 主大陆 = 4 个圆形陆块，用约 250 km 宽的光滑陆桥串起来（窄于亚大陆颈部门槛 500 km，宽于细线门槛 50 km）；
- 主内海 = 最大陆块中挖出的圆形海域，经一条约 100 km 宽的海峡通外洋；
- 次大陆 1 块（圆形，半径用来把海洋占比调到 71%）、极地圆盘大陆 1 块、圆形小岛 4 个；
- 陆块之间用"光滑并集"连接，接口处是圆角。
高程：陆地 = 缓慢抬升的台地 + 随机噪声斑块状"山脉"；海洋 = 窄陆架 + 光滑陆坡 + 平坦深海。

method_class：visual_only（纯视觉的对照样本，没有任何过程）。
输出：data/worlds/rounded_control/rounded_control_<分辨率>.nc（2048x1024 生成，1024x512、512x256 由面积平均降采样），
以及同目录的 params.json（几何参数与调参结果，进 Git）。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_rounded_world.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from mythic.figures.world import code_version
from mythic.metrics.grid import LatLonGrid
from mythic.metrics.spherical import distance_to_set_km

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worlds" / "rounded_control"
SEED = 20261021
R_KM = 6371.0
KM_PER_DEG = np.pi * R_KM / 180.0

# 圆形陆块：(名称, 纬度, 经度, 半径°)
# 陆块之间留 5–8° 的间隙，陆桥够长，颈部在 1024x512 与 2048x1024 上都明显窄于 500 km（第一版间隙 2–3°，
# 光滑并集的圆角把颈部撑宽，1024x512 上只切出 2 个亚大陆）
LOBES = [("L1", 38.0, 15.0, 32.0), ("L2", 44.0, 112.0, 28.0), ("L3", -27.0, 24.0, 25.0), ("L4", -8.0, 150.0, 22.0)]
BRIDGES = [("L1", "L2"), ("L1", "L3"), ("L2", "L4")]
BRIDGE_HALF_WIDTH_KM = 125.0
SEA = (37.0, -6.0, 8.5)                      # 主内海：圆心与半径°
STRAIT = ((37.0, -14.0), (37.0, -40.0), 50.0)  # 海峡：从内海西缘向西通到外洋，半宽 50 km
SECONDARY_CENTER = (5.0, -80.0)              # 次大陆圆心；半径由海洋占比调定
POLAR_RADIUS_DEG = 16.0                      # 极地圆盘大陆（南极）
ISLANDS = [(-35.0, -150.0, 2.5), (20.0, -150.0, 1.5), (-40.0, 70.0, 2.0), (55.0, -30.0, 1.8)]
SMOOTH_KM = 80.0                             # 光滑并集的圆角尺度
TARGET_OCEAN = 0.710


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def ang_km(xyz, lat, lon):
    return np.degrees(np.arccos(np.clip(xyz @ unit(lat, lon), -1, 1))) * KM_PER_DEG


def arc_points(a, b, step_km=10.0):
    """大圆弧 a→b 上的加密点（单位向量）。"""
    pa, pb = unit(*a), unit(*b)
    om = np.arccos(np.clip(pa @ pb, -1, 1))
    n = max(2, int(om * R_KM / step_km))
    t = np.linspace(0, 1, n)[:, None]
    return (np.sin((1 - t) * om) * pa + np.sin(t * om) * pb) / np.sin(om)


def dist_to_arc_km(xyz_flat, pts):
    d, _ = cKDTree(pts).query(xyz_flat, workers=-1)
    return 2 * R_KM * np.arcsin(np.clip(d / 2, 0, 1))


def smax(a, b, k=SMOOTH_KM):
    """光滑并集（log-sum-exp），接口处形成半径约 k 的圆角。"""
    m = np.maximum(a, b)
    return m + k * np.log(np.exp((a - m) / k) + np.exp((b - m) / k))


def smin(a, b, k=SMOOTH_KM / 3):
    return -smax(-a, -b, k)


def signed_land(grid: LatLonGrid, secondary_radius_deg: float) -> tuple[np.ndarray, dict]:
    """陆地的"有符号距离"（km，> 0 为陆地）。"""
    xyz = grid.xyz.reshape(-1, 3)
    lobes = {n: (la, lo, r) for n, la, lo, r in LOBES}
    s = np.full(len(xyz), -1e9)
    for n, (la, lo, r) in lobes.items():
        s = smax(s, r * KM_PER_DEG - ang_km(xyz, la, lo))
    gaps = {}
    for a, b in BRIDGES:
        la_a, lo_a, r_a = lobes[a]
        la_b, lo_b, r_b = lobes[b]
        center_dist = np.degrees(np.arccos(np.clip(unit(la_a, lo_a) @ unit(la_b, lo_b), -1, 1)))
        gaps[f"{a}-{b}"] = round(float(center_dist - r_a - r_b), 2)
        s = smax(s, BRIDGE_HALF_WIDTH_KM - dist_to_arc_km(xyz, arc_points((la_a, lo_a), (la_b, lo_b))))
    s = smax(s, secondary_radius_deg * KM_PER_DEG - ang_km(xyz, *SECONDARY_CENTER))
    s = smax(s, POLAR_RADIUS_DEG * KM_PER_DEG - ang_km(xyz, -90.0, 0.0))
    for la, lo, r in ISLANDS:
        s = smax(s, r * KM_PER_DEG - ang_km(xyz, la, lo), k=30.0)
    sea = SEA[2] * KM_PER_DEG - ang_km(xyz, SEA[0], SEA[1])
    strait = STRAIT[2] - dist_to_arc_km(xyz, arc_points(*STRAIT[:2]))
    s = smin(s, -sea)
    s = smin(s, -strait)
    return s.reshape(grid.shape), gaps


def plane_wave_noise(xyz: np.ndarray, rng, wavelengths_km, n_waves: int) -> np.ndarray:
    """球面上各向同性的噪声：随机方向、随机相位的平面波之和（在三维坐标上取值），归一化为标准差 1。"""
    out = np.zeros(xyz.shape[0])
    for _ in range(n_waves):
        k_dir = rng.normal(size=3)
        k_dir /= np.linalg.norm(k_dir)
        lam = rng.uniform(*wavelengths_km) / R_KM
        out += np.cos(xyz @ (k_dir * 2 * np.pi / lam) + rng.uniform(0, 2 * np.pi))
    return (out - out.mean()) / out.std()


def elevation_field(grid: LatLonGrid, land: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    xyz = grid.xyz.reshape(-1, 3)
    d_land = distance_to_set_km(grid, ~land, query=land)      # 陆地到海岸的距离
    d_sea = distance_to_set_km(grid, land, query=~land)       # 海洋到海岸的距离
    big = plane_wave_noise(xyz, rng, (800.0, 3000.0), 120).reshape(grid.shape)
    small = plane_wave_noise(xyz, rng, (120.0, 500.0), 160).reshape(grid.shape)
    elev = np.empty(grid.shape)
    # 陆地：缓慢抬升的台地 + 随机噪声斑块"山脉"（没有走向、没有构造成因）+ 细小起伏
    plateau = 120.0 + 800.0 * (1.0 - np.exp(-np.nan_to_num(d_land) / 600.0))
    blobs = 3800.0 * np.clip((big - 1.1) / 1.6, 0.0, 1.0) ** 1.3
    elev[land] = np.maximum(plateau + blobs + 120.0 * small, 5.0)[land]
    # 海洋：0–150 km 内的窄陆架（-40 到 -200 m），再以光滑陆坡降到约 -4300 m 的平坦深海。
    # 噪声在陆架外 150–600 km 内逐渐加强，并保证海洋格点高程 ≤ -20 m、陆地 ≥ 5 m：
    # 否则近岸会出现"海里的正高程点、陆上的负高程点"，与海陆掩膜矛盾（第一版出现过）。
    ds = np.nan_to_num(d_sea)
    shelf = -40.0 - 160.0 * np.clip(ds / 150.0, 0, 1)
    ramp = np.clip((ds - 150.0) / 450.0, 0, 1)
    deep = -200.0 - 4100.0 * (1.0 - np.exp(-np.clip(ds - 150.0, 0, None) / 250.0)) + 150.0 * small * ramp
    elev[~land] = np.minimum(np.where(ds <= 150.0, shelf, deep), -20.0)[~land]
    return elev


def downsample(field: np.ndarray, f: int) -> np.ndarray:
    """2^k 倍面积平均降采样（同一行内各格面积相同，行间按面积加权）。"""
    ny, nx = field.shape
    g = LatLonGrid.from_shape(field.shape)
    w = g.row_area_km2[:, None]
    num = (field * w).reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    den = np.broadcast_to(w, field.shape).reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))
    return num / den


def main() -> None:
    t0 = time.perf_counter()
    grid = LatLonGrid(2048, 1024)
    area = grid.cell_area_km2

    def ocean_fraction(r):
        s, _ = signed_land(grid, r)
        return 1.0 - float((area * (s > 0)).sum() / area.sum())

    lo, hi = 10.0, 40.0  # 次大陆半径（°）：半径越大海洋越少
    for _ in range(18):
        mid = (lo + hi) / 2
        if ocean_fraction(mid) > TARGET_OCEAN:
            lo = mid
        else:
            hi = mid
    r_sec = round((lo + hi) / 2, 3)
    s, gaps = signed_land(grid, r_sec)
    land = s > 0
    print(f"次大陆半径 {r_sec}°，海洋占比 {ocean_fraction(r_sec):.4f}，陆块间隙（°）{gaps}", flush=True)
    if min(gaps.values()) <= 0:
        raise SystemExit("陆块相互重叠：颈部会变宽，请调整陆块位置")
    elev = elevation_field(grid, land)

    OUT.mkdir(parents=True, exist_ok=True)
    attrs = {"world_id": "rounded_control", "display_name": "圆润世界（对照样本）", "seed": str(SEED),
             "method_class_elevation": "visual_only", "code_version": code_version(),
             "source": "scripts/make_rounded_world.py：光滑解析几何 + 随机噪声斑块（阶段 0.7 故意做坏的对照样本，不是生成的世界）"}
    files = {}
    for f in (1, 2, 4):
        g = LatLonGrid(2048 // f, 1024 // f)
        e = elev if f == 1 else downsample(elev, f)
        lf = land.astype(float) if f == 1 else downsample(land.astype(float), f)
        ds = xr.Dataset({"elevation": (("lat", "lon"), e.astype("float32"), {"units": "m"}),
                         "land": (("lat", "lon"), lf.astype("float32"), {"description": "陆地比例（≥ 0.5 为陆地）"})},
                        coords={"lat": g.lat, "lon": g.lon}, attrs={**attrs, "grid": g.name})
        p = OUT / f"rounded_control_{g.name}.nc"
        ds.to_netcdf(p)
        files[g.name] = p.relative_to(ROOT).as_posix()
    params = {"description": "阶段 0.7 圆润世界对照样本的生成参数（重建：scripts/make_rounded_world.py）",
              "seed": SEED, "method_class_elevation": "visual_only", "lobes": LOBES, "bridges": BRIDGES,
              "bridge_half_width_km": BRIDGE_HALF_WIDTH_KM, "inland_sea": SEA, "strait": STRAIT,
              "secondary_center": SECONDARY_CENTER, "secondary_radius_deg_tuned": r_sec, "polar_radius_deg": POLAR_RADIUS_DEG,
              "islands": ISLANDS, "smooth_union_km": SMOOTH_KM, "target_ocean_fraction": TARGET_OCEAN,
              "lobe_gaps_deg": gaps, "files": files, "code_version": attrs["code_version"],
              "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"写出 {list(files.values())}，用时 {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
