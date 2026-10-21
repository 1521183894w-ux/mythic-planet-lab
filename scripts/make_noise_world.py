"""阶段 0.7 的第二个对照样本："噪声海岸世界"（故意做坏，但更狡猾）。

做法 = "把圆变成带噪声的圆"：沿用圆润世界的大格局（scripts/make_rounded_world.py 的有符号距离场），
叠加多尺度分形噪声（各向同性、功率谱幂律，Hurst 指数 H），海岸 = 场的零等值线。海岸因此曲折、有半岛海湾和碎岛，
但没有任何地质过程：噪声不知道板块、岛弧、冰川、河流。地形同样由分形噪声生成。

为了让它尽量"骗过"指标，生成器像一个钻空子的程序那样做了两件事（如实记录，见 params.json）：
1. 噪声只加在 ≤ 2000 km 的尺度上（不动大格局），并在陆桥、海峡与主内海岸边附近削弱噪声，保住 A 层要求的连通与海峡；
2. 换随机种子筛选，直到 A 层 7 项全部达标（先在 1024x512 上筛，再在 2048x1024 上确认）。
噪声强度（A0、H）的选取目标是让 B1 海岸指标尽量达标。

method_class：visual_only。输出 data/worlds/noise_control/noise_control_<分辨率>.nc（nc 不进 Git）与 params.json（进 Git）。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_noise_world.py --explore   # 参数探索（1024x512）
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_noise_world.py             # 筛种子并生成（默认 A0 250 km、H 0.5）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

# 同目录脚本：复用圆润世界的几何（运行 scripts/xxx.py 时 Python 会把 scripts/ 放进 sys.path）
from make_rounded_world import (BRIDGES, KM_PER_DEG, LOBES, R_KM, SEA, SECONDARY_CENTER, STRAIT, ang_km, arc_points,
                                dist_to_arc_km, downsample, signed_land, smax)

from mythic.figures.world import code_version
from mythic.metrics.compute import compute_all, load_config
from mythic.metrics.grid import LatLonGrid
from mythic.metrics.spherical import distance_to_set_km
from mythic.metrics.verdict import b_verdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "worlds" / "noise_control"
TARGET_OCEAN = 0.710
LAM0_KM = 2000.0                    # 噪声最大波长：只扰动 ≤ 2000 km 的尺度，大格局保持（第一版用 8000 km，40 个种子都过不了 A 层）
OCTAVES_KM = [LAM0_KM / 2 ** i for i in range(6)]   # 2000 … 62.5 km
WAVES_PER_OCTAVE = 160
PROTECT_INNER_KM, PROTECT_OUTER_KM = 150.0, 500.0  # 陆桥、海峡轴线与主内海岸边附近削弱噪声的范围
SPLIT_KM = 500.0                    # 主内海岸边只挡住 ≥ 500 km 的大尺度噪声
SMALL_FLOOR = 0.4                   # 陆桥、海峡附近保留的小尺度噪声比例


class FbmField:
    """各向同性分形噪声：每个倍频程若干随机方向、随机相位的平面波（在三维坐标上取值），
    振幅 ∝ (波长/最大波长)^H。参数在构造时固定，可以在任意分辨率的格网上取值，得到同一个场。"""

    def __init__(self, seed: int, amp0_km: float, hurst: float, octaves_km=OCTAVES_KM, n_waves=WAVES_PER_OCTAVE):
        rng = np.random.default_rng(seed)
        self.waves = []
        for lam in octaves_km:
            d = rng.normal(size=(n_waves, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            lams = lam * rng.uniform(0.75, 1.33, n_waves)
            k = d * (2 * np.pi / (lams / R_KM))[:, None]
            ph = rng.uniform(0, 2 * np.pi, n_waves)
            amp = amp0_km * (lam / octaves_km[0]) ** hurst * np.sqrt(2.0 / n_waves)  # 每个倍频程标准差 = amp0·(λ/λ0)^H
            self.waves.append((lam, k, ph, amp))

    def __call__(self, xyz: np.ndarray, min_lam: float = 0.0, max_lam: float = np.inf, chunk: int = 32) -> np.ndarray:
        """只累加波长在 [min_lam, max_lam) 内的倍频程。"""
        out = np.zeros(len(xyz))
        for lam, k, ph, amp in self.waves:
            if not (min_lam <= lam < max_lam):
                continue
            for a in range(0, len(k), chunk):
                out += amp * np.cos(xyz @ k[a:a + chunk].T + ph[a:a + chunk]).sum(axis=1)
        return out


def _ramp(d: np.ndarray, inner: float, outer: float) -> np.ndarray:
    return np.clip((d - inner) / (outer - inner), 0, 1)


def protection(grid: LatLonGrid) -> tuple[np.ndarray, np.ndarray]:
    """噪声衰减系数（0 = 不加噪声，1 = 全量）：
    结构保护——陆桥、海峡轴线附近全部尺度都削弱（否则陆桥断开、海峡被堵）；
    岸边保护——主内海岸边只削弱 ≥ 500 km 的大尺度噪声（否则内海被冲出新缺口），小尺度破碎细节照常加上，
    免得内海留下一个正圆的破绽（第三版整片保护，内海是正圆、陆桥笔直，评审一眼就能看穿，测试不公平）。"""
    xyz = grid.xyz.reshape(-1, 3)
    lobes = {n: (la, lo) for n, la, lo, _ in LOBES}
    d = np.full(len(xyz), np.inf)
    for a, b in BRIDGES:
        d = np.minimum(d, dist_to_arc_km(xyz, arc_points(lobes[a], lobes[b])))
    d = np.minimum(d, dist_to_arc_km(xyz, arc_points(*STRAIT[:2])))
    rim = np.abs(ang_km(xyz, SEA[0], SEA[1]) - SEA[2] * np.pi * R_KM / 180.0)
    return (_ramp(d, PROTECT_INNER_KM, PROTECT_OUTER_KM).reshape(grid.shape),
            _ramp(rim, PROTECT_INNER_KM, PROTECT_OUTER_KM).reshape(grid.shape))


def land_field(grid: LatLonGrid, seed: int, amp0: float, hurst: float) -> tuple[np.ndarray, float]:
    """陆地 = 模板有符号距离 + 噪声 > 0。阈值固定为 0，保证模板的海峡、陆桥保持设计宽度
    （第二版曾用全局阈值调海洋占比，阈值偏离 0 把海峡撑宽到 150 km 以上，主内海与外洋连成一片）；
    海洋占比像圆润世界一样，靠二分调整次大陆半径。返回 (陆地场, 次大陆半径°)。"""
    s0, _ = signed_land(grid, 0.001)                      # 不含次大陆的模板（半径 0.001° 可忽略）
    xyz = grid.xyz.reshape(-1, 3)
    sec = ang_km(xyz, *SECONDARY_CENTER).reshape(grid.shape)
    field = FbmField(seed, amp0, hurst)
    p_struct, p_rim = protection(grid)
    large = field(xyz, min_lam=SPLIT_KM).reshape(grid.shape)
    small = field(xyz, max_lam=SPLIT_KM).reshape(grid.shape)
    # 陆桥、海峡附近的小尺度噪声保留 40%：边缘粗糙、不再笔直，又不至于常常断开（偶尔断开交给种子筛选）
    noise = large * p_struct * p_rim + small * np.maximum(p_struct, SMALL_FLOOR)
    a = grid.cell_area_km2
    lo, hi = 5.0, 45.0
    for _ in range(25):
        r = (lo + hi) / 2
        f = smax(s0, r * KM_PER_DEG - sec) + noise
        ocean = 1.0 - float((a * (f > 0)).sum() / a.sum())
        if ocean > TARGET_OCEAN:
            lo = r
        else:
            hi = r
    r = (lo + hi) / 2
    return smax(s0, r * KM_PER_DEG - sec) + noise, r


def screening_config(cfg: dict) -> dict:
    """筛种子按 0.7 制作时的 A 层规则（config v2）：去掉 U18(c) 之后加的主内海接壤与形状条件。
    这个样本的主内海就是圆的、只在一个亚大陆里，按 v3 规则任何种子都过不了；为了能复现 params.json 里的世界，筛选时不用这两条。"""
    out = json.loads(json.dumps(cfg))
    mis = out["A_hard_constraints"]["main_inland_sea"]
    mis.pop("bordering_subcontinents", None)
    mis.pop("min_aspect_ratio", None)
    return out


def evaluate(grid: LatLonGrid, land: np.ndarray, cfg: dict, res: str) -> dict:
    cfg = screening_config(cfg)
    m = compute_all(np.where(land, 500.0, -4000.0), land, cfg=cfg)
    base = json.loads((ROOT / f"data/earth_reference/baselines_{res}.json").read_text(encoding="utf-8"))
    v = b_verdict(m["by_region"], base["landmass_unit_reference"], cfg)
    return {"A_pass": {k: bool(x["pass"]) for k, x in m["A"].items()}, "A_all": all(x["pass"] for x in m["A"].values()),
            "main_share": m["A"]["main_continent"]["share_of_land"], "subcontinents": m["A"]["subcontinents"]["count"],
            "B": {u: f"{x['n_pass']}/{x['n_judged']}" for u, x in v["units"].items()},
            "main_B_all": v["main_continent_all_pass"], "D": m["B1"]["fractal_dimension"]["D"],
            "islands": m["B1"]["islands"]["n_islands"]}


def elevation_field(grid: LatLonGrid, land: np.ndarray, seed: int) -> np.ndarray:
    """噪声地形：内陆随离岸距离缓慢抬升 + 分形噪声"山地"（阈值以上的噪声）+ 细碎起伏；海洋为陆架 + 陆坡 + 带噪声的深海。"""
    xyz = grid.xyz.reshape(-1, 3)
    mtn = FbmField(seed + 1000, 1.0, 0.8, octaves_km=[4000.0 / 2 ** i for i in range(7)])(xyz).reshape(grid.shape)
    mtn /= mtn.std()
    small = FbmField(seed + 2000, 1.0, 0.5, octaves_km=[400.0 / 2 ** i for i in range(3)])(xyz).reshape(grid.shape)
    small /= small.std()
    d_land = np.nan_to_num(distance_to_set_km(grid, ~land, query=land))
    d_sea = np.nan_to_num(distance_to_set_km(grid, land, query=~land))
    elev = np.empty(grid.shape)
    plateau = 60.0 + 700.0 * (1.0 - np.exp(-d_land / 500.0))
    peaks = 4200.0 * np.clip((mtn - 0.9) / 2.2, 0.0, 1.0) ** 1.1
    elev[land] = np.maximum(plateau + peaks + 150.0 * small, 5.0)[land]
    shelf = -40.0 - 160.0 * np.clip(d_sea / 120.0, 0, 1)
    ramp = np.clip((d_sea - 120.0) / 400.0, 0, 1)
    deep = -200.0 - 4100.0 * (1.0 - np.exp(-np.clip(d_sea - 120.0, 0, None) / 250.0)) + 300.0 * small * ramp
    elev[~land] = np.minimum(np.where(d_sea <= 120.0, shelf, deep), -20.0)[~land]
    return elev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true", help="在 1024x512 上比较几组噪声强度（种子 1）")
    # 默认值即 0.7 选用的参数（--explore 中 B1 全过的几组里振幅最小的一组），不带参数运行即可复现 params.json 中的世界
    ap.add_argument("--amp0", type=float, default=250.0, help="最大波长（LAM0_KM）上的噪声标准差（km）")
    ap.add_argument("--hurst", type=float, default=0.5)
    ap.add_argument("--max-seeds", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config()
    g1 = LatLonGrid(1024, 512)

    if args.explore:
        for amp0 in (250.0, 350.0):
            for hurst in (0.5, 0.6):
                t0 = time.perf_counter()
                f, _ = land_field(g1, 1, amp0, hurst)
                land = f > 0
                r = evaluate(g1, land, cfg, "1024x512")
                print(f"amp0={amp0:.0f} H={hurst}: A 全过={r['A_all']} 主大陆占比 {r['main_share']:.2f} 亚大陆 {r['subcontinents']} "
                      f"D={r['D']:.3f} 岛 {r['islands']} B={r['B']} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return

    t_all = time.perf_counter()
    tried = []
    chosen = None
    for seed in range(1, args.max_seeds + 1):
        f, _ = land_field(g1, seed, args.amp0, args.hurst)
        land = f > 0
        r = evaluate(g1, land, cfg, "1024x512")
        tried.append({"seed": seed, "A_all_1024": r["A_all"], "A_failed": [k for k, v in r["A_pass"].items() if not v],
                      "main_B_all_1024": r["main_B_all"]})
        print(f"种子 {seed}: A 全过={r['A_all']}（未过：{tried[-1]['A_failed']}）主大陆 B1/B2 全过={r['main_B_all']} B={r['B']}",
              flush=True)
        if not r["A_all"]:
            continue
        g2 = LatLonGrid(2048, 1024)
        f2, r_sec = land_field(g2, seed, args.amp0, args.hurst)
        land2 = f2 > 0
        r2 = evaluate(g2, land2, cfg, "2048x1024")
        tried[-1]["A_all_2048"] = r2["A_all"]
        print(f"    2048x1024 确认：A 全过={r2['A_all']} B={r2['B']}", flush=True)
        if r2["A_all"]:
            chosen = (seed, g2, f2, r_sec, land2)
            break
    if chosen is None:
        raise SystemExit(f"{args.max_seeds} 个种子都没有通过 A 层")
    seed, g2, f2, r_sec, land2 = chosen
    elev = elevation_field(g2, land2, seed)

    OUT.mkdir(parents=True, exist_ok=True)
    attrs = {"world_id": "noise_control", "display_name": "噪声海岸世界（对照样本）", "seed": str(seed),
             "method_class_elevation": "visual_only", "code_version": code_version(),
             "source": "scripts/make_noise_world.py：圆润世界大格局 + 多尺度分形噪声（阶段 0.7 第二个故意做坏的对照样本，不是生成的世界）"}
    files = {}
    for fct in (1, 2, 4):
        g = LatLonGrid(2048 // fct, 1024 // fct)
        e = elev if fct == 1 else downsample(elev, fct)
        lf = land2.astype(float) if fct == 1 else downsample(land2.astype(float), fct)
        ds = xr.Dataset({"elevation": (("lat", "lon"), e.astype("float32"), {"units": "m"}),
                         "land": (("lat", "lon"), lf.astype("float32"), {"description": "陆地比例（≥ 0.5 为陆地）"})},
                        coords={"lat": g.lat, "lon": g.lon}, attrs={**attrs, "grid": g.name})
        p = OUT / f"noise_control_{g.name}.nc"
        ds.to_netcdf(p)
        files[g.name] = p.relative_to(ROOT).as_posix()
    params = {"description": "阶段 0.7 第二个对照样本（噪声海岸世界）的生成参数（重建：scripts/make_noise_world.py）",
              "method_class_elevation": "visual_only", "template": "scripts/make_rounded_world.py 的有符号距离场",
              "secondary_radius_deg_tuned": r_sec, "noise": {"amp0_km": args.amp0, "hurst": args.hurst,
                                                                     "octaves_km": OCTAVES_KM, "waves_per_octave": WAVES_PER_OCTAVE},
              "protection_km": [PROTECT_INNER_KM, PROTECT_OUTER_KM], "threshold_km": 0.0, "seed": seed,
              "seed_screening": tried, "files": files, "code_version": attrs["code_version"],
              "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"选中种子 {seed}（共试 {len(tried)} 个），写出 {list(files.values())}，用时 {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
