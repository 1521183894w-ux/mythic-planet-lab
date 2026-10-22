"""阶段 1.1 对比图。读取 run_all.py 的结果，写到 figures/stage1/methods/。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u spikes\\stage1_methods\\figures.py --level 7 --seeds 0-7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import CACHE, FIG_DIR, R_KM, RES_DIR, Raster, code_version, load_mesh, unit, xyz_to_lonlat  # noqa: E402
from evaluate import basin_check  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mythic.figures.style import setup_fonts, shaded_elevation  # noqa: E402

RUN_VERSION = {"v": "?"}   # 运行结果所用的代码版本（main 中从结果 JSON 读出）

PC = ccrs.PlateCarree()
PROJ = ccrs.Robinson(central_longitude=10)
ROUTE_NAME = {"A": "路线 A（涌现型，Cortial 式）", "B": "路线 B（pyGPlates 剧本 + 规划）"}
ROUTE_COLOR = {"A": "#d95f02", "B": "#1b6ca8"}
BOUND_COLORS = {0: ("#d62728", "汇聚（俯冲/碰撞）"), 1: ("#1f77b4", "离散（洋脊/裂谷）"), 2: ("#2ca02c", "转换")}
METHOD_NOTE = ("method_class：构造高程 = procedural_constrained（俯冲增厚、碰撞核函数、裂谷阶梯为经验规则）"
               " + parameterized_physics（洋深 GDH1、Airy 均衡）；初始大陆形状为人工设定（几个球冠，无噪声）")


def stamp(fig, extra: str = ""):
    fig.text(0.005, 0.003, f"{extra}{METHOD_NOTE}\n阶段 1.1 构造方案原型 · 运行代码 {RUN_VERSION['v']} · "
             f"出图代码 {code_version()} · spikes/stage1_methods", fontsize=6.5, color="0.3", va="bottom")


# ---- 读取 -----------------------------------------------------------------------------------
def load_run(route: str, seed: int, level: int):
    tag = f"{route}{seed}_L{level}"
    js = json.loads((RES_DIR / f"run_{tag}.json").read_text(encoding="utf-8"))
    z = np.load(CACHE / "runs" / f"{tag}.npz")
    frames = []
    for k, t in enumerate(z["times"]):
        f = {"t": float(t)}
        for name in ("elev", "T", "age", "plate", "ctype", "origin", "oro", "orot"):
            f[name] = z[f"{name}_{k}"]
        f["omega_ids"], f["omega_vec"] = z[f"omega_ids_{k}"], z[f"omega_vec_{k}"]
        frames.append(f)
    return js, frames


def frame_at(frames, t):
    return min(frames, key=lambda f: abs(f["t"] - t))


# ---- 板块边界 -------------------------------------------------------------------------------
def boundaries(mesh, f):
    plate = f["plate"].astype(int)
    e = mesh.edges
    a, b = plate[e[:, 0]], plate[e[:, 1]]
    sel = (a != b) & (a >= 0) & (b >= 0)
    i, j, a, b = e[sel, 0], e[sel, 1], a[sel], b[sel]
    mid = unit(mesh.xyz[i] + mesh.xyz[j])
    om = np.zeros((max(int(plate.max()), int(f["omega_ids"].max())) + 1, 3))
    om[f["omega_ids"].astype(int)] = f["omega_vec"]
    rel = (np.cross(om[a], mid) - np.cross(om[b], mid)) * R_KM / 10.0     # cm/yr
    n = mesh.xyz[j] - mesh.xyz[i]
    n = unit(n - (n * mid).sum(1, keepdims=True) * mid)
    vn = (rel * n).sum(1)
    vt = np.linalg.norm(rel - vn[:, None] * n, axis=1)
    kind = np.where((vn > 0.3) & (vn >= 0.5 * vt), 0, np.where((vn < -0.3) & (-vn >= 0.5 * vt), 1, 2))
    lon, lat = xyz_to_lonlat(mid)
    return lon, lat, kind


def draw_map(ax, raster, mesh, f, bounds=True, extent=None, s=0.5):
    elev = raster(f["elev"])
    rgb = shaded_elevation(elev, raster.grid.pixel_km_equator, land=elev > 0)
    ax.imshow(rgb, transform=PC, extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
    if bounds:
        lon, lat, kind = boundaries(mesh, f)
        for k, (col, _) in BOUND_COLORS.items():
            m = kind == k
            ax.scatter(lon[m], lat[m], s=s, c=col, transform=PC, linewidths=0, rasterized=True)
    if extent:
        ax.set_extent(extent, crs=PC)
    else:
        ax.set_global()
    ax.gridlines(color="0.5", lw=0.3, alpha=0.5)


def bound_legend(fig, y=0.02):
    hs = [Patch(color=c, label=l) for c, l in BOUND_COLORS.values()]
    fig.legend(handles=hs, loc="lower right", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0.99, y))


# ---- 图 1：情景与初始条件 ----------------------------------------------------------------------
def fig_setup(mesh, raster, runs):
    jsB, frB = runs[("B", 0)]
    jsA, frA = runs[("A", 0)]
    fig = plt.figure(figsize=(15, 4.9))
    for k, (js, fr, title) in enumerate([(jsA, frA, "路线 A 起点（种子 0）：初始运动 = 种子扰动后的基准运动"),
                                         (jsB, frB, "路线 B 起点（种子 0）：初始运动 = 规划器修正后的剧本")]):
        ax = fig.add_subplot(1, 2, k + 1, projection=PROJ)
        f = fr[0]
        draw_map(ax, raster, mesh, f, s=1.2)
        # 速度箭头
        lons, lats = np.meshgrid(np.arange(-170, 180, 20), np.arange(-70, 80, 15))
        x = np.stack([np.cos(np.radians(lats)) * np.cos(np.radians(lons)),
                      np.cos(np.radians(lats)) * np.sin(np.radians(lons)), np.sin(np.radians(lats))], -1).reshape(-1, 3)
        _, nn = mesh.tree.query(x)
        pl = f["plate"][nn].astype(int)
        om = np.zeros((int(f["omega_ids"].max()) + 1, 3))
        om[f["omega_ids"].astype(int)] = f["omega_vec"]
        v = np.cross(om[pl], x) * R_KM / 10.0
        east = unit(np.cross([0, 0, 1.0], x))
        north = np.cross(x, east)
        ax.quiver(lons.ravel(), lats.ravel(), (v * east).sum(1), (v * north).sum(1), transform=PC, scale=45,
                  width=0.0018, color="k", alpha=0.75)
        for name, (lo, la) in {"C1 北大陆": (10, 52), "C2 南大陆": (12, -25), "古洋盆（特提斯式）": (8, 12),
                               "西南岬角": (-22, 20), "东北岬角": (46, 8)}.items():
            ax.text(lo, la, name, transform=PC, fontsize=8, ha="center", color="k",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        ax.set_title(title, fontsize=10)
    bound_legend(fig, 0.06)
    fig.suptitle("阶段 1.1 原型情景（100 Ma 起点，两条路线相同的初始几何）：南大陆向北汇聚，"
                 "目标是一端岬角碰撞锁定、另一端留窄海峡，中间留下残留洋盆\n箭头 = 板块速度（最长约 5 cm/yr）；板块 1 = 北大陆，"
                 "板块 2 = 南大陆 + 古洋盆，板块 3 = 远洋。大陆形状故意用球冠拼成、没有噪声，海岸上的任何复杂度只能来自构造过程",
                 fontsize=10)
    stamp(fig)
    fig.savefig(FIG_DIR / "01_setup.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---- 图 2/3：古地理序列 ----------------------------------------------------------------------
def fig_sequence(mesh, raster, route, seed, js, frames, fname):
    times = [100, 80, 60, 40, 20, 0]
    fig = plt.figure(figsize=(16, 11.5))
    for k, t in enumerate(times):
        f = frame_at(frames, t)
        ax = fig.add_subplot(3, 2, k + 1, projection=PROJ)
        draw_map(ax, raster, mesh, f)
        ev = [e for e in js["events"] if f["t"] < e["t"] <= f["t"] + 20 and e["kind"] in
              ("collision_onset", "rift", "terrane_transfer", "continent_contact")] if t < 100 else []
        for e in ev:
            if "lat" in e:
                ax.plot(e["lon"], e["lat"], marker="*", ms=9, mec="k", mfc="yellow", transform=PC)
        ax.set_title(f"{f['t']:.0f} Ma", fontsize=11)
    bound_legend(fig, 0.035)
    plan = js.get("plan")
    extra = ""
    if plan:
        extra = (f"规划：欧拉极 ({plan['pole21'][0]:.1f}°, {plan['pole21'][1]:.1f}°)，{plan['rate21']:.3f}°/Myr，"
                 f"预计 {plan['t_contact']:.1f} Ma 在{'西' if plan['contact_end'] == 'west' else '东'}端接触、"
                 f"{plan['t_lock']:.1f} Ma 锁定后保留 15% 汇聚；裂解 {js['rift_t']:.0f} Ma")
    else:
        extra = f"板片拉力 v_max = {js['v_max_cm_yr']:.1f} cm/yr；随机裂解、碰撞后地体转移"
    fig.suptitle(f"{ROUTE_NAME[route]} · 种子 {seed} · 每 20 Myr 一帧（★ = 前 20 Myr 内的碰撞、裂解、地体转移）\n{extra}",
                 fontsize=11)
    stamp(fig)
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---- 图 4：0 Ma 对比 -----------------------------------------------------------------------
PROV_COLORS = ["#dbe9f3", "#e8d7a9", "#c9d79a", "#9b6fbf", "#f39c34", "#c0392b", "#2ab7c9"]
PROV_LABELS = ["洋壳", "北大陆稳定地块", "南大陆稳定地块", "洋内弧增生地壳", "俯冲带上盘增厚（岛弧/安第斯型）",
               "碰撞造山带", "裂谷减薄陆缘"]


def provenance(f):
    cat = np.zeros(f["ctype"].shape, int)
    cont = f["ctype"] == 1
    cat[cont & (f["origin"] == 1)] = 1
    cat[cont & (f["origin"] == 2)] = 2
    cat[cont & (f["origin"] == 90)] = 3
    cat[cont & (f["oro"] == 1)] = 4
    cat[cont & (f["oro"] == 2)] = 5
    cat[cont & (f["oro"] == 3)] = 6
    return cat


def fig_final(mesh, raster, runs, seed):
    fig = plt.figure(figsize=(17, 9.6))
    ext = (-75, 95, -60, 85)
    for r, route in enumerate("AB"):
        js, fr = runs[(route, seed)]
        f = fr[-1]
        ax = fig.add_subplot(2, 3, 3 * r + 1, projection=PC)
        draw_map(ax, raster, mesh, f, extent=ext, s=1.0)
        ax.set_title(f"{ROUTE_NAME[route]}：0 Ma 高程与板块边界", fontsize=9.5)
        ax = fig.add_subplot(2, 3, 3 * r + 2, projection=PC)
        ax.imshow(raster(provenance(f)), transform=PC, extent=(-180, 180, -90, 90), origin="upper",
                  cmap=ListedColormap(PROV_COLORS), vmin=-0.5, vmax=6.5, interpolation="nearest")
        ax.contour(np.linspace(-180, 180, raster.grid.nx), np.linspace(90, -90, raster.grid.ny),
                   raster(f["elev"]), [0], colors="k", linewidths=0.4, transform=PC)
        ax.set_extent(ext, crs=PC)
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)
        ax.set_title("地壳来源与最近一次构造改造（黑线 = 海岸）", fontsize=9.5)
        ax = fig.add_subplot(2, 3, 3 * r + 3, projection=PC)
        age = np.where(f["ctype"] == 0, f["age"], np.nan)
        im = ax.imshow(raster(age), transform=PC, extent=(-180, 180, -90, 90), origin="upper", cmap="viridis_r",
                       vmin=0, vmax=160, interpolation="nearest")
        ax.set_extent((-180, 180, -90, 90), crs=PC)
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)
        ax.set_title("洋壳年龄（Myr；陆壳留白）", fontsize=9.5)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    hs = [Patch(color=c, label=l) for c, l in zip(PROV_COLORS, PROV_LABELS)]
    fig.legend(handles=hs, loc="lower center", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(f"种子 {seed}：两条路线在 0 Ma 的结果（同一初始几何、同一套地壳规则，只有运动学不同）", fontsize=11)
    stamp(fig)
    fig.savefig(FIG_DIR / "04_final_compare.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


# ---- 图 5：联系表 -----------------------------------------------------------------------------
def fig_contact(mesh, raster, runs, seeds, evals):
    fig = plt.figure(figsize=(2.3 * len(seeds) + 1, 5.2))
    fig.subplots_adjust(hspace=0.35, wspace=0.08, top=0.8, bottom=0.08)
    ext = (-60, 80, -45, 70)
    for r, route in enumerate("AB"):
        for k, s in enumerate(seeds):
            js, fr = runs[(route, s)]
            ev = evals[(route, s)]
            ax = fig.add_subplot(2, len(seeds), r * len(seeds) + k + 1, projection=PC)
            draw_map(ax, raster, mesh, fr[-1], bounds=False, extent=ext)
            off, loose = ev["official"], ev["loose"]
            for lab, col in (("loose", "#ffcc00"), ("official", "#ff2020")):
                if ev[lab]["exists"] and "_mask" in ev[lab]:
                    ax.contour(np.linspace(-180, 180, raster.grid.nx), np.linspace(90, -90, raster.grid.ny),
                               ev[lab]["_mask"].astype(float), [0.5], colors=col, linewidths=0.9, transform=PC)
            tag = "√ 主内海" if off["hit"] else ("△ 残留洋盆" if loose["hit"] else "× 未形成")
            info = (f"{loose['area_Mkm2']:.1f} Mkm² · 长宽比 {loose['aspect_ratio']:.1f} · 海峡 {loose['straits_to_ocean']}"
                    if loose["exists"] else "无半封闭海")
            ax.set_title(f"{route}{s}  {tag}\n{info}", fontsize=8,
                         color="#b00000" if off["hit"] else ("#8a6d00" if loose["hit"] else "0.3"))
    hitsA = sum(evals[("A", s)]["official"]["hit"] for s in seeds)
    hitsB = sum(evals[("B", s)]["official"]["hit"] for s in seeds)
    looseA = sum(evals[("A", s)]["loose"]["hit"] for s in seeds)
    looseB = sum(evals[("B", s)]["loose"]["hit"] for s in seeds)
    fig.suptitle(f"联系表：0 Ma，上排路线 A、下排路线 B，同一列 = 同一种子（初始几何与运动扰动相同）\n"
                 f"√ = 通过 A 层主内海判定（红线；接壤改为两岸分属两块大陆）：A {hitsA}/{len(seeds)}，B {hitsB}/{len(seeds)}；"
                 f"△ = 只通过宽松判定（黄线：海峡 < 300 km、面积 ≥ 50 万 km²、两岸分属两块大陆）：A {looseA}/{len(seeds)}，"
                 f"B {looseB}/{len(seeds)}", fontsize=10)
    stamp(fig)
    fig.savefig(FIG_DIR / "05_contact_sheet.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---- 图 6：运动学与历史 ------------------------------------------------------------------------
def closure_series(mesh, frames):
    from scipy.spatial import cKDTree
    out = []
    for f in frames:
        land = f["elev"] > 0
        a = land & (f["origin"] == 1)
        b = land & (f["origin"] == 2)
        if a.any() and b.any():
            d, _ = cKDTree(mesh.xyz[a]).query(mesh.xyz[b])
            out.append((f["t"], float(2 * R_KM * np.arcsin(min(d.min(), 2) / 2))))
    return np.array(out)


def fig_kinematics(mesh, runs, seeds):
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
    ax = axs[0, 0]
    for route in "AB":
        for s in seeds:
            js, _ = runs[(route, s)]
            t = np.array([r["t"] for r in js["speeds"]])
            v = np.array([r.get("2", np.nan) for r in js["speeds"]])
            ax.plot(t, v, color=ROUTE_COLOR[route], lw=0.9, alpha=0.75, label=ROUTE_NAME[route] if s == seeds[0] else None)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("板块 2（南大陆所在板块）平均速度（cm/yr）")
    ax.axhspan(1, 10, color="0.9", zorder=0)
    ax.text(98, 9.4, "地球现今大板块常见范围约 1–10 cm/yr", fontsize=8, color="0.4")
    ax.legend(fontsize=8)
    ax.set_title("运动学：路线 A 的速度由板片拉力等内部规则决定，路线 B 按剧本", fontsize=10)

    ax = axs[0, 1]
    for route in "AB":
        for s in seeds:
            _, fr = runs[(route, s)]
            cs = closure_series(mesh, fr)
            if cs.size:
                ax.plot(cs[:, 0], cs[:, 1], color=ROUTE_COLOR[route], lw=0.9, alpha=0.75, marker=".", ms=3,
                        label=ROUTE_NAME[route] if s == seeds[0] else None)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("两块大陆陆地（高程 > 0）之间的最近距离（km）")
    ax.set_title("古洋盆闭合过程（降到约一个网格间距 = 两岸陆地相接）", fontsize=10)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    marks = {"collision_onset": ("o", "碰撞开始"), "terrane_transfer": ("s", "地体转移（路线 A）"),
             "rift": ("^", "裂解"), "plate_death": ("x", "板块消亡")}
    for r, route in enumerate("AB"):
        for k, s in enumerate(seeds):
            js, _ = runs[(route, s)]
            y = r * (len(seeds) + 2) + k
            for e in js["events"]:
                if e["kind"] in marks:
                    ax.plot(e["t"], y, marks[e["kind"]][0], color=ROUTE_COLOR[route], ms=6, mfc="none" if e["kind"] == "rift" else ROUTE_COLOR[route])
    ax.set_yticks([r * (len(seeds) + 2) + k for r in range(2) for k in range(len(seeds))])
    ax.set_yticklabels([f"{rt}{s}" for rt in "AB" for s in seeds], fontsize=7)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_title("事件时间线（○ 碰撞开始，□ 地体转移，△ 裂解）", fontsize=10)

    ax = axs[1, 1]
    for route in "AB":
        for s in seeds:
            js, _ = runs[(route, s)]
            h = js["tracers"]["C2 东北岬角"]
            ax.plot([p[0] for p in h], [p[2] for p in h], color=ROUTE_COLOR[route], lw=0.9, alpha=0.75,
                    label=ROUTE_NAME[route] if s == seeds[0] else None)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("古纬度（°）")
    ax.set_title("示踪点“南大陆东北岬角”的古纬度历史（矿产需要的历史量之一）", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"运动学与历史（L7，种子 {seeds[0]}–{seeds[-1]}）", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    stamp(fig)
    fig.savefig(FIG_DIR / "06_kinematics.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


# ---- 图 7：残留洋盆放大，与地中海同比例尺 ---------------------------------------------------------
def fig_zoom(mesh, raster, runs, evals, picks):
    import xarray as xr
    from mythic.figures.world import EARTH_REF
    with xr.open_dataset(EARTH_REF / "earth_grid_1024x512.nc") as ds:
        e_elev = ds.elevation.values.astype(float)
        e_land = ds.land_fraction.values >= 0.5
    W, H = 3600e3, 2300e3
    panels = [(f"{rt}{s}", rt, s) for rt, s in picks] + [("地球：地中海", None, None)]
    fig = plt.figure(figsize=(5.2 * len(panels), 4.6))
    for k, (name, rt, s) in enumerate(panels):
        if rt is None:
            lat0, lon0, elev, land = 37.0, 18.0, e_elev, e_land
            title = "地球：地中海（同比例尺对照，ETOPO 2022 重采样到 1024x512）"
        else:
            js, fr = runs[(rt, s)]
            ev = evals[(rt, s)]["loose"]
            lat0, lon0 = ev["centroid_lat_lon"] if ev["exists"] else (12.0, 10.0)
            elev = raster(fr[-1]["elev"])
            land = elev > 0
            tag = "通过主内海判定" if evals[(rt, s)]["official"]["hit"] else ("宽松判定" if ev["hit"] else "未形成")
            title = f"{ROUTE_NAME[rt]} 种子 {s}（{tag}）"
        proj = ccrs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
        ax = fig.add_subplot(1, len(panels), k + 1, projection=proj)
        rgb = shaded_elevation(elev, 2 * np.pi * R_KM / elev.shape[1], land=land)
        ax.imshow(rgb, transform=PC, extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
        ax.set_extent((-W, W, -H, H), crs=proj)
        ax.gridlines(color="0.5", lw=0.3, alpha=0.6)
        ax.plot([-W + 2e5, -W + 1.2e6], [-H + 2e5, -H + 2e5], color="k", lw=3, transform=proj)
        ax.text(-W + 7e5, -H + 2.8e5, "1000 km", ha="center", fontsize=8, transform=proj)
        ax.set_title(title, fontsize=9)
    fig.suptitle("残留洋盆放大（等距方位投影，各图同比例尺；格网 1024x512，L7 网格间距约 55 km）", fontsize=10.5)
    stamp(fig)
    fig.savefig(FIG_DIR / "07_basin_zoom.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---- 图 8：诊断 -------------------------------------------------------------------------------
def fig_diag(mesh, runs, seeds):
    from engine import Params, gdh1_depth_m
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    js, fr = runs[("B", seeds[0])]
    f = fr[-1]
    oc = (f["ctype"] == 0) & (f["T"] < 7.5)
    ax.scatter(f["age"][oc], f["elev"][oc], s=1, alpha=0.25, color="0.4", rasterized=True, label="原型 0 Ma 的洋壳网格点（含海沟下沉）")
    a = np.linspace(0, 180, 200)
    ax.plot(a, -gdh1_depth_m(a), "r-", lw=1.5, label="GDH1（Stein & Stein 1992）")
    ax.set_xlabel("洋壳年龄（Myr）")
    ax.set_ylabel("高程（m）")
    ax.legend(fontsize=8)
    ax.set_title("洋壳水深—年龄（路线 B 种子 0）", fontsize=10)

    ax = axs[0, 1]
    for route in "AB":
        for s in seeds:
            js, _ = runs[(route, s)]
            cl = js["commit_log"]
            t = [c["t"] for c in cl]
            ax.plot(t, [c["cont_area_km2"] / 1e6 for c in cl], color=ROUTE_COLOR[route], lw=0.9, alpha=0.7,
                    label=f"{ROUTE_NAME[route]} 陆壳面积" if s == seeds[0] else None)
            ax.plot(t, [c["land_area_km2"] / 1e6 for c in cl], color=ROUTE_COLOR[route], lw=0.9, alpha=0.7, ls="--",
                    label=f"{ROUTE_NAME[route]} 陆地面积（高程 > 0）" if s == seeds[0] else None)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("百万 km²")
    ax.legend(fontsize=7)
    ax.set_title("陆壳与陆地面积（碰撞把面积换成厚度，洋内弧增生增加陆壳）", fontsize=10)

    ax = axs[1, 0]
    for route in "AB":
        for s in seeds:
            js, _ = runs[(route, s)]
            cl = js["commit_log"]
            ax.plot([c["t"] for c in cl], [c["covered_area_km2"] / (4 * np.pi * R_KM**2) for c in cl],
                    color=ROUTE_COLOR[route], lw=0.9, alpha=0.7, label=ROUTE_NAME[route] if s == seeds[0] else None)
    ax.set_xlim(100, 0)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("提交后被板块覆盖的球面比例")
    ax.legend(fontsize=8)
    ax.set_title("覆盖完整性（应为 1：没有裂缝、没有漏填的空隙）", fontsize=10)

    ax = axs[1, 1]
    labels, vals = [], []
    for route in "AB":
        for s in seeds:
            js, _ = runs[(route, s)]
            labels.append(f"{route}{s}")
            vals.append(js["wall_s"])
    ax.bar(range(len(vals)), vals, color=[ROUTE_COLOR[l[0]] for l in labels])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("单个种子总耗时（s，含规划、出快照）")
    ax.set_title("耗时（台式机 i5-12600KF，4 个种子并行，L7 约 16.4 万网格点，100 Myr）", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig)
    fig.savefig(FIG_DIR / "08_diagnostics.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    from run_all import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--seeds", default="0-7")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    setup_fonts()
    mesh = load_mesh(args.level)
    raster = Raster(mesh, 512)
    runs = {(r, s): load_run(r, s, args.level) for r in "AB" for s in seeds}
    RUN_VERSION["v"] = ",".join(sorted({js.get("code_version", "?") for js, _ in runs.values()}))
    evals = {}
    for k, (js, fr) in runs.items():
        f = fr[-1]
        evals[k] = basin_check(raster.grid, raster(f["elev"]) > 0, raster(f["origin"]))
    print("loaded", flush=True)
    fig_setup(mesh, raster, runs)
    print("01", flush=True)
    fig_sequence(mesh, raster, "A", seeds[0], *runs[("A", seeds[0])], "02_sequence_A.png")
    fig_sequence(mesh, raster, "B", seeds[0], *runs[("B", seeds[0])], "03_sequence_B.png")
    print("02-03", flush=True)
    fig_final(mesh, raster, runs, seeds[0])
    print("04", flush=True)
    fig_contact(mesh, raster, runs, seeds, evals)
    print("05", flush=True)
    fig_kinematics(mesh, runs, seeds)
    print("06", flush=True)
    picksA = [s for s in seeds if evals[("A", s)]["loose"]["hit"]][:1] or [seeds[0]]
    fig_zoom(mesh, raster, runs, evals, [("A", picksA[0]), ("B", seeds[0])])
    print("07", flush=True)
    fig_diag(mesh, runs, seeds)
    print("08", flush=True)
    from run_all import _jsonable
    summary = {f"{r}{s}": {lab: evals[(r, s)][lab] for lab in ("official", "loose")} for (r, s) in evals}
    (RES_DIR / f"evals_L{args.level}.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1),
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
