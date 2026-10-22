"""阶段 1.2b 检查点（U20 修订）：前史生成 300 Ma 初始大陆（多个种子），用阶段 0 的 B1 判定与地球大陆比较，
按剧本规划器的标准编组（300 Ma 时相距 < 250 km 的陆块一起运动），出图。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\tectonics_prehistory.py --seeds 1-8

输出：figures/stage1/prehistory/（图与 summary.json，不进 Git）；.cache/tectonics/prehistory/（各种子的快照）。
U20 起所有抽样都在 L6 设计网格上做（prehistory.design_prehistory），L7 只按同一份设计生成地壳。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "stage1" / "prehistory"
NOTE = ("method_class：省份拼贴（几何、年龄、类型、厚度）= 程序生成的初始条件（procedural_constrained）；"
        "陆块分界、陆缘减薄、夭折裂谷、漂移、新洋壳、岛弧 = 过程（procedural_constrained / GDH1、Airy = parameterized_physics）")
B1_KEYS = ["fractal_dimension", "peninsula_index@100", "peninsula_index@200", "bay_index@100", "bay_index@200"]
# 盘古大陆裂解留下的碎块（今天的陆地面积约值，Mkm²；西兰大陆为大陆地壳面积）：只用于图 7 比较量级与分布形状
EARTH_FRAGMENTS = {"欧亚（除印度、阿拉伯）": 47.0, "非洲": 30.4, "北美": 22.5, "南美": 17.8, "南极洲": 14.2,
                   "澳大利亚—新几内亚": 8.6, "西兰大陆": 4.9, "印度": 4.4, "阿拉伯": 3.2, "格陵兰": 2.2,
                   "马达加斯加": 0.59, "斯里兰卡": 0.066}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def run_one(level: int, seed: int) -> dict:
    from mythic.figures.world import code_version
    from mythic.metrics.compute import compute_all, load_config
    from mythic.metrics.verdict import b_verdict
    from mythic.tectonics.mesh import CACHE, Raster, load_mesh
    from mythic.tectonics.prehistory import build_initial_world
    from mythic.tectonics.scenario import ScenarioParams, blocks_from_world, group_adjacent, world_usable

    mesh = load_mesh(level)
    t0 = time.perf_counter()
    w = build_initial_world(mesh, seed)
    wall = time.perf_counter() - t0
    raster = Raster(mesh, 512)
    final = w.snapshots[-1]
    E = raster(final["elev"])
    cfg = load_config()
    ref = json.loads((ROOT / "data" / "earth_reference" / "baselines_1024x512.json").read_text(encoding="utf-8"))
    m = compute_all(E, E > 0, cfg=cfg)
    bv = b_verdict(m["by_region"], ref["landmass_unit_reference"], cfg)
    raw = blocks_from_world(mesh, w)
    grp = group_adjacent(mesh, raw, links=w.links)
    usable, why = world_usable(grp, ScenarioParams())
    groups = [{"lead": int(b), "members": [int(x) for x in (G.members or (b,))], "area_km2": float(G.area_km2)}
              for b, G in sorted(grp.items(), key=lambda kv: -kv[1].area_km2)]
    d = CACHE / "prehistory"
    d.mkdir(parents=True, exist_ok=True)
    arrs = {"times": np.array([s["t"] for s in w.snapshots])}
    for k, s in enumerate(w.snapshots):
        for f in ("elev", "T", "t_form"):
            arrs[f"{f}_{k}"] = s[f].astype(np.float32)
        for f in ("plate", "ctype", "origin", "prov", "oro"):
            arrs[f"{f}_{k}"] = s[f].astype(np.int32)
    np.savez_compressed(d / f"prehistory_L{level}_s{seed}.npz", **arrs)
    land = E > 0
    summary = {"seed": seed, "level": level, "code_version": code_version(), "wall_s": wall,
               "blocks": w.blocks, "provinces": w.provinces, "boundaries": w.boundaries, "events": w.events,
               "params": w.params, "rot_path": w.rot_path,
               "land_fraction_300Ma": float((raster.grid.cell_area_km2 * land).sum() / raster.grid.total_area_km2),
               "cont_area_300Ma_km2": float(mesh.area_km2[w.crust.ctype == 1].sum()),
               "b_verdict": bv, "n_units": len(bv["units"]),
               "planner_blocks": sorted(int(b) for b in raw), "groups": groups, "usable": usable, "unusable_why": why,
               "n_units_all_pass": int(sum(u["all_pass"] for u in bv["units"].values())),
               "step_wall_s_mean": None}
    (d / f"prehistory_L{level}_s{seed}.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    return {"seed": seed, "wall_s": round(wall, 1), "blocks": [round(b["area_km2"] / 1e6, 1) for b in w.blocks],
            "groups": [g["members"] for g in groups], "usable": usable,
            "land": round(summary["land_fraction_300Ma"], 3), "units": summary["n_units"],
            "units_pass": summary["n_units_all_pass"]}


# ---- 出图 -------------------------------------------------------------------------------------------
def figures(level: int, seeds: list[int]):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from mythic.figures.style import setup_fonts, shaded_elevation
    from mythic.figures.world import EARTH_REF, code_version
    from mythic.tectonics.geometry import R_KM, km_from_chord, unit, xyz_to_lonlat
    from mythic.tectonics.mesh import CACHE, Raster, load_mesh

    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    PC = ccrs.PlateCarree()
    mesh = load_mesh(level)
    raster = Raster(mesh, 512)
    d = CACHE / "prehistory"
    runs = {}
    for s in seeds:
        js = json.loads((d / f"prehistory_L{level}_s{s}.json").read_text(encoding="utf-8"))
        z = np.load(d / f"prehistory_L{level}_s{s}.npz")
        frames = [{f: z[f"{f}_{k}"] for f in ("elev", "T", "t_form", "plate", "ctype", "origin", "prov", "oro")} |
                  {"t": float(t)} for k, t in enumerate(z["times"])]
        runs[s] = (js, frames)
    ver = ",".join(sorted({js["code_version"] for js, _ in runs.values()}))

    def stamp(fig):
        fig.text(0.005, 0.003, f"{NOTE}\n阶段 1.2b 前史 · L{level} · 运行代码 {ver} · 出图代码 {code_version()} · "
                 f"scripts/tectonics_prehistory.py", fontsize=6.5, color="0.3", va="bottom")

    def shaded(ax, elev, ext=None):
        e = raster(elev)
        ax.imshow(shaded_elevation(e, raster.grid.pixel_km_equator, land=e > 0), transform=PC, extent=(-180, 180, -90, 90),
                  origin="upper", interpolation="nearest")
        if ext:
            ax.set_extent(ext, crs=PC)
        else:
            ax.set_global()
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)

    def sc_center(f):
        c = f["ctype"] == 1
        lon, lat = xyz_to_lonlat(unit((mesh.xyz[c] * mesh.area_km2[c, None]).sum(0)))
        return float(lon), float(lat)

    # 图 1：省份拼贴、薄弱带、分块（第一个种子，400 Ma）
    s0 = seeds[0]
    js, fr = runs[s0]
    f = fr[0]
    lon0, lat0 = sc_center(f)
    proj = ccrs.Orthographic(lon0, lat0)
    prov_type = {p["id"]: p["type"] for p in js["provinces"]}
    colors = ["#d9d9d9", "#8c510a", "#d8b365", "#f6e8c3", "#80cdc1"]
    labels = ["洋壳", "太古宙克拉通", "古元古代活动带", "中—新元古代活动带", "古生代增生带（400 Ma 时的年轻造山带）"]
    typ = np.array([prov_type.get(int(p), 0) for p in f["prov"]])
    typ[f["ctype"] != 1] = 0
    fig = plt.figure(figsize=(16, 7.6))
    ax = fig.add_subplot(1, 2, 1, projection=proj)
    ax.imshow(raster(typ), transform=PC, extent=(-180, 180, -90, 90), origin="upper", cmap=ListedColormap(colors),
              vmin=-0.5, vmax=4.5, interpolation="nearest")
    e = mesh.edges
    pa, pb = f["prov"][e[:, 0]], f["prov"][e[:, 1]]
    sel = (pa != pb) & (pa > 0) & (pb > 0)
    bmap = {(min(b["a"], b["b"]), max(b["a"], b["b"])): b for b in js["boundaries"]}
    info = [bmap.get((min(a, b), max(a, b))) for a, b in zip(pa[sel], pb[sel])]
    mid = unit(mesh.xyz[e[sel, 0]] + mesh.xyz[e[sel, 1]])
    lo, la = xyz_to_lonlat(mid)
    wk = np.array([i["weakness"] if i else 0 for i in info])
    fail = np.array([bool(i and i["failed_rift"]) for i in info], bool)
    # 分块时的分界（沿省份边界）与整理后真正裂开的分界（相对运动极下的洋脊段 + 转换断层，U20）
    bb = {p["id"]: p.get("block_breakup", p["block"]) for p in js["provinces"]}
    blk0 = np.array([bb.get(int(a)) != bb.get(int(b)) for a, b in zip(pa[sel], pb[sel])], bool)
    ax.scatter(lo[~fail], la[~fail], s=1.5, c=wk[~fail], cmap="Greys", vmin=0, vmax=1.6, transform=PC,
               linewidths=0, rasterized=True)
    ax.scatter(lo[blk0], la[blk0], s=5, c="#f4a261", transform=PC, linewidths=0, rasterized=True)
    oa, ob = f["origin"][e[:, 0]], f["origin"][e[:, 1]]
    cc = (f["ctype"][e[:, 0]] == 1) & (f["ctype"][e[:, 1]] == 1) & (oa != ob)
    # 穿过省份内部（截弯）：两侧同一省份，且离最近的省份边界超过 120 km。分界整理在 L6 设计网格上做，省份边界在 L7 上的
    # 位置与 L6 差一个格点，不加距离判断时沿省份边界走的分界在 L7 上也会一段段被算成"穿过"
    from scipy.spatial import cKDTree as _KD
    pb_e = (f["ctype"][e[:, 0]] == 1) & (f["ctype"][e[:, 1]] == 1) & (f["prov"][e[:, 0]] != f["prov"][e[:, 1]])
    d_pb = km_from_chord(_KD(unit(mesh.xyz[e[pb_e, 0]] + mesh.xyz[e[pb_e, 1]])).query(
        unit(mesh.xyz[e[:, 0]] + mesh.xyz[e[:, 1]]))[0])
    thr_e = cc & (f["prov"][e[:, 0]] == f["prov"][e[:, 1]]) & (d_pb > 120.0)
    for m_, col in ((cc & ~thr_e, "#c1121f"), (thr_e, "#1d3fbf")):
        lo_c, la_c = xyz_to_lonlat(unit(mesh.xyz[e[m_, 0]] + mesh.xyz[e[m_, 1]]))
        ax.scatter(lo_c, la_c, s=3, c=col, transform=PC, linewidths=0, rasterized=True)
    ax.scatter(lo[fail], la[fail], s=4, c="#7a0177", transform=PC, linewidths=0, rasterized=True)
    ax.set_global()
    ct_ = js["events"][0]["drift"].get("cut_through_provinces") or {}
    thr_txt = (f"\n设计网格上：裂开的分界共 {ct_['boundary_km'] / 1000:.0f} 千 km，其中穿过省份 "
               f"{ct_['through_km'] / ct_['boundary_km']:.0%}，最长一段 {ct_['longest_through_km']:.0f} km"
               if ct_.get("boundary_km") else "")
    ax.set_title(f"种子 {s0}，{f['t']:.0f} Ma：地壳省份（{len(js['provinces'])} 个）与薄弱带（灰线越深越薄弱）\n"
                 f"橙 = 分块时沿薄弱带的分界；真正裂开的分界：红 = 沿省份边界，蓝 = 穿过省份（截弯）；紫 = 夭折裂谷{thr_txt}",
                 fontsize=9.5)
    ax = fig.add_subplot(1, 2, 2, projection=proj)
    shaded(ax, f["elev"])
    ax.set_global()
    ax.set_title(f"{f['t']:.0f} Ma 高程（裂解刚发生：裂谷陆缘与夭折裂谷减薄，省份厚度不同造成内陆起伏）", fontsize=9.5)
    hs = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    hs += [Patch(color="#f4a261", label="分块时的分界（沿薄弱带）"), Patch(color="#c1121f", label="真正裂开的分界：沿省份边界"),
           Patch(color="#1d3fbf", label="真正裂开的分界：穿过省份"), Patch(color="#7a0177", label="夭折裂谷")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("前史第 1–2 步：老超大陆的地壳省份拼贴，沿薄弱缝合带分成陆块", fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "01_provinces_and_breakup.png", dpi=105, bbox_inches="tight")
    plt.close(fig)

    # 图 2：漂移序列（第一个种子）
    proj2 = ccrs.Robinson(central_longitude=lon0)
    fig = plt.figure(figsize=(16, 11))
    for k, fk in enumerate(fr[:6]):
        ax = fig.add_subplot(3, 2, k + 1, projection=proj2)
        shaded(ax, fk["elev"])
        ax.set_title(f"{fk['t']:.0f} Ma", fontsize=10)
    fig.suptitle(f"前史第 3 步：种子 {s0} 的陆块从老超大陆散开（{fr[0]['t']:.0f} -> {fr[-1]['t']:.0f} Ma，每 20 Myr 一帧）",
                 fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "02_drift_sequence.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 图 3：各种子 300 Ma 联系表与 B1
    ncol = 3
    nrow = int(np.ceil(len(seeds) / ncol))
    fig = plt.figure(figsize=(21, 5.2 * nrow + 0.8))
    for k, s in enumerate(seeds):
        js, fr = runs[s]
        lo_, _ = sc_center(fr[0])
        ax = fig.add_subplot(nrow, ncol, k + 1, projection=ccrs.Robinson(central_longitude=lo_))
        shaded(ax, fr[-1]["elev"])
        bv = js["b_verdict"]["units"]
        big = max(bv.values(), key=lambda u: u["area_km2"]) if bv else None
        n_micro = sum(1 for ev in js["events"] if ev.get("kind") == "ridge_jump_microcontinent")
        areas = sorted((b["area_km2"] / 1e6 for b in js["blocks"]), reverse=True)
        txt = (f"种子 {s}：陆块 {len(js['blocks'])} 块（含微大陆 {n_micro}）、编组后 {len(js['groups'])} 组"
               f"（{'可用' if js['usable'] else '不可用'}），陆地 {js['land_fraction_300Ma']:.0%}\n"
               f"陆块面积 Mkm²：{' '.join(f'{a_:.1f}' if a_ < 3 else f'{a_:.0f}' for a_ in areas)}；"
               f"≥500 万 km² 的 {js['n_units']} 个，B1 全部达标 {js['n_units_all_pass']} 个")
        if big:
            m = big["metrics"]
            txt += (f"\n最大陆块 {big['area_km2'] / 1e6:.0f} Mkm²：分形维数 {m['fractal_dimension']['value']:.3f}"
                    f"（线 {m['fractal_dimension']['line']:.3f}），半岛@100/200 {m['peninsula_index@100']['value']:.3f}/"
                    f"{m['peninsula_index@200']['value']:.3f}（线 {m['peninsula_index@100']['line']:.3f}/"
                    f"{m['peninsula_index@200']['line']:.3f}）")
        ax.set_title(txt, fontsize=8)
    fig.suptitle("各种子的 300 Ma 初始大陆（阶段 1 正式演化的起点）与 B1 海岸复杂度（与地球大陆级陆块本体同类比较，1024x512）",
                 fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "03_initial_worlds_300Ma.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 图 4：B1 各项与地球的比较
    ref = json.loads((ROOT / "data" / "earth_reference" / "baselines_1024x512.json").read_text(encoding="utf-8"))
    lr = ref["landmass_unit_reference"]
    fig, axs = plt.subplots(1, len(B1_KEYS), figsize=(17, 4.2))
    for ax, key in zip(axs, B1_KEYS):
        ev = [u[key] for u in lr["units"].values()]
        ax.axhspan(min(ev), max(ev), color="0.88", label="地球大陆级陆块本体范围")
        ax.axhline(lr["pass_line"][key]["value"], color="k", lw=1, ls="--", label="合格线（地球最小值）")
        for k, s in enumerate(seeds):
            js, _ = runs[s]
            for u in js["b_verdict"]["units"].values():
                v = u["metrics"][key]["value"]
                ax.scatter(k, v, s=20 + u["area_km2"] / 1e6, color="tab:green" if u["metrics"][key]["pass"] else "tab:red",
                           alpha=0.8, edgecolors="k", linewidths=0.4)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("种子（点大小 ∝ 陆块面积）")
        ax.set_title(key, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("300 Ma 各陆块（≥ 500 万 km²）的 B1 指标：绿 = 达到地球合格线，红 = 未达到", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    stamp(fig)
    fig.savefig(OUT / "04_b1_vs_earth.png", dpi=105, bbox_inches="tight")
    plt.close(fig)

    # 图 5：裂谷陆缘放大，与南大西洋两岸同比例尺。选 400 Ma 时共享分界最长的两对陆块，窗口放在 300 Ma 时
    # 两侧相对的裂谷陆缘（oro = 裂谷）之间
    from scipy.spatial import cKDTree

    js, fr = runs[s0]
    f = fr[-1]
    f0 = fr[0]
    o0a, o0b = f0["origin"][e[:, 0]], f0["origin"][e[:, 1]]
    c0 = (f0["ctype"][e[:, 0]] == 1) & (f0["ctype"][e[:, 1]] == 1) & (o0a != o0b) & (o0a > 0) & (o0b > 0)
    pr, cnt = np.unique(np.sort(np.stack([o0a[c0], o0b[c0]], 1), 1), axis=0, return_counts=True)
    wins = []
    for (a_, b_) in pr[np.argsort(cnt)[::-1]]:
        ra = np.flatnonzero((f["origin"] == a_) & (f["ctype"] == 1) & (f["oro"] == 3))
        rb = np.flatnonzero((f["origin"] == b_) & (f["ctype"] == 1) & (f["oro"] == 3))
        if len(ra) < 5 or len(rb) < 5:
            continue
        da = km_from_chord(cKDTree(mesh.xyz[rb]).query(mesh.xyz[ra])[0])
        db = km_from_chord(cKDTree(mesh.xyz[ra]).query(mesh.xyz[rb])[0])
        fa, fb = ra[da < 2500], rb[db < 2500]
        if len(fa) < 5 or len(fb) < 5:
            continue
        cx = unit(unit(mesh.xyz[fa].mean(0)) + unit(mesh.xyz[fb].mean(0)))
        if wins and min(np.arccos(np.clip(cx @ w_[0], -1, 1)) for w_ in wins) < np.radians(25):
            continue
        wins.append((cx, int(a_), int(b_), float(min(da.min(), db.min()))))
        if len(wins) == 2:
            break
    with xr.open_dataset(EARTH_REF / "earth_grid_1024x512.nc") as ds:
        ee = ds.elevation.values.astype(float)
        el = ds.land_fraction.values >= 0.5
    W, H = 3000e3, 2200e3
    E_s = raster(f["elev"])
    panels = []
    for cx, a_, b_, gmin in wins:
        lo_r, la_r = xyz_to_lonlat(cx)
        panels.append((E_s, E_s > 0, float(la_r), float(lo_r),
                       f"种子 {s0}，300 Ma：陆块 {a_} 与 {b_} 之间的裂谷陆缘（最近处 {gmin:.0f} km）"))
    panels.append((ee, el, -12.0, -15.0, "地球：南大西洋两岸（裂开约 1.3 亿年）"))
    fig = plt.figure(figsize=(7.5 * len(panels), 6))
    for k, (elev, land, lat0_, lon0_, title) in enumerate(panels):
        p_ = ccrs.AzimuthalEquidistant(central_longitude=lon0_, central_latitude=lat0_)
        ax = fig.add_subplot(1, len(panels), k + 1, projection=p_)
        ax.imshow(shaded_elevation(elev, 2 * np.pi * R_KM / elev.shape[1], land=land), transform=PC,
                  extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
        ax.set_extent((-W, W, -H, H), crs=p_)
        ax.gridlines(color="0.5", lw=0.3, alpha=0.6)
        ax.plot([-W + 2e5, -W + 1.2e6], [-H + 2e5, -H + 2e5], color="k", lw=3, transform=p_)
        ax.text(-W + 7e5, -H + 2.8e5, "1000 km", ha="center", fontsize=8, transform=p_)
        ax.set_title(title, fontsize=10)
    fig.suptitle("裂谷陆缘放大（同比例尺，1024x512）", fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "05_margin_zoom.png", dpi=105, bbox_inches="tight")
    plt.close(fig)

    # 图 6：300 Ma 陆块与编组（U20 的核心结果）：前史设计阶段判定仍相连的陆块（InitialWorld.links）编成一组一起运动，
    # 组数决定主大陆能由几块拼成
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig = plt.figure(figsize=(21, 5.2 * nrow + 0.8))
    for k, s in enumerate(seeds):
        js, fr = runs[s]
        f = fr[-1]
        lo_, _ = sc_center(fr[0])
        ax = fig.add_subplot(nrow, ncol, k + 1, projection=ccrs.Robinson(central_longitude=lo_))
        gid = np.full(mesh.n, -1)
        g_of_plate = {}
        for gi, g in enumerate(js["groups"]):
            for b_ in g["members"]:
                g_of_plate[b_] = gi
        cont = f["ctype"] == 1
        gp = np.array([g_of_plate.get(int(q), -2) for q in f["plate"]])
        gid[cont] = gp[cont]
        rgb = np.ones((mesh.n, 3)) * np.array([0.86, 0.91, 0.95])
        rgb[gid == -2] = (0.62, 0.62, 0.62)
        for gi in range(len(js["groups"])):
            rgb[gid == gi] = matplotlib.colors.to_rgb(pal[gi]) if gi < len(pal) else (0.24, 0.24, 0.24)
        img = rgb[raster.idx]
        ax.imshow(img, transform=PC, extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
        pa_, pb_ = f["plate"][e[:, 0]], f["plate"][e[:, 1]]
        inner = cont[e[:, 0]] & cont[e[:, 1]] & (pa_ != pb_) & (gid[e[:, 0]] >= 0) & (gid[e[:, 0]] == gid[e[:, 1]])
        lo_i, la_i = xyz_to_lonlat(unit(mesh.xyz[e[inner, 0]] + mesh.xyz[e[inner, 1]]))
        ax.scatter(lo_i, la_i, s=1.5, c="k", transform=PC, linewidths=0, rasterized=True)
        for gi, g in enumerate(js["groups"]):
            m_ = gid == gi
            if not m_.any():
                continue
            glo, gla = xyz_to_lonlat(unit((mesh.xyz[m_] * mesh.area_km2[m_, None]).sum(0)))
            ax.text(float(glo), float(gla), f"G{gi + 1}  {g['area_km2'] / 1e6:.0f}\n块 {','.join(map(str, g['members']))}",
                    transform=PC, fontsize=7, ha="center", va="center", color="k",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_global()
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)
        dr = js["events"][0]["drift"]
        ax.set_title(f"种子 {s}：陆块 {len(js['blocks'])} 块 -> 编组后 {len(js['groups'])} 组"
                     f"（{'可用' if js['usable'] else '不可用：' + js['unusable_why']}）\n"
                     f"刚体估计最近一对 {dr['min_gap_km_rigid']:.0f} km；各组面积单位 Mkm²", fontsize=8)
    fig.suptitle("300 Ma 陆块编组：设计阶段（L6 网格刚体估计）相距 < 250 km 的陆块在剧本里一起运动（颜色 = 组，第 9 组起为深灰、看标签；"
                 "黑点 = 组内陆块的分界；浅灰 = 太小不参与规划的陆壳）", fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "06_groups_300Ma.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 图 7：陆块大小分布（名次—面积），与盘古大陆裂解留下的碎块比较（U20 修订二：第一版陆块大小太平均、没有微大陆）
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, s in enumerate(seeds):
        js, _ = runs[s]
        a_ = sorted((b["area_km2"] / 1e6 for b in js["blocks"]), reverse=True)
        ax.plot(range(1, len(a_) + 1), a_, "o-", ms=3.5, lw=1, alpha=0.8, label=f"种子 {s}")
    efr = sorted(EARTH_FRAGMENTS.items(), key=lambda x: -x[1])
    ax.plot(range(1, len(efr) + 1), [v for _, v in efr], "s--", color="k", ms=5, lw=1.6, label="地球：盘古大陆的碎块（约值）")
    for r_, (nm, v) in enumerate(efr, start=1):
        ax.annotate(nm, (r_, v), textcoords="offset points", xytext=(5, 3), fontsize=7, color="0.25")
    ax.set_yscale("log")
    ax.set_xlabel("名次（按面积从大到小）")
    ax.set_ylabel("面积（Mkm²，对数）")
    ax.axhline(3.0, color="0.6", lw=0.8, ls=":")
    ax.text(0.6, 3.1, "3 Mkm²：小于此的陆块剧本规划器不单独规划", fontsize=7, color="0.4")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("300 Ma 各陆块面积的名次分布（含洋脊跃迁分出的微大陆）与地球比较", fontsize=10)
    fig.text(0.01, 0.005, "地球数值为今天的陆地面积约值（含附属岛屿；西兰大陆为大陆地壳面积，94% 在水下），只看量级与分布形状。\n"
             + NOTE, fontsize=6.5, color="0.3", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "07_block_sizes.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary = {str(s): {k: v for k, v in runs[s][0].items() if k in ("blocks", "land_fraction_300Ma", "cont_area_300Ma_km2",
                                                                      "n_units", "n_units_all_pass", "wall_s", "code_version",
                                                                      "groups", "usable", "unusable_why")}
               for s in seeds}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    if not a.figures_only:
        os.environ["MYTHIC_KD_WORKERS"] = str(max(1, (os.cpu_count() or 4) // a.workers))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(run_one, [a.level] * len(seeds), seeds):
                print(json.dumps(r, ensure_ascii=False), flush=True)
    figures(a.level, seeds)
    print("figures done", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
