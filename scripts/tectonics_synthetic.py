"""阶段 1.2a 检查点：两个合成试验（裂谷 -> 洋盆扩张、碰撞 -> 造山）在 L7 上的图，以及引擎耗时。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\tectonics_synthetic.py

输出 figures/stage1/synthetic/（不进 Git），数值摘要 figures/stage1/synthetic/summary.json。约 1 分钟。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mythic.figures.style import setup_fonts, shaded_elevation  # noqa: E402
from mythic.figures.world import code_version  # noqa: E402
from mythic.tectonics import synthetic  # noqa: E402
from mythic.tectonics.crust import gdh1_depth_m  # noqa: E402
from mythic.tectonics.mesh import CACHE, Raster, load_mesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "stage1" / "synthetic"
PC = ccrs.PlateCarree()
NOTE = ("method_class：洋深 GDH1、Airy 均衡 = parameterized_physics；俯冲增厚、碰撞拼贴、厚度上限流散、裂谷阶梯 = procedural_constrained；"
        "运动 = 剧本（旋转文件 -> pyGPlates）")


def stamp(fig, level):
    fig.text(0.005, 0.003, f"{NOTE}\n阶段 1.2a 合成试验 · 网格 L{level} · 代码 {code_version()} · scripts/tectonics_synthetic.py",
             fontsize=6.5, color="0.3", va="bottom")


def panel(ax, raster, s, ext, title):
    elev = raster(s["elev"])
    ax.imshow(shaded_elevation(elev, raster.grid.pixel_km_equator, land=elev > 0), transform=PC,
              extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
    ax.set_extent(ext, crs=PC)
    ax.gridlines(color="0.5", lw=0.3, alpha=0.6, draw_labels=False)
    ax.set_title(title, fontsize=9)


def equator_profile(mesh, s, field, lon_range):
    sel = (np.abs(mesh.lat) < 1.0) & (mesh.lon > lon_range[0]) & (mesh.lon < lon_range[1])
    o = np.argsort(mesh.lon[sel])
    return mesh.lon[sel][o], s[field][sel][o]


def fig_rift(mesh, raster, r, level):
    fig = plt.figure(figsize=(17, 8.6))
    fig.subplots_adjust(wspace=0.7, hspace=0.3, top=0.9)
    ext = (-50, 70, -45, 45)
    for k, t in enumerate((100, 90, 70, 50)):
        s = min(r.snapshots, key=lambda x: abs(x["t"] - t))
        ax = fig.add_subplot(2, 4, k + 1, projection=PC)
        panel(ax, raster, s, ext, f"{s['t']:.0f} Ma（裂解后 {100 - s['t']:.0f} Myr）")
    s = r.snapshots[-1]
    ax = fig.add_subplot(2, 4, 5, projection=PC)
    age = np.where(s["ctype"] == 0, s["age"], np.nan)
    im = ax.imshow(raster(age), transform=PC, extent=(-180, 180, -90, 90), origin="upper", cmap="viridis_r", vmin=0,
                   vmax=80, interpolation="nearest")
    ax.set_extent(ext, crs=PC)
    ax.set_title("50 Ma 洋壳年龄（Myr；陆壳留白）", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax = fig.add_subplot(2, 4, (6, 7))
    lon, age = equator_profile(mesh, s, "age", (-20, 45))
    _, elev = equator_profile(mesh, s, "elev", (-20, 45))
    _, ctype = equator_profile(mesh, s, "ctype", (-20, 45))
    oc = (ctype == 0) & (age < 60)
    ax.plot(lon[oc], age[oc], "o", ms=3, color="tab:green", label="洋壳年龄（模拟）")
    lo, hi = lon[oc].min(), lon[oc].max()
    mid = 0.5 * (lo + hi)
    xx = np.linspace(lo, hi, 50)
    ax.plot(xx, 50 * np.abs(xx - mid) / (0.5 * (hi - lo)), "k--", lw=1, label="对称扩张预期（陆缘 50 Myr、洋脊 0）")
    ax.set_xlabel("经度（°，赤道剖面）")
    ax.set_ylabel("洋壳年龄（Myr）")
    ax2 = ax.twinx()
    ax2.plot(lon, elev, "-", color="tab:blue", lw=1.2, label="高程（模拟）")
    ax2.plot(xx, -gdh1_depth_m(50 * np.abs(xx - mid) / (0.5 * (hi - lo))), ":", color="tab:red", lw=1.2,
             label="按预期年龄的 GDH1 水深")
    ax2.set_ylabel("高程（m）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
    ax.set_title(f"赤道剖面：新洋盆宽 {hi - lo:.1f}°（预期 0.4°/Myr × 50 Myr = 20°）", fontsize=9)
    ax = fig.add_subplot(2, 4, 8)
    _, T = equator_profile(mesh, s, "T", (-20, 45))
    ax.plot(lon, T, "-o", ms=2.5, color="0.3")
    ax.axhline(35, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("经度（°，赤道剖面）")
    ax.set_ylabel("地壳厚度（km）")
    ax.set_title("裂谷陆缘减薄（陆架、陆坡）", fontsize=9)
    fig.suptitle("合成试验 1：裂谷 -> 洋盆扩张（大陆在 100 Ma 沿 0° 经线裂开，东半块绕北极以 0.4°/Myr 东移，西半块不动）", fontsize=11)
    stamp(fig, level)
    fig.savefig(OUT / "01_rift_spreading.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def fig_collision(mesh, raster, r, level):
    fig = plt.figure(figsize=(16, 11.5))
    fig.subplots_adjust(wspace=0.3, hspace=0.35, top=0.93)
    ext = (-55, 55, -40, 40)
    ev = {e["kind"]: e["t"] for e in r.engine.events}
    for k, t in enumerate((100, 90, 80, 70, 60, 50)):
        s = min(r.snapshots, key=lambda x: abs(x["t"] - t))
        ax = fig.add_subplot(3, 4, k + 1, projection=PC)
        tag = " · 碰撞开始" if abs(s["t"] - ev.get("collision_onset", -99)) < 5 else ""
        panel(ax, raster, s, ext, f"{s['t']:.0f} Ma{tag}")
    s = r.snapshots[-1]
    ax = fig.add_subplot(3, 4, (9, 10), projection=PC)
    im = ax.imshow(raster(np.where(s["ctype"] == 1, s["T"], np.nan)), transform=PC, extent=(-180, 180, -90, 90),
                   origin="upper", cmap="magma_r", vmin=20, vmax=66, interpolation="nearest")
    ax.set_extent(ext, crs=PC)
    ax.set_title("50 Ma 陆壳厚度（km）：碰撞高原在上盘一侧", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax = fig.add_subplot(3, 4, (7, 8))
    for s in r.snapshots:
        lon, e = equator_profile(mesh, s, "elev", (-50, 50))
        ax.plot(lon, e, lw=1.0, label=f"{s['t']:.0f} Ma", color=plt.cm.viridis((100 - s["t"]) / 50))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("经度（°，赤道剖面）")
    ax.set_ylabel("高程（m）")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("赤道剖面高程随时间（海沟、安第斯型山脉 -> 碰撞高原）", fontsize=9)
    ax = fig.add_subplot(3, 4, (11, 12))
    t = [s["t"] for s in r.snapshots]
    ax.plot(t, [s["T"].max() for s in r.snapshots], "-o", ms=3, label="最大地壳厚度（km）")
    ax.axhline(65, color="tab:red", lw=0.8, ls="--", label="厚度上限 65 km")
    ax.set_xlim(100, 50)
    ax.set_xlabel("时间（Ma）")
    ax.set_ylabel("km")
    ax2 = ax.twinx()
    ax2.plot(t, [s["elev"].max() / 1000 for s in r.snapshots], "-s", ms=3, color="tab:brown", label="最高高程（km）")
    ax2.set_ylabel("km")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
    cl0, cl = r.engine.commit_log[0], r.engine.commit_log[-1]
    b = cl["budget"]
    pred = cl0["cont_volume_km3"] + b["arc_added"] + b["arc_converted"] - b["eroded"] - b["rift_thinning"]
    ax.set_title(f"碰撞 {ev.get('collision_onset', float('nan')):.0f} Ma（预期 80 Ma）；陆壳体积收支 实际/预期 = "
                 f"{cl['cont_volume_km3'] / pred:.6f}", fontsize=9)
    fig.suptitle("合成试验 2：碰撞 -> 造山（大陆 B 所在板块以 0.5°/Myr 西移，中间洋壳在大陆 A 东缘俯冲；75 Ma 起汇聚减到 20%）",
                 fontsize=11)
    stamp(fig, level)
    fig.savefig(OUT / "02_collision_orogen.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    level = 7
    mesh = load_mesh(level)
    raster = Raster(mesh, 512)
    work = CACHE / "synthetic"
    summary = {"level": level, "code_version": code_version()}
    t0 = time.perf_counter()
    r = synthetic.rift_spreading(mesh, work / "rift.rot")
    summary["rift"] = {"wall_s": time.perf_counter() - t0,
                       "step_wall_s_mean": float(np.mean([s["wall_s"] for s in r.engine.step_log])),
                       "events": r.engine.events}
    fig_rift(mesh, raster, r, level)
    t0 = time.perf_counter()
    c = synthetic.collision_orogen(mesh, work / "coll.rot")
    summary["collision"] = {"wall_s": time.perf_counter() - t0,
                            "step_wall_s_mean": float(np.mean([s["wall_s"] for s in c.engine.step_log])),
                            "events": c.engine.events,
                            "max_T_km": float(c.snapshots[-1]["T"].max()),
                            "max_elev_m": float(c.snapshots[-1]["elev"].max())}
    fig_collision(mesh, raster, c, level)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v if k in ("level", "code_version") else {kk: vv for kk, vv in v.items() if kk != "events"})
                      for k, v in summary.items()}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
