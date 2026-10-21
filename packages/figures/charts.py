"""与地球同分辨率对比的统计图（C 层分布、B1 海岸、B2 距海距离）。

地球一侧用同一套指标代码在同一分辨率的地球格网上算出（scripts/make_figures.py 中现算，与基准 JSON 一致）。
世界本身就是地球时，两条曲线重合，这是预期结果（用来检查管线本身）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from mythic.metrics.topology import component_sums, label  # noqa: E402

from .style import stamp  # noqa: E402

WORLD_C = "#c0392b"
EARTH_C = "#2c3e50"


def _save(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _labels(world):
    return (f"本世界：{world.display_name}（{world.resolution}）", f"地球基准（{world.resolution}）")


def _wcdf(values, weights):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return v, c / c[-1]


def hypsometry(world, metrics, earth_metrics, path):
    wl, el = _labels(world)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    for m, lab, c, ls in ((metrics, wl, WORLD_C, "-"), (earth_metrics, el, EARTH_C, "--")):
        h = m["C"]["hypsometry"]["histogram_global"]
        edges = np.asarray(h["edges"])
        frac = np.asarray(h["fraction"])
        above = 1.0 - np.concatenate([[0.0], np.cumsum(frac)])  # 高于各边界高程的面积比例
        axes[0].plot(above * 100, edges, color=c, ls=ls, lw=1.8, label=lab)
        centers = (edges[:-1] + edges[1:]) / 2
        axes[1].plot(frac * 100, centers, color=c, ls=ls, lw=1.5, label=lab)
    axes[0].set_xlabel("高于该高程的星球表面比例（%）")
    axes[0].set_ylabel("高程（m）")
    axes[0].set_title("高程曲线（hypsometric curve）")
    axes[1].set_xlabel("每 100 m 高程区间的表面积比例（%）")
    axes[1].set_title("高程分布：地球为“大陆台地 + 深海平原”双峰")
    for ax in axes:
        ax.axhline(0, color="0.5", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(-11000, 9000)
    fig.suptitle(f"{world.display_name} · 高程分布与地球对比（C 层）", fontsize=13)
    stamp(fig, world, f"由高程统计（输入 {world.method_label}）", ["面积加权；地球数据为 ETOPO 2022 重采样到同一分辨率。"])
    _save(fig, path)


def slope_distance_mountains(world, metrics, fields, earth_metrics, earth_fields, earth_world, path):
    wl, el = _labels(world)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(bottom=0.2, wspace=0.28)
    for w, f, m, lab, c, ls in ((world, fields, metrics, wl, WORLD_C, "-"),
                                (earth_world, earth_fields, earth_metrics, el, EARTH_C, "--")):
        a = w.grid.cell_area_km2
        s, cdf = _wcdf(f["slope_deg"][w.land], a[w.land])
        axes[0].plot(np.maximum(s, 1e-3), cdf, color=c, ls=ls, lw=1.8, label=lab)
        d, cdf = _wcdf(f["distance_to_sea_km"][w.land], a[w.land])
        axes[1].plot(d, cdf, color=c, ls=ls, lw=1.8, label=f"{lab}：中位数 {m['B2']['median_km']:.0f} km")
        L = sorted((s_["length_km"] for s_ in m["C"]["mountains"]["systems"]), reverse=True)
        axes[2].plot(np.arange(1, len(L) + 1), L, marker="o", ms=4, color=c, ls=ls, lw=1.5,
                     label=f"{lab}：{m['C']['mountains']['n_longer_than_min_length']} 条 > 1000 km")
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-2, 30)
    axes[0].set_xlabel("陆地坡度（度，对数轴）")
    axes[0].set_ylabel("累积面积比例")
    axes[0].set_title("陆地坡度分布")
    axes[1].set_xlabel("距海距离（km）")
    axes[1].set_title("陆地距海距离分布（B2）")
    axes[2].set_xlabel("山系按长度排名")
    axes[2].set_ylabel("骨架长度（km）")
    axes[2].set_title("山系长度（v1 定义，最多 20 个）")
    axes[2].axhline(1000, color="0.5", lw=0.8, ls=":")
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"{world.display_name} · 坡度、距海距离与山系长度：与同分辨率地球对比", fontsize=13)
    stamp(fig, world, f"由高程派生（输入 {world.method_label}）", ["面积加权累积分布；地球山系统计剔除了南极与格陵兰冰盖。"])
    _save(fig, path)


def _island_areas(w, cont_min):
    lab, n = label(w.land)
    a = component_sums(lab, n, w.grid.cell_area_km2)[1:]
    return np.sort(a[a < cont_min])[::-1]


def coast_vs_earth(world, metrics, earth_metrics, earth_world, cfg, verdict, reference, path):
    """海岸：盒计数（分形维数）、岛屿面积分布（Korčak），以及各大陆 B1/B2 指标与地球各大洲范围。"""
    wl, el = _labels(world)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.45, wspace=0.6, bottom=0.15, top=0.92)
    ax_box = fig.add_subplot(gs[0, 0:3])
    ax_isl = fig.add_subplot(gs[0, 3:6])
    for m, w, lab, c, ls in ((metrics, world, wl, WORLD_C, "-"), (earth_metrics, earth_world, el, EARTH_C, "--")):
        fd = m["B1"]["fractal_dimension"]
        ax_box.plot(fd["sizes_km"], fd["counts"], marker="o", color=c, ls=ls, label=f"{lab}：D = {fd['D']:.3f}")
        ia = _island_areas(w, cfg["B1_coast"]["continent_min_area_km2"])
        ax_isl.plot(ia, np.arange(1, len(ia) + 1), color=c, ls=ls, lw=1.6,
                    label=f"{lab}：{len(ia)} 个，Korčak b = {m['B1']['islands']['korcak']['b']:.2f}")
    ax_box.set_xscale("log")
    ax_box.set_yscale("log")
    ax_box.set_xlabel("盒子边长（km）")
    ax_box.set_ylabel("被海岸线占据的盒子数")
    ax_box.set_title("海岸线盒计数（斜率 = 分形维数 D）")
    ax_isl.set_xscale("log")
    ax_isl.set_yscale("log")
    ax_isl.set_xlabel("岛屿面积 A（km²）")
    ax_isl.set_ylabel("面积 ≥ A 的岛屿数")
    ax_isl.set_title("岛屿面积累积分布（Korčak 律）")
    ax_isl.axvline(metrics["B1"]["islands"]["resolution_min_area_km2"], color="0.5", ls=":", lw=1)
    for ax in (ax_box, ax_isl):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    # 各大陆级陆块的 B1/B2 与地球大陆级陆块本体（U11：同一切分方式）
    titles = {"fractal_dimension": "分形维数 D", "coastline_development": "海岸发育系数", "peninsula_index": "半岛指数",
              "bay_index": "海湾指数", "land_fraction_beyond_1000km": "离海 > 1000 km 陆地比例",
              "distance_to_sea_median_km": "距海距离中位数（km）"}
    items = list(reference["pass_line"])
    wunits = list(verdict["units"])
    sub = gs[1, :].subgridspec(1, len(items), wspace=0.9)
    for n, k in enumerate(items):
        ax = fig.add_subplot(sub[0, n])
        line = reference["pass_line"][k]
        e_vals = [u[k] for u in reference["units"].values()]
        ax.scatter(np.zeros(len(e_vals)), e_vals, s=16, color=EARTH_C, zorder=2)
        ref_only = bool(wunits) and verdict["units"][wunits[0]]["metrics"][k]["pass"] is None
        ax.axhline(line["value"], color="0.45" if ref_only else "#2e7d32", ls="--", lw=1.2, zorder=1)
        for j, u in enumerate(wunits):
            r = verdict["units"][u]["metrics"][k]
            ax.scatter([1 + 0.15 * j], [r["value"]], s=34, marker="D", zorder=3,
                       color=WORLD_C if j == 0 else "#e67e22", edgecolors="k" if r["pass"] else "none")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["地球", "本世界"], fontsize=8)
        ax.set_xlim(-0.6, 1.9)
        base, _, rad = k.partition("@")
        ax.set_title(titles[base] + (f"\n半径 {rad} km" if rad else "") + ("\n（只作参考）" if ref_only else ""), fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    names = "、".join(v["name"] for v in reference["units"].values())

    def label(u):
        n = verdict["units"][u].get("name", u)
        if n != u:
            return n
        return "主大陆" if u == "main_continent" else f"陆块 {u.removeprefix('unit_')}"

    wn = [label(u) for u in wunits]
    fig.text(0.01, 0.06, f"下排（U11 同类比较）：深色点 = 地球大陆级陆块本体（{names}），与世界同一切分方式；"
             "绿色虚线 = 合格线（不低于其中最小值，距海距离类不高于其中最大值）；灰色虚线 = 只作参考的指标。\n"
             f"红色菱形 = {wn[0] if wn else '—'}" + (f"，橙色菱形 = 其余大陆级陆块（{'、'.join(wn[1:])}）" if len(wn) > 1 else "")
             + "；菱形带黑边 = 达标。", fontsize=8.5)
    fig.suptitle(f"{world.display_name} · 海岸复杂度与距海距离：与同分辨率地球对比（B1、B2）", fontsize=13)
    stamp(fig, world, f"由海陆掩膜派生（输入 {world.method_label}）")
    _save(fig, path)
