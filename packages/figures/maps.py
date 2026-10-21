"""全球图与极地图（docs/review_protocol.md §2.1）：
- 全球图：等距圆柱投影（2:1），直接按格网像元显示，经纬网 30°；
- 比例尺：等距圆柱投影的南北向比例处处相同，所以画一根竖直的"南北向 1000 km"比例尺（全图有效）；
- 极地图：以极点为中心的局部等面积方位投影，覆盖到纬度 40°，按格网像元最近邻取样（如实显示格网）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from mythic.metrics.grid import LatLonGrid  # noqa: E402
from mythic.metrics.topology import component_sums, label  # noqa: E402

from .regional import draw_graticule, laea_window, sample_global  # noqa: E402
from .style import (  # noqa: E402
    ICE_SHELF_RGB,
    MONO_FAMILY,
    add_scalebar,
    elevation_cmap_norm,
    has_ice_shelf,
    shaded_elevation,
    stamp,
)

KM_PER_DEG = np.pi * 6371.0 / 180.0
GLOBAL_FIGSIZE = (16, 8.6)
OCEAN_BG = "#dfeaf2"


def _global_axes(fig, rect=(0.04, 0.12, 0.84, 0.8)):
    ax = fig.add_axes(rect)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xticklabels([f"{abs(v)}°{'E' if v > 0 else ('W' if v < 0 else '')}" for v in range(-180, 181, 30)], fontsize=8)
    ax.set_yticklabels([f"{abs(v)}°{'N' if v > 0 else ('S' if v < 0 else '')}" for v in range(-90, 91, 30)], fontsize=8)
    ax.grid(color="0.35", lw=0.4, alpha=0.5)
    return ax


def _meridional_scalebar(ax, length_km=1000.0):
    """放在南太平洋高纬（地球上是海面），文字加白底，任何世界上都看得清。"""
    dy = length_km / KM_PER_DEG
    x, y0 = -174.0, -60.0
    ax.plot([x, x], [y0, y0 + dy], color="k", lw=3, solid_capstyle="butt", zorder=5)
    ax.text(x + 2, y0 + dy / 2, f"南北向 {length_km:g} km\n（全图有效；东西向比例随纬度变化）", fontsize=7, va="center",
            zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0))


def _coastline(ax, grid: LatLonGrid, land: np.ndarray, color="k", lw=0.4):
    ax.contour(grid.lon, grid.lat, land.astype(float), levels=[0.5], colors=color, linewidths=lw)


def _imshow_global(ax, img, **kw):
    return ax.imshow(img, extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest", **kw)


def _save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


def global_elevation(world, path, vert_exag: float = 3.0):
    g = world.grid
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig)
    _imshow_global(ax, shaded_elevation(world.elevation, g.pixel_km_equator, vert_exag=vert_exag, land=world.land))
    _coastline(ax, g, world.land)
    _meridional_scalebar(ax)
    _ice_legend(ax, world.elevation, world.land)
    cmap, norm = elevation_cmap_norm()
    cax = fig.add_axes((0.9, 0.2, 0.015, 0.64))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("高程 / 水深（m）；海平面处色带断开")
    ax.set_title(f"{world.display_name} · 高程（分层设色 + 山体阴影，垂直夸张 {vert_exag:g}×）· 等距圆柱投影", fontsize=13)
    stamp(fig, world, f"{world.method_label}（高程）", [MASK_NOTE])
    _save(fig, path)


MASK_NOTE = "黑线：海岸线（陆地掩膜 0.5 等值线）。着色服从海陆掩膜：低于海平面的陆地（干洼地）按陆地色，海洋不显示正高程。"


def _ice_legend(ax, elev, land, loc="lower right"):
    """掩膜为海、高程为正的格点（漂浮冰架）存在时，加图例说明。"""
    if has_ice_shelf(elev, land):
        ax.legend(handles=[Patch(color=ICE_SHELF_RGB, label="冰架（漂浮冰：海陆掩膜按海计，高程为冰面）")],
                  loc=loc, fontsize=8, framealpha=0.85)


def polar_elevation(world, path, north: bool = True, edge_lat: float = 40.0):
    g = world.grid
    clat = 90.0 if north else -90.0
    size = 2 * (90.0 - edge_lat) * KM_PER_DEG
    lat, lon, ext = laea_window(clat, 0.0, size, g.pixel_km_equator)
    elev = sample_global(world.elevation, lat, lon, order=0)
    land = sample_global(world.land.astype("float32"), lat, lon, order=0)
    fig, ax = plt.subplots(figsize=(10, 10.6))
    fig.subplots_adjust(left=0.03, right=0.86, top=0.94, bottom=0.08)
    xs = np.linspace(ext[0], ext[1], land.shape[1])
    ys = np.linspace(ext[3], ext[2], land.shape[0])
    inside = np.hypot(*np.meshgrid(xs, ys)) <= size / 2  # 只显示到纬度 edge_lat 的圆盘
    rgba = np.dstack([shaded_elevation(elev, g.pixel_km_equator, vert_exag=3.0, land=land)[..., :3], inside.astype(float)])
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    ax.contour(xs, ys, np.where(inside, land, np.nan), levels=[0.5], colors="k", linewidths=0.5)
    draw_graticule(ax, (clat, 0.0), lat, lon, ext, step=10.0)
    disk = plt.Circle((0, 0), size / 2, transform=ax.transData)
    for artist in ax.lines:
        artist.set_clip_path(disk)
    for t in list(ax.texts):  # 圆盘外（低于 edge_lat）的纬度标注去掉
        if np.hypot(*t.get_position()) > size / 2:
            t.remove()
    ax.add_patch(plt.Circle((0, 0), size / 2, fill=False, lw=0.8, color="0.3"))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    add_scalebar(ax, ext, 1000.0)
    cmap, norm = elevation_cmap_norm()
    cax = fig.add_axes((0.88, 0.25, 0.02, 0.5))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("高程 / 水深（m）")
    hemi = "北" if north else "南"
    ax.set_title(f"{world.display_name} · {hemi}极区高程（{hemi}纬 {edge_lat:g}° 以{'北' if north else '南'}）· "
                 f"以极点为中心的等面积方位投影", fontsize=12)
    _ice_legend(ax, elev[inside], land[inside], loc="upper right")
    notes = ["极地图避免等距圆柱投影的高纬拉伸掩盖问题；按格网像元最近邻取样，经纬网 10°。", MASK_NOTE]
    stamp(fig, world, f"{world.method_label}（高程）", notes)
    _save(fig, path)


def landmass_units(world, metrics, fields, cfg, path):
    """陆块单元、亚大陆、主内海与候选海域、远洋空旷点（A 层的空间结构）。"""
    g = world.grid
    area = g.cell_area_km2
    units = fields["units"]
    n_units = int(units.max())
    u_area = component_sums(units, n_units, area)
    main_id = int(np.argmax(u_area[1:]) + 1) if n_units else 0
    lobes = fields["subcontinents"]
    n_lobes = int(lobes.max())
    l_area = component_sums(lobes, n_lobes, area)
    min_lobe = cfg["A_hard_constraints"]["subcontinents"]["min_area_km2"]
    cont_min = cfg["B1_coast"]["continent_min_area_km2"]

    # 类别：0 外洋；1 其他半封闭海域；2 主内海；3 小岛；4 其他大陆级陆块；5 主大陆上的小亚大陆片；6.. 各亚大陆
    cat = np.zeros(g.shape, np.int16)
    sea = fields.get("sea_regions")
    mis = metrics["A"]["main_inland_sea"]
    main_sea = mis["main"]["region_id"] if mis.get("exists") else None
    if sea is not None:
        cand_ids = [s["region_id"] for s in mis.get("candidate_seas", [])]
        cat[np.isin(sea, cand_ids) & ~world.land] = 1
        if main_sea is not None:
            cat[(sea == main_sea) & ~world.land] = 2
    cat[world.land] = 3
    cat[world.land & (u_area[units] >= cont_min)] = 4
    main = units == main_id
    cat[main] = 5
    big_lobes = [k for k in np.argsort(l_area[1:])[::-1] + 1 if l_area[k] >= min_lobe]
    lobe_colors = ["#e6a157", "#7fb069", "#d1748a", "#c9b458", "#8c7ab8", "#5fa8a3", "#b5835a", "#9fb2c4"]
    for n, k in enumerate(big_lobes[:len(lobe_colors)]):
        cat[main & (lobes == k)] = 6 + n
    colors = [OCEAN_BG, "#a9cbe3", "#3f7fbf", "#4d4d4d", "#bdb39a", "#e3dccb"] + lobe_colors[:max(1, len(big_lobes))]
    cmap = ListedColormap(colors)
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig, (0.04, 0.12, 0.7, 0.8))
    _imshow_global(ax, cat, cmap=cmap, norm=BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors)))
    _coastline(ax, g, world.land, lw=0.3)
    for n, k in enumerate(big_lobes[:len(lobe_colors)]):
        m = main & (lobes == k)
        v = (g.xyz * (area * m)[..., None]).sum(axis=(0, 1))
        v /= np.linalg.norm(v)
        la, lo = np.degrees(np.arcsin(v[2])), np.degrees(np.arctan2(v[1], v[0]))
        ax.text(lo, la, f"亚大陆 {n + 1}\n{l_area[k] / 1e6:.1f} 百万 km²", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
    oo = metrics["A"]["open_ocean"]
    ax.plot(oo["at_lat_lon"][1], oo["at_lat_lon"][0], marker="*", ms=14, color="#c0392b", mec="white")
    ax.annotate(f"远洋空旷点：离陆地 {oo['max_distance_to_land_km']:.0f} km", (oo["at_lat_lon"][1], oo["at_lat_lon"][0]),
                xytext=(8, 8), textcoords="offset points", fontsize=8, color="#c0392b")
    _meridional_scalebar(ax)
    handles = [Patch(color=colors[2], label="主内海"), Patch(color=colors[1], label="其他半封闭海域（候选）"),
               Patch(color=colors[5], label="主大陆中不足亚大陆面积的部分"),
               Patch(color=colors[4], label=f"其他大陆级陆块（≥ {cont_min / 1e6:g} 百万 km²）"),
               Patch(color=colors[3], label="岛屿")]
    handles += [Patch(color=lobe_colors[n], label=f"亚大陆 {n + 1}") for n in range(min(len(big_lobes), len(lobe_colors)))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.text(0.755, 0.12, a_layer_text(metrics), fontsize=8.5, va="bottom", family=MONO_FAMILY)
    ax.set_title(f"{world.display_name} · 陆块单元、亚大陆与主内海（A 层的空间结构）", fontsize=13)
    stamp(fig, world, f"由高程/海陆掩膜派生（输入 {world.method_label}）",
          [f"亚大陆：主大陆在宽度 < {2 * cfg['A_hard_constraints']['subcontinents']['neck_radius_km']:g} km 的颈部切开，"
           f"面积 ≥ {min_lobe / 1e6:g} 百万 km² 的才算；海峡：宽度 < {cfg['A_hard_constraints']['main_inland_sea']['strait_max_width_km']:g} km 的水道。"])
    _save(fig, path)


def _ok(x) -> str:
    return "达标" if x else "未达标"


def a_layer_text(metrics) -> str:
    A = metrics["A"]
    lines = ["A 层硬约束"]
    lines.append(f"海洋占比   {A['ocean_fraction']['value']:.3f}  {_ok(A['ocean_fraction']['pass'])}")
    mc = A["main_continent"]
    lines.append(f"主大陆占陆地 {mc['share_of_land']:.2f}  {_ok(mc['pass'])}")
    lines.append(f"亚大陆数   {A['subcontinents']['count']}  {_ok(A['subcontinents']['pass'])}")
    mis = A["main_inland_sea"]
    if mis.get("exists"):
        m = mis["main"]
        lines.append(f"主内海     {m['area_km2'] / 1e6:.2f} 百万 km²，海峡 {m['straits_to_world_ocean']} 条  {_ok(mis['pass'])}")
        geo = mis.get("geometry")
        if geo:
            lines.append(f"  接壤亚大陆 {geo['n_bordering_subcontinents']} 个，主轴长宽比 {geo['aspect_ratio']:.2f}")
    else:
        lines.append(f"主内海     无  {_ok(False)}")
    lat = A["main_continent_latitude"]
    lines.append(f"主大陆纬度 {lat['north_deg']:.0f}°N…{abs(lat['south_deg']):.0f}°{'S' if lat['south_deg'] < 0 else 'N'}  {_ok(lat['pass'])}")
    lines.append(f"极地大陆   {len(A['polar_continent']['found'])} 块  {_ok(A['polar_continent']['pass'])}")
    lines.append(f"远洋空旷   {A['open_ocean']['land_free_circle_diameter_km']:.0f} km  {_ok(A['open_ocean']['pass'])}")
    return "\n".join(lines)


def coast_morphology(world, metrics, fields, path, r_km: int = 100):
    """半岛与海湾：半径 r 的开运算削掉的陆地（半岛、细地峡、小岛）与闭运算填充的海（海湾、窄海峡）。"""
    g = world.grid
    cat = np.zeros(g.shape, np.int8)
    cat[world.land] = 1
    cat[fields[f"peninsula_removed_{r_km}"]] = 2
    cat[fields[f"bay_filled_{r_km}"]] = 3
    colors = [OCEAN_BG, "#cfcac0", "#e66101", "#5e3c99"]
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig, (0.04, 0.12, 0.7, 0.8))
    _imshow_global(ax, cat, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(5) - 0.5, 4))
    _coastline(ax, g, world.land, lw=0.3)
    _meridional_scalebar(ax)
    ax.legend(handles=[Patch(color=colors[2], label=f"半岛 / 细地峡 / 小岛（半径 {r_km} km 开运算削掉的陆地）"),
                       Patch(color=colors[3], label=f"海湾 / 窄海峡（半径 {r_km} km 闭运算填充的海）")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    B1 = metrics["B1"]
    isl = B1["islands"]
    txt = ["B1 海岸复杂度（全球）",
           f"海岸线长度   {B1['coastline_length_km'] / 1e3:.0f} 千 km",
           f"分形维数 D   {B1['fractal_dimension']['D']:.3f}（R² {B1['fractal_dimension']['r2']:.3f}）",
           "半岛指数     " + " / ".join(f"{B1['peninsula_index'][k]:.3f}" for k in sorted(B1['peninsula_index'], key=float)),
           "海湾指数     " + " / ".join(f"{B1['bay_index'][k]:.3f}" for k in sorted(B1['bay_index'], key=float)),
           "  （半径 " + " / ".join(sorted(B1['peninsula_index'], key=float)) + " km）",
           f"岛屿数       {isl['n_islands']}",
           f"Korčak b     {isl['korcak']['b']:.2f}",
           f"大陆架/海洋  {B1['shelf_fraction_of_ocean']:.3f}"]
    fig.text(0.755, 0.12, "\n".join(txt), fontsize=8.5, va="bottom", family=MONO_FAMILY)
    ax.set_title(f"{world.display_name} · 半岛与海湾（半径 {r_km} km 的球面形态学）", fontsize=13)
    stamp(fig, world, f"由海陆掩膜派生（输入 {world.method_label}）",
          ["半岛指数 = 削掉的陆地面积 ÷ 陆地面积；海湾指数 = 填充的海域面积 ÷ 陆地面积。定义见 docs/metrics_definitions.md。"])
    _save(fig, path)


def distance_to_sea(world, metrics, fields, path):
    g = world.grid
    d = np.where(world.land, fields["distance_to_sea_km"], np.nan)
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig)
    ax.set_facecolor(OCEAN_BG)
    vmax = max(2500.0, float(np.nanmax(d)))
    im = _imshow_global(ax, d, cmap="viridis", vmin=0, vmax=vmax)
    _coastline(ax, g, world.land, lw=0.3)
    B2 = metrics["B2"]
    la, lo = B2["max_at_lat_lon"]
    ax.plot(lo, la, marker="*", ms=14, color="#c0392b", mec="white")
    ax.annotate(f"离海最远 {B2['max_km']:.0f} km", (lo, la), xytext=(8, 8), textcoords="offset points", fontsize=9,
                color="#c0392b", path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    _meridional_scalebar(ax)
    cb = fig.colorbar(im, cax=fig.add_axes((0.9, 0.2, 0.015, 0.64)))
    cb.set_label("到最近海洋格点的大圆距离（km）")
    ax.set_title(f"{world.display_name} · 距海距离（B2）  中位数 {B2['median_km']:.0f} km · 90% 分位 {B2['p90_km']:.0f} km · "
                 f"离海 > 1000 km 的陆地 {B2['land_fraction_beyond_1000km']:.1%}", fontsize=12)
    stamp(fig, world, f"由海陆掩膜派生（输入 {world.method_label}）", ["距离为格点中心之间的大圆距离；面积加权统计。"])
    _save(fig, path)


def slope(world, fields, metrics, path):
    g = world.grid
    s = np.where(world.land, np.clip(fields["slope_deg"], 0.01, 30), np.nan)  # 只画陆地：C 层坡度统计只算陆地
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig)
    ax.set_facecolor(OCEAN_BG)
    im = _imshow_global(ax, s, cmap="cividis", norm=LogNorm(vmin=0.02, vmax=5))
    _coastline(ax, g, world.land, lw=0.3)
    _meridional_scalebar(ax)
    cb = fig.colorbar(im, cax=fig.add_axes((0.9, 0.2, 0.015, 0.64)))
    cb.set_label("坡度（度，对数色标）")
    q = metrics["C"]["slope"]["quantiles_land_deg"]
    ax.set_title(f"{world.display_name} · 坡度（格网尺度 {g.pixel_km_equator:.0f} km）  陆地坡度中位数 {q['0.5']:.2f}° · "
                 f"90% 分位 {q['0.9']:.2f}°", fontsize=12)
    stamp(fig, world, f"由高程派生（输入 {world.method_label}）",
          ["坡度随格网分辨率变化很大，只能与同分辨率的地球比较（见分布对比图）。"])
    _save(fig, path)


def mountain_systems(world, metrics, fields, path, n_label: int = 12):
    g = world.grid
    merged = fields["mountain_systems"]
    lab, n = label(merged)
    areas = component_sums(lab, n, g.cell_area_km2)
    order = [k for k in np.argsort(areas[1:])[::-1] + 1]
    systems = metrics["C"]["mountains"]["systems"]
    fig = plt.figure(figsize=GLOBAL_FIGSIZE)
    ax = _global_axes(fig, (0.04, 0.12, 0.7, 0.8))
    base = np.where(world.land, np.clip(world.elevation, 0, 6000), np.nan)
    ax.set_facecolor(OCEAN_BG)
    _imshow_global(ax, base, cmap="Greys", vmin=-1500, vmax=7000)
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    rgba = np.zeros(g.shape + (4,))
    for rank, k in enumerate(order):
        rgba[lab == k] = palette[rank % 20] if rank < n_label else (0.55, 0.2, 0.2, 0.8)
    _imshow_global(ax, rgba)
    _coastline(ax, g, world.land, lw=0.3)
    for rank, k in enumerate(order[:min(n_label, len(systems))]):
        m = lab == k
        v = (g.xyz * (g.cell_area_km2 * m)[..., None]).sum(axis=(0, 1))
        v /= np.linalg.norm(v)
        la, lo = np.degrees(np.arcsin(v[2])), np.degrees(np.arctan2(v[1], v[0]))
        ax.text(lo, la, str(rank + 1), fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.8, lw=0))
    _meridional_scalebar(ax)
    mt = metrics["C"]["mountains"]
    ax.legend(handles=[Patch(color=(0.55, 0.2, 0.2, 0.8), label=f"编号 {n_label} 以外的山系，以及面积不足山系下限的零散山地")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    txt = ["C 层山系（v1 定义 + U16 形态量）", f"山系数       {mt['n_systems']}", f"长 > 1000 km {mt['n_longer_than_min_length']}",
           f"山地占陆地   {mt['mountain_land_fraction']:.3f}", f"最长山系     {mt['longest_km']:.0f} km",
           f"加权长宽比   {mt['area_weighted_aspect_ratio']:.2f}", f"成带面积占比 {mt['elongated_area_fraction']:.2f}",
           "", "编号  长度km 宽km 长宽比 走向°"]
    txt += [f"{i + 1:>3} {s['length_km']:>7.0f} {s['mean_width_km']:>5.0f} {s['aspect_ratio']:>5.2f} {s['orientation_deg']:>5.0f}"
            for i, s in enumerate(systems[:n_label])]
    txt += ["", "长宽比 = 主轴长宽比（圆斑≈1）；", "走向 0° = 南北，90° = 东西"]
    fig.text(0.755, 0.12, "\n".join(txt), fontsize=8.5, va="bottom", family=MONO_FAMILY)
    ax.set_title(f"{world.display_name} · 山系（按面积编号；底图为陆地高程灰度）", fontsize=13)
    stamp(fig, world, f"由高程派生（输入 {world.method_label}）",
          ["山地 = 高程 ≥ 1000 m 且 ±100 km 内起伏 ≥ 1000 m；间隔 < 100 km 的连成山系；长度为骨架最长路径。"
           + ("本世界带冰盖掩膜（高程为冰面高程），冰盖区不计入山系（对所有带冰盖掩膜的世界同样适用）。"
              if world.mountain_exclude is not None else "")])
    _save(fig, path)
