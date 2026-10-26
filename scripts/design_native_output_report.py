"""C6 候选出图：从保存的世界读取，保持地形、回退和视图填洞可区分。"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_west_arc_report as base
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import shaded_elevation

OUT = ROOT / "figures/design/v2/forward/u35_c6"


def save(fig, name, note):
    fig.text(.012, .015, "U35-C6 独立输出候选 · " + fwd.code_version() +
        "\n原生地壳 procedural_constrained；规定历史 user_override；高程 parameterized_physics；输出 interpolated\n" + note,
        fontsize=8, va="bottom")
    fig.savefig(OUT / name, dpi=125)
    plt.close(fig)


def load(path):
    with open_netcdf(path) as ds:
        return {k: ds[k].values.copy() for k in ds.data_vars}


def pixel_extent(grid, sx, sy):
    """imshow 要求像元外边缘，不能把首末像元中心当作边缘。"""
    dx = 180. / grid.nx
    dy = 90. / grid.ny
    return (grid.lon[sx][0] - dx, grid.lon[sx][-1] + dx,
            grid.lat[sy][-1] - dy, grid.lat[sy][0] + dy)


def map_panel(ax, E, land, title, extent, contour):
    g = base.GRID
    sx = (g.lon >= extent[0]) & (g.lon <= extent[1])
    sy = (g.lat >= extent[2]) & (g.lat <= extent[3])
    rgb = shaded_elevation(E, g.pixel_km_equator, vert_exag=3., land=land)[..., :3]
    ax.imshow(rgb[np.ix_(sy, sx)], extent=pixel_extent(g, sx, sy), origin="upper", interpolation="nearest", aspect="equal")
    ax.contour(g.lon[sx], g.lat[sy], contour[np.ix_(sy, sx)], levels=[.5], colors="#b62471", linewidths=.9)
    ax.set(xlim=extent[:2], ylim=extent[2:], title=title, xlabel="经度 °E", ylabel="纬度 °N")
    ax.grid(alpha=.25)
    bar_dx = 500. / (111.2 * np.cos(np.radians(extent[2] + 1)))
    ax.plot([extent[0] + 1, extent[0] + 1 + bar_dx], [extent[2] + .65] * 2, c="#222", lw=2)
    ax.text(extent[0] + 1, extent[2] + .9, "500 km", fontsize=8)


def render(result):
    base.setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    design = fwd.yaml.safe_load((ROOT / "config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    built = fwd.build(design, base.GRID)
    cases = {"C4 固定节点输出": load(ROOT / ".cache/u35_c5/reproduced_c4/latest/world.nc"),
             "C6 直接取样＋沿用填洞": load(ROOT / ".cache/u35_c6/worlds/native_fill_1024x512.nc"),
             "C6 直接取样、不沿用填洞": load(ROOT / ".cache/u35_c6/worlds/native_raw_1024x512.nc")}
    extent = (-20, -10, 34, 40.5)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (name, c) in zip(axes, cases.items()):
        map_panel(ax, c["elevation"], c["land"] > .5, name, extent, built["land"])
        base.feature_lines(ax, design)
        ax.contour(base.GRID.lon, base.GRID.lat, c["elevation"], levels=[1000, 1500, 2000], colors="#6e3f31", linewidths=.6)
    fig.subplots_adjust(bottom=.24, top=.88, wspace=.17)
    base.elevation_legend(fig, .14)
    save(fig, "39_native_output_comparison.png", "三图均1024×512、同一L7演化，含海沟与动力高程；紫线为认可海岸。右侧两图的无覆盖回退有单独标记。")
    p = result["ridge_profiles"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = {"control_1024x512": "#999", "native_fill_1024x512": "#1c9299", "native_fill_2048x1024": "#b56d2c", "native_material_m": "#754282"}
    labels = {"control_1024x512": "C4 1024×512", "native_fill_1024x512": "C6 1024×512", "native_fill_2048x1024": "C6 2048×1024", "native_material_m": "直接原生查询（同样含海沟/动力项）"}
    for key in colors:
        axes[0].plot(p["distance_km"], p[key], label=labels[key], color=colors[key], lw=1.5)
    axes[0].axhline(0, c="#222", lw=.8)
    axes[0].set(title="沿认可M31山脊：同一地壳、同一高程过程项", ylabel="高程 m")
    axes[0].legend(fontsize=9)
    for key, color in (("native_fill_1024x512", "#1c9299"), ("native_fill_2048x1024", "#b56d2c")):
        axes[1].plot(p["distance_km"], p[key], label=labels[key], c=color)
    axes[1].plot(p["distance_km"], p["native_material_m"], "--", c="#754282", label=labels["native_material_m"])
    axes[1].set(xlabel="沿认可山脊距离 km（西→东）", ylabel="高程 m", ylim=(1800, 2300), title="高地段纵轴放大：平顶仍在，完整水下段见上图")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(rect=(0, .14, 1, .96))
    save(fig, "40_native_ridge_profiles.png", "2048×1024仅改变输出取样间距，物理节点平均约60 km；不能当作新的20 km DEM或山峰细节。")
    native = cases["C6 直接取样＋沿用填洞"]
    raw = cases["C6 直接取样、不沿用填洞"]
    status = native["tect_sampling_status"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sx = (base.GRID.lon >= extent[0]) & (base.GRID.lon <= extent[1])
    sy = (base.GRID.lat >= extent[2]) & (base.GRID.lat <= extent[3])
    categories = np.where((status & 1) > 0, 1, 0)
    categories = np.where((status & 2) > 0, 2, categories)
    categories = np.where((status & 3) == 3, 3, categories)
    categories = np.where((status & 4) > 0, 4, categories)
    cmap = ListedColormap(["#dfe9df", "#777777", "#ae8bb5", "#bd875f", "#ddbe57"])
    image = axes[0].imshow(categories[np.ix_(sy, sx)], extent=pixel_extent(base.GRID, sx, sy),
        origin="upper", interpolation="nearest", cmap=cmap, norm=BoundaryNorm(np.arange(6) - .5, 5))
    axes[0].set(title="输出出处：无覆盖与填洞保持可追踪")
    cb = fig.colorbar(image, ax=axes[0], orientation="horizontal", ticks=range(5), fraction=.06, pad=.13)
    cb.ax.set_xticklabels(["原生", "固定回退", "视图填洞", "回退＋填洞", "火山岛"], fontsize=8)
    delta = native["elevation"] - raw["elevation"]
    im = axes[1].imshow(delta[np.ix_(sy, sx)], extent=pixel_extent(base.GRID, sx, sy),
        origin="upper", interpolation="nearest", cmap="RdBu_r", vmin=-3000, vmax=3000)
    axes[1].set(title="只切换旧视图填洞的高程差（m）")
    fig.colorbar(im, ax=axes[1], orientation="horizontal", fraction=.06, pad=.13, label="m，色带截于±3000")
    for ax in axes:
        ax.contour(base.GRID.lon, base.GRID.lat, native["land"], levels=[.5], colors="#222", linewidths=.7)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="经度 °E", ylabel="纬度 °N")
        ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.25, top=.88, wspace=.2)
    save(fig, "41_native_sampling_provenance.png", "回退仅沿用已有固定视图，属于插值表示，不是新增地壳。视图填洞只覆盖相同原始宿主，不覆盖另一个真实陆壳单元。")
    control = cases["C4 固定节点输出"]
    delta = native["elevation"] - control["elevation"]
    changed_land = (native["land"] > .5).astype(int) - (control["land"] > .5).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    im = axes[0].imshow(delta, extent=(-180, 180, -90, 90), origin="upper", cmap="RdBu_r", vmin=-1500, vmax=1500)
    axes[0].set(title="C6 − C4：全球高程差（同为1024×512）")
    fig.colorbar(im, ax=axes[0], shrink=.65, label="m，截于±1500")
    im = axes[1].imshow(changed_land, extent=(-180, 180, -90, 90), origin="upper", cmap="BrBG_r", vmin=-1, vmax=1)
    axes[1].set(title="全球海陆变化：−1陆→海，+1海→陆")
    fig.colorbar(im, ax=axes[1], ticks=[-1, 0, 1], shrink=.65)
    for ax in axes:
        ax.set(xlabel="经度 °E", ylabel="纬度 °N")
    fig.tight_layout(rect=(0, .13, 1, .96))
    save(fig, "42_native_global_change.png", "直接取样是全球输出规则变化，影响不限于西口；独立候选没有替换默认世界。海陆变化必须与连接和质量指标一起审查。")
    high = load(ROOT / ".cache/u35_c6/worlds/native_fill_2048x1024.nc")
    from mythic.metrics.grid import LatLonGrid
    grid = LatLonGrid(2048, 1024)
    sx = (grid.lon >= extent[0]) & (grid.lon <= extent[1])
    sy = (grid.lat >= extent[2]) & (grid.lat <= extent[3])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    map_panel(axes[0], native["elevation"], native["land"] > .5, "直接取样 1024×512", extent, built["land"])
    rgb = shaded_elevation(high["elevation"], grid.pixel_km_equator, vert_exag=3., land=high["land"] > .5)[..., :3]
    axes[1].imshow(rgb[np.ix_(sy, sx)], extent=pixel_extent(grid, sx, sy), origin="upper", interpolation="nearest", aspect="equal")
    for ax in axes:
        base.feature_lines(ax, design)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="经度 °E", ylabel="纬度 °N")
        ax.grid(alpha=.2)
    axes[1].set_title("直接取样 2048×1024；物理仍为L7")
    fig.subplots_adjust(bottom=.23, top=.87, wspace=.18)
    base.elevation_legend(fig, .13)
    save(fig, "43_native_output_spacing.png", "物理单元相同，只改变观察取样间距；不能把更细的格网边界当作真实的小海湾、岛形或更高分辨率DEM。")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, c) in zip(axes, cases.items()):
        map_panel(ax, c["elevation"], c["land"] > .5, name, (11, 21, 43, 49), built["land"])
        ax.scatter([15.996], [46.23], marker="o", s=75, facecolors="none", edgecolors="#8a253e", linewidths=1.6)
    fig.subplots_adjust(bottom=.24, top=.88, wspace=.17)
    base.elevation_legend(fig, .14)
    save(fig, "44_view_fill_interface.png", "圈出不沿用填洞时，A层额外识别的海域接口；封住西口后4/8水路均不通外洋，因此接口计数不能直接当作第二条真实海峡。")
    names = ["39_native_output_comparison.png", "40_native_ridge_profiles.png", "41_native_sampling_provenance.png", "42_native_global_change.png", "43_native_output_spacing.png", "44_view_fill_interface.png"]
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C6 原生输出候选</title><style>body{font-family:system-ui;margin:24px;background:#eee}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style><h1>U35-C6：原生输出候选</h1><p>同一C4演化，直接查询移动地壳单元；物理分辨率仍为L7。默认世界未更换。</p><table><tr><th>候选</th><th>A全过</th><th>水路4/8连通</th><th>末端水下km</th><th>回退面积%</th></tr>"
    for key, row in result["outputs"].items():
        html += f"<tr><td>{key}</td><td>{row['A']['all_pass']}</td><td>{row['water_connectivity']['4_connected']}/{row['water_connectivity']['8_connected']}</td><td>{row['ridge_output_tail_wet_km']:.1f}</td><td>{100*row['native_uncovered_fallback_fraction']:.3f}</td></tr>"
    html += "</table>"
    for name in names:
        html += f"<h2>{name}</h2><img src='{name}'>"
    html += "<p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c6_native/index.html'>候选标准图组</a> · <a href='../u35_c5/index.html'>C5核查</a></p>"
    (OUT / "index.html").write_text(html, encoding="utf-8", newline="\n")
