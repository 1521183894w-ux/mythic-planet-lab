"""同窗口构造叠合图（review_protocol §2.2b，评审 stage1_2026-10-25_3 第 14 条）：区域窗口与高程图同一投影、同一范围，
画地壳类型与厚度、洋底年龄、板块与边界类型、最近一次构造改造四个图层，供"海岸成因抽查"把地貌和构造对上。
只在世界文件带 tect_* 图层时生成（阶段 1 起的正向演化世界；地球与阶段 0 的合成世界没有）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .regional import draw_graticule, laea_window, sample_global  # noqa: E402
from .style import add_scalebar, stamp  # noqa: E402

BOUNDARY_COLORS = {1: ("#d62728", "汇聚"), 2: ("#1f77b4", "离散"), 3: ("#2ca02c", "走滑"), 4: ("#3b3b3b", "缓慢 < 0.5 cm/yr")}
EVENT_COLORS = {1: ("#2e86c1", "俯冲弧"), 2: ("#e67e22", "碰撞"), 3: ("#27ae60", "裂谷"), 4: ("#8e44ad", "板内挤压复活（设计给出）")}
OCEAN_RGB = np.array(to_rgb("#dfe8f1"))
CONT_RGB = np.array(to_rgb("#eeeeee"))


def plate_names(world) -> dict:
    """tect_plate 的属性 plate_codes（"代号=编号,..."）-> {编号: 代号}。"""
    s = (world.tectonics or {}).get("tect_plate", (None, {}))[1].get("plate_codes", "")
    out = {}
    for item in str(s).split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[int(v)] = k.strip()
            except ValueError:
                pass
    return out


PLATE_PALETTE = ["#f4a6a6", "#a6c8f4", "#f4d58d", "#b5e3b0", "#d7b8f0", "#f7c59f", "#9fe0e0", "#e8b4d8",
                 "#c9d98a", "#b0b8f0", "#f0e0a0", "#a8e6c8"]   # 浅色，不含灰色（灰色留给"缓慢"边界与陆壳底色）


def plate_rgb(P: np.ndarray, all_ids) -> np.ndarray:
    """板块编号 -> 浅色：按编号在全世界板块里的名次取 12 色浅色表（编号相近的板块多半相邻，名次相邻就不同色）。"""
    order = {int(q): k for k, q in enumerate(sorted(int(x) for x in all_ids))}
    img = np.zeros(P.shape + (3,))
    for q in np.unique(P):
        img[P == q] = to_rgb(PLATE_PALETTE[order.get(int(q), 0) % len(PLATE_PALETTE)])
    return img


def _cbar(fig, ax, im, label, slot=0):
    """面板内右下角的小色标（slot 0、1 自下而上叠放）。"""
    cax = ax.inset_axes((0.56, 0.05 + 0.11 * slot, 0.4, 0.025))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=7, labelpad=1)
    cax.patch.set_alpha(0.0)


def label_points(P: np.ndarray, min_cells: int) -> list:
    """每块板块一个标注点：离板块边界最远的格点（跨日界线、环形的板块，质心可能落在别的板块上）。"""
    from scipy import ndimage

    out = []
    for q in np.unique(P):
        m = P == q
        if m.sum() < min_cells:
            continue
        d = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        out.append((q, np.unravel_index(int(np.argmax(d)), m.shape)))
    return out


def _sample(world, name, lat, lon):
    return sample_global(np.nan_to_num(world.tectonics[name][0].astype("float64"), nan=-9999.0), lat, lon, order=0)


def _finish(ax, center, lat, lon, ext, land, title):
    ax.contour(land.astype(float), levels=[0.5], colors="k", linewidths=0.6, extent=ext, origin="upper")
    draw_graticule(ax, center, lat, lon, ext, 5.0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    width = ext[1] - ext[0]
    add_scalebar(ax, ext, 1000.0 if width >= 3000 else (500.0 if width >= 1500 else 200.0))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def boundary_overlay(B: np.ndarray) -> np.ndarray:
    """边界类型 -> RGBA（无边界处透明）。"""
    img = np.zeros(B.shape + (4,))
    for k, (c, _) in BOUNDARY_COLORS.items():
        img[B == k] = (*to_rgb(c), 1.0)
    return img


def region_tectonics(pick, world, path, index: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 13.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.04, hspace=0.1)
    if pick.missing:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, f"未找到\n\n{pick.missing}", ha="center", va="center", fontsize=14,
                        transform=axes[0, 0].transAxes)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return
    px = world.grid.pixel_km_equator
    lat, lon, ext = laea_window(pick.center[0], pick.center[1], pick.size_km, px)
    land = sample_global(world.land.astype("float32"), lat, lon, order=0) >= 0.5
    ctype = _sample(world, "tect_crust_type", lat, lon)
    cont = ctype >= 0.5
    T = _sample(world, "tect_crust_km", lat, lon)
    age = _sample(world, "tect_ocean_age_myr", lat, lon)
    P = _sample(world, "tect_plate", lat, lon).astype(int)
    B = _sample(world, "tect_boundary", lat, lon).astype(int)
    ev = _sample(world, "tect_last_event", lat, lon).astype(int)
    evt = _sample(world, "tect_last_event_ma", lat, lon)
    names = plate_names(world)

    # (a) 地壳类型与厚度
    ax = axes[0, 0]
    im1 = ax.imshow(np.ma.masked_where(~cont, T), extent=ext, cmap="YlOrBr", vmin=20, vmax=65, interpolation="nearest")
    im2 = ax.imshow(np.ma.masked_where(cont, T), extent=ext, cmap="Blues", vmin=4, vmax=30, interpolation="nearest")
    _finish(ax, pick.center, lat, lon, ext, land, "(a) 地壳厚度：陆壳（黄褐，km）/ 洋壳（蓝，km）")
    _cbar(fig, ax, im1, "陆壳厚度 km", 1)
    _cbar(fig, ax, im2, "洋壳厚度 km", 0)

    # (b) 洋底年龄
    ax = axes[0, 1]
    base = np.where(cont[..., None], CONT_RGB, OCEAN_RGB)
    ax.imshow(base, extent=ext, interpolation="nearest")
    im = ax.imshow(np.ma.masked_where(cont | (age < 0), age), extent=ext, cmap="RdYlBu", vmin=0, vmax=180,
                   interpolation="nearest")
    _finish(ax, pick.center, lat, lon, ext, land, "(b) 洋底年龄（Myr；灰 = 陆壳）")
    _cbar(fig, ax, im, "洋底年龄 Myr", 0)

    # (c) 板块与边界类型
    ax = axes[1, 0]
    ax.imshow(plate_rgb(P, np.unique(world.tectonics["tect_plate"][0])), extent=ext, interpolation="nearest")
    ax.imshow(boundary_overlay(B), extent=ext, interpolation="nearest")
    n = P.shape[0]
    xs = np.linspace(ext[0], ext[1], n)
    ys = np.linspace(ext[3], ext[2], n)
    for q, (i0, j0) in label_points(P, max(12, n * n // 400)):
        ax.text(xs[j0], ys[i0], names.get(int(q), str(int(q))), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    _finish(ax, pick.center, lat, lon, ext, land, "(c) 板块与边界类型（相对运动取自倒推的旋转文件最后 1 Myr）")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in BOUNDARY_COLORS.values()], loc="upper right", fontsize=7,
              framealpha=0.85)

    # (d) 最近一次构造改造
    ax = axes[1, 1]
    img = np.where(cont[..., None], CONT_RGB, OCEAN_RGB).copy()
    for k, (c, _) in EVENT_COLORS.items():
        m = cont & (ev == k)
        if m.any():
            a = np.clip(1.0 - np.where(evt[m] >= 0, evt[m], 300.0) / 300.0, 0.25, 1.0)[:, None]
            img[m] = a * np.array(to_rgb(c)) + (1 - a) * CONT_RGB
    ax.imshow(img, extent=ext, interpolation="nearest")
    _finish(ax, pick.center, lat, lon, ext, land, "(d) 陆壳最近一次构造改造（颜色越深越近；0 Ma 深 -> 300 Ma 浅）")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in EVENT_COLORS.values()], loc="upper right", fontsize=7,
              framealpha=0.85)

    fig.suptitle(f"区域 {index} · {pick.title}：同窗口构造图层（{pick.size_km:g} km 见方，与高程图同一投影；黑线 = 海岸）",
                 fontsize=12)
    attrs = world.tectonics["tect_plate"][1]
    stamp(fig, world, f"{attrs.get('method_class', '—')}（构造图层，地壳引擎 0 Ma 状态）",
          [f"中心 {pick.center[0]:.1f}°, {pick.center[1]:.1f}°。边界类型按相邻两板块的相对速度分：法向分量至少是切向一半时，"
           "靠近为汇聚、分开为离散，其余为走滑。"])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def global_plates(world, path) -> None:
    """0 Ma 全球板块与边界类型（等距圆柱）。"""
    P = world.tectonics["tect_plate"][0].astype(int)
    B = world.tectonics["tect_boundary"][0].astype(int)
    names = plate_names(world)
    fig = plt.figure(figsize=(16, 9.6))
    ax = fig.add_axes((0.02, 0.14, 0.96, 0.8))
    ext = (-180, 180, -90, 90)
    ax.imshow(plate_rgb(P, np.unique(P)), extent=ext, interpolation="nearest")
    bo = boundary_overlay(B)
    from scipy import ndimage

    for k in BOUNDARY_COLORS:          # 边界加粗一格，全球图上看得见
        m = ndimage.binary_dilation(B == k)
        bo[m & (bo[..., 3] == 0)] = (*to_rgb(BOUNDARY_COLORS[k][0]), 1.0)
    ax.imshow(bo, extent=ext, interpolation="nearest")
    ny, nx = P.shape
    lat = 90 - (np.arange(ny) + 0.5) * 180 / ny
    lon = -180 + (np.arange(nx) + 0.5) * 360 / nx
    ax.contour(lon, lat, world.land.astype(float), levels=[0.5], colors="k", linewidths=0.35)
    for q, (i0, j0) in label_points(P, 400):
        ax.text(lon[j0], lat[i0], names.get(int(q), str(int(q))), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in BOUNDARY_COLORS.values()], loc="lower left", fontsize=8,
              framealpha=0.85)
    ax.set_title("0 Ma 板块与边界类型（相对运动取自倒推的旋转文件最后 1 Myr；黑线 = 海岸）", fontsize=12)
    attrs = world.tectonics["tect_plate"][1]
    stamp(fig, world, f"{attrs.get('method_class', '—')}（构造图层，地壳引擎 0 Ma 状态）")
    fig.savefig(path, dpi=110)
    plt.close(fig)
