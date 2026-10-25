"""统一出图风格（review_protocol §2.1）：
- 高程：分层设色 + 山体阴影，海平面处颜色明确断开，陆地与海洋色带在所有图中一致；
- 中文字体（Windows：微软雅黑 / 黑体）；
- 比例尺。
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

# 指标文本框的字体：显式给出列表，matplotlib 会对缺失字形逐个回退（西文等宽 Consolas，中文雅黑）
MONO_FAMILY = ["Consolas", "Microsoft YaHei", "SimHei", "DejaVu Sans Mono"]

ELEV_MIN_M = -8000.0
ELEV_MAX_M = 6000.0

_OCEAN = LinearSegmentedColormap.from_list(
    "mythic_ocean", ["#08153a", "#0d2f6b", "#1f5fa8", "#4f97d1", "#9fd0ef", "#cfeaf7"])
_LAND = LinearSegmentedColormap.from_list(
    "mythic_land", ["#3f7f3a", "#8fb35a", "#d8cf86", "#c9a064", "#9c6b45", "#8a8073", "#d9d4cf", "#ffffff"])


def elevation_cmap_norm():
    """海陆分段色带：海洋 [-8000, 0) 与陆地 [0, 6000] 各占一半，海平面处颜色跳变。"""
    colors = np.vstack([_OCEAN(np.linspace(0, 1, 128)), _LAND(np.linspace(0, 1, 128))])
    cmap = ListedColormap(colors, name="mythic_elevation")
    norm = TwoSlopeNorm(vmin=ELEV_MIN_M, vcenter=0.0, vmax=ELEV_MAX_M)
    return cmap, norm


ICE_SHELF_RGB = (0.89, 0.94, 0.97)


def shaded_elevation(elev_m: np.ndarray, pixel_km: float, vert_exag: float = 1.0,
                     land: np.ndarray | None = None) -> np.ndarray:
    """分层设色 + 山体阴影的 RGB 图像。

    给出 land（海陆掩膜）时，着色服从掩膜（0.7 校准评审员 B #9）：陆地格点即使低于海平面（干洼地）也用陆地色带，
    海洋格点用海洋色带；掩膜为海、高程却为正的格点（漂浮冰架：海陆按海计，ETOPO 给冰面高程）涂成冰色。"""
    cmap, norm = elevation_cmap_norm()
    z = np.clip(elev_m, ELEV_MIN_M, ELEV_MAX_M)
    ice = None
    if land is not None:
        land = np.asarray(land) >= 0.5
        ice = (~land) & (z > 0)
        z = np.where(land, np.maximum(z, 1.0), np.minimum(z, -1.0))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=vert_exag,
                   dx=pixel_km * 1000.0, dy=pixel_km * 1000.0)
    if ice is not None and ice.any():
        rgb[ice, :3] = ICE_SHELF_RGB
    return rgb


def has_ice_shelf(elev_m: np.ndarray, land: np.ndarray) -> bool:
    return bool(((np.asarray(land) < 0.5) & (elev_m > 0)).any())


def setup_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    # 等宽字体（图中的指标文本框）：西文用等宽字体，中文字形缺失时由 matplotlib 按列表顺序回退到雅黑
    matplotlib.rcParams["font.monospace"] = ["Consolas", "DejaVu Sans Mono", "Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def stamp(fig, world, method_note: str, extra: list[str] | None = None, fontsize: float = 7.5) -> None:
    """图底部的固定说明：世界、格网、种子、代码版本、method_class、数据来源、生成时间（review_protocol §2.1）。"""
    import time

    lines = list(extra or [])
    lines.append(f"世界：{world.display_name}（{world.world_id}） · 格网 {world.resolution} · 种子 {world.seed} · "
                 f"代码 {world.code_version} · 生成 {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"method_class：{method_note} · 数据：{world.source}")
    # 按图宽折行（中文字宽约等于字号；之前一行写完，长的来源说明在右边被截断，评审 stage1_2026-10-25_2 第 9 条）
    import textwrap
    width = max(40, int(fig.get_figwidth() * 72.0 / fontsize) - 6)
    wrapped = [w for ln in lines for w in (textwrap.wrap(ln, width=width) or [""])]
    fig.text(0.01, 0.005, "\n".join(wrapped), fontsize=fontsize, va="bottom", ha="left", color="0.2")


def add_scalebar(ax, extent_km, length_km: float = 500.0, color: str = "k") -> None:
    """在 km 坐标的轴（imshow extent 为 km）左下角画比例尺。"""
    x0, x1, y0, y1 = extent_km
    px = x0 + 0.06 * (x1 - x0)
    py = y0 + 0.06 * (y1 - y0)
    ax.plot([px, px + length_km], [py, py], color=color, lw=3, solid_capstyle="butt")
    ax.text(px + length_km / 2, py + 0.02 * (y1 - y0), f"{length_km:g} km", ha="center", va="bottom",
            fontsize=8, color=color)
