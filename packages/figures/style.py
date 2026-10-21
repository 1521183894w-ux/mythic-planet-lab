"""统一出图风格（review_protocol §2.1）：
- 高程：分层设色 + 山体阴影，海平面处颜色明确断开，陆地与海洋色带在所有图中一致；
- 中文字体（Windows：微软雅黑 / 黑体）；
- 比例尺。
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

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


def shaded_elevation(elev_m: np.ndarray, pixel_km: float, vert_exag: float = 1.0) -> np.ndarray:
    """分层设色 + 山体阴影的 RGB 图像。"""
    cmap, norm = elevation_cmap_norm()
    z = np.clip(elev_m, ELEV_MIN_M, ELEV_MAX_M)
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.shade(z, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=vert_exag,
                    dx=pixel_km * 1000.0, dy=pixel_km * 1000.0)


def setup_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def add_scalebar(ax, extent_km, length_km: float = 500.0, color: str = "k") -> None:
    """在 km 坐标的轴（imshow extent 为 km）左下角画比例尺。"""
    x0, x1, y0, y1 = extent_km
    px = x0 + 0.06 * (x1 - x0)
    py = y0 + 0.06 * (y1 - y0)
    ax.plot([px, px + length_km], [py, py], color=color, lw=3, solid_capstyle="butt")
    ax.text(px + length_km / 2, py + 0.02 * (y1 - y0), f"{length_km:g} km", ha="center", va="bottom",
            fontsize=8, color=color)
