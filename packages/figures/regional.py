"""区域裁剪：在以区域中心为原点的局部等面积方位投影（LAEA）上、按固定的 km 像元重采样全球格网。

用途：评审时"世界区域图"与"地球区域图"并排，必须同比例尺、同分辨率（review_protocol §2.2）。
在 km 坐标上裁剪，高纬地区（挪威、格陵兰）与低纬地区比例一致，不受等经纬度格网的高纬拉伸影响。
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

EARTH_RADIUS_M = 6371000.0


def laea_window(center_lat: float, center_lon: float, size_km: float, pixel_km: float):
    """返回窗口内各像元中心的 (lat, lon)（度）与像元坐标范围（km，用于 imshow 的 extent）。
    第 0 行在北（y 最大）。"""
    n = max(2, int(round(size_km / pixel_km)))
    half = n * pixel_km / 2.0
    xs = (np.arange(n) + 0.5) * pixel_km - half
    ys = half - (np.arange(n) + 0.5) * pixel_km
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(
        f"+proj=laea +lat_0={center_lat} +lon_0={center_lon} +R={EARTH_RADIUS_M} +units=m +no_defs",
        f"+proj=longlat +R={EARTH_RADIUS_M} +no_defs", always_xy=True)
    lon, lat = tr.transform(X * 1000.0, Y * 1000.0)
    return lat, lon, (-half, half, -half, half)


def laea_forward(center_lat: float, center_lon: float, lat, lon):
    """经纬度（度）-> 局部 LAEA 坐标（km），与 laea_window 使用同一投影。"""
    tr = Transformer.from_crs(
        f"+proj=longlat +R={EARTH_RADIUS_M} +no_defs",
        f"+proj=laea +lat_0={center_lat} +lon_0={center_lon} +R={EARTH_RADIUS_M} +units=m +no_defs", always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0


def graticule_lines(center_lat: float, center_lon: float, lat: np.ndarray, lon: np.ndarray, step_deg: float):
    """窗口内的经纬网折线（km 坐标）。直接投影每条经线/纬线，在极点和日界线附近都不会断裂或乱线。
    返回 [(kind, value, x_km, y_km), ...]。"""
    lines = []
    lat_lo, lat_hi = max(-90.0, lat.min() - step_deg), min(90.0, lat.max() + step_deg)
    near_pole = lat.max() > 80 or lat.min() < -80
    t = np.linspace(lat_lo, lat_hi, 400)
    lon_step = 3 * step_deg if near_pole else step_deg
    for lv in np.arange(-180.0, 180.0, lon_step):
        x, y = laea_forward(center_lat, center_lon, t, np.full_like(t, lv))
        lines.append(("lon", lv, x, y))
    s = np.linspace(-180.0, 180.0, 1441)
    for pv in np.arange(np.ceil(lat_lo / step_deg) * step_deg, lat_hi + 1e-9, step_deg):
        if abs(pv) >= 89.9:
            continue
        x, y = laea_forward(center_lat, center_lon, np.full_like(s, pv), s)
        lines.append(("lat", pv, x, y))
    return lines


def draw_graticule(ax, center, lat, lon, extent, step=5.0, label_lat: bool = True) -> None:
    """经纬网：直接投影每条经线/纬线（极点附近经线间隔放大）。纬线在左侧边缘标注。"""
    x0, x1, y0, y1 = extent
    for kind, v, x, y in graticule_lines(center[0], center[1], lat, lon, step):
        ok = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if ok.sum() < 2:
            continue
        xs, ys = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
        ax.plot(xs, ys, color="0.3", lw=0.4, alpha=0.6)
        if kind == "lat" and label_lat:
            k = int(np.nanargmin(np.where(ok, x, np.inf)))
            ax.text(x[k], y[k], f"{abs(v):g}°{'N' if v >= 0 else 'S'}", fontsize=6, color="0.25", va="bottom")


def draw_elevation_panel(ax, elev, land, lat, lon, extent, pixel_km, title, center, graticule_step=5.0) -> None:
    """区域窗口面板：分层设色 + 山体阴影、海岸线（陆地掩膜 0.5 等值线）、经纬网、比例尺。"""
    from .style import add_scalebar, shaded_elevation

    ax.imshow(shaded_elevation(elev, pixel_km, vert_exag=2.0 if pixel_km > 5 else 1.0, land=land), extent=extent,
              interpolation="nearest")
    ax.contour(land.astype(float), levels=[0.5], colors="k", linewidths=0.6, extent=extent, origin="upper")
    draw_graticule(ax, center, lat, lon, extent, graticule_step)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    width = extent[1] - extent[0]
    add_scalebar(ax, extent, 1000.0 if width >= 3000 else (500.0 if width >= 1500 else 200.0))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def sample_global(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, order: int = 1) -> np.ndarray:
    """在全球等经纬度格网（像元配准、第 0 行在北、第 0 列从 180°W 起）上取样，经度环绕。
    order=0 最近邻（如实显示格网像元），order=1 双线性。"""
    ny, nx = field.shape
    row = (90.0 - lat) / (180.0 / ny) - 0.5
    lon = (lon + 180.0) % 360.0 - 180.0
    col = (lon + 180.0) / (360.0 / nx) - 0.5
    pad = 2
    padded = np.concatenate([field[:, -pad:], field, field[:, :pad]], axis=1)
    return ndi.map_coordinates(padded, [row, col + pad], order=order, mode="nearest")
