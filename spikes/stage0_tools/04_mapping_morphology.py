"""Spike 04 · 出图（matplotlib + cartopy）与形态学（scipy.ndimage / scikit-image）。

用 Natural Earth 1:50m 陆地多边形光栅化得到地球陆地掩膜（2048x1024），验证 0.5 指标代码要用的操作：
1. 连通区域标记：经度 180° 处是否需要"环绕"处理？
2. 面积统计：像素是否必须按 cos(纬度) 加权？
3. 距海岸距离：平面欧氏距离变换 vs 球面大圆距离，哪个能复现已知事实？
   对照：欧亚大陆"离海最远点"约 46.3°N 86.7°E，距海约 2510 km（Garcia-Castellanos & Lombardo 2007）。
4. 海岸线提取（find_contours）、形态学开运算、regionprops、盒计数分形维数的速度。
5. cartopy 在 Robinson / 正射 / 区域投影下绘制 2048x1024 场的耗时；Natural Earth 下载目录是否在工作区内。
"""

from __future__ import annotations

import numpy as np
from _common import Recorder, savefig

rec = Recorder("04_mapping_morphology")
import cartopy  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.io.shapereader as shpreader  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage import measure, morphology  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R_KM = 6371.0
NY, NX = 1024, 2048
lat = 90 - (np.arange(NY) + 0.5) * 180 / NY
lon = -180 + (np.arange(NX) + 0.5) * 360 / NX
LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
summary: dict = {"cartopy_data_dir": cartopy.config["data_dir"]}

# ---- 0. 数据 ---------------------------------------------------------------------
with rec.measure("下载/读取 Natural Earth 50m land（cartopy shapereader）") as e:
    path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    land_gdf = gpd.read_file(path)
    e["path"] = str(path)
    e["inside_workspace"] = str(path).lower().startswith(str(rec_root := __import__("_common").ROOT).lower())
    e["n_polygons"] = len(land_gdf)

with rec.measure("光栅化为 2048x1024 陆地掩膜（all_touched=False）") as e:
    land = rasterize(((g, 1) for g in land_gdf.geometry), out_shape=(NY, NX),
                     transform=from_origin(-180, 90, 360 / NX, 180 / NY), fill=0, dtype="uint8").astype(bool)
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, NX))
    e["land_frac_unweighted"] = round(float(land.mean()), 4)
    e["land_frac_area_weighted"] = round(float((land * w).sum() / w.sum()), 4)
summary["land_fraction_area_weighted"] = rec.items[-1].get("land_frac_area_weighted")

# ---- 1. 连通区域：不环绕 vs 经度环绕 ----------------------------------------------
def label_wrapped(mask: np.ndarray, structure):
    """经度方向周期边界的连通区域标记：先普通标记，再把左右边缘相接的标签合并（并查集）。"""
    lab, n = ndi.label(mask, structure=structure)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    left, right = lab[:, 0], lab[:, -1]
    # 8 连通时，左列第 i 行还和右列第 i±1 行相邻
    offsets = (0,) if structure[0, 0] == 0 else (-1, 0, 1)
    for d in offsets:
        r = np.roll(right, d)
        for a, b in zip(left, r):
            if a and b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)
    return new.reshape(-1)[lab], len(uniq) - 1


s8 = np.ones((3, 3), int)
with rec.measure("ndi.label 8 连通（不处理经度环绕）") as e:
    lab_plain, n_plain = ndi.label(land, structure=s8)
    e["n_landmasses"] = int(n_plain)
with rec.measure("ndi.label 8 连通 + 经度环绕合并") as e:
    lab, n_wrap = label_wrapped(land, s8)
    e["n_landmasses"] = int(n_wrap)
    e["merged_across_dateline"] = int(n_plain - n_wrap)

areas_km2 = ndi.sum(w * (2 * np.pi * R_KM / NX) * (np.pi * R_KM / NY), lab, index=np.arange(1, n_wrap + 1))
order = np.argsort(areas_km2)[::-1]
summary["largest_landmasses_Mkm2"] = [round(float(areas_km2[i] / 1e6), 2) for i in order[:6]]
plain_areas = ndi.sum(w * (2 * np.pi * R_KM / NX) * (np.pi * R_KM / NY), lab_plain, index=np.arange(1, n_plain + 1))
summary["largest_landmasses_Mkm2_without_wrap"] = [round(float(a / 1e6), 2) for a in np.sort(plain_areas)[::-1][:6]]

# ---- 2. 距海岸距离：平面 EDT vs 球面大圆距离 -----------------------------------------
coast = land & ~ndi.binary_erosion(land, structure=s8, border_value=0)
coast |= (~land) & ~ndi.binary_erosion(~land, structure=s8, border_value=0)  # 海岸两侧的像素都算
with rec.measure("平面欧氏距离变换 EDT（像素 -> 赤道 km 换算）") as e:
    edt_px = ndi.distance_transform_edt(land)
    km_per_px_eq = 2 * np.pi * R_KM / NX
    i = np.unravel_index(np.argmax(edt_px), edt_px.shape)
    e["max_inland_km_naive"] = round(float(edt_px[i] * km_per_px_eq))
    e["at_lat_lon"] = [round(float(lat[i[0]]), 2), round(float(lon[i[1]]), 2)]

with rec.measure("球面大圆距离（海岸像素 KD 树 + 弦长换算）") as e:
    xyz = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], axis=-1)
    ocean_side = (~land) & ndi.binary_dilation(land, structure=s8)  # 紧贴陆地的海洋像素 = 海岸
    tree = cKDTree(xyz[ocean_side])
    chord, _ = tree.query(xyz[land], k=1, workers=-1)
    dist = np.zeros((NY, NX))
    dist[land] = 2 * R_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    # 只在欧亚大陆（含环绕合并）里找离海最远点
    eurasia_label = lab[int(np.argmin(np.abs(lat - 50))), int(np.argmin(np.abs(lon - 90)))]
    d_eu = np.where(lab == eurasia_label, dist, 0)
    j = np.unravel_index(np.argmax(d_eu), d_eu.shape)
    e["eurasia_max_inland_km"] = round(float(d_eu[j]))
    e["at_lat_lon"] = [round(float(lat[j[0]]), 2), round(float(lon[j[1]]), 2)]
    e["reference"] = "≈2510 km @ 46.3N 86.7E (Garcia-Castellanos & Lombardo 2007)"
summary["eurasia_pole_of_inaccessibility"] = {k: rec.items[-1][k] for k in ("eurasia_max_inland_km", "at_lat_lon")}

# ---- 3. 海岸线、形态学、区域属性、分形维数 -----------------------------------------
with rec.measure("find_contours 海岸线 + 球面长度") as e:
    contours = measure.find_contours(land.astype(float), 0.5)
    total = 0.0
    for c in contours:
        la = np.radians(90 - (c[:, 0] + 0.5) * 180 / NY)
        lo = np.radians(-180 + (c[:, 1] + 0.5) * 360 / NX)
        p = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total += float((2 * R_KM * np.arcsin(np.clip(seg / 2, 0, 1))).sum())
    e["n_contours"] = len(contours)
    e["coastline_km_at_this_resolution"] = round(total)

with rec.measure("形态学开运算 disk(2)：被削掉的细窄陆地比例") as e:
    opened = ndi.binary_opening(land, structure=morphology.disk(2))
    e["land_removed_frac"] = round(float(1 - (opened * w).sum() / (land * w).sum()), 4)

with rec.measure("regionprops：前 5 大陆块的紧凑度 4πA/P²（像素空间）") as e:
    props = measure.regionprops(lab)
    props = sorted(props, key=lambda p: p.area, reverse=True)[:5]
    e["compactness"] = [round(float(4 * np.pi * p.area / p.perimeter**2), 3) for p in props]

with rec.measure("盒计数分形维数（全球海岸像素，等距圆柱投影下）") as e:
    sizes = np.array([2, 4, 8, 16, 32, 64])
    counts = []
    for s in sizes:
        h, wdt = NY // s * s, NX // s * s
        blocks = coast[:h, :wdt].reshape(h // s, s, wdt // s, s).any(axis=(1, 3))
        counts.append(blocks.sum())
    slope = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    e["box_dimension"] = round(float(slope), 3)

# ---- 4. 出图（同时测 cartopy 绘制耗时）--------------------------------------------
signed = np.where(land, dist, 0.0)
with rec.measure("cartopy 绘制：Robinson 2048x1024 + 50m 海岸线 + 保存") as e:
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1.set_global()
    # 按面积排名上色：前 10 大陆块各用一种颜色，其余灰色（避免取余配色让不同陆块撞色）
    rank = np.full(n_wrap + 1, 10)
    rank[order[:10] + 1] = np.arange(10)
    shown = np.where(lab > 0, rank[lab], np.nan)
    cmap = matplotlib.colors.ListedColormap(list(plt.get_cmap("tab10").colors) + [(0.6, 0.6, 0.6)])
    ax1.imshow(shown, extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.5, vmax=10.5,
               interpolation="nearest")
    ax1.set_title(f"Landmasses (8-conn., dateline-wrapped): {n_wrap} (vs {n_plain} unwrapped); top 10 colored", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())
    ax2.set_global()
    im = ax2.imshow(np.where(land, signed, np.nan), extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(),
                    cmap="magma_r", vmin=0, vmax=2600)
    ax2.coastlines("50m", lw=0.3)
    jy, jx = j
    ax2.plot(lon[jx], lat[jy], "c*", ms=12, transform=ccrs.PlateCarree())
    ax2.set_title(f"Distance to coast, great-circle (km)\nEurasia max {d_eu[j]:.0f} km (ref ~2510 km)", fontsize=9)
    fig.colorbar(im, ax=ax2, shrink=0.6, label="km")
    fig.suptitle("Spike 04 - Natural Earth 50m land rasterized to 2048x1024", fontsize=10)
    savefig(fig, "04_landmasses_distance.png")
    plt.close(fig)

with rec.measure("cartopy 绘制：正射投影 + 区域图（地中海）") as e:
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(100, 35))
    ax.set_global()
    ax.imshow(np.where(land, signed, np.nan), extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(),
              cmap="magma_r", vmin=0, vmax=2600)
    ax.coastlines("110m", lw=0.4)
    ax.set_title("Orthographic", fontsize=9)
    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent([-8, 40, 28, 48], crs=ccrs.PlateCarree())
    ax.imshow(land.astype(float), extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap="Greys",
              vmin=0, vmax=1.6, interpolation="nearest")
    ax.coastlines("50m", lw=0.6, color="tab:red")
    ax.set_title("Mediterranean: 2048x1024 mask (grey, ~20 km px) vs NE 50m coastline (red)", fontsize=9)
    savefig(fig, "04_projections_regional.png")
    plt.close(fig)

rec.save(**summary)
