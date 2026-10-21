"""构建地球基准格网数据：ETOPO 2022（高程）+ Natural Earth（海陆、湖泊、大洲）-> 三级格网。

输入（见 data/external_manifests/）：
  data/earth_reference/raw/etopo2022/ETOPO_2022_v1_60s_N90W180_surface.nc
  data/earth_reference/raw/natural_earth/ne_10m_{land,lakes,admin_0_countries}.zip
输出（不进 Git，可随时重算）：
  data/earth_reference/earth_grid_<nx>x<ny>.nc，变量：
    elevation      面积平均高程/水深（m，EGM2008）
    land_fraction  面元中陆地所占比例（Natural Earth 10m 陆地，在 1 角分上光栅化后面积平均）
    lake_fraction  面元中大湖所占比例（Natural Earth 10m 湖泊）
    continent_id   大洲编号（Natural Earth 国家图层 CONTINENT 字段；海洋为 0）

为什么海陆不用 "高程 > 0"：低于海平面的干旱洼地（卡塔拉洼地 −133 m、吐鲁番盆地等）会被误判为海。
虚构世界在阶段 1 没有湖泊和这类洼地的区分，直接用 高程 > 0；地球用权威陆地多边形是同一概念
（"面元多数是陆地"）的正确实现。两种判法对地球海陆比的影响在基准 JSON 里一并记录。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\build_earth_reference.py
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage as ndi

from mythic.metrics.grid import LatLonGrid

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "earth_reference" / "raw"
OUT = ROOT / "data" / "earth_reference"
ETOPO = RAW / "etopo2022" / "ETOPO_2022_v1_60s_N90W180_surface.nc"
NE = RAW / "natural_earth"
GRIDS = [LatLonGrid(512, 256), LatLonGrid(1024, 512), LatLonGrid(2048, 1024)]
SRC_NX, SRC_NY = 21600, 10800  # 60 角秒
SRC_TRANSFORM = from_origin(-180.0, 90.0, 1 / 60, 1 / 60)
CRS = "EPSG:4326"

CONTINENTS = {  # Natural Earth CONTINENT 字段 -> 编号（0 = 海洋/未归属）
    "Africa": 1,
    "Europe": 2,
    "Asia": 3,
    "North America": 4,
    "South America": 5,
    "Oceania": 6,
    "Antarctica": 7,
    "Seven seas (open ocean)": 8,  # 远洋小岛（如圣赫勒拿等）
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_shapes(zip_name: str) -> gpd.GeoDataFrame:
    path = NE / zip_name
    with zipfile.ZipFile(path) as zf:
        shp = [n for n in zf.namelist() if n.endswith(".shp")][0]
    return gpd.read_file(f"zip://{path.as_posix()}!{shp}")


def average_to(src: np.ndarray, grid: LatLonGrid) -> np.ndarray:
    """60 角秒源数据 -> 目标格网的面积平均（rasterio 'average' 重采样，处理非整数倍）。
    注：等经纬度格网上每个源像元面积随纬度变化，但同一目标面元内纬度跨度很小（≤ 0.7°），按等权平均误差可忽略。"""
    dst = np.zeros(grid.shape, dtype="float32")
    reproject(source=src, destination=dst, src_transform=SRC_TRANSFORM, src_crs=CRS,
              dst_transform=grid.transform(), dst_crs=CRS, resampling=Resampling.average)
    return dst


def main() -> None:
    t_start = time.time()
    log(f"读取 ETOPO：{ETOPO.name}")
    with netCDF4.Dataset(ETOPO) as ds:
        lat = ds.variables["lat"][:]
        z = np.asarray(ds.variables["z"][:], dtype="float32")
    if lat[0] < lat[-1]:  # ETOPO 自南向北存放，翻转为自北向南
        z = z[::-1, :]
    assert z.shape == (SRC_NY, SRC_NX), z.shape
    elevations = {g.name: average_to(z, g) for g in GRIDS}
    etopo_land_frac = {g.name: average_to((z > 0).astype("float32"), g) for g in GRIDS}
    del z
    log("ETOPO 重采样完成")

    log("光栅化 Natural Earth 10m 陆地（1 角分）")
    land = read_shapes("ne_10m_land.zip")
    src = rasterize(((geom, 1) for geom in land.geometry), out_shape=(SRC_NY, SRC_NX),
                    transform=SRC_TRANSFORM, fill=0, dtype="uint8")
    land_frac = {g.name: average_to(src.astype("float32"), g) for g in GRIDS}
    del src
    log("光栅化 Natural Earth 10m 湖泊")
    lakes = read_shapes("ne_10m_lakes.zip")
    src = rasterize(((geom, 1) for geom in lakes.geometry), out_shape=(SRC_NY, SRC_NX),
                    transform=SRC_TRANSFORM, fill=0, dtype="uint8")
    lake_frac = {g.name: average_to(src.astype("float32"), g) for g in GRIDS}
    del src

    log("大洲编号")
    countries = read_shapes("ne_10m_admin_0_countries.zip")
    countries["cid"] = countries["CONTINENT"].map(CONTINENTS).fillna(0).astype(int)
    unknown = sorted(set(countries["CONTINENT"]) - set(CONTINENTS))
    if unknown:
        log(f"  未登记的 CONTINENT 值（编号记为 0）：{unknown}")

    for g in GRIDS:
        cont = rasterize(((geom, cid) for geom, cid in zip(countries.geometry, countries["cid"]) if cid),
                         out_shape=g.shape, transform=g.transform(), fill=0, dtype="uint8", all_touched=True)
        land_mask = land_frac[g.name] >= 0.5
        # 陆地面元若未被国家多边形覆盖（海岸线差异），取最近的已编号面元
        missing = land_mask & (cont == 0)
        if missing.any():
            idx = ndi.distance_transform_edt(cont == 0, return_distances=False, return_indices=True)
            cont = np.where(missing, cont[idx[0], idx[1]], cont)
        cont = np.where(land_mask, cont, 0).astype("uint8")
        # Natural Earth 把整个俄罗斯标为 Europe（实测"欧洲"达 23 Mkm²，含西伯利亚直到楚科奇）。
        # 按惯例以乌拉尔山一线（约 60°E）划分：60°E 以东、以及跨日界线的楚科奇（< 100°W）改为 Asia。
        lon2d = np.broadcast_to(g.lon[None, :], g.shape)
        cont = np.where((cont == CONTINENTS["Europe"]) & ((lon2d >= 60.0) | (lon2d < -100.0)),
                        CONTINENTS["Asia"], cont).astype("uint8")

        ds = xr.Dataset(
            {
                "elevation": (("lat", "lon"), elevations[g.name],
                              {"units": "m", "vertical_datum": "EGM2008",
                               "method_class": "interpolated",
                               "method": "ETOPO 2022 v1 60s surface -> 面积平均（rasterio average）"}),
                "land_fraction": (("lat", "lon"), land_frac[g.name],
                                  {"units": "1", "method_class": "interpolated",
                                   "method": "Natural Earth 10m land v5.1.1 于 1 角分光栅化后面积平均"}),
                "land_fraction_etopo": (("lat", "lon"), etopo_land_frac[g.name],
                                        {"units": "1", "method_class": "interpolated",
                                         "method": "ETOPO 高程 > 0 的面积比例（仅作对照，不用于基准）"}),
                "lake_fraction": (("lat", "lon"), lake_frac[g.name],
                                  {"units": "1", "method_class": "interpolated",
                                   "method": "Natural Earth 10m lakes v5.0.0 于 1 角分光栅化后面积平均"}),
                "continent_id": (("lat", "lon"), cont,
                                 {"method_class": "interpolated",
                                  "codes": "; ".join(f"{v}={k}" for k, v in CONTINENTS.items()) + "; 0=ocean",
                                  "method": "Natural Earth 10m admin-0 v5.1.1 CONTINENT 字段，all_touched 光栅化 + 最近邻补齐；"
                                            "Europe 中 60°E 以东及 100°W 以西（楚科奇）改为 Asia（乌拉尔分界）"}),
            },
            coords={"lat": ("lat", g.lat, {"units": "degrees_north"}),
                    "lon": ("lon", g.lon, {"units": "degrees_east"})},
            attrs={"title": f"Earth reference grid {g.name}", "grid": g.name,
                   "sources": "ETOPO 2022 v1 60s surface (doi:10.25921/fd45-gt74); Natural Earth 10m v5.1.1/5.0.0",
                   "manifests": "data/external_manifests/etopo2022.yaml, natural_earth.yaml",
                   "generated_by": "scripts/build_earth_reference.py",
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
        )
        out = OUT / f"earth_grid_{g.name}.nc"
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out, encoding=enc)
        area = g.cell_area_km2
        lf = (land_frac[g.name] * area).sum() / area.sum()
        lf_mask = ((land_frac[g.name] >= 0.5) * area).sum() / area.sum()
        lf_etopo = ((etopo_land_frac[g.name] >= 0.5) * area).sum() / area.sum()
        log(f"  {out.name}: 陆地面积比例 {lf:.4f}（按比例累加）/ {lf_mask:.4f}（≥0.5 判为陆地）/ "
            f"{lf_etopo:.4f}（若用 ETOPO 高程>0）；高程范围 {elevations[g.name].min():.0f}..{elevations[g.name].max():.0f} m")
    log(f"完成，用时 {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
