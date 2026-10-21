"""生成地球区域参照图集：data/earth_reference/regions/<id>.png 与 index.json。

区域清单见 config/earth_regions.yaml。每个区域一张两联图（同一局部等面积方位投影窗口、同比例尺）：
  左：与世界标准格网相同的分辨率（2048x1024，约 19.5 km，最近邻取样，如实显示格网像元）——评审并排对照用；
  右：ETOPO 2022 60 角秒（约 1.9 km）——展示真实地理细节，供区域细化阶段对照。
黑线：海岸线（Natural Earth 陆地）；灰线：经纬网。method_class: interpolated。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\build_region_atlas.py [region_id ...]
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import netCDF4
import numpy as np
import xarray as xr
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_origin

from mythic.figures.regional import draw_elevation_panel, laea_window, sample_global
from mythic.figures.style import elevation_cmap_norm, setup_fonts

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "earth_reference"
OUT = REF / "regions"
GRID_RES = "2048x1024"
GRID_PIXEL_KM = 2 * np.pi * 6371.0 / 2048  # 与世界格网赤道像元一致
FINE_PIXEL_KM = 2.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_fine():
    log("读取 ETOPO 60″ 与光栅化 Natural Earth 陆地（1 角分）")
    with netCDF4.Dataset(REF / "raw" / "etopo2022" / "ETOPO_2022_v1_60s_N90W180_surface.nc") as ds:
        lat = ds.variables["lat"][:]
        z = np.asarray(ds.variables["z"][:], dtype="float32")
    if lat[0] < lat[-1]:
        z = z[::-1, :]
    path = REF / "raw" / "natural_earth" / "ne_10m_land.zip"
    with zipfile.ZipFile(path) as zf:
        shp = [n for n in zf.namelist() if n.endswith(".shp")][0]
    land = gpd.read_file(f"zip://{path.as_posix()}!{shp}")
    mask = rasterize(((g, 1) for g in land.geometry), out_shape=z.shape,
                     transform=from_origin(-180, 90, 1 / 60, 1 / 60), fill=0, dtype="uint8")
    return z, mask


def main(selected: list[str]) -> None:
    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config" / "earth_regions.yaml").read_text(encoding="utf-8"))
    regions = [r for r in cfg["regions"] if not selected or r["id"] in selected]
    ds = xr.open_dataset(REF / f"earth_grid_{GRID_RES}.nc")
    g_elev = ds.elevation.values.astype("float32")
    g_land = (ds.land_fraction.values >= 0.5).astype("float32")
    f_elev, f_land = load_fine()
    cmap, norm = elevation_cmap_norm()
    index = []
    for r in regions:
        t0 = time.perf_counter()
        size = float(r.get("size_km", cfg["default_size_km"]))
        clat, clon = r["center"]
        lat_g, lon_g, ext = laea_window(clat, clon, size, GRID_PIXEL_KM)
        lat_f, lon_f, ext_f = laea_window(clat, clon, size, FINE_PIXEL_KM)
        e_g = sample_global(g_elev, lat_g, lon_g, order=0)
        l_g = sample_global(g_land, lat_g, lon_g, order=0)
        e_f = sample_global(f_elev, lat_f, lon_f, order=1)
        l_f = sample_global(f_land.astype("float32"), lat_f, lon_f, order=0)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
        fig.subplots_adjust(top=0.9, bottom=0.14, wspace=0.05)
        draw_elevation_panel(axes[0], e_g, l_g, lat_g, lon_g, ext, GRID_PIXEL_KM,
                             f"世界标准格网分辨率（2048×1024，约 {GRID_PIXEL_KM:.1f} km）—— 评审并排对照用", (clat, clon))
        draw_elevation_panel(axes[1], e_f, l_f, lat_f, lon_f, ext_f, FINE_PIXEL_KM,
                             f"ETOPO 2022 60″（约 {FINE_PIXEL_KM:.0f} km 重采样）—— 真实细节", (clat, clon))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
        cb.set_label("高程 / 水深（m，EGM2008；海平面处色带断开）")
        fig.suptitle(f"地球参照 · {r['name']}（{r['id']}）  中心 {clat:g}°, {clon:g}°  ·  {size:g} km 见方 · 局部等面积方位投影",
                     fontsize=12)
        foot = []
        if r.get("review_role"):
            foot.append("评审对照：" + "；".join(r["review_role"]))
        if r.get("archetypes"):
            foot.append("原型地球例子：" + "；".join(r["archetypes"]))
        if clat < -60:
            foot.append("注：冰架是漂浮冰，Natural Earth 不算陆地（被海岸线圈在海中），但 ETOPO 给的是冰面高程，所以显示为正高程。")
        foot.append("数据：ETOPO 2022（NOAA NCEI, doi:10.25921/fd45-gt74）、Natural Earth 10m 陆地；method_class: interpolated；"
                    "脚本 scripts/build_region_atlas.py")
        fig.text(0.01, 0.01, "\n".join(foot), fontsize=8, va="bottom")
        out = OUT / f"{r['id']}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        land_frac_g = float(l_g.mean())
        index.append({"id": r["id"], "name": r["name"], "file": f"{r['id']}.png", "center": [clat, clon],
                      "size_km": size, "grid_pixel_km": round(GRID_PIXEL_KM, 2), "fine_pixel_km": FINE_PIXEL_KM,
                      "review_role": r.get("review_role", []), "archetypes": r.get("archetypes", []),
                      "land_fraction_in_window_grid": round(land_frac_g, 3)})
        log(f"  {r['id']:<22} {time.perf_counter() - t0:4.1f}s  陆地占窗口 {land_frac_g:.0%}")
    if not selected:
        (OUT / "index.json").write_text(json.dumps({
            "description": "地球区域参照图集（docs/review_protocol.md §2.2；docs/archetype_library.md §9）",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": "config/earth_regions.yaml", "regions": index}, ensure_ascii=False, indent=1),
            encoding="utf-8", newline="\n")
    log(f"完成 {len(index)} 个区域 -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
