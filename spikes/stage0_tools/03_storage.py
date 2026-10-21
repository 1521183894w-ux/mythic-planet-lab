"""Spike 03 · 数据存储：xarray + zarr（v3）/ netCDF4。

要回答的问题：
1. 标准分辨率栅格（2048x1024）与球面网格数据（65 万节点）能否无损写出/读回？耗时、体积？
2. CLAUDE.md 第 4 节要求每个图层带 method_class 等元数据（含中文说明），能否完整往返？
3. 构造历史按时间逐片追加（append）是否可行？能否只读取某一时刻（惰性读取）？
"""

from __future__ import annotations

import shutil

import numpy as np
from _common import WORK_DIR, Recorder

rec = Recorder("03_storage")
import xarray as xr  # noqa: E402
import zarr  # noqa: E402

work = WORK_DIR / "03_storage"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

rng = np.random.default_rng(3)
ny, nx = 1024, 2048
lat = np.linspace(90, -90, ny, endpoint=False) - 90 / ny
lon = np.linspace(-180, 180, nx, endpoint=False) + 180 / nx
LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
# 近似 DEM 的数据：大尺度光滑场 + 小尺度噪声（影响压缩率）
elev = (3000 * np.sin(3 * LAT) * np.cos(2 * LON) + 800 * rng.standard_normal((ny, nx))).astype("float32")
plate = (np.floor((LON + np.pi) / (2 * np.pi) * 7) + 7 * (LAT > 0)).astype("int16")

LAYER_META = {
    "method_class": "procedural_constrained",
    "method": "spike03 synthetic field",
    "method_version": "0.0",
    "units": "m",
    "resolution": "2048x1024 equirectangular",
    "assumptions": "测试用合成数据：光滑大尺度场叠加高斯噪声，不代表任何真实地形。",
    "references": "none",
}
ds = xr.Dataset(
    {
        "elevation": (("lat", "lon"), elev, LAYER_META),
        "plate_id": (("lat", "lon"), plate, {**LAYER_META, "units": "1", "method_class": "numerical_physics"}),
    },
    coords={"lat": ("lat", lat, {"units": "degrees_north"}), "lon": ("lon", lon, {"units": "degrees_east"})},
    attrs={"world": "spike", "seed": 3, "code_version": "spike-0.4", "说明": "全局属性也可以是中文"},
)
raw_mb = sum(v.nbytes for v in ds.data_vars.values()) / 1e6
summary = {"raster_raw_mb": round(raw_mb, 1), "zarr_version": zarr.__version__, "xarray_version": xr.__version__}


def dir_mb(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def check_roundtrip(back, e):
    same = bool(np.array_equal(back["elevation"].values, elev) and np.array_equal(back["plate_id"].values, plate))
    meta_ok = all(back["elevation"].attrs.get(k) == v for k, v in LAYER_META.items())
    e["values_equal"] = same
    e["metadata_equal"] = meta_ok
    e["global_attr_zh"] = back.attrs.get("说明")
    e["status"] = "ok" if (same and meta_ok) else "fail"


# ---- 1. 栅格：zarr v3 ------------------------------------------------------------
zpath = work / "raster.zarr"
with rec.measure("zarr v3 写出 2048x1024（2 变量，块 512x512）") as e:
    ds.to_zarr(zpath, mode="w", encoding={"elevation": {"chunks": (512, 512)}, "plate_id": {"chunks": (512, 512)}})
    e["size_mb"] = round(dir_mb(zpath), 1)
    e["n_files"] = sum(1 for f in zpath.rglob("*") if f.is_file())
with rec.measure("zarr 读回并核对数值与元数据") as e:
    back = xr.open_zarr(zpath).load()
    check_roundtrip(back, e)

# ---- 2. 栅格：netCDF4（zlib 压缩）------------------------------------------------
npath = work / "raster.nc"
with rec.measure("netCDF4 写出 2048x1024（zlib 4，块 512x512）") as e:
    enc = {v: {"zlib": True, "complevel": 4, "chunksizes": (512, 512)} for v in ("elevation", "plate_id")}
    ds.to_netcdf(npath, engine="netcdf4", encoding=enc)
    e["size_mb"] = round(npath.stat().st_size / 1e6, 1)
with rec.measure("netCDF4 读回并核对数值与元数据") as e:
    with xr.open_dataset(npath, engine="netcdf4") as b:
        back = b.load()
    check_roundtrip(back, e)

# ---- 3. 球面网格（非结构化，节点维度）---------------------------------------------
n_nodes = 655362
mesh_ds = xr.Dataset(
    {"elevation": (("node",), rng.standard_normal(n_nodes).astype("float32"), {"method_class": "numerical_physics"}),
     "crust_age": (("node",), rng.uniform(0, 300, n_nodes).astype("float32"), {"units": "Myr"})},
    coords={"lon": ("node", rng.uniform(-180, 180, n_nodes)), "lat": ("node", rng.uniform(-90, 90, n_nodes))},
)
mpath = work / "mesh.zarr"
with rec.measure("zarr 写出/读回 球面网格 655,362 节点") as e:
    mesh_ds.to_zarr(mpath, mode="w")
    mb = xr.open_zarr(mpath).load()
    e["equal"] = bool(np.array_equal(mb["crust_age"].values, mesh_ds["crust_age"].values))
    e["size_mb"] = round(dir_mb(mpath), 1)
    e["status"] = "ok" if e["equal"] else "fail"

# ---- 4. 时间序列逐片追加（模拟构造历史快照）----------------------------------------
hpath = work / "history.zarr"
n_steps = 30
with rec.measure(f"zarr 按时间追加 {n_steps} 片 1024x512（模拟 300 Myr 每 10 Myr 一片）") as e:
    for i in range(n_steps):
        t = 300 - 10 * i
        snap = xr.Dataset(
            {"elevation": (("time", "lat", "lon"), elev[None, ::2, ::2] + i, LAYER_META)},
            coords={"time": ("time", [t], {"units": "Ma"}), "lat": lat[::2], "lon": lon[::2]},
        )
        if i == 0:
            snap.to_zarr(hpath, mode="w", encoding={"elevation": {"chunks": (1, 512, 512)}})
        else:
            snap.to_zarr(hpath, append_dim="time")
    e["size_mb"] = round(dir_mb(hpath), 1)
with rec.measure("惰性打开历史，只读取 150 Ma 一片") as e:
    hist = xr.open_zarr(hpath)
    e["n_times"] = int(hist.sizes["time"])
    one = hist["elevation"].sel(time=150).values
    i150 = (300 - 150) // 10
    e["equal"] = bool(np.array_equal(one, elev[::2, ::2] + i150))
    e["status"] = "ok" if e["equal"] else "fail"

rec.save(**summary)
