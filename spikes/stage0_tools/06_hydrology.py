"""Spike 06 · 水文：richdem 与 pysheds（洼地填充、D8 流向、汇水面积）。

要回答的问题（阶段 2 河网与流域）：
1. 在同一 DEM 上，两者算出的汇水面积是否与 Fastscape 内核自己的汇流结果一致？
2. 在全球标准分辨率 2048x1024 上耗时与内存如何？
3. 有没有需要注意的接口坑（输入格式、输出单位、边界处理）？

输入：spike 05 生成的稳态地形（.cache/spikes/05_lem/dem_fastscape_256_steady.npy）。
"""

from __future__ import annotations

import io
import contextlib

import numpy as np
from _common import WORK_DIR, Recorder, available_gb, savefig

rec = Recorder("06_hydrology")
import fastscapelib_fortran as fsf  # noqa: E402
import pysheds  # noqa: E402
import rasterio  # noqa: E402
import richdem as rd  # noqa: E402
from fastscape.processes.flow import _flow_accumulate_sd  # noqa: E402
from pysheds.grid import Grid  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

# pysheds 0.5 在 sgrid.py 中 9 处调用 np.in1d，而 numpy 2.4+ 已删除该函数（实测 numpy 2.5.4 下
# accumulation 等直接报 AttributeError）。为评估 pysheds 其余部分，本 spike 临时补一个等价实现；
# 这只是测试手段，不进入项目代码（项目代码不应依赖对 numpy 打补丁）。
PYSHEDS_SHIM = not hasattr(np, "in1d")
if PYSHEDS_SHIM:
    np.in1d = lambda ar1, ar2, assume_unique=False, invert=False: np.isin(  # noqa: E731
        np.ravel(ar1), ar2, assume_unique=assume_unique, invert=invert)

work = WORK_DIR / "06_hydro"
work.mkdir(parents=True, exist_ok=True)
DX = 500.0
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # pysheds 的 ESRI 方向编码（N, NE, E, SE, S, SW, W, NW）

dem_path = WORK_DIR / "05_lem" / "dem_fastscape_256_steady.npy"
z = np.load(dem_path)
summary: dict = {"richdem_version": rd.__version__, "pysheds_version": pysheds.__version__, "dem": str(dem_path),
                 "pysheds_needed_np_in1d_shim": PYSHEDS_SHIM}


def quiet(func, *args, **kwargs):
    """richdem 会向标准输出打印大量进度与文献信息，这里吞掉。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def richdem_area(dem: np.ndarray, dx: float):
    r = rd.rdarray(dem.astype("float64"), no_data=-9999.0)
    r.geotransform = [0.0, dx, 0.0, 0.0, 0.0, -dx]
    filled = quiet(rd.FillDepressions, r, epsilon=True, in_place=False)
    acc = quiet(rd.FlowAccumulation, filled, method="D8")
    return np.asarray(acc) * dx * dx  # richdem 输出的是上游格点数（含自身）


def pysheds_area(dem: np.ndarray, dx: float, name: str):
    path = work / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=dem.shape[1], height=dem.shape[0], count=1,
                       dtype="float64", crs="EPSG:3857", transform=from_origin(0, dem.shape[0] * dx, dx, dx),
                       nodata=-9999.0) as dst:
        dst.write(dem.astype("float64"), 1)
    grid = Grid.from_raster(str(path))
    d = grid.read_raster(str(path))
    filled = grid.fill_depressions(grid.fill_pits(d))
    inflated = grid.resolve_flats(filled)
    fdir = grid.flowdir(inflated, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return np.asarray(acc) * dx * dx


def fastscape_area(dem: np.ndarray, dx: float):
    ny, nx = dem.shape
    fsf.fastscape_init()
    fsf.fastscape_set_nx_ny(nx, ny)
    fsf.fastscape_setup()
    fsf.fastscape_set_xl_yl((nx - 1) * dx, (ny - 1) * dx)
    fsf.fastscape_set_bc(1111)
    ctx = fsf.fastscapecontext
    ctx.h = dem.ravel().astype("float64")
    fsf.flowroutingsingleflowdirection()
    area = np.full(nx * ny, dx * dx)
    _flow_accumulate_sd(area, ctx.stack.astype("int64") - 1, ctx.rec.astype("int64") - 1)
    fsf.fastscape_destroy()
    return area.reshape(ny, nx)


# ---- 1. 正确性：稳态 DEM 上三者对比 --------------------------------------------------
with rec.measure("richdem：填洼(epsilon) + D8 汇流，256x256") as e:
    a_rd = richdem_area(z, DX)
    e["max_area_km2"] = round(float(a_rd.max() / 1e6), 1)
with rec.measure("pysheds：fill_pits + fill_depressions + resolve_flats + D8，256x256") as e:
    a_ps = pysheds_area(z, DX, "steady256")
    e["max_area_km2"] = round(float(a_ps.max() / 1e6), 1)
with rec.measure("Fastscape 内核单流向汇流（参照），256x256") as e:
    a_fs = fastscape_area(z, DX)
    e["max_area_km2"] = round(float(a_fs.max() / 1e6), 1)

with rec.measure("三者汇水面积一致性（内部节点，log10 相关、主河道重合度）") as e:
    core = (slice(2, -2), slice(2, -2))
    L = {k: np.log10(v[core]) for k, v in (("richdem", a_rd), ("pysheds", a_ps), ("fastscape", a_fs))}
    e["corr_richdem_fastscape"] = round(float(np.corrcoef(L["richdem"].ravel(), L["fastscape"].ravel())[0, 1]), 4)
    e["corr_pysheds_fastscape"] = round(float(np.corrcoef(L["pysheds"].ravel(), L["fastscape"].ravel())[0, 1]), 4)
    e["corr_richdem_pysheds"] = round(float(np.corrcoef(L["richdem"].ravel(), L["pysheds"].ravel())[0, 1]), 4)
    # 河道（汇水面积 > 10 km²）像素的重合比例（Jaccard）
    ch = {k: v[core] > 1e7 for k, v in (("richdem", a_rd), ("pysheds", a_ps), ("fastscape", a_fs))}
    jac = lambda a, b: float((a & b).sum() / (a | b).sum())  # noqa: E731
    e["channel_jaccard_rd_fs"] = round(jac(ch["richdem"], ch["fastscape"]), 3)
    e["channel_jaccard_ps_fs"] = round(jac(ch["pysheds"], ch["fastscape"]), 3)

# ---- 2. 性能：2048x1024 合成 DEM（多尺度噪声 + 人为洼地）---------------------------------
rng = np.random.default_rng(6)
ny, nx = 1024, 2048
big = np.zeros((ny, nx))
for sigma, amp in ((64, 3000), (16, 800), (4, 200), (1, 30)):
    big += amp * ndi.gaussian_filter(rng.standard_normal((ny, nx)), sigma, mode="wrap") * sigma
big -= big.min()
summary["perf_dem"] = {"shape": [ny, nx], "range_m": [round(float(big.min())), round(float(big.max()))]}

if available_gb() > 2.0:
    with rec.measure("richdem：2048x1024 填洼 + D8 汇流") as e:
        a1 = richdem_area(big, 20000.0)
        e["n_cells"] = int(big.size)
    with rec.measure("pysheds：2048x1024 填洼 + 平地处理 + D8 汇流") as e:
        a2 = pysheds_area(big, 20000.0, "synthetic2048")
    with rec.measure("Fastscape 内核：2048x1024 单流向汇流（含其内部洼地处理）") as e:
        a3 = fastscape_area(big, 20000.0)
    with rec.measure("2048x1024 三者 log 汇水面积相关") as e:
        c = (slice(2, -2), slice(2, -2))
        e["corr_rd_ps"] = round(float(np.corrcoef(np.log10(a1[c]).ravel(), np.log10(a2[c]).ravel())[0, 1]), 4)
        e["corr_rd_fs"] = round(float(np.corrcoef(np.log10(a1[c]).ravel(), np.log10(a3[c]).ravel())[0, 1]), 4)
    # 解释低相关：合成 DEM 有大片封闭洼地，填平后成为大面积平地，平地上的流向各算法规则不同
    with rec.measure("诊断：被填洼抬高的格点比例，以及仅在未填格点上的相关") as e:
        r = rd.rdarray(big.astype("float64"), no_data=-9999.0)
        r.geotransform = [0.0, 20000.0, 0.0, 0.0, 0.0, -20000.0]
        filled = np.asarray(quiet(rd.FillDepressions, r, epsilon=False, in_place=False))
        raised = (filled - big) > 1e-6
        e["filled_fraction"] = round(float(raised.mean()), 3)
        keep = ~raised[c]
        e["corr_rd_ps_unfilled_only"] = round(float(np.corrcoef(np.log10(a1[c][keep]), np.log10(a2[c][keep]))[0, 1]), 4)
        e["corr_rd_fs_unfilled_only"] = round(float(np.corrcoef(np.log10(a1[c][keep]), np.log10(a3[c][keep]))[0, 1]), 4)
        e["corr_rd_ps_filled_only"] = round(float(np.corrcoef(np.log10(a1[c][~keep]), np.log10(a2[c][~keep]))[0, 1]), 4)
else:
    rec.skip("2048x1024 性能测试", f"可用内存 {available_gb():.1f} GB < 2 GB")

# ---- 出图 --------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ext = (0, z.shape[1] * DX / 1e3, 0, z.shape[0] * DX / 1e3)
for ax, a, title in zip(axes, (a_rd, a_ps, a_fs), ("richdem (FillDepressions + D8)",
                                                  "pysheds (fill + resolve_flats + D8)",
                                                  "Fastscape kernel (reference)")):
    im = ax.imshow(np.log10(a), extent=ext, cmap="Blues", vmin=5.4, vmax=9.5)
    ax.set_title(f"{title}\nmax drainage area {a.max() / 1e6:.0f} km²", fontsize=9)
    ax.set_xlabel("km")
fig.colorbar(im, ax=axes, shrink=0.75, label="log10 drainage area (m²)")
fig.suptitle("Spike 06 - flow accumulation on the steady-state Fastscape DEM (256x256 @ 500 m)", fontsize=10)
savefig(fig, "06_hydrology_compare.png")
plt.close(fig)

rec.save(**summary)
