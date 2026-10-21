"""项目环境自检。

检查三类内容：
1. 关键包能否导入，记录版本；
2. 地理空间栈（GDAL / PROJ / rasterio / cartopy / matplotlib）的最小冒烟测试；
3. 各类缓存与数据目录是否都在工作区内（CLAUDE.md 第 5 节：不得写 C 盘）。

由 scripts/setup_env.ps1 安装后自动运行，也可单独运行：
    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\check_env.py

结果打印到屏幕并写入 environment/env_report.json；任一项失败时退出码为 1。
各科学工具的功能验证（最小样例）不在这里做，见阶段 0.4 的 docs/tool_evaluation.md。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata as md
import io
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("MYTHIC_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORT = ROOT / "environment" / "env_report.json"

# (导入名, 发行包名)：与 environment.yml 对应
PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("xarray", "xarray"),
    ("zarr", "zarr"),
    ("netCDF4", "netCDF4"),
    ("osgeo.gdal", "GDAL"),
    ("rasterio", "rasterio"),
    ("pyproj", "pyproj"),
    ("shapely", "shapely"),
    ("geopandas", "geopandas"),
    ("cartopy", "cartopy"),
    ("matplotlib", "matplotlib"),
    ("skimage", "scikit-image"),
    ("pygplates", "pygplates"),
    ("gplately", "gplately"),
    ("stripy", "stripy"),
    ("landlab", "landlab"),
    ("fastscape", "fastscape"),
    ("fastscapelib_fortran", "fastscapelib-fortran"),
    ("richdem", "richdem"),
    ("pysheds", "pysheds"),
    ("yaml", "PyYAML"),
    ("jinja2", "Jinja2"),
    ("PIL", "pillow"),
    ("pooch", "pooch"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]

results: list[dict] = []


def record(group: str, name: str, status: str, detail: str = "") -> None:
    results.append({"group": group, "name": name, "status": status, "detail": detail})


def version_of(module, dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return str(getattr(module, "__version__", "?"))


def inside_root(path: str | os.PathLike) -> bool:
    try:
        Path(path).resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def check(group: str, name: str):
    """装饰器：运行一个检查函数，异常记为 fail。函数返回 (status, detail)。"""

    def wrap(func):
        try:
            status, detail = func()
        except Exception as exc:  # noqa: BLE001  自检需要捕获一切异常并继续
            status = "fail"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        record(group, name, status, detail)
        return func

    return wrap


# gplately 导入时按它自带的日志配置在"当前目录"创建 gplately.log 和 ptt.log（文件名写死、不可配置），
# 所以导入它时先切换到 .cache/logs，避免在仓库根目录留下日志文件。
LOGS = ROOT / ".cache" / "logs"
CWD_SENSITIVE = {"gplately"}

# ---- 1. 导入 -------------------------------------------------------------------
for mod_name, dist in PACKAGES:
    t0 = time.perf_counter()
    try:
        if mod_name in CWD_SENSITIVE:
            LOGS.mkdir(parents=True, exist_ok=True)
            with contextlib.chdir(LOGS):
                mod = importlib.import_module(mod_name)
        else:
            mod = importlib.import_module(mod_name)
        record("import", mod_name, "ok", f"{version_of(mod, dist)}  ({time.perf_counter() - t0:.2f}s)")
    except Exception as exc:  # noqa: BLE001
        record("import", mod_name, "fail", f"{type(exc).__name__}: {exc}")


# ---- 2. 地理空间冒烟测试 ---------------------------------------------------------
@check("smoke", "pyproj: EPSG:4326 -> EPSG:3857")
def _pyproj():
    from pyproj import Transformer
    from pyproj.datadir import get_data_dir

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(10.0, 50.0)
    ok = abs(x - 1113194.9) < 1.0 and abs(y - 6446275.8) < 1.0
    return ("ok" if ok else "fail"), f"x={x:.1f}, y={y:.1f}; proj data={get_data_dir()}"


@check("smoke", "GDAL: MEM 数据集 + EPSG:4326 坐标系")
def _gdal():
    from osgeo import gdal, osr

    gdal.UseExceptions()
    ds = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ok = "WGS 84" in ds.GetProjection()
    return ("ok" if ok else "fail"), f"GDAL {gdal.__version__}; GDAL_DATA={gdal.GetConfigOption('GDAL_DATA')}"


@check("smoke", "rasterio: 内存 GeoTIFF 写入/读回")
def _rasterio():
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.arange(64, dtype="float32").reshape(8, 8)
    profile = dict(driver="GTiff", width=8, height=8, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(0, 8, 1, 1))
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        with mem.open() as src:
            back = src.read(1)
            crs = src.crs.to_epsg()
    ok = bool(np.array_equal(back, data)) and crs == 4326
    return ("ok" if ok else "fail"), f"rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}, EPSG={crs}"


@check("smoke", "cartopy: Robinson 投影变换")
def _cartopy():
    import cartopy.crs as ccrs
    import numpy as np

    lon = np.array([-170.0, 0.0, 120.0])
    lat = np.array([-60.0, 0.0, 45.0])
    pts = ccrs.Robinson().transform_points(ccrs.PlateCarree(), lon, lat)
    ok = bool(np.isfinite(pts).all())
    return ("ok" if ok else "fail"), f"points={pts[:, :2].round(0).tolist()}"


@check("smoke", "matplotlib: Agg 后台出图")
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    size = buf.getbuffer().nbytes
    return ("ok" if size > 1000 else "fail"), f"backend={matplotlib.get_backend()}, png={size} bytes"


# ---- 3. 缓存与数据目录位置 -------------------------------------------------------
def location(name: str, path: str | os.PathLike | None, note: str = "") -> None:
    if not path:
        record("location", name, "warn", f"未设置 {note}".strip())
        return
    status = "ok" if inside_root(path) else "fail"
    record("location", name, status, f"{path} {note}".strip())


location("sys.prefix（Python 环境）", sys.prefix)
location("临时目录 tempfile.gettempdir()", tempfile.gettempdir())
location("pip 缓存 PIP_CACHE_DIR", os.environ.get("PIP_CACHE_DIR"))
location("micromamba 包缓存 CONDA_PKGS_DIRS", os.environ.get("CONDA_PKGS_DIRS"))
location("micromamba 根目录 MAMBA_ROOT_PREFIX", os.environ.get("MAMBA_ROOT_PREFIX"))

try:
    import matplotlib

    location("matplotlib 配置目录", matplotlib.get_configdir())
    location("matplotlib 缓存目录", matplotlib.get_cachedir())
except Exception as exc:  # noqa: BLE001
    record("location", "matplotlib", "fail", repr(exc))

try:
    import cartopy

    location("cartopy 数据目录（Natural Earth 下载）", cartopy.config.get("data_dir"))
    location("cartopy 缓存目录", cartopy.config.get("cache_dir"))
except Exception as exc:  # noqa: BLE001
    record("location", "cartopy", "fail", repr(exc))

try:
    from pyproj.datadir import get_user_data_dir

    location("PROJ 用户可写目录", get_user_data_dir())
except Exception as exc:  # noqa: BLE001
    record("location", "pyproj user dir", "fail", repr(exc))

try:
    import numba

    location("numba 缓存目录", numba.config.CACHE_DIR or None, "(NUMBA_CACHE_DIR)")
except Exception as exc:  # noqa: BLE001
    record("location", "numba", "fail", repr(exc))

try:
    import platformdirs

    location("platformdirs 用户缓存目录", platformdirs.user_cache_dir("probe"), "(WIN_PD_OVERRIDE_LOCAL_APPDATA)")
    location("platformdirs 用户数据目录（roaming）", platformdirs.user_data_dir("probe", roaming=True), "(WIN_PD_OVERRIDE_APPDATA)")
except Exception as exc:  # noqa: BLE001
    record("location", "platformdirs", "fail", repr(exc))

try:
    import pooch

    location("pooch 默认缓存（gplately 等）", pooch.os_cache("gplately"))
except Exception as exc:  # noqa: BLE001
    record("location", "pooch", "fail", repr(exc))

stray_logs = sorted(p.name for p in ROOT.glob("*.log"))
record("location", "仓库根目录无日志文件", "fail" if stray_logs else "ok", ", ".join(stray_logs))


# ---- 输出 ---------------------------------------------------------------------
def main() -> int:
    width = max(len(r["name"]) for r in results) + 2
    mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
    current = None
    for r in results:
        if r["group"] != current:
            current = r["group"]
            print(f"\n== {current} ==")
        print(f"  {mark[r['status']]} {r['name']:<{width}} {r['detail']}")

    n_fail = sum(r["status"] == "fail" for r in results)
    n_warn = sum(r["status"] == "warn" for r in results)
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.node(),
        "python": sys.version,
        "prefix": sys.prefix,
        "root": str(ROOT),
        "fail": n_fail,
        "warn": n_warn,
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合计：{len(results)} 项，失败 {n_fail}，警告 {n_warn}。报告：{REPORT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
