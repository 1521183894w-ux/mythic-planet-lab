"""世界数据的统一入口：标准出图（scripts/make_figures.py）与指标计算都从这里取"一个世界"。

两种来源：
- 地球：data/earth_reference/earth_grid_<分辨率>.nc（scripts/build_earth_reference.py 生成）；
- 虚构世界：一个 netCDF 文件，至少含 elevation（m，形状 (ny, nx)，nx = 2·ny，第 0 行在北、第 0 列从 180°W 起），
  可选 land（0/1 或陆地比例）。全局属性：world_id、seed、method_class_elevation（必填）、code_version、source。
  没有 land 变量时，陆地 = 高程 > 0。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from mythic.metrics.grid import LatLonGrid
from .netcdf import open_netcdf

ROOT = Path(__file__).resolve().parents[2]
EARTH_REF = ROOT / "data" / "earth_reference"

METHOD_CLASSES = {"numerical_physics", "parameterized_physics", "procedural_constrained", "interpolated",
                  "visual_only", "user_override", "fantasy_exception"}

# ---- 地球专用的输入处理（与 scripts/compute_earth_baselines.py 共用，基准数值依赖它们，修改需同步重算基准）----
EARTH_REGION_NAMES = {1: "Africa", 2: "Europe", 3: "Asia", 4: "North America", 5: "South America", 6: "Oceania",
                      7: "Antarctica"}
EARTH_REGION_GROUPS = {"Afro-Eurasia": [1, 2, 3], "Americas": [4, 5]}
# "地球合理范围"取哪些区域；南极洲（冰架/冰崖海岸）单独列出但不计入范围
EARTH_RANGE_UNITS = ["Afro-Eurasia", "Europe", "Asia", "Africa", "North America", "South America", "Oceania"]


RANGE_KEYS = ["fractal_dimension", "coastline_development", "peninsula_index", "bay_index",
              "land_fraction_beyond_1000km", "distance_to_sea_median_km"]


def earth_ranges(by_region: dict, units: list[str] = EARTH_RANGE_UNITS) -> dict:
    """各指标在地球各大洲（units）上的 [最小, 最大] 及对应大洲。按半径分的指标逐个半径给出。"""
    out = {}
    for key in RANGE_KEYS:
        vals = {u: by_region[u][key] for u in units if u in by_region}
        if isinstance(next(iter(vals.values())), dict):  # 按半径分的指标
            radii = next(iter(vals.values())).keys()
            out[key] = {r: {"min": min(v[r] for v in vals.values()), "max": max(v[r] for v in vals.values()),
                            "min_unit": min(vals, key=lambda u: vals[u][r]), "max_unit": max(vals, key=lambda u: vals[u][r])}
                        for r in radii}
        else:
            finite = {u: v for u, v in vals.items() if v is not None and np.isfinite(v)}
            out[key] = {"min": min(finite.values()), "max": max(finite.values()),
                        "min_unit": min(finite, key=finite.get), "max_unit": max(finite, key=finite.get)}
    return out


def to_jsonable(x):
    """把含 numpy 类型的结果转成可写 JSON 的结构；以 "_" 开头的键（内部数据）丢弃，非有限浮点数写 null。"""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def earth_ice_mask(continent_id: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """山系统计要剔除的冰盖：南极洲全部、格陵兰（北美编号中 59°N 以北、75°W–10°W）。
    ETOPO 给的是冰面高程，冰穹不是山。"""
    lat2 = np.broadcast_to(np.asarray(lat)[:, None], continent_id.shape)
    lon2 = np.broadcast_to(np.asarray(lon)[None, :], continent_id.shape)
    return (continent_id == 7) | ((continent_id == 4) & (lat2 > 59) & (lon2 > -75) & (lon2 < -10))


CODE_PATHS = ["packages", "scripts", "config", "pyproject.toml", "environment.yml", "environment.win-64.lock.txt"]


def code_version() -> str:
    """当前代码版本：Git 短提交号；代码、配置或环境文件（CODE_PATHS）有未提交的改动或未跟踪的新文件时加 "-dirty"。
    数据、文档、评审报告的改动不算：它们不影响计算结果，而且重算数据本身就会改写已跟踪的数据文件
    （例如先重算地球基准，后面的脚本就会误报 dirty）。未跟踪的新脚本算 dirty：0.7 曾出现过用未提交的脚本生成数据、
    版本号却显示干净的情况。"""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class World:
    world_id: str
    display_name: str
    elevation: np.ndarray            # m，(ny, nx)
    land: np.ndarray                 # bool
    method_class: str                # 高程图层的 method_class
    source: str                      # 数据来源（一行）
    seed: str = "—"
    code_version: str = field(default_factory=code_version)
    regions: np.ndarray | None = None          # 分区域统计用的标签（地球 = 大洲）；None 时按陆块单元
    region_names: dict | None = None
    region_groups: dict | None = None
    mountain_exclude: np.ndarray | None = None
    is_earth: bool = False
    method_label_override: str | None = None   # 只用于盲评时在图中隐去 method_class；真实值仍在 method_class
    tectonics: dict | None = None              # U34：可选的构造图层（世界文件里的 tect_* 变量：名 -> (数组, 属性)）

    @property
    def method_label(self) -> str:
        """图中显示的 method_class（盲评时为"隐去"说明）。"""
        return self.method_label_override or self.method_class

    def __post_init__(self):
        if self.method_class not in METHOD_CLASSES:
            raise ValueError(f"未知的 method_class：{self.method_class}（可选：{sorted(METHOD_CLASSES)}）")
        if self.elevation.shape != self.land.shape:
            raise ValueError("elevation 与 land 形状不一致")

    @property
    def grid(self) -> LatLonGrid:
        return LatLonGrid.from_shape(self.elevation.shape)

    @property
    def resolution(self) -> str:
        return self.grid.name

    def compute_kwargs(self) -> dict:
        """传给 mythic.metrics.compute.compute_all 的世界相关参数。"""
        return {"regions": self.regions, "region_names": self.region_names, "region_groups": self.region_groups,
                "mountain_exclude": self.mountain_exclude}


def load_earth(resolution: str = "2048x1024") -> World:
    path = EARTH_REF / f"earth_grid_{resolution}.nc"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行 scripts/download_earth_data.py 与 scripts/build_earth_reference.py")
    with open_netcdf(path) as ds:
        elev = ds.elevation.values.astype("float64")
        land = ds.land_fraction.values >= 0.5
        cont = ds.continent_id.values.astype(int)
        lat, lon = ds.lat.values, ds.lon.values
    return World(
        world_id=f"earth_{resolution}", display_name="地球", elevation=elev, land=land,
        method_class="interpolated",
        source=f"ETOPO 2022 60″（doi:10.25921/fd45-gt74）面积平均重采样到 {resolution}；海陆 = Natural Earth 10m 陆地比例 ≥ 0.5",
        seed="—（真实地球）", regions=cont, region_names=EARTH_REGION_NAMES, region_groups=EARTH_REGION_GROUPS,
        mountain_exclude=earth_ice_mask(cont, lat, lon), is_earth=True)


def load_world_nc(path: str | Path) -> World:
    path = Path(path)
    with open_netcdf(path) as ds:
        elev = ds["elevation"].values.astype("float64")
        land = (ds["land"].values >= 0.5) if "land" in ds else elev > 0
        attrs = dict(ds.attrs)
        tect = {k: (ds[k].values, dict(ds[k].attrs)) for k in ds.data_vars if str(k).startswith("tect_")} or None
    mc = attrs.get("method_class_elevation")
    if not mc:
        raise ValueError(f"{path} 缺少全局属性 method_class_elevation（CLAUDE.md 第 4 节：每个图层必须标注）")
    return World(world_id=str(attrs.get("world_id", path.stem)), display_name=str(attrs.get("display_name", path.stem)),
                 elevation=elev, land=land, method_class=mc, source=str(attrs.get("source", path.name)),
                 seed=str(attrs.get("seed", "—")), code_version=str(attrs.get("code_version", code_version())),
                 tectonics=tect)


BLIND_NOTE = "（盲评隐去，评审后在 reviews/stage0_calibration.md 公开）"


def mirrored(world: World, shift_deg: float) -> World:
    """东西镜像（经度 λ → −λ）后向东平移 shift_deg（取整到格点）。这是球面上的保距变换，
    纬度不变，所以所有指标（面积、距离、形态学、纬度判据）都不变；只用于盲评时降低"一眼认出地图"的可能。"""
    nx = world.grid.nx
    k = int(round(shift_deg / 360.0 * nx))

    def f(a):
        return None if a is None else np.roll(a[:, ::-1], k, axis=1)

    return replace(world, elevation=f(world.elevation), land=f(world.land), regions=f(world.regions),
                   mountain_exclude=f(world.mountain_exclude), tectonics=None,
                   source=f"{world.source}；东西镜像后向东平移 {k * 360.0 / nx:.2f}°")


def blinded(world: World, name: str, world_id: str) -> World:
    """盲评版：隐去名称、来源、种子与 method_class，按陆块单元（而不是地球大洲）分区统计。
    山系统计的冰盖剔除（地球）保留，因为它是输入处理，不是可见信息。"""
    return replace(world, world_id=world_id, display_name=name, seed=BLIND_NOTE, source=BLIND_NOTE,
                   method_label_override=BLIND_NOTE, is_earth=False, regions=None, region_names=None,
                   region_groups=None)


def load(spec: str) -> World:
    """"earth" / "earth:1024x512" 读地球格网；其他视为 netCDF 路径。"""
    if spec == "earth" or spec.startswith("earth:"):
        return load_earth(spec.split(":", 1)[1] if ":" in spec else "2048x1024")
    return load_world_nc(spec)
