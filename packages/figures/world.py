"""世界数据的统一入口：标准出图（scripts/make_figures.py）与指标计算都从这里取"一个世界"。

两种来源：
- 地球：data/earth_reference/earth_grid_<分辨率>.nc（scripts/build_earth_reference.py 生成）；
- 虚构世界：一个 netCDF 文件，至少含 elevation（m，形状 (ny, nx)，nx = 2·ny，第 0 行在北、第 0 列从 180°W 起），
  可选 land（0/1 或陆地比例）。全局属性：world_id、seed、method_class_elevation（必填）、code_version、source。
  没有 land 变量时，陆地 = 高程 > 0。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from mythic.metrics.grid import LatLonGrid

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


def code_version() -> str:
    """当前代码版本：Git 短提交号；有未提交的已跟踪文件改动时加 "-dirty"。"""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
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
    with xr.open_dataset(path) as ds:
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
    with xr.open_dataset(path) as ds:
        elev = ds["elevation"].values.astype("float64")
        land = (ds["land"].values >= 0.5) if "land" in ds else elev > 0
        attrs = dict(ds.attrs)
    mc = attrs.get("method_class_elevation")
    if not mc:
        raise ValueError(f"{path} 缺少全局属性 method_class_elevation（CLAUDE.md 第 4 节：每个图层必须标注）")
    return World(world_id=str(attrs.get("world_id", path.stem)), display_name=str(attrs.get("display_name", path.stem)),
                 elevation=elev, land=land, method_class=mc, source=str(attrs.get("source", path.name)),
                 seed=str(attrs.get("seed", "—")), code_version=str(attrs.get("code_version", code_version())))


def load(spec: str) -> World:
    """"earth" / "earth:1024x512" 读地球格网；其他视为 netCDF 路径。"""
    if spec == "earth" or spec.startswith("earth:"):
        return load_earth(spec.split(":", 1)[1] if ":" in spec else "2048x1024")
    return load_world_nc(spec)
