"""计算地球基准：对三级格网运行与虚构世界相同的指标代码，输出 data/earth_reference/baselines_<分辨率>.json。

前置：scripts/build_earth_reference.py 已生成 data/earth_reference/earth_grid_*.nc。
用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\compute_earth_baselines.py

地球专用的输入处理（虚构世界不需要）：
- 陆地 = Natural Earth 陆地比例 ≥ 0.5（理由见 build_earth_reference.py）；
- 区域 = 大洲编号；组合区域：亚非欧（Afro-Eurasia）、美洲（Americas）；
- 山系统计剔除冰盖：南极洲全部、格陵兰（北美编号中 59°N 以北、75°W–10°W）。ETOPO 给的是冰面高程，冰穹不是山。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from mythic.metrics.compute import DEFAULT_CONFIG, compute_all, load_config

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "earth_reference"
RESOLUTIONS = ["512x256", "1024x512", "2048x1024"]
NAMES = {1: "Africa", 2: "Europe", 3: "Asia", 4: "North America", 5: "South America", 6: "Oceania", 7: "Antarctica"}
GROUPS = {"Afro-Eurasia": [1, 2, 3], "Americas": [4, 5]}
# 用于"地球合理范围"的区域；南极洲（冰架/冰崖海岸）单独列出但不计入范围
RANGE_UNITS = ["Afro-Eurasia", "Europe", "Asia", "Africa", "North America", "South America", "Oceania"]
RANGE_KEYS = ["fractal_dimension", "coastline_development", "peninsula_index", "bay_index",
              "land_fraction_beyond_1000km", "distance_to_sea_median_km"]


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def ranges(by_region: dict) -> dict:
    out = {}
    for key in RANGE_KEYS:
        vals = {u: by_region[u][key] for u in RANGE_UNITS if u in by_region}
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


def main() -> None:
    cfg = load_config()
    cfg_text = Path(DEFAULT_CONFIG).read_bytes()
    manifests = {p.name: yaml.safe_load(p.read_text(encoding="utf-8"))
                 for p in sorted((ROOT / "data" / "external_manifests").glob("*.yaml"))}
    for res in RESOLUTIONS:
        t0 = time.perf_counter()
        ds = xr.open_dataset(REF / f"earth_grid_{res}.nc")
        elev = ds.elevation.values.astype("float64")
        land = ds.land_fraction.values >= 0.5
        cont = ds.continent_id.values.astype(int)
        lat2 = np.broadcast_to(ds.lat.values[:, None], cont.shape)
        lon2 = np.broadcast_to(ds.lon.values[None, :], cont.shape)
        ice = (cont == 7) | ((cont == 4) & (lat2 > 59) & (lon2 > -75) & (lon2 < -10))
        result = compute_all(elev, land, cont, NAMES, GROUPS, cfg, mountain_exclude=ice)
        land_etopo = ds.land_fraction_etopo.values >= 0.5
        from mythic.metrics.grid import LatLonGrid
        g = LatLonGrid.from_shape(land.shape)
        a = g.cell_area_km2
        payload = {
            "description": "地球基准：与虚构世界相同的指标代码在同一格网上计算（docs/quality_metrics.md §F）",
            "grid": res,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code_commit": git_commit(),
            "config": {"path": "config/metrics.yaml", "version": cfg.get("version"),
                       "sha256": hashlib.sha256(cfg_text).hexdigest()},
            "inputs": {
                "grid_file": f"data/earth_reference/earth_grid_{res}.nc (scripts/build_earth_reference.py)",
                "land_mask": "Natural Earth 10m land fraction >= 0.5",
                "regions": "Natural Earth admin-0 CONTINENT（俄罗斯按 60°E 划分欧亚）",
                "mountain_exclusion": "南极洲 + 格陵兰（59°N 以北、75°W–10°W）冰盖",
                "manifests": {k: {f["name"]: f.get("sha256") for f in v.get("files", [])} for k, v in manifests.items()},
            },
            "sensitivity": {
                "ocean_fraction_if_land_is_etopo_elevation_gt_0": float(1 - (a * land_etopo).sum() / a.sum()),
            },
            "earth_ranges_over_continents": {"units": RANGE_UNITS, "values": ranges(result["by_region"])},
            **result,
        }
        out = REF / f"baselines_{res}.json"
        out.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"[{res}] {time.perf_counter() - t0:.1f}s -> {out.name} ({out.stat().st_size / 1e3:.0f} KB); timings {result['timings_s']}",
              flush=True)


if __name__ == "__main__":
    main()
