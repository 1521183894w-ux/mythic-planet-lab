"""B 层（B1 海岸复杂度、B2 距海距离）的合格判定：与地球大陆级陆块本体同类比较。

U11（2026-10-21 用户同意，依据 reviews/stage0_calibration.md §5 R1）：
- 地球参照与虚构世界用**同一种切分**：陆块单元（连通本体，同样的细线连接规则），不用含近岸岛屿的"大洲"；
- 参与判定的陆块：面积 ≥ unit_min_area_km2 的陆块单元（地球：亚非欧、美洲、澳大利亚、南极洲）；
- 合格线：每一项不低于地球这些陆块中的最小值（"地球上最平滑的大陆级陆块"）；距海距离类（越小越湿润）
  不高于其中的最大值；
- 海岸发育系数随陆块面积增大，只作参考，不参与判定；
- 地球亚非欧本体的数值作为主大陆的"目标"，用于评分，不作硬门槛。

地球参照由 scripts/compute_earth_baselines.py 写入 baselines_<分辨率>.json 的 landmass_unit_reference。
"""

from __future__ import annotations

import numpy as np

KEYS = ["fractal_dimension", "coastline_development", "peninsula_index", "bay_index",
        "land_fraction_beyond_1000km", "distance_to_sea_median_km"]
SMALLER_IS_BETTER = {"land_fraction_beyond_1000km", "distance_to_sea_median_km"}


def flat(region: dict) -> dict:
    """一个区域的 B1/B2 数值展开为 {指标@半径: 值}。"""
    out = {}
    for key in KEYS:
        v = region[key]
        if isinstance(v, dict):
            for r in sorted(v, key=float):
                out[f"{key}@{r}"] = v[r]
        else:
            out[key] = v
    return out


def base_key(k: str) -> str:
    return k.split("@")[0]


def unit_regions(by_region: dict, min_area_km2: float) -> dict:
    """compute_all 的 by_region 中按陆块单元命名、面积达标的区域（main_continent、unit_k）。"""
    return {k: v for k, v in by_region.items()
            if (k == "main_continent" or k.startswith("unit_")) and v["area_km2"] >= min_area_km2}


def earth_unit_reference(by_region_units: dict, min_area_km2: float, names: dict | None = None) -> dict:
    """由地球按陆块单元切分的 by_region 构造参照：各陆块数值、合格线（最小值/最大值）与主大陆目标。"""
    units = unit_regions(by_region_units, min_area_km2)
    vals = {k: flat(v) for k, v in units.items()}
    keys = list(next(iter(vals.values())).keys())
    lines = {}
    for k in keys:
        col = {u: vals[u][k] for u in vals if vals[u][k] is not None and np.isfinite(vals[u][k])}
        pick = max if base_key(k) in SMALLER_IS_BETTER else min
        u = pick(col, key=col.get)
        lines[k] = {"value": col[u], "unit": u, "direction": "max" if base_key(k) in SMALLER_IS_BETTER else "min"}
    return {"description": "地球陆块单元本体（与虚构世界同一切分方式）的 B1/B2 数值与合格线（U11）",
            "unit_min_area_km2": min_area_km2,
            "units": {u: {"name": (names or {}).get(u, u), "area_km2": units[u]["area_km2"], **vals[u]} for u in units},
            "pass_line": lines, "target_main_continent": vals.get("main_continent")}


def b_verdict(by_region: dict, reference: dict, cfg: dict) -> dict:
    """世界各大陆级陆块的 B1/B2 判定（by_region 为 compute_all 的输出）。"""
    units = unit_regions(by_region, cfg["B1_coast"]["verdict"]["unit_min_area_km2"])
    return b_verdict_flat({u: {"area_km2": r["area_km2"], **flat(r)} for u, r in units.items()}, reference, cfg)


def b_verdict_flat(units: dict, reference: dict, cfg: dict) -> dict:
    """units = {陆块: {area_km2, 指标@半径: 值, ...}}。返回各陆块各指标的 {value, line, pass|None} 与汇总。
    reference_only 中的指标（海岸发育系数）pass 为 None，不计入判定。"""
    vcfg = cfg["B1_coast"]["verdict"]
    ref_only = set(vcfg.get("reference_only", []))
    out = {}
    for u, region in units.items():
        rows = {}
        for k, v in region.items():
            if k in ("area_km2", "name"):
                continue
            line = reference["pass_line"][k]
            if base_key(k) in ref_only or v is None or not np.isfinite(v):
                ok = None
            elif line["direction"] == "max":
                ok = bool(v <= line["value"])
            else:
                ok = bool(v >= line["value"])
            rows[k] = {"value": v, "line": line["value"], "line_unit": line["unit"], "direction": line["direction"], "pass": ok}
        judged = [r["pass"] for r in rows.values() if r["pass"] is not None]
        out[u] = {"area_km2": region["area_km2"], "name": region.get("name", u), "metrics": rows, "n_judged": len(judged),
                  "n_pass": int(sum(judged)), "all_pass": bool(all(judged)) if judged else False}
    main = out.get("main_continent")
    return {"method": vcfg["method"], "units": out,
            "main_continent_all_pass": bool(main["all_pass"]) if main else False,
            "all_units_pass": bool(all(x["all_pass"] for x in out.values())) if out else False}
