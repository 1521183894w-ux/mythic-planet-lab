"""阶段 0.7 尺子校准的指标对比：地球 vs 对照样本（圆润世界、噪声海岸世界；2048x1024、1024x512）。
输出 reviews/stage0_calibration_data.json。

内容：
- 两个世界都按"陆块单元"分区（虚构世界的标准处理；地球也这样算，才能同类比较）的 A 层、B1/B2 分区数值、全球 B1/B2/C 摘要；
- 地球按"大洲"分区的现行 B1 范围（基准文件 earth_ranges_over_continents）下，各主大陆的判定；
- PROGRESS U11 的两种候选比较方式的试算（只是试算，不改 config/metrics.yaml）：
  方式 1：主大陆不低于地球"海岸复杂"的参照（亚非欧、欧洲、亚洲、北美，大洲口径）中的最小值；
  方式 3（同类比较）：主大陆不低于地球大陆级陆块本体（陆块单元口径）中的最小值；
- 正式的 B1/B2 判定（b_verdict，U11 采纳后的同类比较）；
- 高程分布的 Wasserstein-1 距离（每个对照样本 vs 地球，按世界名分键）。

前置：scripts/make_rounded_world.py、scripts/make_noise_world.py 已生成 data/worlds/rounded_control/、noise_control/
（缺哪个就跳过哪个）。
用法：powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\calibration_metrics.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from mythic.figures.world import EARTH_RANGE_UNITS, RANGE_KEYS, ROOT, code_version, load_earth, load_world_nc, to_jsonable
from mythic.metrics.compute import compute_all, load_config
from mythic.metrics.distributions import wasserstein
from mythic.metrics.verdict import b_verdict

COMPLEX_REFS = ["Afro-Eurasia", "Europe", "Asia", "North America"]
SMALLER_IS_BETTER = {"land_fraction_beyond_1000km", "distance_to_sea_median_km"}
RESOLUTIONS = ["2048x1024", "1024x512"]
CONTROLS = {"rounded": "data/worlds/rounded_control/rounded_control_{res}.nc",   # 第一个对照样本
            "noise": "data/worlds/noise_control/noise_control_{res}.nc"}         # 第二个对照样本（噪声海岸）


def flat(region: dict) -> dict:
    out = {}
    for key in RANGE_KEYS:
        v = region[key]
        if isinstance(v, dict):
            for r in sorted(v, key=float):
                out[f"{key}@{r}"] = v[r]
        else:
            out[key] = v
    return out


def summarize(m: dict, cfg: dict) -> dict:
    A = m["A"]
    mis = A["main_inland_sea"]
    units = {k: v for k, v in m["by_region"].items() if k == "main_continent" or k.startswith("unit_")}
    return {
        "A_pass": {k: bool(v["pass"]) for k, v in A.items()},
        "A_values": {"ocean_fraction": A["ocean_fraction"]["value"], "main_share_of_land": A["main_continent"]["share_of_land"],
                     "subcontinents": A["subcontinents"]["count"], "subcontinent_areas_Mkm2": [a / 1e6 for a in A["subcontinents"]["areas_km2"]],
                     "inland_sea_Mkm2": mis["main"]["area_km2"] / 1e6 if mis.get("exists") else None,
                     "inland_sea_straits": mis["main"]["straits_to_world_ocean"] if mis.get("exists") else None,
                     "inland_sea_checks": mis.get("checks"),
                     "inland_sea_aspect_ratio": mis.get("geometry", {}).get("aspect_ratio"),
                     "inland_sea_bordering_subcontinents": mis.get("geometry", {}).get("n_bordering_subcontinents"),
                     "inland_sea_border_shares": [s["border_share"] for s in mis.get("geometry", {}).get("subcontinent_border_shares", [])],
                     "north_deg": A["main_continent_latitude"]["north_deg"], "south_deg": A["main_continent_latitude"]["south_deg"],
                     "polar_continents": len(A["polar_continent"]["found"]), "open_ocean_diameter_km": A["open_ocean"]["land_free_circle_diameter_km"]},
        "units": {k: {"area_Mkm2": v["area_km2"] / 1e6, **flat(v)} for k, v in units.items()},
        "all_land": flat(m["by_region"]["all_land"]),
        "global": {"fractal_D": m["B1"]["fractal_dimension"]["D"], "coastline_km": m["B1"]["coastline_length_km"],
                   "islands": m["B1"]["islands"]["n_islands"], "korcak_b": m["B1"]["islands"]["korcak"]["b"],
                   "shelf_fraction": m["B1"]["shelf_fraction_of_ocean"], "B2": m["B2"],
                   "coast_complexity_spread": m["B1"].get("coast_complexity_spread"),
                   "mountains": {k: m["C"]["mountains"][k] for k in (
                       "n_systems", "n_longer_than_min_length", "mountain_land_fraction", "longest_km", "top3_length_km",
                       "area_weighted_aspect_ratio", "elongated_area_fraction", "low_tier")},
                   "mountain_top8": [{k: s[k] for k in ("length_km", "mean_width_km", "sinuosity", "aspect_ratio", "orientation_deg")}
                                     for s in m["C"]["mountains"]["systems"][:8]],
                   "slope_land_quantiles_deg": m["C"]["slope"]["quantiles_land_deg"],
                   "elevation_land_quantiles_m": m["C"]["hypsometry"]["quantiles_land_m"]},
    }


def verdicts(main: dict, continents: dict, earth_units: dict) -> dict:
    out = {}
    big_earth_units = [u for u in earth_units.values() if u["area_Mkm2"] >= 5.0]
    for k, v in main.items():
        if k == "area_Mkm2":
            continue
        cont = [continents[u][k] for u in EARTH_RANGE_UNITS]
        cx = [continents[u][k] for u in COMPLEX_REFS]
        units = [u[k] for u in big_earth_units]
        if k in SMALLER_IS_BETTER:
            o1_ref, o3_ref = max(cx), max(units)
            o1, o3 = v <= o1_ref, v <= o3_ref
        else:
            o1_ref, o3_ref = min(cx), min(units)
            o1, o3 = v >= o1_ref, v >= o3_ref
        out[k] = {"value": v, "current_range": [min(cont), max(cont)], "current_in_range": min(cont) <= v <= max(cont),
                  "opt1_ref": o1_ref, "opt1_pass": bool(o1), "opt3_ref": o3_ref, "opt3_pass": bool(o3)}
    return out


def main() -> None:
    cfg = load_config()
    result = {"description": "阶段 0.7 尺子校准的指标对比（reviews/stage0_calibration.md）", "code_version": code_version(),
              "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "resolutions": {}}
    for res in RESOLUTIONS:
        base = json.loads((ROOT / f"data/earth_reference/baselines_{res}.json").read_text(encoding="utf-8"))
        continents = {u: flat(base["by_region"][u]) for u in EARTH_RANGE_UNITS}
        ew = load_earth(res)
        # 所有世界一视同仁：按陆块单元分区；地球的山系统计照基准剔除冰盖（对照样本没有冰盖，不受影响）
        em = compute_all(ew.elevation, ew.land, cfg=cfg, mountain_exclude=ew.mountain_exclude)
        es = summarize(em, cfg)
        reference = base["landmass_unit_reference"]
        r = {"earth_continents_current_reference": continents, "earth_as_world": es,
             "verdicts_main_continent": {"earth": verdicts(es["units"]["main_continent"], continents, es["units"])},
             # 正式判定（U11 采纳后，config/metrics.yaml B1_coast.verdict；实现 packages/metrics/verdict.py）
             "b_verdict": {"earth": b_verdict(em["by_region"], reference, cfg)},
             "wasserstein_elevation_m": {}}
        summaries = {"earth": es}
        for name, path in CONTROLS.items():
            p = ROOT / path.format(res=res)
            if not p.exists():
                continue
            w = load_world_nc(p)
            m = compute_all(w.elevation, w.land, cfg=cfg)
            s = summarize(m, cfg)
            summaries[name] = r[name] = s
            r["verdicts_main_continent"][name] = verdicts(s["units"]["main_continent"], continents, es["units"])
            r["b_verdict"][name] = b_verdict(m["by_region"], reference, cfg)
            a_w, a_e = w.grid.cell_area_km2, ew.grid.cell_area_km2
            r["wasserstein_elevation_m"][name] = {
                "global": wasserstein(w.elevation, a_w, ew.elevation, a_e),
                "land": wasserstein(w.elevation[w.land], a_w[w.land], ew.elevation[ew.land], a_e[ew.land])}
        result["resolutions"][res] = r
        v = r["verdicts_main_continent"]
        bv = r["b_verdict"]
        for name, s in summaries.items():
            n = len(v[name])
            mt = s["global"]["mountains"]
            print(f"[{res}] {name}: A 全部达标={all(s['A_pass'].values())}；主大陆 B1/B2 {n} 项中 "
                  f"旧范围内 {sum(d['current_in_range'] for d in v[name].values())}，"
                  f"方式 1 通过 {sum(d['opt1_pass'] for d in v[name].values())}，"
                  f"方式 3 通过 {sum(d['opt3_pass'] for d in v[name].values())}；正式判定 "
                  + "、".join(f"{u} {x['n_pass']}/{x['n_judged']}" for u, x in bv[name]["units"].items())
                  + f"；山系 最长 {mt['longest_km']:.0f} km，加权长宽比 {mt['area_weighted_aspect_ratio']:.2f}，"
                  f"成带面积占比 {mt['elongated_area_fraction']:.2f}"
                  f"；主内海 长宽比 {s['A_values']['inland_sea_aspect_ratio'] or float('nan'):.2f}、"
                  f"接壤亚大陆 {s['A_values']['inland_sea_bordering_subcontinents']}"
                  f"；海岸复杂度 CV {s['global']['coast_complexity_spread']['cv']:.2f}、"
                  f"较平直占比 {s['global']['coast_complexity_spread']['straight_fraction']:.2f}", flush=True)
    out = ROOT / "reviews" / "stage0_calibration_data.json"
    out.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"写出 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
