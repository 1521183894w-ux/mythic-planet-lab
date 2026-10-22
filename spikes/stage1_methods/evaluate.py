"""原型结果的"命中"判定：残留洋盆有没有形成主内海。直接调用阶段 0 的 A 层代码（mythic.metrics.hard），不另写判据。

两档：
- official：按 config/metrics.yaml 的主内海定义（海峡宽 < 150 km、面积 100–500 万 km²、1–3 条海峡通外洋、
  至少是第二大内部水体的 2 倍、主轴长宽比 ≥ 2），"接壤 ≥ 2 个亚大陆"在这里换成"两岸分属两块初始大陆（各占沿岸 ≥ 15%）"，
  因为原型只有两块大陆，亚大陆切分没有意义；
- loose：把海峡宽度放宽到 300 km、面积下限放宽到 50 万 km²，只问"两块大陆之间是否留下了一个半封闭的残留洋盆"。
"""

from __future__ import annotations

import numpy as np
from mythic.metrics.compute import load_config
from mythic.metrics.earthlike import shape_axes
from mythic.metrics.hard import inland_seas, landmass_units, main_inland_sea_check
from mythic.metrics.topology import any_neighbor

_CFG = load_config()["A_hard_constraints"]


def basin_check(grid, land: np.ndarray, origin: np.ndarray) -> dict:
    """land：陆地掩膜；origin：地体来源（1 = 北大陆，2 = 南大陆，90 = 洋内弧，0 = 洋壳）。"""
    area = grid.cell_area_km2
    units = landmass_units(grid, land, _CFG["main_continent"]["thin_link_width_km"])
    main = units["main_mask"]
    out = {"land_area_Mkm2": float((area * land).sum() / 1e6), "main_area_Mkm2": float((area * main).sum() / 1e6),
           "main_has_both_continents": bool(((origin == 1) & main).any() and ((origin == 2) & main).any())}
    cfg = _CFG["main_inland_sea"]
    for label, strait, min_area in (("official", cfg["strait_max_width_km"], cfg["area_km2"]["target"][0]),
                                    ("loose", 300.0, 0.5e6)):
        seas = inland_seas(grid, land, main, strait, cfg["min_main_border_fraction"], cfg["min_candidate_area_km2"])
        check = main_inland_sea_check(seas["seas"], {**cfg, "area_km2": {"target": [min_area, cfg["area_km2"]["target"][1]]}})
        r = {"exists": bool(check.get("exists"))}
        if check.get("exists"):
            sea = seas["region_labels"] == check["main"]["region_id"]
            bw = area * (land & any_neighbor(sea))
            tot = float(bw.sum())
            s1 = float((bw * (origin == 1)).sum() / tot) if tot else 0.0
            s2 = float((bw * (origin == 2)).sum() / tot) if tot else 0.0
            asp = float(shape_axes(grid, sea)["aspect_ratio"])
            checks = dict(check["checks"])
            checks["two_continents_ok"] = s1 >= 0.15 and s2 >= 0.15
            checks["elongation_ok"] = asp >= cfg["min_aspect_ratio"]
            if label == "loose":   # loose 只看"存在、两岸分属两块大陆、面积 ≥ 50 万 km²"
                checks = {k: checks[k] for k in ("area_in_range", "two_continents_ok")}
                checks["area_in_range"] = check["main"]["area_km2"] >= min_area
            r.update({"area_Mkm2": check["main"]["area_km2"] / 1e6, "straits_to_ocean": check["main"]["straits_to_world_ocean"],
                      "aspect_ratio": asp, "border_share_C1": s1, "border_share_C2": s2,
                      "centroid_lat_lon": check["main"]["sample_lat_lon"], "checks": checks,
                      "hit": bool(all(checks.values())), "_mask": sea})
        else:
            r.update({"hit": False, "reason": check.get("reason", "")})
        out[label] = r
    return out
