"""U35-C6：捕获独立 C4 最终状态，并验证直接原生单元输出候选。

不改变物理更新或默认世界。--capture 完整演化；其后可从本地状态重建输出。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd

CACHE = ROOT / ".cache/u35_c6"


def capture():
    original = fwd.simulate

    def observed(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        result = original(*args, **kwargs)
        eng, frames, vframes = result
        payload = {"engine": eng, "fixed_frame": frames[-1], "volcanoes": vframes[-1] if vframes else None,
                   "plumes": bound.arguments.get("plumes"), "rifts": bound.arguments.get("rifts"),
                   "rot": str(bound.arguments["rot"])}
        with (CACHE / "final_state.pkl").open("wb") as stream:
            pickle.dump(payload, stream, protocol=5)
        print("[C6] 已保存完整原生单元、旋转、海沟前缘与最终快照", flush=True)
        return result

    fwd.simulate = observed
    previous = sys.argv
    sys.argv = ["design_forward.py", "--design", "v2", "--history", "v2",
                "--history-config", "config/design/history_v2_u35_c4_strain.yaml",
                "--candidate", "u35_c6_state", "--no-figures"]
    try:
        return fwd.main()
    finally:
        fwd.simulate = original
        sys.argv = previous


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.capture:
        return capture()
    return export()


def export():
    from mythic.figures.netcdf import open_netcdf, write_netcdf
    from mythic.figures.world import to_jsonable
    from mythic.metrics.compute import compute_all, load_config
    from mythic.metrics.grid import LatLonGrid
    from mythic.tectonics.mesh import Raster
    from mythic.tectonics.sampling import native_cover, prepare_output, sample_output, FIXED_FALLBACK, VIEW_FILL, VOLCANIC_ISLAND
    from mythic.tectonics.volcanism import island_mask
    from design_final_map import MIN_ISLAND_RADIUS_KM, densify
    from design_west_arc_c5_report import PROTECTED, hashes, sample_grid, tail_length
    import design_west_arc_report as base
    import xarray as xr

    before = hashes()
    with (CACHE / "final_state.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    eng = payload["engine"]
    ctx = prepare_output(eng, payload["fixed_frame"])
    original = eng.cover_fixed()
    exact = native_cover(eng, eng.m.xyz)
    different = original != exact
    raw_fixed_class = ctx.raw["ctype"]
    safe = np.maximum(exact, 0)
    fixed_check = {"nodes": eng.m.n, "different_owners": int(different.sum()),
                   "different_ctype": int(np.sum((exact >= 0) & (eng.c.ctype[safe] != raw_fixed_class))),
                   "uncovered": int((exact < 0).sum()),
                   "view_fill_nodes": int(ctx.filled.sum()),
                   "view_fill_examples": [{"node": int(i), "lon": float(eng.m.lon[i]), "lat": float(eng.m.lat[i]),
                       "raw_uid": int(ctx.raw["uid"][i]), "raw_ctype": int(ctx.raw["ctype"][i]),
                       "display_ctype": int(ctx.display["ctype"][i])} for i in np.flatnonzero(ctx.filled)]}
    # 新取样器不能偷偷改变既有固定节点的覆盖选择。
    if different.any():
        base.write_json(CACHE / "fixed_cover_disagreement.json", fixed_check)
        raise RuntimeError("精确原生查询与引擎固定覆盖存在差异，先诊断，禁止生成候选")
    baseline = CACHE / "worlds"
    baseline.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    summaries = {}
    design = fwd.yaml.safe_load((ROOT / "config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    mountain = next(x for x in design["mountains"] if x["id"] == "M31")
    ridge, distance = densify(mountain["line"], 2.)
    direct = sample_output(ctx, ridge, dynamic_height(payload, ridge), carry_view_fill=True)
    profiles = {"distance_km": distance.tolist(), "native_material_m": direct["elev"].tolist()}
    old_run = ROOT / ".cache/u35_c5/reproduced_c4/latest"
    with open_netcdf(old_run / "world.nc") as old_ds:
        inherited_source = old_ds.attrs["source"]
    hcfg = fwd.yaml.safe_load((ROOT / "config/design/history_v2_u35_c4_strain.yaml").read_text(encoding="utf-8"))
    from design_west_arc_report import compare_npz
    state_run = ROOT / ".cache/tectonics/design/candidates/u35_c6_state/forward_v2_v2_L7"
    neutrality = compare_npz(old_run / "forward.npz", state_run.with_suffix(".npz"))
    old_meta = base.read_json(old_run / "forward.json")
    new_meta = base.read_json(state_run.with_suffix(".json"))
    neutrality.update(budget_identical=old_meta["budget"] == new_meta["budget"], events_identical=old_meta["events"] == new_meta["events"])
    if not all(neutrality[k] for k in ("identical", "budget_identical", "events_identical")):
        raise RuntimeError("状态捕获改变了演化，禁止输出候选：" + str(neutrality))

    for ny in (512, 1024):
        grid = LatLonGrid(2 * ny, ny)
        xyz = grid.xyz.reshape(-1, 3)
        dynamic = dynamic_height(payload, xyz)
        old_raster = Raster(eng.m, ny)
        control_e = old_raster(payload["fixed_frame"]["elev"]).astype(float)
        volcanic = island_mask(grid, payload["volcanoes"], MIN_ISLAND_RADIUS_KM) if payload["volcanoes"] else np.zeros(grid.shape, bool)
        control_e = np.where(volcanic & (control_e <= 0), 100., control_e)
        if ny == 512:
            with open_netcdf(old_run / "world.nc") as ds:
                if not np.array_equal(control_e.astype("float32"), ds.elevation.values):
                    raise RuntimeError("旧输出重建不一致")
        ri, rj = sample_grid(grid, ridge)
        profiles["control_" + grid.name] = control_e[ri, rj].tolist()
        for name, carry in (("native_fill", True), ("native_raw", False)):
            start = time.perf_counter()
            values = sample_output(ctx, xyz, dynamic, carry_view_fill=carry)
            values = {k: v.reshape(grid.shape) for k, v in values.items()}
            e = values["elev"]
            raised = volcanic & (e <= 0)
            e[raised] = 100.
            values["status"][raised] |= VOLCANIC_ISLAND
            C, P = values["ctype"], values["plate"]
            cls, rate = fwd.boundary_classes(P, grid, eng.omega, smooth_cells=7 * ny // 512)
            oro = np.where(C == 1, values["oro"], 0).astype("int8")
            tect = {"tect_plate": P.astype("int16"), "tect_crust_type": C.astype("int8"),
                    "tect_crust_km": values["T"].astype("float32"),
                    "tect_ocean_age_myr": np.where(C == 1, np.nan, values["age"]).astype("float32"),
                    "tect_last_event": oro, "tect_last_event_ma": np.where(oro > 0, values["oro_t"], np.nan).astype("float32"),
                    "tect_boundary": cls, "tect_boundary_cm_yr": rate}
            status = values["status"]
            world_id = f"design_v2_forward_v2_0Ma_u35_c6_{name}_{grid.name}"
            source = ("同一C4 L7存活单元直接查询到输出像元；海沟与动力高程保留。原生无覆盖处沿用固定节点值，"
                      "tect_sampling_status位1明确标记，不能解释为实体地壳；位2为沿用的旧视图填洞，位4为既有火山岛掩膜。"
                      "输出间距不是物理分辨率，原生节点约60 km；不是新的地壳演化或新地形细节。")
            dataset = xr.Dataset({"elevation": (("lat", "lon"), e.astype("float32"), {"units": "m"}),
                "land": (("lat", "lon"), (e > 0).astype("float32")),
                **{k: (("lat", "lon"), v, {**fwd.TECT_ATTRS[k], **({"plate_codes": ",".join(
                    f"{c}={q}" for c, q in fwd.plate_codes(hcfg).items())} if k == "tect_plate" else {})}) for k, v in tect.items()},
                "tect_sampling_status": (("lat", "lon"), status, {"description": "输出表示出处位掩膜，0=直接原生覆盖",
                    "flag_masks": [1, 2, 4], "flag_meanings": "fixed_node_fallback legacy_view_fill volcanic_island", "method_class": "interpolated"}),
                "tect_material_uid": (("lat", "lon"), values["uid"], {"description": "输出宿主永久编号；-1=显示代理/空隙，无实体来源证明", "method_class": "interpolated"})},
                coords={"lat": grid.lat, "lon": grid.lon}, attrs={"world_id": world_id, "display_name": "C6 原生输出候选（" + name + "）",
                    "seed": "design_v2_history_v2", "method_class_elevation": "interpolated", "code_version": fwd.code_version(),
                    "source": inherited_source + "；U35-C6输出：" + source, "grid": grid.name, "native_mean_node_spacing_km": float(eng.m.h_km),
                    "method_class_underlying_crust": "procedural_constrained", "method_class_prescribed_history": "user_override",
                    "candidate_status": "independent_output_candidate_not_adopted", "legacy_view_fill_carried": int(carry)})
            nc = baseline / f"{name}_{grid.name}.nc"
            write_netcdf(dataset, nc)
            m, fields = compute_all(e, e > 0, cfg=cfg, return_fields=True)
            a = grid.cell_area_km2
            lo, la = np.meshgrid(grid.lon, grid.lat)
            west = (lo >= -30) & (lo <= -5) & (la >= 33) & (la <= 40)
            delta = e - control_e
            record = {"A": fwd.a_summary(m), "water_connectivity": water_connectivity(grid, e > 0, tect["tect_ocean_age_myr"], status),
                "sea_ocean_interfaces": sea_interfaces(grid, e > 0, m, fields),
                "native_uncovered_fallback_area_km2": float(a[(status & FIXED_FALLBACK) > 0].sum()),
                "native_uncovered_fallback_fraction": float(a[(status & FIXED_FALLBACK) > 0].sum() / a.sum()),
                "view_fill_area_km2": float(a[(status & VIEW_FILL) > 0].sum()),
                "volcano_override_area_km2": float(a[(status & VOLCANIC_ISLAND) > 0].sum()),
                "land_mask_change_km2": float(a[(e > 0) != (control_e > 0)].sum()),
                "outside_west_land_change_km2": float(a[(~west) & ((e > 0) != (control_e > 0))].sum()),
                "max_height_change_m": float(abs(delta).max()),
                "ridge_output_tail_wet_km": tail_length(distance, e[ri, rj] <= 0),
                "ridge_output_minmax_m": [float(e[ri, rj].min()), float(e[ri, rj].max())],
                "world_file": nc.relative_to(ROOT).as_posix(), "wall_s": round(time.perf_counter() - start, 2)}
            key = name + "_" + grid.name
            summaries[key] = record
            profiles[key] = e[ri, rj].tolist()
            base.write_json(CACHE / (key + "_metrics.json"), m)
            print("[C6]", key, "A", record["A"]["all_pass"], "末端水下", record["ridge_output_tail_wet_km"],
                  "km；回退面积", round(record["native_uncovered_fallback_fraction"] * 100, 3), "%", flush=True)
            del values, dataset, m, fields
    result = {"checkpoint": "U35-C6", "baseline_commit": "add960a", "terrain_evolution_changed": False,
              "default_world_unchanged": True, "candidate_adopted": False, "physics_neutrality": neutrality,
              "fixed_cover_validation": fixed_check, "outputs": summaries, "ridge_profiles": profiles,
              "protected_sha256": before, "protected_unchanged": before == hashes(),
              "method_class": {"underlying_crust": "procedural_constrained", "prescribed_history": "user_override", "height": "parameterized_physics", "output": "interpolated"},
              "limitations": ["固定节点回退不是实体地壳", "输出密集取样不增加物理分辨率", "平顶与构造成因未改变", "所有输出仍为独立候选"]}
    if not result["protected_unchanged"]:
        raise RuntimeError("默认配置或数据被修改")
    base.write_json(ROOT / "data/worlds/design_v2/u35_c6_diagnostic.json", result)
    figures(result)
    return 0


def dynamic_height(payload, points):
    from mythic.tectonics.plumes import swell_m
    from mythic.tectonics.rifts import rift_relief_m
    from mythic.tectonics.kinematics import RotationFileKinematics
    from mythic.tectonics.geometry import unit
    dyn = swell_m(points, 0., (payload["plumes"] or {}).get("swells", []))
    kin = RotationFileKinematics(Path(payload["rot"]))
    for r in payload["rifts"] or []:
        line = unit(r["pts0"] @ kin.total_from(r["anchor"], 0., 0.).T)
        dyn += rift_relief_m(points, line, r["half_km"], r["flank_amp_m"], r["flank_decay_km"], r["swell_amp_m"], r["swell_radius_km"])
    return dyn


def water_connectivity(grid, land, age, status):
    from design_strait_pixels import label_wrap
    lo, la = np.meshgrid(grid.lon, grid.lat)
    def cell(x, y):
        return np.argmin(abs(grid.lat - y)), np.argmin(abs(grid.lon - x))
    sea, ocean = cell(19, 37), cell(-40, 0)
    west = (lo >= -30) & (lo <= -5) & (la >= 33) & (la <= 40)
    water = ~land
    young = west & water & np.isfinite(age) & (age < 10)
    result = {}
    for key, mask, eight in (("4_connected", water, False), ("8_connected", water, True),
            ("4_remove_young", water & ~young, False), ("4_remove_fallback", water & ~((status & 1) > 0), False),
            ("4_block_west", water & ~west, False), ("8_block_west", water & ~west, True)):
        lab = label_wrap(mask, eight)
        result[key] = bool(lab[sea] != 0 and lab[sea] == lab[ocean])
    return result


def sea_interfaces(grid, land, metrics, fields):
    """重用现有海域标签识别海峡接口，只定位，不修改指标算法或阈值。"""
    from mythic.metrics.topology import any_neighbor, label
    sea = metrics["A"]["main_inland_sea"]
    if not sea.get("exists"):
        return []
    region = fields["sea_regions"]
    areas = np.bincount(region.ravel(), weights=grid.cell_area_km2.ravel())
    areas[0] = 0
    world = int(areas.argmax())
    reg = region == sea["main"]["region_id"]
    water = ~land
    interface = reg & any_neighbor(water & ~reg)
    labels, n = label(interface)
    result = []
    for c in range(1, n + 1):
        mask = labels == c
        neighbors = any_neighbor(mask) & water & ~reg
        if np.any(region[neighbors] == world):
            i, j = np.nonzero(mask)
            result.append({"lat_minmax": [float(grid.lat[i].min()), float(grid.lat[i].max())],
                           "lon_minmax": [float(grid.lon[j].min()), float(grid.lon[j].max())],
                           "sample_lat_lon": [float(grid.lat[i].mean()), float(grid.lon[j].mean())], "pixels": len(i)})
    return result


def figures(result):
    # 绘图实现与导出分开；读取已保存候选，保证展示的是实际输出。
    import design_native_output_report
    design_native_output_report.render(result)


if __name__ == "__main__":
    raise SystemExit(main())
