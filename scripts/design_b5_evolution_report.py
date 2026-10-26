"""U35-C8：B5逐过程收支与隔离伸展候选的图和数值。"""
from pathlib import Path
import sys
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_arc_sections as sec
import design_west_arc_report as base
from mythic.tectonics.geometry import lonlat_to_xyz, xyz_to_lonlat
from mythic.tectonics.kinematics import RotationFileKinematics
from mythic.tectonics.sampling import prepare_output, sample_output
from mythic.tectonics.sections import spherical_sections
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm
from design_native_output import dynamic_height, water_connectivity
from design_west_arc_c5_report import hashes

OUT = ROOT / "figures/design/v2/forward/u35_c8"
CACHE = ROOT / ".cache/u35_c8"
LABELS = {"collision": "碰撞转入", "arc": "弧增生/缩短", "erosion": "剥蚀", "collapse": "垮塌净转移",
          "lateral_flow": "侧向流动", "commit_birth_or_borrow": "提交：出生/借料", "graben_thinning": "地堑减薄",
          "lip": "大火成岩省", "prescribed_shortening": "规定缩短"}


def save(fig, name, note):
    fig.text(.01, .012, "U35-C8 · " + fwd.code_version() +
        "\n规定历史 user_override；地壳 procedural_constrained；高程 parameterized_physics；观察取样 interpolated\n" + note,
        fontsize=8, va="bottom")
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def baseline_report():
    OUT.mkdir(parents=True, exist_ok=True)
    base.setup_fonts()
    ledger = base.read_json(CACHE / "baseline/ledger.json")
    frames = ledger["timeline"]
    ci = ledger["watch_uids"].index(52985)
    times = np.array([r["t_Ma"] for r in frames])
    final = frames[-1]
    area = final["stock_km3"][ci] / final["T_km"][ci]
    neutral = base.compare_npz(CACHE / "baseline/forward.npz", ROOT / ".cache/u35_c5/reproduced_c4/latest/forward.npz")
    meta = base.read_json(CACHE / "baseline/forward.json")
    old_meta = base.read_json(ROOT / ".cache/u35_c5/reproduced_c4/latest/forward.json")
    neutral.update(budget_identical=meta["budget"] == old_meta["budget"], events_identical=meta["events"] == old_meta["events"])
    if not all(neutral[k] for k in ("identical", "budget_identical", "events_identical")):
        raise RuntimeError("观测运行与基准不一致")
    start = next(r for r in frames if r["t_Ma"] == 36)
    processes = final["process_cumulative_km3"]
    late = {k: (v[ci] - start["process_cumulative_km3"].get(k, np.zeros(len(ledger["watch_uids"])))[ci]) / area for k, v in processes.items()}
    center = [{"t_Ma": r["t_Ma"], "T_km": r["T_km"][ci], "base_elevation_m": r["base_elevation_m"][ci],
               "stock_km3": r["stock_km3"][ci], "lon_lat": [r["lon"][ci], r["lat"][ci]],
               "process_cumulative_km3": {k: v[ci] for k, v in r["process_cumulative_km3"].items()},
               "unassigned_residual_km3": r["unassigned_residual_km3"][ci],
               **({"nearest_D": r["center_to_nearest_D_cont"]} if "center_to_nearest_D_cont" in r else {})} for r in frames]
    donations = [r for r in ledger["donations"] if 52985 in r["target_uid"]]
    attribution = {}
    for r in donations:
        key = f"plate{r['source_plate']}_origin{r['source_origin']}"
        value = r["deposited_km3"][r["target_uid"].index(52985)]
        record = attribution.setdefault(key, {"all_km3": 0., "since_36Ma_km3": 0., "source_uids": set()})
        record["all_km3"] += value
        if r["t_Ma"] < 36:
            record["since_36Ma_km3"] += value
        record["source_uids"].add(r["source_uid"])
    for record in attribution.values():
        record["source_uids"] = sorted(record["source_uids"])
    result = {"checkpoint": "U35-C8", "baseline_commit": "03cab8f", "center_uid": 52985, "watch_uids": ledger["watch_uids"],
        "observation_neutrality": neutral, "ledger_closure_max_km3": max(abs(x) for r in frames for x in r["unassigned_residual_km3"]),
        "collision_attribution_max_error_km3": ledger["collision_reconstruction_max_error_km3"],
        "center_area_km2": area, "center_timeline": center, "center_since_36Ma_equivalent_T_km": late,
        "center_collision_attribution": attribution, "candidates": {},
        "method_class": {"history": "user_override", "process": "procedural_constrained", "height": "parameterized_physics", "query": "interpolated"},
        "limitations": ["累计厚度收支不是现今厚度，也不是沉积层厚度", "提交借料项不是持续构造伸展",
            "最近C/D宿主连线法向不是断层法向或盆内已求解应变", "基底低于海平面不代表已求解湖面、排水与封闭水文"]}
    base.write_json(ROOT / "data/worlds/design_v2/u35_c8_diagnostic.json", sec.clean(result))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(times, [r["T_km"][ci] for r in frames], c="#754282", lw=1.5)
    axes[0].set(title="B5定位点的原生宿主UID52985：完整300→0Ma厚度演化", ylabel="地壳厚度 km")
    for k in ("collision", "arc", "erosion", "collapse", "lateral_flow", "commit_birth_or_borrow", "graben_thinning"):
        values = [r["process_cumulative_km3"].get(k, np.zeros(len(ledger["watch_uids"])))[ci] / area for r in frames]
        axes[1].plot(times, values, lw=1.2, label=LABELS[k])
    axes[1].set(xlabel="Ma（左早右晚）", ylabel="累计体积÷本宿主面积 km", title="逐过程实际增减：累计量可远大于现存厚度")
    axes[1].legend(ncol=4, fontsize=9)
    for ax in axes:
        ax.axvline(35, c="#444", ls=":")
        ax.grid(alpha=.2)
        ax.set_xlim(300, 0)
    fig.tight_layout(rect=(0, .13, 1, .95))
    save(fig, "51_B5_material_history.png", "基底高程按地壳均衡；当前宿主非固定地理点。35Ma线为C/D拉分规则首次可在提交中生效的时刻。")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    keys = [k for k in LABELS if k in late]
    values = [late[k] for k in keys]
    axes[0].barh([LABELS[k] for k in keys], values, color=["#b67d4d" if v > 0 else "#6096ad" for v in values])
    axes[0].axvline(0, c="#333", lw=.8)
    axes[0].set(xlabel="36→0Ma净体积变化÷宿主面积 km", title=f"晚期库存：{start['T_km'][ci]:.2f}→{final['T_km'][ci]:.2f}km")
    late_rows = [r for r in frames if r["t_Ma"] <= 36]
    axes[1].plot([r["t_Ma"] for r in late_rows], [r["T_km"][ci] for r in late_rows], c="#754282")
    axes[1].set(xlim=(36, 0), xlabel="Ma", ylabel="地壳厚度 km", title="多次借料、碰撞与流动叠加，不能只看最后事件")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, .17, 1, .94))
    save(fig, "52_B5_late_budget.png", "地堑减薄为零；提交借料是给邻近小洞供料，不能当作已实现局部伸展。全部21个追踪宿主均独立核查闭合。")

    kin = RotationFileKinematics(ROOT / "data/worlds/design_v2/history_v2.rot")
    c7 = base.read_json(ROOT / "data/worlds/design_v2/u35_c7_diagnostic.json")
    poly = np.asarray(c7["B5"]["design"]["poly"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    p = np.vstack([poly, poly[0]])
    axes[0].plot(p[:, 0], p[:, 1], "--", c="#b43e78", label="B5设计原顶点")
    for t, color in ((35, "#654a8e"), (20, "#7e8e3b"), (10, "#b58939"), (0, "#a84442")):
        r = next(r for r in frames if r["t_Ma"] == t)
        cp = lonlat_to_xyz(r["lon"][ci], r["lat"][ci])
        dp = lonlat_to_xyz(*r["center_to_nearest_D_cont"]["D_lon_lat"])
        R = kin.total_from(301, 0., t)
        lo, la = xyz_to_lonlat(np.stack([cp, dp]) @ R)
        axes[0].plot(lo, la, "o-", c=color, lw=1, ms=4, label=f"{t}Ma最近C/D宿主连线")
    axes[0].scatter(*c7["B5"]["center_lon_lat"], marker="*", c="#ffdc55", edgecolors="#222", s=100, zorder=8)
    axes[0].set(xlim=(17.5, 21.), ylim=(-5., -.2), title="换到洲C现今参照系：连线不是断层线", xlabel="经度 °E", ylabel="纬度 °N")
    axes[0].set_aspect(1.)
    axes[0].legend(fontsize=8)
    tr = [r for r in frames if r["t_Ma"] <= 36 and "center_to_nearest_D_cont" in r]
    axes[1].plot([r["t_Ma"] for r in tr], [r["center_to_nearest_D_cont"]["separation_cm_yr"] for r in tr], label="连线法向分离（＋张开）")
    axes[1].plot([r["t_Ma"] for r in tr], [r["center_to_nearest_D_cont"]["shear_cm_yr"] for r in tr], label="连线切向相对运动")
    axes[1].axhline(0, c="#333", lw=.7)
    axes[1].set(xlim=(36, 0), xlabel="Ma", ylabel="cm/yr", title="最近宿主会切换：不能把投影当作唯一释放段应变")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, .17, 1, .95))
    save(fig, "53_B5_motion_and_location.png", "星为定位点，圆点为真实单元中心。模型没有在目标范围求解连续伸展；末帧标签与单次邻接投影不能代替完整成因。")
    return result


def candidate_report(result):
    """相同输出规则下比较物理候选；原生局部图不回退、不填洞。"""
    c7 = base.read_json(ROOT / "data/worlds/design_v2/u35_c7_diagnostic.json")
    lon, lat, xyz, area = sec.local_map((16.5, 21., -6., 1.))
    with np.load(ROOT / ".cache/u35_c7/maps.npz") as z:
        meander = z["meander"].copy()
    from matplotlib.path import Path as MPath
    xx, yy = np.meshgrid(lon, lat)
    region = MPath(meander).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    point = lonlat_to_xyz(*c7["B5"]["center_lon_lat"])[None, :]
    bp, _ = spherical_sections([[18.65, -.4], [18.7, -4.6]], c7["B5"]["section_along_km"], c7["B5"]["cross_offset_km"])
    local, profiles, evolutions, worlds = {}, {}, {}, {}
    protected = hashes()
    reference_protected = base.read_json(ROOT / "data/worlds/design_v2/u35_c7_provenance.json")["protected_sha256"]
    for case in ("baseline", "instant", "progressive"):
        with (CACHE / case / "final_state.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        eng = payload["engine"]
        ctx = prepare_output(eng, payload["fixed_frame"])
        def query(points):
            shape = points.shape[:-1]
            p = points.reshape(-1, 3)
            q = sample_output(ctx, p, dynamic_height(payload, p), carry_view_fill=False)
            valid = q["owner"] >= 0
            q["elev"][~valid] = np.nan
            q["T"][~valid] = np.nan
            for field in ("uid", "origin", "plate"):
                q[field] = q[field].astype(np.int64)
                q[field][~valid] = -1
            return {k: v.reshape(shape) for k, v in q.items()}
        q, cp, cross = query(xyz), query(point), query(bp)
        local[case] = {k: q[k].copy() for k in ("elev", "T", "owner")}
        profiles[case] = cross["elev"].copy()
        ledger = base.read_json(CACHE / case / "ledger.json")
        ci = ledger["watch_uids"].index(52985)
        evolutions[case] = {"t_Ma": [r["t_Ma"] for r in ledger["timeline"]], "T_km": [r["T_km"][ci] for r in ledger["timeline"]]}
        valid = region & (q["owner"] >= 0)
        with open_netcdf(CACHE / case / "world.nc") as ds:
            worlds[case] = {k: ds[k].values.copy() for k in ds.data_vars}
            attrs = dict(ds.attrs)
        meta = base.read_json(CACHE / case / "forward.json")
        summary = base.read_json(CACHE / case / "summary.json")
        all_closure = max(abs(x) for r in ledger["timeline"] for x in r["unassigned_residual_km3"])
        center_closure = max(abs(r["unassigned_residual_km3"][ci]) for r in ledger["timeline"])
        start_row = next(r for r in ledger["timeline"] if r["t_Ma"] == 36)
        center_process = {k: (v[ci] - start_row["process_cumulative_km3"].get(k, np.zeros(len(ledger["watch_uids"])))[ci]) / result["center_area_km2"]
                          for k, v in ledger["timeline"][-1]["process_cumulative_km3"].items()}
        result["candidates"][case] = {"center_current_host": {"uid": int(cp["uid"][0]), "covered": bool(cp["owner"][0] >= 0),
                "height_m": float(cp["elev"][0]), "T_km": float(cp["T"][0]), "origin": int(cp["origin"][0]), "plate": int(cp["plate"][0])},
            "design_polygon": {"area_km2": float(area[region].sum()), "covered_area_fraction": float(area[valid].sum() / area[region].sum()),
                "height_m": sec.stats(q["elev"][valid]), "T_km": sec.stats(q["T"][valid]),
                "below_sea_fraction_of_covered_area": float(area[valid & (q["elev"] < 0)].sum() / area[valid].sum())},
            "A": summary["A_engine_0Ma"], "global_budget_km3": meta["budget"],
            "initial_uid52985_T_km": evolutions[case]["T_km"][-1], "initial_uid52985_closure_max_km3": center_closure,
            "initial_uid52985_since_36Ma_equivalent_T_km": center_process,
            "all_watched_UID_closure_max_km3": all_closure,
            "UID_comparison_note": "初始UID52985可跨候选追踪；候选出生顺序改变后的新增UID不保证仍对应基准的同一地理来源",
            "world_attrs": {k: attrs[k] for k in ("world_id", "method_class_elevation", "source", "method_class_prescribed_local_extension", "prescribed_local_extension") if k in attrs}}
        metric_path = ROOT / f"figures/stage1/design_v2_forward_v2_0Ma_u35_c8_{case}/metrics.json"
        if metric_path.exists():
            result["candidates"][case]["b_verdict"] = base.read_json(metric_path)["b_verdict"]
        del eng, payload, ctx, q

    grid = base.GRID
    lo, la = np.meshgrid(grid.lon, grid.lat)
    bwindow = (lo >= 16.5) & (lo <= 21.) & (la >= -6.) & (la <= 1.)
    m31 = (lo >= -20.) & (lo <= -10.) & (la >= 34.) & (la <= 41.)
    for case in ("baseline", "instant", "progressive"):
        e = worlds[case]["elevation"]
        connect = water_connectivity(grid, e > 0, worlds[case]["tect_ocean_age_myr"], np.zeros(e.shape, np.uint8))
        connect.pop("4_remove_fallback")  # 旧固定输出没有原生回退来源位，不能编造该敏感性结果。
        row = result["candidates"][case]
        row["western_sea_connectivity"] = connect
        delta = e.astype(float) - worlds["baseline"]["elevation"].astype(float)
        changed_land = (e > 0) != (worlds["baseline"]["elevation"] > 0)
        row["difference_vs_same_fixed_output_baseline"] = {"max_height_m": float(abs(delta).max()),
            "height_gt1m_area_km2": float(grid.cell_area_km2[abs(delta) > 1].sum()),
            "outside_B5_window_height_gt1m_area_km2": float(grid.cell_area_km2[(abs(delta) > 1) & ~bwindow].sum()),
            "land_mask_changed_area_km2": float(grid.cell_area_km2[changed_land].sum()),
            "M31_max_height_m": float(abs(delta[m31]).max())}
        # 候选激活前的保存数组必须完全一致。
        with np.load(CACHE / "baseline/forward.npz") as a, np.load(CACHE / case / "forward.npz") as b:
            n = np.flatnonzero(a["times"] >= 40)
            keys = [k for k in a.files if k.rsplit("_", 1)[-1].isdigit() and int(k.rsplit("_", 1)[-1]) in n]
            same = all(np.array_equal(a[k], b[k], equal_nan=True) for k in keys)
            row["saved_arrays_40Ma_and_earlier"] = {"arrays_checked": len(keys), "identical": same}
            if not same:
                raise RuntimeError(case + "在B5激活前已改变演化")
    plain = ROOT / ".cache/tectonics/design/candidates/u35_c8_plain/forward_v2_v2_L7"
    result["new_feature_disabled_neutrality"] = base.compare_npz(CACHE / "baseline/forward.npz", plain.with_suffix(".npz"))
    pm = base.read_json(plain.with_suffix(".json"))
    bm = base.read_json(CACHE / "baseline/forward.json")
    result["new_feature_disabled_neutrality"].update(budget_identical=pm["budget"] == bm["budget"], events_identical=pm["events"] == bm["events"])
    if not all(result["new_feature_disabled_neutrality"][k] for k in ("identical", "budget_identical", "events_identical")):
        raise RuntimeError("持续伸展未启用时改变了旧配置结果")
    result["protected_sha256"] = protected
    result["protected_unchanged_from_C7"] = protected == reference_protected
    if not result["protected_unchanged_from_C7"]:
        raise RuntimeError("默认配置或摘要被改变")
    result["profiles"] = {"offset_km": c7["B5"]["cross_offset_km"], "height_m": profiles,
        "profile_in_design_polygon": c7["B5"]["profile_in_design_polygon"]}
    result["initial_uid52985_evolution_by_case"] = evolutions
    result["selection"] = "independent_candidates_not_adopted"
    base.write_json(ROOT / "data/worlds/design_v2/u35_c8_diagnostic.json", sec.clean(result))
    np.savez_compressed(CACHE / "candidate_maps.npz", lon=lon, lat=lat, meander=meander,
        **{case + "_" + field: values[field] for case, values in local.items() for field in ("elev", "T")})
    cmap, norm = elevation_cmap_norm()
    cmap = cmap.copy()
    cmap.set_bad("#888888")
    names = {"baseline": "基准：没有B5地堑减薄", "instant": "35Ma一次减薄，总β=1.5", "progressive": "35→0Ma持续减薄，总β=1.5"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, case in zip(axes, names):
        im = ax.pcolormesh(lon, lat, local[case]["elev"], shading="nearest", cmap=cmap, norm=norm)
        p = np.vstack([meander, meander[0]])
        ax.plot(p[:, 0], p[:, 1], c="#bd3b7c", lw=1.2)
        ax.scatter(*c7["B5"]["center_lon_lat"], marker="*", c="#ffde5a", edgecolors="#222", s=90)
        center = result["candidates"][case]["center_current_host"]["height_m"]
        ax.set(title=names[case] + f"\n定位点原生{center:.0f}m", xlabel="经度 °E", ylabel="纬度 °N", xlim=(16.5, 21), ylim=(-6, 1), aspect=1.)
        ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23, top=.87, wspace=.22)
    cax = fig.add_axes([.65, .13, .3, .025])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="原生基底高程 m；灰色无覆盖")
    save(fig, "54_B5_extension_candidates.png", "同一L7状态查询规则，不填洞/不回退。粉线为B5设计轮廓；β、位置、时间为user_override，未求解水位、沉积、热沉降。")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = {"baseline": "#888", "instant": "#d38b36", "progressive": "#704295"}
    for i, ax in enumerate(axes.ravel()[:3]):
        for case in names:
            ax.plot(c7["B5"]["cross_offset_km"], profiles[case][i], c=colors[case], label=names[case])
        mask = np.flatnonzero(c7["B5"]["profile_in_design_polygon"][i])
        ax.axvspan(c7["B5"]["cross_offset_km"][mask[0]], c7["B5"]["cross_offset_km"][mask[-1]], color="#e9b9d4", alpha=.2)
        ax.axhline(0, c="#333", lw=.7)
        ax.axhline(-400, c="#bc3d7b", ls="--", lw=.9, label="谷底设计目标−400m")
        ax.set(title=f"B5北→南第{i+1}横剖面", xlabel="横向 km（＋东方）", ylabel="基底高程 m")
        ax.grid(alpha=.2)
    for case in names:
        ev = evolutions[case]
        t = np.asarray(ev["t_Ma"])
        mask = t <= 40
        axes[1, 1].plot(t[mask], np.asarray(ev["T_km"])[mask], c=colors[case], label=names[case])
    axes[1, 1].set(xlim=(40, 0), title="同一初始UID52985：末帧定位点宿主另行查询", xlabel="Ma", ylabel="地壳厚度 km")
    axes[1, 1].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    save(fig, "55_B5_profiles_and_persistence.png", "两候选使用同一总β和相同对数空间权重，仅改变时间分配；后续碰撞/流动/提交仍照常，未锁定最终盆底高程。")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), gridspec_kw={"width_ratios": [2.1, 1.]})
    for row, case in zip(axes, ("instant", "progressive")):
        delta = worlds[case]["elevation"].astype(float) - worlds["baseline"]["elevation"].astype(float)
        row[0].imshow(delta, origin="upper", extent=(-180, 180, -90, 90), vmin=-1500, vmax=1500, cmap="RdBu_r")
        row[0].add_patch(matplotlib.patches.Rectangle((11, -12), 16, 18, fill=False, edgecolor="#555", linewidth=.8))
        row[0].set(title=names[case] + "：全球高程差", xlabel="经度 °E", ylabel="纬度 °N")
        sx = (grid.lon >= 11) & (grid.lon <= 27)
        sy = (grid.lat >= -12) & (grid.lat <= 6)
        im = row[1].pcolormesh(grid.lon[sx], grid.lat[sy], delta[np.ix_(sy, sx)], shading="nearest", vmin=-1500, vmax=1500, cmap="RdBu_r")
        row[1].contour(grid.lon[sx], grid.lat[sy], abs(delta[np.ix_(sy, sx)]) > 1, levels=[.5], colors="#333", linewidths=.7)
        row[1].set(title="区域放大；黑线=|变化|>1m边界", xlabel="经度 °E", ylabel="纬度 °N", aspect=1.)
        fig.colorbar(im, ax=row[1], shrink=.7, label="高程差 m（色带截于±1500）")
    fig.tight_layout(rect=(0, .13, 1, .95))
    save(fig, "56_B5_global_cost.png", "完整全球输出均保持旧固定节点规则；比较的是地壳演化变化，未混入C6的全球直接取样变化。默认C2未替换。")
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C8 B5演化与伸展候选</title><style>body{font-family:system-ui;max-width:1600px;margin:24px auto;padding:16px;background:#eee}img{max-width:100%}</style><h1>U35-C8：B5演化收支与伸展候选</h1><p>相同总β=1.5，位置/时间/倍率为设计处方；没有锁定盆底高程，没有采用候选或改变默认世界。负基底高程不代表水文已求解。</p>"
    for name in ("51_B5_material_history.png", "52_B5_late_budget.png", "53_B5_motion_and_location.png", "54_B5_extension_candidates.png", "55_B5_profiles_and_persistence.png", "56_B5_global_cost.png"):
        html += f"<h2>{name}</h2><img src='{name}'>"
    html += "<p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c8_instant/index.html'>一次减薄标准图组</a> · <a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c8_progressive/index.html'>持续减薄标准图组</a></p>"
    (OUT / "index.html").write_text(html, encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = baseline_report()
    if all((CACHE / case / "final_state.pkl").exists() for case in ("instant", "progressive")):
        candidate_report(result)
