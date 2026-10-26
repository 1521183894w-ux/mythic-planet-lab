"""U35-C10：共享供体约束的2×2完整对照，不以谷底阈值调参。"""
from pathlib import Path
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_arc_sections as sec
import design_west_arc_report as base
from design_native_output import dynamic_height, water_connectivity
from design_west_arc_c5_report import hashes
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm, setup_fonts
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.sampling import prepare_output, sample_output
from mythic.tectonics.sections import spherical_sections
from mythic.tectonics.crust import Crust

CACHE = ROOT / ".cache/u35_c10"
OUT = ROOT / "figures/design/v2/forward/u35_c10"
NAMES = {"old_base": "旧借料：无B5减薄", "new_base": "共享限额：无B5减薄",
    "old_prog": "旧借料：持续减薄β1.5", "new_prog": "共享限额：持续减薄β1.5"}


def save(fig, name, note):
    fig.text(.01, .012, "U35-C10 · " + fwd.code_version() +
        "\n历史/伸展处方 user_override；地壳 procedural_constrained；高程 parameterized_physics；查询 interpolated；L7约60km\n" + note,
        fontsize=8, va="bottom")
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def unchanged_before_40(old, now):
    with np.load(old / "forward.npz") as a, np.load(now / "forward.npz") as b:
        times = np.flatnonzero(a["times"] >= 40)
        keys = [k for k in a.files if k.rsplit("_", 1)[-1].isdigit() and int(k.rsplit("_", 1)[-1]) in times]
        identical = all(np.array_equal(a[k], b[k], equal_nan=True) for k in keys)
    if not identical:
        raise RuntimeError("借料约束实际触发前已改变结果")
    return {"arrays_checked": len(keys), "identical": identical}


def disabled_check():
    aroot, broot = ROOT / ".cache/u35_c9/progressive", CACHE / "legacy"
    result = base.compare_npz(aroot / "forward.npz", broot / "forward.npz")
    am, bm = base.read_json(aroot / "forward.json"), base.read_json(broot / "forward.json")
    result.update(budget_identical=am["budget"] == bm["budget"], events_identical=am["events"] == bm["events"])
    with (aroot / "final_state.pkl").open("rb") as f:
        a = pickle.load(f)["engine"]
    with (broot / "final_state.pkl").open("rb") as f:
        b = pickle.load(f)["engine"]
    result["native_fields_identical"] = all(np.array_equal(getattr(a.c, k), getattr(b.c, k), equal_nan=True) for k in Crust.FIELDS)
    result["alive_identical"] = np.array_equal(a.alive, b.alive)
    result["rotations_identical"] = set(a.R) == set(b.R) and all(np.array_equal(a.R[k], b.R[k]) for k in a.R)
    with open_netcdf(aroot / "world.nc") as a, open_netcdf(broot / "world.nc") as b:
        result["world_arrays_identical"] = set(a.data_vars) == set(b.data_vars) and all(np.array_equal(a[k].values, b[k].values, equal_nan=True) for k in a.data_vars)
    if not all(result[k] for k in ("identical", "budget_identical", "events_identical", "native_fields_identical", "alive_identical", "rotations_identical", "world_arrays_identical")):
        raise RuntimeError("关闭共享限额后旧结果改变")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    c7 = base.read_json(ROOT / "data/worlds/design_v2/u35_c7_diagnostic.json")
    c8 = base.read_json(ROOT / "data/worlds/design_v2/u35_c8_diagnostic.json")
    result = {"checkpoint": "U35-C10", "base_commit": "0145060", "candidate_status": "isolated_not_adopted",
        "method_class": {"prescribed_extension": "user_override", "budget_rule": "procedural_constrained", "height": "parameterized_physics", "query": "interpolated"},
        "feature_disabled_neutrality": disabled_check(), "protected_sha256": hashes(), "cases": {}}
    result["protected_unchanged_from_C8"] = result["protected_sha256"] == c8["protected_sha256"]
    lon, lat, xyz, area = sec.local_map((16.5, 21., -6., 1.))
    with np.load(ROOT / ".cache/u35_c7/maps.npz") as z:
        polygon = z["meander"].copy()
    xx, yy = np.meshgrid(lon, lat)
    region = MPath(polygon).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    center = lonlat_to_xyz(*c7["B5"]["center_lon_lat"])[None, :]
    line = [[18.65, -.4], [18.7, -4.6]]
    cross_xyz, length = spherical_sections(line, c7["B5"]["section_along_km"], c7["B5"]["cross_offset_km"])
    along = np.arange(-60., length + 65., 5.)
    long_xyz, _ = spherical_sections(line, along, [0.])
    local, cross, long, worlds, histories = {}, {}, {}, {}, {}
    paths = {"old_base": ROOT / ".cache/u35_c9/baseline", "old_prog": ROOT / ".cache/u35_c9/progressive",
        "new_base": CACHE / "baseline", "new_prog": CACHE / "progressive"}
    for name in NAMES:
        directory = paths[name]
        with (directory / "final_state.pkl").open("rb") as f:
            payload = pickle.load(f)
        eng = payload["engine"]
        ctx = prepare_output(eng, payload["fixed_frame"])
        def query(points):
            shape = points.shape[:-1]
            p = points.reshape(-1, 3)
            q = sample_output(ctx, p, dynamic_height(payload, p), carry_view_fill=False)
            valid = q["owner"] >= 0
            for k in ("elev", "T"):
                q[k][~valid] = np.nan
            return {k: v.reshape(shape) for k, v in q.items()}
        q, cp, cx, lg = query(xyz), query(center), query(cross_xyz), query(long_xyz)
        local[name], cross[name], long[name] = q["elev"].copy(), cx["elev"].copy(), lg["elev"].ravel().copy()
        ledger = base.read_json(directory / "ledger.json")
        ci = ledger["watch_uids"].index(52985)
        histories[name] = ledger
        last, first = ledger["timeline"][-1], next(r for r in ledger["timeline"] if r["t_Ma"] == 36)
        processes = {k: (v[ci] - first["process_cumulative_km3"].get(k, np.zeros(len(ledger["watch_uids"])))[ci]) / c8["center_area_km2"] for k, v in last["process_cumulative_km3"].items()}
        with open_netcdf(directory / "world.nc") as ds:
            worlds[name] = {k: ds[k].values.copy() for k in ds.data_vars}
        summary, meta = base.read_json(directory / "summary.json"), base.read_json(directory / "forward.json")
        valid = region & (q["owner"] >= 0)
        long_inside = (along >= 0) & (along <= length)
        row = {"center_current_host": {"uid": int(cp["uid"][0]), "origin": int(cp["origin"][0]), "plate": int(cp["plate"][0]),
                "covered": bool(cp["owner"][0] >= 0), "height_m": float(cp["elev"][0]), "T_km": float(cp["T"][0])},
            "design_polygon": {"area_km2": float(area[region].sum()), "covered_area_fraction": float(area[valid].sum() / area[region].sum()),
                "height_m": sec.stats(q["elev"][valid]), "below_sea_fraction_of_covered_area": float(area[valid & (q["elev"] < 0)].sum() / area[valid].sum())},
            "long_axis_inside_height_m": sec.stats(long[name][long_inside]), "cross_profiles_height_m": cross[name], "long_profile_height_m": long[name],
            "initial_uid52985_since_36Ma_equivalent_T_km": processes,
            "initial_uid52985_final_T_km": float(last["T_km"][ci]),
            "ledger_closure_max_km3": max(abs(v) for r in ledger["timeline"] for v in r["unassigned_residual_km3"]),
            "global_budget_km3": meta["budget"], "A": summary["A_engine_0Ma"]}
        if name.startswith("new"):
            d = base.read_json(directory / "coupling.json")
            checks = d["all_commit_borrow_checks"]
            row["shared_budget_audit"] = {"overdraft_km3_sum": sum(r["overdraft_km3"] for r in checks),
                "max_overdraft_equivalent_km": max(r["max_overdraft_equivalent_km"] for r in checks),
                "oversubscribed_commit_count": sum(r["overdraft_km3"] > 1e-6 for r in checks),
                "borrow_reconstruction_max_error_km3": d["borrow_reconstruction_max_error_km3"]}
            if row["shared_budget_audit"]["oversubscribed_commit_count"]:
                raise RuntimeError("限额候选仍有供体超额分配")
            old_name = "old_base" if name == "new_base" else "old_prog"
            row["saved_arrays_40Ma_and_earlier"] = unchanged_before_40(paths[old_name], directory)
            suffix = "baseline" if name == "new_base" else "progressive"
            metric = ROOT / f"figures/stage1/design_v2_forward_v2_0Ma_u35_c10_{suffix}/metrics.json"
            if metric.exists():
                row["b_verdict"] = base.read_json(metric)["b_verdict"]
        result["cases"][name] = row
        del eng, payload, ctx, q
    grid = base.GRID
    lo, la = np.meshgrid(grid.lon, grid.lat)
    window = (lo >= 16.5) & (lo <= 21) & (la >= -6) & (la <= 1)
    m31 = (lo >= -20) & (lo <= -10) & (la >= 34) & (la <= 41)
    differences = {}
    for name, reference in (("new_base", "old_base"), ("new_prog", "old_prog")):
        a, b = worlds[name]["elevation"], worlds[reference]["elevation"]
        delta = a.astype(float) - b.astype(float)
        differences[name] = delta
        result["cases"][name]["difference_vs_same_extension_old_borrowing"] = {"reference": reference,
            "max_height_m": float(abs(delta).max()), "height_gt1m_area_km2": float(grid.cell_area_km2[abs(delta) > 1].sum()),
            "outside_B5_window_height_gt1m_area_km2": float(grid.cell_area_km2[(abs(delta) > 1) & ~window].sum()),
            "land_mask_changed_area_km2": float(grid.cell_area_km2[(a > 0) != (b > 0)].sum()), "M31_max_height_m": float(abs(delta[m31]).max())}
    for name in NAMES:
        e = worlds[name]["elevation"]
        water = water_connectivity(grid, e > 0, worlds[name]["tect_ocean_age_myr"], np.zeros(e.shape, np.uint8))
        water.pop("4_remove_fallback")
        result["cases"][name]["western_sea_connectivity"] = water
    legacy_audit = base.read_json(CACHE / "legacy/coupling.json")
    result["old_rule_worst_donors_by_commit_C_present_frame"] = [{"t_Ma": r["t_Ma"], **r["worst_donor"],
        "commit_overdraft_km3": r["overdraft_km3"], "worst_donor_overdraft_equivalent_km": r["max_overdraft_equivalent_km"]} for r in legacy_audit["all_commit_borrow_checks"] if "worst_donor" in r]
    result["long_axis_along_km"], result["long_axis_length_km"] = along, length
    result["cross_offset_km"] = c7["B5"]["cross_offset_km"]
    result["limitations"] = ["34km是借料规则门槛，不是地球物理下限", "限额缺额按旧显式hole_created规则补足，未求解材料来源",
        "共享余额修复不等于连续谷形、面积伸展或水文已求解", "密集查询不增加L7物理分辨率，灰区仍为无原生覆盖",
        "初始UID52985可跨候选比较，新增UID编号不可直接对应同一材料"]
    base.write_json(ROOT / "data/worlds/design_v2/u35_c10_diagnostic.json", sec.clean(result))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    labels = [NAMES[n] for n in NAMES]
    old_c9 = base.read_json(ROOT / "data/worlds/design_v2/u35_c9_diagnostic.json")
    over = [old_c9["cases"]["baseline"]["global_overdraft_km3_sum"], result["cases"]["new_base"]["shared_budget_audit"]["overdraft_km3_sum"], old_c9["cases"]["progressive"]["global_overdraft_km3_sum"], result["cases"]["new_prog"]["shared_budget_audit"]["overdraft_km3_sum"]]
    axes[0].barh(labels, np.asarray(over) / 1e6, color=["#888", "#5c84a5", "#b78d56", "#78478e"])
    axes[0].set(title="跨提交累计供体超额分配", xlabel="百万km³；不是现存或新增体积")
    axes[1].barh(labels, [result["cases"][n]["global_budget_km3"]["hole_created"] / 1e6 for n in NAMES], color=["#888", "#5c84a5", "#b78d56", "#78478e"])
    axes[1].set(title="实际显式补洞新增体积", xlabel="百万km³；不同于超额借料")
    fig.tight_layout(rect=(0, .15, 1, .94))
    save(fig, "61_shared_budget_constraints.png", "限额仅修正共享供体约束；现有缺额规则仍会新增体积，必须同时报告，不称为完整质量守恒伸展。")

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    cmap, norm = elevation_cmap_norm()
    cmap = cmap.copy()
    cmap.set_bad("#888888")
    p = np.vstack([polygon, polygon[0]])
    for ax, name in zip(axes.ravel(), NAMES):
        im = ax.pcolormesh(lon, lat, local[name], shading="nearest", cmap=cmap, norm=norm)
        ax.plot(p[:, 0], p[:, 1], c="#b5427d", lw=1.)
        ax.scatter(*c7["B5"]["center_lon_lat"], marker="*", c="#ffdd55", edgecolors="#333", s=80)
        h = result["cases"][name]["center_current_host"]["height_m"]
        ax.set(title=NAMES[name] + f"\n定位点{h:.0f}m", xlabel="经度 °E", ylabel="纬度 °N", aspect=1., xlim=(16.5,21), ylim=(-6,1))
        ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.16, top=.94, hspace=.22)
    cax = fig.add_axes([.67,.08,.28,.018])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="原生基底高程 m；灰色无覆盖")
    save(fig, "62_B5_budget_candidates.png", "四案相同原生查询，无填洞/回退；只改变共享借料约束和是否规定β1.5持续减薄，未锁定最终高程。")

    colors = dict(old_base="#888", new_base="#477fa5", old_prog="#be8b49", new_prog="#76438b")
    fig, axes = plt.subplots(2,2,figsize=(16,11))
    for i, ax in enumerate(axes.ravel()[:3]):
        for name in NAMES:
            ax.plot(c7["B5"]["cross_offset_km"], cross[name][i], color=colors[name], label=NAMES[name])
        mask = np.flatnonzero(c7["B5"]["profile_in_design_polygon"][i])
        ax.axvspan(c7["B5"]["cross_offset_km"][mask[0]], c7["B5"]["cross_offset_km"][mask[-1]], color="#deb3ca", alpha=.22)
        ax.set(title=f"北→南第{i+1}横剖面", xlabel="横向 km（＋东方）", ylabel="基底高程 m")
    for name in NAMES:
        axes[1,1].plot(along, long[name], color=colors[name], label=NAMES[name])
    axes[1,1].axvspan(0,length,color="#deb3ca",alpha=.22)
    axes[1,1].set(title="沿设计伸展轴与两端各60km；不是已求解谷底线", xlabel="从轴线北端向南 km", ylabel="基底高程 m")
    for ax in axes.ravel():
        ax.axhline(0,c="#333",lw=.7)
        ax.axhline(-400,c="#b9487e",ls="--",lw=.8)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.96))
    save(fig,"63_B5_cross_and_long_sections.png","粉影为设计目标范围；虚线−400m为谷底目标而非整块铺平要求。约5km查询间隔不等于物理分辨率。")

    fig,axes=plt.subplots(1,2,figsize=(17,8))
    for name in NAMES:
        ledger=histories[name]
        ci=ledger["watch_uids"].index(52985)
        rows=[r for r in ledger["timeline"] if r["t_Ma"]<=40]
        axes[0].plot([r["t_Ma"] for r in rows],[r["T_km"][ci] for r in rows],c=colors[name],label=NAMES[name])
    axes[0].set(xlim=(40,0),title="同一初始材料UID52985",xlabel="Ma",ylabel="厚度 km")
    keys=["collision","lateral_flow","commit_birth_or_borrow","graben_thinning"]
    labels=["碰撞","侧向流动","提交借料","地堑减薄"]
    for index,name in enumerate(NAMES):
        axes[1].bar(np.arange(4)+index*.18-.27,[result["cases"][name]["initial_uid52985_since_36Ma_equivalent_T_km"].get(k,0) for k in keys],width=.18,color=colors[name],label=NAMES[name])
    axes[1].set(xticks=np.arange(4),xticklabels=labels,title="36→0Ma主要累计账项",ylabel="体积÷本宿主面积 km")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.96))
    save(fig,"64_B5_budget_history.png","累计账项不是最终厚度；剥蚀/其他过程见JSON。关闭新限额的独立完整运行与C9持续候选352数组/预算/事件相同。")

    fig,axes=plt.subplots(2,2,figsize=(17,10),gridspec_kw={"width_ratios":[2.1,1.]})
    for row,name in zip(axes,("new_base","new_prog")):
        delta=differences[name]
        row[0].imshow(delta,origin="upper",extent=(-180,180,-90,90),vmin=-1500,vmax=1500,cmap="RdBu_r")
        row[0].contour(grid.lon,grid.lat,abs(delta)>1,levels=[.5],colors="#333",linewidths=.5)
        changed=result["cases"][name]["difference_vs_same_extension_old_borrowing"]["height_gt1m_area_km2"]
        row[0].set(title=NAMES[name]+f" − 同伸展旧借料\n|变化|>1m面积{changed/1e4:.1f}万km²",xlabel="经度 °E",ylabel="纬度 °N")
        sx=(grid.lon>=11)&(grid.lon<=27);sy=(grid.lat>=-12)&(grid.lat<=6)
        im=row[1].pcolormesh(grid.lon[sx],grid.lat[sy],delta[np.ix_(sy,sx)],shading="nearest",vmin=-1500,vmax=1500,cmap="RdBu_r")
        row[1].contour(grid.lon[sx],grid.lat[sy],abs(delta[np.ix_(sy,sx)])>1,levels=[.5],colors="#333",linewidths=.6)
        row[1].set(title="区域影响；黑线为|变化|>1m边界",xlabel="经度 °E",ylabel="纬度 °N",aspect=1.)
        fig.colorbar(im,ax=row[1],shrink=.7,label="高程差 m；截于±1500")
    fig.tight_layout(rect=(0,.14,1,.96))
    save(fig,"65_B5_budget_global_cost.png","比较各自同β情景，保持C4固定节点输出规则。共享预算改善不能自动认定谷地更好或外部代价可接受。")
    html="<!doctype html><meta charset='utf-8'><title>U35-C10共享供体约束候选</title><style>body{font-family:system-ui;max-width:1600px;margin:24px auto;background:#eee}img{max-width:100%}</style><h1>U35-C10：共享供体预算与B5地貌对照</h1><p>两个隔离候选，仅增加供体总余额限额；β、历史、几何不调。限额缺额仍显式新增，不称为真实沉积或完整质量守恒。未采用候选。</p>"
    for name in ("61_shared_budget_constraints.png","62_B5_budget_candidates.png","63_B5_cross_and_long_sections.png","64_B5_budget_history.png","65_B5_budget_global_cost.png"):
        html+=f"<h2>{name}</h2><img src='{name}'>"
    for case in ("baseline","progressive"):
        html+=f"<p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c10_{case}/index.html'>{case}完整标准图组</a></p>"
    (OUT/"index.html").write_text(html,encoding="utf-8")
    print(json.dumps({n:{"center_m":r["center_current_host"]["height_m"],"polygon_median_m":r["design_polygon"]["height_m"]["p50"],"below_sea":r["design_polygon"]["below_sea_fraction_of_covered_area"]} for n,r in result["cases"].items()},ensure_ascii=False))


if __name__=="__main__":
    main()
