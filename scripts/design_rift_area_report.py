"""U36-S2：实际三亿年运行的守恒、地图、副作用与兼容性对照。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_final_map as final
import design_arc_sections as sec
import design_west_arc_report as base
from design_native_output import dynamic_height
from design_west_arc_c5_report import hashes
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm
from mythic.tectonics.crust import CONT
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.sampling import prepare_output, sample_output

CACHE = ROOT / ".cache/u36_s2"
OUT = ROOT / "figures/design/v2/forward/u36_s2"
DATA = ROOT / "data/worlds/design_v2/u36_s2_diagnostic.json"


def save(fig, name, note):
    fig.text(.012, .014, "U36-S2 · " + fwd.code_version() +
             "\n种子 design_v2/history_v2；L7约60 km；规定历史 user_override；地壳 procedural_constrained；高程 parameterized_physics；取样 interpolated\n" +
             note, fontsize=8, va="bottom")
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def query(payload, points):
    shape = points.shape[:-1]
    xyz = points.reshape(-1, 3)
    ctx = prepare_output(payload["engine"], payload["fixed_frame"])
    q = sample_output(ctx, xyz, dynamic_height(payload, xyz), carry_view_fill=False)
    covered = q["owner"] >= 0
    for key in ("elev", "T"):
        q[key][~covered] = np.nan
    return {**{key: value.reshape(shape) for key, value in q.items() if key in ("elev", "T", "uid", "ctype")},
            "covered": covered.reshape(shape)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base.setup_fonts()
    payloads, worlds, metas, summaries = {}, {}, {}, {}
    for case in ("control", "conservative"):
        with (CACHE / case / "final_state.pkl").open("rb") as stream:
            payloads[case] = pickle.load(stream)
        with open_netcdf(CACHE / case / "world.nc") as ds:
            worlds[case] = ds.elevation.values.copy()
            lon, lat = ds.lon.values.copy(), ds.lat.values.copy()
        metas[case] = base.read_json(CACHE / case / "forward.json")
        summaries[case] = base.read_json(CACHE / case / "summary.json")
    old = ROOT / ".cache/u35_c11/progressive"
    compatibility = base.compare_npz(old / "forward.npz", CACHE / "control/forward.npz")
    old_meta = base.read_json(old / "forward.json")
    compatibility.update(budget_identical=old_meta["budget"] == metas["control"]["budget"],
                         events_identical=old_meta["events"] == metas["control"]["events"],
                         native_state_identical=sec.state_hash(pickle.load((old / "final_state.pkl").open("rb"))["engine"]) ==
                                                sec.state_hash(payloads["control"]["engine"]))
    with open_netcdf(old / "world.nc") as ds:
        compatibility["world_elevation_identical"] = np.array_equal(ds.elevation.values, worlds["control"])
    if not all(compatibility[k] for k in ("identical", "budget_identical", "events_identical", "native_state_identical", "world_elevation_identical")):
        raise RuntimeError("关闭候选后的兼容对照发生变化")
    protected = hashes()
    protected_identical_to_head = {}
    for path, digest in protected.items():
        raw = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
        protected_identical_to_head[path] = hashlib.sha256(raw).hexdigest() == digest
    if not all(protected_identical_to_head.values()):
        raise RuntimeError("默认保护文件发生变化")
    records = payloads["conservative"]["engine"].rift_area_audit
    np.savez_compressed(CACHE / "conservative/rift_area_edges.npz",
                        **{f"commit{i}_{k}": v for i, r in enumerate(records) for k, v in r.items()})
    pair_stats = {}
    for r in records:
        for pair in np.unique(r["target_pair"]):
            mask_pair = r["target_pair"] == pair
            item = pair_stats.setdefault(str(int(pair)), dict(target_births=0, new_area_km2=0., transferred_km3=0.))
            item["target_births"] += int(mask_pair.sum())
            item["new_area_km2"] += float(r["target_area_km2"][mask_pair].sum())
            item["transferred_km3"] += float((r["target_area_km2"]*r["target_thickness_km"])[mask_pair].sum())
    loss = sum(r["source_loss_km3"] for r in records)
    gain = sum(r["target_gain_km3"] for r in records)
    audit = dict(by_pair=pair_stats, commits=len(records), target_births=sum(len(r["target_uid"]) for r in records),
                 transferred_km3=gain, source_loss_km3=loss, cumulative_closure_km3=gain-loss,
                 max_commit_abs_closure_km3=max(abs(r["closure_km3"]) for r in records),
                 source_beta_stats=sec.stats(np.concatenate([r["source_beta"] for r in records])),
                 target_birth_T_km_stats=sec.stats(np.concatenate([r["target_thickness_km"] for r in records])),
                 source_after_T_km_stats=sec.stats(np.concatenate([r["source_after_km"] for r in records])),
                 cumulative_target_area_km2=sum(r["target_area_km2"].sum() for r in records),
                 target_below_7km_birth_area_km2=sum(r["target_area_km2"][r["target_thickness_km"] < 7].sum() for r in records),
                 ledger_file=".cache/u36_s2/conservative/rift_area_edges.npz",
                 ledger_sha256=hashlib.sha256((CACHE / "conservative/rift_area_edges.npz").read_bytes()).hexdigest(),
                 timeline=[{k: r[k] for k in ("t_ma", "source_loss_km3", "target_gain_km3", "closure_km3")} for r in records])
    design = fwd.yaml.safe_load((ROOT / "config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    basin = next(b for b in design["basins"] if b["id"] == "B5")
    meander = final.meander_polygon(basin["poly"], final.feature_rng("B5"), basin["meander_km"])
    blon, blat, bxyz, barea = sec.local_map((16.5, 21., -6., 1.))
    blo, bla = np.meshgrid(blon, blat)
    mask = MPath(meander).contains_points(np.c_[blo.ravel(), bla.ravel()]).reshape(blo.shape)
    axis_lat = np.linspace(-4.6, -.4, 421)
    axis_lon = np.interp(axis_lat, [-4.6, -.4], [18.7, 18.65])
    axis_xyz = lonlat_to_xyz(axis_lon, axis_lat)
    b5, maps, profiles = {}, {}, {}
    for case, payload in payloads.items():
        q = maps[case] = query(payload, bxyz)
        valid = mask & q["covered"]
        cp = query(payload, lonlat_to_xyz(18.675, -2.5)[None, :])
        ap = profiles[case] = query(payload, axis_xyz)
        b5[case] = dict(center_native_m=float(cp["elev"][0]), center_uid=int(cp["uid"][0]),
                        center_T_km=float(cp["T"][0]), polygon_area_km2=float(barea[mask].sum()),
                        covered_fraction=float(barea[valid].sum()/barea[mask].sum()),
                        native_height_m=sec.stats(q["elev"][valid]),
                        below_sea_fraction_of_covered=float(barea[valid & (q["elev"] < 0)].sum()/barea[valid].sum()),
                        axis_height_m=sec.stats(ap["elev"]), axis_covered_fraction=float(ap["covered"].mean()),
                        axis_lat=axis_lat, axis_native_m=ap["elev"])
    # 固定输出用于全球面积积分；精细B5查询只观察同一L7材料。
    d = worlds["conservative"].astype(float) - worlds["control"]
    grid = base.GRID
    area = grid.cell_area_km2
    from design_strait_pixels import label_wrap
    b5_world_water = {}
    jj = int(np.argmin(abs(lon-18.675)))
    ii = int(np.argmin(abs(lat+2.5)))
    for case, e in worlds.items():
        a4 = label_wrap(e <= 0, False)
        a8 = label_wrap(e <= 0, True)
        largest4 = int(np.bincount(a4.ravel(), weights=np.where(e <= 0, area, 0).ravel())[1:].argmax()+1)
        largest8 = int(np.bincount(a8.ravel(), weights=np.where(e <= 0, area, 0).ravel())[1:].argmax()+1)
        b5_world_water[case] = dict(pixel_elevation_m=float(e[ii, jj]), four_connected_to_largest_water=bool(a4[ii, jj]==largest4),
                                  eight_connected_to_largest_water=bool(a8[ii, jj]==largest8))
    audit["center_uid52985_direct_donations"] = [
        dict(t_ma=r["t_ma"], volume_km3=float(((r["source_before_km"]-r["source_after_km"])*r["source_area_km2"])[r["source_uid"]==52985].sum()))
        for r in records if np.any(r["source_uid"]==52985)]
    global_change = dict(land_change_km2=float(area[(worlds["conservative"] > 0) != (worlds["control"] > 0)].sum()),
                         height_change_gt100m_area_km2=float(area[abs(d) > 100].sum()),
                         minmax_change_m=[float(d.min()), float(d.max())])
    cases = {}
    for case, p in payloads.items():
        eng = p["engine"]
        c, a = eng.c, eng.area
        cont = eng.alive & (c.ctype == CONT)
        cases[case] = dict(budget_km3=metas[case]["budget"], A=summaries[case]["A_engine_0Ma"],
                           sea_level_m=summaries[case]["sea_level_m"],
                           land_iou=summaries[case]["comparison"]["land_iou"],
                           continental_volume_km3=float((c.T[cont] * a[cont]).sum()),
                           cont_below7km_area_km2=float(a[cont & (c.T < 7)].sum()),
                           cont_below15km_area_km2=float(a[cont & (c.T < 15)].sum()),
                           cont_T_km_stats=sec.stats(c.T[cont]),
                           remaining_65km_cap_area_km2=float(a[cont & (c.T >= 65-1e-9)].sum()),
                           feature_output=[r for key in ("mountains", "plateaus_basins")
                                           for r in summaries[case]["comparison"][key]
                                           if r["id"] in ("M3f", "M31", "B5", "B6", "PL8")])
    result = dict(checkpoint="U36-S2", baseline_commit="3a2b371", code_version=fwd.code_version(),
                  method_class=dict(history="user_override", crust="procedural_constrained",
                                    elevation="parameterized_physics", queries="interpolated"),
                  units=dict(thickness="km", area="km2", volume="km3", elevation="m", time="Ma"),
                  assumptions=["固定单元表上的面积展开代理，未求解完整应变、应力、断层与热沉降",
                               "既有张开/最大宽度门槛不变；12近邻，限同板块对陆壳，按面积/软化距离平方分摊",
                               "所有供体同时变薄；不设最低厚度，不补显式来源；新单元厚度由转入材料计算",
                               "地壳年龄/省份取体积最大供体的代表值，不是混合组分追踪",
                               "hole_created、旧垮塌和输出回退仍保留；内部rift_transferred不是物质源"],
                  compatibility=compatibility, protected_identical_to_head=protected_identical_to_head,
                  transport=audit, cases=cases, B5=b5, B5_world_water=b5_world_water, global_change=global_change,
                  candidate_adopted=False, default_world="C2",
                  limitations=["本试验仅隔离裂谷填充来源，不是已验证的全球连续变形求解器",
                               "守恒不保证地形好看或地质正确；网格几何、借料范围、再碰撞可反馈",
                               "L7约60km；局部0.025度查询不增加物理细节；海底低地不等于已求解湖泊"])
    base.write_json(DATA, sec.clean(result))
    cmap, norm = elevation_cmap_norm()
    cmap = cmap.copy()
    cmap.set_bad("#888888")

    def map_ax(ax, x, y, e, title, diff=False):
        im = ax.pcolormesh(x, y, e, cmap="RdBu_r" if diff else cmap,
                          norm=matplotlib.colors.TwoSlopeNorm(0, -1500, 1500) if diff else norm,
                          shading="nearest", rasterized=True)
        ax.set(title=title, xlabel="经度 °E", ylabel="纬度 °N")
        ax.grid(alpha=.18)
        ax.set_aspect(1 / np.cos(np.radians(np.mean(y))))
        span = float(max(x)-min(x))
        scale = 2000 if span > 100 else 500 if span > 15 else 100
        x0 = float(min(x)) + span*.08
        y0 = float(min(y)) + float(max(y)-min(y))*.06
        dx = scale / (111.195 * np.cos(np.radians(y0)))
        ax.plot([x0, x0+dx], [y0, y0], c="#222", lw=2.5)
        ax.text(x0+dx/2, y0+float(max(y)-min(y))*.02, f"{scale} km", ha="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=1))
        return im

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for case, color in (("control", "#777777"), ("conservative", "#74449a")):
        rows = metas[case]["commit_log"]
        tt = [r["t"] for r in rows]
        for key, ax, label in (("rift_fill", axes[0, 0], "裂谷整柱新增"),
                               ("hole_created", axes[0, 1], "小洞显式缺额")):
            ax.plot(tt, [r["budget"].get(key, 0)/1e6 for r in rows], c=color, label=case)
            ax.set(title=label, ylabel="累计物质源 百万km³", xlabel="Ma")
        axes[1, 1].plot(tt, [r["cont_volume_km3"]/1e9 for r in rows], c=color, label=case)
    axes[1, 0].plot([r["t_ma"] for r in records], np.cumsum([r["target_gain_km3"] for r in records])/1e6, c="#74449a")
    axes[1, 0].set(title="候选实际内部转移（不是新增）", ylabel="累计 百万km³", xlabel="Ma")
    axes[1, 1].set(title="全球陆壳库存（含其他过程反馈）", ylabel="十亿km³", xlabel="Ma")
    for ax in axes.ravel():
        ax.invert_xaxis()
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout(rect=(0, .13, 1, .96))
    save(fig, "71_material_budget.png", f"逐提交最大体积残差 {audit['max_commit_abs_closure_km3']:.3g} km³；新目标数{audit['target_births']}。净物质源与内部运输分开记账。")

    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    for ax, case in zip(axes, ("control", "conservative")):
        im = map_ax(ax, lon, lat, worlds[case], case + "：同色带全球高程")
        fig.colorbar(im, ax=ax, shrink=.7, label="m")
    im = map_ax(axes[2], lon, lat, d, "候选 − 对照（超出色带饱和；海平面调整已包含）", True)
    fig.colorbar(im, ax=axes[2], shrink=.7, label="m")
    fig.subplots_adjust(bottom=.12, top=.96, hspace=.28)
    save(fig, "72_world_comparison.png", "固定1024×512输出用于全球对照；守恒不等于平顶、高原分配或盆地均已修好。")

    xx = (lon >= -15) & (lon <= 50)
    yy = (lat >= -25) & (lat <= 45)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, case in zip(axes, ("control", "conservative")):
        im = map_ax(ax, lon[xx], lat[yy], worlds[case][np.ix_(yy, xx)], case + "：C/D与主内海")
        fig.colorbar(im, ax=ax, shrink=.55, label="m")
    im = map_ax(axes[2], lon[xx], lat[yy], d[np.ix_(yy, xx)], "候选 − 对照", True)
    fig.colorbar(im, ax=axes[2], shrink=.55, label="m")
    for ax in axes:
        ax.plot(*np.asarray(meander).T, c="#d12a9b", lw=1.2)
        ax.text(20, -3, "B5", color="#d12a9b")
    fig.subplots_adjust(bottom=.16, top=.93, wspace=.3)
    save(fig, "73_rift_regions.png", "同窗口暴露裂谷扩张、内海、走滑盆地及邻岸副作用。差值±1500m，超出色带饱和。洋壳生成等其他规则未改。")

    fig = plt.figure(figsize=(16, 11))
    axes = [fig.add_subplot(2, 2, k) for k in (1, 2)]
    for ax, case in zip(axes, ("control", "conservative")):
        im = map_ax(ax, blon, blat, maps[case]["elev"], case + "：B5直接原生（灰=无覆盖）")
        ax.plot(*np.asarray(meander).T, c="#d12a9b", lw=1.2)
        ax.plot(axis_lon, axis_lat, "--", c="#333", lw=.7)
        ax.scatter([18.675], [-2.5], marker="+", c="#f6268f")
        fig.colorbar(im, ax=ax, shrink=.75, label="m")
    ax = fig.add_subplot(2, 1, 2)
    for case, color in (("control", "#777"), ("conservative", "#74449a")):
        ax.plot(axis_lat, profiles[case]["elev"], c=color, label=case)
    ax.axhline(0, c="#222", lw=.7)
    ax.axhline(-400, c="#d12a9b", ls="--", label="设计谷底参考 −400m")
    ax.set(xlabel="B5观察轴纬度 °N（虚线所示）", ylabel="原生高程 m", title="同一观察轴：有低点不等于形成连续盆底")
    ax.grid(alpha=.2)
    ax.legend()
    fig.subplots_adjust(bottom=.15, top=.94, hspace=.35)
    save(fig, "74_b5_native.png", "粉线为实际扰动后的B5设计轮廓；中心18.675°E/2.5°S。只查询L7材料，原生空隙断线，不借东侧水体代替B5。")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    beta = np.concatenate([r["source_beta"] for r in records])
    birth = np.concatenate([r["target_thickness_km"] for r in records])
    axes[0, 0].hist(beta, bins=35, color="#74449a")
    axes[0, 0].set(title="每次提交供体面积展开比例", xlabel="β=1+分配新增面积/旧面积", ylabel="供体记录数（可重复）")
    axes[0, 1].hist(birth, bins=35, color="#74449a")
    axes[0, 1].axvline(7, c="#d12a9b", ls="--", label="7km观察线")
    axes[0, 1].set(title="裂谷接收体出生厚度", xlabel="km", ylabel="新单元数")
    axes[0, 1].legend()
    for case, color in (("control", "#777"), ("conservative", "#74449a")):
        eng = payloads[case]["engine"]
        mask_c = eng.alive & (eng.c.ctype == CONT)
        axes[1, 0].hist(eng.c.T[mask_c], bins=np.arange(0, 67, 1), weights=eng.area[mask_c]/1e6,
                        histtype="step", lw=1.5, color=color, label=case)
    axes[1, 0].set(title="现今陆壳厚度分布", xlabel="km", ylabel="百万km²")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    lines = ["实际守恒闭合与剩余问题", f"转移 {gain/1e6:.3f} 百万km³（内部）",
             f"最大逐次残差 {audit['max_commit_abs_closure_km3']:.3g} km³",
             f"出生厚度范围 {birth.min():.2f}–{birth.max():.2f} km",
             f"单次β范围 {beta.min():.3f}–{beta.max():.3f}",
             "厚度不设下限；过薄陆壳须单列",
             "未求解断层、应力、热结构或组分混合",
             "旧hole_created与65km平台仍需处理"]
    axes[1, 1].text(.05, .95, "\n\n".join(lines), va="top", fontsize=11)
    fig.tight_layout(rect=(0, .13, 1, .96))
    save(fig, "75_thinning_and_limits.png", "β仅为本次面积分摊因子，不是从300Ma累积恢复出的连续应变；7km不是本项目验收阈值，未据此调参。")

    images = sorted(OUT.glob("*.png"))
    html = '<meta charset="utf-8"><title>U36-S2 守恒面积展开试验</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%}</style><h1>U36-S2：实际历史候选与对照，未采用</h1>'
    html += '<p>五张专项图；完整标准图组与独立评审另见交接文档。守恒改善不能替代世界观感评审。</p>'
    html += "".join(f'<h2>{p.name}</h2><img src="{p.name}">' for p in images)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(sec.clean(dict(compatibility=compatibility, transport=audit, B5={case: {k:v for k,v in row.items() if k not in ("axis_lat","axis_native_m")} for case,row in b5.items()}, cases=cases, global_change=global_change)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
