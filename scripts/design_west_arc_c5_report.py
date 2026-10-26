"""U35-C5：只读核查 M31 端部的实体覆盖、双重取样和规定缩短权重。

先用 design_west_arc_diagnostic.py 完整复算 C4；本脚本不修改地形或历史。
连续位置查询沿用引擎的旋转参照网格覆盖规则，不产生更细的科学 DEM。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_west_arc_report as base
from design_final_map import densify
from mythic.tectonics.geometry import km_from_chord, lonlat_to_xyz, xyz_to_lonlat
from mythic.tectonics.mesh import load_mesh
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm

PROTECTED = (
    "config/design/history_v2.yaml", "config/design/history_v2_u35_c4_strain.yaml",
    "config/design/final_map_v2.yaml", "config/metrics.yaml",
    "data/worlds/design_v2/history_v2.rot", "data/worlds/design_v2/forward_v2.json",
)


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}


def read_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def sample_grid(grid, points):
    lon, lat = xyz_to_lonlat(points)
    i = np.clip(((90 - lat) / 180 * grid.ny).astype(int), 0, grid.ny - 1)
    j = ((lon + 180) / 360 * grid.nx).astype(int) % grid.nx
    return i, j


def material_cover(mesh, parcels, points):
    """重建各板块参照旋转并查询原生 Voronoi 单元；重叠按引擎的陆壳/厚度/年龄优先级选取。

    只读观察文件只存局部存活单元，故调用方须在保存窗口内部查询，并与完整固定网格核对。
    不按最近陆地填洞，不跨海插值。返回 -1 的位置没有被本地记录覆盖。
    """
    pos = lonlat_to_xyz(parcels["lon"], parcels["lat"])
    best = np.full(len(points), -1, dtype=int)
    score = np.full(len(points), -np.inf)
    fits = []
    for q in np.unique(parcels["plate"]):
        ix = np.flatnonzero(parcels["plate"] == q)
        ref = mesh.xyz[parcels["node"][ix]]
        if len(ix) < 3 or np.linalg.matrix_rank(ref) < 3:
            raise ValueError(f"板块 {q} 在保存窗口内不足以重建旋转")
        # pos = ref @ R.T；投影到最近正交旋转，排除舍入引起的伸缩。
        a = np.linalg.lstsq(ref, pos[ix], rcond=None)[0]
        u, _, vt = np.linalg.svd(a)
        rotation_t = u @ vt
        if np.linalg.det(rotation_t) <= 0:
            raise ValueError("不能以反射代替板块旋转")
        err = km_from_chord(np.linalg.norm(ref @ rotation_t - pos[ix], axis=1))
        fits.append({"plate": int(q), "parcels": len(ix), "max_fit_error_km": float(err.max())})
        _, nodes = mesh.tree.query(points @ rotation_t.T, workers=4)
        owner = np.full(mesh.n, -1, dtype=int)
        owner[parcels["node"][ix]] = ix
        hit = owner[nodes]
        valid = hit >= 0
        safe = np.maximum(hit, 0)
        s = parcels["ctype"][safe] * 1e6 + parcels["T"][safe] * 1e3 - parcels["age"][safe]
        use = valid & (s > score)
        best[use] = hit[use]
        score[use] = s[use]
    return best, fits


def tail_length(distance, wet):
    """从线尾向回到最后一个干采样点；离散采样长度，不声称精确海岸位置。"""
    dry = np.flatnonzero(~wet)
    return float(distance[-1] - distance[dry[-1]]) if dry.size else float(distance[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=Path(".cache/u35_c5/reproduced_c4/latest"))
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    if not run.is_relative_to(ROOT):
        ap.error("运行结果必须位于项目工作区内")
    before = hashes()
    out = ROOT / "figures/design/v2/forward/u35_c5"
    out.mkdir(parents=True, exist_ok=True)
    base.setup_fonts()
    mesh = load_mesh(7)
    grid = base.GRID
    design = fwd.apply_options(fwd.yaml.safe_load((ROOT / "config/design/final_map_v2.yaml").read_text(encoding="utf-8")), set())
    hcfg = fwd.yaml.safe_load((ROOT / "config/design/history_v2_u35_c4_strain.yaml").read_text(encoding="utf-8"))
    built = fwd.build(design, grid)
    mountain = next(m for m in design["mountains"] if m["id"] == "M31")
    points, distance = densify(mountain["line"], 2.0)
    lon, lat = xyz_to_lonlat(points)
    ii, jj = sample_grid(grid, points)
    _, nodes = mesh.tree.query(points, workers=4)
    parcels = read_npz(run / "parcels_000.npz")
    snap = read_npz(run / "snapshot_000.npz")
    with open_netcdf(run / "world.nc") as ds:
        E = ds.elevation.values.copy()
        land = ds.land.values > .5
    cover, fits = material_cover(mesh, parcels, points)
    if np.any(cover < 0):
        raise RuntimeError("山脊查询超出已保存实体范围，不能下覆盖结论")
    raw = parcels["raw_elevation_m"][cover]
    direct = snap["elev"][nodes]
    exported = E[ii, jj]
    # 先验证重建的覆盖查询，不能把局部最近单元误称为引擎覆盖。
    extent = (-18.5, -10.5, 34.5, 40.)
    sel = (mesh.lon > extent[0]) & (mesh.lon < extent[1]) & (mesh.lat > extent[2]) & (mesh.lat < extent[3])
    checked, _ = material_cover(mesh, parcels, mesh.xyz[sel])
    valid = checked >= 0
    view_check = {k: int(np.sum(parcels[k][checked[valid]] != snap[k][sel][valid])) for k in ("plate", "ctype", "origin")}
    validation_differences = []
    for fixed_node, p in zip(np.flatnonzero(sel)[valid], checked[valid]):
        if any(parcels[k][p] != snap[k][fixed_node] for k in ("plate", "ctype", "origin")):
            validation_differences.append({"node": int(fixed_node), "lon": float(mesh.lon[fixed_node]), "lat": float(mesh.lat[fixed_node]),
                "material_ctype": int(parcels["ctype"][p]), "snapshot_ctype": int(snap["ctype"][fixed_node]),
                "material_age_Myr": float(parcels["age"][p])})
    ridge_nodes = np.unique(nodes)
    ridge_cover, _ = material_cover(mesh, parcels, mesh.xyz[ridge_nodes])
    ridge_matches = bool(np.all(ridge_cover >= 0) and all(
        np.array_equal(parcels[k][ridge_cover], snap[k][ridge_nodes]) for k in ("plate", "ctype", "origin")))
    if not ridge_matches:
        raise RuntimeError("山脊附近的覆盖查询与已保存节点不一致，不能归因为取样")
    # snapshot 可以显示填洞；不掩盖任何差异，查询结论只针对已核对的山脊实体。
    native_to_output = mesh.tree.query(grid.xyz[ii, jj], workers=4)[1]
    source = (parcels["origin"] == 304) & (parcels["ctype"] == 1)
    source_xyz = lonlat_to_xyz(parcels["lon"][source], parcels["lat"][source])
    nearest_source_km = km_from_chord(cKDTree(source_xyz).query(points)[0])
    react = next(r for r in fwd.build_reactivations(hcfg, design, grid, built["land"]) if r["id"] == "M31_internal")
    si, sj = sample_grid(grid, source_xyz)
    weights = react["W"][si, sj]
    core = source.copy()
    core[source] = weights == 1.
    reference = base.read_json(ROOT / "data/worlds/design_v2/u35_c4_diagnostic.json")
    summary = base.read_json(run / "summary.json")
    budgets = base.read_json(run / "forward.json")["budget"]
    m31 = next(m for m in summary["comparison"]["mountains"] if m["id"] == "M31")
    expected_budget = reference["budget_km3"]["C4"]
    budget_deltas = {k: budgets[k] - expected_budget[k] for k in expected_budget}
    # 跨机浮点累加允许舍入误差；这不是质量指标容差，也不称为逐位相同。
    budget_close = set(budgets) == set(expected_budget) and all(
        np.isclose(budgets[k], v, rtol=1e-12, atol=1e-8) for k, v in expected_budget.items())
    reproduction = {"M31": m31 == reference["M31"]["C4"],
                    "budget_within_roundoff": bool(budget_close),
                    "A": summary["A_engine_0Ma"] == reference["A"]["C4"]}
    if not all(reproduction.values()):
        raise RuntimeError("C4 复现未对齐，停止解释新结果：" + str(reproduction))
    rows = []
    for k in range(len(points)):
        p = int(cover[k])
        rows.append({"s_km": float(distance[k]), "lon": float(lon[k]), "lat": float(lat[k]),
                     "parcel_uid": int(parcels["uid"][p]), "plate": int(parcels["plate"][p]),
                     "origin": int(parcels["origin"][p]), "ctype": int(parcels["ctype"][p]),
                     "T_km": float(parcels["T"][p]), "raw_height_m": float(raw[k]),
                     "native_fixed_height_m": float(direct[k]), "output_height_m": float(exported[k]),
                     "native_fixed_node": int(nodes[k]), "output_owner_node": int(native_to_output[k]),
                     "nearest_original_source_km": float(nearest_source_km[k])})
    history = []
    uid_now = set(map(int, parcels["uid"][source]))
    for t in (38, 30, 10, 6, 5, 0):
        p = read_npz(run / f"parcels_{t:03d}.npz")
        orig = (p["origin"] == 304) & (p["ctype"] == 1)
        history.append({"t_Ma": t, "n": int(orig.sum()),
                        "surviving_at_0Ma": len(set(map(int, p["uid"][orig])) & uid_now),
                        "source_area_km2": float(p["area"][orig].sum()),
                        "T_min_km": float(p["T"][orig].min()), "T_max_km": float(p["T"][orig].max())})
    wet_raw = raw <= 0
    wet_direct = direct <= 0
    wet_export = exported <= 0
    stats = {"ridge_length_km": float(distance[-1]), "source_n": int(source.sum()),
             "full_weight_source_n": int(core.sum()), "source_weight_minmax": [float(weights.min()), float(weights.max())],
             "core_raw_height_minmax_m": [float(parcels["raw_elevation_m"][core].min()), float(parcels["raw_elevation_m"][core].max())],
             "raw_tail_wet_km": tail_length(distance, wet_raw),
             "ridge_raw_height_minmax_m": [float(raw.min()), float(raw.max())],
             "ridge_source_origins": sorted(set(map(int, parcels["origin"][cover]))),
             "fixed_tail_wet_km": tail_length(distance, wet_direct),
             "output_tail_wet_km": tail_length(distance, wet_export),
             "different_native_output_nodes_samples": int((nodes != native_to_output).sum()),
             "samples": len(points), "unique_raw_parcels": len(set(cover.tolist())),
             "unique_fixed_nodes": len(set(nodes.tolist())), "unique_output_cells": len(set(zip(ii.tolist(), jj.tolist()))),
             "raw_wet_ctype": sorted(set(map(int, parcels["ctype"][cover[wet_raw]]))),
             "view_validation": {"nodes": int(sel.sum()), "covered": int(valid.sum()), "different": view_check,
                                 "differences": validation_differences,
                                 "ridge_nodes_checked": len(ridge_nodes), "ridge_nodes_match": ridge_matches}}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    base.map_panel(axes[0], E, land, "C4：输出像元与原 ALB 宿主单元", extent=extent, contour=built["land"])
    base.feature_lines(axes[0], design)
    axes[0].scatter(parcels["lon"][source], parcels["lat"][source], s=25, facecolors="none", edgecolors="#111", label="原 ALB 单元中心")
    axes[0].plot(lon, lat, color="#ffec93", lw=2, label="M31 认可山脊")
    axes[0].legend(fontsize=8)
    sx = (grid.lon >= extent[0]) & (grid.lon <= extent[1])
    sy = (grid.lat >= extent[2]) & (grid.lat <= extent[3])
    glon, glat = np.meshgrid(grid.lon[sx], grid.lat[sy])
    exact, _ = material_cover(mesh, parcels, lonlat_to_xyz(glon.ravel(), glat.ravel()))
    uidmap = np.where(exact >= 0, parcels["origin"][np.maximum(exact, 0)], -1).reshape(glon.shape)
    axes[1].pcolormesh(glon, glat, np.where(uidmap == 304, 1., 0.), cmap="Greys", vmin=0, vmax=1, shading="nearest", alpha=.45)
    axes[1].plot(lon, lat, "-", c="#743d7f", lw=2, label="认可山脊")
    for k in np.unique(cover):
        hit = cover == k
        mid = np.flatnonzero(hit)[len(np.flatnonzero(hit)) // 2]
        axes[1].scatter(lon[hit], lat[hit], s=12, c="#238476" if parcels["ctype"][k] == 1 else "#d35245")
        axes[1].annotate(str(int(parcels["uid"][k])), (lon[mid], lat[mid]), xytext=(0, 8), textcoords="offset points", fontsize=8)
    axes[1].set(xlim=extent[:2], ylim=extent[2:], aspect="equal", title="实体覆盖：深灰为原 ALB 来源宿主\n山脊绿色=陆壳、红色=洋壳；数字=永久 UID", xlabel="经度 °E", ylabel="纬度 °N")
    axes[1].grid(alpha=.25)
    fig.subplots_adjust(bottom=.22, top=.85, wspace=.18)
    stamp(fig, "实体覆盖重建自参照格网与板块旋转；图中格点只是查询位置，未加密物理模型。")
    save(fig, out / "35_m31_material_coverage.png")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for h, label, color in ((exported, "C4：经纬格网再取样", "#2789a1"), (direct, "直接取固定 L7 节点", "#bd8a3d"), (raw, "连续位置实体覆盖（无海沟显示项）", "#4b3e72")):
        axes[0].plot(distance, h, label=label, color=color)
    axes[0].axhline(0, lw=.8, c="black")
    axes[0].set(ylabel="高程 m", title="同一 L7 结果的三种查询：取样差异与实体海段分开")
    axes[0].legend(fontsize=9)
    axes[1].plot(distance, nearest_source_km, color="#515b60", label="到最近原 ALB 单元中心的距离")
    for name, wet, color in (("实体水下段", wet_raw, "#4b3e72"), ("输出水下段", wet_export, "#2789a1")):
        axes[1].fill_between(distance, 0, nearest_source_km, where=wet, color=color, alpha=.2, label=name)
    axes[1].set(xlabel="沿认可山脊距离 km（西→东）", ylabel="km")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(rect=(0, .15, 1, .96))
    stamp(fig, f"2 km 是查询步距；L7 平均节点间距 {mesh.h_km:.1f} km。密集查询不能增加山峰或海岸分辨率。")
    save(fig, out / "36_m31_sampling_profile.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(parcels["lon"][source], parcels["lat"][source], c=weights, cmap="viridis", vmin=0, vmax=1, s=55, edgecolors="#222")
    axes[0].plot(lon, lat, "--", color="#743d7f")
    axes[0].set(aspect="equal", xlabel="经度 °E", ylabel="纬度 °N", title="原 ALB 单元的规定增厚权重（现今参照系）")
    fig.colorbar(sc, ax=axes[0], shrink=.7, label="无量纲；非求解的应变场")
    axes[1].plot([r["t_Ma"] for r in history], [r["n"] for r in history], "-o", label="当时的原 ALB 单元")
    axes[1].plot([r["t_Ma"] for r in history], [r["surviving_at_0Ma"] for r in history], "--x", label="其中现今仍存活")
    axes[1].set(xlim=(38, 0), xlabel="距今 Ma", ylabel="单元数", title="按 UID 追踪来源单元，核查是否被消耗")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(rect=(0, .16, 1, .95))
    stamp(fig, "C4 的规定缩短保留原来的平顶权重核；本图只测量处方覆盖，不把它称为动力学验证。")
    save(fig, out / "37_m31_weights_and_survival.png")
    # 直接展示原生单元覆盖边界；更密的查询只让已有单元可见，不增加物理细节。
    qlon, qlat = np.meshgrid(np.linspace(extent[0], extent[1], 321), np.linspace(extent[2], extent[3], 221))
    lookup, _ = material_cover(mesh, parcels, lonlat_to_xyz(qlon.ravel(), qlat.ravel()))
    # 固定节点覆盖完整不保证节点之间无空隙；保留并明确显示这些空白，绝不填陆或填海。
    uncovered = (lookup < 0).reshape(qlon.shape)
    stats["native_window_uncovered_queries"] = int(uncovered.sum())
    stats["native_window_queries"] = len(lookup)
    qheight = np.ma.array(parcels["raw_elevation_m"][np.maximum(lookup, 0)].reshape(qlon.shape), mask=uncovered)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    base.map_panel(axes[0], E, land, "既有输出：固定节点 → 1024×512", extent=extent, contour=built["land"])
    cmap, norm = elevation_cmap_norm()
    axes[1].pcolormesh(qlon, qlat, qheight, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    axes[1].set_facecolor("#a7a7a7")
    axes[1].contour(qlon, qlat, qheight > 0, levels=[.5], colors="#222", linewidths=.8)
    axes[1].set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="经度 °E", ylabel="纬度 °N",
                title="原生单元观察：按板块参照格网覆盖查询\n无山体阴影、无海沟显示项、无新增 DEM 细节")
    for ax in axes:
        ax.plot(lon, lat, color="#743d7f", lw=2)
        ax.scatter(parcels["lon"][source], parcels["lat"][source], s=10, c="#222")
        ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22, top=.86, wspace=.16)
    base.elevation_legend(fig, .13)
    fig.axes[-1].set_xlabel("高程 m；左图含阴影，右图无阴影", fontsize=8)
    stamp(fig, "右图灰色=无原生单元覆盖，未填陆/填海；只显示 L7 单元边界，不提高模拟分辨率。未替换正式输出或指标。")
    save(fig, out / "38_m31_native_cells.png")
    after = hashes()
    if before != after:
        raise RuntimeError("只读报告意外修改了保护文件")
    result = {"checkpoint": "U35-C5", "kind": "read_only_material_and_sampling_diagnosis",
              "baseline_commit": "db02e15", "candidate_adopted": False, "terrain_changed": False,
              "source_run": run.relative_to(ROOT).as_posix(), "reproduction": reproduction,
              "budget_comparison": {"exact": budgets == expected_budget, "rtol": 1e-12, "atol_km3": 1e-8, "deltas_km3": budget_deltas},
              "resolution": {"mesh_level": 7, "mean_node_spacing_km": mesh.h_km, "export": "1024x512", "query_step_km": 2.},
              "method_class": {"tectonics": "procedural_constrained", "height": "parameterized_physics", "prescribed_shortening": "user_override", "export": "interpolated"},
              "stats": stats, "rotation_fit": fits, "source_history": history, "ridge_profile": rows,
              "protected_sha256": before, "protected_unchanged": before == after,
              "limitations": ["查询步距不等于 DEM 分辨率", "局部覆盖重建需由固定节点对照验证", "实体高程未加海沟显示及动力地形", "未修复端部、海口填隙或 R17，也未采用 C4"]}
    base.write_json(ROOT / "data/worlds/design_v2/u35_c5_diagnostic.json", result)
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C5 山弧实体与取样核查</title><style>body{margin:24px;font-family:system-ui;background:#eee}img{max-width:100%}</style><h1>U35-C5：山弧实体与取样核查</h1><p>只读重算 C4，尚未采用候选；没有改动地形、终图、运动或指标。</p>"
    for name in ("35_m31_material_coverage.png", "36_m31_sampling_profile.png", "37_m31_weights_and_survival.png", "38_m31_native_cells.png"):
        html += f"<h2>{name}</h2><img src='{name}'>"
    html += "<p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c4_strain/index.html'>C4 标准图组与指标</a></p>"
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    print(json.dumps({"reproduction": reproduction, "stats": stats, "source_history": history, "protected_unchanged": before == after}, ensure_ascii=False, indent=2))


def stamp(fig, note):
    fig.text(.015, .018, "U35-C5 · C4 只读诊断 · " + fwd.code_version() +
             "\n地壳 procedural_constrained；高程 parameterized_physics；终图/规定缩短 user_override；栅格 interpolated\n" + note,
             fontsize=8, va="bottom")


def save(fig, path):
    fig.savefig(path, dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
