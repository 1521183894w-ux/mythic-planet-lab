"""U35-C7：从C6已捕获状态核查M31横剖面、二维宽度、两端和B5定位。

只读诊断，不修改历史、地壳或正式输出。先按C5/C6交接准备本地缓存。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_final_map as final
import design_west_arc_report as base
from design_native_output import dynamic_height
from design_west_arc_c5_report import hashes, sample_grid
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm
from mythic.tectonics.crust import Crust
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, xyz_to_lonlat
from mythic.tectonics.sampling import prepare_output, sample_output
from mythic.tectonics.sections import spherical_sections, axis_connected_span

OUT = ROOT / "figures/design/v2/forward/u35_c7"
CACHE = ROOT / ".cache/u35_c7"
DATA = ROOT / "data/worlds/design_v2/u35_c7_diagnostic.json"


def state_hash(eng):
    h = hashlib.sha256()
    for name in Crust.FIELDS:
        h.update(getattr(eng.c, name).tobytes())
    h.update(eng.alive.tobytes())
    h.update(eng.area.tobytes())
    for q in sorted(eng.R):
        h.update(eng.R[q].tobytes())
    h.update(json.dumps([eng.t, eng.budget, eng.events], sort_keys=True).encode())
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def stats(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return {"n": len(a), "min": float(a.min()), "p50": float(np.median(a)), "max": float(a.max())} if len(a) else {"n": 0}


def save(fig, filename, note):
    fig.text(.012, .014, "U35-C7 · " + fwd.code_version() +
        "\n同一C4/C6地壳；规定历史 user_override；地壳 procedural_constrained；高程 parameterized_physics；观察取样 interpolated\n" + note,
        fontsize=8, va="bottom")
    fig.savefig(OUT / filename, dpi=130)
    plt.close(fig)


def local_map(bounds, step=.025):
    x0, x1, y0, y1 = bounds
    lon = np.linspace(x0, x1, round((x1 - x0) / step) + 1)
    lat = np.linspace(y0, y1, round((y1 - y0) / step) + 1)
    lo, la = np.meshgrid(lon, lat)
    # 每个查询中心的球面矩形面积；精细的是观察积分，非地壳分辨率。
    dlon, dlat = np.radians(lon[1] - lon[0]), np.radians(lat[1] - lat[0])
    a = R_KM ** 2 * dlon * (np.sin(np.radians(lat) + dlat / 2) - np.sin(np.radians(lat) - dlat / 2))
    return lon, lat, lonlat_to_xyz(lo, la), np.broadcast_to(a[:, None], lo.shape)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    base.setup_fonts()
    protected = hashes()
    with (ROOT / ".cache/u35_c6/final_state.pkl").open("rb") as f:
        payload = pickle.load(f)
    eng = payload["engine"]
    before = state_hash(eng)
    ctx = prepare_output(eng, payload["fixed_frame"])
    design = fwd.yaml.safe_load((ROOT / "config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    history = fwd.yaml.safe_load((ROOT / "config/design/history_v2_u35_c4_strain.yaml").read_text(encoding="utf-8"))
    mountain = next(x for x in design["mountains"] if x["id"] == "M31")
    basin = next(x for x in design["basins"] if x["id"] == "B5")
    built = fwd.build(design, base.GRID)
    design_e = fwd.design_elevation(design, base.GRID, built)
    reaction = fwd.build_reactivations(history, design, base.GRID, built["land"])[0]
    assert reaction["id"] == "M31_internal"
    worlds = {}
    for key, path in {"C4": ".cache/u35_c5/reproduced_c4/latest/world.nc",
                      "C6_1024": ".cache/u35_c6/worlds/native_fill_1024x512.nc",
                      "C6_2048": ".cache/u35_c6/worlds/native_fill_2048x1024.nc"}.items():
        with open_netcdf(ROOT / path) as ds:
            worlds[key] = ds.elevation.values.copy()

    def query(points):
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        q = sample_output(ctx, flat, dynamic_height(payload, flat), carry_view_fill=False)
        covered = q["owner"] >= 0
        # 不让fallback进入“实体曲线”；另行保存未覆盖位置。
        q["elev"][~covered] = np.nan
        q["T"][~covered] = np.nan
        for field in ("uid", "origin", "plate", "ctype"):
            q[field] = q[field].astype(np.int64)
            q[field][~covered] = -1
        q["covered"] = covered
        return {k: v.reshape(shape) for k, v in q.items()}

    def raster_sample(points, field):
        from mythic.metrics.grid import LatLonGrid
        grid = base.GRID if field.shape == base.GRID.shape else LatLonGrid(field.shape[1], field.shape[0])
        i, j = sample_grid(grid, points.reshape(-1, 3))
        return field[i, j].reshape(points.shape[:-1])

    # 五条正交横剖面与整条带状观察窗，含两端外延。
    _, length = spherical_sections(mountain["line"], [0.], [0.])
    along = np.linspace(0., length, 5)
    cross = np.arange(-300., 301., 5.)
    points, _ = spherical_sections(mountain["line"], along, cross)
    sections = query(points)
    comparisons = {name: raster_sample(points, e) for name, e in worlds.items()}
    schematic = raster_sample(points, design_e)
    rows = []
    for i, s in enumerate(along):
        center = len(cross) // 2
        lon, lat = xyz_to_lonlat(points[i, center])
        host = sections["covered"][i] & (sections["origin"][i] == 304) & (sections["plate"][i] == 304)
        host_span = axis_connected_span(cross, sections["elev"][i], 1500., host)
        if host_span["axis_above"]:
            for side, outside in (("left", 0), ("right", 1)):
                bracket = host_span[side + "_bracket_km"]
                if bracket is not None:
                    j = int(np.flatnonzero(cross == bracket[outside])[0])
                    if sections["covered"][i, j] and not host[j]:
                        host_span[side + "_stop"] = "other_host"
        rows.append({"section": f"S{i + 1}", "along_km": float(s), "axis_lon_lat": [float(lon), float(lat)],
            "native_axis_m": float(sections["elev"][i, center]),
            "native_axis_uid": int(sections["uid"][i, center]),
            "native_spans": [axis_connected_span(cross, sections["elev"][i], h, sections["covered"][i]) for h in (1000., 1500., 2000.)],
            "native_304_host_1500m_span": host_span,
            "C6_1024_spans": [axis_connected_span(cross, comparisons["C6_1024"][i], h) for h in (1000., 1500., 2000.)],
            "native_distinct_uids": len(np.unique(sections["uid"][i, sections["covered"][i]]))})
    s2 = np.unique(np.r_[np.arange(-180., length + 181., 5.), 0., length])
    n2 = np.arange(-240., 241., 5.)
    strip_points, _ = spherical_sections(mountain["line"], s2, n2)
    strip = query(strip_points)
    endpoint = query(strip_points[:, [len(n2) // 2]])
    width_rows = [axis_connected_span(n2, h, 1500., ok) for h, ok in zip(strip["elev"], strip["covered"])]

    lon, lat, xyz, _ = local_map((-20., -10., 34., 41.))
    local = query(xyz)
    map_W = raster_sample(xyz, reaction["W"])
    # 原ALB宿主中心上的处方权重，与模拟的最近栅格取权重一致。
    pos = eng.positions()
    source = eng.alive & (eng.c.origin == 304) & (eng.c.ctype == 1)
    u = np.flatnonzero(source)
    w = raster_sample(pos[u], reaction["W"])
    source_rows = [{"uid": int(eng.c.uid[j]), "lon": float(x), "lat": float(y), "T_km": float(eng.c.T[j]), "weight_0Ma": float(ww)}
                   for j, x, y, ww in zip(u, *xyz_to_lonlat(pos[u]), w)]

    # B5同时画原始顶点、实际扰动后的设计轮廓，避免用自动r8代替定位。
    blon, blat, bxyz, barea = local_map((16.5, 21., -6., 1.))
    bq = query(bxyz)
    blo, bla = np.meshgrid(blon, blat)
    meander = final.meander_polygon(basin["poly"], final.feature_rng("B5"), basin["meander_km"])
    masks = {"vertices_polygon": MPath(basin["poly"]).contains_points(np.c_[blo.ravel(), bla.ravel()]).reshape(blo.shape),
             "design_meander_polygon": MPath(meander).contains_points(np.c_[blo.ravel(), bla.ravel()]).reshape(blo.shape)}
    center = np.mean(np.asarray(basin["poly"]), axis=0)
    cp = lonlat_to_xyz(*center)[None, :]
    cq = query(cp)
    basin_rows = {}
    for name, mask in masks.items():
        valid = mask & bq["covered"]
        basin_rows[name] = {"area_km2": float(barea[mask].sum()),
            "covered_area_fraction": float(barea[valid].sum() / barea[mask].sum()),
            "native_height_m": stats(bq["elev"][valid]), "native_T_km": stats(bq["T"][valid]),
            "native_below_sea_fraction_of_covered_area": float(barea[valid & (bq["elev"] < 0)].sum() / barea[valid].sum()),
            "native_uids": sorted(int(x) for x in np.unique(bq["uid"][valid])),
            "native_origins": sorted(int(x) for x in np.unique(bq["origin"][valid])),
            "native_last_event_counts_queries": {int(k): int(np.sum(bq["oro"][valid] == k)) for k in np.unique(bq["oro"][valid])},
            "output_height_m": {k: stats(raster_sample(bxyz, e)[mask]) for k, e in worlds.items()},
            "schematic_height_m": stats(raster_sample(bxyz, design_e)[mask])}
    bsections, blength = spherical_sections([[18.65, -.4], [18.7, -4.6]], [0.], [0.])
    bs = np.linspace(0., blength, 5)[1:-1]
    bcross = np.arange(-220., 221., 5.)
    bpoints, _ = spherical_sections([[18.65, -.4], [18.7, -4.6]], bs, bcross)
    bp = query(bpoints)
    px, py = xyz_to_lonlat(bpoints)
    profile_in_basin = MPath(meander).contains_points(np.c_[px.ravel(), py.ravel()]).reshape(px.shape)
    result = {"checkpoint": "U35-C7", "baseline_commit": "972637c", "method_class": {"history": "user_override",
        "crust": "procedural_constrained", "elevation": "parameterized_physics", "sections": "interpolated", "design_elevation": "visual_only"},
        "native_node_spacing_km": float(eng.m.h_km), "query_step_cross_km": 5., "physics_changed": False,
        "M31": {"line": mountain["line"], "length_km": length, "design_width_km": mountain["width_km"],
            "design_width_meaning": "终图高斯横剖面尺度；不是1500m等高带的合格线", "cross_positive": "沿线向东行进时的左侧，约西北",
            "sections": rows, "cross_offset_km": cross, "native_profiles_m": sections["elev"], "native_profiles_uid": sections["uid"],
            "native_profiles_origin": sections["origin"], "native_profiles_plate": sections["plate"],
            "native_profiles_covered": sections["covered"], "output_profiles_m": comparisons, "design_profiles_m": schematic,
            "strip_along_km": s2, "strip_1500m_spans": width_rows,
            "endpoint_axis_m": endpoint["elev"].ravel(), "endpoint_axis_uid": endpoint["uid"].ravel(),
            "endpoint_axis_origin": endpoint["origin"].ravel(), "endpoint_axis_plate": endpoint["plate"].ravel(),
            "endpoint_axis_covered": endpoint["covered"].ravel(), "source_parcels": source_rows,
            "source_weight_core_n": int(np.sum(w >= 1 - 1e-12)), "source_weight_transition_n": int(np.sum((w > 1e-3) & (w < 1 - 1e-12)))},
        "B5": {"design": basin, "center_lon_lat": center, "center_native": {k: v for k, v in cq.items() if k in ("elev", "T", "uid", "origin", "plate", "oro", "oro_t", "covered")},
            "center_outputs_m": {k: float(raster_sample(cp, e)[0]) for k, e in worlds.items()}, "polygons": basin_rows,
            "cross_offset_km": bcross, "section_along_km": bs, "native_profiles_m": bp["elev"], "native_profiles_T_km": bp["T"]},
        "protected_sha256": protected, "protected_unchanged": hashes() == protected,
        "physical_state_sha256": before, "physical_state_unchanged": state_hash(eng) == before,
        "limitations": ["只读现今状态，未追踪B5演化收支，不能从末帧唯一归因", "灰色未覆盖区不使用固定视图回退",
            "5km查询不是5km物理精度", "高度阈值宽度仅为诊断，不替代质量指标", "终图高程为visual_only，非物理结果"]}
    result["B5"]["profile_in_design_polygon"] = profile_in_basin
    result["B5"]["profile_axis_lon_lat"] = np.stack(xyz_to_lonlat(bpoints[:, len(bcross) // 2]), axis=-1)
    if not result["protected_unchanged"] or not result["physical_state_unchanged"]:
        raise RuntimeError("只读核查改变了保护文件或物理状态")
    result["wall_s_before_plot"] = round(time.perf_counter() - started, 2)
    base.write_json(DATA, clean(result))
    np.savez_compressed(CACHE / "maps.npz", lon=lon, lat=lat, E=local["elev"], T=local["T"], W=map_W,
        origin=local["origin"], plate=local["plate"], covered=local["covered"], strip_E=strip["elev"], strip_T=strip["T"],
        strip_origin=strip["origin"], strip_ctype=strip["ctype"], strip_covered=strip["covered"],
        blon=blon, blat=blat, bE=bq["elev"], bT=bq["T"], bplate=bq["plate"], boro=bq["oro"],
        bcovered=bq["covered"], meander=meander)
    render(clean(result))
    print(json.dumps(clean({"M31_sections": rows, "B5_center": result["B5"]["center_native"],
        "B5_polygons": basin_rows, "state_unchanged": result["physical_state_unchanged"], "total_wall_s": round(time.perf_counter() - started, 2)}), ensure_ascii=False, indent=2))
    return 0


def render(result):
    with np.load(CACHE / "maps.npz") as z:
        m = {k: z[k] for k in z.files}
    mountain = result["M31"]
    basin = result["B5"]
    cmap, norm = elevation_cmap_norm()
    cmap = cmap.copy()
    cmap.set_bad("#888888")

    def axes_map(ax, lon, lat, value, title, cm=cmap, nm=norm):
        if isinstance(cm, str):
            cm = matplotlib.colormaps[cm].copy()
            cm.set_bad("#888888")
        im = ax.pcolormesh(lon, lat, value, cmap=cm, norm=nm, shading="nearest", rasterized=True)
        ax.set(title=title, xlabel="经度 °E", ylabel="纬度 °N")
        ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
        ax.grid(alpha=.2)
        return im

    line = np.asarray(mountain["line"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im = axes_map(axes[0], m["lon"], m["lat"], m["E"], "M31：直接原生高程（灰色无覆盖，不填洞）")
    fig.colorbar(im, ax=axes[0], shrink=.6, label="m")
    selected_host = (m["origin"] == 304) & (m["plate"] == 304)
    effective_w = np.where(m["covered"], np.where(selected_host, m["W"], 0.), np.nan)
    im = axes_map(axes[1], m["lon"], m["lat"], effective_w, "现今处方权重：仅原ALB宿主，含满额内核", "viridis", None)
    axes[1].contour(m["lon"], m["lat"], selected_host, levels=[.5], colors="white", linewidths=.8)
    fig.colorbar(im, ax=axes[1], shrink=.6, label="规定增厚权重，非求解应变")
    sections, _ = spherical_sections(line, [r["along_km"] for r in mountain["sections"]], [-300., 300.])
    for k, ax in enumerate(axes):
        ax.plot(line[:, 0], line[:, 1], "--", c="#c54183", lw=2, label="认可轴线")
        for row, pts in zip(mountain["sections"], sections):
            x, y = xyz_to_lonlat(pts)
            color = "#333" if k == 0 else "#ddd"
            ax.plot(x, y, c=color, lw=.7)
            ax.text(x[1], y[1], row["section"], fontsize=9, color=color)
        source = mountain["source_parcels"]
        ax.scatter([p["lon"] for p in source], [p["lat"] for p in source], s=13, facecolors="none", edgecolors="#222", lw=.7)
    fig.subplots_adjust(bottom=.22, top=.87, wspace=.2)
    save(fig, "45_arc_sections_map.png", "圆点为29个原ALB宿主中心；白线为其当前覆盖边界。S1–S5为600km横剖面；背景查询约2–3km，物理节点仍约60km。")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    x = mountain["cross_offset_km"]
    for i, (ax, row) in enumerate(zip(axes.ravel(), mountain["sections"])):
        ax.plot(x, mountain["design_profiles_m"][i], "--", c="#b34182", label="终图示意 visual_only")
        for name, color in (("C4", "#aaa"), ("C6_1024", "#28a0a0"), ("C6_2048", "#d38c35")):
            ax.plot(x, mountain["output_profiles_m"][name][i], c=color, lw=1, label=name)
        ax.plot(x, mountain["native_profiles_m"][i], c="#542678", lw=1.6, label="直接原生（空隙断线）")
        ax.axvline(0, c="#333", lw=.6)
        ax.axhline(0, c="#333", lw=.6)
        ax.set(title=f"{row['section']} · 轴向{row['along_km']:.0f} km", xlabel="横向距离 km（＋左侧/西北）", ylabel="高程 m", ylim=(-6500, 5200))
        ax.grid(alpha=.2)
    axes.ravel()[-1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc="center", fontsize=10)
    fig.tight_layout(rect=(0, .14, 1, .95))
    save(fig, "46_arc_transverse_profiles.png", "五剖面使用相同海沟/动力高程；输出包含回退/填洞，原生曲线不含。终图130km是高斯尺度，不是等高带合格线。")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.4, .65, 1.]})
    s = np.asarray(mountain["strip_along_km"])
    n = np.arange(-240., 241., 5.)
    im = axes[0].pcolormesh(s, n, m["strip_E"].T, cmap=cmap, norm=norm, shading="nearest")
    axes[0].contour(s, n, m["strip_E"].T, levels=[1000, 1500, 2000], colors=["#555", "#c0447a", "#111"], linewidths=.8)
    axes[0].axhline(0, c="white", lw=.8, ls="--")
    axes[0].set(title="M31带状观察窗：1500m带是否围着轴线，端部是否收拢", ylabel="横向 km（＋左侧）")
    cax = fig.add_axes([.915, .62, .015, .27])
    fig.colorbar(im, cax=cax, label="m")
    categories = np.where(~m["strip_covered"], 0, np.where(m["strip_origin"] == 304, 1, np.where(m["strip_ctype"] == 1, 2, 3)))
    category_cmap = matplotlib.colors.ListedColormap(["#888", "#bda6d6", "#c1b77f", "#6ba5bf"])
    im = axes[1].pcolormesh(s, n, categories.T, cmap=category_cmap, norm=matplotlib.colors.BoundaryNorm(np.arange(5) - .5, 4), shading="nearest")
    axes[1].axhline(0, c="#222", ls="--", lw=.7)
    axes[1].set(ylabel="横向 km", title="宿主来源：紫=原ALB；褐=其他陆壳；蓝=洋壳；灰=无覆盖")
    axes[2].plot(s, mountain["endpoint_axis_m"], c="#542678", label="原生轴线及两端外延")
    axes[2].axhline(0, c="#222", lw=.7)
    axes[2].set(xlabel="沿线距离 km（0至277为认可山脊；两端各外延180）", ylabel="高程 m", title="同一轴线首尾外延；灰色/断线不作实体连接")
    for ax in axes:
        ax.axvspan(0, mountain["length_km"], color="#c6b8d0", alpha=.16)
        ax.axvline(0, c="#555", ls=":")
        ax.axvline(mountain["length_km"], c="#555", ls=":")
        ax.grid(alpha=.2)
    fig.subplots_adjust(left=.08, right=.9, bottom=.16, top=.91, hspace=.32)
    save(fig, "47_arc_width_and_ends.png", "等高带只量与轴线相接的连续段，仍可能包含其他地体高地；原ALB宿主宽度另列JSON。5km查询步距不等于物理精度。")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    # 三种实际输出在同一固定窗口与精确B5轮廓下比较。
    for ax, (name, path) in zip(axes, (("C4", ".cache/u35_c5/reproduced_c4/latest/world.nc"),
            ("C6 1024", ".cache/u35_c6/worlds/native_fill_1024x512.nc"), ("C6 2048", ".cache/u35_c6/worlds/native_fill_2048x1024.nc"))):
        with open_netcdf(ROOT / path) as ds:
            crop = ds.sel(lon=slice(16.5, 21.), lat=slice(1., -6.))
            im = axes_map(ax, crop.lon.values, crop.lat.values, crop.elevation.values, name + "：精确定位B5")
    for ax in axes:
        for poly, style in ((np.asarray(basin["design"]["poly"]), "--"), (m["meander"], "-")):
            p = np.vstack([poly, poly[0]])
            ax.plot(p[:, 0], p[:, 1], style, c="#c12f79", lw=1.3)
        ax.scatter(*basin["center_lon_lat"], marker="*", c="#ffda55", edgecolors="#222", s=95, zorder=6)
        ax.set(xlim=(16.5, 21.), ylim=(-6., 1.))
    fig.subplots_adjust(bottom=.23, top=.88, wspace=.24)
    base.elevation_legend(fig, .13)
    fig.axes[-1].set_xlabel("高程 m（统一色带；无地形阴影）", fontsize=8)
    save(fig, "48_B5_location_comparison.png", "粉虚线为B5原始顶点，实线为固定种子的设计轮廓，星为顶点均值定位点；三图同窗口，不用自动r8代替B5。")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes_map(axes[0], m["blon"], m["blat"], m["bE"], "B5：直接原生高程")
    im = axes_map(axes[1], m["blon"], m["blat"], m["bT"], "B5：地壳厚度 km", "viridis", None)
    fig.colorbar(im, ax=axes[1], shrink=.55, label="km")
    event_cmap = matplotlib.colors.ListedColormap(["#ddd", "#cf7738", "#9a6262", "#5d9ca9", "#a17ac0"])
    event_cmap.set_bad("#888888")
    im = axes_map(axes[2], m["blon"], m["blat"], np.where(m["bcovered"], m["boro"], np.nan), "最后改造：0无/1弧/2碰撞/3裂谷/4复活", event_cmap, matplotlib.colors.BoundaryNorm(np.arange(6) - .5, 5))
    fig.colorbar(im, ax=axes[2], ticks=range(5), shrink=.55)
    for ax in axes:
        p = np.vstack([m["meander"], m["meander"][0]])
        ax.plot(p[:, 0], p[:, 1], c="#c12f79", lw=1.5)
        ax.contour(m["blon"], m["blat"], np.where(m["bcovered"], m["bplate"] == 301, np.nan), levels=[.5], colors="#222", linewidths=.7)
        ax.scatter(*basin["center_lon_lat"], marker="*", c="#ffda55", edgecolors="#222", s=95, zorder=6)
    fig.subplots_adjust(bottom=.23, top=.88, wspace=.27)
    save(fig, "49_B5_native_crust.png", "灰色无原生覆盖；高程配色同图48。黑线为现今C/D分类边界（非精确断层）；最后改造仅是末帧标签，不是完整收支归因。")

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(basin["cross_offset_km"], basin["native_profiles_m"][i], c="#542678", label="原生高程，空隙断线")
        ax.axhline(-400, c="#c12f79", ls="--", label="终图谷底目标−400m")
        ax.axhline(0, c="#555", lw=.7)
        within = np.flatnonzero(basin["profile_in_design_polygon"][i])
        for k, group in enumerate(np.split(within, np.flatnonzero(np.diff(within) > 1) + 1)):
            if len(group):
                ax.axvspan(basin["cross_offset_km"][group[0]], basin["cross_offset_km"][group[-1]], color="#edc4dc", alpha=.3,
                           label="实际设计轮廓内查询" if k == 0 else None)
        ax.set(ylabel="高程 m", title=f"B5自北向南第{i + 1}剖面 · 沿长轴{basin['section_along_km'][i]:.0f}km")
        ax.grid(alpha=.2)
        twin = ax.twinx()
        twin.plot(basin["cross_offset_km"], basin["native_profiles_T_km"][i], c="#a76e20", lw=.8, alpha=.7)
        twin.set(ylabel="地壳厚度 km", ylim=(15, 65))
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("横向距离 km（＋东方；北→南长轴的左侧）")
    fig.tight_layout(rect=(0, .13, 1, .95))
    save(fig, "50_B5_profiles.png", "目标是谷底高程，不是要求整个轮廓均为−400m；窄洼地宽度接近原生节点间距，不能用密集查询伪造精细盆底。")

    names = ["45_arc_sections_map.png", "46_arc_transverse_profiles.png", "47_arc_width_and_ends.png", "48_B5_location_comparison.png", "49_B5_native_crust.png", "50_B5_profiles.png"]
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C7 山弧横剖面与B5定位</title><style>body{font-family:system-ui;max-width:1500px;margin:24px auto;padding:16px;background:#eee}img{max-width:100%}</style><h1>U35-C7：山弧横剖面、二维宽度与B5精确定位</h1><p>只读同一C4/C6原生状态；没有修改地形。灰色是无原生覆盖，密集取样不是更高物理分辨率。</p>"
    for name in names:
        html += f"<h2>{name}</h2><img src='{name}'>"
    html += "<p><a href='../u35_c6/index.html'>C6对照</a> · <a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c6_native/index.html'>未改变的C6标准图组</a></p>"
    (OUT / "index.html").write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
