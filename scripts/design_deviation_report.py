"""向用户说明终图偏差：读取现有 C2/C11，不运行模拟、不改变验收定义。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import yaml

from design_final_map import apply_options, build, elevation
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm, setup_fonts
from mythic.metrics.grid import LatLonGrid

ROOT = Path(__file__).resolve().parents[1]


def read_world(path):
    with open_netcdf(path) as ds:
        return np.asarray(ds.elevation.values, float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-world", default=".cache/u35_c11/progressive/world.nc")
    parser.add_argument("--candidate-summary", default="data/worlds/design_v2/candidates/u35_c11_progressive/forward_v2.json")
    args = parser.parse_args()
    official_path = ROOT / "data/worlds/design_v2/design_v2_forward_v2_0Ma_1024x512.nc"
    candidate_path = ROOT / args.candidate_world
    design_path = ROOT / "config/design/final_map_v2.yaml"
    official_summary = ROOT / "data/worlds/design_v2/forward_v2.json"
    candidate_summary = ROOT / args.candidate_summary
    out = ROOT / "figures/design/v2/deviation_report"
    data = ROOT / "data/worlds/design_v2/deviation_report"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    c2 = json.loads(official_summary.read_text(encoding="utf-8"))
    c11 = json.loads(candidate_summary.read_text(encoding="utf-8"))
    eng0, eng1 = read_world(official_path), read_world(candidate_path)
    assert eng0.shape == eng1.shape == (512, 1024)
    grid = LatLonGrid.from_shape(eng1.shape)
    design = apply_options(yaml.safe_load(design_path.read_text(encoding="utf-8")), set())
    print("构建现行终图参考（仅示意图）", flush=True)
    b = build(design, grid)
    target = elevation(design, grid, b)
    old_text = subprocess.check_output(
        ["git", "show", "baf014e:config/design/final_map_v2.yaml"], cwd=ROOT
    ).decode("utf-8")
    old = apply_options(yaml.safe_load(old_text), set())
    print("构建10月24日配置参考；两版均使用当前渲染器", flush=True)
    b_old = build(old, grid)
    area = grid.cell_area_km2
    def mask_stats(ref, z):
        le = z > 0
        return {
            "target_land_km2": float(area[ref].sum()),
            "engine_land_km2": float(area[le].sum()),
            "missing_land_km2": float(area[ref & ~le].sum()),
            "extra_land_km2": float(area[~ref & le].sum()),
            "land_iou": float(area[ref & le].sum() / area[ref | le].sum()),
        }
    result = {
        "scope": "已有世界的解释性对照；无新演化、无验收阈值变更",
        "grid": "1024x512；L7物理网格约60km；赤道输出像元约39km",
        "references": {
            "current_design": "final_map_v2，含U35用户要求的西口修订（5d7a5a0）",
            "approved_configuration": "baf014e，2026-10-24；以当前渲染器重新栅格化，并非当时PNG",
            "official": "C2，默认世界",
            "candidate": "C11 progressive，未采用",
        },
        "current_design": {"C2": mask_stats(b["land"], eng0), "C11": mask_stats(b["land"], eng1)},
        "approved_configuration": {"C2": mask_stats(b_old["land"], eng0), "C11": mask_stats(b_old["land"], eng1)},
        "design_mask_change_km2": float(area[b["land"] != b_old["land"]].sum()),
        "candidate_large_deviation_regions": c11["deviation_regions"],
        "method_class": {
            "target_geometry": "user_override", "target_elevation": "visual_only",
            "crust": "procedural_constrained", "engine_elevation": "parameterized_physics",
            "output_sampling": "interpolated",
        },
        "assumptions": [
            "正高程作为引擎陆地，沿用既有comparison口径；干洼地须另查原生宿主。",
            "山系取设计走廊内目标陆地像元高程P90；高原盆地取原设计多边形内目标陆地像元中位数，沿用既有摘要。",
            "示意DEM不等于名义峰高或谷底；岛区有覆盖顺序问题，极区含示意冰面；二者不作裸岩高程误差结论。",
            "海陆重合度不能衡量山地、谷地连续性或海湾成因；没有计算气候、水文影响。",
        ],
    }
    rows = []
    for category in ["mountains", "plateaus_basins"]:
        base = {r["id"]: r for r in c2["comparison"][category]}
        for row in c11["comparison"][category]:
            r = base[row["id"]]
            is_mountain = category == "mountains"
            fk = "final_p90_m" if is_mountain else "final_median_m"
            ek = "engine_p90_m" if is_mountain else "engine_median_m"
            # 旧摘要的多边形统计字段名称保持原样，不重新定义。
            rows.append({
                "id": row["id"], "name": row["name"],
                "statistic": "走廊P90" if is_mountain else "多边形中位数",
                "design_nominal_m": row["design_height_m"],
                "target_schematic_m": row[fk], "official_C2_m": r[ek],
                "candidate_C11_m": row[ek], "C11_minus_schematic_m": row[ek]-row[fk],
                "caveat": "示意高度非名义峰顶/谷底；岛区覆盖与极地冰面须单独解释",
            })
    result["features"] = rows
    paths = [design_path, official_path, candidate_path, official_summary, candidate_summary]
    result["source_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["report_code_base"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    (data / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (data / "all_feature_heights.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    setup_fonts()
    cmap, norm = elevation_cmap_norm()
    extent = (-180, 180, -90, 90)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, z, title in zip(axes.flat[:3], [target, eng0, eng1], [
        "现行终图：位置设计 + 示意高程", "正式默认 C2：当前仍在使用", "最新 C11 持续减薄：未采用"]):
        im = ax.imshow(z, extent=extent, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=12)
    code = np.zeros(grid.shape, int)
    code[b["land"] & (eng1 > 0)] = 1
    code[b["land"] & (eng1 <= 0)] = 2
    code[~b["land"] & (eng1 > 0)] = 3
    ax = axes.flat[3]
    ax.imshow(code, extent=extent, origin="upper", interpolation="nearest",
              cmap=ListedColormap(["#edf4fa", "#9ba797", "#d45c54", "#436cbe"]), vmin=0, vmax=3)
    ax.set_title("C11 海陆差异：红=目标陆地变海；蓝=新增陆地", fontsize=12)
    for reg in c11["deviation_regions"]:
        ax.text(reg["lon"], reg["lat"], str(reg["no"]), fontsize=7,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=.7))
    windows = [(99, 116, 20, 35), (57, 79, 33, 41), (16, 21, -6, 1)]
    for a in axes.flat:
        a.set_xlim(-180, 180); a.set_ylim(-90, 90)
        a.set_xticks(np.arange(-180, 181, 60)); a.set_yticks(np.arange(-60, 61, 30))
        a.tick_params(labelsize=8); a.grid(alpha=.15)
    for x0, x1, y0, y1 in windows:
        axes.flat[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="#db4054", lw=1))
    fig.colorbar(im, ax=list(axes.flat[:3]), orientation="horizontal", fraction=.025, pad=.09, label="高程 / m")
    fig.suptitle("大陆轮廓接近，不代表山脉与盆地已经实现", fontsize=17)
    fig.text(.02, .02, "同窗1024×512；物理L7约60km。终图 user_override / visual_only；引擎 procedural_constrained + parameterized_physics；取样 interpolated。\n颜色表达现有高程，不含河流/冰川侵蚀与气候模拟；差异编号详见 comparison.json。", fontsize=8)
    fig.subplots_adjust(left=.04, right=.98, top=.91, bottom=.16, hspace=.30, wspace=.12)
    fig.savefig(out / "01_global_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    feature_map = {p["id"]: p for cat in ["mountains", "plateaus", "basins"] for p in design[cat]}
    row_images = []
    for i, (window, ids, label) in enumerate(zip(windows,
            [["B6", "PL8"], ["M3f"], ["B5"]], ["红层盆地与喀斯特高原", "高原中北部山脉", "死海式洼地"])):
        x0, x1, y0, y1 = window
        rownorm = TwoSlopeNorm(vmin=-500, vcenter=0, vmax=5500 if i < 2 else 2000)
        for j, (z, title) in enumerate(zip([target, eng0, eng1], ["终图示意", "正式 C2", "候选 C11"])):
            ax = axes[i, j]
            im = ax.imshow(z, extent=extent, cmap=cmap, norm=rownorm, origin="upper", interpolation="nearest")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_aspect(1 / np.cos(np.deg2rad((y0+y1)/2)))
            for fid in ids:
                f = feature_map[fid]
                xy = np.array(f.get("line", f.get("poly")), float)
                if "poly" in f: xy = np.vstack([xy, xy[0]])
                ax.plot(xy[:, 0], xy[:, 1], color="#e43f67", lw=1.2)
                ax.text(*xy.mean(axis=0), fid, color="#791535", fontsize=9,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1))
            ax.set_title(f"{label} / {title}", fontsize=10)
            ax.tick_params(labelsize=8); ax.grid(alpha=.2)
        row_images.append(im)
    fig.suptitle("同一地理窗口对照：红线为设计要素原始轴线或多边形", fontsize=15)
    fig.text(.03, .035, "示意高程与名义峰顶/谷底不同：B5目标谷底−400m，不要求整块多边形都是−400m。\nB5这里沿用1024输出；实际弯曲轮廓的原生查询见C11报告：中位857m，仅0.565%面积低于海面。\n各行共用各自行色标；最近邻显示不提高L7物理分辨率。方法与来源同图01。", fontsize=9)
    fig.subplots_adjust(left=.06, right=.89, top=.93, bottom=.13, hspace=.32, wspace=.20)
    for i, im in enumerate(row_images):
        pos = axes[i, 2].get_position()
        cax = fig.add_axes([.91, pos.y0, .012, pos.height])
        fig.colorbar(im, cax=cax, label="高程 / m")
    fig.savefig(out / "02_key_regions.png", dpi=160)
    plt.close(fig)
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>终图偏差说明</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;background:#f5f5f1}img{width:100%;background:white}p{line-height:1.7}</style><h1>已有世界与终图的偏差</h1><p>现行终图包含U35西口修订。正式世界仍C2；C11为未采用候选。本页仅说明现有结果，不代表新模拟或阶段通过。名义高度、示意DEM统计与引擎统计须分别理解。</p><img src="01_global_comparison.png"><img src="02_key_regions.png">', encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k in ["current_design", "approved_configuration", "design_mask_change_km2"]}, ensure_ascii=False, indent=2))
    print(f"已写入{len(rows)}项高度对照及两张图：{out}", flush=True)


if __name__ == "__main__":
    main()
