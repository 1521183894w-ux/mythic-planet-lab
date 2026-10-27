"""运行U36-S1，保存合成试验图、参数、数值与保护文件哈希。"""
from __future__ import annotations
import json
import hashlib
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
import importlib.metadata as md

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from study import METHOD_VERSION, REFERENCES, Settings, run_case

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/design/v2/distribution_spike"
DATA = ROOT / "data/worlds/design_v2/distribution_spike"
KINDS = ["legacy_cap", "legacy_uncapped", "uniform", "strong_basin", "directional"]
NAMES = {
    "legacy_cap": "原规则：扩散300 + 65km逐圈填充",
    "legacy_uncapped": "对照：只取消原规则的65km截断",
    "uniform": "Landlab：均质扩散，K=12000",
    "strong_basin": "Landlab：强盆地，K外=12000",
    "directional": "Landlab：强盆地 + 南北定向",
}
PROTECTED = [
    "packages/tectonics/engine.py", "packages/tectonics/crust.py",
    "config/design/final_map_v2.yaml", "config/design/history_v2.yaml",
    "config/metrics.yaml", "data/worlds/design_v2/forward_v2.json",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in PROTECTED if (ROOT / name).exists()}


def setup():
    plt.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                        "axes.unicode_minus": False, "font.size": 10})


def map_image(ax, result, field, title, vmin, vmax, cmap="terrain"):
    grid = result["grid"]
    values = np.where(result["core"], field, np.nan).reshape(grid.shape)
    image = ax.imshow(values[1:-1, 1:-1], origin="lower", extent=(-1200, 1200, -1000, 1000),
                      cmap=(ListedColormap(plt.colormaps["terrain"](np.linspace(0.25, 1, 256))) if cmap == "terrain" else cmap), vmin=vmin, vmax=vmax, interpolation="nearest")
    for mask, color in [(result["basin"], "black"), (result["corridor"], "purple")]:
        ax.contour(result["x"].reshape(grid.shape), result["y"].reshape(grid.shape),
                   mask.reshape(grid.shape).astype(float), levels=[0.5], colors=[color], linewidths=0.8)
    ax.set(title=title, xlabel="局部 x / km", ylabel="局部 y / km")
    ax.set_aspect("equal")
    ax.plot([700, 1100], [-850, -850], color="black", lw=2)
    ax.text(900, -810, "400 km", ha="center", fontsize=8)
    return image


def save(fig, filename, note):
    fig.text(0.03, 0.015, note, ha="left", va="bottom", fontsize=8, linespacing=1.5)
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def make_plots(results, sweep, convergence, revision):
    setup()
    r = results["directional"]
    footer = ("合成试验；不是本星球地图 | procedural_constrained / " + METHOD_VERSION
              + " | 主格网40km / 40Myr | 输入分区 user_override | seed=无随机 | code=" + revision)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
    fields = [r["initial"], -np.log10(np.maximum(1e-12, 1000.0 ** (-r["basin_weight"]))),
              r["source"]]
    titles = ["初始地壳厚度 / km", "假设的log10黏度对比；黑圈为盆地核心", "同一外部碰撞供给 / km/Myr"]
    for ax, field, title, lo, hi, cmap in zip(axes, fields, titles, [35, 0, 0], [35.33, 3, 1.8],
                                            ["viridis", "magma", "inferno"]):
        im = map_image(ax, r, field, title, lo, hi, cmap)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("U36-S1 输入：闭边界合成域，紫框是北向检查走廊；没有最终高程目标")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.83, wspace=0.4)
    save(fig, "01_inputs.png", footer + "\n方向方案：y向K不变、x向K×0.25；分区边界是输入，不是自然生成地貌。")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    norm = Normalize(0, 10000)
    for ax, kind in zip(axes.flat, KINDS):
        rr = results[kind]
        s = rr["summary"]
        map_image(ax, rr, rr["height"], NAMES[kind] + "\n"
                  + f"盆地中位 {s['basin_median_m']:.0f}m；走廊 {s['corridor_median_m']:.0f}m；最高 {s['maximum_m']:.0f}m",
                  0, 10000)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.95,
        "所有方案：同一初始柱、同一输入体积、40Myr、闭边界\n\n"
        "原规则两案只差是否截断，隔离齐高平台的形成。\n"
        "均质→强盆地只改材料对比，隔离强盆地的影响。\n"
        "强盆地→定向只改x向系数，隔离方向性的影响。\n\n"
        "原规则→Landlab同时改变K及分配规则，\n不能据此把全部差异归因于材料分区。\n"
        "统一0–10000m色带；高程是Airy公式诊断量。\n"
        "原规则四邻接适配不是C11全球复跑。", va="top", linespacing=1.7)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(plt.colormaps["terrain"](np.linspace(0.25, 1, 256)))),
                 ax=list(axes.flat), fraction=0.018, pad=0.03, label="诊断高程 / m")
    fig.suptitle("同输入材料分配对照：硬截断、有限时间扩散、材料分区与方向性")
    fig.subplots_adjust(left=0.06, right=0.85, bottom=0.14, top=0.88, hspace=0.45, wspace=0.3)
    save(fig, "02_comparison.png", footer + "\n65km是现有工程上限，不是普适物理天花板；无截断对照出现96km厚地壳，不能采用。")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for kind in KINDS:
        rr = results[kind]
        shape = rr["grid"].shape
        zz = rr["height"].reshape(shape)
        xx, yy = rr["x"].reshape(shape)[0], rr["y"].reshape(shape)[:, 0]
        axes[0, 0].plot(xx[1:-1], zz[np.argmin(abs(yy)), 1:-1], label=NAMES[kind])
        axes[0, 1].plot(yy[1:-1], zz[1:-1, np.argmin(abs(xx + 420))])
        hist = rr["history"]
        axes[1, 0].plot(hist[:, 0], hist[:, 1])
        axes[1, 1].plot(hist[:, 0], hist[:, 2])
    for ax in axes[0]:
        ax.axhline(4945, color="gray", ls="--", lw=0.8)
        ax.set(ylabel="诊断高程 / m", ylim=(0, 10500))
    axes[0, 0].set(title="东西剖面：y=0", xlabel="x / km")
    axes[0, 0].axvspan(40, 400, color="gray", alpha=0.15)
    axes[0, 1].set(title="南北剖面：x≈−420km", xlabel="y / km")
    for ax, title in zip(axes[1], ["强盆地核心中位", "北向走廊中位"]):
        ax.set(title=title, xlabel="演化时间 / Myr", ylabel="诊断高程 / m")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.07), ncol=2, fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.94, hspace=0.38, wspace=0.3)
    save(fig, "03_sections_history.png", footer + "\n剖面与统计均为合成结构检查窗；不对应B6、PL8、M3f，不把北向走廊称作实际山系。")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    labels = ["basin_median_m", "corridor_median_m", "maximum_thickness_ever_km"]
    titles = ["强盆地中位 / m", "北向走廊中位 / m", "全时段最大地壳厚度 / km"]
    ks, contrasts = [6000, 12000, 24000], [1, 10, 100, 1000]
    for ax, key, title in zip(axes, labels, titles):
        vals = np.array([[next(s[key] for s in sweep if s["K"] == k and s["contrast"] == c)
                          for c in contrasts] for k in ks])
        im = ax.imshow(vals, cmap="viridis", aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{vals[i,j]:.0f}", ha="center", va="center",
                        color="white" if vals[i,j] < (vals.min()+vals.max())/2 else "black")
        ax.set(xticks=range(4), xticklabels=contrasts, yticks=range(3), yticklabels=ks,
               xlabel="假设黏度对比", ylabel="y向K / km²/Myr", title=title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("定向方案参数敏感性：12组全部展示；不是按验收分数筛选")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.25, top=0.82, wspace=0.5)
    save(fig, "04_sensitivity.png", footer + "\nK与黏度对比未用真实区域校准；输入增大/流动变慢可过厚，不能只看盆地低位。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for item in convergence:
        rr = item["result"]
        shape = rr["grid"].shape
        zz = rr["height"].reshape(shape)
        xx, yy = rr["x"].reshape(shape)[0], rr["y"].reshape(shape)[:, 0]
        label = f"dx={item['dx']}km,dt={item['dt']}Myr" + item.get("label", "")
        axes[0].plot(yy[1:-1], zz[1:-1, np.argmin(abs(xx+420))], label=label)
        axes[1].plot(xx[1:-1], zz[np.argmin(abs(yy)), 1:-1], label=label)
    axes[0].set(title="南北剖面", xlabel="y / km", ylabel="诊断高程 / m")
    axes[1].set(title="东西剖面", xlabel="x / km", ylabel="诊断高程 / m")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_ylim(0, 4500)
        ax.set_xlim(-1200, 1200)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.87, wspace=0.3)
    save(fig, "05_resolution.png", footer + "\n定向方案：网格/步长细化，及闭边界移远一倍；剖面仅画活动单元。局部窗口相同，不以平滑代替收敛。")


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    before = hashes()
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    settings = Settings()
    results = {kind: run_case(kind, settings) for kind in KINDS}
    sweep = []
    for k in (6000, 12000, 24000):
        for contrast in (1, 10, 100, 1000):
            s = run_case("directional", replace(settings, diffusivity_km2_myr=k, viscosity_contrast=contrast))["summary"]
            sweep.append({"K": k, "contrast": contrast, **s})
    convergence = [{"dx": dx, "dt": dt, "result": run_case("directional", replace(settings, spacing_km=dx, dt_myr=dt))}
                   for dx, dt in [(40, 0.25), (40, 0.125), (20, 0.125)]]
    convergence.append({"dx": 40, "dt": 0.125, "label": "；域加倍",
                        "result": run_case("directional", replace(settings, dt_myr=0.125,
                                                               width_km=4800, height_km=4000))})
    make_plots(results, sweep, convergence, revision)
    arrays = {}
    for kind, r in results.items():
        for field in ("thickness", "height", "source", "initial", "history", "basin_weight", "diffusivity"):
            arrays[kind + "_" + field] = r[field]
    np.savez_compressed(DATA / "arrays.npz", **arrays)
    after = hashes()
    if before != after:
        raise RuntimeError("保护文件被改变")
    report = {
        "method_class": "procedural_constrained",
        "method_name": "固定网格材料扩散类比；Landlab LinearDiffuser",
        "method_version": METHOD_VERSION, "settings": asdict(settings),
        "units": {"length": "km", "thickness": "km", "time": "Myr", "height": "m", "volume": "km3"},
        "assumptions": ["闭边界固定面积，无球面板块运动",
                        "外部供给库按输入等量扣除；不求真实碰撞速率",
                        "不求温度、黏弹塑性、应力、沉积或剥蚀；不等同下地壳动力学",
                        "空间结构及方向为user_override；地形未按终图高度回填",
                        "两个原规则对照隔离截断；三个Landlab对照隔离结构与方向性",
                        "原规则与Landlab之间同时改变系数及方法，不能单独归因"],
        "layers": {"initial_structure": "user_override", "source": "user_override",
                   "thickness": "procedural_constrained", "height": "parameterized_physics",
                   "plot": "visual_only"},
        "references": REFERENCES, "base_commit": revision,
        "code_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")},
        "versions": {n: md.version(n) for n in ("numpy", "scipy", "landlab", "matplotlib")},
        "platform": platform.platform(),
        "cases": {k: r["summary"] for k, r in results.items()},
        "sensitivity": sweep,
        "refinement": [{"dx": v["dx"], "dt": v["dt"], "label": v.get("label", ""), **v["result"]["summary"]} for v in convergence],
        "protected_hashes_before": before, "protected_hashes_after": after,
        "elapsed_s": time.perf_counter() - started,
    }
    (DATA / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    images = sorted(OUT.glob("*.png"))
    html = '<!doctype html><meta charset="utf-8"><title>U36-S1地壳分配试验</title><style>body{max-width:1500px;margin:30px auto;background:#eee;font-family:sans-serif}img{width:100%;background:white}h1,p{padding:0 20px}</style><h1>U36-S1 地壳分配方法小试</h1><p>合成输入与局部方法验证；不对应实际世界，未关闭原有问题。</p>'
    html += "".join(f'<h2>{p.name}</h2><img src="{p.name}" alt="{p.stem}">' for p in images)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "refinement", "elapsed_s")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
