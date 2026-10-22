"""阶段 1.2c 检查点：剧本生成器（300 -> 0 Ma 板块运动剧本）+ 引擎完整模拟，0 Ma 用阶段 0 的 A 层、B1 代码评估，出图。

    # 先筛前史种子（只建前史 + 编组，约 10 s/个），再对可用的世界种子规划、模拟
    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\tectonics_scenario.py --runs 19:1,11:1 --level 7

--runs 为 "世界种子:剧本种子" 列表。输出：figures/stage1/scenario/（图与 summary.json，不进 Git）；
.cache/tectonics/scenario/（各次运行每 10 Ma 一帧的快照、剧本、事件日志、指标；U8）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "stage1" / "scenario"
NOTE = ("method_class：板块运动 = 剧本设定（procedural_constrained：情景族 + 随机抽样，速度受地球范围约束）；"
        "俯冲、碰撞增厚、岛弧、新洋壳 = 过程规则（procedural_constrained）；洋深 GDH1、陆地 Airy 均衡 = parameterized_physics")
FIELDS_F = ("elev", "T", "t_form", "age")
FIELDS_I = ("plate", "ctype", "origin", "prov", "oro")


def parse_runs(s):
    out = []
    for part in s.split(","):
        w, _, sc = part.partition(":")
        out.append((int(w), int(sc or 1)))
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _ll(v):
    from mythic.tectonics.geometry import xyz_to_lonlat
    lon, lat = xyz_to_lonlat(np.asarray(v, float))
    return [round(float(lat), 2), round(float(lon), 2)]


def plan_record(plan, planner) -> dict:
    segs = lambda ss: [{"from_Ma": float(a), "to_Ma": float(b), "pole_lat_lon": _ll(ax), "angle_deg": float(np.degrees(ang))}
                       for a, b, ax, ang in ss]
    return {
        "seed": plan.seed, "anchor": plan.anchor, "roles": plan.roles, "score": plan.score,
        "groups": {b: list(B.members) for b, B in planner.blocks.items() if B.members},
        "block_area_km2": {b: B.area_km2 for b, B in planner.blocks.items()},
        "docks": {b: {"t_dock": d.t_dock, "t_contact": d.t_contact, "inner_sea": d.inner_sea, "segs": segs(d.segs)}
                  for b, d in plan.docks.items()},
        "moves": {b: segs(ss) for b, ss in plan.moves.items()},
        "placement": {**plan.placement, "pole_lat_lon": _ll(plan.G_axis), "angle_deg": float(np.degrees(plan.G_angle))},
        "breakups": [{k: (_ll(v) if k in ("pole", "axis") else v) for k, v in e.items()} for e in plan.breakups],
        "inner_sea": {k: (_ll(v) if k == "centroid" else v) for k, v in plan.inner_sea.items()},
        "conflicts": plan.conflicts, "notes": plan.notes,
    }


def cached_initial_world(mesh, wseed: int):
    """300 Ma 初始世界（前史）。前史只取决于种子、参数和代码，按前史相关源码的哈希缓存（.cache，不进 Git）：
    调剧本时每轮省 3–8 分钟。源码一改，哈希就变，自动重算。"""
    import hashlib
    import pickle

    from mythic.tectonics.mesh import CACHE
    from mythic.tectonics.prehistory import build_initial_world

    src = ROOT / "packages" / "tectonics"
    h = hashlib.sha1()
    for f in ("prehistory.py", "engine.py", "crust.py", "geometry.py", "kinematics.py", "mesh.py"):
        h.update((src / f).read_bytes())
    path = CACHE / "prehistory" / f"world_L{mesh.level}_s{wseed}_{h.hexdigest()[:10]}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    w = build_initial_world(mesh, wseed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(w, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return w


def run_one(level: int, wseed: int, sseed: int, loop: bool = True) -> dict:
    from mythic.figures.world import code_version
    from mythic.metrics.compute import compute_all, load_config
    from mythic.metrics.verdict import b_verdict
    from mythic.tectonics.mesh import CACHE, load_mesh
    from mythic.tectonics.scenario import run_scenario, run_scenario_loop

    mesh = load_mesh(level)
    t0 = time.perf_counter()
    w = cached_initial_world(mesh, wseed)
    t_pre = time.perf_counter() - t0
    run = run_scenario_loop(mesh, w, sseed) if loop else run_scenario(mesh, w, sseed)
    cfg = load_config()
    ref = json.loads((ROOT / "data" / "earth_reference" / "baselines_1024x512.json").read_text(encoding="utf-8"))
    if run.metrics is not None:
        m = run.metrics
    else:
        from mythic.tectonics.mesh import Raster
        raster = Raster(mesh, 512)
        E = raster(run.snapshots[-1]["elev"])
        m = compute_all(E, E > 0, cfg=cfg)
    bv = b_verdict(m["by_region"], ref["landmass_unit_reference"], cfg)
    d = CACHE / "scenario"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"scenario_L{level}_w{wseed}_s{sseed}"
    arrs = {"times": np.array([s["t"] for s in run.snapshots])}
    for k, s in enumerate(run.snapshots):
        for f in FIELDS_F:
            arrs[f"{f}_{k}"] = s[f].astype(np.float32)
        for f in FIELDS_I:
            arrs[f"{f}_{k}"] = s[f].astype(np.int32)
    np.savez_compressed(d / f"{tag}.npz", **arrs)
    A = m["A"]
    summary = {"world_seed": wseed, "scenario_seed": sseed, "level": level, "code_version": code_version(),
               "wall_s": {"prehistory": t_pre, "scenario_total": run.wall_s,
                          "step_mean": float(np.mean([s["wall_s"] for s in run.engine.step_log]))},
               "plan": plan_record(run.plan, run.planner), "rot_path": run.rot_path,
               "engine_events": run.engine.events, "A": A, "soft": m.get("soft"), "b_verdict": bv,
               "A_all_pass": all(x["pass"] for x in A.values()), "inner_sea_loop": run.loop,
               "attached_small_plates": getattr(run.planner, "attached", {})}
    (d / f"{tag}.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    return {"run": f"w{wseed}:s{sseed}", "plan_score": run.plan.score, "wall_s": round(run.wall_s),
            "A": {k: bool(x["pass"]) for k, x in A.items()},
            "main_share": round(A["main_continent"]["share_of_land"], 3),
            "inner_sea": {k: A["main_inland_sea"].get(k) for k in ("exists", "passes")},
            "loop": [(h["round"], h["dgap_km"], h["planned_sea"].get("state"), h["inner_sea_pass"]) for h in run.loop],
            "B1_units_all_pass": f"{sum(u['all_pass'] for u in bv['units'].values())}/{len(bv['units'])}"}


# ---- 出图 -------------------------------------------------------------------------------------------
def figures(level: int, runs: list[tuple[int, int]]):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from mythic.figures.style import setup_fonts, shaded_elevation
    from mythic.figures.world import code_version
    from mythic.metrics.compute import compute_all, load_config
    from mythic.tectonics.geometry import unit, xyz_to_lonlat
    from mythic.tectonics.mesh import CACHE, Raster, load_mesh

    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    PC = ccrs.PlateCarree()
    mesh = load_mesh(level)
    raster = Raster(mesh, 512)
    d = CACHE / "scenario"
    data = {}
    for w, s in runs:
        tag = f"scenario_L{level}_w{w}_s{s}"
        js = json.loads((d / f"{tag}.json").read_text(encoding="utf-8"))
        z = np.load(d / f"{tag}.npz")
        frames = [{f: z[f"{f}_{k}"] for f in FIELDS_F + FIELDS_I} | {"t": float(t)} for k, t in enumerate(z["times"])]
        data[(w, s)] = (js, frames)
    ver = ",".join(sorted({js["code_version"] for js, _ in data.values()}))
    cfg = load_config()

    def stamp(fig):
        fig.text(0.005, 0.003, f"{NOTE}\n阶段 1.2c 剧本 · L{level} · 运行代码 {ver} · 出图代码 {code_version()} · "
                 f"scripts/tectonics_scenario.py", fontsize=6.5, color="0.3", va="bottom")

    def shaded(ax, elev):
        e = raster(elev)
        ax.imshow(shaded_elevation(e, raster.grid.pixel_km_equator, land=e > 0), transform=PC, extent=(-180, 180, -90, 90),
                  origin="upper", interpolation="nearest")
        ax.set_global()
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)

    def main_lon(f):
        c = (f["ctype"] == 1) & (f["elev"] > 0)
        lon, _ = xyz_to_lonlat(unit((mesh.xyz[c] * mesh.area_km2[c, None]).sum(0)))
        return float(lon)

    def frame_at(fr, t):
        return min(fr, key=lambda f: abs(f["t"] - t))

    def a_text(js):
        A = js["A"]
        ok = lambda k: "√" if A[k]["pass"] else "×"
        isea = A["main_inland_sea"]
        it = (f"{isea['main']['area_km2'] / 1e6:.2f} Mkm²，海峡 {isea['main']['straits_to_world_ocean']}"
              if isea.get("exists") else isea.get("reason", "无"))
        pol = A["polar_continent"]["found"]
        return (f"海洋 {A['ocean_fraction']['value']:.0%} {ok('ocean_fraction')} · 主大陆占陆地 "
                f"{A['main_continent']['share_of_land']:.0%} {ok('main_continent')} · 亚大陆 {A['subcontinents']['count']} "
                f"{ok('subcontinents')}\n主内海 {it} {ok('main_inland_sea')} · 纬度 "
                f"{A['main_continent_latitude']['north_deg']:.0f}°N/{A['main_continent_latitude']['south_deg']:.0f}° "
                f"{ok('main_continent_latitude')} · 极地大陆 "
                f"{(str(round(pol[0]['area_km2'] / 1e6, 1)) + ' Mkm²') if pol else '无'} {ok('polar_continent')} · 开阔大洋 "
                f"{ok('open_ocean')}")

    # 图 1：每次运行的古地理序列
    times = [300, 260, 220, 180, 140, 100, 50, 0]
    for (w, s), (js, fr) in data.items():
        lon0 = main_lon(fr[-1])
        proj = ccrs.Robinson(central_longitude=lon0)
        fig = plt.figure(figsize=(17, 15.5))
        plan = js["plan"]
        for k, t in enumerate(times):
            f = frame_at(fr, t)
            ax = fig.add_subplot(4, 2, k + 1, projection=proj)
            shaded(ax, f["elev"])
            t_prev = times[k - 1] if k else 301
            ev = []
            for b, dk in plan["docks"].items():
                if dk["segs"] and t <= dk["t_dock"] < t_prev:
                    ev.append(f"陆块 {b} 拼上" + ("（圈出主内海）" if dk["inner_sea"] else ""))
            for e in plan["breakups"]:
                if t <= e["t"] < t_prev:
                    ev.append({"reopen": f"陆块 {e['block']} 沿缝合带张开", "micro": f"陆块 {e['block']} 裂出微大陆"}[e["kind"]])
            ax.set_title(f"{f['t']:.0f} Ma" + ("：" + "；".join(ev) if ev else ""), fontsize=9.5)
        roles = plan["roles"]
        rl = "；".join(f"{b} {dict(main='主大陆', polar='极地', secondary='次大陆').get(r, r)}" for b, r in roles.items())
        fig.suptitle(f"世界种子 {w} · 剧本种子 {s}：300 -> 0 Ma 古地理（陆块角色：{rl}）\n0 Ma A 层：{a_text(js)}",
                     fontsize=10.5)
        stamp(fig)
        fig.savefig(OUT / f"01_paleogeography_w{w}_s{s}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)

    # 图 2：各次运行 0 Ma 联系表 + A 层要素（主大陆、主内海、极地大陆）
    n = len(data)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(17, 5.4 * nrow + 0.8))
    for k, ((w, s), (js, fr)) in enumerate(data.items()):
        f = fr[-1]
        E = raster(f["elev"])
        mt, fields = compute_all(E, E > 0, cfg=cfg, return_fields=True)
        ax = fig.add_subplot(nrow, ncol, k + 1, projection=ccrs.Robinson(central_longitude=main_lon(f)))
        shaded(ax, f["elev"])
        units = fields["units"]
        main_id = int(np.bincount(units[units > 0].ravel(), weights=raster.grid.cell_area_km2[units > 0]).argmax()) \
            if (units > 0).any() else 0
        lat = np.linspace(90, -90, raster.grid.ny)
        lon = np.linspace(-180, 180, raster.grid.nx)
        ax.contour(lon, lat, (units == main_id).astype(float), [0.5], colors="#d7301f", linewidths=0.7, transform=PC)
        isea = mt["A"]["main_inland_sea"]
        if isea.get("exists") and fields.get("sea_regions") is not None:
            sea = fields["sea_regions"] == isea["main"]["region_id"]
            ax.contourf(lon, lat, sea.astype(float), [0.5, 1.5], colors=["#2171b5"], alpha=0.6, transform=PC)
        ax.set_title(f"世界 {w} · 剧本 {s}（规划评分 {js['plan']['score']}）\n{a_text(js)}", fontsize=8)
    fig.suptitle("0 Ma：各次运行的世界（红线 = A 层主大陆，蓝 = A 层识别的主内海候选）", fontsize=11)
    stamp(fig)
    fig.savefig(OUT / "02_worlds_0Ma.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 图 3：事件时间线
    fig, axs = plt.subplots(n, 1, figsize=(15, 3.2 * n + 0.6), squeeze=False)
    for ax, ((w, s), (js, fr)) in zip(axs[:, 0], data.items()):
        plan = js["plan"]
        rows = sorted(plan["roles"])
        ypos = {b: i for i, b in enumerate(rows)}
        for b in rows:
            y = ypos[b]
            r = plan["roles"][b]
            dk = plan["docks"].get(b)
            if dk and dk["segs"]:
                for sg in dk["segs"]:
                    ax.plot([sg["from_Ma"], sg["to_Ma"]], [y, y], color="#3182bd", lw=5, solid_capstyle="butt")
                ax.plot(dk["t_dock"], y, "k|", ms=14, mew=2)
                if dk["t_contact"]:
                    ax.plot(dk["t_contact"], y, "o", color="#3182bd", ms=5)
            for sg in plan["moves"].get(b, []):
                ax.plot([sg["from_Ma"], sg["to_Ma"]], [y, y], color="#31a354" if r == "polar" else "#d95f0e", lw=5,
                        solid_capstyle="butt")
            for e in plan["breakups"]:
                if str(e["block"]) == str(b):
                    ax.plot([e["t"], max(e["t"] - e["dur"], 0)], [y + 0.25, y + 0.25], color="#e7298a", lw=3)
        kinds = {"continent_contact": "#000000", "collision_onset": "#e31a1c", "rift": "#e7298a",
                 "terrane_transfer": "#ff7f00", "subduction_start": "#6a3d9a"}
        for kk, col in kinds.items():
            ts = [e["t"] for e in js["engine_events"] if e["kind"] == kk]
            ax.plot(ts, np.full(len(ts), -1.0 - 0.25 * list(kinds).index(kk)), "|", color=col, ms=7, label=kk)
        ax.set_yticks(list(ypos.values()) + [-1.5])
        ax.set_yticklabels([f"陆块 {b}（{dict(main='主', polar='极地', secondary='次大陆').get(plan['roles'][b], plan['roles'][b])}）"
                            for b in rows] + ["引擎事件"], fontsize=8)
        ax.set_xlim(300, 0)
        ax.set_ylim(-2.4, len(rows) - 0.4)
        ax.set_xlabel("Ma")
        ax.set_title(f"世界 {w} · 剧本 {s}：蓝 = 拼合运动（圆点 = 计划首次接触，竖线 = 锁定），绿 = 极地陆块漂移，橙 = 次大陆漂移，"
                     f"粉 = 裂解；下方为引擎记录的事件", fontsize=8.5)
        ax.legend(fontsize=7, ncol=5, loc="lower left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stamp(fig)
    fig.savefig(OUT / "03_timeline.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    summary = {f"w{w}:s{s}": {"plan_score": js["plan"]["score"], "A_all_pass": js["A_all_pass"],
                              "A": {k: v["pass"] for k, v in js["A"].items()}, "wall_s": js["wall_s"],
                              "notes": js["plan"]["notes"], "code_version": js["code_version"]}
               for (w, s), (js, _) in data.items()}
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--runs", default="19:1,11:1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-loop", action="store_true", help="不做主内海闭环修正（只规划、模拟一次）")
    a = ap.parse_args()
    runs = parse_runs(a.runs)
    if not a.figures_only:
        os.environ["MYTHIC_KD_WORKERS"] = str(max(1, (os.cpu_count() or 4) // a.workers))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(run_one, [a.level] * len(runs), [w for w, _ in runs], [s for _, s in runs],
                            [not a.no_loop] * len(runs)):
                print(json.dumps(r, ensure_ascii=False), flush=True)
    figures(a.level, runs)
    print("figures done", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
