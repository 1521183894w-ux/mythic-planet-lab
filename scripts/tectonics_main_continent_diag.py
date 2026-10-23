"""阶段 1.2c 诊断：主大陆为什么破碎。读剧本运行的 0 Ma 结果（.cache/tectonics/scenario/），统计：
- 主陆块（规划角色为 main，含编组成员与跟随的小陆块，按当前板块归属）有几块、各多大；
- 主陆块的陆地（海面以上）连成几片、最大一片占多少；
- 两两之间"焊上"（陆地与陆地在网格上相接）的长度，与"隔窄海面对面"（海岸离对方陆地 < 400 km 但不相接）的长度。
出一张按主陆块着色、标出焊缝的图。这是诊断量，不是质量指标（不进 config/metrics.yaml）。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\tectonics_main_continent_diag.py --worlds 1-8

输出：figures/stage1/scenario/06_main_continent_welds.png 与 06_main_continent_welds.json（不进 Git）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "stage1" / "scenario"


def parse_worlds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def analyse(mesh, w: int, level: int, sseed: int) -> dict:
    from mythic.tectonics.geometry import chord_from_km, km_from_chord
    from mythic.tectonics.engine import graph_components
    from mythic.tectonics.mesh import CACHE

    E, h = mesh.edges, mesh.h_km
    tag = f"scenario_L{level}_w{w}_s{sseed}"
    js = json.loads((CACHE / "scenario" / f"{tag}.json").read_text(encoding="utf-8"))
    z = np.load(CACHE / "scenario" / f"{tag}.npz")
    k = len(z["times"]) - 1
    elev, plate = z[f"elev_{k}"], z[f"plate_{k}"]
    plan = js["plan"]
    roles = {int(b): r for b, r in plan["roles"].items()}
    lead = {int(q): int(b) for b, mem in plan["groups"].items() for q in mem}
    for q, info in (js.get("attached_small_plates") or {}).items():
        lead[int(q)] = int(info["lead"])
    for b in roles:
        lead.setdefault(b, b)
    main = sorted((b for b, r in roles.items() if r == "main"), key=lambda b: -plan["block_area_km2"][str(b)])
    blk = np.array([lead.get(int(q), -1) for q in plate])
    land = elev > 0
    is_main = land & np.isin(blk, main)
    lab = graph_components(mesh.n, E[is_main[E[:, 0]] & is_main[E[:, 1]]])
    comp = np.sort(np.bincount(lab[is_main], weights=mesh.area_km2[is_main]))[::-1]
    comp = comp[comp > 0]
    tot = float(mesh.area_km2[is_main].sum())
    coastal = np.zeros(mesh.n, bool)
    coastal[E[land[E[:, 0]] & ~land[E[:, 1]], 0]] = True
    coastal[E[land[E[:, 1]] & ~land[E[:, 0]], 1]] = True
    trees = {b: cKDTree(mesh.xyz[land & (blk == b)]) for b in main if (land & (blk == b)).any()}
    pairs = []
    for i, a in enumerate(main):
        for b in main[i + 1:]:
            if a not in trees or b not in trees:
                continue
            ia = np.flatnonzero(land & (blk == a))
            e = E[land[E[:, 0]] & land[E[:, 1]] & (((blk[E[:, 0]] == a) & (blk[E[:, 1]] == b)) |
                                                   ((blk[E[:, 0]] == b) & (blk[E[:, 1]] == a)))]
            weld = len(np.unique(e[blk[e] == a])) * h if len(e) else 0.0
            ca = ia[coastal[ia]]
            dc = trees[b].query(mesh.xyz[ca], distance_upper_bound=chord_from_km(400.0))[0] if len(ca) else np.array([])
            ok = np.isfinite(dc)                       # 查不到时是 inf，不能先换算成公里
            facing = float((km_from_chord(dc[ok]) > 1.5 * h).sum()) * h
            if weld + facing >= 300.0:
                pairs.append({"pair": [a, b], "weld_km": weld, "facing_km": facing})
    wt, ft = sum(p["weld_km"] for p in pairs), sum(p["facing_km"] for p in pairs)
    return {"world": w, "main_blocks": {b: plan["block_area_km2"][str(b)] for b in main},
            "main_land_km2": tot, "pieces_ge_1Mkm2": int((comp >= 1e6).sum()),
            "largest_piece_share": float(comp[0] / tot) if tot else 0.0, "pieces_km2": comp[:6].tolist(),
            "weld_km": wt, "facing_km": ft, "weld_share": wt / max(wt + ft, 1.0), "pairs": pairs,
            "breakups": [(e["kind"], e["block"]) for e in plan["breakups"]],
            "A_main_share": js["A"]["main_continent"]["share_of_land"],
            "_elev": elev, "_blk": blk, "_main": main}


def figure(mesh, res: list[dict], path: Path):
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mythic.figures.style import setup_fonts
    from mythic.figures.world import code_version
    from mythic.tectonics.geometry import xyz_to_lonlat
    from mythic.tectonics.mesh import Raster

    setup_fonts()
    raster = Raster(mesh, 512)
    PC = ccrs.PlateCarree()
    E = mesh.edges
    n = len(res)
    fig = plt.figure(figsize=(17, 5.2 * ((n + 1) // 2) + 0.8))
    cmap = plt.get_cmap("tab10")
    for kk, r in enumerate(res):
        elev, blk, main = r["_elev"], r["_blk"], r["_main"]
        land = elev > 0
        col = np.zeros((mesh.n, 3)) + np.array([0.80, 0.86, 0.93])
        col[land] = (0.75, 0.75, 0.72)
        for i, b in enumerate(main):
            col[land & (blk == b)] = cmap(i % 10)[:3]
        ew = E[land[E[:, 0]] & land[E[:, 1]] & (blk[E[:, 0]] != blk[E[:, 1]]) & np.isin(blk[E[:, 0]], main)
               & np.isin(blk[E[:, 1]], main)]
        img = np.stack([raster(col[:, c]) for c in range(3)], -1)
        ml = land & np.isin(blk, main)
        lon_c, _ = xyz_to_lonlat((mesh.xyz[ml] * mesh.area_km2[ml, None]).sum(0))
        ax = fig.add_subplot((n + 1) // 2, 2, kk + 1, projection=ccrs.Robinson(central_longitude=float(lon_c)))
        ax.imshow(img, transform=PC, extent=(-180, 180, -90, 90), origin="upper", interpolation="nearest")
        if len(ew):
            lo, la = xyz_to_lonlat(mesh.xyz[ew].mean(1))
            ax.scatter(lo, la, s=1.2, c="k", transform=PC, lw=0)
        ax.set_global()
        ax.gridlines(color="0.5", lw=0.3, alpha=0.5)
        ax.set_title(f"世界 {r['world']}：主陆块 {len(main)} 块（各色）· 陆地连成 {r['pieces_ge_1Mkm2']} 片，最大占 "
                     f"{r['largest_piece_share']:.0%} · 焊上 {r['weld_km'] / 1e3:.1f} 千 km / 隔窄海面对面 "
                     f"{r['facing_km'] / 1e3:.1f} 千 km（焊上 {r['weld_share']:.0%}）", fontsize=8.5)
    fig.suptitle("0 Ma：主大陆由哪些陆块拼成、哪里真正焊上（黑点 = 不同主陆块的陆地相接处；灰 = 其他陆地：极地、次大陆、岛弧等）",
                 fontsize=11)
    fig.text(0.005, 0.003, "诊断图：按当前板块归属着色（procedural_constrained 剧本运行的结果）· 阶段 1.2c · 出图代码 "
             f"{code_version()} · scripts/tectonics_main_continent_diag.py", fontsize=7, color="0.3")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    from mythic.tectonics.mesh import load_mesh

    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="1-8")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--scenario-seed", type=int, default=1)
    a = ap.parse_args()
    mesh = load_mesh(a.level)
    res = [analyse(mesh, w, a.level, a.scenario_seed) for w in parse_worlds(a.worlds)]
    OUT.mkdir(parents=True, exist_ok=True)
    figure(mesh, res, OUT / "06_main_continent_welds.png")
    keep = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
    (OUT / "06_main_continent_welds.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1, default=float),
                                                      encoding="utf-8")
    for r in keep:
        print(f"世界 {r['world']}：主陆块 {len(r['main_blocks'])} 块，陆地 {r['main_land_km2'] / 1e6:.0f} Mkm²，"
              f"连成 {r['pieces_ge_1Mkm2']} 片（最大 {r['largest_piece_share']:.0%}），焊上 {r['weld_km']:.0f} km，"
              f"隔窄海面对面 {r['facing_km']:.0f} km（焊上 {r['weld_share']:.0%}），A 层主大陆 {r['A_main_share']:.0%}",
              flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
