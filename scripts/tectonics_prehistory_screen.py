"""U20 开发用：前史种子批量筛查。每个种子建前史（300 Ma），列出陆块数、按剧本规划器同一套标准编组后的组数、
是否可用（world_usable），以及漂移设计的刚体诊断。不出图。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\tectonics_prehistory_screen.py 6 4 1,2,3,4,5,6,7,8

参数：网格级别、并行进程数、种子列表（逗号分隔）。L6 上 8 个种子 4 进程约 4 分钟。
"""
import os
import sys
import time
from multiprocessing import Pool


def one(args):
    level, seed = args
    from mythic.tectonics.mesh import load_mesh
    from mythic.tectonics.prehistory import build_initial_world
    from mythic.tectonics.scenario import ScenarioParams, blocks_from_world, group_adjacent, world_usable
    t = time.perf_counter()
    mesh = load_mesh(level)
    w = build_initial_world(mesh, seed)
    raw = blocks_from_world(mesh, w)
    g = group_adjacent(mesh, raw, links=w.links)
    ok, why = world_usable(g, ScenarioParams())
    dr = w.events[0]["drift"]
    return (seed, round(time.perf_counter() - t), w.events[0]["n_blocks"], len(raw), len(g),
            sorted([round(b.area_km2 / 1e6, 1) for b in g.values()], reverse=True), ok, why,
            {"rigid_groups": dr.get("groups_rigid"), "rigid_min_gap": round(dr["min_gap_km_rigid"] or 0),
             "crossings": dr["crossings"], "moved": dr["cells_reassigned_by_cuts"], "links": dr.get("links")},
            {b: B.members for b, B in g.items() if B.members})


if __name__ == "__main__":
    level, workers = int(sys.argv[1]), int(sys.argv[2])
    seeds = [int(x) for x in sys.argv[3].split(",")]
    os.environ["MYTHIC_KD_WORKERS"] = str(max(1, (os.cpu_count() or 4) // workers))
    with Pool(workers) as p:
        res = sorted(p.imap_unordered(one, [(level, s) for s in seeds]))
    print("（种子, 秒, 陆块数, 规划器陆块数, 编组数, 各组面积 Mkm², 可用, 原因, 刚体诊断, 编组成员）")
    for r in res:
        print(r)
    print("usable:", [r[0] for r in res if r[6]], "of", len(res), "| groups:", [r[4] for r in res])
