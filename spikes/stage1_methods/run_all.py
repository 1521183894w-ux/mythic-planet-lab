"""阶段 1.1：两条路线 × 多个种子，并行运行、判定、保存。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u spikes\\stage1_methods\\run_all.py --level 7 --seeds 0-7

每个运行保存：.cache/spikes/stage1/runs/<route><seed>_L<level>.npz（每 10 Myr 一帧的地壳属性），
figures/stage1/methods/results/run_<route><seed>_L<level>.json（事件、提交诊断、速度、示踪点、规划、判定、耗时）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def one(route: str, seed: int, level: int) -> dict:
    import numpy as np
    import psutil
    from common import CACHE, RES_DIR, Raster, code_version, load_mesh
    from evaluate import basin_check

    mod = __import__("route_a" if route == "A" else "route_b")
    res = mod.run(seed, level)
    eng, log = res.pop("engine"), res.pop("log")
    mesh = load_mesh(level)
    raster = Raster(mesh, 512)
    final = log.snapshots[-1]
    ev = basin_check(raster.grid, raster(final["elev"]) > 0, raster(final["origin"]))

    run_dir = CACHE / "runs"
    run_dir.mkdir(exist_ok=True)
    tag = f"{route}{seed}_L{level}"
    arrs = {}
    for k, s in enumerate(log.snapshots):
        for f in ("elev", "T", "age"):
            arrs[f"{f}_{k}"] = s[f].astype(np.float32)
        for f in ("plate", "ctype", "origin", "oro"):
            arrs[f"{f}_{k}"] = s[f].astype(np.int16)
        arrs[f"orot_{k}"] = s["oro_t"].astype(np.float32)
        qs = sorted(s["omega"])
        arrs[f"omega_ids_{k}"] = np.array(qs, np.int16)
        arrs[f"omega_vec_{k}"] = np.array([s["omega"][q] for q in qs])
    arrs["times"] = np.array([s["t"] for s in log.snapshots])
    np.savez_compressed(run_dir / f"{tag}.npz", **arrs)

    steps = [s["wall_s"] for s in eng.step_log]
    summary = {
        **res,
        "code_version": code_version(),
        "eval_0Ma": ev,
        "events": eng.events,
        "commit_log": eng.commit_log,
        "budget": eng.budget,
        "n_steps": len(steps),
        "step_wall_s_mean": float(np.mean(steps)),
        "step_wall_s_max": float(np.max(steps)),
        "speeds": log.speed_hist,
        "tracers": log.tracer_hist,
        "peak_rss_mb": psutil.Process().memory_info().rss / 1e6,
        "peak_wset_mb": getattr(psutil.Process().memory_info(), "peak_wset", 0) / 1e6,
    }
    (RES_DIR / f"run_{tag}.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {"route": route, "seed": seed, "wall_s": round(res["wall_s"], 1),
             "official": ev["official"]["hit"], "loose": ev["loose"]["hit"],
             "area": round(ev["loose"].get("area_Mkm2", 0.0), 2), "aspect": round(ev["loose"].get("aspect_ratio", 0.0), 2),
             "straits": ev["loose"].get("straits_to_ocean"), "events": [e["kind"] for e in eng.events]}
    if route == "B":
        brief["plan_ok"] = res["plan"]["ok"]
        brief["rot_err"] = res["rot_roundtrip_max_err"]
    return brief


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--routes", default="A,B")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    routes = args.routes.split(",")
    jobs = [(r, s, args.level) for s in seeds for r in routes]
    t0 = time.perf_counter()
    if args.workers <= 1:
        for j in jobs:
            print(json.dumps(one(*j), ensure_ascii=False), flush=True)
    else:
        os.environ["MYTHIC_KD_WORKERS"] = str(max(1, (os.cpu_count() or 4) // args.workers))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one, *j): j for j in jobs}
            for f in as_completed(futs):
                try:
                    print(json.dumps(f.result(), ensure_ascii=False), flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    print("FAILED", futs[f], repr(e), traceback.format_exc(), flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s for {len(jobs)} runs", flush=True)


if __name__ == "__main__":
    main()
