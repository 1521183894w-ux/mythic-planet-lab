"""U35-C8：观测B5现今宿主的逐过程体积账；完整重跑，候选输出隔离。

包装器仅记录函数前后的实际状态，预算闭合与旧运行逐数组一致性另行核查。
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_arc_sections as sections
from mythic.tectonics.crust import CONT, elevation
from mythic.tectonics.geometry import R_KM, chord_from_km, km_from_chord, xyz_to_lonlat, unit
from mythic.tectonics.volcanism import VolcanoField


class Observer:
    def __init__(self, case, checkpoint="u35_c8"):
        data = json.loads((ROOT / "data/worlds/design_v2/u35_c7_diagnostic.json").read_text(encoding="utf-8"))
        self.uids = np.array(sorted(set(sum((r["native_uids"] for r in data["B5"]["polygons"].values()), []))), np.int64)
        self.center = int(np.flatnonzero(self.uids == 52985)[0])
        self.dest = ROOT / ".cache" / checkpoint / case
        self.dest.mkdir(parents=True, exist_ok=True)
        self.process = {}
        self.timeline = []
        self.donations = []
        self.resolve_max_error = 0.
        self.initial = None

    def slots(self, eng):
        j = np.searchsorted(eng.c.uid, self.uids)
        safe = np.minimum(j, len(eng.c) - 1)
        ok = (j < len(eng.c)) & (eng.c.uid[safe] == self.uids) & eng.alive[safe]
        return safe, ok

    def stock(self, eng):
        j, ok = self.slots(eng)
        return np.where(ok, eng.c.T[j] * eng.area[j], 0.)

    def add(self, name, change):
        self.process[name] = self.process.get(name, np.zeros(len(self.uids))) + change

    def positions(self, eng, j):
        matrices = np.stack([eng.R[int(q)] for q in eng.c.plate[j]])
        return np.einsum("nij,nj->ni", matrices, eng.m.xyz[eng.c.node[j]])

    def capture(self, eng):
        if self.initial is None:
            self.initial = self.stock(eng)
        j, ok = self.slots(eng)
        pos = self.positions(eng, j)
        lo, la = xyz_to_lonlat(pos)
        row = {"t_Ma": float(eng.t), "alive": ok.copy(), "stock_km3": self.stock(eng),
            "T_km": np.where(ok, eng.c.T[j], np.nan), "base_elevation_m": np.where(ok, elevation(eng.c.gather(j), eng.p), np.nan),
            "plate": np.where(ok, eng.c.plate[j], -1), "origin": np.where(ok, eng.c.origin[j], -1),
            "lon": np.where(ok, lo, np.nan), "lat": np.where(ok, la, np.nan),
            "process_cumulative_km3": {k: v.copy() for k, v in self.process.items()}}
        residual = row["stock_km3"] - self.initial - sum(self.process.values(), np.zeros(len(self.uids)))
        row["unassigned_residual_km3"] = residual
        if eng.t <= 60 and ok[self.center] and eng.c.plate[j[self.center]] == 301 and 401 in eng.R:
            dj = np.flatnonzero(eng.alive & (eng.c.plate == 401) & (eng.c.ctype == CONT))
            dpos = self.positions(eng, dj)
            target = pos[self.center]
            dist, k = cKDTree(dpos).query(target)
            normal = unit(dpos[k] - (dpos[k] @ target) * target)
            tangent = np.cross(target, normal)
            relative = np.cross(eng.omega[401] - eng.omega[301], target) * R_KM / 10.
            dx, dy = xyz_to_lonlat(dpos[k])
            row["center_to_nearest_D_cont"] = {"distance_km": float(km_from_chord(dist)), "D_uid": int(eng.c.uid[dj[k]]),
                "D_lon_lat": [float(dx), float(dy)], "separation_cm_yr": float(relative @ normal),
                "shear_cm_yr": float(relative @ tangent),
                "meaning": "两宿主连线法向的相对速度；不是已求解的盆内应变或断层释放率"}
        self.timeline.append(row)

    def record_resolve(self, original, eng, pos, U, Q, B):
        # 只对有追踪宿主位于上盘的调用保存必要的前态。
        j, ok = self.slots(eng)
        if not np.any(ok & np.isin(eng.c.plate[j], Q)):
            return original(eng, pos, U, Q, B)
        stock = self.stock(eng)
        was_alive = eng.alive.copy()
        donor_T = eng.c.T[U].copy()
        donor_type = eng.c.ctype[U].copy()
        donor_plate = eng.c.plate[U].copy()
        donor_origin = eng.c.origin[U].copy()
        donor_uid = eng.c.uid[U].copy()
        result = original(eng, pos, U, Q, B)
        change = self.stock(eng) - stock
        self.add("collision", change)
        removed = was_alive[U] & ~eng.alive[U] & (donor_type == CONT)
        if not removed.any() or not np.any(change > 0):
            return result
        predicted = np.zeros(len(self.uids))
        active = was_alive.copy()
        # 洋壳已在碰撞前消耗；陆壳按引擎同样的上盘顺序移除。
        active[U[was_alive[U] & ~eng.alive[U] & (donor_type != CONT)]] = False
        for o in np.unique(Q[removed]):
            source = np.flatnonzero(removed & (Q == o))
            tgt = np.flatnonzero(active & (eng.c.plate == o) & (eng.c.ctype == CONT))
            tree = cKDTree(pos[tgt])
            for k, neighbors in zip(source, tree.query_ball_point(pos[U[source]], chord_from_km(eng.p.collision_radius_km))):
                if not neighbors:
                    continue
                local = tgt[neighbors]
                distance = km_from_chord(np.linalg.norm(pos[local] - pos[U[k]], axis=1))
                weight = (1. - (distance / eng.p.collision_radius_km) ** 2) ** 2
                norm = (weight * eng.area[local]).sum()
                incoming = donor_T[k] * eng.area[U[k]]
                uid = eng.c.uid[local]
                select = np.isin(uid, self.uids)
                if not select.any():
                    continue
                target_rows = np.searchsorted(self.uids, uid[select])
                amounts = incoming * weight[select] * eng.area[local[select]] / norm
                predicted[target_rows] += amounts
                sx, sy = xyz_to_lonlat(pos[U[k]])
                self.donations.append({"t_Ma": float(eng.t), "source_uid": int(donor_uid[k]), "source_plate": int(donor_plate[k]),
                    "source_origin": int(donor_origin[k]), "source_lon_lat": [float(sx), float(sy)],
                    "source_volume_km3": float(incoming), "overriding": int(o),
                    "target_uid": uid[select].copy(), "deposited_km3": amounts,
                    "source_distance_km": distance[select].copy()})
            active[U[source]] = False
        # 被删宿主的负库存不属于接收量。
        error = np.max(abs(predicted - np.maximum(change, 0.)))
        self.resolve_max_error = max(self.resolve_max_error, float(error))
        return result

    def finish(self):
        output = {"watch_uids": self.uids, "center_uid": 52985, "initial_stock_km3": self.initial,
            "timeline": self.timeline, "donations": self.donations, "collision_reconstruction_max_error_km3": self.resolve_max_error,
            "method_class": "procedural_constrained", "purpose": "实际演化逐过程观测；不是新物理求解器"}
        (self.dest / "ledger.json").write_text(json.dumps(sections.clean(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(observer_class=Observer, checkpoint="u35_c8"):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--history-config", default="config/design/history_v2_u35_c4_strain.yaml")
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.case):
        ap.error("case须为小写字母、数字和下划线")
    observer = observer_class(args.case, checkpoint=checkpoint)
    originals = {name: getattr(fwd.Engine, name) for name in ("step", "_resolve", "_arc", "_surface", "_collapse", "commit", "erosion_rate")}
    original_simulate = fwd.simulate
    original_volcano = VolcanoField.step
    from mythic.tectonics import rifts, reactivation, plumes
    functions = [(rifts, "apply_graben"), (reactivation, "apply_shortening"), (plumes, "apply_lip"), (plumes, "apply_lip_weights")]
    saved = [(module, name, getattr(module, name)) for module, name in functions]

    def wrap(name, original):
        @functools.wraps(original)
        def call(eng, *a, **kw):
            before = observer.stock(eng)
            collapse_before = observer.process.get("collapse", np.zeros(len(observer.uids))).copy()
            erosion_before = observer.process.get("erosion", np.zeros(len(observer.uids))).copy()
            result = original(eng, *a, **kw)
            delta = observer.stock(eng) - before
            if name == "surface":
                delta -= observer.process.get("collapse", collapse_before) - collapse_before
                delta -= observer.process.get("erosion", erosion_before) - erosion_before
                name_out = "lateral_flow"
            else:
                name_out = name
            observer.add(name_out, delta)
            return result
        return call

    def step(eng, *a, **kw):
        if observer.initial is None:
            observer.capture(eng)
        return originals["step"](eng, *a, **kw)

    def erosion(eng, *a, **kw):
        rate = originals["erosion_rate"](eng, *a, **kw)
        j, ok = observer.slots(eng)
        loss = np.minimum(rate[j] * eng.p.dt, np.maximum(eng.c.T[j] - eng.p.T_cont, 0.))
        observer.add("erosion", np.where(ok & (eng.c.ctype[j] == CONT), -loss * eng.area[j], 0.))
        return rate

    def volcano(field, eng):
        result = original_volcano(field, eng)
        observer.capture(eng)
        return result

    def simulate(*a, **kw):
        bound = inspect.signature(original_simulate).bind(*a, **kw)
        bound.apply_defaults()
        result = original_simulate(*a, **kw)
        eng, frames, vframes = result
        with (observer.dest / "final_state.pkl").open("wb") as stream:
            pickle.dump({"engine": eng, "fixed_frame": frames[-1], "volcanoes": vframes[-1],
                "plumes": bound.arguments.get("plumes"), "rifts": bound.arguments.get("rifts"), "rot": str(bound.arguments["rot"])}, stream, protocol=5)
        observer.finish()
        return result

    fwd.Engine.step = step
    fwd.Engine._resolve = functools.wraps(originals["_resolve"])(lambda eng, *a, **kw: observer.record_resolve(originals["_resolve"], eng, *a, **kw))
    for name, label in (("_arc", "arc"), ("_surface", "surface"), ("_collapse", "collapse"), ("commit", "commit_birth_or_borrow")):
        setattr(fwd.Engine, name, wrap(label, originals[name]))
    fwd.Engine.erosion_rate = erosion
    for module, name, original in saved:
        setattr(module, name, wrap({"apply_graben": "graben_thinning", "apply_shortening": "prescribed_shortening"}.get(name, "lip"), original))
    VolcanoField.step = volcano
    fwd.simulate = simulate
    old_argv = sys.argv
    candidate = checkpoint + "_" + args.case
    sys.argv = ["design_forward.py", "--design", "v2", "--history", "v2", "--history-config", args.history_config,
                "--candidate", candidate, "--no-figures"]
    started = time.perf_counter()
    try:
        rc = fwd.main()
        for suffix in (".npz", ".json"):
            shutil.copy2(ROOT / f".cache/tectonics/design/candidates/{candidate}/forward_v2_v2_L7{suffix}", observer.dest / ("forward" + suffix))
        world_dir = ROOT / "data/worlds/design_v2/candidates" / candidate
        shutil.copy2(world_dir / "forward_v2.json", observer.dest / "summary.json")
        shutil.copy2(world_dir / f"design_v2_forward_v2_0Ma_{candidate}_1024x512.nc", observer.dest / "world.nc")
        print(f"[{checkpoint}] {args.case} 完整运行结束，{time.perf_counter() - started:.2f} s；UID数{len(observer.uids)}；碰撞归因最大残差{observer.resolve_max_error:.3g} km³", flush=True)
        return rc
    finally:
        for name, original in originals.items():
            setattr(fwd.Engine, name, original)
        for module, name, original in saved:
            setattr(module, name, original)
        VolcanoField.step = original_volcano
        fwd.simulate = original_simulate
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
