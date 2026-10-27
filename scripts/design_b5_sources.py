"""U35-C11：只读记录补洞出生、宿主碰撞去向及流量；不是成分示踪。"""
from __future__ import annotations
import functools
import inspect
import json
import sys
import numpy as np
import design_b5_coupling as coupling
import design_b5_evolution as evolution
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import orthonormalize
from mythic.tectonics.kinematics import RotationFileKinematics

def birth_balance(total, shares):
    """逐洞核对借入与新增，拒绝用截断掩盖超额。"""
    total, shares = np.asarray(total, float), np.asarray(shares, float)
    if total.ndim != 1 or shares.ndim != 2 or len(total) != len(shares):
        raise ValueError("数组形状不一致")
    if not np.isfinite(total).all() or not np.isfinite(shares).all() or (total < 0).any() or (shares < 0).any():
        raise ValueError("体积须有限且非负")
    borrowed = shares.sum(axis=1)
    created = total - borrowed
    if (created < -1e-7).any():
        raise ValueError("借入超过洞总体积")
    return borrowed, created

class SourceObserver(coupling.CouplingObserver):
    def __init__(self, case, checkpoint="u35_c11"):
        evolution.Observer.__init__(self, case, checkpoint)
        self.checkpoint = checkpoint
        coupling.CouplingObserver.current = self
        self.flow_rows, self.borrow_rows, self.borrow_checks = [], [], []
        self.flow_error = self.borrow_error = self.flow_global_balance = 0.
        self.pending = {}
        kin = RotationFileKinematics(evolution.ROOT / "data/worlds/design_v2/history_v2.rot")
        ref = np.eye(3)
        for t in range(300, 0, -1):
            ref = kin.stage(301, t) @ ref
            if (t - 1) % 5 == 0:
                ref = orthonormalize(ref)
        self.R_C_present = ref
        self.births = {}
        self.collisions, self.deaths, self.birth_host_flow = [], [], []
        self.last_engine = None

    def borrow(self, eng, loc):
        super().borrow(eng, loc)
        g, hg, new = loc["g"], loc["hg"], loc["new"]
        borrowed, created = birth_balance(loc["V"], loc["share"])
        xy = self.coordinates(eng, g.node[hg], g.plate[hg])
        nb, share = loc["nb"], loc["share"]
        for r, h in enumerate(np.flatnonzero(hg)):
            uid = int(g.uid[h])
            if uid in self.births:
                raise RuntimeError("出生UID重复")
            self.births[uid] = {
                "uid": uid, "t_Ma": float(eng.t), "plate": int(g.plate[h]), "origin": int(g.origin[h]),
                "C_frame_lon_lat": xy[r], "T_km": float(g.T[h]), "area_km2": float(eng.m.area_km2[g.node[h]]),
                "kind": "hole_fill", "total_km3": float(loc["V"][r]), "borrowed_km3": float(borrowed[r]), "created_km3": float(created[r]),
                "donors": [{"uid": int(new.uid[n]), "volume_km3": float(v), "plate": int(new.plate[n]),
                    "origin": int(new.origin[n])} for n, v in zip(nb[r], share[r]) if v > 0]}

    def rift_birth(self, eng, loc):
        g = loc["g"]
        cg = loc["cg"]
        if not cg.any():
            return
        xy = self.coordinates(eng, g.node[cg], g.plate[cg])
        for r, k in enumerate(np.flatnonzero(cg)):
            uid = int(g.uid[k])
            if uid in self.births:
                raise RuntimeError("裂谷出生UID重复")
            area = float(eng.m.area_km2[g.node[k]])
            volume = float(g.T[k] * area)
            self.births[uid] = {"uid": uid, "t_Ma": float(eng.t), "plate": int(g.plate[k]),
                "origin": int(g.origin[k]), "C_frame_lon_lat": xy[r], "T_km": float(g.T[k]),
                "area_km2": area, "kind": "rift_fill", "total_km3": volume,
                "borrowed_km3": 0., "created_km3": volume, "donors": []}

    def flow(self, eng, loc):
        super().flow(eng, loc)
        if eng.t > 36 or not self.births:
            return
        c, i, j, flux = eng.c, loc["i"], loc["j"], loc["flux"]
        born = np.fromiter(self.births, np.int64)
        for target, other, sign in ((i, j, 1.), (j, i, -1.)):
            for h in np.flatnonzero(np.isin(c.uid[target], born)):
                a, b = target[h], other[h]
                xy = self.coordinates(eng, c.node[[a, b]], c.plate[[a, b]])
                self.birth_host_flow.append({"t_Ma": float(eng.t), "target_uid": int(c.uid[a]),
                    "neighbor_uid": int(c.uid[b]), "volume_into_host_km3": float(sign * flux[h]),
                    "C_frame_target_lon_lat": xy[0], "C_frame_neighbor_lon_lat": xy[1]})

    def collision_transfer(self, eng, loc):
        s = int(loc["s_node"])
        uid = int(eng.c.uid[s])
        if uid not in self.births:
            return
        j, w = loc["j"], loc["w"]
        volume = eng.c.T[s] * eng.area[s] * w * eng.area[j] / (w * eng.area[j]).sum()
        xy = self.coordinates(eng, eng.c.node[j], eng.c.plate[j])
        self.collisions.append({"t_Ma": float(eng.t), "source_uid": uid,
            "source_whole_host_volume_km3": float(eng.c.T[s] * eng.area[s]),
            "targets": [{"uid": int(eng.c.uid[k]), "plate": int(eng.c.plate[k]),
                "volume_km3": float(v), "C_frame_lon_lat": xy[q]} for q, (k, v) in enumerate(zip(j, volume))]})

    def record_resolve(self, original, eng, pos, U, Q, B):
        s = U[np.isin(eng.c.uid[U], list(self.births)) & eng.alive[U]]
        uid, vol = eng.c.uid[s].copy(), (eng.c.T[s] * eng.area[s]).copy()
        result = super().record_resolve(original, eng, pos, U, Q, B)
        for k in np.flatnonzero(~eng.alive[s]):
            self.deaths.append({"t_Ma": float(eng.t), "uid": int(uid[k]), "whole_host_volume_km3": float(vol[k])})
        return result

    def capture(self, eng):
        super().capture(eng)
        self.last_engine = eng

    def finish(self):
        super().finish()
        eng = self.last_engine
        current = {int(u): int(i) for i, u in enumerate(eng.c.uid) if eng.alive[i]}
        deaths = {r["uid"]: r for r in self.deaths}
        for uid, birth in self.births.items():
            if uid in current:
                k = current[uid]
                birth["final_host"] = {"alive": True, "T_km": float(eng.c.T[k]),
                    "volume_km3": float(eng.c.T[k] * eng.area[k]), "plate": int(eng.c.plate[k]),
                    "C_frame_lon_lat": self.coordinates(eng, eng.c.node[[k]], eng.c.plate[[k]])[0]}
            else:
                birth["final_host"] = {"alive": False, "removal": deaths.get(uid)}
                if uid not in deaths:
                    raise RuntimeError("宿主消失但无去向记录")
        residuals = {kind: sum(r["created_km3"] for r in self.births.values() if r["kind"] == kind) - eng.budget.get(budget, 0.) for kind, budget in (("hole_fill", "hole_created"), ("rift_fill", "rift_fill"))}
        residual = max(abs(v) for v in residuals.values())
        if abs(residual) > 1e-5:
            raise RuntimeError("出生账与新增预算不闭合")
        error = max((abs(sum(t["volume_km3"] for t in r["targets"]) - r["source_whole_host_volume_km3"]) for r in self.collisions), default=0.)
        result = {"checkpoint": "U35-C11", "method_class": "procedural_constrained",
            "method": "只读行事件提取出生分配和碰撞；实际宿主通量；非成分守恒示踪",
            "resolution": "L7约60km；洲C现今参照系", "units": {"volume": "km3", "time": "Ma", "thickness": "km"},
            "births": list(self.births.values()), "collision_transfers": self.collisions, "host_removals": self.deaths,
            "birth_host_flow_since36Ma": self.birth_host_flow, "created_budget_residual_km3": residual, "birth_budget_residuals_km3": residuals,
            "collision_distribution_max_residual_km3": error,
            "C_frame_final_matrix_max_error": float(abs(eng.R[301] - self.R_C_present).max()),
            "limitations": ["新增与借入在同一宿主混合；后续全宿主流量不等于新增成分通量",
                "存活宿主库存不是新增量存留；不推定混合后的成分分数",
                "显式新增不是求解的沉积或岩浆；没有新物理或最终地形修正"]}
        (self.dest / "sources.json").write_text(json.dumps(evolution.sections.clean(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def collision_trace(original):
    lines, start = inspect.getsourcelines(original)
    points = [start + i for i, l in enumerate(lines) if "add[j] += c.T[s_node]" in l]
    if len(points) != 1:
        raise RuntimeError("碰撞观测点改变")
    @functools.wraps(original)
    def call(eng, *args, **kwargs):
        if sys.gettrace() is not None:
            raise RuntimeError("不能覆盖已有追踪器")
        def trace(frame, event, arg):
            if frame.f_code is not original.__code__:
                return None
            if event == "line" and frame.f_lineno == points[0]:
                coupling.CouplingObserver.current.collision_transfer(eng, frame.f_locals)
            return trace
        sys.settrace(trace)
        try:
            return original(eng, *args, **kwargs)
        finally:
            sys.settrace(None)
    return call

def commit_trace(original):
    lines, start = inspect.getsourcelines(original)
    needles = {"hg = hole_src[uniq] >= 0": "rift_birth",
        "np.subtract.at(new.T, nb.ravel()": "borrow",
        'self.budget["hole_created"]': "check_borrow"}
    targets = {start + i: label for i, line in enumerate(lines) for needle, label in needles.items() if needle in line}
    if set(targets.values()) != set(needles.values()):
        raise RuntimeError("提交观测点改变")
    @functools.wraps(original)
    def call(eng, *args, **kwargs):
        if sys.gettrace() is not None:
            raise RuntimeError("不能覆盖已有追踪器")
        def trace(frame, event, arg):
            if frame.f_code is not original.__code__:
                return None
            if event == "line" and frame.f_lineno in targets:
                obs = coupling.CouplingObserver.current
                label = targets[frame.f_lineno]
                if label == "check_borrow":
                    obs.check_borrow()
                else:
                    getattr(obs, label)(eng, frame.f_locals)
            return trace
        sys.settrace(trace)
        try:
            return original(eng, *args, **kwargs)
        finally:
            sys.settrace(None)
    return call

def main():
    if "--control" in sys.argv:
        sys.argv.remove("--control")
        return evolution.main(evolution.Observer, "u35_c11_control")
    surface, commit, resolve = Engine._surface, Engine.commit, Engine._resolve
    Engine._surface, Engine.commit = coupling.traced(surface, "surface"), commit_trace(commit)
    Engine._resolve = collision_trace(resolve)
    try:
        return evolution.main(SourceObserver, "u35_c11")
    finally:
        Engine._surface, Engine.commit, Engine._resolve = surface, commit, resolve
        coupling.CouplingObserver.current = None

if __name__ == "__main__":
    raise SystemExit(main())
