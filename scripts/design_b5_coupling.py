"""U35-C9：只读捕获实际侧向通量与补洞分配，不重写求解函数。"""
from __future__ import annotations

import functools
import inspect
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_b5_evolution as evolution
from mythic.tectonics.engine import Engine, HOLE_BORROW_T_KM
from mythic.tectonics.geometry import xyz_to_lonlat


class CouplingObserver(evolution.Observer):
    current = None

    def __init__(self, case, checkpoint="u35_c9"):
        super().__init__(case, checkpoint)
        CouplingObserver.current = self
        self.flow_rows, self.borrow_rows, self.borrow_checks = [], [], []
        self.flow_error = 0.
        self.borrow_error = 0.
        self.flow_global_balance = 0.
        self.pending = {}
        with (ROOT / ".cache/u35_c8/baseline/final_state.pkl").open("rb") as stream:
            self.R_C_present = pickle.load(stream)["engine"].R[301].copy()

    def coordinates(self, eng, node, plate):
        pos = np.einsum("nij,nj->ni", np.stack([eng.R[int(q)] for q in plate]), eng.m.xyz[node])
        # 转到洲C现今参照系，便于跨时间对照；不是变形网格。
        if 301 in eng.R:
            pos = pos @ eng.R[301] @ self.R_C_present.T
        lo, la = xyz_to_lonlat(pos)
        return np.column_stack([lo, la])

    def flow(self, eng, loc):
        c = eng.c
        i, j, flux = loc["i"], loc["j"], loc["flux"]
        before = c.T.copy()
        self.pending["flow"] = (eng, before, loc["dv"].copy())
        self.flow_global_balance = max(self.flow_global_balance, abs(float(loc["dv"].sum())))
        if eng.t > 36:
            return
        # 保存涉及初始21个宿主的每一条真实边，保留正反向，不只挑输入。
        for endpoint, other, sign in ((i, j, 1.), (j, i, -1.)):
            hit = np.flatnonzero(np.isin(c.uid[endpoint], self.uids))
            for h in hit:
                a, b = endpoint[h], other[h]
                xy = self.coordinates(eng, c.node[[a, b]], c.plate[[a, b]])
                self.flow_rows.append({"t_Ma": eng.t, "target_uid": int(c.uid[a]), "neighbor_uid": int(c.uid[b]),
                    "target_plate": int(c.plate[a]), "neighbor_plate": int(c.plate[b]), "neighbor_origin": int(c.origin[b]),
                    "T_target_km": float(before[a]), "T_neighbor_km": float(before[b]),
                    "target_area_km2": float(eng.area[a]), "volume_into_target_km3": float(sign * flux[h]),
                    "C_frame_target_lon_lat": xy[0], "C_frame_neighbor_lon_lat": xy[1]})

    def check_flow(self, eng):
        state = self.pending.pop("flow", None)
        if state is not None:
            _, before, dv = state
            error = np.max(abs((eng.c.T - before) * eng.area - dv))
            self.flow_error = max(self.flow_error, float(error))

    def borrow(self, eng, loc):
        new, g = loc["new"], loc["g"]
        nb, share, V = loc["nb"], loc["share"], loc["V"]
        area = eng.m.area_km2[new.node]
        used = np.bincount(nb.ravel(), weights=share.ravel(), minlength=len(new))
        avail = np.where(new.ctype == evolution.CONT, np.maximum(new.T - HOLE_BORROW_T_KM, 0.) * area, 0.)
        overdraft = np.maximum(used - avail, 0.)
        post = new.T - used / area
        self.pending["borrow"] = (new, new.T.copy(), used, area)
        involved = used > 0
        check = {"t_Ma": eng.t, "holes": len(V), "borrowed_km3": float(used.sum()),
            "created_km3": float((V - share.sum(axis=1)).sum()),
            "oversubscribed_donors": int((overdraft > 1e-6).sum()), "overdraft_km3": float(overdraft.sum()),
            "max_overdraft_equivalent_km": float((overdraft / area).max()),
            "minimum_post_donor_T_km": float(post[involved].min()) if involved.any() else None}
        self.borrow_checks.append(check)
        if eng.t > 36:
            return
        hg = loc["hg"]
        newborn = g.uid[hg]
        gxy = self.coordinates(eng, g.node[hg], g.plate[hg])
        for row in range(len(V)):
            select = np.flatnonzero(np.isin(new.uid[nb[row]], self.uids) & (share[row] > 0))
            for k in select:
                a = nb[row, k]
                xy = self.coordinates(eng, new.node[[a]], new.plate[[a]])[0]
                self.borrow_rows.append({"t_Ma": eng.t, "donor_uid": int(new.uid[a]), "newborn_uid": int(newborn[row]),
                    "donor_origin": int(new.origin[a]), "newborn_origin": int(g.origin[hg][row]),
                    "newborn_plate": int(g.plate[hg][row]), "donor_T_before_km": float(new.T[a]),
                    "donor_T_after_all_holes_km": float(post[a]), "donor_available_km3": float(avail[a]),
                    "donor_total_taken_km3": float(used[a]), "volume_to_hole_km3": float(share[row, k]),
                    "hole_total_volume_km3": float(V[row]), "hole_created_km3": float(V[row] - share[row].sum()),
                    "C_frame_donor_lon_lat": xy, "C_frame_hole_lon_lat": gxy[row]})

    def check_borrow(self):
        state = self.pending.pop("borrow", None)
        if state is not None:
            new, before, used, area = state
            self.borrow_error = max(self.borrow_error, float(np.max(abs((before - new.T) * area - used))))

    def finish(self):
        super().finish()
        data = {"checkpoint": "U35-C9", "method_class": "procedural_constrained", "reference_frame": "洲C现今参照系",
            "method": "Python行事件只读提取实际求解函数局部变量；观测不修改数组或分配",
            "flow_reconstruction_max_error_km3": self.flow_error,
            "flow_global_balance_max_km3": self.flow_global_balance,
            "borrow_reconstruction_max_error_km3": self.borrow_error,
            "flow_edges_since_36Ma": self.flow_rows, "borrow_transfers_since_36Ma": self.borrow_rows,
            "all_commit_borrow_checks": self.borrow_checks,
            "limitations": ["侧向通量是厚度扩散代理，不是流变方程求得的下地壳流", "出生UID仅在本候选内唯一，跨候选不能按编号对应"]}
        (self.dest / "coupling.json").write_text(json.dumps(evolution.sections.clean(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def traced(original, mode):
    lines, start = inspect.getsourcelines(original)
    target = {}
    needles = {"surface": {"c.T += dv / self.area": "flow", "self._collapse(cont)": "check_flow"},
        "commit": {"np.subtract.at(new.T, nb.ravel()": "borrow", 'self.budget["hole_created"]': "check_borrow"}}[mode]
    for offset, line in enumerate(lines):
        for needle, label in needles.items():
            if needle in line:
                target[start + offset] = label
    if set(target.values()) != set(needles.values()):
        raise RuntimeError("求解函数位置改变，请核查只读观测点")

    @functools.wraps(original)
    def call(eng, *a, **kw):
        obs = CouplingObserver.current
        if obs is None:
            return original(eng, *a, **kw)
        old_trace = sys.gettrace()
        if old_trace is not None:
            raise RuntimeError("不能覆盖已有调试追踪器")
        def trace(frame, event, arg):
            if frame.f_code is not original.__code__:
                return None
            if event == "line" and frame.f_lineno in target:
                label = target[frame.f_lineno]
                if label in ("flow", "borrow"):
                    getattr(obs, label)(eng, frame.f_locals)
                elif label == "check_flow":
                    obs.check_flow(eng)
                else:
                    obs.check_borrow()
            return trace
        sys.settrace(trace)
        try:
            return original(eng, *a, **kw)
        finally:
            sys.settrace(old_trace)
    return call


def main():
    surface, commit = Engine._surface, Engine.commit
    Engine._surface, Engine.commit = traced(surface, "surface"), traced(commit, "commit")
    try:
        return evolution.main(CouplingObserver, "u35_c9")
    finally:
        Engine._surface, Engine.commit = surface, commit
        CouplingObserver.current = None


if __name__ == "__main__":
    raise SystemExit(main())
