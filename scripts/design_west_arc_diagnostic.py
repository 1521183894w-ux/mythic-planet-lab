"""U35-C1：观测既有正向演化；不修改引擎、终图或磁盘上的历史参数。

用法：python scripts/design_west_arc_diagnostic.py --case previous --no-figures
      python scripts/design_west_arc_diagnostic.py --case latest
previous 仅在本进程内恢复 WT—ALB 在 6 Ma 结束的旧设定，用于重算对照。
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
from mythic.tectonics.crust import CONT, elevation
from mythic.tectonics.geometry import xyz_to_lonlat, unit
from mythic.tectonics.kinematics import RotationFileKinematics
from mythic.tectonics.plumes import swell_m
from mythic.tectonics.rifts import rift_relief_m
from mythic.tectonics.volcanism import VolcanoField

OUT = ROOT / ".cache/u35_c1"
KEY_TIMES = (38, 30, 10, 6, 5, 0)


def finite_percentile(values, q):
    a = np.asarray(values)
    return float(np.percentile(a, q)) if a.size else None


class Observer:
    """读取物理更新后的状态；单元来源和当前板块归属分别记录。"""

    def __init__(self, bound, case):
        self.args = bound.arguments
        self.mesh = self.args["mesh"]
        self.kin = RotationFileKinematics(self.args["rot"])
        self.rows = []
        self.frames = []
        self.dest = OUT / case
        self.dest.mkdir(parents=True, exist_ok=True)
        self.hcfg = self.args["arcs_hcfg"]
        self.ids = fwd.plate_codes(self.hcfg)
        self.alb = self.ids["ALB"]
        self.event_count = 0
        self.events = []
        self.transfers = []
        self.transferred_volume = 0.0
        world = self.args["world"]
        c = world.crust
        m = c.origin == self.alb
        np.savez_compressed(self.dest / "initial_source.npz",
                            node=c.node[m], plate=c.plate[m], T=c.T[m],
                            ctype=c.ctype[m], origin=c.origin[m],
                            xyz=self.mesh.xyz[c.node[m]], area=self.mesh.area_km2[c.node[m]])

    def capture(self, eng):
        c, area = eng.c, eng.area
        t = int(round(eng.t))
        source = eng.alive & (c.origin == self.alb) & (c.ctype == CONT)
        own = eng.alive & (c.plate == self.alb) & (c.ctype == CONT)
        inherited = own & source
        added = own & ~source
        row = {
            "t_Ma": t, "committed": eng.steps_since_commit == 0,
            "collision_deposited_cumulative_km3": self.transferred_volume,
            "source_n": int(source.sum()),
            "source_area_km2": float(area[source].sum()),
            "source_volume_km3": float((area[source] * c.T[source]).sum()),
            "source_T_p90_km": finite_percentile(c.T[source], 90),
            "alb_cont_n": int(own.sum()),
            "alb_cont_volume_km3": float((area[own] * c.T[own]).sum()),
            "alb_original_volume_km3": float((area[inherited] * c.T[inherited]).sum()),
            "alb_other_volume_km3": float((area[added] * c.T[added]).sum()),
            "alb_T_p90_km": finite_percentile(c.T[own], 90),
            "alb_at_cap_area_km2": float(area[own & (c.T >= eng.p.T_max_km - 1e-6)].sum()),
            "source_by_plate_km2": {
                str(int(q)): float(area[source & (c.plate == q)].sum())
                for q in np.unique(c.plate[source])
            },
        }
        self.rows.append(row)
        if t <= 40:
            pos = eng.positions()
            lon, lat = xyz_to_lonlat(pos)
            sel = eng.alive & (((lon >= -30) & (lon <= 10) & (lat >= 28) & (lat <= 48)) | source)
            data = {k: getattr(c, k)[sel].copy() for k in c.FIELDS}
            data.update(lon=lon[sel], lat=lat[sel], area=area[sel],
                        raw_elevation_m=elevation(c, eng.p)[sel])
            np.savez_compressed(self.dest / f"parcels_{t:03d}.npz", **data)
        for e in eng.events[self.event_count:]:
            if any(e.get(k) == self.alb for k in ("src", "dst", "down", "over", "plate", "downgoing", "overriding")):
                self.events.append(dict(e))
        self.event_count = len(eng.events)
        if t in KEY_TIMES:
            f = eng.snapshot()
            plumes = self.args.get("plumes") or {"swells": []}
            dyn = swell_m(self.mesh.xyz, float(t), plumes["swells"])
            for r in self.args.get("rifts") or []:
                if t <= r["t_start"] + 1e-9:
                    line = unit(r["pts0"] @ self.kin.total_from(r["anchor"], 0., float(t)).T)
                    dyn = dyn + rift_relief_m(self.mesh.xyz, line, r["half_km"],
                                r["flank_amp_m"], r["flank_decay_km"],
                                r["swell_amp_m"], r["swell_radius_km"])
            f["dyn"] = dyn.astype(np.float32)
            f["elev"] = f["elev"] + f["dyn"]
            self.frames.append(f)
            saved = {k: np.asarray(f[k]).astype(dt) for k, dt in fwd.FIELDS.items()}
            np.savez_compressed(self.dest / f"snapshot_{t:03d}.npz", **saved)
        if t <= 40 and t % 5 == 0:
            print(f"[U35-C1] {t} Ma: source={source.sum()}, ALB volume={row['alb_cont_volume_km3']:.0f} km3", flush=True)

    def finish(self):
        levels = fwd.eustatic_sea_level(self.frames, self.mesh.area_km2)
        data = {
            "case": self.dest.name,
            "observer": "VolcanoField.step 返回后只读采样；原方法每次仍调用一次",
            "method_class": {"crust": "procedural_constrained",
                             "elevation": "parameterized_physics",
                             "positions": "user_override"},
            "mesh_level": self.mesh.level,
            "units": {"time": "Ma", "thickness": "km", "volume": "km3", "elevation": "m"},
            "rows": self.rows,
            "events": self.events,
            "collision_transfers": self.transfers,
            "material_note": "origin 是宿主单元来源，不是混合岩石成分；碰撞转入体积单独记录，累计值未扣后续剥蚀。",
            "snapshot_sea_level_m": {str(t): float(s) for t, s in zip(KEY_TIMES, levels)},
        }
        (self.dest / "observations.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def archive_run(case):
    dest = OUT / case
    dest.mkdir(parents=True, exist_ok=True)
    run = ROOT / ".cache/tectonics/design/forward_v2_v2_L7"
    for suffix in (".npz", ".json"):
        shutil.copy2(run.with_suffix(suffix), dest / ("forward" + suffix))
    shutil.copy2(ROOT / "data/worlds/design_v2/forward_v2.json", dest / "summary.json")
    shutil.copy2(ROOT / "data/worlds/design_v2/design_v2_forward_v2_0Ma_1024x512.nc",
                 dest / "world.nc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("previous", "latest"), required=True)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-collision-ledger", action="store_true",
                        help="仅采样逐步状态，不包裹碰撞函数；用于复现额外体积账本的无干扰对照")
    args = parser.parse_args()
    original_yaml = fwd.yaml
    original_simulate = fwd.simulate
    original_step = VolcanoField.step
    original_resolve = fwd.Engine._resolve
    observers = []

    def load_config(stream):
        data = original_yaml.safe_load(stream)
        if args.case == "previous" and isinstance(data, dict) and "forward" in data and "blocks" in data:
            rules = data["forward"].get("no_collision", [])
            found = [r for r in rules if set(r["pair"]) == {"WT", "ALB"}]
            if len(found) != 1 or found[0]["to"] != 0:
                raise ValueError("旧配置对照只允许将唯一 WT—ALB 的 to=0 在内存中改回 6")
            found[0]["to"] = 6
        return data

    def observed_simulate(*positional, **keywords):
        bound = inspect.signature(original_simulate).bind(*positional, **keywords)
        bound.apply_defaults()
        observer = Observer(bound, args.case)
        observers.append(observer)

        def observed_step(field, eng):
            result = original_step(field, eng)
            observer.capture(eng)
            return result

        def observed_resolve(eng, pos, U, Q, B):
            c = eng.c
            if eng.t > 40 or not (np.any(Q == observer.alb) or np.any(c.origin[U] == observer.alb)):
                return original_resolve(eng, pos, U, Q, B)
            was_alive = eng.alive[U].copy()
            types = c.ctype[U].copy()
            donor_volume = c.T[U] * eng.area[U]
            donor_plate = c.plate[U].copy()
            donor_origin = c.origin[U].copy()
            target = np.flatnonzero(c.plate == observer.alb)
            before = c.T[target].copy()
            result = original_resolve(eng, pos, U, Q, B)
            removed = was_alive & ~eng.alive[U] & (types == CONT)
            incoming = removed & (Q == observer.alb)
            deposited = float(((c.T[target] - before) * eng.area[target]).sum())
            if incoming.any() or deposited > 0:
                entry = {"t_Ma": float(eng.t), "donor_n": int(incoming.sum()),
                    "donor_volume_km3": float(donor_volume[incoming].sum()),
                    "deposited_on_ALB_km3": deposited,
                    "donor_volume_by_plate_km3": {str(int(q)): float(donor_volume[incoming & (donor_plate == q)].sum())
                         for q in np.unique(donor_plate[incoming])},
                    "donor_volume_by_origin_km3": {str(int(q)): float(donor_volume[incoming & (donor_origin == q)].sum())
                         for q in np.unique(donor_origin[incoming])}}
                observer.transfers.append(entry)
                observer.transferred_volume += deposited
            return result

        VolcanoField.step = observed_step
        fwd.Engine._resolve = original_resolve if args.no_collision_ledger else observed_resolve
        try:
            result = original_simulate(*positional, **keywords)
            observer.finish()
            return result
        finally:
            VolcanoField.step = original_step
            fwd.Engine._resolve = original_resolve

    fwd.yaml = SimpleNamespace(safe_load=load_config)
    fwd.simulate = observed_simulate
    previous_argv = sys.argv
    sys.argv = ["design_forward.py", "--design", "v2", "--history", "v2"]
    if args.no_figures:
        sys.argv.append("--no-figures")
    try:
        rc = fwd.main()
        if len(observers) != 1:
            raise RuntimeError("必须恰好完成一次完整演化")
        archive_run(args.case)
        return rc
    finally:
        fwd.yaml = original_yaml
        fwd.simulate = original_simulate
        VolcanoField.step = original_step
        fwd.Engine._resolve = original_resolve
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
