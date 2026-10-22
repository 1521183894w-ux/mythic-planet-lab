"""两条路线共用的运行记录：板块速度、示踪点（古纬度历史）、快照。"""

from __future__ import annotations

import numpy as np
from common import R_KM, lonlat_to_xyz, xyz_to_lonlat
from engine import Engine

# 示踪点（100 Ma 时的位置，经度、纬度）：北大陆西南岬角、南大陆东北岬角、北大陆西北角（路线 B 剧本里裂离的那块）
TRACERS = {"C1 西南岬角": (-14.0, 20.0), "C2 东北岬角": (38.0, 12.0), "C1 西北角": (-30.0, 62.0)}


class RunLog:
    def __init__(self, eng: Engine):
        self.tracer_names = list(TRACERS)
        self.tracer_x = lonlat_to_xyz(*np.array(list(TRACERS.values())).T)
        self.tracer_hist = {n: [] for n in self.tracer_names}     # (t, lon, lat, plate)
        self.speed_hist: list[dict] = []
        self.snapshots: list[dict] = [eng.snapshot()]
        self._record_tracers(eng, owners=self._owners(eng))

    def _owners(self, eng: Engine) -> list[int]:
        out = [-1] * len(self.tracer_x)
        for q in eng.plates:
            par = eng.covering(q, self.tracer_x)
            for k in np.flatnonzero(par >= 0):
                if out[k] < 0:
                    out[k] = q
        return out

    def _record_tracers(self, eng, owners):
        lon, lat = xyz_to_lonlat(self.tracer_x)
        for n, lo, la, q in zip(self.tracer_names, lon, lat, owners):
            self.tracer_hist[n].append((eng.t, float(lo), float(la), int(q)))

    def before_step(self, eng: Engine) -> list[int]:
        return self._owners(eng)

    def after_step(self, eng: Engine, dR: dict, omega: dict, owners: list[int], committed: bool):
        for k, q in enumerate(owners):
            if q in dR:
                self.tracer_x[k] = dR[q] @ self.tracer_x[k]
        self._record_tracers(eng, owners)
        pos = eng.positions()
        rec = {"t": eng.t}
        for q in eng.plates:
            sel = eng.alive & (eng.c.plate == q)
            if sel.any() and q in omega:   # 这一步刚裂出的新板块下一步才有速度
                a = eng.area[sel]
                v = np.linalg.norm(np.cross(omega[q], pos[sel]), axis=1) * R_KM / 10.0
                rec[int(q)] = float((v * a).sum() / a.sum())
        self.speed_hist.append(rec)
        if committed and abs(eng.t - 10 * round(eng.t / 10)) < 1e-6:
            self.snapshots.append(eng.snapshot())
