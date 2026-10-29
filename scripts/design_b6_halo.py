"""U39：B6 核心阻流与外侧过渡的两案正向敏感性试验。

范围、阻力和 180 Ma 激活均为 user_override 假设，非地质重建。
复用 U38 的地壳规则，只改权重；不修改默认引擎或终态高程。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import design_b6_transition as previous
import design_forward as fwd
from mythic.tectonics.crust import CONT
from mythic.tectonics.engine import Engine

CASE = 'joint'
WIDTH_KM = 250.0
CACHE = ROOT / '.cache/u39_halo'
CACHE.mkdir(parents=True, exist_ok=True)
rows = []
ledger = []
original = Engine._surface


def region_weights(points):
    """B6 内部为 1，外侧 250 km 平滑归零；PL8 沿用 U38 内侧渐变。"""
    weights = []
    masks = []
    for polygon, tree in previous.regions:
        inside = previous.in_sphere_polygon(points, polygon)
        chord = tree.query(points)[0]
        distance = 2 * 6371.0 * np.arcsin(np.minimum(chord / 2, 1))
        q = np.clip(distance / WIDTH_KM, 0, 1)
        smooth = q * q * (3 - 2 * q)
        masks.append(inside)
        weights.append(smooth)
    basin = np.where(masks[0], 1.0, 1.0 - weights[0])
    plateau = np.where(masks[1], weights[1], 0.0)
    return basin, plateau, masks


def resistance(points):
    basin, plateau, _ = region_weights(points)
    return np.maximum(basin, plateau) if CASE == 'joint' else basin


def candidate_surface(self, dt, pos=None):
    if self.t > 180.0:
        return original(self, dt, pos)
    c, p = self.c, self.p
    pos = self.positions() if pos is None else pos
    ids = np.flatnonzero(self.alive & (c.plate == 101) & (c.ctype == CONT))
    future = pos[ids] @ previous.kin.total_from(101, float(self.t), 0.0).T
    wb, wp, inside = region_weights(future)
    weight = np.zeros(len(c))
    weight[ids] = np.maximum(wb, wp) if CASE == 'joint' else wb
    masks = {}
    for name, local in zip(('B6', 'PL8'), inside):
        mask = np.zeros(len(c), bool)
        mask[ids[local]] = True
        masks[name] = mask
    masks['PL8'] &= ~masks['B6']  # 账本区域互斥，交界优先 B6。
    cont = self.alive & (c.ctype == CONT)
    masks['other_continent'] = cont & ~masks['B6'] & ~masks['PL8']
    ex = np.where(cont, np.maximum(c.T - p.T_cont, 0.0), 0.0)
    if p.erosion_law == 'relief':
        loss = np.minimum(self.erosion_rate(pos) * dt, ex)
    elif p.erosion_law == 'linear':
        loss = ex * dt / p.tau_oro_myr
    else:
        raise ValueError('unknown erosion law')
    step = dict(t_ma=float(self.t), regions={})
    for name, mask in masks.items():
        step['regions'][name] = dict(
            cells=int(mask.sum()), area_km2=float(self.area[mask].sum()),
            erosion_removed_km3=float(np.dot(loss[mask], self.area[mask])))
    c.T -= loss
    self.budget['eroded'] += float((loss * self.area).sum())
    i, j = self.pedges[:, 0], self.pedges[:, 1]
    ok = cont[i] & cont[j] & (np.maximum(c.T[i], c.T[j]) > p.flow_T_km)
    if p.flow_min_T_km is not None:
        ok &= np.minimum(c.T[i], c.T[j]) >= p.flow_min_T_km
    dv = np.zeros(len(c))
    if ok.any():
        i, j = i[ok], j[ok]
        coef = p.kappa_km2_myr * dt * 2.0 / (3.0 * self.m.h_km**2)
        flux = coef * (c.T[j] - c.T[i]) * 0.5 * (self.area[i] + self.area[j])
        flux *= 1.0 - 0.99 * np.maximum(weight[i], weight[j])
        dv = np.bincount(i, flux, minlength=len(c)) - np.bincount(j, flux, minlength=len(c))
        c.T += dv / self.area
    for name, mask in masks.items():
        step['regions'][name]['lateral_net_in_km3'] = float(dv[mask].sum())
    # 临时占用的是接收容量；结束后逐格减回，不作为真实库存。
    before = c.T.copy()
    reserve = weight * np.maximum(p.T_max_km - c.T, 0.0)
    volume_before = float(np.dot(c.T, self.area))
    c.T += reserve
    self._collapse(cont)
    c.T -= reserve
    closure = float(np.dot(c.T, self.area) - volume_before)
    if abs(closure) > 0.01:
        raise RuntimeError(f'collapse balance failed: {closure} km3')
    step['collapse_balance_km3'] = closure
    for name, mask in masks.items():
        step['regions'][name]['collapse_net_in_km3'] = float(np.dot((c.T-before)[mask], self.area[mask]))
    ledger.append(step)
    if int(self.t) % 10 == 0:
        rows.append(dict(t_ma=float(self.t),
                         affected_cells=int(np.count_nonzero(weight)),
                         collapse_balance_km3=closure,
                         regions=step['regions']))
        print('U39', rows[-1], flush=True)


def main():
    global CASE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=['basin_only', 'joint'], default='joint')
    CASE = ap.parse_args().case
    candidate = 'u39_b6_halo' if CASE == 'basin_only' else 'u39_b6_pl8_halo'
    cache = CACHE / CASE
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    Engine._surface = candidate_surface
    saved_argv = sys.argv[:]
    sys.argv = ['design_forward.py', '--history-config', 'config/design/history_v2_u35_c10_progressive.yaml',
                '--candidate', candidate, '--no-figures']
    success = False
    try:
        fwd.main()
        success = True
    finally:
        Engine._surface = original
        sys.argv = saved_argv
        report = dict(
            success=success, candidate=candidate, case=CASE,
            method_class={'region':'user_override', 'crust':'procedural_constrained',
                          'elevation':'parameterized_physics', 'output':'interpolated'},
            method='u39_core_outward_halo_v1', resolution='L7 about 60 km; output 1024x512',
            units={'thickness':'km','volume':'km3','time':'Ma','elevation':'m'},
            assumptions=['B6 and PL8 follow plate 101; active since 180 Ma',
                         'B6 weight=1 inside; 250 km outward smoothstep halo',
                         'joint adds identical PL8 inward transition from U38',
                         'maximum lateral flux reduction 99%; core overflow capacity fully reserved',
                         '65 km cap and ring redistribution retained; no stress solver'],
            limits=['unvalidated inherited structure, strength and activation',
                    'ledger partitions current design windows on plate 101 with B6 priority, not material provenance',
                    'ledger covers surface erosion/flow/collapse only; excludes collisions and other sources',
                    'rift source and all previous world-quality issues unresolved'],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT/'config/design/final_map_v2.yaml',
                                    ROOT/'config/design/history_v2_u35_c10_progressive.yaml',
                                    ROOT/'data/worlds/design_v2/history_v2.rot']},
            reference=['docs/tectonics_method.md', 'docs/u37_b6_structured.md', 'docs/u38_transition.md'],
            diagnostic_rows=rows, step_ledger=ledger, elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text = json.dumps(report, ensure_ascii=False, indent=2)
        (cache/'run.json').write_text(text, encoding='utf8')
        if success:
            dest=ROOT/'data/worlds/design_v2/candidates'/candidate
            (dest/'run.json').write_text(text,encoding='utf8')
        print('U39_END',CASE,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)


if __name__ == '__main__':
    main()
