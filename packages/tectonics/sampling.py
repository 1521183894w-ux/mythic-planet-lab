"""直接查询移动的原生地壳单元，保留输出回退与视图填洞的出处。

这是输出取样器，不参与碰撞、提交或物理更新。原生单元间未覆盖的查询使用
既有固定节点视图回退，显式标记；不把回退值称为新生地壳或更细的物理结果。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crust import Crust, elevation

FIXED_FALLBACK = 1
VIEW_FILL = 2
VOLCANIC_ISLAND = 4


def native_cover(eng, points: np.ndarray, chunk_size: int = 65536) -> np.ndarray:
    """绝对位置单位向量 -> 存活单元索引；无覆盖为 -1。重叠沿用陆壳/厚度/年龄优先级。"""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("查询点必须是有限的 (n, 3) 单位向量")
    if not np.allclose(np.linalg.norm(points, axis=1), 1., atol=1e-8, rtol=0):
        raise ValueError("查询点必须位于单位球面")
    if chunk_size < 1:
        raise ValueError("chunk_size 必须为正")
    best = np.full(len(points), -1, np.int64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        part = points[start:stop]
        scores = np.full(len(part), -np.inf)
        chosen = best[start:stop]
        for q in eng.plates:
            rows = np.flatnonzero(eng._in_cap(q, part))
            if not rows.size:
                continue
            hit = eng.covering(q, part[rows])
            valid = hit >= 0
            rows, hit = rows[valid], hit[valid]
            s = eng.c.ctype[hit] * 1e6 + eng.c.T[hit] * 1e3 - eng.c.age[hit]
            use = s > scores[rows]
            chosen[rows[use]] = hit[use]
            scores[rows[use]] = s[use]
    return best


@dataclass
class OutputContext:
    engine: object
    raw: dict
    display: dict
    filled: np.ndarray
    parcel_trench: np.ndarray


def prepare_output(eng, fixed_frame: dict) -> OutputContext:
    """保存填洞前后视图；核对既有最终快照，拒绝把不同年代或不同过程项混在一起。"""
    raw, trench = eng.view(with_trench=True)
    displayed = raw.gather(np.arange(len(raw)))
    shown_trench = trench.copy()
    if eng.p.view_hole_max_cells is not None:
        eng._fill_view_holes(displayed, shown_trench)
    filled = np.zeros(eng.m.n, bool)
    for field in Crust.FIELDS:
        a, b = getattr(raw, field), getattr(displayed, field)
        equal = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
        filled |= ~equal
    for field in ("plate", "ctype", "origin", "T"):
        if not np.allclose(getattr(displayed, field), fixed_frame[field], rtol=0, atol=1e-8, equal_nan=True):
            raise ValueError("完整状态与固定快照不一致：" + field)
    dyn = np.asarray(fixed_frame.get("dyn", np.zeros(eng.m.n)))
    display_e = elevation(displayed, eng.p, shown_trench) + dyn
    if not np.allclose(display_e, fixed_frame["elev"], rtol=0, atol=1e-3):
        raise ValueError("固定快照的海沟/动力高程过程项不一致")
    raw_data = {field: getattr(raw, field).copy() for field in Crust.FIELDS}
    display_data = {field: getattr(displayed, field).copy() for field in Crust.FIELDS}
    raw_data["elev"] = elevation(raw, eng.p, trench) + dyn
    display_data["elev"] = np.asarray(fixed_frame["elev"]).copy()
    return OutputContext(eng, raw_data, display_data, filled, eng.trench_now())


def sample_output(context: OutputContext, points: np.ndarray, dynamic_m: np.ndarray,
                  carry_view_fill: bool = True) -> dict:
    """直接取原生单元；无覆盖时回退固定视图，填洞沿用时仅覆盖同一原始宿主。

    返回 status 位掩膜：1=固定节点回退，2=沿用旧视图填洞。原生状态不变。
    该回退是 interpolated 表示约定，不能用于证明空隙中的真实地壳成因。
    """
    eng = context.engine
    points = np.asarray(points, float)
    dyn = np.asarray(dynamic_m, float)
    if dyn.shape != (len(points),) or not np.isfinite(dyn).all():
        raise ValueError("dynamic_m 必须为每个查询点提供有限高程")
    owner = native_cover(eng, points)
    _, fixed = eng.m.tree.query(points, workers=4)
    safe = np.maximum(owner, 0)
    data = {field: getattr(eng.c, field)[safe].copy() for field in Crust.FIELDS}
    sampled = eng.c.gather(safe)
    data["elev"] = elevation(sampled, eng.p, context.parcel_trench[safe]) + dyn
    status = np.zeros(len(points), np.uint8)
    gap = owner < 0
    fallback = context.display if carry_view_fill else context.raw
    for field in data:
        data[field][gap] = fallback[field][fixed[gap]]
    status[gap] |= FIXED_FALLBACK
    if carry_view_fill:
        # 不把邻近填洞像元的陆地盖到另一个真实宿主上；同一宿主才沿用旧显示处方。
        same_host = data["uid"] == context.raw["uid"][fixed]
        fill = context.filled[fixed] & (gap | same_host)
        for field in data:
            data[field][fill] = context.display[field][fixed[fill]]
        status[fill] |= VIEW_FILL
    data.update(status=status, owner=owner, fixed_node=fixed)
    return data
