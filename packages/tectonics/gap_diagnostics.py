"""只读诊断实际 commit 的裂谷候选和独立的陆块边缘切线拟合。

method_class=interpolated（边缘法向）；绝不将拟合当成已恢复的断层几何。
所有拟合尺度为物理 km，不依赖固定近邻个数。
"""
import numpy as np
from scipy.spatial import cKDTree
from .crust import CONT, ORIGIN_RIFT
from .engine import gap_profile
from .geometry import R_KM, chord_from_km, km_from_chord, unit
from .deformation_kinematics import boundary_jump


def rift_candidates(eng):
    """重现现有窄缝候选及去重，只读，不新增物质或改动预算。"""
    m, c, p = eng.m, eng.c, eng.p
    pos = eng.positions()
    gap = np.flatnonzero(eng.cover_fixed(pos) < 0)
    keep = np.flatnonzero(eng.alive)
    if not len(gap) or not len(keep) or not eng.cont_gap_pairs:
        return {}
    d, k = cKDTree(pos[keep]).query(m.xyz[gap], k=min(12, len(keep)), workers=2)
    pl = c.plate[keep[k]]
    owner = pl[:, 0].copy()
    other = pl != owner[:, None]
    has, first = other.any(1), np.argmax(other, axis=1)
    ii = np.arange(len(gap))
    i1, i2 = keep[k[:, 0]], keep[k[ii, first]]
    p2 = pl[ii, first]
    pair = np.minimum(pl[:, 0], p2).astype(np.int64)*1000 + np.maximum(pl[:, 0], p2)
    omega = np.zeros((max(eng.R)+1, 3))
    for q, w in eng.omega.items():
        omega[q] = w
    proxy = np.sum(np.cross(omega[pl[:, 0]]-omega[p2], m.xyz[gap])*(pos[i1]-pos[i2]), axis=1)
    apart = proxy > 0
    for key, q in eng.gap_prefer.items():
        use = has & (pair == key) & (apart if p.gap_prefer_opening_only else True)
        owner[use] = q
    if eng.gap_prefer and p.gap_prefer_opening_only:
        conv = np.flatnonzero(has & ~apart)
        for key in np.unique(pair[conv]):
            sub = eng.polarity.get(int(key))
            if sub is None or (eng.hole_down_pairs is not None and int(key) not in eng.hole_down_pairs):
                continue
            rr = conv[pair[conv] == key]
            owner[rr[(c.ctype[i1[rr]] != CONT) & (c.ctype[i2[rr]] != CONT)]] = sub
    both = has & (c.ctype[i1] == CONT) & (c.ctype[i2] == CONT)
    selected, width = np.zeros(len(gap), bool), np.full(len(gap), np.nan)
    for key, spec in eng.cont_gap_pairs.items():
        if eng.t > eng.cont_gap_since.get(key, np.inf) + 1e-9:
            continue
        rows = np.flatnonzero(both & apart & (pair == key))
        if not len(rows):
            continue
        kw, _, _ = gap_profile(spec)
        width[rows] = 0.
        for q in (int(key)//1000, int(key)%1000):
            orig = np.flatnonzero(eng.alive & (c.plate == q) & (c.ctype == CONT) & (c.origin != ORIGIN_RIFT))
            if not len(orig):
                width[rows] = np.inf
                continue
            dq, _ = cKDTree(pos[orig]).query(m.xyz[gap[rows]], workers=2)
            width[rows] += km_from_chord(dq)
        selected[rows] = width[rows] <= kw[-1]
    # 对所有空隙先去重，保持与 Engine.commit 顺序相同。
    bn = np.empty(len(gap), np.int64)
    for q in np.unique(owner):
        rows = owner == q
        bn[rows] = m.tree.query(m.xyz[gap[rows]] @ eng.R[int(q)], workers=2)[1]
    _, uniq = np.unique(owner.astype(np.int64)*m.n + bn, return_index=True)
    rows = uniq[selected[uniq]]
    return dict(gap=gap[rows], xyz=m.xyz[gap[rows]], pair=pair[rows],
                i1=i1[rows], i2=i2[rows], neighbor=keep[k[rows]],
                distance_km=km_from_chord(d[rows]), width_proxy_km=width[rows],
                target_area_km2=m.area_km2[bn[rows]], target_node=bn[rows], target_owner=owner[rows], raw_gap_count=len(gap))


def margin_normal_diagnostics(eng, rows, scales=(180., 360., 720.)):
    """两侧独立边缘的 PCA 切线拟合；输出的是诊断支持度，不是物理真值。

    各侧至少6个边缘节点、协方差小/大特征值<0.25、两侧法向相差<45°。
    保守诊断余量 |dv|*(h/L+sin(两侧法向夹角/2))，不是统计置信区间。
    缺任一尺度、拐角或正负不一致均保留未定，不据此给地壳减薄。
    """
    if not rows or not len(rows["gap"]):
        return {}
    m, c = eng.m, eng.c
    pos = eng.positions()
    margin = eng.alive & (c.ctype == CONT) & (eng._alive_nb() < eng.full_deg)
    trees = {}
    for q in np.unique(c.plate[np.r_[rows["i1"], rows["i2"]]]):
        ids = np.flatnonzero(margin & (c.plate == q))
        if len(ids):
            trees[int(q)] = (ids, cKDTree(pos[ids]))
    nrow, ns = len(rows["gap"]), len(scales)
    vn, vt, uncertainty, angle = [np.full((nrow, ns), np.nan) for _ in range(4)]
    for j, (x, ia, ib) in enumerate(zip(rows["xyz"], rows["i1"], rows["i2"])):
        sep = pos[ib]-pos[ia]
        sep -= sep.dot(x)*x
        if np.linalg.norm(sep) < 1e-12:
            continue
        ex = unit(sep)
        ey = np.cross(x, ex)
        basis = np.stack([ex, ey], axis=1)
        sides = [int(c.plate[ia]), int(c.plate[ib])]
        for si, scale in enumerate(scales):
            normals = []
            for plate in sides:
                if plate not in trees:
                    break
                ids, tree = trees[plate]
                _, near = tree.query(x)
                local = tree.query_ball_point(pos[ids[near]], chord_from_km(scale))
                if len(local) < 6:
                    break
                xy = (pos[ids[local]]-x) @ basis * R_KM
                weights = np.exp(-np.sum((xy-xy.mean(0))**2, axis=1)/(scale*.7)**2)
                center = np.average(xy, axis=0, weights=weights)
                delta = xy-center
                cov = (delta*weights[:, None]).T @ delta / weights.sum()
                eig, vec = np.linalg.eigh(cov)
                if eig[-1] < 1e-12 or eig[0]/eig[-1] >= .25:
                    break
                n = basis @ vec[:, 0]
                if n.dot(ex) < 0:
                    n = -n
                normals.append(n)
            if len(normals) != 2:
                continue
            a = np.arccos(np.clip(normals[0].dot(normals[1]), -1, 1))
            angle[j, si] = np.degrees(a)
            if a > np.pi/4:
                continue
            n = unit(normals[0]+normals[1])
            vn[j, si], vt[j, si] = boundary_jump(x, n, eng.omega[sides[0]], eng.omega[sides[1]])
            speed = np.hypot(vn[j, si], vt[j, si])
            uncertainty[j, si] = speed*(m.h_km/scale + np.sin(a/2))
    complete = np.isfinite(vn).all(axis=1)
    status = np.full(nrow, "unresolved", dtype="<U32")
    status[complete & np.all(vn > uncertainty, axis=1)] = "opening_supported"
    status[complete & np.all(vn < -uncertainty, axis=1)] = "closing_supported"
    status[complete & np.all(np.abs(vn) <= uncertainty, axis=1)] = "shear_unresolved"
    return dict(scales_km=np.asarray(scales), normal_speed_km_myr=vn,
                slip_speed_km_myr=vt, diagnostic_margin_km_myr=uncertainty,
                normal_disagreement_deg=angle, status=status)
