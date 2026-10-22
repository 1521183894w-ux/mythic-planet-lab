"""路线 B：情景剧本 -> 运动学规划 -> PLATES4 旋转文件 -> pyGPlates 管运动学，地壳规则用共用引擎。

流程：
1. 规划（plan_kinematics）：只做刚体旋转，不跑地壳规则（每个候选不到 1 秒）。在板块 1 的参照系里，
   让南大陆绕候选欧拉极旋转，找到两块大陆第一次接触的角度；接触后再推进 300 km 即"锁定"（碰撞使汇聚减速到 15%），
   一直推进到 0 Ma；要求接触发生在洋盆一端、另一端留下窄海峡、中段仍然开着。从种子给出的欧拉极出发，
   按与它的距离由近到远试候选极，第一个满足条件、并且用阶段 0 的 A 层主内海代码核对通过的就采用（"最小修正"）。
2. 写旋转文件（build_script / write_rot）：每块板块每 1 Myr 一条总旋转记录，层级为 1、3 相对 000，2、4 相对 1。
   文件可以用 GPlates 桌面软件打开。
3. 运行：每一步由 pyGPlates 给出各板块 1 Myr 的阶段旋转，交给共用引擎；剧本里的裂解在指定时刻执行。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from common import (DEG, KD_WORKERS, R_KM, ROT_DIR, Raster, axis_angle, chord_from_km, km_from_chord, load_mesh,
                    lonlat_to_xyz, omega_from_rotmat, omega_rotmat, rotmat, unit, xyz_to_lonlat)
from engine import CONT, Engine, Params
from evaluate import basin_check
from runlog import RunLog
from scenario import T0, Kinematics, initial_crust, land_mask_initial, seed_kinematics
from scipy.spatial import cKDTree

D_PEN_KM = 300.0      # 接触后再汇聚 300 km 锁定（Cortial 2019 的地体转移阈值也是 300 km）
RESIDUAL = 0.15       # 锁定后保留 15% 的汇聚速率（碰撞后的持续楔入，类似印度—欧亚）
WEST_LON, MID_LON = (-40.0, 0.0), (3.0, 25.0)   # 板块 1 参照系中的分区（按初始几何）


@dataclass
class Plan:
    pole21: tuple
    rate21: float          # °/Myr
    t_contact: float       # Ma
    t_lock: float
    residual: float
    contact_end: str
    contact_lat_lon: tuple
    strait_node_km: float
    middle_min_km: float
    n_poles_tried: int
    n_verified: int
    ok: bool
    note: str
    wall_s: float


def _rot_pts(X, axis, ang):
    return X @ rotmat(axis, ang).T


def plan_kinematics(mesh, crust, kin: Kinematics, t_contact_target: float, raster: Raster) -> Plan:
    t_start = time.perf_counter()
    c = crust
    c1 = (c.plate == 1) & (c.ctype == CONT)
    c2c = (c.plate == 2) & (c.ctype == CONT)
    c2l = c2c & land_mask_initial(c)
    tree1 = cKDTree(mesh.xyz[c1])
    X2c, X2l = mesh.xyz[c2c], mesh.xyz[c2l]
    h = mesh.h_km
    base = kin.pole21
    cands = [(base[0] + a, base[1] + b) for a in range(-25, 26, 5) for b in range(-25, 26, 5)]
    bx = lonlat_to_xyz(base[1], base[0])
    cands.sort(key=lambda p: -float(lonlat_to_xyz(p[1], p[0]) @ bx))
    coarse = np.radians(np.arange(0.5, 45.0, 0.5))
    tried = verified = 0
    best = None
    for lat_p, lon_p in cands:
        if abs(lat_p) > 89:
            continue
        tried += 1
        P = lonlat_to_xyz(lon_p, lat_p)
        a_c = None
        for a in coarse:
            if np.isfinite(tree1.query(_rot_pts(X2c, P, a), distance_upper_bound=chord_from_km(0.6 * h),
                                       workers=KD_WORKERS)[0]).any():
                lo, hi = a - np.radians(0.5), a
                for _ in range(8):   # 二分到 0.002°
                    mid = 0.5 * (lo + hi)
                    if np.isfinite(tree1.query(_rot_pts(X2c, P, mid), distance_upper_bound=chord_from_km(0.6 * h),
                                               workers=KD_WORKERS)[0]).any():
                        hi = mid
                    else:
                        lo = mid
                a_c = hi
                break
        if a_c is None:
            continue
        Y = _rot_pts(X2c, P, a_c)
        d, _ = tree1.query(Y, workers=KD_WORKERS)
        xc = unit(Y[km_from_chord(d) < 0.6 * h].mean(0))
        lonc, latc = (float(v) for v in xyz_to_lonlat(xc))
        end = "west" if lonc < MID_LON[0] else ("east" if lonc > MID_LON[1] else "middle")
        if end == "middle":
            continue
        s_dist = np.sin(np.arccos(np.clip(P @ xc, -1, 1)))
        a_lock = a_c + D_PEN_KM / (R_KM * max(s_dist, 0.05))
        omega = a_c / (T0 - t_contact_target)                 # rad/Myr
        t_lock = t_contact_target - (a_lock - a_c) / omega
        if t_lock <= 1.0:
            continue
        a_final = a_lock + RESIDUAL * omega * t_lock
        Y = _rot_pts(X2l, P, a_final)
        dl = km_from_chord(tree1.query(Y, workers=KD_WORKERS)[0])
        lonY, _ = xyz_to_lonlat(Y)
        other = (lonY > MID_LON[1]) if end == "west" else (lonY < MID_LON[0])
        mid = (lonY > MID_LON[0]) & (lonY < MID_LON[1])
        if not other.any() or not mid.any():
            continue
        strait, middle = float(dl[other].min()), float(dl[mid].min())
        plan = Plan((float(lat_p), float(lon_p)), float(np.degrees(omega)), float(t_contact_target), float(t_lock),
                    RESIDUAL, end, (latc, lonc), strait, middle, tried, verified, False, "", 0.0)
        score = abs(strait - (h + 80.0)) + max(0.0, 600.0 - middle)
        if best is None or score < best[0]:
            best = (score, plan)
        # 海峡水面宽度 ≈ 节点距离 − 网格间距：要求 30–140 km；中段节点距离 ≥ 600 km
        if not (h + 30.0 <= strait <= h + 140.0 and middle >= 600.0):
            continue
        verified += 1
        rm = rotmat(P, a_final)
        g = raster.grid
        _, nn = mesh.tree.query(g.xyz.reshape(-1, 3) @ rm, workers=KD_WORKERS)
        in2 = c2l[nn].reshape(g.shape)
        in1 = c1[raster.idx]
        land = in1 | in2
        origin = np.where(in1, 1, np.where(in2, 2, 0))
        res = basin_check(g, land, origin)
        if res["official"]["hit"]:
            plan.ok, plan.n_verified, plan.note = True, verified, "规划通过（运动学代理 + A 层主内海核对）"
            plan.wall_s = time.perf_counter() - t_start
            return plan
        if verified >= 8:
            break
    plan = best[1]
    plan.n_poles_tried, plan.n_verified = tried, verified
    plan.note = "没有候选同时满足海峡与主内海核对，采用海峡宽度最接近目标的候选"
    plan.wall_s = time.perf_counter() - t_start
    return plan


# ---- 剧本与旋转文件 -----------------------------------------------------------------------
def build_stages(kin: Kinematics, plan: Plan, rift_t: float) -> dict:
    """每块板块每 1 Myr 的前进阶段旋转（绝对参照系，x(t-1) = S @ x(t)）。
    板块 2、4 的相对运动在板块 1 的初始参照系里给出：S = R1(t-1) · E · R1(t)^T。"""
    E1, E3 = omega_rotmat(kin.omega1, 1.0), omega_rotmat(kin.omega3, 1.0)
    R1 = {int(T0): np.eye(3)}
    for t in range(int(T0), 0, -1):
        R1[t - 1] = E1 @ R1[t]
    P21 = lonlat_to_xyz(plan.pole21[1], plan.pole21[0])
    st = {1: {}, 2: {}, 3: {}, 4: {}}
    for t in range(int(T0), 0, -1):
        st[1][t], st[3][t] = E1, E3
        rate = plan.rate21 if (t - 0.5) > plan.t_lock else plan.rate21 * plan.residual
        st[2][t] = R1[t - 1] @ rotmat(P21, rate * DEG) @ R1[t].T
        if t <= rift_t:
            st[4][t] = R1[t - 1] @ rotmat(kin.rift["axis"], kin.rift["rate"] * DEG) @ R1[t].T
        else:
            st[4][t] = E1
    return st


PARENTS = {1: 0, 3: 0, 2: 1, 4: 1}
NAMES = {1: "North continent C1", 2: "South continent C2 + Tethys", 3: "Far ocean", 4: "C1 NW rift block"}


def totals(stages: dict) -> dict:
    """总旋转（现今 -> t 时刻）：Rot(0) = I，Rot(t) = S(t)^T · Rot(t-1)。"""
    out = {}
    for q, st in stages.items():
        rot = {0: np.eye(3)}
        for t in range(1, int(T0) + 1):
            rot[t] = st[t].T @ rot[t - 1]
        out[q] = rot
    return out


def write_rot(stages: dict, path) -> None:
    tot = totals(stages)
    lines = []
    for q in sorted(stages):
        par = PARENTS[q]
        for t in range(0, int(T0) + 1):
            rm = tot[q][t] if par == 0 else tot[par][t].T @ tot[q][t]
            ax, ang = axis_angle(rm)
            lon, lat = xyz_to_lonlat(ax)
            if ang < 1e-12:
                lat, lon = 90.0, 0.0
            lines.append(f"{q:4d} {t:7.1f} {float(lat):11.6f} {float(lon):12.6f} {np.degrees(ang):11.6f} {par:4d} !{NAMES[q]}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def fr_to_mat(fr) -> np.ndarray:
    if fr.represents_identity_rotation():
        return np.eye(3)
    lat, lon, ang = fr.get_lat_lon_euler_pole_and_angle_degrees()
    return rotmat(lonlat_to_xyz(lon, lat), np.radians(ang))


# ---- 运行 ---------------------------------------------------------------------------------
def run(seed: int, level: int = 7) -> dict:
    import pygplates

    t_all = time.perf_counter()
    mesh = load_mesh(level)
    raster = Raster(mesh, 512)
    crust = initial_crust(mesh)
    kin = seed_kinematics(seed)
    rng_t = np.random.default_rng(3000 + seed)
    t_contact_target = float(rng_t.uniform(40.0, 55.0))
    plan = plan_kinematics(mesh, crust, kin, t_contact_target, raster)
    rift_t = float(5 * round(kin.rift["t"] / 5))
    stages = build_stages(kin, plan, rift_t)
    rot_path = ROT_DIR / f"route_b_seed{seed}.rot"
    write_rot(stages, rot_path)
    model = pygplates.RotationModel(str(rot_path))

    # 旋转文件往返核对：pyGPlates 给出的 1 Myr 阶段旋转与剧本一致
    rt_err = 0.0
    for t in (100, 73, rift_t, 31, 1):
        t = int(t)
        for q in (1, 2, 3, 4):
            m = fr_to_mat(model.get_rotation(float(t - 1), q, float(t), anchor_plate_id=0))
            rt_err = max(rt_err, float(np.abs(m - stages[q][t]).max()))

    eng = Engine(mesh, crust, Params(), T0, {q: omega_from_rotmat(stages[q][int(T0)], 1.0) for q in (1, 2, 3)})
    log = RunLog(eng)
    rift_rng = np.random.default_rng(kin.rift["rng_seed"])
    t = int(T0)
    while t > 0:
        dR = {q: fr_to_mat(model.get_rotation(float(t - 1), q, float(t), anchor_plate_id=0)) for q in eng.plates}
        om = {q: omega_from_rotmat(dR[q], 1.0) for q in eng.plates}
        owners = log.before_step(eng)
        committed = eng.step(dR, om)
        t -= 1
        if committed and t == int(rift_t) and 1 in eng.plates:
            R1 = fr_to_mat(model.get_rotation(float(t), 1, float(T0), anchor_plate_id=0))   # 板块 1 从 100 Ma 到现在的总旋转
            s1, s2, ax = R1 @ kin.rift["s1"], R1 @ kin.rift["s2"], R1 @ kin.rift["axis"]
            om4 = omega_from_rotmat(fr_to_mat(model.get_rotation(float(t - 1), 4, float(t), anchor_plate_id=0)), 1.0)
            eng.split_plate(1, s1, s2, ax, 4, rift_rng, omega_new=om4)
        log.after_step(eng, dR, om, owners, committed)

    # 示踪点与 pyGPlates 重建的一致性：示踪点 0 Ma 位置用 pyGPlates 转回 100 Ma，应回到出发点
    tracer_check = {}
    for name, hist in log.tracer_hist.items():
        q = hist[-1][3]
        x0 = lonlat_to_xyz(*np.array(hist[0][1:3]))
        x1 = lonlat_to_xyz(*np.array(hist[-1][1:3]))
        back = fr_to_mat(model.get_rotation(float(T0), q, 0.0, anchor_plate_id=0)) @ x1
        tracer_check[name] = {"plate_at_0Ma": q, "error_km": float(km_from_chord(np.linalg.norm(back - x0)))}
    return {"route": "B", "seed": seed, "level": level, "mesh_n": mesh.n, "h_km": mesh.h_km,
            "plan": asdict(plan), "rift_t": rift_t, "kin_jitter": kin.jitter, "rift": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kin.rift.items()},
            "rot_file": str(rot_path), "rot_roundtrip_max_err": rt_err, "tracer_vs_pygplates": tracer_check,
            "engine": eng, "log": log, "wall_s": time.perf_counter() - t_all}
