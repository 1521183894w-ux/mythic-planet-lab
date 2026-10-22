"""路线 A：Cortial 式"涌现型"运动学，地壳规则用共用引擎。

起点与路线 B 完全相同（同一个种子的初始几何和扰动后的初始运动），之后板块运动由内部规则决定，不再有剧本：
- 板片拉力：正在俯冲的板块，其角速度向"把板块质心拉向俯冲前缘"的方向松弛（时间尺度 10 Myr），
  目标速度随海沟长度增大：v = v_max · L / (L + 3000 km)，v_max 每个种子取 6–10 cm/yr；
  另保留初始运动作为背景（地幔拖曳、洋脊推力的代理）。（Cortial 2019 §4.1 的做法是按俯冲前缘修正旋转轴；这里写成松弛形式。）
- 碰撞：某对板块累计被拼贴的陆壳面积达到 30 万 km²（约 300 km 楔入 × 1000 km 接触带）时，
  下行板块上与上盘相接的整块大陆改属上盘板块（Cortial 2019 的地体转移），之后它随上盘运动。
- 裂解：陆壳面积 ≥ 1500 万 km² 的板块按泊松过程裂开（平均等待 120 Myr），位置、方向、速率随机；
  裂谷几何用与路线 B 相同的洋脊—转换断层阶梯。
"""

from __future__ import annotations

import time

import numpy as np
from common import DEG, KD_WORKERS, R_KM, chord_from_km, km_from_chord, load_mesh, omega_rotmat, unit
from engine import CONT, Engine, Params
from runlog import RunLog
from scenario import T0, initial_crust, seed_kinematics

TAU_MYR = 10.0
L0_KM = 3000.0
A_TRIGGER_KM2 = 3.0e5
RIFT_WAIT_MYR = 120.0
RIFT_MIN_CONT_KM2 = 1.5e7


def run(seed: int, level: int = 7) -> dict:
    t_all = time.perf_counter()
    mesh = load_mesh(level)
    crust = initial_crust(mesh)
    kin = seed_kinematics(seed)
    rng = np.random.default_rng(5000 + seed)
    v_max = float(rng.uniform(6.0, 10.0))          # cm/yr
    omega = {1: kin.omega1.copy(), 2: kin.omega2.copy(), 3: kin.omega3.copy()}
    bg = {q: w.copy() for q, w in omega.items()}
    eng = Engine(mesh, crust, Params(), T0, omega)
    log = RunLog(eng)
    transferred: set = set()
    next_id = 4
    cen = eng.plate_centroids()   # 提交时的板块质心；两次提交之间用 R 的增量转到当前位置
    R_at_cen = {q: eng.R[q].copy() for q in eng.plates}

    def centroids():
        cen.clear()
        cen.update(eng.plate_centroids())
        R_at_cen.clear()
        R_at_cen.update({q: eng.R[q].copy() for q in eng.plates})
    pull_log = []
    while eng.t > 0.5:
        dR = {q: omega_rotmat(omega[q], 1.0) for q in eng.plates}
        owners = log.before_step(eng)
        om_used = {q: omega[q].copy() for q in eng.plates}
        committed = eng.step(dR, om_used)

        # 板片拉力
        fr = eng._recent_fronts()
        for q in eng.plates:
            target = bg[q].copy()
            if fr is not None and q in cen:
                X, S, _, V = fr
                sel = S == q
                if sel.any():
                    cq = eng.R[q] @ R_at_cen[q].T @ cen[q]
                    tq = unit(np.cross(cq, X[sel]))
                    n = unit(tq.sum(0))
                    vbar = max(float(V[sel].mean()) * 10.0, 5.0)          # km/Myr
                    L = sel.sum() * mesh.h_km**2 / (vbar * eng.p.front_memory_myr)
                    v_sp = v_max * 10.0 * L / (L + L0_KM)                  # km/Myr
                    target = target + n * v_sp / R_KM
                    pull_log.append({"t": eng.t, "plate": int(q), "trench_km": float(L), "v_pull_cm_yr": v_sp / 10.0})
            omega[q] = omega[q] + (target - omega[q]) * (1.0 / TAU_MYR)

        if committed:
            centroids()
            # 碰撞 -> 地体转移
            for (down, over), a in list(eng.coll_area.items()):
                if a >= A_TRIGGER_KM2 and (down, over) not in transferred and down in eng.plates and over in eng.plates:
                    eng.transfer_terrane(down, over)
                    transferred.add((down, over))
            # 随机裂解
            for q in list(eng.plates):
                if eng.cont_area.get(q, 0.0) < RIFT_MIN_CONT_KM2:
                    continue
                if rng.random() >= 1.0 - np.exp(-eng.p.commit_every * eng.p.dt / RIFT_WAIT_MYR):
                    continue
                pos = eng.positions()
                cont = np.flatnonzero(eng.alive & (eng.c.plate == q) & (eng.c.ctype == CONT))
                s1 = pos[rng.choice(cont)]
                far = cont[km_from_chord(np.linalg.norm(pos[cont] - s1, axis=1)) > 1500.0]
                if far.size == 0:
                    continue
                s2 = pos[rng.choice(far)]
                axis0 = unit(np.cross(s1, s2))
                tilt = unit(np.cross(axis0, unit(s1 + s2)))
                ob = np.radians(rng.uniform(20, 50)) * rng.choice([-1, 1])
                axis = unit(np.cos(ob) * axis0 + np.sin(ob) * tilt)
                rate = float(rng.uniform(0.2, 0.35))
                new = eng.split_plate(q, s1, s2, axis, next_id, rng, omega_new=omega[q] + axis * rate * DEG)
                if new is not None:
                    omega[next_id] = omega[q] + axis * rate * DEG
                    bg[next_id] = bg[q] + axis * rate * DEG
                    next_id += 1
                    centroids()
        log.after_step(eng, dR, om_used, owners, committed)

    return {"route": "A", "seed": seed, "level": level, "mesh_n": mesh.n, "h_km": mesh.h_km, "v_max_cm_yr": v_max,
            "kin_jitter": kin.jitter, "transfers": sorted(transferred), "pull_log": pull_log[::10],
            "engine": eng, "log": log, "wall_s": time.perf_counter() - t_all}
