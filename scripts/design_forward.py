"""U25 正向演化 I：引擎按倒推的旋转文件从 300 Ma 跑到 0 Ma，与终图对照（PROGRESS U22、docs/tectonics_method.md §4）。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\design_forward.py --design v2 --history v2

输入：U24 的初始世界 .cache/tectonics/design/initial_<design>_<history>_L<level>.pkl（先跑 scripts/design_initial_world.py）、
旋转文件 data/worlds/design_<design>/history_<history>.rot（scripts/design_history.py）、终图 config/design/final_map_<design>.yaml。
输出：
- .cache/tectonics/design/forward_<design>_<history>_L<level>.npz：每 10 Myr 一帧（不进 Git，可重算）；
- data/worlds/design_<design>/forward_<history>.json：0 Ma 与终图的对照、A 层、山系与高原、引擎事件与收支（进 Git）；
- figures/design/<design>/forward/：01 终图与 0 Ma 并排、02 海陆差异与偏差编号、03 古地理序列（引擎与刚体块轮廓）、
  04 山系与高原对照、05 0 Ma 地壳（厚度、最近一次造山的类型与时间）。
method_class：地壳 procedural_constrained（运动由倒推设定，地壳过程是引擎的经验规则）；高程是构造尺度的
（Airy 均衡 + GDH1 洋深，parameterized_physics），不含河流与冰川侵蚀。
--figures-only：不重跑模拟，读缓存出图与对照。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from design_final_map import apply_options, build, elevation as design_elevation, label_points, line_distance_km, \
    lonlat_mask  # noqa: E402
from design_history import PANEL_TIMES, assign_blocks, build_history, raster_at  # noqa: E402
from mythic.figures.world import code_version  # noqa: E402
from mythic.metrics.grid import LatLonGrid  # noqa: E402
from mythic.tectonics.crust import CONT, ORO_ARC, ORO_COLL, ORO_RIFT, Params  # noqa: E402
from mythic.tectonics.engine import Engine  # noqa: E402
from mythic.tectonics.geometry import R_KM, km_from_chord, unit, xyz_to_lonlat  # noqa: E402
from mythic.tectonics.kinematics import RotationFileKinematics  # noqa: E402
from mythic.tectonics.mesh import CACHE, Raster, load_mesh  # noqa: E402
from mythic.tectonics.prehistory import InitialWorld  # noqa: E402,F401  （反序列化要用）

FIELDS = {"elev": np.float32, "T": np.float32, "age": np.float32, "t_form": np.float32, "oro_t": np.float32,
          "plate": np.int16, "ctype": np.int8, "origin": np.int16, "oro": np.int8, "prov": np.int32}
MIN_REGION_KM2 = 50_000.0          # 偏差清单只列面积不小于约 220 km × 220 km 的差异区


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------
def preset_polarity(hcfg: dict) -> dict:
    """按设计预设俯冲方向：{板块对键: 下行板块}。键与 Engine.polarity 相同（小编号 × 1000 + 大编号）。
    history yaml 的 ocean.subduction 每条 down × over；先写的优先；外大洋板块与任何大陆相遇都是它下行。
    （第一次运行按首次接触时的洋壳年龄决定，269 Ma 洲 B 被当成下行板块、陆壳被拼贴到洲 A 上。）"""
    oc = hcfg.get("ocean") or {}
    code_id = {b["code"]: int(b["id"]) for b in hcfg["blocks"]}
    out = {}
    for r in oc.get("subduction", []):
        for d in r["down"]:
            for o in r["over"]:
                a, b = code_id[d], code_id[o]
                out.setdefault(min(a, b) * 1000 + max(a, b), a)
    oceans = ([int(oc["background"]["id"])] if oc.get("background") else []) + [int(p["id"]) for p in oc.get("plates", [])]
    for bg in oceans:                       # 外大洋板块与单独运动的大洋板块（U26）遇到任何大陆都是它下行
        for q in code_id.values():
            out.setdefault(min(q, bg) * 1000 + max(q, bg), bg)
    return out


def over_down_pairs(hcfg: dict) -> dict:
    """{上盘板块: [下行板块, ...]}，来自 history yaml 的 ocean.subduction（不含外大洋板块）。"""
    code_id = {b["code"]: int(b["id"]) for b in hcfg["blocks"]}
    out: dict[int, list[int]] = {}
    for r in (hcfg.get("ocean") or {}).get("subduction", []):
        for o in r["over"]:
            lst = out.setdefault(code_id[o], [])
            lst += [code_id[d] for d in r["down"] if code_id[d] not in lst and code_id[d] != code_id[o]]
    return out


def strip_aprons(eng, pairs: dict, forearc_km: float, approach_km: float, between_cos: float = -0.5,
                 min_conv_cm_yr: float = 0.5) -> float:
    """上盘板块不留挡路的洋壳"围裙"（U26）：上盘板块离自己陆壳 forearc_km 以外、夹在它和某块设计中的下行板块的陆壳之间
    （两个方向夹角的余弦小于 between_cos）、离那块陆壳 approach_km 以内的洋壳，转给那块下行板块，
    下行陆块前进方向上的上盘陆缘成为俯冲带（海沟在弧前外缘）。
    U25 第三次运行：南方板块早期离开时张开的新洋壳按"就近"归了洲 B（洲 A 同理），积成约 600 km 宽的古老洋壳围裙，
    俯冲带在围裙外缘、围裙上长出洋内弧，南方地盾陆块与洲 C 北端先撞上弧地壳，提前碰撞被消耗。
    地球上被动陆缘外的洋壳在汇聚开始后也会转成俯冲带的下行一侧（俯冲在被动陆缘起始）。
    （U26 第一版只要求离下行板块的任何地壳 1200 km 以内，洲 B 北岸、洲 A 西岸等不在任何陆块前进路上的海岸也都变成俯冲带，
    沿海岸长出一圈 1 km 高的弧地壳，陆壳体积 300 Myr 增加 15%。）返回转移的面积（km²）。"""
    from scipy.spatial import cKDTree

    from mythic.tectonics.crust import OCEAN
    from mythic.tectonics.geometry import chord_from_km

    def tangent(Q, X):
        v = Q - X
        v = v - np.sum(v * X, axis=1, keepdims=True) * X
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    moved_km2 = 0.0
    for o, downs in pairs.items():
        if o not in eng.R:
            continue
        c, alive, pos = eng.c, eng.alive, eng.positions()
        oc = np.flatnonzero(alive & (c.plate == o) & (c.ctype == OCEAN))
        cont = np.flatnonzero(alive & (c.plate == o) & (c.ctype == CONT))
        if oc.size == 0 or cont.size == 0:
            continue
        d_coast, k_coast = cKDTree(pos[cont]).query(pos[oc], workers=-1)
        far = km_from_chord(d_coast) > forearc_km
        cand, p_o = oc[far], pos[cont[k_coast[far]]]
        if cand.size == 0:
            continue
        X = pos[cand]
        t_o = tangent(p_o, X)
        best = np.full(cand.size, np.inf)
        to = np.zeros(cand.size, np.int64)
        for d in downs:
            if d not in eng.R:
                continue
            du = np.flatnonzero(alive & (c.plate == d) & (c.ctype == CONT))
            if du.size == 0:
                continue
            dd, kd = cKDTree(pos[du]).query(X, distance_upper_bound=chord_from_km(approach_km), workers=-1)
            hit = np.isfinite(dd)
            dd = np.where(hit, km_from_chord(np.where(hit, dd, 0.0)), np.inf)
            p_d = pos[du[np.where(hit, kd, 0)]]
            between = np.sum(t_o * tangent(p_d, X), axis=1) < between_cos
            # 下行板块正在朝上盘汇聚：相对速度（cm/yr）在朝向上盘陆缘方向上的分量超过 min_conv_cm_yr。
            # 已经拼合、一起运动的板块不再接收围裙（第二版里洲 B 的围裙转给了 120 Ma 就拼上 B 的东南增生地块，
            # 围裙原地不动，南方地盾陆块撞上它的弧地壳，东方大高原长到设计位置以南）
            v_rel = np.cross(eng.omega[d] - eng.omega[o], X) * R_KM / 10.0
            conv = np.sum(v_rel * t_o, axis=1) > min_conv_cm_yr
            dd = np.where(between & conv, dd, np.inf)
            better = dd < best
            best[better], to[better] = dd[better], d
        for d in np.unique(to[np.isfinite(best)]):
            sel = cand[np.isfinite(best) & (to == d)]
            moved_km2 += float(eng.area[sel].sum())
            eng.transfer_units(int(o), int(d), sel, kind="apron_transfer")
    return moved_km2


def ocean_gap_prefer(hcfg: dict) -> dict:
    """外大洋板块与单独运动的大洋板块之间张开的新洋壳全部归运动的那块（U26）：按最近单元归属时一半归了固定的外大洋板块，
    运动的大洋板块前缘俯冲、后缘只长一半，几千万年后俯冲完消失（第一次试算两块分别在约 60、20 Ma 消失，
    之后远方大陆西岸、洲 B 东岸又只剩大陆自己的慢速俯冲）。"""
    oc = hcfg.get("ocean") or {}
    if not oc.get("background"):
        return {}
    bg = int(oc["background"]["id"])
    out = {min(bg, int(p["id"])) * 1000 + max(bg, int(p["id"])): int(p["id"]) for p in oc.get("plates", [])}
    # U27 试验（默认关）：大洋板块与大陆之间张开的新洋壳也全归大洋板块（东大洋由大洋板块主导，太平洋式）。
    # 按最近单元归属时，远方大陆 300 Myr 一路北移、西移，后缘张开的新洋壳归了它，积成占满东南大洋的围裙，南方几条热点链
    # 落在它的洋壳上。打开后南方珊瑚岛群方向对了（296° / 终图 288°），但大洋板块更大、俯冲更久，洲 B 东岸与远方大陆南段
    # 山系都被弧增厚推到引擎上限 4945 m（设计 2300、3000 m），陆壳体积 +7%：弧增厚在长期快速俯冲下饱和，留到 U28 一起标定
    if (hcfg.get("forward") or {}).get("ocean_plates_take_continent_gaps"):
        for p in oc.get("plates", []):
            for b in hcfg["blocks"]:
                a, q = int(b["id"]), int(p["id"])
                out[min(a, q) * 1000 + max(a, q)] = q
    return out


def build_hotspots(hcfg: dict, kin, vp) -> list:
    """history yaml 的 hotspots 目录 -> Hotspot 列表（U27）。已熄灭的老海山（old）：热点位置 = 所在大洋板块上该点在形成时刻的位置。"""
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.volcanism import Hotspot

    oc = hcfg.get("ocean") or {}
    ids = {p["code"]: int(p["id"]) for p in oc.get("plates", [])}
    if oc.get("background"):
        ids["OCEAN"] = int(oc["background"]["id"])
    out = []
    for h in hcfg.get("hotspots", []):
        flux, smax = float(vp.flux_km3_myr[h["flux"]]), float(vp.summit_max_m[h["flux"]])
        if "old" in h:
            o = h["old"]
            age, half = float(o["age"]), 0.5 * float(h["active_myr"])
            x = unit(kin.total_from(ids[o["plate"]], 0.0, age) @ lonlat_to_xyz(*o["at"]))
            out.append(Hotspot(h["id"], h["name"], x, age + half, max(age - half, 0.0), flux, smax))
        else:
            out.append(Hotspot(h["id"], h["name"], lonlat_to_xyz(*h["at"]), float(h["active"][0]),
                               float(h["active"][1]), flux, smax))
    return out


def simulate(world, mesh, rot: Path, snap_every: int, params: Params, polarity: dict, aprons: dict | None = None,
             gap_prefer: dict | None = None, volc=None):
    kin = RotationFileKinematics(rot)
    c = world.crust.gather(np.arange(len(world.crust)))
    c.uid = np.arange(len(c), dtype=np.int64)
    c.node = np.arange(len(c), dtype=np.int64)
    t0 = int(round(world.t))
    plates0 = sorted(int(q) for q in np.unique(c.plate))
    _, om = kin.step_rotations(plates0, t0)
    eng = Engine(mesh, c, params, float(t0), om)
    eng.polarity.update(polarity)
    eng.gap_prefer.update(gap_prefer or {})
    if aprons:
        aprons["pairs"] = over_down_pairs(aprons["hcfg"])
    moved = 0.0
    frames = [eng.snapshot()]
    vframes = [volc.state(eng)] if volc is not None else []
    t = t0
    tw = time.perf_counter()
    while t > 0:
        dR, om = kin.step_rotations(eng.plates, t)
        committed = eng.step(dR, om)
        t -= 1
        if committed and aprons:
            moved += strip_aprons(eng, aprons["pairs"], aprons["forearc_km"], aprons["approach_km"], aprons["between_cos"],
                                  aprons["min_conv_cm_yr"])
        if volc is not None:
            volc.step(eng)
        if committed and t % snap_every == 0:
            frames.append(eng.snapshot())
            if volc is not None:
                vframes.append(volc.state(eng))
            cl = eng.commit_log[-1]
            log(f"  {t:3d} Ma：陆壳 {cl['cont_area_km2'] / 1e6:6.1f}、海平面以上 {cl['land_area_km2'] / 1e6:6.1f} 百万 km²，"
                f"陆壳体积 {cl['cont_volume_km3'] / 1e9:6.3f}×10⁹ km³，单元 {cl['n_parcels']}，"
                f"板块 {len(eng.plates)}" + (f"，累计转给下行板块的洋壳围裙 {moved / 1e6:.1f} 百万 km²" if aprons else "")
                + (f"，火山 {int(volc.alive.sum())} 座" if volc is not None else "")
                + f"（{time.perf_counter() - tw:.0f} s）")
    return eng, frames, vframes


VFIELDS = ("lon", "lat", "plate", "hotspot", "t_birth", "height_km", "summit_m", "kind", "radius_km")


def save_run(path: Path, eng, frames, meta: dict, vframes=None, volc=None) -> None:
    arrs = {"times": np.array([f["t"] for f in frames], float)}
    for k, f in enumerate(frames):
        for name, dt in FIELDS.items():
            arrs[f"{name}_{k}"] = np.asarray(f[name]).astype(dt)
    if vframes:                             # 火山：各帧拼成一张表，v_frame 是帧序号
        arrs["v_frame"] = np.concatenate([np.full(len(v["lon"]), k) for k, v in enumerate(vframes)])
        for name in VFIELDS:
            col = np.concatenate([np.asarray(v[name]) for v in vframes])
            arrs[f"v_{name}"] = col.astype(str) if name == "hotspot" else col
    np.savez_compressed(path.with_suffix(".npz"), **arrs)
    js = {**meta, "events": eng.events, "budget": eng.budget,
          "commit_log": [{k: v for k, v in cl.items() if k != "plate_area_km2"} for cl in eng.commit_log],
          "step_wall_s_total": round(sum(s["wall_s"] for s in eng.step_log), 1)}
    if volc is not None:
        js["volcano_events"] = volc.events
        js["hotspot_on_continent"] = volc.continental
    path.with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, default=float), encoding="utf-8")


def load_run(path: Path):
    z = np.load(path.with_suffix(".npz"))
    frames = [{name: z[f"{name}_{k}"] for name in FIELDS} | {"t": float(t)} for k, t in enumerate(z["times"])]
    vframes = []
    if "v_frame" in z.files:
        for k in range(len(frames)):
            s = z["v_frame"] == k
            vframes.append({name: z[f"v_{name}"][s] for name in VFIELDS})
    js = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return frames, js, vframes


# ---------------------------------------------------------------------------
# 对照
# ---------------------------------------------------------------------------
def label_wrap(mask: np.ndarray):
    """8 连通标记，经度方向首尾相接（180° 经线两侧算同一块）。"""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ny = mask.shape[0]
    for i in range(ny):
        for di in (-1, 0, 1):
            k = i + di
            if 0 <= k < ny:
                a, b = lab[i, 0], lab[k, -1]
                if a and b:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(a) for a in range(n + 1)])
    _, new = np.unique(roots, return_inverse=True)
    return new[lab], int(new.max())


def coast_mask(land: np.ndarray) -> np.ndarray:
    er = ndimage.binary_erosion(land, structure=np.ones((3, 3), bool), border_value=0)
    return land & ~er


def compare(design, grid, b, ids, info, E_eng, E_fin, labels):
    area = grid.cell_area_km2
    Lf, Le = b["land"], E_eng > 0
    both, miss, extra = Lf & Le, Lf & ~Le, Le & ~Lf
    out = {"final_land_Mkm2": round(float(area[Lf].sum()) / 1e6, 1),
           "engine_land_Mkm2": round(float(area[Le].sum()) / 1e6, 1),
           "both_land_Mkm2": round(float(area[both].sum()) / 1e6, 1),
           "missing_land_Mkm2": round(float(area[miss].sum()) / 1e6, 1),
           "extra_land_Mkm2": round(float(area[extra].sum()) / 1e6, 1)}
    out["land_iou"] = round(float(area[both].sum() / area[Lf | Le].sum()), 3)
    # 海岸线偏移：引擎海岸点到终图海岸的距离，反之亦然
    cf, ce = coast_mask(Lf), coast_mask(Le)
    tf, te = cKDTree(grid.xyz[cf]), cKDTree(grid.xyz[ce])
    d_ef = km_from_chord(tf.query(grid.xyz[ce])[0])
    d_fe = km_from_chord(te.query(grid.xyz[cf])[0])
    out["coast_offset_km"] = {"engine_to_final": {"median": round(float(np.median(d_ef))),
                                                  "p90": round(float(np.percentile(d_ef, 90)))},
                              "final_to_engine": {"median": round(float(np.median(d_fe))),
                                                  "p90": round(float(np.percentile(d_fe, 90)))}}
    # 各刚体块与新生地壳在 0 Ma 仍是陆地的比例
    per = {}
    for code, blk in info.items():
        m = ids == blk["id"]
        per[code] = {"name": blk["name"], "area_Mkm2": round(float(area[m].sum()) / 1e6, 2),
                     "land_fraction": round(float(area[m & Le].sum() / max(area[m].sum(), 1)), 3)}
    m = ids == -1
    per["juvenile"] = {"name": "新生地壳（岛弧、热点岛、孤岛等）", "area_Mkm2": round(float(area[m].sum()) / 1e6, 2),
                       "land_fraction": round(float(area[m & Le].sum() / max(area[m].sum(), 1)), 3)}
    out["blocks"] = per
    # 偏差区：差异的连通块（≥ MIN_REGION_KM2），按面积排序，用最近的终图要素命名
    lab_pts = [(k, f, n, lo, la) for k, f, n, lo, la in labels]
    lp_xyz = unit(np.stack([np.cos(np.radians([p[4] for p in lab_pts])) * np.cos(np.radians([p[3] for p in lab_pts])),
                            np.cos(np.radians([p[4] for p in lab_pts])) * np.sin(np.radians([p[3] for p in lab_pts])),
                            np.sin(np.radians([p[4] for p in lab_pts]))], axis=1))
    lp_tree = cKDTree(lp_xyz)
    regions = []
    for kind, msk in (("missing", miss), ("extra", extra)):
        lab, n = label_wrap(msk)
        if n == 0:
            continue
        a = ndimage.sum(area, lab, index=np.arange(1, n + 1))
        for k in np.flatnonzero(a >= MIN_REGION_KM2):
            sel = lab == k + 1
            v = unit((grid.xyz[sel] * area[sel][:, None]).sum(0))
            lo, la = xyz_to_lonlat(v)
            near = lp_tree.query(v, k=2)[1]
            blk_ids, cnt = np.unique(ids[sel], return_counts=True)
            regions.append({"kind": kind, "area_Mkm2": round(float(a[k]) / 1e6, 2), "lon": round(float(lo), 1),
                            "lat": round(float(la), 1),
                            "near_features": [lab_pts[j][2] for j in np.atleast_1d(near)],
                            "final_block_cells": {str(int(x)): int(c) for x, c in zip(blk_ids, cnt)},
                            "engine_elev_m_median": round(float(np.median(E_eng[sel]))),
                            "_mask": sel})
    regions.sort(key=lambda r: -r["area_Mkm2"])
    for i, r in enumerate(regions):
        r["no"] = i + 1
    out["regions"] = regions
    # 山系：沿山脊线 width_km/2 以内，引擎与终图示意高程的 90 分位
    mts = []
    for m in design.get("mountains", []):
        hw = 0.5 * float(m.get("width_km", 150))
        d = line_distance_km(grid, m["line"], np.ones(grid.shape, bool), hw)
        sel = np.isfinite(d) & Lf
        if not sel.any():
            continue
        mts.append({"id": m["id"], "name": m["name"], "design_height_m": int(m.get("height_m", 0)),
                    "final_p90_m": round(float(np.percentile(E_fin[sel], 90))),
                    "engine_p90_m": round(float(np.percentile(E_eng[sel], 90))),
                    "engine_land_fraction": round(float(Le[sel].mean()), 2), "cause": m.get("cause", "")})
    out["mountains"] = mts
    pls = []
    for kind, key in (("plateau", "plateaus"), ("basin", "basins")):
        for p in design.get(key, []):
            sel = lonlat_mask(grid, p["poly"]) & Lf
            if not sel.any():
                continue
            pls.append({"kind": kind, "id": p["id"], "name": p["name"], "design_height_m": int(p["height_m"]),
                        "engine_median_m": round(float(np.median(E_eng[sel]))),
                        "final_median_m": round(float(np.median(E_fin[sel]))), "cause": p.get("cause", "")})
    out["plateaus_basins"] = pls
    return out


def bearing_deg(lon1, lat1, lon2, lat2) -> float:
    """从 (lon1, lat1) 指向 (lon2, lat2) 的方位角（0 = 北，90 = 东）。"""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def compare_hotspots(design: dict, hcfg: dict, v0: dict) -> list:
    """0 Ma 各热点的火山（岛、环礁、海山）与终图对照：岛数与大小、链的方向（热点指向 3–40 Myr 的老火山的质心）、链长。"""
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.volcanism import ATOLL, ISLAND, SEAMOUNT

    lines = {i["id"]: i for i in design.get("islands", [])}
    lone = {i["id"]: i for i in design.get("lone_islands", [])}
    out = []
    for h in hcfg.get("hotspots", []):
        hid = h["id"]
        sel = v0["hotspot"] == hid
        kind, age = v0["kind"][sel], v0["t_birth"][sel]
        lon, lat, rad = v0["lon"][sel], v0["lat"][sel], v0["radius_km"][sel]
        at = h["old"]["at"] if "old" in h else h["at"]
        x0 = lonlat_to_xyz(*at)
        X = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)), np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                      np.sin(np.radians(lat))], axis=1) if sel.any() else np.zeros((0, 3))
        dist = km_from_chord(np.linalg.norm(X - x0, axis=1)) if sel.any() else np.zeros(0)
        row = {"id": hid, "name": h["name"], "n_island": int((kind == ISLAND).sum()), "n_atoll": int((kind == ATOLL).sum()),
               "n_seamount": int((kind == SEAMOUNT).sum()),
               "max_island_radius_km": round(float(rad.max()), 1) if sel.any() else 0.0,
               "oldest_Ma": round(float(age.max()), 1) if sel.any() else None,
               "max_dist_km": round(float(dist.max())) if sel.any() else 0,
               "engine_bearing_deg": None, "design_bearing_deg": None, "design_length_km": None, "design_size_km": None}
        old = sel.any() and (age >= 3) & (age <= 40)
        if np.any(old):
            cen = unit(X[old].mean(0))
            clo, cla = xyz_to_lonlat(cen)
            row["engine_bearing_deg"] = round(bearing_deg(at[0], at[1], float(clo), float(cla)))
        key = hid if hid in lines or hid in lone else hid.rstrip("ab")
        if key in lines:
            ln = np.asarray(lines[key]["line"], float)
            row["design_bearing_deg"] = round(bearing_deg(ln[0, 0], ln[0, 1], ln[1, 0], ln[1, 1]))
            P = np.stack([lonlat_to_xyz(*q) for q in ln])
            row["design_length_km"] = round(float(km_from_chord(np.linalg.norm(np.diff(P, axis=0), axis=1)).sum()))
            if lines[key]["type"] == "spine":
                row["design_size_km"] = max(np.atleast_1d(lines[key]["width_km"]))
        elif key in lone:
            row["design_size_km"] = max(np.atleast_1d(lone[key]["size_km"]))
        if "old" in h and sel.any():
            row["nearest_to_design_km"] = round(float(dist.min()))
        out.append(row)
    return out


def a_summary(m: dict) -> dict:
    A = m["A"]
    isea = A["main_inland_sea"]
    return {"all_pass": all(v["pass"] for v in A.values()),
            "pass": {k: bool(v["pass"]) for k, v in A.items()},
            "ocean_fraction": round(A["ocean_fraction"]["value"], 3),
            "main_continent_share": round(A["main_continent"].get("share_of_land", float("nan")), 3),
            "subcontinents": A["subcontinents"].get("count"),
            "subcontinent_areas_Mkm2": [round(x / 1e6, 1) for x in A["subcontinents"].get("areas_km2", [])],
            "main_inland_sea": ({"area_Mkm2": round(isea["main"]["area_km2"] / 1e6, 2),
                                 "straits": isea["main"]["straits_to_world_ocean"],
                                 "aspect": (isea.get("geometry") or {}).get("aspect_ratio"),
                                 "checks": isea.get("checks")} if isea.get("exists") else
                                {"reason": isea.get("reason", "无")}),
            "latitude": [A["main_continent_latitude"].get("north_deg"), A["main_continent_latitude"].get("south_deg")],
            "polar_Mkm2": [round(p["area_km2"] / 1e6, 1) for p in A["polar_continent"].get("found", [])]}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="v2")
    ap.add_argument("--history", default="v2")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    tag = f"{args.design}_{args.history}_L{args.level}"
    run_path = CACHE / "design" / f"forward_{tag}"
    rot = ROOT / "data" / "worlds" / f"design_{args.design}" / f"history_{args.history}.rot"
    mesh = load_mesh(args.level)
    hcfg = yaml.safe_load((ROOT / "config" / "design" / f"history_{args.history}.yaml").read_text(encoding="utf-8"))
    fcfg = hcfg.get("forward") or {}
    params = Params(**fcfg.get("engine_params", {}))
    polarity = preset_polarity(hcfg)
    if not args.figures_only:
        pkl = CACHE / "design" / f"initial_{tag}.pkl"
        if not pkl.exists():
            raise SystemExit(f"缺少初始世界 {pkl}：先跑 scripts/design_history.py 与 scripts/design_initial_world.py")
        with open(pkl, "rb") as fh:
            world = pickle.load(fh)
        log(f"正向演化：{world.t:g} -> 0 Ma，L{mesh.level}（{mesh.n} 个节点），旋转文件 {rot.relative_to(ROOT).as_posix()}")
        tw = time.perf_counter()
        log(f"引擎参数（history yaml forward.engine_params）：{fcfg.get('engine_params', {})}；预设俯冲方向 {len(polarity)} 对")
        ap = fcfg.get("aprons")
        oc = hcfg.get("ocean") or {}
        aprons = ({"hcfg": hcfg, "forearc_km": float(oc.get("forearc_km", 120.0)),
                   "between_cos": float(oc.get("between_cos", -0.5)),
                   "approach_km": float(ap["approach_km"]), "min_conv_cm_yr": float(ap.get("min_conv_cm_yr", 0.5))}
                  if ap and ap.get("strip", True) else None)
        volc = None
        if hcfg.get("hotspots"):
            from mythic.tectonics.volcanism import VolcanoField, VolcanoParams
            vp = VolcanoParams(**fcfg.get("volcano_params", {}))
            hs = build_hotspots(hcfg, RotationFileKinematics(rot), vp)
            volc = VolcanoField(hs, vp)
            log(f"热点 {len(hs)} 个（history yaml hotspots）")
        eng, frames, vframes = simulate(world, mesh, rot, args.snap_every, params, polarity, aprons,
                                        ocean_gap_prefer(hcfg), volc)
        wall = time.perf_counter() - tw
        log(f"模拟完成，用时 {wall:.0f} s")
        save_run(run_path, eng, frames, {"design": args.design, "history": args.history, "level": mesh.level,
                                         "code_version": code_version(), "wall_s": round(wall, 1),
                                         "rotation_file": rot.relative_to(ROOT).as_posix(),
                                         "engine_params": fcfg.get("engine_params", {}),
                                         "aprons": fcfg.get("aprons"),
                                         "preset_polarity": {str(k): v for k, v in polarity.items()}},
                 vframes, volc)
    frames, run_js, vframes = load_run(run_path)

    log("对照终图（1024x512）")
    design = apply_options(yaml.safe_load((ROOT / "config" / "design" / f"final_map_{args.design}.yaml")
                                          .read_text(encoding="utf-8")), set())
    design["_ver"] = args.design
    grid = LatLonGrid(1024, 512)
    b = build(design, grid)
    ids, info = assign_blocks(hcfg, grid, b)
    E_fin = design_elevation(design, grid, b)
    raster = Raster(mesh, 512)
    E_eng = raster(frames[-1]["elev"]).astype(float)
    v0 = vframes[-1] if vframes else None
    hs_cmp = []
    if v0 is not None:
        # 火山岛：半径不小于约半个格点的画进海陆掩膜（与终图同一约定），高程至少 +100 m 以便出图着色
        from design_final_map import MIN_ISLAND_RADIUS_KM
        from mythic.tectonics.volcanism import island_mask
        isl = island_mask(grid, v0, MIN_ISLAND_RADIUS_KM)
        E_eng = np.where(isl & (E_eng <= 0), 100.0, E_eng)
        hs_cmp = compare_hotspots(design, hcfg, v0)
        for r in hs_cmp:
            log(f"  热点 {r['id']:4s} {r['name']}：岛 {r['n_island']}（最大半径 {r['max_island_radius_km']} km，终图 {r['design_size_km']}），"
                f"环礁 {r['n_atoll']}，海山 {r['n_seamount']}；链的方向 {r['engine_bearing_deg']}°（终图 {r['design_bearing_deg']}°），"
                f"最老的 {r['oldest_Ma']} Ma，离热点最远 {r['max_dist_km']} km（终图链长 {r['design_length_km']} km）")
    cmp_ = compare(design, grid, b, ids, info, E_eng, E_fin, label_points(design))

    # 体积账（U26）：写了 volume_balance 的碰撞，终图上它垫出的高原与山系范围内，0 Ma 实际有多少超额地壳
    from design_final_map import feature_mask
    from design_initial_world import excess_thickness_km
    T_r = raster(frames[-1]["T"]).astype(float)
    C_r = raster(frames[-1]["ctype"])
    have = np.where(C_r == CONT, np.clip(T_r - Params().T_cont, 0.0, None), 0.0)
    need = excess_thickness_km(E_fin)
    vbal = []
    for col in hcfg.get("collisions", []):
        if col.get("volume_balance"):
            tm = feature_mask(design, grid, b["land"], col["volume_balance"]["builds"])
            a = grid.cell_area_km2
            vbal.append({"down": col["down"], "up": col["up"], "builds": col["volume_balance"]["builds"],
                         "target_excess_1e6_km3": round(float((a[tm] * need[tm]).sum()) / 1e6, 1),
                         "engine_0Ma_excess_1e6_km3": round(float((a[tm] * have[tm]).sum()) / 1e6, 1),
                         "engine_0Ma_elev_m_median": round(float(np.median(E_eng[tm])))})
            log(f"体积账 {col['down']} <- {col['up']}：终图需要超额地壳 {vbal[-1]['target_excess_1e6_km3']}×10⁶ km³，"
                f"引擎 0 Ma 有 {vbal[-1]['engine_0Ma_excess_1e6_km3']}，该范围高程中位 {vbal[-1]['engine_0Ma_elev_m_median']} m")

    from mythic.metrics.compute import compute_all, load_config
    cfg = load_config()
    m_eng, f_eng = compute_all(E_eng, E_eng > 0, cfg=cfg, return_fields=True)
    m_fin, f_fin = compute_all(E_fin, b["land"], cfg=cfg, return_fields=True)
    A_eng, A_fin = a_summary(m_eng), a_summary(m_fin)

    c0 = frames[-1]
    A7 = mesh.area_km2
    cont = c0["ctype"] == CONT
    oro_area = {name: round(float(A7[cont & (c0["oro"] == k)].sum()) / 1e6, 2)
                for name, k in (("collision", ORO_COLL), ("arc", ORO_ARC), ("rift", ORO_RIFT))}
    ev_counts = {}
    for e in run_js["events"]:
        ev_counts[e["kind"]] = ev_counts.get(e["kind"], 0) + 1
    cl0, cl1 = run_js["commit_log"][0], run_js["commit_log"][-1]
    summary = {
        "description": "U25 正向演化：引擎按倒推的旋转文件从 300 Ma 跑到 0 Ma，与终图对照（重建：scripts/design_forward.py；"
                       "每 10 Myr 一帧的缓存在 .cache/tectonics/design/，不进 Git）",
        "design": args.design, "history": args.history, "mesh_level": args.level,
        "run_code_version": run_js["code_version"], "code_version": code_version(), "wall_s": run_js["wall_s"],
        "compare_grid": "1024x512",
        "comparison": {k: v for k, v in cmp_.items() if k != "regions"},
        "deviation_regions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in cmp_["regions"]],
        "A_engine_0Ma": A_eng, "A_final_map_1024": A_fin,
        "volume_balance_0Ma": vbal,
        "hotspots_0Ma": hs_cmp,
        "crust_0Ma": {"continental_Mkm2": round(float(A7[cont].sum()) / 1e6, 1),
                      "arc_origin_continental_Mkm2": round(float(A7[cont & (c0["origin"] == 90)].sum()) / 1e6, 2),
                      "last_orogeny_area_Mkm2": oro_area,
                      "thickness_km_p99": round(float(np.percentile(c0["T"][cont], 99)), 1),
                      "elev_m_p99_land": round(float(np.percentile(c0["elev"][c0["elev"] > 0], 99)))},
        "volume_change": round(cl1["cont_volume_km3"] / cl0["cont_volume_km3"] - 1, 4),
        "budget_km3": {k: round(v) for k, v in run_js["budget"].items()},
        "engine_event_counts": ev_counts,
        "collision_onsets": [e for e in run_js["events"] if e["kind"] in ("collision_onset", "continent_contact")],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_w = ROOT / "data" / "worlds" / f"design_{args.design}"
    from mythic.figures.world import to_jsonable
    (out_w / f"forward_{args.history}.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=1),
                                                       encoding="utf-8", newline="\n")
    c = summary["comparison"]
    log(f"终图陆地 {c['final_land_Mkm2']}，引擎 0 Ma 陆地 {c['engine_land_Mkm2']}，两者都是陆地 {c['both_land_Mkm2']}，"
        f"终图有而引擎没有 {c['missing_land_Mkm2']}，引擎多出 {c['extra_land_Mkm2']} 百万 km²；IoU {c['land_iou']}")
    log(f"海岸偏移（中位 / 90 分位）：引擎 -> 终图 {c['coast_offset_km']['engine_to_final']}，"
        f"终图 -> 引擎 {c['coast_offset_km']['final_to_engine']} km")
    for k, v in c["blocks"].items():
        log(f"  {k:9s} {v['name']}：{v['area_Mkm2']} 百万 km²，0 Ma 仍是陆地 {v['land_fraction']:.0%}")
    for r in cmp_["regions"][:25]:
        log(f"  偏差 {r['no']:2d} {'缺陆地' if r['kind'] == 'missing' else '多出陆地'} {r['area_Mkm2']:5.2f} 百万 km² "
            f"({r['lon']}, {r['lat']}) 近 {'、'.join(r['near_features'])}，引擎高程中位 {r['engine_elev_m_median']} m")
    for mt in cmp_["mountains"]:
        log(f"  山系 {mt['id']:5s} {mt['name']}：设计 {mt['design_height_m']} m，引擎 90 分位 {mt['engine_p90_m']} m"
            f"（陆地 {mt['engine_land_fraction']:.0%}）")
    for p in cmp_["plateaus_basins"]:
        log(f"  {p['kind']} {p['id']} {p['name']}：设计 {p['design_height_m']} m，引擎中位 {p['engine_median_m']} m")
    log(f"A 层（引擎 0 Ma）：{A_eng}")
    log(f"A 层（终图，1024x512）：{A_fin}")
    log(f"陆壳体积变化 {summary['volume_change'] * 100:.2f}%；引擎事件 {ev_counts}")
    if not args.no_figures:
        log("出图")
        figures(args, design, hcfg, grid, b, ids, info, mesh, raster, frames, E_eng, E_fin, cmp_, m_eng, f_eng,
                A_eng, A_fin, run_js, v0, hs_cmp)
    log(f"完成，用时 {time.perf_counter() - t_start:.0f} s")
    return 0


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def figures(args, design, hcfg, grid, b, ids, info, mesh, raster, frames, E_eng, E_fin, cmp_, m_eng, f_eng,
            A_eng, A_fin, run_js, v0=None, hs_cmp=None):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgb
    from matplotlib.patches import Patch

    from mythic.figures.style import setup_fonts, shaded_elevation

    setup_fonts()
    out = ROOT / "figures" / "design" / args.design / "forward"
    out.mkdir(parents=True, exist_ok=True)
    PC = ccrs.PlateCarree()
    proj = ccrs.EqualEarth(central_longitude=float(design.get("map_central_longitude", 30)))
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    ver = code_version()
    lon = grid.lon
    lat = grid.lat
    Lf = b["land"]

    def stamp(fig, extra=""):
        fig.text(0.01, 0.004, f"U25 正向演化 · 终图 final_map_{args.design}.yaml · 历史 history_{args.history}.yaml · "
                 f"L{args.level} · scripts/design_forward.py · 运行代码 {run_js['code_version']} · 出图代码 {ver} · "
                 f"method_class：地壳 procedural_constrained，高程为构造尺度（Airy + GDH1，不含侵蚀）{extra}",
                 fontsize=7.5, color="#555555")

    def show(ax, E, land=None):
        ax.set_global()
        ax.imshow(shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=(E > 0) if land is None else land)[..., :3],
                  extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(2000, 1000),
                  interpolation="nearest")
        ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)

    def a_line(A):
        ok = lambda k: "√" if A["pass"][k] else "×"
        s = A["main_inland_sea"]
        sea = (f"{s['area_Mkm2']} Mkm²、海峡 {s['straits']}、长宽比 {s['aspect']:.1f}" if "area_Mkm2" in s
               else s.get("reason", "无"))
        return (f"海洋 {A['ocean_fraction']:.1%} {ok('ocean_fraction')} · 主大陆占陆地 {A['main_continent_share']:.0%} "
                f"{ok('main_continent')} · 亚大陆 {A['subcontinents']} {ok('subcontinents')} · 主内海 {sea} "
                f"{ok('main_inland_sea')} · 极地大陆 {A['polar_Mkm2'] or '无'} {ok('polar_continent')} · "
                f"纬度 {ok('main_continent_latitude')} · 开阔大洋 {ok('open_ocean')}")

    # ---- 01 终图与 0 Ma 并排 ----
    fig = plt.figure(figsize=(20, 21))
    ax = fig.add_axes([0.02, 0.52, 0.96, 0.42], projection=proj)
    show(ax, E_fin, Lf)
    ax.set_title(f"(a) 终图 v2（设计；示意高程 visual_only）\nA 层（1024x512）：{a_line(A_fin)}", fontsize=11)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.42], projection=proj)
    show(ax, E_eng)
    ax.contour(lon, lat, Lf.astype(float), levels=[0.5], colors="#d62728", linewidths=0.5, transform=PC)
    ax.set_title(f"(b) 引擎正向演化 0 Ma（300 Ma 初始世界 + 倒推的板块运动；红线 = 终图海岸）\nA 层：{a_line(A_eng)}",
                 fontsize=11)
    fig.text(0.5, 0.985, "终图与正向演化的 0 Ma 对照（同一投影、同一配色；引擎高程是构造尺度，没有侵蚀、河谷与海岸细节）",
             fontsize=15, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "01_final_vs_0Ma.png", dpi=110)
    plt.close(fig)

    # ---- 02 海陆差异与偏差编号 ----
    fig = plt.figure(figsize=(20, 11.5))
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.84], projection=proj)
    ax.set_global()
    cat = np.zeros(grid.shape, int)
    cat[Lf & (E_eng > 0)] = 1
    cat[Lf & (E_eng <= 0)] = 2
    cat[~Lf & (E_eng > 0)] = 3
    cmap = ListedColormap(["#dbe9f6", "#bdbdbd", "#e6550d", "#3182bd"])
    ax.imshow(cat, extent=(-180, 180, -90, 90), transform=PC, origin="upper", cmap=cmap, vmin=-0.5, vmax=3.5,
              regrid_shape=(2000, 1000), interpolation="nearest")
    ax.contour(lon, lat, Lf.astype(float), levels=[0.5], colors="#333333", linewidths=0.35, transform=PC)
    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
    for r in cmp_["regions"][:30]:
        ax.text(r["lon"], r["lat"], str(r["no"]), transform=PC, fontsize=9, weight="bold", ha="center", va="center",
                color="#a63603" if r["kind"] == "missing" else "#08519c", path_effects=halo)
    cc = cmp_
    hs = [Patch(fc="#bdbdbd", label=f"两者都是陆地 {cc['both_land_Mkm2']} 百万 km²"),
          Patch(fc="#e6550d", label=f"终图是陆地、引擎是海 {cc['missing_land_Mkm2']}"),
          Patch(fc="#3182bd", label=f"引擎多出的陆地 {cc['extra_land_Mkm2']}")]
    ax.legend(handles=hs, loc="lower left", fontsize=10, framealpha=0.92)
    co = cc["coast_offset_km"]
    fig.text(0.5, 0.975, f"海陆差异（陆地 IoU {cc['land_iou']}；海岸偏移中位 {co['engine_to_final']['median']} km、"
             f"90 分位 {co['engine_to_final']['p90']} km）· 数字 = 偏差编号（≥ 5 万 km² 的差异区，按面积排序，见 forward_"
             f"{args.history}.json）", fontsize=13, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "02_land_difference.png", dpi=120)
    plt.close(fig)

    # ---- 03 古地理序列：引擎与刚体块轮廓 ----
    area = grid.cell_area_km2
    blocks = {}
    for code, blk in info.items():
        msk = ids == blk["id"]
        blocks[blk["id"]] = {"code": code, "color": blk["color"],
                             "centroid": unit((grid.xyz[msk] * area[msk][:, None]).sum(0))}
    h = build_history(hcfg, {k: v["centroid"] for k, v in blocks.items()})
    tgt = LatLonGrid(720, 360)
    times = [t for t in PANEL_TIMES if any(abs(f["t"] - t) < 0.5 for f in frames)]
    nrow = int(np.ceil(len(times) / 2))
    fig = plt.figure(figsize=(18, 4.6 * nrow + 0.8))
    for k, t in enumerate(times):
        f = min(frames, key=lambda fr: abs(fr["t"] - t))
        ax = fig.add_subplot(nrow, 2, k + 1, projection=proj)
        show(ax, raster(f["elev"]).astype(float))
        own, _ = raster_at(h, blocks, ids, grid, float(t), tgt)
        for pid, v in blocks.items():
            ax.contour(tgt.lon, tgt.lat, (own == pid).astype(float), levels=[0.5], colors=[v["color"]],
                       linewidths=0.8, transform=PC)
        ax.set_title(f"{t} Ma", fontsize=11)
    fig.suptitle("古地理序列：底图 = 引擎正向演化（构造尺度高程），彩色线 = U23 刚体块在同一时刻的轮廓（只倒推运动的设计位置）",
                 fontsize=13)
    stamp(fig)
    fig.savefig(out / "03_paleogeography.png", dpi=95, bbox_inches="tight")
    plt.close(fig)

    # ---- 04 山系与高原：引擎 vs 设计 ----
    mts = cmp_["mountains"]
    pls = cmp_["plateaus_basins"]
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_axes([0.01, 0.40, 0.98, 0.54], projection=proj)
    show(ax, E_eng)
    for m in design.get("mountains", []):
        p = np.asarray(m["line"], float)
        ax.plot(p[:, 0], p[:, 1], color="#7f0000", lw=1.0, alpha=0.7, transform=ccrs.Geodetic())
    for mt in mts:
        m = next(x for x in design["mountains"] if x["id"] == mt["id"])
        p = np.asarray(m["line"], float)[len(m["line"]) // 2]
        ax.text(p[0], p[1], mt["id"], transform=PC, fontsize=7.5, color="#7f0000", path_effects=halo)
    ax.set_title("引擎 0 Ma 高程，暗红线 = 终图山脊线（编号见下表）", fontsize=12)
    ax2 = fig.add_axes([0.05, 0.05, 0.55, 0.30])
    x = np.arange(len(mts))
    ax2.bar(x - 0.2, [m["design_height_m"] for m in mts], 0.4, color="#bdbdbd", label="终图设计高度")
    ax2.bar(x + 0.2, [m["engine_p90_m"] for m in mts], 0.4, color="#8c510a", label="引擎 0 Ma（山脊带 90 分位）")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m["id"] for m in mts], rotation=90, fontsize=7)
    ax2.set_ylabel("m")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(fontsize=9)
    ax2.set_title("山系：设计高度 vs 引擎（不含侵蚀的构造高程）", fontsize=11)
    ax3 = fig.add_axes([0.66, 0.05, 0.32, 0.30])
    y = np.arange(len(pls))
    ax3.barh(y + 0.2, [p["design_height_m"] for p in pls], 0.4, color="#bdbdbd", label="设计")
    ax3.barh(y - 0.2, [p["engine_median_m"] for p in pls], 0.4, color="#8c510a", label="引擎中位")
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"{p['id']} {p['name']}" for p in pls], fontsize=7.5)
    ax3.invert_yaxis()
    ax3.axvline(0, color="k", lw=0.5)
    ax3.legend(fontsize=9)
    ax3.set_title("高原与盆地", fontsize=11)
    stamp(fig)
    fig.savefig(out / "04_mountains_plateaus.png", dpi=110)
    plt.close(fig)

    # ---- 05 0 Ma 地壳 ----
    c0 = frames[-1]
    fig = plt.figure(figsize=(20, 11))
    cont = raster(c0["ctype"]) == CONT
    panels = [("a", "陆壳厚度（km）"), ("b", "最近一次构造改造的时间（Ma；碰撞 = 暖色，俯冲弧 = 冷色）"),
              ("c", "地壳来源（初始陆块 / 岛弧增生）"), ("d", "洋底年龄（Myr）")]
    for k, (tg, title) in enumerate(panels):
        r, cc_ = divmod(k, 2)
        ax = fig.add_axes([0.01 + cc_ * 0.495, 0.50 - r * 0.46, 0.48, 0.42], projection=proj)
        ax.set_global()
        if tg == "a":
            im = ax.imshow(np.ma.masked_where(~cont, raster(c0["T"])), extent=(-180, 180, -90, 90), transform=PC,
                           origin="upper", cmap="YlOrBr", vmin=20, vmax=65, regrid_shape=(1600, 800),
                           interpolation="nearest")
        elif tg == "b":
            oro = raster(c0["oro"])
            ot = raster(c0["oro_t"]).astype(float)
            ax.imshow(np.ma.masked_where(~cont, np.zeros(grid.shape)), extent=(-180, 180, -90, 90), transform=PC,
                      origin="upper", cmap=ListedColormap(["#eeeeee"]), regrid_shape=(1600, 800), interpolation="nearest")
            for code, cm in ((ORO_COLL, "autumn"), (ORO_ARC, "winter")):
                im = ax.imshow(np.ma.masked_where(~cont | (oro != code), ot), extent=(-180, 180, -90, 90), transform=PC,
                               origin="upper", cmap=cm, vmin=0, vmax=300, regrid_shape=(1600, 800),
                               interpolation="nearest")
        elif tg == "c":
            org = raster(c0["origin"]).astype(float)
            img = np.broadcast_to(np.array([0.86, 0.91, 0.96]), grid.shape + (3,)).copy()
            for code, blk in info.items():
                img[cont & (org == blk["id"])] = to_rgb(blk["color"])
            img[cont & (org == 90)] = (0.1, 0.1, 0.1)
            ax.imshow(img, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(1600, 800),
                      interpolation="nearest")
            im = None
        else:
            im = ax.imshow(np.ma.masked_where(cont, raster(c0["age"])), extent=(-180, 180, -90, 90), transform=PC,
                           origin="upper", cmap="RdYlBu", vmin=0, vmax=180, regrid_shape=(1600, 800),
                           interpolation="nearest")
        if im is not None:
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        ax.contour(lon, lat, Lf.astype(float), levels=[0.5], colors="#444444", linewidths=0.3, transform=PC)
        ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
        ax.set_title(f"({tg}) {title}", fontsize=11)
    hs = [Patch(fc=v["color"], label=f"{k} {v['name']}") for k, v in info.items()] + [Patch(fc="#1a1a1a", label="岛弧增生")]
    fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=8)
    fig.text(0.5, 0.99, "0 Ma 地壳（灰线 = 终图海岸）", fontsize=14, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "05_crust_0Ma.png", dpi=105)
    plt.close(fig)


    # ---- 06 热点：终图的热点链与孤岛（灰）vs 引擎 0 Ma 的火山（U27） ----
    if v0 is not None:
        from mythic.tectonics.volcanism import ATOLL, ISLAND, SEAMOUNT
        hs_at = {h["id"]: (h["old"]["at"] if "old" in h else h["at"]) for h in hcfg.get("hotspots", [])}
        GEO = ccrs.Geodetic()

        def draw(ax, ms_scale=1.0, labels=False):
            for isl in design.get("islands", []):
                if isl["type"] == "hotspot":
                    q = np.asarray(isl["line"], float)
                    ax.plot(q[:, 0], q[:, 1], "--", color="#7f7f7f", lw=1.0, transform=GEO, zorder=3)
            for sym in b["symbols"]:
                mk = "o" if "atoll" in sym["kind"] else "^"
                ax.plot(sym["lon"], sym["lat"], mk, ms=5 * ms_scale, mfc="none", mec="#7f7f7f", mew=0.8, transform=PC, zorder=3)
            k = v0["kind"]
            age = v0["t_birth"]
            s = k == SEAMOUNT
            ax.scatter(v0["lon"][s], v0["lat"][s], s=4 * ms_scale, c=age[s], cmap="viridis_r", vmin=0, vmax=100,
                       transform=PC, zorder=4, linewidths=0)
            s = k == ATOLL
            ax.scatter(v0["lon"][s], v0["lat"][s], s=30 * ms_scale, facecolors="none", edgecolors="#0868ac", linewidths=1.2,
                       transform=PC, zorder=5)
            s = k == ISLAND
            ax.scatter(v0["lon"][s], v0["lat"][s], s=(8 + 2.0 * v0["radius_km"][s]) * ms_scale, marker="^", c="#d7301f",
                       edgecolors="k", linewidths=0.4, transform=PC, zorder=6)
            for hid, (lo, la) in hs_at.items():
                ax.plot(lo, la, "*", ms=9 * ms_scale, mfc="#ffd92f", mec="k", mew=0.5, transform=PC, zorder=7)
                if labels:
                    ax.text(lo + 1.0, la + 1.0, hid, transform=PC, fontsize=8, path_effects=halo, zorder=8, clip_on=True)

        fig = plt.figure(figsize=(20, 19))
        ax = fig.add_axes([0.02, 0.50, 0.96, 0.44], projection=proj)
        show(ax, E_eng)
        draw(ax, 0.8)
        ax.set_title("全球：灰色虚线与空心符号 = 终图的热点链与孤岛；红三角 = 引擎 0 Ma 的火山岛（大小 ∝ 半径），"
                     "蓝圈 = 环礁，点 = 海山（颜色 = 形成时刻，深 = 老）；黄星 = 热点（地幔参照系）", fontsize=11)
        regions = [("东大洋", (-180, -105, -45, 45)), ("西大洋与北洋", (-75, -20, -60, 75)), ("东洋与南大洋", (25, 90, -60, -5))]
        for j, (nm, (x0, x1, y0, y1)) in enumerate(regions):
            ax = fig.add_axes([0.02 + j * 0.33, 0.03, 0.30, 0.38], projection=PC)
            ax.set_extent((x0, x1, y0, y1), crs=PC)
            ax.imshow(shaded_elevation(E_eng, grid.pixel_km_equator, vert_exag=3.0, land=E_eng > 0)[..., :3],
                      extent=(-180, 180, -90, 90), transform=PC, origin="upper", interpolation="nearest")
            draw(ax, 1.3, labels=True)
            ax.gridlines(draw_labels=True, color="#888888", lw=0.3)
            ax.set_title(nm, fontsize=12)
        rows = [f"{r['id']} {r['name']}：岛 {r['n_island']} / 环礁 {r['n_atoll']} / 海山 {r['n_seamount']}"
                + (f"，方向 {r['engine_bearing_deg']}°（终图 {r['design_bearing_deg']}°）" if r["design_bearing_deg"] is not None else "")
                for r in (hs_cmp or [])]
        fig.text(0.5, 0.985, "U27 热点岛链与孤岛：终图 vs 正向演化 0 Ma", fontsize=15, ha="center", va="top")
        fig.text(0.01, 0.485, "；".join(rows[:8]) + "\n" + "；".join(rows[8:16]) + "\n" + "；".join(rows[16:]),
                 fontsize=8, va="top")
        stamp(fig)
        fig.savefig(out / "06_hotspots.png", dpi=105)
        plt.close(fig)

if __name__ == "__main__":
    raise SystemExit(main())
