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
from mythic.tectonics.crust import CONT, ORO_ARC, ORO_COLL, ORO_REACT, ORO_RIFT, Params  # noqa: E402
from mythic.tectonics.engine import Engine  # noqa: E402
from mythic.tectonics.geometry import R_KM, km_from_chord, unit, xyz_to_lonlat  # noqa: E402
from mythic.tectonics.kinematics import RotationFileKinematics  # noqa: E402
from mythic.tectonics.mesh import CACHE, Raster, load_mesh  # noqa: E402
from mythic.tectonics.prehistory import InitialWorld  # noqa: E402,F401  （反序列化要用）

FIELDS = {"elev": np.float32, "T": np.float32, "age": np.float32, "t_form": np.float32, "oro_t": np.float32,
          "plate": np.int16, "ctype": np.int8, "origin": np.int16, "oro": np.int8, "prov": np.int32,
          "dyn": np.float32}                # U29：动力地形（m，已加在 elev 里）
MIN_REGION_KM2 = 50_000.0          # 偏差清单只列面积不小于约 220 km × 220 km 的差异区


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------
def plate_codes(hcfg: dict) -> dict:
    """代号 -> 板块编号：陆块、外大洋板块（OCEAN）、单独运动的大洋板块（U26）、岛弧系统的新板块（U28 第二轮）。"""
    oc = hcfg.get("ocean") or {}
    out = {b["code"]: int(b["id"]) for b in hcfg["blocks"]}
    if oc.get("background"):
        out["OCEAN"] = int(oc["background"]["id"])
    out.update({p["code"]: int(p["id"]) for p in oc.get("plates", [])})
    out.update({p["code"]: int(p["id"]) for p in (hcfg.get("arcs") or {}).get("plates", [])})
    return out


def preset_polarity(hcfg: dict) -> dict:
    """按设计预设俯冲方向：{板块对键: 下行板块}。键与 Engine.polarity 相同（小编号 × 1000 + 大编号）。
    先 history yaml 的 arcs.subduction（U28 第二轮岛弧系统），再 ocean.subduction，每条 down × over，先写的优先；
    最后外大洋板块、大洋板块与岛弧系统的新洋板块遇到任何大陆都是它下行。
    （第一次运行按首次接触时的洋壳年龄决定，269 Ma 洲 B 被当成下行板块、陆壳被拼贴到洲 A 上。）"""
    oc = hcfg.get("ocean") or {}
    code_id = plate_codes(hcfg)
    out = {}
    for r in ((hcfg.get("forward") or {}).get("polarity", []) + (hcfg.get("arcs") or {}).get("subduction", [])
              + oc.get("subduction", [])):          # U33：forward.polarity 只定方向，优先
        for d in r["down"]:
            for o in r["over"]:
                a, b = code_id[d], code_id[o]
                out.setdefault(min(a, b) * 1000 + max(a, b), a)
    oceans = (([int(oc["background"]["id"])] if oc.get("background") else []) + [int(p["id"]) for p in oc.get("plates", [])]
              + [int(p["id"]) for p in (hcfg.get("arcs") or {}).get("plates", [])])
    for bg in oceans:                       # 外大洋板块与单独运动的大洋板块（U26）遇到任何大陆都是它下行
        for q in (int(b["id"]) for b in hcfg["blocks"]):
            out.setdefault(min(q, bg) * 1000 + max(q, bg), bg)
    return out


def designed_pairs(hcfg: dict) -> set:
    """设计的俯冲带（arcs.subduction 与 ocean.subduction 的每个下行 × 上盘）的板块对键（U28 第二轮）。"""
    code_id = plate_codes(hcfg)
    out = set()
    for r in (hcfg.get("arcs") or {}).get("subduction", []) + (hcfg.get("ocean") or {}).get("subduction", []):
        for d in r["down"]:
            for o in r["over"]:
                a, b = code_id[d], code_id[o]
                out.add(min(a, b) * 1000 + max(a, b))
    return out


def over_down_pairs(hcfg: dict) -> dict:
    """{上盘板块: [下行板块, ...]}，来自 history yaml 的 ocean.subduction 与 forward.polarity 里标了 aprons 的条目（不含外大洋板块）。"""
    code_id = {b["code"]: int(b["id"]) for b in hcfg["blocks"]}
    out: dict[int, list[int]] = {}
    # U33：forward.polarity 里标了 aprons: true 的条目也参加（主内海北岸微陆块对洲 A、B，与 ocean.subduction 的规则同一做法：
    # 挡在逼近的微陆块前方的上盘洋壳转给微陆块）。第九次运行加它是为了 6–9°E 的深水窄湾（评审 stage1_2026-10-25_3 第 1 条），
    # 窄湾没变；真正的原因是内海微大陆岛 I4 记在洲 A 名下、挡在西段地体的路上（见 history yaml 的 blocks）
    rules = list((hcfg.get("ocean") or {}).get("subduction", []))
    rules += [r for r in (hcfg.get("forward") or {}).get("polarity", []) if r.get("aprons")]
    for r in rules:
        for o in r["over"]:
            lst = out.setdefault(code_id[o], [])
            lst += [code_id[d] for d in r["down"] if code_id[d] not in lst and code_id[d] != code_id[o]]
    return out


def strip_aprons(eng, pairs: dict, forearc_km: float, approach_km: float, between_cos: float = -0.5,
                 min_conv_cm_yr: float = 0.5, ocean_downs: dict | None = None, protected: np.ndarray | None = None) -> float:
    """上盘板块不留挡路的洋壳"围裙"（U26）：上盘板块离自己陆壳 forearc_km 以外、夹在它和某块设计中的下行板块的陆壳之间
    （两个方向夹角的余弦小于 between_cos）、离那块陆壳 approach_km 以内的洋壳，转给那块下行板块，
    下行陆块前进方向上的上盘陆缘成为俯冲带（海沟在弧前外缘）。
    U25 第三次运行：南方板块早期离开时张开的新洋壳按"就近"归了洲 B（洲 A 同理），积成约 600 km 宽的古老洋壳围裙，
    俯冲带在围裙外缘、围裙上长出洋内弧，南方地盾陆块与洲 C 北端先撞上弧地壳，提前碰撞被消耗。
    地球上被动陆缘外的洋壳在汇聚开始后也会转成俯冲带的下行一侧（俯冲在被动陆缘起始）。
    （U26 第一版只要求离下行板块的任何地壳 1200 km 以内，洲 B 北岸、洲 A 西岸等不在任何陆块前进路上的海岸也都变成俯冲带，
    沿海岸长出一圈 1 km 高的弧地壳，陆壳体积 300 Myr 增加 15%。）
    U28 第二轮 ocean_downs = {上盘陆块: [大洋板块, ...]}：大洋板块（外大洋、东大洋两块、岛弧系统的新洋板块）朝陆块汇聚、
    按预设是它下行时，挡在前面的围裙也转给它（"离下行板块近"按它的任何单元算）。否则大陆漂移时前方自己名下的旧洋壳与
    大洋板块相撞，海沟落在离岸几百公里外，长出终图没有的洋内弧（第二轮第一次运行：极地大陆一圈、洲 C 西岸外、洲 A 西北等，
    1200 多座弧火山岛）。protected：岛弧系统划给陆块的洋壳（琉球、千岛、班达、汤加—克马德克式的洋内弧所在）单元编号，不转。
    返回转移的面积（km²）。"""
    from scipy.spatial import cKDTree

    from mythic.tectonics.crust import OCEAN
    from mythic.tectonics.geometry import chord_from_km

    def tangent(Q, X):
        v = Q - X
        v = v - np.sum(v * X, axis=1, keepdims=True) * X
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    moved_km2 = 0.0
    ocean_downs = ocean_downs or {}
    for o in sorted(set(pairs) | set(ocean_downs)):
        downs = list(pairs.get(o, [])) + [d for d in ocean_downs.get(o, []) if d not in pairs.get(o, [])]
        if o not in eng.R:
            continue
        c, alive, pos = eng.c, eng.alive, eng.positions()
        oc = np.flatnonzero(alive & (c.plate == o) & (c.ctype == OCEAN))
        if protected is not None and protected.size and oc.size:
            oc = oc[~np.isin(c.uid[oc], protected)]
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
            du = np.flatnonzero(alive & (c.plate == d) & ((c.ctype == CONT) | (d in ocean_downs.get(o, []))))
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
    code_id = plate_codes(hcfg)
    for g in (hcfg.get("arcs") or {}).get("gaps", []):     # U28 第二轮：岛弧系统各板块之间张开的新洋壳
        a, b = (code_id[x] for x in g["pair"])
        out[min(a, b) * 1000 + max(a, b)] = code_id[g["to"]]
    return out


def compress_pairs(hcfg: dict) -> set:
    """岛弧系统里标了 compress: true 的俯冲带（挤压型岛弧，U28 第三轮）的板块对键。"""
    code_id = plate_codes(hcfg)
    out = set()
    for r in (hcfg.get("arcs") or {}).get("subduction", []):
        if r.get("compress"):
            for d in r["down"]:
                for o in r["over"]:
                    a, b = code_id[d], code_id[o]
                    out.add(min(a, b) * 1000 + max(a, b))
    return out


def apply_basement(eng, q: int, poly_xyz: np.ndarray, T_km: float) -> int:
    """新洋板块继承的老岛弧基底（U28 第三轮）：板块 q 落在多边形里的存活洋壳单元，厚度至少 T_km，记为岛弧来源。返回单元数。"""
    from mythic.tectonics.crust import OCEAN, ORIGIN_ARC
    from mythic.tectonics.reverse import in_sphere_polygon

    c = eng.c
    cand = np.flatnonzero(eng.alive & (c.plate == q) & (c.ctype == OCEAN))
    cand = cand[in_sphere_polygon(eng.positions()[cand], poly_xyz)]
    c.T[cand] = np.maximum(c.T[cand], float(T_km))
    c.origin[cand] = ORIGIN_ARC
    return int(cand.size)


def arc_regions(hcfg: dict, kin, key: str = "regions") -> list:
    """history yaml arcs.regions（或 arcs.basement）-> [{t, plate, omega_plate, poly_fn, what}]（U28 第二轮）。poly_fn(t) 给出
    t 时刻的多边形顶点：现今坐标的顶点按各自锚定的板块转回 t（默认锚在 plate 上；第三项为上盘代号；0 = 地幔参照系）。"""
    from mythic.tectonics.geometry import lonlat_to_xyz

    code_id = plate_codes(hcfg)
    out = []
    for r in (hcfg.get("arcs") or {}).get(key, []):
        q = code_id[r["plate"]]
        verts = []
        for v in r["poly"]:
            a = q if len(v) < 3 else (0 if v[2] in (0, "0") else code_id[v[2]])
            verts.append((a, lonlat_to_xyz(float(v[0]), float(v[1]))))

        def poly_fn(t, verts=verts):
            return np.array([x if a == 0 else unit(kin.total_from(a, 0.0, t) @ x) for a, x in verts])

        out.append({"t": float(r["t"]), "plate": q, "code": r["plate"], "poly_fn": poly_fn, "what": r.get("what", ""),
                    "T_km": r.get("T_km")})
    return sorted(out, key=lambda e: -e["t"])


def apply_region(eng, q: int, poly_xyz: np.ndarray, omega=None) -> dict:
    """把当前落在多边形里的存活洋壳单元（不论原属哪块板块）整批转给板块 q（U28 第二轮：俯冲起始时划定新的板块边界）；
    q 还不存在时先新增（参照系取当前地幔参照系）。陆壳不动。返回 {原板块: 转出的单元数}。"""
    from mythic.tectonics.crust import OCEAN
    from mythic.tectonics.reverse import in_sphere_polygon

    if q not in eng.R:
        eng.add_plate(q, np.zeros(3) if omega is None else omega)
    c = eng.c
    cand = np.flatnonzero(eng.alive & (c.ctype == OCEAN) & (c.plate != q))
    cand = cand[in_sphere_polygon(eng.positions()[cand], poly_xyz)]
    moved = {}
    for s in np.unique(c.plate[cand]):
        sel = cand[eng.c.plate[cand] == s]
        eng.transfer_units(int(s), int(q), sel, kind="region_transfer")
        moved[int(s)] = int(sel.size)
    return moved


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


def build_plumes(hcfg: dict, design: dict | None = None, grid=None, land: np.ndarray | None = None) -> dict:
    """history yaml plumes -> {"swells": [...], "lips": [...]}（U29）。swell 的 at 在地幔参照系里固定；lip 的范围是现今坐标、
    锚在 anchor 板块上（不写或 0 = 地幔参照系），t 时刻的范围 = 把它按 anchor 板块的运动从现今转回 t（与岛弧系统的区域同一约定）。
    范围三选一：at（点）或 line（折线）加 radius_km / flat_km；或 feature = 终图高原编号，取终图同一轮廓（带平滑起伏）与
    从边缘往里 ramp_km 的渐变（design_final_map.region_ramp），这时要给 design、grid、land（终图 1024x512 栅格）。"""
    from design_final_map import PLATEAU_MEANDER, region_ramp
    from mythic.tectonics.geometry import lonlat_to_xyz

    plateaus = {q["id"]: q for q in (design or {}).get("plateaus", [])}
    code_id = plate_codes(hcfg)
    swells, lips = [], []
    for pl in hcfg.get("plumes", []):
        if pl.get("swell"):
            sw = pl["swell"]
            swells.append({"id": pl["id"], "name": pl["name"], "center": lonlat_to_xyz(*pl["at"]),
                           "amp_m": float(sw["amp_m"]), "radius_km": float(sw["radius_km"]),
                           "active": (float(pl["active"][0]), float(pl["active"][1]))})
        for lp in pl.get("lips", []):
            a = lp.get("anchor", 0)
            e = {"id": pl["id"], "name": pl["name"], "what": lp.get("what", ""), "t": float(lp["t"]),
                 "anchor": 0 if a in (0, "0", None) else code_id[a], "anchor_code": a, "add_km": float(lp["add_km"]),
                 "pts0": None, "W": None, "radius_km": float(lp.get("radius_km", 0.0)), "flat_km": float(lp.get("flat_km", 0.0))}
            if "feature" in lp:
                if design is None:
                    raise ValueError(f"地幔柱 {pl['id']} 的大火成岩省按终图要素 {lp['feature']} 取范围，要给 design、grid、land")
                _, ramp = region_ramp(plateaus[lp["feature"]], grid, land, PLATEAU_MEANDER)
                e.update({"W": ramp, "grid": grid, "feature": lp["feature"]})
            else:
                q = np.asarray(lp["line"] if "line" in lp else [lp["at"]], float)
                e["pts0"] = np.atleast_2d(lonlat_to_xyz(q[:, 0], q[:, 1]))
            lips.append(e)
    return {"swells": swells, "lips": sorted(lips, key=lambda e: -e["t"])}


def build_reactivations(hcfg: dict, design: dict, grid, land: np.ndarray) -> list:
    """history yaml reactivations -> 板内挤压复活列表（U33，packages/tectonics/reactivation.py）。范围是终图要素：山脊线（mountains，
    两侧 width_km/2 内权重 1、到 width_km 升余弦减到 0）或高原（plateaus，同一轮廓与从边缘往里 ramp_km 的渐变），现今坐标、
    锚在 anchor 板块上；[t_start, t_end] 期间按 strain（总增厚比 − 1）匀速纯剪切增厚。"""
    from design_final_map import PLATEAU_MEANDER, region_ramp
    from mythic.tectonics.reactivation import line_weight

    code_id = plate_codes(hcfg)
    mts = {m["id"]: m for m in design.get("mountains", [])}
    pls = {q["id"]: q for q in design.get("plateaus", [])}
    out = []
    for r in hcfg.get("reactivations", []):
        fid = r["feature"]
        if fid in mts:
            w_km = float(mts[fid].get("width_km", 150))
            d = line_distance_km(grid, mts[fid]["line"], np.ones(grid.shape, bool), w_km)
            W = line_weight(np.where(np.isfinite(d), d, 1e9), 0.5 * w_km)
        else:
            _, W = region_ramp(pls[fid], grid, land, PLATEAU_MEANDER)
            W = np.zeros(grid.shape) if W is None else W
        t0, t1 = float(r["t_start"]), float(r.get("t_end", 0.0))
        out.append({"id": r["id"], "name": r.get("name", fid), "what": r.get("what", ""), "anchor": code_id[r["anchor"]],
                    "t_start": t0, "t_end": t1, "strain": float(r["strain"]),
                    "rate": float(np.log1p(float(r["strain"])) / max(t0 - t1, 1e-9)), "W": W, "grid": grid})
    return out


def build_rifts(hcfg: dict) -> list:
    """history yaml rifts -> 陆内裂谷列表（U30）。line 是现今坐标、锚在 anchor 板块上；t_start 起地堑减薄、两肩抬升与轴部隆起。"""
    from mythic.tectonics.geometry import lonlat_to_xyz

    code_id = plate_codes(hcfg)
    out = []
    for r in hcfg.get("rifts", []):
        q = np.asarray(r["line"], float)
        out.append({"id": r["id"], "name": r["name"], "what": r.get("what", ""), "t_start": float(r["t_start"]),
                    "anchor": code_id[r["anchor"]], "pts0": lonlat_to_xyz(q[:, 0], q[:, 1]),
                    "half_km": float(r["graben_half_km"]), "beta": float(r["beta"]),
                    "flank_amp_m": float(r["flank_amp_m"]), "flank_decay_km": float(r["flank_decay_km"]),
                    "swell_amp_m": float(r.get("swell_amp_m", 0.0)), "swell_radius_km": float(r.get("swell_radius_km", 1.0))})
    return out


def cont_gap_pairs(hcfg: dict) -> dict:
    """history yaml forward.pull_apart -> Engine.cont_gap_pairs（U30）：{板块对键: (T_narrow, w1, T_wide, w_max)}；
    写了 knots（U32 裂开的被动陆缘：[[累计张开宽度 km, 厚度 km], ...]）或 holes 的，给 {"knots": ..., "holes": ...}。"""
    code_id = plate_codes(hcfg)
    out = {}
    for r in (hcfg.get("forward") or {}).get("pull_apart", []):
        a, b = (code_id[x] for x in r["pair"])
        if "knots" in r:
            spec = {"knots": [(float(w), float(t)) for w, t in r["knots"]], "holes": bool(r.get("holes", False))}
        else:
            spec = (float(r["T_narrow_km"]), float(r["w1_km"]), float(r["T_wide_km"]), float(r["w_max_km"]))
            if r.get("holes"):
                spec = {"knots": [(0.0, spec[0]), (spec[1], spec[0]), (spec[3], spec[2])], "holes": True}
        out[min(a, b) * 1000 + max(a, b)] = spec
    return out


def arc_cont_pairs(hcfg: dict) -> set:
    """U32：陆壳上盘的弧增厚只用于这些板块对（Engine.arc_cont_pairs）：设计的俯冲带（arcs.subduction、ocean.subduction），
    加上 forward.arc_continental_extra_pairs 列出的大洋板块俯冲带（东大洋东侧板块 -> 远方大陆西岸，安第斯式）。
    固定的外大洋板块不在内；大洋板块与其他陆块的接触也不在内（第一次运行把大洋板块 × 所有陆块都算进去，洲 C 南岸、
    极地大陆、洲 B 北极海岸的陆架被增厚抬出海面）。"""
    code_id = plate_codes(hcfg)
    out = set(designed_pairs(hcfg))
    for a, b in (hcfg.get("forward") or {}).get("arc_continental_extra_pairs", []):
        out.add(min(code_id[a], code_id[b]) * 1000 + max(code_id[a], code_id[b]))
    return out


def cont_gap_since(hcfg: dict) -> dict:
    """history yaml forward.pull_apart 的 since（Ma）-> Engine.cont_gap_since（U30）：从这个时刻起才填拉伸陆壳。"""
    code_id = plate_codes(hcfg)
    out = {}
    for r in (hcfg.get("forward") or {}).get("pull_apart", []):
        if "since" in r:
            a, b = (code_id[x] for x in r["pair"])
            out[min(a, b) * 1000 + max(a, b)] = float(r["since"])
    return out


def eustatic_sea_level(frames: list, area_km2: np.ndarray, factor: float = 0.69) -> np.ndarray:
    """各帧的全球海平面（m，U30）：海水体积不变（取 0 Ma 海平面 0 时的体积），按每帧的高程分布（洋底随年龄的水深、陆地面积）
    解出刚性洋盆下的海平面，再乘海水加载的均衡改正 factor = (ρm − ρw) / ρm ≈ (3300 − 1030) / 3300 ≈ 0.69：
    海平面升 Δ 时海底在海水重压下下沉，相对大陆只升约 0.69Δ（Airy 均衡推得）。洋中脊年轻、洋底浅时海水被挤上大陆（陆表海）。"""
    e0 = np.asarray(frames[-1]["elev"], float)
    V0 = float((np.maximum(-e0, 0.0) * area_km2).sum())
    out = []
    for f in frames:
        e = np.asarray(f["elev"], float)
        lo, hi = -2000.0, 2000.0
        for _ in range(60):              # 二分：体积随海平面单调增
            mid = 0.5 * (lo + hi)
            if float((np.maximum(mid - e, 0.0) * area_km2).sum()) < V0:
                lo = mid
            else:
                hi = mid
        out.append(factor * 0.5 * (lo + hi))
    return np.asarray(out)


def simulate(world, mesh, rot: Path, snap_every: int, params: Params, polarity: dict, aprons: dict | None = None,
             gap_prefer: dict | None = None, volc=None, arcs_hcfg: dict | None = None, plumes: dict | None = None,
             rifts: list | None = None, gap_cont: dict | None = None, reacts: list | None = None):
    kin = RotationFileKinematics(rot)
    regions = arc_regions(arcs_hcfg, kin) if arcs_hcfg else []     # U28 第二轮：岛弧系统的俯冲起始
    basement = arc_regions(arcs_hcfg, kin, "basement") if arcs_hcfg else []   # U28 第三轮：新洋板块继承的老岛弧基底
    ri = bi = 0
    c = world.crust.gather(np.arange(len(world.crust)))
    c.uid = np.arange(len(c), dtype=np.int64)
    c.node = np.arange(len(c), dtype=np.int64)
    t0 = int(round(world.t))
    plates0 = sorted(int(q) for q in np.unique(c.plate))
    _, om = kin.step_rotations(plates0, t0)
    eng = Engine(mesh, c, params, float(t0), om)
    eng.polarity.update(polarity)
    eng.gap_prefer.update(gap_prefer or {})
    if gap_cont:                                    # U30：走滑拉分与裂谷陆缘
        eng.cont_gap_pairs.update(gap_cont[0])
        eng.cont_gap_since.update(gap_cont[1])
    if arcs_hcfg and (arcs_hcfg.get("forward") or {}).get("arc_continental_designed_only"):
        eng.arc_cont_pairs = arc_cont_pairs(arcs_hcfg)     # U32：陆壳上的弧增厚只用于设计的俯冲带（见 Engine.arc_cont_pairs）
    if aprons:
        aprons["pairs"] = over_down_pairs(aprons["hcfg"])
        aprons["ocean_downs"] = {}
        if aprons.get("ocean_plates"):        # U28 第二轮：大洋板块朝陆块汇聚时也不留围裙（见 strip_aprons）
            blocks = [int(b["id"]) for b in aprons["hcfg"]["blocks"]]
            oceans = [q for q in plate_codes(aprons["hcfg"]).values() if q not in blocks]
            for o in blocks:
                aprons["ocean_downs"][o] = [d for d in oceans if polarity.get(min(o, d) * 1000 + max(o, d)) == d]
    protected = np.zeros(0, np.int64)       # 岛弧系统划给陆块的洋壳（洋内弧所在），不当围裙转走
    blocks_ids = {int(b["id"]) for b in arcs_hcfg["blocks"]} if arcs_hcfg else set()
    if arcs_hcfg and params.gap_prefer_opening_only:
        # 汇聚边界上的网格小洞归下行板块：只限大洋板块下行的板块对与岛弧系统设计的俯冲带（见 Engine.commit）
        oceans = {q for q in plate_codes(arcs_hcfg).values() if q not in blocks_ids}
        arc_pairs = set()
        for r in arcs_hcfg["arcs"].get("subduction", []):
            for d in r["down"]:
                for o in r["over"]:
                    a, bb = plate_codes(arcs_hcfg)[d], plate_codes(arcs_hcfg)[o]
                    arc_pairs.add(min(a, bb) * 1000 + max(a, bb))
        eng.hole_down_pairs = arc_pairs | {k for k, sub in polarity.items() if sub in oceans}
        # U33：forward.polarity 里标了 hole_down 的板块对也按此处理（主内海北岸：洲 A、B 与逼近的微陆块之间、汇聚边界上的小洞归下行的
        # 微陆块，不按最近单元一半补给上盘的弧前）。第十次运行为 6–9°E 的深水窄湾加的，窄湾没变；真正的原因见 history yaml 的 blocks（I4）
        for r in (arcs_hcfg.get("forward") or {}).get("polarity", []):
            if r.get("hole_down"):
                for d in r["down"]:
                    for o in r["over"]:
                        a, bb = plate_codes(arcs_hcfg)[d], plate_codes(arcs_hcfg)[o]
                        eng.hole_down_pairs.add(min(a, bb) * 1000 + max(a, bb))
        # 弧—陆拼贴只在岛弧系统设计的俯冲带上（东南群岛东大岛、东岸大岛）：第二轮第十次运行，南方地盾陆块北上时擦过东南增生地块
        # 西侧的弧，弧被拼贴成陆、再被碰撞增厚，苏门答腊式长岛西边多出 0.29 百万 km²、约 3 km 高的陆地
        eng.arc_accrete_pairs = arc_pairs
        # 挤压型洋内弧（构造缩短可越过岩浆增生的上限，吕宋式）只用于设计里标了 compress 的俯冲带（U28 第三轮）。第一次运行按
        # "上盘朝海沟推进"自动判断、用于岛弧系统的全部俯冲带：南洋陆块板块随极地大陆运动，东缘一条约 3000 km 的弧增厚到 38 km、
        # 多出 0.93 百万 km² 陆地（地球上汤加—克马德克是后撤型岛弧）；上盘的绝对运动是倒推设定的，不能代表上盘应变状态
        eng.arc_compress_pairs = compress_pairs(arcs_hcfg)
        # U35：后撤型俯冲带（arcs.subduction 的 rollback: true）上盘不做构造缩短；设计的走滑撕裂边界（forward.no_collision）时段内不碰撞
        cid = plate_codes(arcs_hcfg)
        for r in arcs_hcfg["arcs"].get("subduction", []):
            if r.get("rollback"):
                for d in r["down"]:
                    for o in r["over"]:
                        eng.arc_rollback_pairs.add(min(cid[d], cid[o]) * 1000 + max(cid[d], cid[o]))
        for r in (arcs_hcfg.get("forward") or {}).get("no_collision", []):
            a, bb = (cid[x] for x in r["pair"])
            eng.no_collision_pairs[min(a, bb) * 1000 + max(a, bb)] = (float(r["from"]), float(r["to"]))
    if arcs_hcfg:                           # 洋内弧增厚加快只用于设计的俯冲带（U28 第二轮，见 Engine.arc_flux_pairs）
        eng.arc_flux_pairs = designed_pairs(arcs_hcfg)
    if arcs_hcfg and volc is not None:      # 岛弧火山只长在设计的俯冲带上（U28 第二轮，见 VolcanoField.arc_pairs）
        volc.arc_pairs = designed_pairs(arcs_hcfg)
        volc.arc_cont_plates = blocks_ids
        cid = plate_codes(arcs_hcfg)               # U35：按俯冲带单独设定弧火山离上盘陆壳的最小距离（arcs.subduction 的 volcano_min_cont_km）
        for r in (arcs_hcfg.get("arcs") or {}).get("subduction", []):
            if r.get("volcano_min_cont_km") is not None:
                for d in r["down"]:
                    for o in r["over"]:
                        a, bb = cid[d], cid[o]
                        volc.arc_min_cont_by_pair[min(a, bb) * 1000 + max(a, bb)] = float(r["volcano_min_cont_km"])
    from mythic.tectonics.plumes import apply_lip, apply_lip_weights, swell_m

    from mythic.tectonics.reactivation import apply_shortening
    from mythic.tectonics.rifts import apply_graben, rift_relief_m

    plumes = plumes or {"swells": [], "lips": []}
    rifts = rifts or []
    li = 0
    code_id_all = plate_codes(arcs_hcfg) if arcs_hcfg else {}
    rifts_from = sorted(({"id": int(b["id"]), "code": b["code"], "t": float(b["rift_from"]["t"]),
                          "parent": int(code_id_all[b["rift_from"]["plate"]]), "parent_code": b["rift_from"]["plate"],
                          "preserve_frame": bool(b["rift_from"].get("preserve_frame", False))}
                         for b in (arcs_hcfg or {}).get("blocks", []) if b.get("rift_from")), key=lambda e: -e["t"])
    rift_done = set()
    # U35：嵌套的裂开（西口弧形地体先是西段地体的一部分，西段地体又先是洲 C 的一部分）：母块裂开时连同它名下还没裂开的子地体一起转出
    rift_parent = {int(b["id"]): int(code_id_all[b["rift_from"]["plate"]])
                   for b in (arcs_hcfg or {}).get("blocks", []) if b.get("rift_from")}

    def rift_family(q):
        fam, todo = {q}, [q]
        while todo:
            x = todo.pop()
            for ch, par in rift_parent.items():
                if par == x and ch not in fam:
                    fam.add(ch)
                    todo.append(ch)
        return np.array(sorted(fam), np.int64)

    def rift_line(r, tt):
        return unit(r["pts0"] @ kin.total_from(r["anchor"], 0.0, float(tt)).T)

    def snap():
        """快照，叠加动力地形（U29：地幔参照系 = 固定网格，只改高程）与陆内裂谷两肩、轴部隆起（U30，随所在板块）。"""
        f = eng.snapshot()
        dyn = swell_m(mesh.xyz, float(f["t"]), plumes["swells"])
        for r in rifts:
            if float(f["t"]) <= r["t_start"] + 1e-9:
                dyn = dyn + rift_relief_m(mesh.xyz, rift_line(r, f["t"]), r["half_km"], r["flank_amp_m"],
                                          r["flank_decay_km"], r["swell_amp_m"], r["swell_radius_km"])
        f["dyn"] = dyn.astype(np.float32)
        f["elev"] = f["elev"] + f["dyn"]
        return f

    moved = 0.0
    frames = [snap()]
    vframes = [volc.state(eng)] if volc is not None else []
    t = t0
    tw = time.perf_counter()
    while t > 0:
        dR, om = kin.step_rotations(eng.plates, t)
        committed = eng.step(dR, om)
        t -= 1
        if committed and aprons:
            moved += strip_aprons(eng, aprons["pairs"], aprons["forearc_km"], aprons["approach_km"], aprons["between_cos"],
                                  aprons["min_conv_cm_yr"], aprons["ocean_downs"], protected)
        while committed and ri < len(regions) and regions[ri]["t"] >= t - 1e-9:
            e = regions[ri]
            ri += 1
            om_q = kin.step_rotations([e["plate"]], max(t, 1))[1][e["plate"]]
            poly = e["poly_fn"](float(t))
            mv = apply_region(eng, e["plate"], poly, om_q)
            if e["plate"] in blocks_ids:     # 划给陆块的洋壳：设计中的洋内弧所在（不当围裙转走；弧火山只长在这里）
                from mythic.tectonics.crust import OCEAN
                from mythic.tectonics.reverse import in_sphere_polygon
                mine = np.flatnonzero(eng.alive & (eng.c.plate == e["plate"]) & (eng.c.ctype == OCEAN))
                mine = mine[in_sphere_polygon(eng.positions()[mine], poly)]
                protected = np.union1d(protected, eng.c.uid[mine])
                if volc is not None:
                    volc.arc_protected_uids = protected
            log(f"  {t:3d} Ma 岛弧系统：{e['code']} 接收洋壳单元 {sum(mv.values())} 个（来自 {mv}）")
        while committed and bi < len(basement) and basement[bi]["t"] >= t - 1e-9:
            e = basement[bi]                 # 在同一时刻的区域划分（上面的循环）之后
            bi += 1
            n = apply_basement(eng, e["plate"], e["poly_fn"](float(t)), float(e["T_km"]))
            log(f"  {t:3d} Ma 老岛弧基底：{e['code']} 的 {n} 个洋壳单元厚度至少 {e['T_km']} km")
        while committed and li < len(plumes["lips"]) and plumes["lips"][li]["t"] >= t - 1e-9:
            e = plumes["lips"][li]             # U29 大火成岩省：范围按 anchor 板块从现今转回此刻
            li += 1
            Rt = np.eye(3) if e["anchor"] == 0 else kin.total_from(e["anchor"], 0.0, float(t))   # 现今 -> t
            if e["W"] is not None:          # 终图要素的范围：单元此刻的位置转回现今坐标，查栅格上的渐变
                g = e["grid"]
                lo, la = xyz_to_lonlat(eng.positions() @ Rt)
                i = np.clip(((90.0 - la) / 180.0 * g.ny).astype(int), 0, g.ny - 1)
                j = np.clip(((lo + 180.0) / 360.0 * g.nx).astype(int), 0, g.nx - 1)
                r = apply_lip_weights(eng, np.where(eng.alive, e["W"][i, j], 0.0), e["add_km"])
            else:
                r = apply_lip(eng, unit(e["pts0"] @ Rt.T), e["radius_km"], e["add_km"], e["flat_km"])
            log(f"  {t:3d} Ma 大火成岩省 {e['id']}（{e['what']}）：{r['n']} 个单元加厚至多 {e['add_km']} km，"
                f"{r['volume_km3'] / 1e6:.2f}×10⁶ km³")
        for rf in rifts:                           # U30 陆内裂谷：开始时地堑减薄（两肩与轴部隆起在快照里叠加）
            if committed and rf["id"] not in rift_done and rf["t_start"] >= t - 1e-9:
                rift_done.add(rf["id"])
                r = apply_graben(eng, rift_line(rf, t), rf["half_km"], rf["beta"])
                log(f"  {t:3d} Ma 陆内裂谷 {rf['id']}（{rf['name']}）：{r['n']} 个陆壳单元减薄（β ≤ {rf['beta']}），"
                    f"{r['volume_km3'] / 1e6:.2f}×10⁶ km³")
        while (arcs_hcfg is not None and rifts_from and rifts_from[0]["t"] >= t - 1e-9
               and (committed or rifts_from[0]["preserve_frame"])):
            e = rifts_from.pop(0)                  # U33：微陆块从母板块裂开（之前是母板块的一部分）
            om_q = kin.step_rotations([e["id"]], max(t, 1))[1][e["id"]]
            sel = np.flatnonzero(eng.alive & (eng.c.plate == e["parent"]) & np.isin(eng.c.origin, rift_family(e["id"])))
            if e["preserve_frame"]:
                # U35-C2：逐 Myr 准时裂离，继承母块参照系，不做重采样或额外全局提交。
                n = eng.rift_units(e["parent"], e["id"], sel, om_q)
            else:
                eng.add_plate(e["id"], om_q)
                n = eng.transfer_units(e["parent"], e["id"], sel, kind="terrane_rift")
            log(f"  {t:3d} Ma 微陆块 {e['code']} 从 {e['parent_code']} 裂开：{sel.size} 个单元转出（新板块 {n} 个单元）")
        for ra in reacts or []:                    # U33 板内挤压复活：期间每 Myr 纯剪切增厚（范围按 anchor 板块从现今转回此刻）
            if ra["t_end"] <= t < ra["t_start"]:
                Rt = kin.total_from(ra["anchor"], 0.0, float(t))
                lo, la = xyz_to_lonlat(eng.positions() @ Rt)
                g = ra["grid"]
                i = np.clip(((90.0 - la) / 180.0 * g.ny).astype(int), 0, g.ny - 1)
                j = np.clip(((lo + 180.0) / 360.0 * g.nx).astype(int), 0, g.nx - 1)
                r = apply_shortening(eng, np.where(eng.alive, ra["W"][i, j], 0.0), ra["rate"], 1.0, plate=ra["anchor"])
                ra["vol"] = ra.get("vol", 0.0) + r["volume_km3"]
                if t == ra["t_end"]:
                    log(f"  {t:3d} Ma 板内挤压复活 {ra['id']}（{ra['name']}）：{ra['t_start']:g}–{ra['t_end']:g} Ma 共加厚 "
                        f"{ra['vol'] / 1e6:.2f}×10⁶ km³（总增厚比 {1 + ra['strain']:.2f}）")
        if volc is not None:
            volc.step(eng)
        if committed and t % snap_every == 0:
            frames.append(snap())
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
    frames = [{name: (z[f"{name}_{k}"] if f"{name}_{k}" in z.files else np.zeros_like(z[f"elev_{k}"])) for name in FIELDS}
              | {"t": float(t)} for k, t in enumerate(z["times"])]
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


def auto_shelf_emerged(grid, E_eng: np.ndarray, path: Path, min_Mkm2: float = 0.03, buffer_km: float = 150.0) -> dict | None:
    """U32：初始世界自动补的陆架（scripts/design_initial_world.py 存的现今参照系掩膜）在 0 Ma 有多少被抬出海面。
    露出的地方是正向演化里实际在汇聚的陆缘（俯冲上盘的弧前增厚，或下行一侧撞进海沟），按"活动陆缘的陆架窄"不该补；
    连通的片给出中心与半径（最远格点 + buffer_km），可写进 history yaml initial.margins 的 active_points。"""
    if not path.exists():
        return None
    sm = np.load(path)
    if sm.shape != E_eng.shape:
        return None
    a = grid.cell_area_km2
    em = sm & (E_eng > 0)
    lab, n = label_wrap(em)
    clusters = []
    for k in range(1, n + 1):
        m = lab == k
        ar = float(a[m].sum()) / 1e6
        if ar < min_Mkm2:
            continue
        X = grid.xyz[m]
        cen = X.sum(0)
        cen /= np.linalg.norm(cen)
        lo, la = xyz_to_lonlat(cen)
        r = float(km_from_chord(np.linalg.norm(X - cen, axis=1)).max()) + buffer_km
        clusters.append({"lon": round(float(lo), 1), "lat": round(float(la), 1), "radius_km": int(round(r, -1)),
                         "area_Mkm2": round(ar, 3)})
    clusters.sort(key=lambda c: -c["area_Mkm2"])
    return {"area_Mkm2": round(float(a[sm].sum()) / 1e6, 2), "emerged_Mkm2": round(float(a[em].sum()) / 1e6, 2),
            "emerged_fraction": round(float(a[em].sum()) / max(float(a[sm].sum()), 1.0), 3), "clusters": clusters}


def write_shelf_active(path: Path, grid, E_eng: np.ndarray, shelf_path: Path, run_tag: str, spacing_km: float = 150.0,
                       radius_km: float = 200.0, min_elev_m: float = 300.0, min_Mkm2: float = 0.03) -> int:
    """U32：把 0 Ma 被明显抬起的自动陆架写成"实测活动陆缘"点清单（history yaml initial.margins 自动陆架的 active_points_file）。
    只算高出海面 min_elev_m 以上、连成 min_Mkm2 以上的片（第一次把所有露出海面的都算进去：贴海岸的一格陆架被旁边的地壳流动
    垫高几十米就露出，517 个点几乎罩住了所有海岸，自动陆架 18.5 -> 0.66 百万 km²）；按 spacing_km 间隔贪心取点，每点半径 radius_km。"""
    sm = np.load(shelf_path)
    em = sm & (E_eng > min_elev_m)
    lab, n = label_wrap(em)
    if n:
        a = grid.cell_area_km2
        big = np.bincount(lab.ravel(), weights=a.ravel(), minlength=n + 1) / 1e6 >= min_Mkm2
        big[0] = False
        em = big[lab]
    X = grid.xyz[em]
    order = np.argsort(-np.abs(X[:, 2]))        # 先高纬再低纬，结果与格点顺序无关地稳定
    pts = []
    cos_s = np.cos(spacing_km / R_KM)
    for k in order:
        if not pts or (X[pts] @ X[k]).max() < cos_s:
            pts.append(int(k))
    lo, la = xyz_to_lonlat(X[pts])
    lines = ["# U32：实测活动陆缘——自动陆架（老被动陆缘的陆架与陆坡）在正向演化 0 Ma 被明显抬起的地方，下一次构建初始世界时不补陆架。",
             "# 由 scripts/design_forward.py --write-shelf-active 生成，不要手改。露出的原因是那里在正向演化中实际在汇聚：",
             "# 下行一侧的陆架撞进海沟、撞上岛弧（碰撞增厚），或贴着增厚的造山带被侧向流动的地壳垫高（陆架本身不接受弧增厚，",
             "# engine_params.arc_cont_min_T_km）。地球上活动陆缘的陆架窄。",
             f"# 来源运行：{run_tag}；高出海面 {min_elev_m:g} m 以上、成片 {min_Mkm2:g} 百万 km² 以上的 {float(grid.cell_area_km2[em].sum()) / 1e6:.2f} 百万 km²，"
             f"按 {spacing_km:g} km 间隔取 {len(pts)} 个点，每点半径 {radius_km:g} km。",
             "points:"]
    lines += [f"  - [{float(a):.2f}, {float(b):.2f}, {radius_km:g}]" for a, b in zip(np.atleast_1d(lo), np.atleast_1d(la))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(pts)


def hypsometry(grid, E: np.ndarray) -> dict:
    """U32：陆地高程的面积加权分位与大陆架（0 ~ −200 m）占海洋面积的比例（评审 stage1_2026-10-25_1 第 1、3 条的客观证据）。"""
    a = grid.cell_area_km2
    land, oc = E > 0, E <= 0

    def wpct(x, w, ps):
        o = np.argsort(x)
        c = np.cumsum(w[o]) / w[o].sum()
        return [int(round(float(np.interp(q / 100.0, c, x[o])))) for q in ps]

    return {"land_elev_m_percentiles": wpct(E[land], a[land], [5, 10, 25, 50, 75, 90]),
            "shelf_fraction_of_ocean": round(float(a[oc & (E > -200)].sum() / a[oc].sum()), 4)}


BOUNDARY_CLASSES = {0: "无", 1: "汇聚", 2: "离散", 3: "走滑", 4: "缓慢（< 0.5 cm/yr）"}


def boundary_classes(plate: np.ndarray, grid, omega: dict, min_cm_yr: float = 0.5,
                     smooth_cells: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """U34：0 Ma 板块边界类型（评审 stage1_2026-10-25_3 第 14 条）。在经纬度格网上，相邻（右、下）两格属于不同板块处，
    用旋转文件最后 1 Myr 的角速度算相对速度，按法向分量分类：|法向| ≥ 一半切向且 < −min 为汇聚、> min 为离散；
    其余切向 > min 为走滑；都不到 min 为缓慢。返回 (类型 int8, 相对速度 cm/yr float32)，每个边界格点取最快的那条边。
    法向取边界两侧各自在 smooth_cells × smooth_cells 窗口（1024x512 上 7 格约 270 km）里的质心连线：栅格边界一格一齿，
    逐格的法向是锯齿状的，斜向运动的边界会逐格换类型（评审 stage1_2026-10-25_4 第 12 条）。smooth_cells ≤ 1 用逐格法向。"""
    ny, nx = plate.shape
    cls = np.zeros(plate.shape, np.int8)
    rate = np.zeros(plate.shape, np.float32)
    X = grid.xyz
    cen = {}
    if smooth_cells > 1:
        def box(f):              # 经向环绕、纬向贴边的方窗求和
            return ndimage.uniform_filter(f, size=smooth_cells, mode=("nearest", "wrap"))
        for q in np.unique(plate):
            mq = (plate == q).astype(np.float32)
            nq = box(mq)
            s = np.stack([box(mq * X[..., k].astype(np.float32)) for k in range(3)], axis=-1)
            cen[int(q)] = s / np.maximum(nq, 1e-6)[..., None]
    for di, dj in ((0, 1), (1, 0)):
        a = plate[: ny - di, :]
        b = np.roll(plate, -dj, axis=1)[di:, :] if dj else plate[di:, :]
        xa = X[: ny - di, :]
        xb = np.roll(X, -dj, axis=1)[di:, :] if dj else X[di:, :]
        ii, jj = np.nonzero(a != b)
        if ii.size == 0:
            continue
        pa, pb = a[ii, jj], b[ii, jj]
        m = unit(xa[ii, jj] + xb[ii, jj])
        wa = np.stack([omega.get(int(q), np.zeros(3)) for q in pa])
        wb = np.stack([omega.get(int(q), np.zeros(3)) for q in pb])
        v = np.cross(wb - wa, m) * R_KM / 10.0                     # b 相对 a，cm/yr
        n = xb[ii, jj] - xa[ii, jj]
        if cen:                                                    # 平滑法向：a 侧质心 -> b 侧质心（在 a 格点处的窗口里量）
            ns = np.stack([cen[int(q)][i_, j_] for q, i_, j_ in zip(pb, ii, jj)]) - \
                np.stack([cen[int(q)][i_, j_] for q, i_, j_ in zip(pa, ii, jj)])
            ok = np.linalg.norm(ns, axis=1) > 1e-9
            n = np.where(ok[:, None], ns, n)
        n = n - (n * m).sum(1, keepdims=True) * m
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        vn = (v * n).sum(1)
        vt = np.linalg.norm(v - vn[:, None] * n, axis=1)
        c = np.full(ii.size, 4, np.int8)
        normal = np.abs(vn) >= 0.5 * vt
        c[normal & (vn < -min_cm_yr)] = 1
        c[normal & (vn > min_cm_yr)] = 2
        c[(c == 4) & (vt > min_cm_yr)] = 3
        sp = np.hypot(vn, vt).astype(np.float32)
        jb = (jj + dj) % nx
        for (i_, j_) in ((ii, jj), (ii + di, jb)):
            upd = sp >= rate[i_, j_]
            rate[i_[upd], j_[upd]] = sp[upd]
            cls[i_[upd], j_[upd]] = c[upd]
    return cls, rate


def tectonic_layers(f0: dict, raster, grid, rot: Path) -> dict:
    """U34：0 Ma 构造图层，随标准世界文件导出（标准图组的区域窗口构造叠合图，评审第 14 条）。"""
    kin = RotationFileKinematics(rot)
    P = raster(f0["plate"]).astype(np.int32)
    omega = {}
    for q in np.unique(P):
        try:
            omega[int(q)] = kin.step_rotations([int(q)], 1)[1][int(q)]
        except Exception:  # noqa: BLE001  旋转文件里没有的板块（不动）
            omega[int(q)] = np.zeros(3)
    cls, rate = boundary_classes(P, grid, omega)
    C = raster(f0["ctype"]).astype(np.int8)
    oro = np.where(C == CONT, raster(f0["oro"]), 0).astype(np.int8)
    return {"tect_plate": P, "tect_crust_type": C, "tect_crust_km": raster(f0["T"]).astype(np.float32),
            "tect_ocean_age_myr": np.where(C == CONT, np.nan, raster(f0["age"])).astype(np.float32),
            "tect_last_event": oro,
            "tect_last_event_ma": np.where(oro > 0, raster(f0["oro_t"]), np.nan).astype(np.float32),
            "tect_boundary": cls, "tect_boundary_cm_yr": rate}


TECT_ATTRS = {
    "tect_plate": {"description": "0 Ma 板块编号（history yaml 的 blocks / ocean / arcs）", "method_class": "procedural_constrained"},
    "tect_crust_type": {"description": "地壳类型：0 洋壳，1 陆壳", "method_class": "procedural_constrained"},
    "tect_crust_km": {"units": "km", "description": "地壳厚度", "method_class": "procedural_constrained"},
    "tect_ocean_age_myr": {"units": "Myr", "description": "洋底年龄（陆壳为 NaN）", "method_class": "procedural_constrained"},
    "tect_last_event": {"description": "陆壳最近一次构造改造：0 无，1 俯冲弧，2 碰撞，3 裂谷，4 板内挤压复活（设计给出）",
                        "method_class": "procedural_constrained"},
    "tect_last_event_ma": {"units": "Ma", "description": "最近一次构造改造的时间", "method_class": "procedural_constrained"},
    "tect_boundary": {"description": "板块边界类型：0 无，1 汇聚，2 离散，3 走滑，4 缓慢（< 0.5 cm/yr）；"
                                     "相对速度取自倒推的旋转文件最后 1 Myr（设计的运动，不是动力学算出的）",
                      "method_class": "procedural_constrained"},
    "tect_boundary_cm_yr": {"units": "cm/yr", "description": "边界两侧板块的相对速度", "method_class": "procedural_constrained"},
}


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


def compare_arcs(design: dict, v0: dict) -> dict:
    """0 Ma 岛弧火山（U28）与终图岛弧对照：每条终图岛弧沿线每 50 km 一个点，150 km 内有引擎弧火山的比例（覆盖率），
    沿线的岛与海山数；离任何终图岛弧都超过 300 km 的引擎弧火山（多出的岛弧）。"""
    from design_final_map import densify
    from mythic.tectonics.volcanism import ISLAND

    arc = v0["hotspot"] == "ARC"
    X = np.stack([np.cos(np.radians(v0["lat"])) * np.cos(np.radians(v0["lon"])),
                  np.cos(np.radians(v0["lat"])) * np.sin(np.radians(v0["lon"])), np.sin(np.radians(v0["lat"]))], axis=1)
    Xa, kind = X[arc], v0["kind"][arc]
    tree = cKDTree(Xa) if arc.any() else None
    rows, near_any = [], np.zeros(int(arc.sum()), bool)
    for isl in design.get("islands", []):
        if isl["type"] != "arc":
            continue
        P, s = densify(isl["line"], 50.0)
        if tree is None:
            rows.append({"id": isl["id"], "name": isl["name"], "coverage": 0.0, "n_island": 0, "n_seamount": 0})
            continue
        d, _ = tree.query(P)
        cov = float((km_from_chord(d) <= 150.0).mean())
        dd, _ = cKDTree(P).query(Xa)
        close = km_from_chord(dd) <= 150.0
        near_any |= km_from_chord(dd) <= 300.0
        rows.append({"id": isl["id"], "name": isl["name"], "length_km": round(float(s[-1])), "coverage": round(cov, 2),
                     "n_island": int((close & (kind == ISLAND)).sum()), "n_seamount": int((close & (kind != ISLAND)).sum())})
    extra = ~near_any
    return {"arcs": rows, "engine_arc_volcanoes": int(arc.sum()), "engine_arc_islands": int((kind == ISLAND).sum()),
            "far_from_design_arcs": int(extra.sum()), "far_from_design_arcs_islands": int((extra & (kind == ISLAND)).sum())}


def compare_gulfs(hcfg: dict, design: dict, b: dict, grid, E_eng: np.ndarray, E_fin: np.ndarray) -> list:
    """0 Ma 浅海湾对照（U30）：initial.margins 里按 gulfs 补了陆壳的海湾（终图的海湾范围），引擎的水深分布与设计水深。"""
    rows = []
    for mg in (hcfg.get("initial") or {}).get("margins", []):
        for gid in mg.get("gulfs", []):
            m = b["sea_masks"].get(gid)
            if m is None:
                continue
            m = m & ~b["land"]
            if not m.any():
                continue
            e = E_eng[m]
            names = {g["id"]: g["name"] for g in design.get("gulfs", [])}
            rows.append({"id": gid, "name": names.get(gid, gid), "design_depth_m": mg["depth_m"],
                         "area_Mkm2": round(float(grid.cell_area_km2[m].sum()) / 1e6, 3),
                         "engine_elev_m_p50": round(float(np.median(e))), "engine_elev_m_p10": round(float(np.percentile(e, 10))),
                         "engine_shallower_200m": round(float((e > -200).mean()), 2),
                         "engine_land_fraction": round(float((e > 0).mean()), 2)})
    return rows


def compare_plumes(hcfg: dict, design: dict, grid, E_eng: np.ndarray, E_fin: np.ndarray, Lf: np.ndarray,
                   dyn: np.ndarray) -> list:
    """0 Ma 地幔柱对照（U29）：每次大火成岩省的现今范围（剖面过半处）里引擎与终图的高程、陆地比例；动力地形穹隆中心的抬升量。"""
    from mythic.tectonics.plumes import distance_km, taper

    pl = build_plumes(hcfg, design, grid, Lf)
    X = grid.xyz.reshape(-1, 3)
    rows = []
    for e in pl["lips"]:
        w = e["W"] if e["W"] is not None else taper(distance_km(X, e["pts0"]), e["radius_km"], e["flat_km"]).reshape(grid.shape)
        m = w > 0.5
        rows.append({"id": e["id"], "kind": "lip", "what": e["what"], "t": e["t"], "add_km": e["add_km"],
                     "area_Mkm2": round(float(grid.cell_area_km2[m].sum()) / 1e6, 3),
                     "engine_elev_m_p50": round(float(np.median(E_eng[m]))), "engine_elev_m_p90": round(float(np.percentile(E_eng[m], 90))),
                     "final_elev_m_p50": round(float(np.median(E_fin[m]))), "final_elev_m_p90": round(float(np.percentile(E_fin[m], 90))),
                     "engine_land_fraction": round(float((E_eng[m] > 0).mean()), 2), "final_land_fraction": round(float(Lf[m].mean()), 2)})
    for sw in pl["swells"]:
        d = distance_km(X, sw["center"][None]).reshape(grid.shape)
        rows.append({"id": sw["id"], "kind": "swell", "name": sw["name"], "amp_m": sw["amp_m"], "radius_km": sw["radius_km"],
                     "dyn_m_at_center_0Ma": round(float(dyn[np.unravel_index(np.argmin(d), d.shape)]))})
    return rows


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
    ap.add_argument("--write-shelf-active", action="store_true",
                    help="U32：把 0 Ma 露出海面的自动陆架写进 config/design/shelf_active_<history>.yaml（实测活动陆缘）")
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
    design = apply_options(yaml.safe_load((ROOT / "config" / "design" / f"final_map_{args.design}.yaml")
                                          .read_text(encoding="utf-8")), set())
    design["_ver"] = args.design
    grid = LatLonGrid(1024, 512)
    b = build(design, grid)
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
                   "approach_km": float(ap["approach_km"]), "min_conv_cm_yr": float(ap.get("min_conv_cm_yr", 0.5)),
                   "ocean_plates": bool(ap.get("ocean_plates", False))}
                  if ap and ap.get("strip", True) else None)
        volc = None
        if hcfg.get("hotspots"):
            from mythic.tectonics.volcanism import VolcanoField, VolcanoParams
            vp = VolcanoParams(**fcfg.get("volcano_params", {}))
            log(f"火山参数（history yaml forward.volcano_params）：{fcfg.get('volcano_params', {})}")
            hs = build_hotspots(hcfg, RotationFileKinematics(rot), vp)
            volc = VolcanoField(hs, vp)
            log(f"热点 {len(hs)} 个（history yaml hotspots）")
        plumes = build_plumes(hcfg, design, grid, b["land"])
        if hcfg.get("plumes"):
            log(f"地幔柱 {len(hcfg['plumes'])} 个（history yaml plumes）：动力地形穹隆 {len(plumes['swells'])} 个，"
                f"大火成岩省 {len(plumes['lips'])} 次")
        rifts = build_rifts(hcfg)
        gap_cont = (cont_gap_pairs(hcfg), cont_gap_since(hcfg))
        if rifts or gap_cont[0]:
            log(f"陆内裂谷 {len(rifts)} 条（history yaml rifts），走滑拉分与裂谷陆缘的板块对 {len(gap_cont[0])} 个（forward.pull_apart）")
        reacts = build_reactivations(hcfg, design, grid, b["land"])
        if reacts:
            log(f"板内挤压复活 {len(reacts)} 处（history yaml reactivations）：" + "，".join(
                f"{r['id']} {r['t_start']:g}–{r['t_end']:g} Ma 增厚比 {1 + r['strain']:.2f}" for r in reacts))
        eng, frames, vframes = simulate(world, mesh, rot, args.snap_every, params, polarity, aprons,
                                        ocean_gap_prefer(hcfg), volc, hcfg if hcfg.get("arcs") else None, plumes,
                                        rifts, gap_cont, reacts)
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
    sea_level = eustatic_sea_level(frames, mesh.area_km2)       # U30：全球海平面（0 Ma 为 0）
    for f, sl in zip(frames, sea_level):
        f["sea_level"] = float(sl)
    cont_flooded = [round(float(mesh.area_km2[(f["ctype"] == CONT) & (f["elev"] < f["sea_level"])].sum()) / 1e6, 1)
                    for f in frames]
    log("全球海平面（m）：" + "，".join(f"{f['t']:g} Ma {f['sea_level']:+.0f}" for f in frames[::3]))

    log("对照终图（1024x512）")
    ids, info = assign_blocks(hcfg, grid, b)
    E_fin = design_elevation(design, grid, b)
    raster = Raster(mesh, 512)
    E_eng = raster(frames[-1]["elev"]).astype(float)
    v0 = vframes[-1] if vframes else None
    hs_cmp, arc_cmp = [], None
    if v0 is not None:
        # 火山岛：半径不小于约半个格点的画进海陆掩膜（与终图同一约定），高程至少 +100 m 以便出图着色
        from design_final_map import MIN_ISLAND_RADIUS_KM
        from mythic.tectonics.volcanism import island_mask
        isl = island_mask(grid, v0, MIN_ISLAND_RADIUS_KM)
        E_eng = np.where(isl & (E_eng <= 0), 100.0, E_eng)
        hs_cmp = compare_hotspots(design, hcfg, v0)
        arc_cmp = compare_arcs(design, v0)
        log(f"  岛弧火山 {arc_cmp['engine_arc_volcanoes']} 座（其中岛 {arc_cmp['engine_arc_islands']}），"
            f"离终图岛弧都超过 300 km 的 {arc_cmp['far_from_design_arcs']} 座（岛 {arc_cmp['far_from_design_arcs_islands']}）")
        for r in arc_cmp["arcs"]:
            log(f"  岛弧 {r['id']:4s} {r['name']}：覆盖率 {r['coverage']:.0%}，沿线岛 {r['n_island']}、海山 {r['n_seamount']}")
        for r in hs_cmp:
            log(f"  热点 {r['id']:4s} {r['name']}：岛 {r['n_island']}（最大半径 {r['max_island_radius_km']} km，终图 {r['design_size_km']}），"
                f"环礁 {r['n_atoll']}，海山 {r['n_seamount']}；链的方向 {r['engine_bearing_deg']}°（终图 {r['design_bearing_deg']}°），"
                f"最老的 {r['oldest_Ma']} Ma，离热点最远 {r['max_dist_km']} km（终图链长 {r['design_length_km']} km）")
    cmp_ = compare(design, grid, b, ids, info, E_eng, E_fin, label_points(design))
    shelf_npy = CACHE / "design" / f"shelf_auto_{args.design}_{args.history}_{grid.name}.npy"
    shelf_cmp = auto_shelf_emerged(grid, E_eng, shelf_npy)
    if args.write_shelf_active and shelf_npy.exists():
        out_sa = ROOT / "config" / "design" / f"shelf_active_{args.history}.yaml"
        n_sa = write_shelf_active(out_sa, grid, E_eng, shelf_npy, f"{run_js['code_version']}，{time.strftime('%Y-%m-%d %H:%M')}")
        log(f"写 {out_sa.relative_to(ROOT).as_posix()}：{n_sa} 个点")
    if shelf_cmp:
        log(f"  自动陆架：{shelf_cmp['area_Mkm2']} 百万 km²，0 Ma 露出海面 {shelf_cmp['emerged_Mkm2']}（{shelf_cmp['emerged_fraction']:.0%}）；"
            f"露出的片（> 0.03 百万 km²，可写进 initial.margins 的 active_points）："
            + "；".join(f"[{c['lon']}, {c['lat']}, {c['radius_km']}] {c['area_Mkm2']}" for c in shelf_cmp["clusters"]))
    gulf_cmp = compare_gulfs(hcfg, design, b, grid, E_eng, E_fin)
    for r in gulf_cmp:
        log(f"  浅海湾 {r['id']} {r['name']}：设计水深 {r['design_depth_m']} m，引擎中位 {r['engine_elev_m_p50']} m"
            f"（浅于 200 m 的占 {r['engine_shallower_200m']:.0%}、露出海面的占 {r['engine_land_fraction']:.0%}）")
    plume_cmp = []
    if hcfg.get("plumes"):
        plume_cmp = compare_plumes(hcfg, design, grid, E_eng, E_fin, b["land"], raster(frames[-1]["dyn"]).astype(float))
    for r in plume_cmp:
        if r["kind"] == "lip":
            log(f"  大火成岩省 {r['id']}（{r['what']}，{r['t']:g} Ma）：现今范围 {r['area_Mkm2']} 百万 km²，引擎高程中位 / 90 分位 "
                f"{r['engine_elev_m_p50']} / {r['engine_elev_m_p90']} m（终图 {r['final_elev_m_p50']} / {r['final_elev_m_p90']}），"
                f"陆地 {r['engine_land_fraction']:.0%}（终图 {r['final_land_fraction']:.0%}）")
        else:
            log(f"  动力地形 {r['id']} {r['name']}：中心抬升 {r['dyn_m_at_center_0Ma']} m（设定 {r['amp_m']:g} m、半径 {r['radius_km']:g} km）")

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
                for name, k in (("collision", ORO_COLL), ("arc", ORO_ARC), ("rift", ORO_RIFT),
                                ("reactivation", ORO_REACT))}
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
        "arcs_0Ma": arc_cmp,
        "plumes_0Ma": plume_cmp,
        "sea_level_m": [{"t": f["t"], "sea_level_m": round(f["sea_level"], 1), "flooded_continental_crust_Mkm2": a}
                        for f, a in zip(frames, cont_flooded)],
        "gulfs_0Ma": gulf_cmp,
        "auto_shelf_0Ma": shelf_cmp,
        "hypsometry_0Ma": hypsometry(grid, E_eng),
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
    # U31：0 Ma 世界导出为标准世界文件（packages/figures/world.py），供标准图组（scripts/make_figures.py --stage 1）与评审用
    import xarray as xr
    world_id = f"design_{args.design}_forward_{args.history}_0Ma"
    tect = tectonic_layers(frames[-1], raster, grid, rot)       # U34：构造图层（标准图组的区域窗口构造叠合图）
    ds = xr.Dataset({"elevation": (("lat", "lon"), E_eng.astype("float32"), {"units": "m"}),
                     "land": (("lat", "lon"), (E_eng > 0).astype("float32"), {"description": "陆地（高程 > 0）"}),
                     **{k: (("lat", "lon"), v, {**TECT_ATTRS[k], **({"plate_codes": ",".join(
                         f"{c}={q}" for c, q in plate_codes(hcfg).items())} if k == "tect_plate" else {})})
                        for k, v in tect.items()}},
                    coords={"lat": grid.lat, "lon": grid.lon},
                    attrs={"world_id": world_id, "display_name": f"终图 {args.design} 正向演化 0 Ma",
                           "seed": f"design_{args.design}_history_{args.history}",
                           "method_class_elevation": "procedural_constrained", "code_version": code_version(),
                           "source": "scripts/design_forward.py：300 Ma 初始世界 + 倒推的板块运动正向演化到 0 Ma（L7 地壳引擎，"
                                     "构造尺度高程：Airy + GDH1 + 按起伏的剥蚀，无河流侵蚀）；火山岛按半径进海陆掩膜。"
                                     "注意：几百公里以上的海陆轮廓主要来自终图（user_override）切出的刚体块随板块搬运，"
                                     "不是过程产生的；过程产生的是岛弧、热点岛、碰撞与裂谷带来的变化（评审 stage1_2026-10-25_1 第 17 条）。"
                                     "浅海湾（陆表海、陆架海、神怒湾、东湾）的陆壳厚度按终图水深设定（user_override），"
                                     "0 Ma 水深等于设计值不算过程的验证（评审 stage1_2026-10-25_2 第 9 条）；"
                                     "老被动陆缘的陆架陆壳由初始条件设定（initial.margins）；主内海北岸三块微陆块的位置、裂开与碰撞时刻由设计给出；"
                                     "板内挤压复活（中央山系、北部断块山、北方高原、内海南岸山系）的范围、时间与增厚比由设计给出、"
                                     "增厚比按终图高度估定（性质同 user_override），所以这几条山的高度接近终图不算过程的验证（评审 stage1_2026-10-25_3 第 2、10 条）；"
                                     "U34 起，设计的裂开陆缘与走滑拉分板块对（history yaml forward.pull_apart）上，两侧正在分开处的陆—陆重叠"
                                     "按网格对不齐处理、不当碰撞增厚（之前会把张开的陆缘一点点泵厚），300 Myr 里因此没被当成碰撞的陆壳单元共 "
                                     f"约 {run_js['budget'].get('opening_overlap_skipped', 0.0) / 1e4:.0f} 万 km³（评审 stage1_2026-10-25_4 第 13 条）",
                           "grid": grid.name})
    nc_path = out_w / f"{world_id}_{grid.name}.nc"
    ds.to_netcdf(nc_path)
    log(f"写 {nc_path.relative_to(ROOT).as_posix()}（标准图组：scripts/make_figures.py --world 该文件 --stage 1）")
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
    hyp = summary["hypsometry_0Ma"]
    log(f"陆地高程 5/10/25/50/75/90 分位 {hyp['land_elev_m_percentiles']} m（地球除冰盖 31/62/158/373/812/1437）；"
        f"大陆架（0 ~ −200 m）占海洋 {hyp['shelf_fraction_of_ocean']}（地球同分辨率 0.0675）")
    log(f"A 层（终图，1024x512）：{A_fin}")
    log(f"陆壳体积变化 {summary['volume_change'] * 100:.2f}%；引擎事件 {ev_counts}")
    if not args.no_figures:
        log("出图")
        figures(args, design, hcfg, grid, b, ids, info, mesh, raster, frames, E_eng, E_fin, cmp_, m_eng, f_eng,
                A_eng, A_fin, run_js, v0, hs_cmp, vframes)
    log(f"完成，用时 {time.perf_counter() - t_start:.0f} s")
    return 0


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def figures(args, design, hcfg, grid, b, ids, info, mesh, raster, frames, E_eng, E_fin, cmp_, m_eng, f_eng,
            A_eng, A_fin, run_js, v0=None, hs_cmp=None, vframes=None):
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
        show(ax, raster(f["elev"]).astype(float) - f.get("sea_level", 0.0))
        own, _ = raster_at(h, blocks, ids, grid, float(t), tgt)
        for pid, v in blocks.items():
            ax.contour(tgt.lon, tgt.lat, (own == pid).astype(float), levels=[0.5], colors=[v["color"]],
                       linewidths=0.8, transform=PC)
        ax.set_title(f"{t} Ma（全球海平面 {f.get('sea_level', 0.0):+.0f} m）", fontsize=11)
    fig.suptitle("古地理序列：底图 = 引擎正向演化（构造尺度高程，相对当时的海平面），彩色线 = U23 刚体块在同一时刻的轮廓（只倒推运动的设计位置）",
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
    panels = [("a", "陆壳厚度（km）"), ("b", "最近一次构造改造的时间（Ma；碰撞 = 暖色，俯冲弧 = 冷色，板内挤压复活 = 紫色）"),
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
            for code, cm in ((ORO_COLL, "autumn"), (ORO_ARC, "winter"), (ORO_REACT, "Purples_r")):
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

    # ---- 07 岛弧：终图的岛弧（灰）vs 引擎 0 Ma 的弧火山（U28） ----
    if v0 is not None and (v0["hotspot"] == "ARC").any():
        from mythic.tectonics.volcanism import ISLAND
        GEO = ccrs.Geodetic()
        arc = v0["hotspot"] == "ARC"

        def draw_arcs(ax, ms_scale=1.0, labels=False):
            for isl in design.get("islands", []):
                if isl["type"] == "arc":
                    q = np.asarray(isl["line"], float)
                    ax.plot(q[:, 0], q[:, 1], "-", color="#7f7f7f", lw=2.0, alpha=0.7, transform=GEO, zorder=3)
                    if labels:
                        m = q[len(q) // 2]
                        ax.text(m[0], m[1], isl["id"], transform=PC, fontsize=8, color="#444444", path_effects=halo,
                                zorder=8, clip_on=True)
            s = arc & (v0["kind"] != ISLAND)
            ax.scatter(v0["lon"][s], v0["lat"][s], s=5 * ms_scale, c="#253494", transform=PC, zorder=4, linewidths=0)
            s = arc & (v0["kind"] == ISLAND)
            ax.scatter(v0["lon"][s], v0["lat"][s], s=(8 + 2.0 * v0["radius_km"][s]) * ms_scale, marker="^", c="#fd8d3c",
                       edgecolors="k", linewidths=0.4, transform=PC, zorder=6)

        fig = plt.figure(figsize=(20, 19))
        ax = fig.add_axes([0.02, 0.50, 0.96, 0.44], projection=proj)
        show(ax, E_eng)
        draw_arcs(ax, 0.8)
        ax.set_title("全球：灰线 = 终图岛弧；橙三角 = 引擎 0 Ma 的弧火山岛（大小 ∝ 半径），深蓝点 = 水下的弧火山", fontsize=11)
        regions = [("东亚—东南群岛", (95, 180, -50, 60)), ("主内海", (-25, 60, 25, 52)), ("远方大陆北缘", (-130, -60, 0, 40))]
        for j, (nm, (x0, x1, y0, y1)) in enumerate(regions):
            ax = fig.add_axes([0.02 + j * 0.33, 0.03, 0.30, 0.40], projection=PC)
            ax.set_extent((x0, x1, y0, y1), crs=PC)
            ax.imshow(shaded_elevation(E_eng, grid.pixel_km_equator, vert_exag=3.0, land=E_eng > 0)[..., :3],
                      extent=(-180, 180, -90, 90), transform=PC, origin="upper", interpolation="nearest")
            draw_arcs(ax, 1.3, labels=True)
            ax.gridlines(draw_labels=True, color="#888888", lw=0.3)
            ax.set_title(nm, fontsize=12)
        fig.text(0.5, 0.985, "U28 岛弧火山岛：终图 vs 正向演化 0 Ma", fontsize=15, ha="center", va="top")
        stamp(fig)
        fig.savefig(out / "07_arcs.png", dpi=105)
        plt.close(fig)

    # ---- 08 岛弧系统的形成：几个时刻的板块归属、弧地壳与弧火山岛（U28 第二轮） ----
    if hcfg.get("arcs") and vframes:
        from mythic.tectonics.crust import OCEAN
        from mythic.tectonics.volcanism import ISLAND
        names = {v: k for k, v in plate_codes(hcfg).items()}
        label = {int(bk["id"]): bk["name"] for bk in hcfg["blocks"]}
        label.update({int(p["id"]): p["name"] for p in (hcfg.get("ocean") or {}).get("plates", [])})
        label.update({int(p["id"]): p["name"] for p in hcfg["arcs"].get("plates", [])})
        if (hcfg.get("ocean") or {}).get("background"):
            label[int(hcfg["ocean"]["background"]["id"])] = "外大洋板块"
        times = np.array([f["t"] for f in frames])
        rows = [("西太平洋：菲律宾海式板块（50 Ma 起）", (105, 170, -5, 40), [50, 30, 10, 0]),
                ("东南群岛—南洋：微陆块所在板块与南洋陆块的洋壳（60 Ma 起）", (95, 185, -45, 8), [60, 30, 10, 0]),
                ("加勒比式板块（50 Ma 起）", (-100, -45, -2, 30), [50, 30, 10, 0])]
        pal = matplotlib.colormaps["tab20"]
        allq = sorted(names)
        pcol = {q: pal(i % 20) for i, q in enumerate(allq)}
        fig, axs = plt.subplots(3, 4, figsize=(22, 15.5))
        for r, (title, (x0, x1, y0, y1), ts) in enumerate(rows):
            for cidx, tt in enumerate(ts):
                ax = axs[r, cidx]
                k = int(np.argmin(np.abs(times - tt)))
                f = frames[k]
                P = raster(f["plate"]).astype(int)
                C = raster(f["ctype"])
                T = raster(f["T"]).astype(float)
                ny, nx = P.shape
                lo = np.linspace(-180, 180, nx, endpoint=False) + 180 / nx
                la = np.linspace(90, -90, ny, endpoint=False) - 90 / ny
                sx, sy = (lo >= x0) & (lo <= x1), (la >= y0) & (la <= y1)
                Ps = P[np.ix_(sy, sx)]
                img = np.ones(Ps.shape + (3,))
                for q in np.unique(Ps):
                    img[Ps == q] = np.asarray(pcol.get(int(q), (0.6, 0.6, 0.6, 1)))[:3] * 0.55 + 0.45
                ext = [lo[sx][0], lo[sx][-1], la[sy][-1], la[sy][0]]
                ax.imshow(img, extent=ext, interpolation="nearest", aspect="auto")
                ax.contour(lo[sx], la[sy], C[np.ix_(sy, sx)], levels=[0.5], colors="k", linewidths=0.9)
                yy, xx = np.nonzero(((C == OCEAN) & (T >= 12.0))[np.ix_(sy, sx)])
                ax.scatter(lo[sx][xx], la[sy][yy], s=1.2, c="#d7301f", linewidths=0, zorder=3)
                v = vframes[k]
                s = (v["hotspot"] == "ARC") & (v["kind"] == ISLAND)
                ax.scatter(v["lon"][s], v["lat"][s], marker="^", s=16, c="#fd8d3c", edgecolors="k", linewidths=0.3, zorder=5)
                for q in np.unique(Ps):
                    yy, xx = np.nonzero(Ps == q)
                    if yy.size > 120:
                        ax.text(lo[sx][int(np.median(xx))], la[sy][int(np.median(yy))], label.get(int(q), names.get(int(q), str(q))),
                                fontsize=8.5, ha="center", path_effects=halo, zorder=7)
                if tt == 0:
                    ax.contour(lon, lat, Lf.astype(float), levels=[0.5], colors="#1a9850", linewidths=0.9, zorder=4)
                    for isl in design.get("islands", []):
                        if isl["type"] == "arc":
                            q = np.asarray(isl["line"], float)
                            ax.plot(q[:, 0], q[:, 1], "-", color="#6a3d9a", lw=1.4, zorder=6)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.grid(alpha=0.3)
                ax.set_title(f"{title if cidx == 0 else ''}{chr(10) if cidx == 0 else ''}{times[k]:g} Ma", fontsize=10)
        fig.text(0.5, 0.995, "U28 岛弧系统的形成（第二轮起）：浅色块 = 板块；黑线 = 陆壳边界；红点 = 增厚的洋型弧地壳（≥ 12 km）；"
                 "橙三角 = 弧火山岛；0 Ma 绿线 = 终图陆地、紫线 = 终图岛弧", fontsize=12, ha="center", va="top")
        fig.tight_layout(rect=(0, 0.015, 1, 0.975))
        stamp(fig, " · 板块边界与运动由设计设定（history yaml arcs）")
        fig.savefig(out / "08_arc_systems.png", dpi=95)
        plt.close(fig)

    # ---- 09 东岸大岛（台湾式）与吕宋式近景：终图、引擎 0 Ma 高程、引擎 0 Ma 地壳厚度（U28 第三轮） ----
    if hcfg.get("arcs"):
        x0, x1, y0, y1 = 110.0, 128.0, 6.0, 28.0
        sx, sy = (lon >= x0) & (lon <= x1), (lat >= y0) & (lat <= y1)
        ext = [lon[sx][0], lon[sx][-1], lat[sy][-1], lat[sy][0]]
        T0 = raster(frames[-1]["T"]).astype(float)
        P0 = raster(frames[-1]["plate"]).astype(int)
        fig, axs = plt.subplots(1, 3, figsize=(21, 9.6))
        panels = [("(a) 终图（设计，示意高程 visual_only）；红线 = 引擎 0 Ma 海岸", E_fin, Lf, E_eng > 0),
                  ("(b) 引擎 0 Ma 高程（构造尺度）；绿线 = 终图海岸", E_eng, E_eng > 0, Lf)]
        for ax, (title, E, land, other) in zip(axs[:2], panels):
            rgb = shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=land)[..., :3]
            ax.imshow(rgb[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto")
            ax.contour(lon[sx], lat[sy], other[np.ix_(sy, sx)].astype(float), levels=[0.5],
                       colors="#d7301f" if other is not Lf else "#1a9850", linewidths=1.0)
            ax.set_title(title, fontsize=11)
        ax = axs[2]
        im = ax.imshow(T0[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto", cmap="magma_r", vmin=5, vmax=65)
        Ps = P0[np.ix_(sy, sx)]
        bnd = np.zeros(Ps.shape, bool)
        bnd[:, 1:] |= Ps[:, 1:] != Ps[:, :-1]
        bnd[1:, :] |= Ps[1:, :] != Ps[:-1, :]
        yy, xx = np.nonzero(bnd)
        ax.scatter(lon[sx][xx], lat[sy][yy], s=0.6, c="#2c7fb8", linewidths=0)
        ax.contour(lon[sx], lat[sy], Lf[np.ix_(sy, sx)].astype(float), levels=[0.5], colors="#1a9850", linewidths=1.0)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label="地壳厚度 km（32.4 km 以上露出海面）")
        ax.set_title("(c) 引擎 0 Ma 地壳厚度；蓝点 = 板块边界，绿线 = 终图海岸", fontsize=11)
        for ax in axs:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.grid(alpha=0.3)
            for isl in design.get("islands", []):
                if isl["id"] in ("I3", "I18"):
                    q = np.asarray(isl["line"], float)
                    ax.plot(q[:, 0], q[:, 1], "--", color="#6a3d9a", lw=1.2)
                    ax.text(q[0, 0] + 0.4, q[0, 1], isl["name"], fontsize=9, color="#4a1486", path_effects=halo)
        fig.text(0.5, 0.995, "U28 第三轮 东岸大岛（台湾式弧—陆碰撞）与吕宋式东南弧北岛（挤压型岛弧 + 老岛弧基底）近景；"
                 "紫虚线 = 终图岛屿脊线", fontsize=13, ha="center", va="top")
        fig.tight_layout(rect=(0, 0.02, 1, 0.965))
        stamp(fig, " · 陆架陆壳、老岛弧基底、挤压型俯冲带由设计设定（history yaml initial.margins、arcs.basement、arcs.subduction）")
        fig.savefig(out / "09_taiwan_luzon.png", dpi=100)
        plt.close(fig)

    # ---- 10 地幔柱：动力地形穹隆与大火成岩省近景（U29） ----
    if hcfg.get("plumes"):
        wins = [("南方高原 PL2、火山隆起 PL6 与南方火山高地 M22", (-22.0, 16.0, -36.0, 32.0)),
                ("东南火山高原 PL4（埃塞俄比亚式）", (12.0, 40.0, -3.0, 21.0)),
                ("北方火山岛 I9", (-64.0, -32.0, 56.0, 71.0)),
                ("南大洋高原（凯尔盖朗式）", (58.0, 86.0, -56.0, -37.0))]
        dyn = raster(frames[-1]["dyn"]).astype(float)
        fig, axs = plt.subplots(2, 4, figsize=(24, 13))
        for cidx, (title, (x0, x1, y0, y1)) in enumerate(wins):
            sx, sy = (lon >= x0) & (lon <= x1), (lat >= y0) & (lat <= y1)
            ext = [lon[sx][0], lon[sx][-1], lat[sy][-1], lat[sy][0]]
            for r, (E, land, other, lab) in enumerate(((E_fin, Lf, E_eng > 0, "终图（示意高程）"),
                                                       (E_eng, E_eng > 0, Lf, "引擎 0 Ma（构造尺度 + 动力地形）"))):
                ax = axs[r, cidx]
                rgb = shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=land)[..., :3]
                ax.imshow(rgb[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto")
                ax.contour(lon[sx], lat[sy], other[np.ix_(sy, sx)].astype(float), levels=[0.5],
                           colors="#d7301f" if r == 0 else "#1a9850", linewidths=0.9)
                if r == 1 and np.nanmax(dyn[np.ix_(sy, sx)]) > 100:
                    cs = ax.contour(lon[sx], lat[sy], dyn[np.ix_(sy, sx)], levels=[200, 500, 800], colors="#6a3d9a",
                                    linewidths=0.8, linestyles="--")
                    ax.clabel(cs, fmt="%d m", fontsize=7)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.grid(alpha=0.3)
                ax.set_title(f"{lab}\n{title}", fontsize=10)
        fig.text(0.5, 0.995, "U29 地幔柱：动力地形穹隆（紫虚线 = 引擎的动力地形等值线）与大火成岩省；"
                 "上排终图（红线 = 引擎 0 Ma 海岸），下排引擎 0 Ma（绿线 = 终图海岸）", fontsize=13, ha="center", va="top")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        stamp(fig, " · 地幔柱位置、时间与规模由设计设定（history yaml plumes）")
        fig.savefig(out / "10_plumes.png", dpi=95)
        plt.close(fig)

    # ---- 11 拉分盆地、裂谷与陆表海近景（U30） ----
    if (hcfg.get("forward") or {}).get("pull_apart") or hcfg.get("rifts"):
        wins = [("洲 C、D 之间的走滑断层谷、死海式洼地与南裂谷海", (8.0, 30.0, -37.0, 6.0)),
                ("南方地盾裂谷 M10（地堑、两肩、轴部隆起）", (-9.0, 17.0, -29.0, 13.0)),
                ("南方陆表海 G17 与中央盆地", (-37.0, -7.0, -12.0, 8.0)),
                ("神怒湾 G7 与主内海北岸", (6.0, 28.0, 33.0, 49.0))]
        dyn = raster(frames[-1]["dyn"]).astype(float)
        fig, axs = plt.subplots(2, 4, figsize=(24, 13))
        for cidx, (title, (x0, x1, y0, y1)) in enumerate(wins):
            sx, sy = (lon >= x0) & (lon <= x1), (lat >= y0) & (lat <= y1)
            ext = [lon[sx][0], lon[sx][-1], lat[sy][-1], lat[sy][0]]
            for r, (E, land, other, lab) in enumerate(((E_fin, Lf, E_eng > 0, "终图（示意高程）"),
                                                       (E_eng, E_eng > 0, Lf, "引擎 0 Ma"))):
                ax = axs[r, cidx]
                rgb = shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=land)[..., :3]
                ax.imshow(rgb[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto")
                ax.contour(lon[sx], lat[sy], other[np.ix_(sy, sx)].astype(float), levels=[0.5],
                           colors="#d7301f" if r == 0 else "#1a9850", linewidths=0.9)
                if r == 1:
                    cs = ax.contour(lon[sx], lat[sy], E[np.ix_(sy, sx)], levels=[-200], colors="#08519c",
                                    linewidths=0.7, linestyles=":")
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.grid(alpha=0.3)
                ax.set_title(f"{lab}\n{title}", fontsize=10)
        fig.text(0.5, 0.995, "U30 拉分盆地、裂谷与陆表海：上排终图（红线 = 引擎 0 Ma 海岸），下排引擎 0 Ma（绿线 = 终图海岸，"
                 "蓝点线 = −200 m 等深线，陆架与陆表海在它内侧）", fontsize=13, ha="center", va="top")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        stamp(fig, " · 拉分陆壳、裂谷、浅海湾的陆壳由设计设定（history yaml forward.pull_apart、rifts、initial.margins）")
        fig.savefig(out / "11_pullapart_rift_shelf_seas.png", dpi=95)
        plt.close(fig)

    # ---- 12 全球海平面与被淹没的陆壳（U30） ----
    if all("sea_level" in f for f in frames):
        tt = np.array([f["t"] for f in frames])
        sl = np.array([f["sea_level"] for f in frames])
        fl = np.array([float(mesh.area_km2[(f["ctype"] == CONT) & (f["elev"] < f["sea_level"])].sum()) / 1e6 for f in frames])
        fig, ax = plt.subplots(figsize=(13, 5.6))
        ax.plot(tt, sl, "-o", color="#08519c", ms=3.5, label="全球海平面（m，相对 0 Ma）")
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_xlim(tt.max(), 0)
        ax.set_xlabel("Ma")
        ax.set_ylabel("海平面 m", color="#08519c")
        ax2 = ax.twinx()
        ax2.plot(tt, fl, "-s", color="#b15928", ms=3, label="海平面以下的陆壳面积（百万 km²：陆架、陆表海、被淹的陆缘）")
        ax2.set_ylabel("百万 km²", color="#b15928")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title("U30 全球海平面：海水体积不变，按每一帧的洋底水深（随洋底年龄）与陆地面积解出海平面，再乘海水加载的均衡改正 0.69",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        stamp(fig, " · method_class：parameterized_physics（洋盆容积—海平面，Airy 均衡）")
        fig.savefig(out / "12_sea_level.png", dpi=100)
        plt.close(fig)

    # ---- 14 西口弧形地体（直布罗陀—阿尔沃兰式后撤弧，U35）：终图、引擎 0 Ma、地壳厚度与板块边界，地体沿走廊西移的古地理 ----
    if "ALB" in info:
        x0, x1, y0, y1 = -26.0, 6.0, 30.0, 44.0
        sx, sy = (lon >= x0) & (lon <= x1), (lat >= y0) & (lat <= y1)
        ext = [lon[sx][0], lon[sx][-1], lat[sy][-1], lat[sy][0]]
        alb_id = int(info["ALB"]["id"])
        fig = plt.figure(figsize=(24, 14))
        top = [fig.add_axes((0.02 + k * 0.33, 0.53, 0.3, 0.4)) for k in range(3)]
        bot = [fig.add_axes((0.02 + k * 0.245, 0.06, 0.23, 0.38)) for k in range(4)]
        for ax, (title, E, land, other) in zip(top[:2], (
                ("(a) 终图（设计，示意高程 visual_only）；红线 = 引擎 0 Ma 海岸", E_fin, Lf, E_eng > 0),
                ("(b) 引擎 0 Ma 高程；绿线 = 终图海岸，橙三角 = 西口弧形地体上的弧火山（实心 = 露出海面）", E_eng, E_eng > 0, Lf))):
            rgb = shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=land)[..., :3]
            ax.imshow(rgb[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto")
            ax.contour(lon[sx], lat[sy], other[np.ix_(sy, sx)].astype(float), levels=[0.5],
                       colors="#d7301f" if other is not Lf else "#1a9850", linewidths=1.0)
            ax.set_title(title, fontsize=10)
        v0 = vframes[-1] if vframes else None
        if v0 is not None and len(v0["lon"]):
            m = np.asarray(v0["plate"]) == alb_id
            isl = m & (np.asarray(v0["summit_m"], float) > 0)
            top[1].scatter(np.asarray(v0["lon"])[m & ~isl], np.asarray(v0["lat"])[m & ~isl], marker="^", s=28,
                           facecolors="none", edgecolors="#e6550d", linewidths=1.0)
            top[1].scatter(np.asarray(v0["lon"])[isl], np.asarray(v0["lat"])[isl], marker="^", s=36, color="#e6550d")
        T0 = raster(frames[-1]["T"]).astype(float)
        P0 = raster(frames[-1]["plate"]).astype(int)
        ax = top[2]
        im = ax.imshow(T0[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto", cmap="magma_r", vmin=5, vmax=65)
        Ps = P0[np.ix_(sy, sx)]
        bnd = np.zeros(Ps.shape, bool)
        bnd[:, 1:] |= Ps[:, 1:] != Ps[:, :-1]
        bnd[1:, :] |= Ps[1:, :] != Ps[:-1, :]
        yy, xx = np.nonzero(bnd)
        ax.scatter(lon[sx][xx], lat[sy][yy], s=0.8, c="#2c7fb8", linewidths=0)
        ax.contour(lon[sx], lat[sy], (Ps == alb_id).astype(float), levels=[0.5], colors="#6a3d9a", linewidths=1.4)
        ax.contour(lon[sx], lat[sy], Lf[np.ix_(sy, sx)].astype(float), levels=[0.5], colors="#1a9850", linewidths=0.9)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label="地壳厚度 km")
        ax.set_title("(c) 引擎 0 Ma 地壳厚度；蓝点 = 板块边界，紫线 = 西口弧形地体所在板块，绿线 = 终图海岸", fontsize=10)
        for ax in top:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.grid(alpha=0.3)
            for key in ("mountains", "arcs", "gulfs", "straits"):
                for f in design.get(key, []) or []:
                    if f["id"] in ("M31", "R17", "T1", "G28", "R13"):
                        q = np.asarray(f["line"], float)
                        ax.plot(q[:, 0], q[:, 1], "--", color="#6a3d9a", lw=1.1)
                        ax.text(q[-1, 0] + 0.2, q[-1, 1], f["id"], fontsize=9, color="#4a1486", path_effects=halo)
        # 下排：地体沿走廊西移（引擎高程 + 刚体块轮廓，地幔参照系）
        area = grid.cell_area_km2
        blocks = {}
        for code, blk in info.items():
            msk = ids == blk["id"]
            blocks[blk["id"]] = {"code": code, "color": blk["color"],
                                 "centroid": unit((grid.xyz[msk] * area[msk][:, None]).sum(0))}
        h = build_history(hcfg, {k: v["centroid"] for k, v in blocks.items()})
        tgt = LatLonGrid(1440, 720)
        for ax, t in zip(bot, (38.0, 30.0, 20.0, 10.0)):
            f = min(frames, key=lambda fr: abs(fr["t"] - t))
            E = raster(f["elev"]).astype(float) - f.get("sea_level", 0.0)
            rgb = shaded_elevation(E, grid.pixel_km_equator, vert_exag=3.0, land=E > 0)[..., :3]
            ax.imshow(rgb[np.ix_(sy, sx)], extent=ext, interpolation="nearest", aspect="auto")
            own, _ = raster_at(h, blocks, ids, grid, float(f["t"]), tgt)
            tsx, tsy = (tgt.lon >= x0) & (tgt.lon <= x1), (tgt.lat >= y0) & (tgt.lat <= y1)
            for code in ("ALB", "WT", "A", "C", "ADR"):
                if code in info:
                    pid = info[code]["id"]
                    ax.contour(tgt.lon[tsx], tgt.lat[tsy], (own[np.ix_(tsy, tsx)] == pid).astype(float), levels=[0.5],
                               colors=[info[code]["color"]], linewidths=1.6 if code == "ALB" else 0.8)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.grid(alpha=0.3)
            ax.set_title(f"{f['t']:g} Ma：引擎高程，彩线 = 刚体块（粗紫 = 西口弧形地体）", fontsize=10)
        fig.text(0.5, 0.995, "U35 西口弧形地体（直布罗陀—阿尔沃兰式后撤弧）：38 Ma 起从西段地体南缘（西北湾 G28 一带）被拖出，沿走廊西移约 1250 km，"
                 "楔进洲 A 与洲 C 之间；紫虚线 = 终图的西海峡 T1、弧形山系 M31、火山岛群 R17、西北湾 G28、西部群岛 R13",
                 fontsize=12, ha="center", va="top")
        stamp(fig, " · 地体的位置与运动由设计设定（history yaml blocks 的 ALB、keys、arcs.regions、arcs.subduction）；"
                   "火山与地壳增厚由引擎按俯冲带规则产生")
        fig.savefig(out / "14_west_arc.png", dpi=90)
        plt.close(fig)

if __name__ == "__main__":
    raise SystemExit(main())
