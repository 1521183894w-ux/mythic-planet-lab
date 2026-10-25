"""U24：补回碰撞挤掉的地壳，构建终图倒推历史的 300 Ma 初始世界（PROGRESS U22、docs/tectonics_method.md §4）。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\design_initial_world.py --design v2 --history v2

输入：终图 config/design/final_map_<design>.yaml、倒推历史 config/design/history_<history>.yaml（刚体块、关键帧、
collisions、initial）、旋转文件 data/worlds/design_<design>/history_<history>.rot（先跑 scripts/design_history.py）。
输出：
- .cache/tectonics/design/initial_<design>_<history>_L<level>.pkl：引擎可直接用的 InitialWorld（不进 Git，可重算）；
- data/worlds/design_<design>/initial_<history>.json：面积、厚度、高程等统计（进 Git）；
- figures/design/<design>/initial/：01 要补回的陆缘（现今位置）、02 300 Ma 初始世界（高程、厚度、省份年龄、板块与洋底年龄）。
method_class：procedural_constrained（初始条件由倒推设计设定）。
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from design_final_map import apply_options, build, densify, feature_mask, lonlat_mask  # noqa: E402
from design_final_map import elevation as design_elevation  # noqa: E402
from design_history import assign_blocks, build_history, consumed_margins  # noqa: E402
from mythic.figures.world import code_version  # noqa: E402
from mythic.metrics.grid import LatLonGrid  # noqa: E402
from mythic.tectonics.crust import CONT, Params, elevation  # noqa: E402
from mythic.tectonics.geometry import (R_KM, chord_from_km, km_from_chord, lonlat_to_xyz, smoothstep, unit,  # noqa: E402
                                       xyz_to_lonlat)
from mythic.tectonics.kinematics import RotationFileKinematics  # noqa: E402
from mythic.tectonics.mesh import CACHE, load_mesh  # noqa: E402
from mythic.tectonics.prehistory import InitialWorld  # noqa: E402
from mythic.tectonics.reverse import assign_ocean_plates, fill_ocean, initial_crust  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def excess_thickness_km(elev_m: np.ndarray, p: Params | None = None) -> np.ndarray:
    """终图高度按 Airy 均衡换算成高出正常陆壳的厚度（km），封顶于引擎的厚度上限（与 crust.elevation 同一组常数）。"""
    p = p or Params()
    e_max = p.e_cont_ref_m + (p.T_max_km - p.T_cont) * p.iso_m_per_km
    return np.clip((np.clip(elev_m, p.e_cont_ref_m, e_max) - p.e_cont_ref_m) / p.iso_m_per_km, 0.0, None)


def old_orogen_add_km(oo: dict, hcfg: dict, t0: float) -> float:
    """300 Ma 以前的古老造山带在 t0 时刻比正常陆壳多出的厚度（km）。
    写了 add_km 就直接用（U24–U30）；写了 formed_ma 与 h0_m（U31），就按正向演化用的剥蚀规律（erosion_law = "relief"）
    从造山结束时的超高 h0 剥蚀到 t0：孤立山脉的起伏 ≈ 超高，dh/dt = −a·h − b·h²（a = iso·bg，b = iso·K），
    h(t) = a·h0·e^(−at) / (a + b·h0·(1 − e^(−at)))。不论 h0 多高，几亿年后都收敛到约 1/(b·t)，即"老山的持久高度"。"""
    if "add_km" in oo:
        return float(oo["add_km"])
    p = Params(**hcfg.get("forward", {}).get("engine_params", {}))
    if p.erosion_law != "relief" or p.ero_n != 2.0:
        raise ValueError("old_orogens.formed_ma 只用于 erosion_law = relief、ero_n = 2（解析解）")
    iso = p.iso_m_per_km / 1000.0
    a, b = iso * p.ero_bg_per_myr, iso * p.ero_K_km_myr
    t = float(oo["formed_ma"]) - t0
    h0 = float(oo["h0_m"]) / 1000.0
    h = a * h0 * np.exp(-a * t) / (a + b * h0 * (1.0 - np.exp(-a * t)))
    return float(h / iso)


def province_relief(mesh, c, owner: np.ndarray, kind: np.ndarray, cfg: dict, seed: int,
                    keep_out: np.ndarray | None = None, prov_keys: dict | None = None) -> dict:
    """U32：陆壳按省份给厚度起伏（原地修改 c.T，返回统计）。
    每个古老地壳省份（前史式的泊松划分，约 700 km 一个）形成于不同的构造事件，继承了不同的地壳厚度与岩石圈状态；地球上稳定
    大陆的高程因此有几百米的起伏（克拉通内的沉降盆地与隆起、台地与低地）。每个省份的高程偏移从正态分布里抽（mean_m、sd_m，
    截在 [min_m, max_m]），按 Airy 换成厚度；在网格上同一块内的相邻节点之间平均 smooth_iter 次（约 150 km 的过渡，
    省份多边形的直边不留下来，评审 stage1_2026-10-25_1 第 15 条）；离本块边缘 ramp_km 以内渐弱到 0（现今海岸不动）；
    降低后的厚度不低于 T_floor_km（内陆不低于约 +65 m，不凭空出现海）。只加在块本体的陆壳上，补回的陆缘与陆架不加；
    厚度低于 min_base_T_km 的（设计的盆地、海岸减薄带）也不加，免得设计盆地被再压一次、压到下限。
    method_class：procedural_constrained（结构跟着继承的省份走，幅度按地球稳定大陆的高程分布定）。"""
    rng = np.random.default_rng(seed)
    p0 = Params()
    body = (kind == 1) & (c.ctype == CONT) & (c.T >= float(cfg.get("min_base_T_km", 0.0)))   # 设计改低过的（盆地、海岸减薄带）不再叠加
    provs = np.unique(c.prov[body])
    if prov_keys:     # U33：每个省份按 (种子, 块编号, 块内序号) 单独抽，加减别的块不影响它
        draw = np.array([np.random.default_rng([seed, *prov_keys[int(q)]]).normal(float(cfg["mean_m"]), float(cfg["sd_m"]))
                         for q in provs])
    else:
        draw = rng.normal(float(cfg["mean_m"]), float(cfg["sd_m"]), provs.size)
    off = np.clip(draw, float(cfg["min_m"]), float(cfg["max_m"]))
    lut = np.zeros(int(c.prov.max()) + 1)
    lut[provs] = off
    d = np.where(body, lut[c.prov], 0.0)
    i, j = mesh.edges[:, 0], mesh.edges[:, 1]
    same = body[i] & body[j] & (owner[i] == owner[j])
    i, j = i[same], j[same]
    for _ in range(int(cfg["smooth_iter"])):
        s, n = d.copy(), np.ones(mesh.n)
        np.add.at(s, i, d[j])
        np.add.at(s, j, d[i])
        np.add.at(n, i, 1.0)
        np.add.at(n, j, 1.0)
        d = np.where(body, s / n, 0.0)
    if keep_out is not None and keep_out.any():
        # 终图设计了山系、高原、盆地的地方不叠加（它们的高度由各自的过程决定；第一版在北方高原抽到低值，73 m / 终图 1500）：
        # 权重按同样的平滑过渡到 0
        wk = keep_out.astype(float)
        for _ in range(int(cfg["smooth_iter"])):
            s_, n_ = wk.copy(), np.ones(mesh.n)
            np.add.at(s_, i, wk[j])
            np.add.at(s_, j, wk[i])
            np.add.at(n_, i, 1.0)
            np.add.at(n_, j, 1.0)
            wk = np.where(body, s_ / n_, wk)
        d = d * (1.0 - np.clip(wk, 0.0, 1.0))
    dk, _ = cKDTree(mesh.xyz[~body]).query(mesh.xyz[body])
    ramp = np.zeros(mesh.n)
    ramp[body] = smoothstep(np.maximum(km_from_chord(dk) - mesh.h_km, 0.0) / float(cfg["ramp_km"]))   # 贴海岸的一圈为 0
    dT = d * ramp / p0.iso_m_per_km
    T_new = c.T + dT
    low = dT < 0
    T_new[low] = np.maximum(T_new[low], np.minimum(c.T[low], float(cfg["T_floor_km"])))
    applied = T_new - c.T
    c.T = T_new
    A = mesh.area_km2
    return {"provinces": int(provs.size), "offset_m_draw_sd": round(float(off.std()), 1),
            "applied_m_percentiles": {str(q): round(float(np.percentile(applied[body] * p0.iso_m_per_km, q)))
                                      for q in (5, 25, 50, 75, 95)},
            "volume_change_1e6_km3": round(float((applied * A).sum()) / 1e6, 2)}


def volume_balance(hcfg, design, grid, b, ext_masks, T_ext) -> list:
    """碰撞的体积账（U26）：补回的陆缘（下行块被挤掉的陆壳）有多少体积；写了 volume_balance 的碰撞，
    再算终图上它垫出的高原与山系需要多少超额地壳（Airy 均衡由终图高度反算，乘面积）。"""
    area = grid.cell_area_km2
    rows, need = [], None
    for col, m in zip(hcfg.get("collisions", []), ext_masks):
        row = {"down": col["down"], "up": col["up"], "t": col["t"],
               "consumed_margin_Mkm2": round(float(area[m].sum()) / 1e6, 2),
               "supply_1e6_km3": round(float((area[m] * T_ext[m]).sum()) / 1e6, 1)}
        vb = col.get("volume_balance")
        if vb:
            if need is None:
                need = excess_thickness_km(design_elevation(design, grid, b))
            tm = feature_mask(design, grid, b["land"], vb["builds"])
            row.update({"builds": vb["builds"], "target_area_Mkm2": round(float(area[tm].sum()) / 1e6, 2),
                        "target_excess_1e6_km3": round(float((area[tm] * need[tm]).sum()) / 1e6, 1)})
        rows.append(row)
    return rows


def shelf_margins(ini: dict, grid: LatLonGrid, ids: np.ndarray, ext: np.ndarray, code_id: dict, h, t0: float,
                  b: dict | None = None, step_myr: float = 5.0, design: dict | None = None) -> tuple[list, list]:
    """陆架下的减薄陆壳（U28 第三轮，history yaml 的 initial.margins）：现今在海平面以下、属于某块的被动陆缘，按该块的陆缘补进
    初始世界（写进 ext，与被挤掉的陆缘同一套省份与减薄）。范围二选一：poly + block（多边形里的海与没划块的新生地壳岛，归 block）；
    或 gulfs（U30：终图海湾的范围，每格归最近的陆块，要给 b = design_final_map.build 的结果）。300 Ma 以来任何时刻会压到
    别的块（或别的块补回的陆缘）上的格点不补，免得提前碰撞。原地修改 ext，返回 (每条的统计, 每条的掩膜)。"""
    infos, masks = [], []
    others = list(code_id.values())
    id_code = {v: k for k, v in code_id.items()}
    for mg in ini.get("margins", []):
        if mg.get("auto"):
            # U32：300 Ma 以前就有的被动陆缘（老陆缘）的陆架与陆坡。现今海岸外 shelf_km + slope_km 以内的海，归最近的陆块；
            # 只占到对岸（别的块或新生地壳岛）距离的 gap_frac（两岸的陆架不相接，海峡不被堵）；离设计的活动陆缘要素（俯冲型山系、
            # 岛弧）active_km 以内的不补（活动陆缘的陆架窄，而且在弧前）；exclude_gulfs 的海湾不补（残留洋盆、弧后盆地、裂谷海，
            # 以及 U30 查明会被碰撞抬起的海湾）。300 Ma 以来裂开的陆缘（共轭陆缘）在 300 Ma 时与对岸贴在一起，下面的冲突检查会去掉，
            # 由正向演化在裂开时填（forward.pull_apart 的 knots）
            cand = ~b["land"] & (ids <= 0) & (ext == 0)
            for gid in mg.get("exclude_gulfs", []):
                cand &= ~b["sea_masks"][gid]
            Xc = grid.xyz[cand]
            reach = float(mg["shelf_km"]) + float(mg["slope_km"])
            bids = sorted(code_id.values())
            D = np.full((Xc.shape[0], len(bids) + 1), np.inf)
            near = np.zeros((Xc.shape[0], len(bids), 3))     # 每块离格点最近的海岸格点（判断这段海岸是不是活动陆缘）
            for col, q in enumerate(bids):
                Xq = grid.xyz[ids == q]
                d, kq = cKDTree(Xq).query(Xc, distance_upper_bound=chord_from_km(4.0 * reach))
                D[:, col] = km_from_chord(np.where(np.isfinite(d), d, 2.0))
                near[:, col] = Xq[np.minimum(kq, Xq.shape[0] - 1)]
            juv = b["land"] & (ids <= 0)                       # 新生地壳岛（岛弧、热点岛）也算对岸
            if juv.any():
                d, _ = cKDTree(grid.xyz[juv]).query(Xc, distance_upper_bound=chord_from_km(4.0 * reach))
                D[:, -1] = km_from_chord(np.where(np.isfinite(d), d, 2.0))
            o = np.argsort(D[:, :-1], axis=1)[:, 0]
            d_own = D[np.arange(Xc.shape[0]), o]
            D_other = D.copy()
            D_other[np.arange(Xc.shape[0]), o] = np.inf
            ok = (d_own <= reach) & (d_own < float(mg["gap_frac"]) * D_other.min(axis=1))
            coast = near[np.arange(Xc.shape[0]), o]           # 这片陆架所属的海岸格点
            act = []
            mts = {m_["id"]: m_ for m_ in design.get("mountains", [])}
            for fid in mg.get("active", []):
                act.append(densify(mts[fid]["line"], 20.0)[0])
            if mg.get("active_arcs", True):
                for isl in design.get("islands", []):
                    if isl.get("type") == "arc":
                        act.append(densify(isl["line"], 20.0)[0])
            if act:
                # 按所属海岸离活动要素的距离判断（一段海岸是活动陆缘，它外面整片陆架、陆坡都不补）。第一次按格点自己的距离判断，
                # 离岸较远的陆坡格点超出 active_km，在远方大陆西岸外 150–230 km 处留下一串孤立的陆壳碎片，海沟被推离海岸，
                # 弧增厚的峰值落在不增厚的陆架上（远方大陆西岸山系 4150 -> 3431 m）
                d, _ = cKDTree(np.concatenate(act)).query(coast)
                ok &= km_from_chord(d) > float(mg["active_km"])
            apts = list(mg.get("active_points", []))
            if mg.get("active_points_file") and (ROOT / mg["active_points_file"]).exists():
                apts += yaml.safe_load((ROOT / mg["active_points_file"]).read_text(encoding="utf-8")).get("points") or []
            for lo_, la_, r_ in apts:
                # 按上一次正向运行实测的活动陆缘：自动补的陆架在 0 Ma 被抬出海面的地方（俯冲上盘的弧前增厚、下行一侧撞进海沟），
                # 见 scripts/design_forward.py 的 auto_shelf_emerged
                # 实测点取在露出的陆架格点上：格点本身或它所属的海岸任一在半径内都不补
                xp = lonlat_to_xyz(lo_, la_)
                ok &= ((km_from_chord(np.linalg.norm(Xc - xp, axis=1)) > float(r_))
                       & (km_from_chord(np.linalg.norm(coast - xp, axis=1)) > float(r_)))
            own = np.zeros(grid.shape, int)
            ci = np.flatnonzero(cand.ravel())
            own.ravel()[ci[ok]] = np.array(bids)[o[ok]]
            cand = own > 0
        elif "gulfs" in mg:
            cand = np.zeros(grid.shape, bool)
            for gid in mg["gulfs"]:
                cand |= b["sea_masks"][gid]
            cand &= ~b["land"] & (ids <= 0) & (ext == 0)
            blk = ids > 0
            own = np.zeros(grid.shape, int)
            if mg.get("block"):          # U33：指定归属（神怒湾整片是亚得里亚式地体的前陆海，不按最近的块分给上盘）
                own[cand] = code_id[mg["block"]]
            elif cand.any():
                _, kk = cKDTree(grid.xyz[blk]).query(grid.xyz[cand])
                own[cand] = ids[blk][kk]
        else:
            cand = lonlat_mask(grid, mg["poly"]) & (ids <= 0) & (ext == 0)
            own = np.where(cand, code_id[mg["block"]], 0)
        keep_all = np.zeros(grid.shape, bool)
        trimmed = 0.0
        owners = {}
        for pid in np.unique(own[cand]):
            m = cand & (own == pid)
            Xe = grid.xyz[m]
            bad = np.zeros(Xe.shape[0], bool)
            for t in np.arange(0.0, t0 + 0.1, step_myr):
                P = Xe @ h.absolute(int(pid), t).T
                for other in others:
                    if other == pid:
                        continue
                    lo, la = xyz_to_lonlat(P @ h.absolute(other, t))
                    i = np.clip(((90.0 - la) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
                    j = np.clip(((lo + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
                    bad |= (ids[i, j] == other) | (ext[i, j] == other)
            trimmed += float(grid.cell_area_km2[m][bad].sum())
            keep = np.zeros_like(m)
            keep[m] = ~bad
            ext[keep] = pid
            keep_all |= keep
            owners[id_code[int(pid)]] = round(float(grid.cell_area_km2[keep].sum()) / 1e6, 3)
        masks.append(keep_all)
        infos.append({**{k: v for k, v in mg.items() if k not in ("poly", "names")}, "owners_Mkm2": owners,
                      "area_Mkm2": round(float(grid.cell_area_km2[keep_all].sum()) / 1e6, 3),
                      "trimmed_conflict_Mkm2": round(trimmed / 1e6, 3)})
    return infos, masks


def dist_to_mask_km(grid: LatLonGrid, mask: np.ndarray, query: np.ndarray) -> np.ndarray:
    """query 格点到 mask 格点的大圆距离（km）。"""
    out = np.full(grid.shape, np.inf)
    if mask.any() and query.any():
        d, _ = cKDTree(grid.xyz[mask]).query(grid.xyz[query])
        out[query] = km_from_chord(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="v2")
    ap.add_argument("--history", default="v2")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    design = apply_options(yaml.safe_load((ROOT / "config" / "design" / f"final_map_{args.design}.yaml")
                                          .read_text(encoding="utf-8")), set())
    design["_ver"] = args.design
    hcfg = yaml.safe_load((ROOT / "config" / "design" / f"history_{args.history}.yaml").read_text(encoding="utf-8"))
    ini = hcfg["initial"]
    t0 = float(ini["t0"])
    rng = np.random.default_rng(int(ini.get("seed", 1)))
    grid = LatLonGrid(1024, 512)
    log("栅格化终图、划块、倒推历史")
    b = build(design, grid)
    ids, info = assign_blocks(hcfg, grid, b)
    area = grid.cell_area_km2
    code_id = {k: v["id"] for k, v in info.items()}
    id_code = {v: k for k, v in code_id.items()}
    cent = {code_id[k]: unit((grid.xyz[ids == code_id[k]] * area[ids == code_id[k]][:, None]).sum(axis=0))
            for k in info}
    h = build_history(hcfg, cent)
    rot = ROOT / "data" / "worlds" / f"design_{args.design}" / f"history_{args.history}.rot"
    if not rot.exists():
        raise SystemExit(f"缺少旋转文件 {rot}：先跑 scripts/design_history.py")

    # ---- 1. 被挤掉的陆缘（与 design_history.py 的重叠审计同一套计算） ----
    ext, ext_info, ext_masks = consumed_margins(hcfg, grid, ids, h, log, return_masks=True)
    shelf_info, shelf_masks = shelf_margins(ini, grid, ids, ext, code_id, h, t0, b, design=design)
    auto = np.zeros(grid.shape, bool)
    for mg, m in zip(ini.get("margins", []), shelf_masks):
        if mg.get("auto"):
            auto |= m
    (CACHE / "design").mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "design" / f"shelf_auto_{args.design}_{args.history}_{grid.name}.npy", auto)   # 正向演化对照用（不进 Git）
    for s in shelf_info:
        log(f"  陆架下的减薄陆壳 {s['owners_Mkm2']}：{s['area_Mkm2']} 百万 km²（会压到别的块而不补的 {s['trimmed_conflict_Mkm2']}）"
            f"——{s.get('what', '')}")

    # ---- 2. 厚度、省份与年龄（现今参照系栅格；本体与被挤掉的陆缘各一套） ----
    log("厚度、省份与年龄")
    LO, LA = np.meshgrid(grid.lon, grid.lat)
    T = np.where(ids > 0, float(ini["T_cont_km"]), 0.0)
    lith = {x["id"]: x for x in design.get("lithology", [])}
    craton = np.zeros(grid.shape, bool)
    for lid in ini["craton_lithology"]:
        craton |= lonlat_mask(grid, lith[lid]["poly"])
    craton &= ids > 0
    T[craton] = float(ini["craton_T_km"])
    mts = {m["id"]: m for m in design.get("mountains", [])}
    oo = ini["old_orogens"]
    oo_add = old_orogen_add_km(oo, hcfg, float(ini["t0"]))
    for mid in oo["ids"]:
        p, _ = densify(mts[mid]["line"], 20.0)
        dd, _ = cKDTree(p).query(grid.xyz.reshape(-1, 3))
        dk = km_from_chord(dd).reshape(grid.shape)
        add = oo_add * np.exp(-(dk / float(oo["half_width_km"])) ** 2)
        T = np.where(ids > 0, T + add, T)
    pls = {p["id"]: p for p in design.get("plateaus", [])}
    for pid in ini["tablelands"]["ids"]:
        m = lonlat_mask(grid, pls[pid]["poly"]) & (ids > 0)
        dk = dist_to_mask_km(grid, ~m, m)
        T[m] += float(ini["tablelands"]["add_km"]) * smoothstep(dk[m] / float(ini["tablelands"]["ramp_km"]))
    bas = {p["id"]: p for p in design.get("basins", [])}
    p_b = Params()
    for bid in ini["basins"]["ids"]:
        m = lonlat_mask(grid, bas[bid]["poly"]) & (ids > 0)
        dk = dist_to_mask_km(grid, ~m, m)
        if ini["basins"].get("from_design"):
            # U32：下沉量按终图的盆地高程由 Airy 均衡反算（从 400 m 的正常陆面降到设计高程），过渡宽度用终图的 ramp_km
            sub = (p_b.e_cont_ref_m - float(bas[bid]["height_m"])) / p_b.iso_m_per_km
            ramp = float(bas[bid].get("ramp_km", ini["basins"]["ramp_km"]))
        else:
            sub, ramp = float(ini["basins"]["sub_km"]), float(ini["basins"]["ramp_km"])
        T[m] -= sub * smoothstep(dk[m] / ramp)
    # 被挤掉的陆缘：贴着下行块处厚、往外减薄（超伸展陆缘）
    T_ext = np.zeros(grid.shape)
    for pid in id_code:
        m = ext == pid
        if m.any():
            dk = dist_to_mask_km(grid, ids == pid, m)
            far, near = float(ini["ext_T_far_km"]), float(ini["ext_T_near_km"])
            T_ext[m] = far + (near - far) * np.exp(-dk[m] / float(ini["ext_taper_km"]))
    no_thin = np.zeros(grid.shape, bool)
    p0 = Params()
    for mg, m in zip(ini.get("margins", []), shelf_masks):     # 陆架下的减薄陆壳用自己的厚度剖面
        if not m.any():
            continue
        if mg.get("auto"):      # U32：老陆缘的陆架—陆坡，厚度按到本块现今海岸的距离（自带减薄剖面，不再做陆缘减薄）
            T_ext[m] = np.nan
            for pid in np.unique(ext[m]):
                mm = m & (ext == pid)
                dk = dist_to_mask_km(grid, ids == pid, mm)[mm]
                T_ext[mm] = np.interp(dk, [0.0, float(mg["shelf_km"]), float(mg["shelf_km"]) + float(mg["slope_km"])],
                                      [float(mg["T_coast_km"]), float(mg["T_break_km"]), float(mg["T_foot_km"])])
            no_thin |= m
        elif "depth_m" in mg:   # U30：被淹没的陆壳，厚度按设计水深由 Airy 均衡反算（与 crust.elevation 同一组常数）
            T_ext[m] = p0.T_cont + (float(mg["depth_m"]) - p0.e_cont_ref_m) / p0.iso_m_per_km
        else:
            dk = dist_to_mask_km(grid, ids == code_id[mg["block"]], m)
            far, near = float(mg["T_far_km"]), float(mg["T_near_km"])
            T_ext[m] = far + (near - far) * np.exp(-dk[m] / float(mg["taper_km"]))
        if mg.get("edge_thinning", True) is False:
            no_thin |= m
    # 块边缘（本体 + 被挤掉的陆缘合起来算边缘）往海岸减薄：本体是现今陆地，只减薄到 coast_T_km（约 +100 m 的滨海平原），
    # 现今海岸留在海平面附近；被挤掉的陆缘按被动陆缘减薄到 margin_min（大部分在海平面以下）。
    # （U24 曾把本体边缘也减薄到 72%，约 −1 km：U25 第一次正向运行时现今海岸带与窄小陆块整片沉到海下。）
    mk, mmin, cT = float(ini["margin_km"]), float(ini["margin_min"]), float(ini.get("coast_T_km", 0.0))
    for pid in id_code:
        whole = (ids == pid) | (ext == pid)
        dk = dist_to_mask_km(grid, ~whole, whole)
        s = smoothstep(dk / mk)
        f = mmin + (1.0 - mmin) * s
        body = ids == pid
        T = np.where(body & (T > cT), cT + (T - cT) * s, T) if cT > 0 else np.where(body, T * f, T)
        T_ext = np.where((ext == pid) & ~no_thin, T_ext * f, T_ext)    # 海湾里的陆表海不做陆缘减薄（U30，edge_thinning: false）
    vb_rows = volume_balance(hcfg, design, grid, b, ext_masks, T_ext)
    # 体积平衡（U28）：写了保留率 efficiency 的碰撞（补回陆缘的体积有多少留在终图要垫高的范围里，由上一次正向运行实测），
    # 补回陆缘按"需要量 ÷ 保留率"统一放大厚度，上限是正常陆壳厚度
    for col, m, r in zip(hcfg.get("collisions", []), ext_masks, vb_rows):
        eff = (col.get("volume_balance") or {}).get("efficiency")
        if not eff or "target_excess_1e6_km3" not in r or r["supply_1e6_km3"] <= 0:
            continue
        f = r["target_excess_1e6_km3"] / (float(eff) * r["supply_1e6_km3"])
        T_ext[m] = np.minimum(T_ext[m] * f, float(ini["T_cont_km"]))
        r.update({"efficiency": eff, "thickness_scale": round(f, 3),
                  "supply_scaled_1e6_km3": round(float((area[m] * T_ext[m]).sum()) / 1e6, 1),
                  "mean_T_km": round(float((area[m] * T_ext[m]).sum() / area[m].sum()), 1)})
    for r in vb_rows:
        log(f"  体积账 {r['down']} <- {r['up']}（{r['t']} Ma）：补回陆缘 {r['consumed_margin_Mkm2']} 百万 km²，"
            f"{r['supply_1e6_km3']}×10⁶ km³" + (f"；终图上要垫出的 {'、'.join(r['builds'])} 共 {r['target_area_Mkm2']} 百万 km²，"
                                               f"需要超额地壳 {r['target_excess_1e6_km3']}×10⁶ km³" if "builds" in r else "")
            + (f"；按保留率 {r['efficiency']} 厚度放大 {r['thickness_scale']} 倍，补回 {r['supply_scaled_1e6_km3']}×10⁶ km³、"
               f"平均 {r['mean_T_km']} km" if "thickness_scale" in r else ""))
    # 省份：每块（本体 + 被挤掉的陆缘）泊松撒点、就近划分；古陆核省份最老，往外变年轻，被挤掉的陆缘最年轻
    sep = float(ini["province_sep_km"])
    prov = np.zeros(grid.shape, int)
    prov_ext = np.zeros(grid.shape, int)
    tf = np.zeros(grid.shape)
    tf_ext = np.zeros(grid.shape)
    provinces = []
    for pid in id_code:
        # U33：每块用自己的随机数（种子 + 块编号）。之前整个世界共用一串随机数，加一块微陆块就让排在后面的所有块的省份划分、
        # 年龄与省份起伏都重抽一遍（东南增生半岛与苏门答腊式长岛因此连成一片，多出一个亚大陆）
        rng = np.random.default_rng([int(ini.get("seed", 1)), int(pid)])
        whole = (ids == pid) | (ext == pid)
        idx = np.flatnonzero(whole.ravel())
        if idx.size == 0:
            continue
        X = grid.xyz.reshape(-1, 3)[idx]
        order = rng.permutation(idx.size)
        seeds = []
        cos_sep = np.cos(sep / R_KM)
        for k in order:
            if not seeds or (X[seeds] @ X[k]).max() < cos_sep:
                seeds.append(k)
        S = X[seeds]
        _, own = cKDTree(S).query(X)
        s_flat = idx[seeds]
        s_cr = craton.ravel()[s_flat]
        s_ext = (ext.ravel()[s_flat] == pid)
        blk_area = float(area[ids == pid].sum()) / 1e6
        if not s_cr.any() and blk_area >= float(ini["craton_min_block_Mkm2"]):
            dk = dist_to_mask_km(grid, ~whole, whole).ravel()[s_flat]
            s_cr[int(np.argmax(np.where(s_ext, -1, dk)))] = True
        if s_cr.any():
            dcr, _ = cKDTree(S[s_cr]).query(S)
            dcr = km_from_chord(dcr)
        else:
            dcr = np.full(len(seeds), np.inf)
        base = len(provinces)
        for k in range(len(seeds)):
            if s_cr[k]:
                age, kind = float(rng.uniform(2600, 3400)), "craton"
            elif s_ext[k]:
                age, kind = float(rng.uniform(500, 800)), "margin"
            elif np.isfinite(dcr[k]):
                age, kind = float(np.clip(2400 - 0.45 * dcr[k] + rng.normal(0, 150), 700, 2400)), "belt"
            else:
                age, kind = float(rng.uniform(900, 1800)), "belt"
            provinces.append({"id": base + k + 1, "block": id_code[pid], "block_id": int(pid), "local": k,
                              "age_ma": round(age), "kind": kind})
        pv = base + own + 1
        ages = np.array([provinces[i - 1]["age_ma"] for i in pv], float)
        for arr_p, arr_t, sel in ((prov, tf, ids.ravel()[idx] == pid), (prov_ext, tf_ext, ext.ravel()[idx] == pid)):
            fl = idx[sel]
            arr_p.ravel()[fl] = pv[sel]
            arr_t.ravel()[fl] = ages[sel]
    log(f"  省份 {len(provinces)} 个（古陆核 {sum(p['kind'] == 'craton' for p in provinces)}，"
        f"被挤掉的陆缘 {sum(p['kind'] == 'margin' for p in provinces)}）")

    # ---- 3. 300 Ma 的地壳（L7 网格） ----
    mesh = load_mesh(int(ini["mesh_level"]))
    log(f"构建 {t0:g} Ma 地壳（L{mesh.level}，{mesh.n} 个节点）")
    order = [code_id[c] for c in ini["block_order"]]
    c, owner, kind = initial_crust(mesh, h, t0, order, ids, ext,
                                   {"T": T, "t_form": tf, "prov": prov}, {"T": T_ext, "t_form": tf_ext, "prov": prov_ext})
    c = fill_ocean(mesh, c, owner, t0, float(ini["half_spreading_km_myr"]))
    prov_relief = None
    if ini.get("province_relief"):          # U32：陆壳按省份给厚度起伏（克拉通内的盆地与隆起）
        fids = [x["id"] for key in ("plateaus", "basins", "mountains") for x in design.get(key, [])]
        feat = feature_mask(design, grid, b["land"], fids)      # 终图设计的山系、高原、盆地（现今参照系）
        keep_out = np.zeros(mesh.n, bool)
        for pid in np.unique(owner[owner > 0]):
            lo, la = xyz_to_lonlat(mesh.xyz @ h.absolute(int(pid), t0))
            i_ = np.clip(((90.0 - la) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
            j_ = np.clip(((lo + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
            keep_out |= (owner == pid) & feat[i_, j_]
        prov_relief = province_relief(mesh, c, owner, kind, ini["province_relief"], int(ini.get("seed", 1)) + 101,
                                      keep_out=keep_out,
                                      prov_keys={pv["id"]: (pv["block_id"], pv["local"]) for pv in provinces})
        log(f"  省份厚度起伏：{prov_relief}")
    oc_cfg = hcfg.get("ocean")
    if oc_cfg:                              # 洋壳归属按俯冲设计（U25），见 history yaml 的 ocean 一节
        rules = [{"down": [code_id[x] for x in r["down"]], "over": [code_id[x] for x in r["over"]]}
                 for r in oc_cfg.get("subduction", [])]
        bg = int(oc_cfg["background"]["id"])
        ops = [{**op, "near": code_id[op["near"]]} if op.get("near") else dict(op) for op in oc_cfg.get("plates", [])]
        # 陆架下的减薄陆壳（水下的被动陆缘）不参与洋壳划分：第三轮第一、二次运行把它当大陆，东南增生地块名下的洋壳由 10.1 变成
        # 9.3 百万 km²（离谁近、夹在谁之间都变了），连带改了东南群岛一带的洋壳归属，东大岛山脊 1953 -> 1185 m
        shelf_nodes = np.zeros(mesh.n, bool)
        if any(m.any() for m in shelf_masks):
            sr = np.zeros(grid.shape, int)
            for m in shelf_masks:
                sr[m] = ext[m]
            for pid in np.unique(sr[sr > 0]):
                lo, la = xyz_to_lonlat(mesh.xyz @ h.absolute(int(pid), t0))
                i = np.clip(((90.0 - la) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
                j = np.clip(((lo + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
                shelf_nodes |= (sr[i, j] == pid) & (owner == pid) & (kind == 2)
        c.plate = assign_ocean_plates(mesh, c.plate.astype(np.int64), (c.ctype == CONT) & ~shelf_nodes, rules,
                                      float(oc_cfg["forearc_km"]), float(oc_cfg["corridor_km"]), bg,
                                      float(oc_cfg.get("between_cos", -0.5)), ops).astype(c.plate.dtype)
        c.plate[shelf_nodes] = owner[shelf_nodes]
    rift_of = {b["code"]: b["rift_from"]["plate"] for b in hcfg["blocks"] if b.get("rift_from")}
    for blk in hcfg["blocks"]:
        if blk.get("rift_from"):
            # U33：裂开之前是母板块的一部分（地体来源 origin 保留块编号，正向演化到 rift_from.t 时按它分出来）。
            # U35：嵌套时（西口弧形地体 -> 西段地体 -> 洲 C）记在最上一级的母板块名下
            root = blk["rift_from"]["plate"]
            while root in rift_of:
                root = rift_of[root]
            c.plate[c.plate == int(blk["id"])] = int(code_id[root])
        log(f"  洋壳归属：外大洋板块 {mesh.area_km2[c.plate == bg].sum() / 1e6:.0f} 百万 km²；"
            + "".join(f"{op['name']} {mesh.area_km2[c.plate == int(op['id'])].sum() / 1e6:.0f}；" for op in ops)
            + "，".join(f"{id_code[q]} {mesh.area_km2[(c.plate == q) & (c.ctype != CONT)].sum() / 1e6:.0f}"
                       for q in order))
    kin = RotationFileKinematics(rot)
    plates = sorted(int(q) for q in np.unique(c.plate))
    _, omega = kin.step_rotations(plates, int(t0))
    elev = elevation(c, Params())
    A = mesh.area_km2
    cont = c.ctype == CONT
    blocks = []
    for pid in order:
        blocks.append({"id": int(pid), "code": id_code[pid], "name": info[id_code[pid]]["name"],
                       "core_Mkm2": round(float(A[(owner == pid) & (kind == 1)].sum()) / 1e6, 2),
                       "consumed_margin_Mkm2": round(float(A[(owner == pid) & (kind == 2)].sum()) / 1e6, 2),
                       "plate_Mkm2": round(float(A[c.plate == pid].sum()) / 1e6, 1)})
    world = InitialWorld(level=mesh.level, seed=0, t=t0, crust=c, omega={int(k): v for k, v in omega.items()},
                         provinces=provinces, blocks=blocks, events=hcfg.get("events", []), rot_path=str(rot),
                         params={"design": args.design, "history": args.history, **{k: v for k, v in ini.items()}})
    out_c = CACHE / "design"
    out_c.mkdir(parents=True, exist_ok=True)
    pkl = out_c / f"initial_{args.design}_{args.history}_L{mesh.level}.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(world, fh)
    land = elev > 0
    stats = {
        "description": "终图倒推历史的 300 Ma 初始世界（重建：scripts/design_initial_world.py；"
                       "InitialWorld 缓存在 .cache/tectonics/design/，不进 Git）",
        "design": args.design, "history": args.history, "code_version": code_version(), "t0": t0,
        "mesh_level": mesh.level, "rotation_file": rot.relative_to(ROOT).as_posix(),
        "consumed_margins": ext_info,
        "shelf_margins": [{**s, "volume_1e6_km3": round(float((area[m] * T_ext[m]).sum()) / 1e6, 2)}
                          for s, m in zip(shelf_info, shelf_masks)],
        "volume_balance": vb_rows,
        "old_orogen_add_km": round(oo_add, 2),
        "province_relief": prov_relief,
        "continental_crust_Mkm2": round(float(A[cont].sum()) / 1e6, 1),
        "of_which_consumed_margin_Mkm2": round(float(A[kind == 2].sum()) / 1e6, 1),
        "present_rigid_blocks_Mkm2": round(float(area[ids > 0].sum()) / 1e6, 1),
        "land_above_sea_level_Mkm2": round(float(A[land].sum()) / 1e6, 1),
        "ocean_fraction": round(1 - float(A[land].sum()) / float(A.sum()), 3),
        "elevation_m_percentiles_land": {str(p): round(float(np.percentile(elev[land], p))) for p in (5, 50, 95, 99)},
        "continental_thickness_km": {"median": round(float(np.median(c.T[cont])), 1),
                                     "min": round(float(c.T[cont].min()), 1), "max": round(float(c.T[cont].max()), 1)},
        "ocean_age_myr": {"median": round(float(np.median(c.age[~cont])), 1), "max": round(float(c.age[~cont].max()), 1)},
        "n_provinces": len(provinces), "blocks": blocks,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_w = ROOT / "data" / "worlds" / f"design_{args.design}"
    (out_w / f"initial_{args.history}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                                       encoding="utf-8", newline="\n")
    log(f"陆壳 {stats['continental_crust_Mkm2']} 百万 km²（其中被挤掉的陆缘 {stats['of_which_consumed_margin_Mkm2']}；"
        f"现今刚体块 {stats['present_rigid_blocks_Mkm2']}），海平面以上 {stats['land_above_sea_level_Mkm2']}，"
        f"海洋占 {stats['ocean_fraction'] * 100:.1f}%；陆地高程中位 {stats['elevation_m_percentiles_land']['50']} m")
    log(f"写 {pkl.relative_to(ROOT).as_posix()}")
    if not args.no_figures:
        log("出图")
        figures(args, design, grid, b, ids, ext, info, id_code, mesh, c, owner, kind, elev, ext_info, ext_masks,
                list(zip(shelf_info, shelf_masks)))
    log(f"完成，用时 {time.perf_counter() - t_start:.0f} s")
    return 0


def figures(args, design, grid, b, ids, ext, info, id_code, mesh, c, owner, kind, elev, ext_info, ext_masks, shelves=()):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    from mythic.figures.style import setup_fonts, shaded_elevation
    from mythic.tectonics.mesh import Raster

    setup_fonts()
    out = ROOT / "figures" / "design" / args.design / "initial"
    out.mkdir(parents=True, exist_ok=True)
    PC = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    ver = code_version()
    col = {v["id"]: np.array(to_rgb(v["color"])) for v in info.values()}

    def stamp(fig):
        fig.text(0.01, 0.005, f"300 Ma 初始世界 · config/design/history_{args.history}.yaml · scripts/design_initial_world.py · "
                 f"代码版本 {ver} · method_class：procedural_constrained（初始条件由倒推设计设定）", fontsize=7.5, color="#555555")

    # ---- 01 要补回的陆缘（现今位置） ----
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84], projection=ccrs.EqualEarth(central_longitude=30))
    ax.set_global()
    img = np.broadcast_to(np.array([0.86, 0.91, 0.96]), grid.shape + (3,)).copy()
    for pid, cc in col.items():
        img[ids == pid] = 0.45 * cc + 0.55
    img[ids == -1] = 0.8
    for pid, cc in col.items():
        img[ext == pid] = cc
    ax.imshow(img, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(2400, 1200),
              interpolation="nearest")
    ax.contour(grid.lon, grid.lat, b["land"].astype(float), levels=[0.5], colors="#555555", linewidths=0.4,
               transform=PC)
    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
    for e, m in zip(ext_info, ext_masks):          # 每次碰撞单独标注（同一下行块可能有几处）
        if not m.any():
            continue
        v = unit((grid.xyz[m] * grid.cell_area_km2[m][:, None]).sum(0))
        lo, la = xyz_to_lonlat(v)
        ax.text(float(lo), float(la), f"{e['down']} 补回的陆缘\n{e['continental_Mkm2']} 百万 km²\n（{e['t']} Ma 起被 {e['up']} 覆盖）",
                transform=PC, fontsize=9, ha="center", va="center", path_effects=halo)
    for s_, m in shelves:                          # 陆架下的减薄陆壳（U28 第三轮）
        if not m.any():
            continue
        v = unit((grid.xyz[m] * grid.cell_area_km2[m][:, None]).sum(0))
        lo, la = xyz_to_lonlat(v)
        ax.text(float(lo) + 4, float(la), f"{'、'.join(s_['owners_Mkm2'])} 陆架下的减薄陆壳\n{s_['area_Mkm2']} 百万 km²", transform=PC,
                fontsize=8, ha="left", va="center", path_effects=halo)
    hs = [Patch(fc=v["color"], label=f"{k} {v['name']}（深色 = 要补回的陆缘）") for k, v in info.items()]
    ax.legend(handles=hs, loc="lower left", fontsize=8, ncol=2, framealpha=0.92)
    fig.text(0.5, 0.97, "要补回的陆缘（画在现今位置）：碰撞后上盘块扫过、一路走在陆地上就碰到下行块的地方，"
             "原来是下行块的陆缘，正向演化时被俯冲、拼贴进造山带", fontsize=13, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "01_consumed_margins.png", dpi=130)
    plt.close(fig)

    # ---- 02 300 Ma 初始世界 ----
    ras = Raster(mesh, 512)
    E = ras(elev)
    landm = E > 0
    rgb = shaded_elevation(E, ras.grid.pixel_km_equator, vert_exag=3.0, land=landm)[..., :3]
    plate_r = ras(c.plate)
    bnd = np.zeros(plate_r.shape, bool)
    bnd[:, 1:] |= plate_r[:, 1:] != plate_r[:, :-1]
    bnd[1:, :] |= plate_r[1:, :] != plate_r[:-1, :]
    fig = plt.figure(figsize=(22, 13))
    panels = [("a", "300 Ma 示意高程（Airy 均衡 + 洋底年龄水深，引擎的初始状态）"),
              ("b", "陆壳厚度（km）与板块边界"), ("c", "省份年龄（Ma，陆壳）"), ("d", "洋底年龄（Myr）")]
    for k, (tag, title) in enumerate(panels):
        r, cc = divmod(k, 2)
        ax = fig.add_axes([0.01 + cc * 0.495, 0.50 - r * 0.46, 0.48, 0.42], projection=ccrs.EqualEarth(central_longitude=30))
        ax.set_global()
        if tag == "a":
            ax.imshow(rgb, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(1600, 800),
                      interpolation="nearest")
        elif tag == "b":
            Tm = np.ma.masked_where(ras(c.ctype) != CONT, ras(c.T))
            im = ax.imshow(Tm, extent=(-180, 180, -90, 90), transform=PC, origin="upper", cmap="YlOrBr", vmin=20, vmax=45,
                           regrid_shape=(1600, 800), interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        elif tag == "c":
            Am = np.ma.masked_where(ras(c.ctype) != CONT, ras(c.t_form))
            im = ax.imshow(Am, extent=(-180, 180, -90, 90), transform=PC, origin="upper", cmap="viridis_r", vmin=400,
                           vmax=3400, regrid_shape=(1600, 800), interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        else:
            Om = np.ma.masked_where(ras(c.ctype) == CONT, ras(c.age))
            im = ax.imshow(Om, extent=(-180, 180, -90, 90), transform=PC, origin="upper", cmap="RdYlBu", vmin=0, vmax=180,
                           regrid_shape=(1600, 800), interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        if tag in ("b", "d"):
            ys, xs = np.nonzero(bnd)
            ax.plot(ras.grid.lon[xs], ras.grid.lat[ys], ",", color="#1b1b1b", transform=PC)
        ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
        for pid in col:
            sel = (owner == pid) & (kind == 1)
            if sel.any():
                v = unit((mesh.xyz[sel] * mesh.area_km2[sel][:, None]).sum(0))
                lo, la = xyz_to_lonlat(v)
                ax.text(float(lo), float(la), id_code[pid], transform=PC, fontsize=9, weight="bold", ha="center",
                        path_effects=halo)
        ax.set_title(f"({tag}) {title}", fontsize=12)
    fig.text(0.5, 0.995, "300 Ma 初始世界（L7 网格；被挤掉的陆缘已补回，300 Ma 以后形成的山与高原不预先放）",
             fontsize=15, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "02_initial_300Ma.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
