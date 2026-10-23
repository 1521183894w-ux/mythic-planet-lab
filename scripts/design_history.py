"""倒推终图的板块运动：刚体块划分、关键帧 -> 旋转文件、古地理序列与审计（PROGRESS U22）。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\design_history.py --design v2 --history v2

输入：config/design/final_map_<design>.yaml（终图）、config/design/history_<history>.yaml（板块划分、关键帧、事件）。
输出：
- data/worlds/design_<design>/history_<history>.rot（PLATES4 旋转文件，进 Git）与 history_<history>.json（审计结果，进 Git）；
- figures/design/<design>/history/：01 刚体块划分、02 古地理序列、03 速度、04 事件时间表、05 碰撞时的空隙与重叠。
审计：各块绝对速度与相对父板块的速度（cm/yr，按 1 Myr 差分）、块与块的重叠面积、热点轨迹检验。
method_class：procedural_constrained（运动由设计设定）。只倒推运动，不倒推地壳过程。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from design_final_map import apply_options, build  # noqa: E402
from mythic.figures.world import code_version  # noqa: E402
from mythic.metrics.grid import LatLonGrid  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from mythic.tectonics.geometry import R_KM, km_from_chord, lonlat_to_xyz, rotmat, unit, xyz_to_lonlat  # noqa: E402
from mythic.tectonics.reverse import PlateHistory, consumed_margin, pose_from_ops  # noqa: E402

PANEL_TIMES = [300, 250, 220, 180, 150, 120, 90, 60, 35, 0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inside(poly, lon, lat) -> np.ndarray:
    return MPath(np.asarray(poly, float)).contains_points(np.c_[lon.ravel(), lat.ravel()]).reshape(lon.shape)


def assign_blocks(hist_cfg: dict, grid: LatLonGrid, b: dict) -> tuple[np.ndarray, dict]:
    """每个陆地格点归到一个刚体块（编号）；0 = 海，-1 = 新生地壳（不参与倒推）。"""
    land, cont, fid = b["land"], b["cont"], b["fid"]
    LO, LA = np.meshgrid(grid.lon, grid.lat)
    out = np.zeros(grid.shape, int)
    out[land] = -1
    info = {}
    for blk in hist_cfg["blocks"]:
        m = np.zeros(grid.shape, bool)
        if blk.get("labels"):
            lab = np.isin(cont, blk["labels"]) & (out == -1)
            if blk.get("poly"):
                lab &= inside(blk["poly"], LO, LA)
            m |= lab
        if blk.get("islands"):
            m |= np.isin(fid, blk["islands"]) & land
        out[m] = blk["id"]
        info[blk["code"]] = blk
    return out, info


def _key_entries(hist_cfg: dict, blk: dict) -> dict:
    """关键帧：{时间: {"ops": [...], "frame": 代号或 None}}。值可以直接写操作列表（父板块参照系），
    也可以写 {frame: 代号, ops: [...]}（在另一块板块的现今参照系里给位置，例如裂开之前贴在哪块旁边）。
    follow: {frame: 代号, from: t}：t 之前一直跟着那块板块走（每 5 Myr 自动补一个关键帧，位置同 t 时刻）。"""
    out = {}
    for t, v in (hist_cfg.get("keys", {}).get(blk["code"]) or {}).items():
        out[float(t)] = dict(v) if isinstance(v, dict) else {"ops": v}
    fol = blk.get("follow")
    if fol:
        t1 = float(fol["from"])
        base = out[t1]
        if base.get("frame") != fol["frame"]:
            raise ValueError(f"{blk['code']}：follow 起点 {t1} Ma 的关键帧必须写在 {fol['frame']} 的参照系里")
        for t in np.arange(t1 + 5.0, float(hist_cfg["t0"]) + 0.1, 5.0):
            out.setdefault(float(t), {"ops": base["ops"], "frame": fol["frame"]})
    return out


def topo_order(hist_cfg: dict) -> list:
    blocks = hist_cfg["blocks"]
    code_id = {b["code"]: b["id"] for b in blocks}
    deps = {}
    for b in blocks:
        d = {0 if b["parent"] in (0, "0") else code_id[b["parent"]]}
        d |= {code_id[e["frame"]] for e in _key_entries(hist_cfg, b).values() if e.get("frame")}
        deps[b["id"]] = d
    done, order, pending = {0}, [], list(blocks)
    while pending:
        progressed = False
        for b in list(pending):
            if deps[b["id"]] <= done:
                order.append((b, 0 if b["parent"] in (0, "0") else code_id[b["parent"]]))
                done.add(b["id"])
                pending.remove(b)
                progressed = True
        if not progressed:
            raise ValueError(f"板块层级有环或依赖的板块不存在：{[b['code'] for b in pending]}")
    return order


def build_history(hist_cfg: dict, centroids: dict) -> PlateHistory:
    h = PlateHistory()
    code_id = {b["code"]: b["id"] for b in hist_cfg["blocks"]}
    for blk, par in topo_order(hist_cfg):
        c0 = centroids[blk["id"]]
        keys = [(0.0, np.eye(3))]
        for t, e in sorted(_key_entries(hist_cfg, blk).items()):
            R = pose_from_ops(c0, e["ops"])
            if e.get("frame"):                         # 另一块板块的参照系 -> 父板块参照系
                R_abs = h.absolute(code_id[e["frame"]], t) @ R
                R = (h.absolute(par, t).T @ R_abs) if par else R_abs
            keys.append((t, R))
        # 最后一个关键帧之后保持不动（相对父板块）
        if keys[-1][0] < hist_cfg["t0"]:
            keys.append((float(hist_cfg["t0"]), keys[-1][1]))
        h.add(blk["id"], par, blk["code"], keys)
    oc = hist_cfg.get("ocean") or {}
    t0 = float(hist_cfg["t0"])
    if oc.get("background"):                # 外大洋板块：固定在地幔（热点）参照系里（U25）
        h.add(int(oc["background"]["id"]), 0, "OCEAN", [(0.0, np.eye(3)), (t0, np.eye(3))])
    for op in oc.get("plates", []):         # 单独运动的大洋板块：绕固定欧拉极分段匀速转动（U26、U27）
        # motion：[{from, to, pole: [经, 纬], rate_deg_myr}]（按时间往回排）；没写时用 pole + speed_cm_yr 全程一段
        # （欧拉极离 300 Ma 范围的质心 90°，质心处速度 = speed）
        segs = op.get("motion") or [{"from": 0.0, "to": t0, "pole": op["pole"],
                                     "rate_deg_myr": np.degrees(float(op["speed_cm_yr"]) * 10.0 / R_KM)}]
        R, keys = np.eye(3), [(0.0, np.eye(3))]
        for sg in sorted(segs, key=lambda s: float(s["from"])):
            a, b = float(sg["from"]), float(sg["to"])
            axis, w = lonlat_to_xyz(*sg["pole"]), np.radians(float(sg["rate_deg_myr"]))
            # 现今 -> t：先倒回这一段之前已走过的（R），再绕本段的极倒转 w·(t − a)。关键帧在分段处与每 60 Myr
            ts = sorted(set(np.arange(a, b, 60.0).tolist()[1:] + [b]))
            for t in ts:
                keys.append((float(t), rotmat(axis, -w * (t - a)) @ R))
            R = rotmat(axis, -w * (b - a)) @ R
        h.add(int(op["id"]), 0, op["code"], keys)
    return h


def consumed_margins(hist_cfg: dict, grid: LatLonGrid, ids: np.ndarray, h: PlateHistory, log_fn=None, return_masks=False):
    """各碰撞下行块要补回的陆缘（现今参照系栅格上的块编号，0 = 无）。范围按运动学算（reverse.consumed_margin），
    陆壳部分只取离下行块 continental_km 以内（更远处是大印度式陆缘外侧的洋盆，按洋壳处理，不补陆壳）。"""
    code_id = {b["code"]: b["id"] for b in hist_cfg["blocks"]}
    ini = hist_cfg.get("initial", {})
    ext = np.zeros(grid.shape, int)
    infos, masks = [], []
    for col in hist_cfg.get("collisions", []):
        dn, up = code_id[col["down"]], code_id[col["up"]]
        t_c = float(col["t"])
        R = h.absolute(dn, t_c).T @ h.absolute(up, t_c)
        m = consumed_margin(grid.xyz, ids == dn, ids == up, R, float(col["max_km"])) & (ext == 0)
        full = float(grid.cell_area_km2[m].sum())
        cap = float(col.get("continental_km", ini.get("ext_continental_km", 1e9)))
        if m.any():
            d, _ = cKDTree(grid.xyz[ids == dn]).query(grid.xyz[m])
            keep = np.zeros_like(m)
            keep[m] = km_from_chord(d) <= cap
            m = keep
        # 碰撞之前任何时刻会压到别的块上的陆缘格点不补（按洋壳处理）：否则正向演化会提前碰撞
        trimmed = 0.0
        if m.any():
            Xe = grid.xyz[m]
            bad = np.zeros(Xe.shape[0], bool)
            for t in np.arange(t_c, float(hist_cfg["t0"]) + 0.1, 5.0):
                P = Xe @ h.absolute(dn, t).T
                for other in code_id.values():
                    if other == dn:
                        continue
                    lo, la = xyz_to_lonlat(P @ h.absolute(other, t))
                    i = np.clip(((90.0 - la) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
                    j = np.clip(((lo + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
                    bad |= (ids[i, j] == other) | (ext[i, j] == other)
            if bad.any():
                trimmed = float(grid.cell_area_km2[m][bad].sum())
                keep = np.zeros_like(m)
                keep[m] = ~bad
                m = keep
        ext[m] = dn
        masks.append(m.copy())
        info = {**col, "swept_Mkm2": round(full / 1e6, 2), "continental_Mkm2": round(float(grid.cell_area_km2[m].sum()) / 1e6, 2),
                "continental_km": cap, "trimmed_conflict_Mkm2": round(trimmed / 1e6, 2)}
        infos.append(info)
        if log_fn:
            log_fn(f"  被挤掉的陆缘 {col['down']} <- {col['up']}（{t_c:g} Ma）：扫过 {info['swept_Mkm2']} 百万 km²，"
                   f"补回陆壳 {info['continental_Mkm2']}（离下行块 {cap:g} km 以内；碰撞前会压到别的块而不补的 "
                   f"{info['trimmed_conflict_Mkm2']}）")
    return (ext, infos, masks) if return_masks else (ext, infos)


def raster_at(h: PlateHistory, blocks: dict, ids: np.ndarray, grid: LatLonGrid, t: float, tgt: LatLonGrid):
    """t 时刻各块在目标格网上的覆盖（反向查找：目标格点转回现今位置，看落在哪块里，没有空洞）。
    返回 (块编号, 覆盖次数)。"""
    X = tgt.xyz.reshape(-1, 3)
    owner = np.zeros(X.shape[0], int)
    count = np.zeros(X.shape[0], int)
    for pid in blocks:
        Y = X @ h.absolute(pid, t)                                       # R^T x：回到现今位置
        lon, lat = xyz_to_lonlat(Y)
        i = np.clip(((90.0 - lat) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
        j = np.clip(((lon + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
        hit = ids[i, j] == pid
        owner[hit] = pid
        count[hit] += 1
    return owner.reshape(tgt.shape), count.reshape(tgt.shape)


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
    t0 = int(hcfg["t0"])
    grid = LatLonGrid(1024, 512)
    log("栅格化终图（1024x512）")
    b = build(design, grid)
    ids, info = assign_blocks(hcfg, grid, b)
    area = grid.cell_area_km2
    blocks = {}
    for code, blk in info.items():
        m = ids == blk["id"]
        X = grid.xyz[m]
        w = area[m]
        c = unit((X * w[:, None]).sum(axis=0))
        blocks[blk["id"]] = {"code": code, "name": blk["name"], "color": blk["color"], "mask": m, "centroid": c,
                             "area_km2": float(w.sum())}
        lo, la = xyz_to_lonlat(c)
        log(f"  {code:4s} {blk['name']}：{w.sum() / 1e6:6.2f} 百万 km²，质心 ({float(lo):7.1f}, {float(la):6.1f})")
    juv = float(area[ids == -1].sum())
    log(f"  新生地壳（不参与倒推）：{juv / 1e6:.2f} 百万 km²")
    h = build_history(hcfg, {k: v["centroid"] for k, v in blocks.items()})
    out_w = ROOT / "data" / "worlds" / f"design_{args.design}"
    rot = h.write_rot(out_w / f"history_{args.history}.rot")
    log(f"写旋转文件 {rot.relative_to(ROOT).as_posix()}")

    # ---- 审计 ----
    log("审计：速度")
    rng = np.random.default_rng(0)
    samples = {}
    for pid, v in blocks.items():
        X = grid.xyz[v["mask"]]
        samples[pid] = X[rng.choice(len(X), size=min(400, len(X)), replace=False)]
    times = np.arange(1, t0 + 1)
    code_id = {v["code"]: k for k, v in blocks.items()}
    speed = {}
    for pid, v in blocks.items():
        X = samples[pid]
        absv = np.array([h.velocity_cm_yr(pid, X, float(t)) for t in times])           # (T, n)
        par = h.parent[pid]
        relv = (np.array([h.relative_velocity_cm_yr(pid, par, X, float(t)) for t in times]) if par else absv)
        speed[v["code"]] = {"abs_median": np.median(absv, axis=1), "abs_max": absv.max(axis=1),
                            "rel_median": np.median(relv, axis=1), "rel_max": relv.max(axis=1)}
    ext, ext_info = consumed_margins(hcfg, grid, ids, h, log)
    # 被挤掉的陆缘与它的上盘在接触之后重叠是预期的（正在被消耗），不计
    consumed_ok = [(code_id[c["down"]], code_id[c["up"]], float(c["t"])) for c in hcfg.get("collisions", [])]
    log("审计：重叠（360x180，每 1 Myr；块本体 + 补回的陆缘）")
    tgt = LatLonGrid(360, 180)
    tarea = tgt.cell_area_km2
    overlap_t = []
    pair_max = {}
    for t in range(0, t0 + 1):
        # 重叠：逐块覆盖相加；超过 1 的格点是重叠
        X = tgt.xyz.reshape(-1, 3)
        hits, hits_ext = {}, {}
        for pid in blocks:
            Y = X @ h.absolute(pid, float(t))
            lon, lat = xyz_to_lonlat(Y)
            i = np.clip(((90.0 - lat) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
            j = np.clip(((lon + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
            hits[pid] = ((ids[i, j] == pid) | (ext[i, j] == pid)).reshape(tgt.shape)
            hits_ext[pid] = (ext[i, j] == pid).reshape(tgt.shape)
        for dn, up, tc in consumed_ok:
            if t < tc:
                hits[dn] = hits[dn] & ~(hits_ext[dn] & hits[up])
        tot = sum(m.astype(int) for m in hits.values())
        overlap_t.append(float(tarea[tot > 1].sum()))
        keys = list(hits)
        for a in range(len(keys)):
            for c in range(a + 1, len(keys)):
                ov = float(tarea[hits[keys[a]] & hits[keys[c]]].sum())
                if ov > 0:
                    pk = f"{blocks[keys[a]]['code']}-{blocks[keys[c]]['code']}"
                    if ov > pair_max.get(pk, (0, 0))[0]:
                        pair_max[pk] = (ov, t)
    log("审计：热点轨迹")
    hs = []
    for hc in hcfg.get("hotspot_checks", []):
        pid = code_id[hc["plate"]]
        p = lonlat_to_xyz(*hc["point"])
        q = h.positions(pid, p[None, :], float(hc["t"]))[0]
        d = float(np.arccos(np.clip(q @ lonlat_to_xyz(*hc["hotspot"]), -1, 1)) * R_KM)
        lo, la = xyz_to_lonlat(q)
        hs.append({**hc, "position_at_t": [round(float(lo), 2), round(float(la), 2)], "distance_km": round(d)})
        log(f"  {hc['plate']} 上 {hc['point']} 在 {hc['t']} Ma 位于 ({float(lo):.1f}, {float(la):.1f})，"
            f"离热点 {hc['hotspot']} {d:.0f} km")

    def summ(x):
        return {"median": round(float(np.median(x)), 2), "p95": round(float(np.percentile(x, 95)), 2),
                "max": round(float(np.max(x)), 2), "t_of_max": int(times[int(np.argmax(x))])}

    summary = {
        "description": "终图倒推历史的审计（重建：scripts/design_history.py）",
        "design": args.design, "history": args.history, "code_version": code_version(),
        "rotation_file": rot.relative_to(ROOT).as_posix(),
        "blocks": {v["code"]: {"id": k, "name": v["name"], "area_Mkm2": round(v["area_km2"] / 1e6, 2),
                               "parent": blocks[h.parent[k]]["code"] if h.parent[k] else 0}
                   for k, v in blocks.items()},
        "juvenile_Mkm2": round(juv / 1e6, 2),
        "speed_cm_yr": {c: {"abs_median_over_time": summ(s["abs_median"]), "abs_max_point": summ(s["abs_max"]),
                            "rel_median_over_time": summ(s["rel_median"])} for c, s in speed.items()},
        "speed_over_10cm_yr_myr": {c: int((s["abs_median"] > 10).sum()) for c, s in speed.items()},
        "overlap_max_Mkm2_by_pair": {k: {"Mkm2": round(v[0] / 1e6, 2), "t": v[1]} for k, v in
                                     sorted(pair_max.items(), key=lambda kv: -kv[1][0])},
        "hotspot_checks": hs,
        "consumed_margins": ext_info,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_w / f"history_{args.history}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                       encoding="utf-8", newline="\n")
    for c, s in speed.items():
        log(f"  速度 {c:4s} 绝对中位数：全程中位 {np.median(s['abs_median']):.1f}，最大 {s['abs_median'].max():.1f} cm/yr"
            f"（{int(times[np.argmax(s['abs_median'])])} Ma）；相对父板块最大 {s['rel_median'].max():.1f}")
    for k, v in list(summary["overlap_max_Mkm2_by_pair"].items())[:8]:
        log(f"  重叠 {k}：最大 {v['Mkm2']} 百万 km²（{v['t']} Ma）")
    if not args.no_figures:
        log("出图")
        figures(design, hcfg, grid, b, ids, blocks, h, speed, times, overlap_t, args)
    log(f"完成，用时 {time.perf_counter() - t_start:.0f} s")
    return 0


def figures(design, hcfg, grid, b, ids, blocks, h, speed, times, overlap_t, args):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mythic.figures.style import setup_fonts

    setup_fonts()
    out = ROOT / "figures" / "design" / args.design / "history"
    out.mkdir(parents=True, exist_ok=True)
    PC = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    ver = code_version()

    def stamp(fig):
        fig.text(0.01, 0.005, f"倒推历史 {args.history} · config/design/history_{args.history}.yaml · "
                 f"scripts/design_history.py · 代码版本 {ver} · method_class：procedural_constrained（运动由设计设定，"
                 f"只倒推运动、不倒推地壳过程；是许多可能历史中的一种）", fontsize=7.5, color="#555555")

    ocean = np.array([0.85, 0.91, 0.96])
    cols = {pid: np.array(to_rgb(v["color"])) for pid, v in blocks.items()}
    hotspots = [(np.asarray(i["line"], float)[0], i["name"]) for i in design.get("islands", [])
                if i["type"] == "hotspot"]

    def paint(ax, t, tgt):
        owner, cnt = raster_at(h, blocks, ids, grid, float(t), tgt)
        img = np.broadcast_to(ocean, owner.shape + (3,)).copy()
        for pid, c in cols.items():
            img[owner == pid] = c
        img[cnt > 1] = np.array([0.1, 0.1, 0.1])
        if t == 0:                                     # 现今的新生地壳（岛弧、热点岛等）画成灰色
            LO, LA = np.meshgrid(tgt.lon, tgt.lat)
            i = np.clip(((90.0 - LA) / 180.0 * grid.ny).astype(int), 0, grid.ny - 1)
            j = np.clip(((LO + 180.0) / 360.0 * grid.nx).astype(int), 0, grid.nx - 1)
            img[ids[i, j] == -1] = np.array([0.55, 0.55, 0.55])
        ax.imshow(img, extent=(-180, 180, -90, 90), transform=PC, origin="upper", regrid_shape=(1400, 700),
                  interpolation="nearest")
        ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color="#888888", lw=0.3)
        for (lo, la), name in hotspots:
            ax.plot(lo, la, marker="*", ms=7, color="#c0392b", mec="white", mew=0.5, transform=PC)
        for pid, v in blocks.items():
            c = h.positions(pid, v["centroid"][None, :], float(t))[0]
            lo, la = xyz_to_lonlat(c)
            ax.text(float(lo), float(la), v["code"], transform=PC, fontsize=8, ha="center", va="center",
                    weight="bold", color="#1b1b1b", path_effects=halo)

    # ---- 01 刚体块 ----
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.80], projection=ccrs.EqualEarth(central_longitude=30))
    ax.set_global()
    paint(ax, 0, grid)
    for pid, v in blocks.items():
        lo, la = xyz_to_lonlat(v["centroid"])
        ax.text(float(lo), float(la) - 6, f"{v['name']}\n{v['area_km2'] / 1e6:.1f} 百万 km²", transform=PC, fontsize=9,
                ha="center", va="top", path_effects=halo)
    handles = [Patch(fc=v["color"], label=f"{v['code']} {v['name']}") for v in blocks.values()]
    handles += [Patch(fc="#8c8c8c", label="新生地壳（岛弧、热点岛等，不参与倒推）"),
                Line2D([], [], marker="*", ls="none", color="#c0392b", ms=9, label="热点（活动端，地幔里近似固定）")]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, ncol=2, framealpha=0.92)
    fig.text(0.5, 0.97, f"终图 {args.design} 的刚体块划分：沿缝合带、裂谷与岛屿成因切开（倒推历史 {args.history}）",
             fontsize=15, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "01_blocks_0Ma.png", dpi=130)
    plt.close(fig)

    # ---- 02 古地理序列 ----
    tgt = LatLonGrid(720, 360)
    fig = plt.figure(figsize=(22, 15.5))
    for k, t in enumerate(PANEL_TIMES):
        r, c = divmod(k, 3)
        ax = fig.add_axes([0.005 + c * 0.333, 0.715 - r * 0.232, 0.325, 0.215],
                          projection=ccrs.EqualEarth(central_longitude=30))
        ax.set_global()
        paint(ax, t, tgt)
        ax.text(0.02, 0.97, f"{t} Ma", transform=ax.transAxes, fontsize=15, weight="bold", va="top")
    lax = fig.add_axes([0.36, 0.02, 0.62, 0.16])
    lax.axis("off")
    lh = [Patch(fc=v["color"], label=f"{v['code']} {v['name']}") for v in blocks.values()]
    lh += [Patch(fc="#1a1a1a", label="两块重叠（碰撞带被挤掉的地壳，正向演化前补回）"),
           Line2D([], [], marker="*", ls="none", color="#c0392b", ms=10, label="热点（地幔里近似固定，各时刻同一位置）")]
    lax.legend(handles=lh, loc="center", ncol=3, fontsize=10, frameon=False)
    fig.text(0.5, 0.995, "倒推的古地理序列：300 Ma -> 现今（刚体块；等面积投影，中央经线 30°E，绝对参照系）",
             fontsize=16, ha="center", va="top")
    stamp(fig)
    fig.savefig(out / "02_reconstructions.png", dpi=110)
    plt.close(fig)

    # ---- 03 速度 ----
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for pid, v in blocks.items():
        s = speed[v["code"]]
        axs[0].plot(times, s["abs_median"], color=v["color"], lw=1.6, label=v["code"])
        axs[1].plot(times, s["rel_median"], color=v["color"], lw=1.6, label=v["code"])
    for a, ttl in zip(axs, ("绝对速度（块内各点中位数）", "相对父板块的速度（块内各点中位数）")):
        a.axhline(10, color="#c0392b", ls="--", lw=1)
        a.text(298, 10.2, "陆地过半的板块一般不超过约 10 cm/yr（Zahirovic 等 2015）", ha="right", va="bottom",
               fontsize=8.5, color="#c0392b")
        a.set_ylabel("cm/yr")
        a.set_title(ttl, fontsize=11)
        a.grid(alpha=0.3)
    axs[1].set_xlabel("Ma")
    axs[1].invert_xaxis()
    axs[0].legend(ncol=5, fontsize=8.5)
    fig.suptitle("各刚体块的速度（每 1 Myr 差分）", fontsize=14)
    stamp(fig)
    fig.savefig(out / "03_speeds.png", dpi=120)
    plt.close(fig)

    # ---- 04 事件时间表 ----
    ev = hcfg.get("events", [])
    kc = {"collision": "#c0392b", "rift": "#e67e22", "drift": "#2e86c1", "subduction": "#7d3c98"}
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(ev) + 1.5))
    for k, e in enumerate(ev):
        a, bb = e["t"]
        ax.barh(k, a - bb, left=bb, color=kc.get(e["kind"], "#888888"), height=0.6)
        ax.text(min(a, bb) - 3, k, e["what"], va="center", ha="left", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(t_max := int(hcfg["t0"]), -150)
    ax.set_xticks(range(0, t_max + 1, 25))
    ax.set_xlabel("Ma")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(fc=c, label=n) for n, c in
                       zip(("碰撞", "裂谷 / 裂离", "漂移", "俯冲"), kc.values())], loc="upper right", fontsize=9)
    ax.set_title(f"倒推历史 {args.history} 的事件时间表（右侧为现今）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "04_timeline.png", dpi=120)
    plt.close(fig)

    # ---- 05 重叠面积随时间 ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(0, int(hcfg["t0"]) + 1), np.array(overlap_t) / 1e6, color="#1b1b1b")
    ax.invert_xaxis()
    ax.set_xlabel("Ma")
    ax.set_ylabel("百万 km²")
    ax.set_title("刚体块之间的重叠面积（碰撞带被挤掉的地壳；倒推正确时只在碰撞前后出现）", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_overlap.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
