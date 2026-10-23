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
from mythic.tectonics.geometry import R_KM, km_from_chord, smoothstep, unit, xyz_to_lonlat  # noqa: E402
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
    for mid in oo["ids"]:
        p, _ = densify(mts[mid]["line"], 20.0)
        dd, _ = cKDTree(p).query(grid.xyz.reshape(-1, 3))
        dk = km_from_chord(dd).reshape(grid.shape)
        add = float(oo["add_km"]) * np.exp(-(dk / float(oo["half_width_km"])) ** 2)
        T = np.where(ids > 0, T + add, T)
    pls = {p["id"]: p for p in design.get("plateaus", [])}
    for pid in ini["tablelands"]["ids"]:
        m = lonlat_mask(grid, pls[pid]["poly"]) & (ids > 0)
        dk = dist_to_mask_km(grid, ~m, m)
        T[m] += float(ini["tablelands"]["add_km"]) * smoothstep(dk[m] / float(ini["tablelands"]["ramp_km"]))
    bas = {p["id"]: p for p in design.get("basins", [])}
    for bid in ini["basins"]["ids"]:
        m = lonlat_mask(grid, bas[bid]["poly"]) & (ids > 0)
        dk = dist_to_mask_km(grid, ~m, m)
        T[m] -= float(ini["basins"]["sub_km"]) * smoothstep(dk[m] / float(ini["basins"]["ramp_km"]))
    # 被挤掉的陆缘：贴着下行块处厚、往外减薄（超伸展陆缘）
    T_ext = np.zeros(grid.shape)
    for pid in id_code:
        m = ext == pid
        if m.any():
            dk = dist_to_mask_km(grid, ids == pid, m)
            far, near = float(ini["ext_T_far_km"]), float(ini["ext_T_near_km"])
            T_ext[m] = far + (near - far) * np.exp(-dk[m] / float(ini["ext_taper_km"]))
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
        T_ext = np.where(ext == pid, T_ext * f, T_ext)
    vb_rows = volume_balance(hcfg, design, grid, b, ext_masks, T_ext)
    for r in vb_rows:
        log(f"  体积账 {r['down']} <- {r['up']}（{r['t']} Ma）：补回陆缘 {r['consumed_margin_Mkm2']} 百万 km²，"
            f"{r['supply_1e6_km3']}×10⁶ km³" + (f"；终图上要垫出的 {'、'.join(r['builds'])} 共 {r['target_area_Mkm2']} 百万 km²，"
                                               f"需要超额地壳 {r['target_excess_1e6_km3']}×10⁶ km³" if "builds" in r else ""))
    # 省份：每块（本体 + 被挤掉的陆缘）泊松撒点、就近划分；古陆核省份最老，往外变年轻，被挤掉的陆缘最年轻
    sep = float(ini["province_sep_km"])
    prov = np.zeros(grid.shape, int)
    prov_ext = np.zeros(grid.shape, int)
    tf = np.zeros(grid.shape)
    tf_ext = np.zeros(grid.shape)
    provinces = []
    for pid in id_code:
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
            provinces.append({"id": base + k + 1, "block": id_code[pid], "age_ma": round(age), "kind": kind})
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
    oc_cfg = hcfg.get("ocean")
    if oc_cfg:                              # 洋壳归属按俯冲设计（U25），见 history yaml 的 ocean 一节
        rules = [{"down": [code_id[x] for x in r["down"]], "over": [code_id[x] for x in r["over"]]}
                 for r in oc_cfg.get("subduction", [])]
        bg = int(oc_cfg["background"]["id"])
        ops = [{**op, "near": code_id[op["near"]]} if op.get("near") else dict(op) for op in oc_cfg.get("plates", [])]
        c.plate = assign_ocean_plates(mesh, c.plate.astype(np.int64), c.ctype == CONT, rules,
                                      float(oc_cfg["forearc_km"]), float(oc_cfg["corridor_km"]), bg,
                                      float(oc_cfg.get("between_cos", -0.5)), ops).astype(c.plate.dtype)
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
        "volume_balance": vb_rows,
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
        figures(args, design, grid, b, ids, ext, info, id_code, mesh, c, owner, kind, elev, ext_info, ext_masks)
    log(f"完成，用时 {time.perf_counter() - t_start:.0f} s")
    return 0


def figures(args, design, grid, b, ids, ext, info, id_code, mesh, c, owner, kind, elev, ext_info, ext_masks):
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
