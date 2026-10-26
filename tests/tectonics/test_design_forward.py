"""U25：正向演化与终图对照的辅助函数（scripts/design_forward.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fwd = _load("design_forward", "design_forward.py")


def test_label_wrap_joins_regions_across_the_dateline():
    m = np.zeros((20, 40), bool)
    m[5:9, :3] = True           # 180°E 一侧
    m[6:10, -2:] = True         # 180°W 一侧，与上面隔着 180° 经线相接
    m[15:17, 18:22] = True      # 另一块，不相接
    lab, n = fwd.label_wrap(m)
    assert n == 2
    assert lab[6, 0] == lab[7, -1] != 0
    assert lab[16, 20] not in (0, lab[6, 0])


def test_coast_mask_is_the_outer_ring_of_land():
    land = np.zeros((10, 10), bool)
    land[2:8, 2:8] = True
    c = fwd.coast_mask(land)
    assert c[2, 2] and c[7, 5] and not c[4, 4]
    assert not (c & ~land).any()
    assert c.sum() == 6 * 6 - 4 * 4


def test_excess_thickness_follows_airy_and_engine_cap():
    """U26 体积平衡：终图高度按 Airy 均衡换算成超出 35 km 的厚度，与引擎同一组常数，封顶于 65 km。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.tectonics.crust import Crust, CONT, Params, elevation

    p = Params()
    ex = init.excess_thickness_km(np.array([-500.0, 400.0, 2000.0, 4945.0, 8000.0]))
    assert ex[0] == 0 and ex[1] == 0
    assert abs(ex[3] - (p.T_max_km - p.T_cont)) < 0.01 and ex[4] == ex[3]
    c = Crust(1)                                  # 反过来：引擎按这个厚度算出的高程就是 2000 m
    c.ctype[:] = CONT
    c.T[:] = p.T_cont + ex[2]
    assert abs(float(elevation(c, p)[0]) - 2000.0) < 1.0


def test_feature_mask_unions_polygons_and_ridge_bands():
    fm = _load("design_final_map", "design_final_map.py")
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(360, 180)
    land = np.ones(g.shape, bool)
    design = {"plateaus": [{"id": "P", "poly": [[10, 10], [20, 10], [20, 20], [10, 20]]}],
              "mountains": [{"id": "M", "line": [[-40, 0], [-30, 0]], "width_km": 400}]}
    LO, LA = np.meshgrid(g.lon, g.lat)
    m = fm.feature_mask(design, g, land, ["P", "M"])
    assert m[(LO > 11) & (LO < 19) & (LA > 11) & (LA < 19)].all()
    assert m[(np.abs(LA) < 1) & (LO > -39) & (LO < -31)].all()
    assert not m[(np.abs(LA) > 3) & (LO > -39) & (LO < -31)].any()        # 山脊两侧各 200 km（约 1.8°）以外
    assert not fm.feature_mask(design, g, np.zeros(g.shape, bool), ["P", "M"]).any()


def _hcfg_arcs():
    return {"blocks": [{"code": "B", "id": 101}, {"code": "BSE", "id": 102}],
            "ocean": {"background": {"id": 900}, "plates": [{"id": 920, "code": "PA"}]},
            "arcs": {"plates": [{"id": 930, "code": "PS", "t_init": 50}],
                     "subduction": [{"down": ["BSE"], "over": ["PS"]}, {"down": ["PA"], "over": ["PS"]}],
                     "gaps": [{"pair": ["PS", "PA"], "to": "PS"}]}}


def test_arc_polarity_overrides_the_ocean_plate_default():
    """U28 第二轮：岛弧系统的俯冲方向优先于"大洋板块遇到大陆都下行"（马尼拉式：陆块的洋壳俯冲到新洋板块之下）。"""
    pol = fwd.preset_polarity(_hcfg_arcs())
    assert pol[102 * 1000 + 930] == 102          # 东南增生地块下行（设计）
    assert pol[101 * 1000 + 930] == 930          # 洲 B 与新洋板块：默认新洋板块下行（琉球式）
    assert pol[920 * 1000 + 930] == 920
    gp = fwd.ocean_gap_prefer(_hcfg_arcs())
    assert gp[920 * 1000 + 930] == 930


def test_apply_region_moves_only_ocean_units_inside_the_polygon():
    """U28 第二轮：俯冲起始时把多边形里的洋壳（不论原属哪块板块）划给新板块，陆壳不动。"""
    from mythic.tectonics.crust import CONT, Crust, Params
    from mythic.tectonics.engine import Engine
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.mesh import load_mesh

    mesh = load_mesh(5)
    c = Crust(mesh.n)
    c.plate[:] = np.where(mesh.lon < 20, 1, 2)
    land = (np.abs(mesh.lat) < 5) & (np.abs(mesh.lon - 20) < 5)
    c.ctype[land], c.T[land] = CONT, 35.0
    eng = Engine(mesh, c, Params(), 50.0, {1: np.zeros(3), 2: np.zeros(3)})
    poly = lonlat_to_xyz(np.array([5.0, 35.0, 35.0, 5.0]), np.array([-20.0, -20.0, 20.0, 20.0]))
    moved = fwd.apply_region(eng, 3, poly, np.zeros(3))
    assert set(moved) == {1, 2}
    on3 = eng.alive & (eng.c.plate == 3)
    assert on3.sum() > 0 and (eng.c.ctype[on3] == 0).all()
    assert (eng.alive & (eng.c.ctype == CONT) & (eng.c.plate != 3)).sum() == land.sum()


def test_shelf_margin_fills_sea_in_polygon_and_skips_cells_another_block_crosses():
    """U28 第三轮：陆架下的减薄陆壳（initial.margins）只补多边形里的海与新生地壳岛，归指定的块；
    300 Ma 以来会压到别的块上的格点不补（否则正向演化会提前碰撞）。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.metrics.grid import LatLonGrid
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.reverse import PlateHistory, pose_from_ops

    g = LatLonGrid(360, 180)
    LO, LA = np.meshgrid(g.lon, g.lat)
    ids = np.zeros(g.shape, int)
    ids[(LO > -10) & (LO < 10) & (LA > -20) & (LA < 0)] = 7          # 有陆架的块（不动）
    ids[(LO > 40) & (LO < 50) & (LA > -20) & (LA < 0)] = 8           # 另一块：100 Ma 时在西边 2000 km（约 18°）
    ids[(LO > 12) & (LO < 14) & (LA > -6) & (LA < -4)] = -1          # 陆架上的新生地壳岛（台湾式）
    h = PlateHistory()
    h.add(7, 0, "X", [(0, np.eye(3)), (100, np.eye(3))])
    h.add(8, 0, "Y", [(0, np.eye(3)), (100, pose_from_ops(lonlat_to_xyz(45, -10), [{"move": [2000.0, 270.0]}]))])
    ext = np.zeros(g.shape, int)
    ini = {"margins": [{"block": "X", "poly": [[8, -20], [26, -20], [26, 0], [8, 0]], "T_near_km": 30.0, "T_far_km": 22.0,
                        "taper_km": 300}]}
    info, masks = init.shelf_margins(ini, g, ids, ext, {"X": 7, "Y": 8}, h, 100.0)
    m = masks[0]
    assert (ext[m] == 7).all() and not (m & (ids > 0)).any()          # 只补海与新生地壳，块本体不动
    assert m[(LO > 12) & (LO < 14) & (LA > -6) & (LA < -4)].all()     # 新生地壳岛下面也是陆架
    assert m[(LO > 11) & (LO < 18) & (LA > -18) & (LA < -2)].all()     # 块 Y 走不到的地方补上
    assert not m[(LO > 24) & (LO < 26) & (LA > -18) & (LA < -2)].any()  # 块 Y 100 Ma 时压过的地方不补
    assert info[0]["trimmed_conflict_Mkm2"] > 0


def test_compress_pairs_only_from_flagged_subduction_zones():
    """U28 第三轮：挤压型岛弧只取设计里标了 compress 的俯冲带（马尼拉式），后撤型（伊豆—马里亚纳式）不算。"""
    h = _hcfg_arcs()
    assert fwd.compress_pairs(h) == set()
    h["arcs"]["subduction"][0]["compress"] = True
    assert fwd.compress_pairs(h) == {102 * 1000 + 930}


def test_apply_basement_thickens_only_the_plates_ocean_inside_the_polygon():
    """U28 第三轮：新洋板块继承的老岛弧基底：只加厚该板块多边形里的洋壳（厚度至少 T_km、记为岛弧来源），已经更厚的不变。"""
    from mythic.tectonics.crust import CONT, ORIGIN_ARC, Crust, Params
    from mythic.tectonics.engine import Engine
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.mesh import load_mesh

    mesh = load_mesh(5)
    c = Crust(mesh.n)
    c.plate[:] = np.where(mesh.lon < 20, 1, 2)
    box = (np.abs(mesh.lat) < 15) & (mesh.lon > 5) & (mesh.lon < 35)
    land = box & (mesh.lon > 28)
    c.ctype[land], c.T[land] = CONT, 35.0
    thick = box & (mesh.lon > 22) & (mesh.lon < 26)
    c.T[thick] = 25.0
    eng = Engine(mesh, c, Params(), 50.0, {1: np.zeros(3), 2: np.zeros(3)})
    poly = lonlat_to_xyz(np.array([5.0, 35.0, 35.0, 5.0]), np.array([-15.0, -15.0, 15.0, 15.0]))
    n = fwd.apply_basement(eng, 2, poly, 20.0)
    sel = box & (eng.c.plate == 2) & (eng.c.ctype == 0)
    assert n >= sel.sum() > 0                                          # 多边形边是大圆弧，比经纬度框略大
    assert (eng.c.T[sel & ~thick] == 20.0).all() and (eng.c.T[sel & thick] == 25.0).all()
    assert (eng.c.origin[sel] == ORIGIN_ARC).all()
    assert (eng.c.T[box & (eng.c.plate == 1)] == 7.0).all()          # 别的板块不动
    assert (eng.c.T[land] == 35.0).all()                              # 陆壳不动


def test_build_plumes_reads_swells_and_anchored_lips():
    """U29：地幔柱目录 -> 动力地形穹隆（地幔参照系）与大火成岩省（锚在板块上、按时间先后排）。"""
    h = _hcfg_arcs()
    h["plumes"] = [
        {"id": "P1", "name": "甲", "at": [10.0, 5.0], "active": [30, 0], "swell": {"amp_m": 800, "radius_km": 1000},
         "lips": [{"t": 20, "anchor": "BSE", "at": [10.0, 5.0], "radius_km": 300, "add_km": 5.0}]},
        {"id": "P2", "name": "乙", "lips": [{"t": 60, "line": [[0.0, 0.0], [2.0, 1.0]], "radius_km": 200, "flat_km": 50,
                                              "add_km": 8.0}]}]
    pl = fwd.build_plumes(h)
    assert [s["id"] for s in pl["swells"]] == ["P1"] and pl["swells"][0]["active"] == (30.0, 0.0)
    assert [e["t"] for e in pl["lips"]] == [60.0, 20.0]                 # 先发生的在前
    assert pl["lips"][0]["anchor"] == 0 and pl["lips"][0]["pts0"].shape == (2, 3)
    assert pl["lips"][1]["anchor"] == 102 and pl["lips"][1]["flat_km"] == 0.0


def test_build_plumes_lip_footprint_from_final_map_plateau():
    """U29：大火成岩省的范围可以取终图高原的轮廓与从边缘往里的渐变（与终图示意高程同一画法）。"""
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(360, 180)
    design = {"plateaus": [{"id": "PX", "poly": [[10, 0], [30, 0], [30, 20], [10, 20]], "ramp_km": 300, "meander_km": [0, 0]}]}
    h = _hcfg_arcs()
    h["plumes"] = [{"id": "P3", "name": "丙", "lips": [{"t": 30, "anchor": "BSE", "feature": "PX", "add_km": 10.0}]}]
    e = fwd.build_plumes(h, design, g, np.ones(g.shape, bool))["lips"][0]
    LO, LA = np.meshgrid(g.lon, g.lat)
    W = e["W"]
    assert e["pts0"] is None and e["anchor"] == 102
    assert W[(np.abs(LO - 20) < 2) & (np.abs(LA - 10) < 2)].min() > 0.99      # 中间满额
    assert (W[(LO < 8) | (LO > 32)] == 0).all()                                # 轮廓外为 0
    edge = W[(np.abs(LO - 10.5) < 0.1) & (np.abs(LA - 10) < 0.6)]             # 紧贴西缘的一列格点
    assert 0 < edge.max() < 0.5 and W[(np.abs(LO - 11.5) < 0.1) & (np.abs(LA - 10) < 0.6)].min() > edge.max()  # 往里渐升


def test_gulf_margins_belong_to_the_nearest_block():
    """U30：浅海湾补的陆壳按终图海湾范围，每格归最近的陆块（不补陆地、不补别的块本体）。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.metrics.grid import LatLonGrid
    from mythic.tectonics.reverse import PlateHistory

    g = LatLonGrid(360, 180)
    LO, LA = np.meshgrid(g.lon, g.lat)
    ids = np.zeros(g.shape, int)
    ids[(LO > -20) & (LO < 0) & (np.abs(LA) < 20)] = 7
    ids[(LO > 10) & (LO < 30) & (np.abs(LA) < 20)] = 8
    land = ids > 0
    gulf = (LO > -2) & (LO < 12) & (np.abs(LA) < 5)            # 夹在两块之间的海湾
    b = {"sea_masks": {"G1": gulf}, "land": land & ~gulf}
    ids[gulf] = 0
    h = PlateHistory()
    for pid, code in ((7, "X"), (8, "Y")):
        h.add(pid, 0, code, [(0, np.eye(3)), (100, np.eye(3))])
    ext = np.zeros(g.shape, int)
    info, masks = init.shelf_margins({"margins": [{"gulfs": ["G1"], "depth_m": -80}]}, g, ids, ext, {"X": 7, "Y": 8}, h, 100.0, b)
    m = masks[0]
    assert m.sum() == (gulf & ~(ids > 0)).sum()
    assert (ext[m & (LO < 4)] == 7).all() and (ext[m & (LO > 6)] == 8).all()
    assert set(info[0]["owners_Mkm2"]) == {"X", "Y"}


def test_eustatic_sea_level_rises_when_ocean_floor_is_shallower():
    """U30：海水体积不变，洋底整体变浅时海平面上升（乘海水加载改正 0.69），0 Ma 为 0。"""
    area = np.ones(1000)
    e0 = np.r_[np.full(700, -4000.0), np.linspace(0, 1000, 300)]
    e1 = np.r_[np.full(700, -3700.0), np.linspace(0, 1000, 300)]      # 洋底浅 300 m（年轻洋脊多）
    sl = fwd.eustatic_sea_level([{"elev": e1}, {"elev": e0}], area)
    assert sl[-1] == pytest.approx(0.0, abs=0.5)
    assert 0.69 * 150 < sl[0] < 0.69 * 300                             # 一部分水漫上陆地，升得比 300 m 少


def test_design_rifts_and_pull_apart_pairs():
    """U30：history yaml 的陆内裂谷与走滑拉分板块对。"""
    h = _hcfg_arcs()
    h["rifts"] = [{"id": "R1", "name": "甲", "anchor": "BSE", "t_start": 25, "line": [[0, 0], [0, 5]], "graben_half_km": 40,
                   "beta": 1.1, "flank_amp_m": 900, "flank_decay_km": 60}]
    h["forward"] = {"pull_apart": [{"pair": ["B", "BSE"], "since": 36, "T_narrow_km": 33, "w1_km": 80, "T_wide_km": 20,
                                    "w_max_km": 250}]}
    r = fwd.build_rifts(h)[0]
    assert r["anchor"] == 102 and r["pts0"].shape == (2, 3) and r["swell_amp_m"] == 0.0
    assert fwd.cont_gap_pairs(h) == {101 * 1000 + 102: (33.0, 80.0, 20.0, 250.0)}
    assert fwd.cont_gap_since(h) == {101 * 1000 + 102: 36.0}


def test_pull_apart_knots_holes_and_designed_arc_pairs():
    """U32：forward.pull_apart 的分段剖面（裂开的被动陆缘）与 holes；陆壳弧增厚只用于设计的俯冲带与按设计运动的大洋板块。"""
    h = _hcfg_arcs()
    h["forward"] = {"pull_apart": [{"pair": ["B", "BSE"], "since": 36, "T_narrow_km": 33, "w1_km": 80, "T_wide_km": 20,
                                    "w_max_km": 250, "holes": True},
                                   {"pair": ["BSE", "PA"], "knots": [[0, 32.0], [200, 31.4], [360, 18.0]]}]}
    g = fwd.cont_gap_pairs(h)
    assert g[101 * 1000 + 102] == {"knots": [(0.0, 33.0), (80.0, 33.0), (250.0, 20.0)], "holes": True}
    assert g[102 * 1000 + 920] == {"knots": [(0.0, 32.0), (200.0, 31.4), (360.0, 18.0)], "holes": False}
    from mythic.tectonics.engine import gap_profile
    kw, kT, holes = gap_profile((33.0, 80.0, 20.0, 250.0))          # U30 的四元组照旧
    assert kw.tolist() == [0.0, 80.0, 250.0] and kT.tolist() == [33.0, 33.0, 20.0] and holes is False
    h["forward"]["arc_continental_extra_pairs"] = [["PA", "B"]]
    pairs = fwd.arc_cont_pairs(h)
    assert 102 * 1000 + 930 in pairs and 920 * 1000 + 930 in pairs  # 设计的俯冲带
    assert 101 * 1000 + 920 in pairs                                # 显式列出的大洋板块俯冲带
    assert 102 * 1000 + 920 not in pairs                            # 大洋板块与其他陆块的接触不算
    assert 101 * 1000 + 900 not in pairs and 102 * 1000 + 900 not in pairs   # 固定的外大洋板块不算


def test_province_relief_keeps_coasts_and_floor():
    """U32：省份厚度起伏只加在块本体陆壳上，海岸处为 0、往内陆渐强，降低后不低于 T_floor_km，补回的陆缘不动。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.tectonics.crust import CONT, Crust
    from mythic.tectonics.geometry import lonlat_to_xyz
    from mythic.tectonics.mesh import load_mesh

    mesh = load_mesh(5)
    c = Crust(mesh.n)
    cosd = mesh.xyz @ lonlat_to_xyz(0.0, 0.0)
    body = cosd >= np.cos(np.radians(40.0))
    margin = (cosd >= np.cos(np.radians(45.0))) & ~body
    c.ctype[body | margin] = CONT
    c.T[body] = 35.0
    c.T[margin] = 30.0
    c.prov[body] = np.where(mesh.xyz[body, 0] > 0.9, 1, 2)      # 两个省份
    c.prov[margin] = 3
    owner = np.where(body | margin, 7, 0)
    kind = np.where(body, 1, np.where(margin, 2, 0))
    T0 = c.T.copy()
    cfg = {"mean_m": -150, "sd_m": 400, "min_m": -600, "max_m": 900, "smooth_iter": 3, "ramp_km": 600, "T_floor_km": 33.0}
    st = init.province_relief(mesh, c, owner, kind, cfg, seed=3)
    assert st["provinces"] == 2
    assert (c.T[margin] == T0[margin]).all() and (c.T[~(body | margin)] == T0[~(body | margin)]).all()
    edge = body & np.isin(np.arange(mesh.n), mesh.edges[(body[mesh.edges[:, 0]] != body[mesh.edges[:, 1]])].ravel())
    assert np.abs(c.T[edge] - 35.0).max() < 0.05                 # 海岸处几乎不变
    assert c.T[body].min() >= 33.0 - 1e-9                         # 不低于下限
    assert np.abs(c.T[body] - 35.0).max() > 0.3                   # 内陆有起伏
    # 终图设计了要素的地方（keep_out）不叠加
    c2 = Crust(mesh.n)
    for f in ("ctype", "T", "prov"):
        setattr(c2, f, getattr(c, f).copy())
    c2.T[:] = T0
    ko = body & (cosd > np.cos(np.radians(12.0)))
    init.province_relief(mesh, c2, owner, kind, cfg, seed=3, keep_out=ko)
    core = body & (cosd > np.cos(np.radians(6.0)))
    assert np.abs(c2.T[core] - 35.0).max() < 0.1 * np.abs(c.T[core] - 35.0).max() + 1e-6


def test_auto_shelf_stays_off_the_far_shore_and_active_margins():
    """U32：老陆缘的自动陆架——只补到对岸距离的 gap_frac、离活动陆缘要素 active_km 以内与 active_points 附近不补，
    厚度剖面按到本块海岸的距离（陆架 -> 陆坡）。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.metrics.grid import LatLonGrid
    from mythic.tectonics.reverse import PlateHistory

    g = LatLonGrid(720, 360)
    LO, LA = np.meshgrid(g.lon, g.lat)
    ids = np.zeros(g.shape, int)
    ids[(LO > -20) & (LO < 0) & (LA > -20) & (LA < 20)] = 7          # 块 X
    ids[(LO > 3) & (LO < 20) & (LA > -20) & (LA < 20)] = 8           # 块 Y，东边隔着约 330 km 的海峡
    h = PlateHistory()
    for q, code in ((7, "X"), (8, "Y")):
        h.add(q, 0, code, [(0, np.eye(3)), (100, np.eye(3))])
    ext = np.zeros(g.shape, int)
    b = {"land": ids > 0, "sea_masks": {}}
    design = {"mountains": [{"id": "M9", "line": [[-19, 10], [-19, 20]]}], "islands": []}
    ini = {"margins": [{"auto": True, "shelf_km": 120, "slope_km": 80, "T_coast_km": 32.0, "T_break_km": 31.4,
                        "T_foot_km": 18.0, "gap_frac": 0.4, "active_km": 300, "active": ["M9"],
                        "active_points": [[-10.0, -21.5, 200]]}]}
    info, masks = init.shelf_margins(ini, g, ids, ext, {"X": 7, "Y": 8}, h, 100.0, b, design=design)
    m = masks[0]
    assert not (m & (ids > 0)).any()
    strait = (LO > 0) & (LO < 3) & (np.abs(LA) < 5)
    assert m[strait & (LO < 1.0)].any() and m[strait & (LO > 1.0) & (LO < 2.0)].sum() == 0   # 只占到对岸距离的 40%
    west = (LO > -21.5) & (LO < -20) & (np.abs(LA) < 5)
    assert (ext[west] == 7).all()                                      # 西岸外的陆架归 X
    assert not m[(LO > -20.8) & (LO < -20) & (LA > 12) & (LA < 19)].any()   # 离活动陆缘要素 300 km 内不补
    assert not m[(LO > -11) & (LO < -9) & (LA > -22) & (LA < -21)].any()    # 实测活动陆缘点附近不补
    assert m[(LO > -16) & (LO < -14) & (LA > -22) & (LA < -21)].any()       # 别处的南岸照补


def test_hypsometry_and_emerged_auto_shelf(tmp_path):
    """U32：陆地高程分位、大陆架占海洋比例；自动陆架在 0 Ma 露出海面的片。"""
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(360, 180)
    LO, LA = np.meshgrid(g.lon, g.lat)
    E = np.full(g.shape, -4000.0)
    E[(np.abs(LO) < 20) & (np.abs(LA) < 20)] = 300.0
    E[(np.abs(LO) < 22) & (np.abs(LA) < 22) & (E < 0)] = -100.0          # 一圈陆架
    hy = fwd.hypsometry(g, E)
    assert hy["land_elev_m_percentiles"][3] == 300 and 0 < hy["shelf_fraction_of_ocean"] < 0.1
    sm = (np.abs(LO) < 22) & (np.abs(LA) < 22) & ~((np.abs(LO) < 20) & (np.abs(LA) < 20))
    E2 = E.copy()
    E2[sm & (LO > 20)] = 200.0                                          # 东侧的陆架被抬出海面
    p = tmp_path / "shelf.npy"
    np.save(p, sm)
    r = fwd.auto_shelf_emerged(g, E2, p)
    assert r["emerged_fraction"] > 0.1 and len(r["clusters"]) == 1
    c = r["clusters"][0]
    assert 20 < c["lon"] < 22 and abs(c["lat"]) < 1


def test_write_shelf_active_points_cover_the_emerged_shelf(tmp_path):
    """U32：明显抬起（高出海面 300 m 以上、成片）的自动陆架写成点清单，每个这样的格点都落在某个点的半径内；只露出一点的不算。"""
    import yaml
    from mythic.metrics.grid import LatLonGrid
    from mythic.tectonics.geometry import km_from_chord, lonlat_to_xyz

    g = LatLonGrid(360, 180)
    LO, LA = np.meshgrid(g.lon, g.lat)
    sm = (LO > 0) & (LO < 8) & (np.abs(LA) < 10)
    E = np.where((LO > 2) & (LO < 6) & sm, 800.0, -100.0)
    E[(LO > 0) & (LO < 1) & sm] = 50.0                                 # 只露出一点的一条：不算
    np.save(tmp_path / "s.npy", sm)
    n = fwd.write_shelf_active(tmp_path / "a.yaml", g, E, tmp_path / "s.npy", "test")
    pts = yaml.safe_load((tmp_path / "a.yaml").read_text(encoding="utf-8"))["points"]
    assert n == len(pts) > 1
    P = np.array([lonlat_to_xyz(lo, la) for lo, la, _ in pts])
    em = g.xyz[sm & (E > 300)]
    P_far = km_from_chord(np.min(np.linalg.norm(g.xyz[sm & (E > 0) & (E < 300)][:, None, :] - P[None], axis=2), axis=1))
    assert P_far.min() > 0
    d = km_from_chord(np.min(np.linalg.norm(em[:, None, :] - P[None], axis=2), axis=1))
    assert d.max() <= pts[0][2]


def test_forward_polarity_overrides_designed_and_default_polarity():
    """U33：forward.polarity 只定俯冲方向，优先于 arcs.subduction、ocean.subduction 与"大洋板块遇到大陆都下行"。"""
    h = _hcfg_arcs()
    h["forward"] = {"polarity": [{"down": ["PS"], "over": ["BSE"]}, {"down": ["B"], "over": ["BSE"]}]}
    pol = fwd.preset_polarity(h)
    assert pol[102 * 1000 + 930] == 930          # arcs.subduction 写的是 BSE 下行，被 forward.polarity 改成 PS 下行
    assert pol[101 * 1000 + 102] == 101          # 两块陆块之间没有别的设定时按 forward.polarity
    assert pol[920 * 1000 + 930] == 920          # 其余照旧


def test_boundary_classes_divergent_convergent_transform_slow():
    """U34：0 Ma 边界类型。东半球板块绕地轴向东转：0° 经线处与西半球板块分开（离散），180° 经线处靠近（汇聚）；
    南北两半球以赤道为界、南半球向东转是走滑；转得很慢是"缓慢"。"""
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(64, 32)
    lon = np.broadcast_to(g.lon[None, :], g.shape)
    lat = np.broadcast_to(g.lat[:, None], g.shape)
    w = 5.0 * 10.0 / fwd.R_KM                        # 赤道上 5 cm/yr
    P = np.where(lon > 0, 2, 1)
    cls, rate = fwd.boundary_classes(P, g, {1: np.zeros(3), 2: np.array([0.0, 0.0, w])})
    eq = np.argmin(np.abs(g.lat))
    j0 = np.argmin(np.abs(g.lon))                    # 0° 经线两侧
    assert set(cls[eq, j0 - 1:j0 + 2]) - {0} == {2}
    assert cls[eq, 0] == 1 and cls[eq, -1] == 1      # 180° 经线（环绕）
    assert 4.5 < rate[eq, 0] < 5.1
    assert cls[eq, 10] == 0                          # 板块内部
    P2 = np.where(lat > 0, 1, 2)
    cls2, _ = fwd.boundary_classes(P2, g, {1: np.zeros(3), 2: np.array([0.0, 0.0, w])})
    assert set(np.unique(cls2[cls2 > 0])) == {3}
    cls3, _ = fwd.boundary_classes(P, g, {1: np.zeros(3), 2: np.array([0.0, 0.0, w / 50])})
    assert set(np.unique(cls3[cls3 > 0])) == {4}


@pytest.mark.parametrize("preserve,expected_t", [(False, 35.), (True, 38.)])
def test_terrane_rift_timing_and_nested_family_without_extra_commits(monkeypatch, preserve, expected_t):
    """U35-C2：新方式准时分离并携带未裂子地体；旧配置仍保留原提交节奏和行为。"""
    from types import SimpleNamespace
    from mythic.tectonics.crust import CONT, Crust, Params
    from mythic.tectonics.mesh import load_mesh

    class StationaryKinematics:
        def step_rotations(self, plates, t):
            return ({q: np.eye(3) for q in plates}, {q: np.zeros(3) for q in plates})

    monkeypatch.setattr(fwd, "RotationFileKinematics", lambda _: StationaryKinematics())
    mesh = load_mesh(4)
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.ctype[:] = CONT
    c.T[:] = 35.
    c.origin[:] = 1
    child = (abs(mesh.lon) < 15) & (abs(mesh.lat) < 15)
    grandchild = child & (mesh.lon > 0)
    c.origin[child] = 2
    c.origin[grandchild] = 3
    original_ids = c.uid[child].copy()
    hcfg = {"blocks": [
        {"code": "WT", "id": 1},
        {"code": "ALB", "id": 2, "rift_from": {"plate": "WT", "t": 38, "preserve_frame": preserve}},
        {"code": "CHILD", "id": 3, "rift_from": {"plate": "ALB", "t": 10, "preserve_frame": True}}],
        "arcs": {"subduction": []}}
    eng, frames, _ = fwd.simulate(SimpleNamespace(crust=c, t=40), mesh, Path("unused.rot"), 10,
                                  Params(), {}, arcs_hcfg=hcfg)
    rifts = [e for e in eng.events if e["kind"] == "terrane_rift"]
    assert [(e["t"], e["src"], e["dst"]) for e in rifts] == [(expected_t, 1, 2), (10., 2, 3)]
    assert len(eng.commit_log) == 9
    assert [f["t"] for f in frames] == [40., 30., 20., 10., 0.]
    assert np.all(eng.c.plate[eng.alive & (eng.c.origin == 2)] == 2)
    assert np.all(eng.c.plate[eng.alive & (eng.c.origin == 3)] == 3)
    if preserve:
        np.testing.assert_array_equal(np.sort(eng.c.uid[eng.alive & np.isin(eng.c.origin, [2, 3])]),
                                      np.sort(original_ids))
