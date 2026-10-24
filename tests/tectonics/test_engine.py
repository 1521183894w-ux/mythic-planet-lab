"""地壳引擎的不变量（tasks/stage1_tectonics.md 验收）：球面接缝与极区无裂缝、刚体运动与精确旋转逐节点一致（面积守恒）、
板块 ID 稳定、无瞬移、慢速运动不丢失、水深—年龄关系。"""

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, Crust, Params, elevation, gdh1_depth_m
from mythic.tectonics.engine import Engine, graph_components
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, omega_rotmat, pole_vector
from mythic.tectonics.mesh import load_mesh


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(5)


def _two_plates(mesh, cap_center=(170.0, 70.0), r=25.0):
    """板块 1：一块大陆（中心在 170°E、70°N，跨 ±180° 经线、靠近北极）及其周围；板块 2：其余。"""
    c = Crust(mesh.n)
    cen = lonlat_to_xyz(*cap_center)
    cont = mesh.xyz @ cen >= np.cos(np.radians(r))
    c.plate[:] = np.where(mesh.xyz @ cen >= np.cos(np.radians(r + 15)), 1, 2)
    c.ctype[cont] = CONT
    c.T[cont] = 35.0
    c.origin[cont] = 1
    c.age[~cont] = 60.0
    return c, cont


@pytest.mark.parametrize("rate", [0.6, 0.05])
def test_rigid_rotation_exact_across_seam_and_pole(mesh, rate):
    """所有板块一起绕赤道上的轴旋转 60 Myr：大陆跨过 ±180° 经线和北极。结果应与精确旋转逐节点一致、无裂缝、面积不变；
    慢速（0.05°/Myr，约 0.55 cm/yr）也要正确移动（1.1 原型第一版会把它取整吞掉）。"""
    c, cont0 = _two_plates(mesh)
    w = pole_vector(0.0, 80.0, rate)
    eng = Engine(mesh, c, Params(), 100.0, {1: w, 2: w})
    for _ in range(60):
        eng.step({q: omega_rotmat(w, 1.0) for q in eng.plates}, {q: w for q in eng.plates})
        if eng.steps_since_commit == 0:
            assert eng.commit_log[-1]["covered_area_km2"] == pytest.approx(mesh.area_km2.sum(), rel=1e-12)
    v = eng.view()
    _, nn = mesh.tree.query(mesh.xyz @ omega_rotmat(w, 60.0))
    exact = cont0[nn]
    got = v.ctype == CONT
    assert (exact == got).all()
    assert eng.events == []
    # 大陆仍是一个连通块（接缝两侧没有被切开）
    lab = graph_components(mesh.n, mesh.edges[got[mesh.edges[:, 0]] & got[mesh.edges[:, 1]]])
    assert np.unique(lab[got]).size == 1


def test_plate_ids_stable_and_no_teleport(mesh):
    """两块板块相向运动 30 Myr（俯冲）：每个存活单元始终属于同一板块，每步位移不超过所属板块的旋转量。"""
    c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=20.0)
    w = {1: pole_vector(0.0, 90.0, 0.5), 2: pole_vector(0.0, -90.0, 0.3)}
    eng = Engine(mesh, c, Params(), 100.0, w)
    plate_of = dict(zip(eng.c.uid.tolist(), eng.c.plate.tolist()))
    for _ in range(30):
        uid0, pos0 = eng.c.uid.copy(), eng.positions()
        alive0 = eng.alive.copy()
        eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        uid1, pos1 = eng.c.uid, eng.positions()
        common, i0, i1 = np.intersect1d(uid0[alive0], uid1[eng.alive], return_indices=True)
        a0 = np.flatnonzero(alive0)[i0]
        a1 = np.flatnonzero(eng.alive)[i1]
        disp = R_KM * np.arccos(np.clip((pos0[a0] * pos1[a1]).sum(1), -1, 1))
        bound = R_KM * np.radians(np.where(eng.c.plate[a1] == 1, 0.5, 0.3)) + 1e-6
        assert (disp <= bound).all()
        for u, q in zip(eng.c.uid.tolist(), eng.c.plate.tolist()):
            assert plate_of.setdefault(u, q) == q
    assert any(e["kind"] == "subduction_start" for e in eng.events)
    assert eng.commit_log[-1]["covered_area_km2"] == pytest.approx(mesh.area_km2.sum(), rel=1e-12)


def test_gdh1_and_isostasy():
    assert gdh1_depth_m(0.0) == pytest.approx(2600.0)
    assert gdh1_depth_m(20.0) == pytest.approx(2600 + 365 * np.sqrt(20), abs=2.0)   # 两段在 20 Myr 处连续
    assert gdh1_depth_m(100.0) == pytest.approx(5651 - 2473 * np.exp(-2.78), abs=1e-6)
    c = Crust(3)
    c.ctype[:] = CONT
    c.T[:] = [35.0, 65.0, 20.0]
    e = elevation(c, Params())
    assert e[0] == pytest.approx(400.0)
    assert e[1] == pytest.approx(400.0 + 30 * 151.5)
    assert e[2] < -1000.0                       # 减薄陆缘在水下


def test_transfer_units_keeps_positions_and_attributes(mesh):
    """U26：一批洋壳单元从板块 2 改属板块 1（两块板块已转过不同角度）：位置与属性不变，原单元删除，板块 1 不重叠。"""
    c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=20.0)
    c.age[:] = np.arange(mesh.n) % 97          # 用年龄当标签核对属性随单元走
    w = {1: pole_vector(0.0, 90.0, 0.4), 2: pole_vector(30.0, 0.0, 0.3)}
    eng = Engine(mesh, c, Params(), 100.0, w)
    for _ in range(10):
        eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
    assert eng.steps_since_commit == 0
    pos = eng.positions()
    cen = lonlat_to_xyz(0.0, 0.0)
    ring = np.flatnonzero(eng.alive & (eng.c.plate == 2) & (pos @ cen > np.cos(np.radians(45))))
    assert ring.size > 50
    added = eng.transfer_units(2, 1, ring)
    assert added > 0.8 * ring.size
    assert not eng.alive[ring].any()
    new = np.arange(len(eng.c) - added, len(eng.c))        # 新单元接在单元表末尾
    assert (eng.c.plate[new] == 1).all()
    # 新单元在板块 1 参照格网上各占一个节点，且不与原有板块 1 单元重叠
    nodes = eng.c.node[eng.alive & (eng.c.plate == 1)]
    assert np.unique(nodes).size == nodes.size
    # 每个新单元离原来某个被转移单元的位置不超过一个网格间距，年龄取自覆盖它的原单元
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pos[ring]).query(eng.positions()[new])
    assert (d * R_KM < mesh.h_km).all()
    assert np.isin(eng.c.age[new], eng.c.age[ring]).all()
    assert eng.events[-1]["kind"] == "unit_transfer"


def test_gap_prefer_gives_new_ocean_to_one_plate(mesh):
    """U26：指定板块对之间张开的新洋壳全部归指定的一块（默认按最近单元，两边各一半）。"""
    def run(prefer):
        c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=20.0)
        w = {1: pole_vector(0.0, 90.0, 0.5), 2: np.zeros(3)}         # 绕赤道上东经 90° 的极转：板块 1 向南移动，北侧张开
        eng = Engine(mesh, c, Params(), 100.0, w)
        eng.gap_prefer.update(prefer)
        n0 = {q: int((eng.c.plate == q).sum()) for q in (1, 2)}
        for _ in range(10):
            eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        new = eng.c.uid >= len(c)
        return {q: int((new & (eng.c.plate == q)).sum()) for q in (1, 2)}, n0

    both, _ = run({})
    assert both[1] > 0 and both[2] > 0
    only2, _ = run({1 * 1000 + 2: 2})
    assert only2[1] == 0 and only2[2] >= 0.9 * (both[1] + both[2])


def test_thick_intraoceanic_arc_is_accreted_not_subducted(mesh):
    """U28 弧—陆碰撞：板块 2 上一条厚的洋内弧（洋壳类型、25 km、离板块 2 的陆壳很远）被板块 1 的大陆撞上时转成陆壳拼贴，
    不被俯冲；不开这一项时被俯冲掉。"""
    def run(accrete):
        c, cont = _two_plates(mesh, cap_center=(0.0, 0.0), r=15.0)     # 板块 1：大陆（及其周围洋壳）；板块 2：其余洋壳
        arc = (c.plate == 2) & (np.abs(mesh.lat) < 25) & (mesh.lon > 31) & (mesh.lon < 36)
        c.T[arc] = 25.0
        p = Params(arc_accrete_T_km=20.0 if accrete else None, arc_ocean_T_max_km=30.0, arc_to_cont_T_km=31.0)
        w = {1: pole_vector(90.0, 0.0, 0.5), 2: np.zeros(3)}            # 绕北极转：板块 1 向东撞向弧
        eng = Engine(mesh, c, p, 100.0, w)
        eng.polarity[1 * 1000 + 2] = 1                                  # 带大陆的板块 1 下行，弧在上盘（新几内亚、台湾式）
        uid_arc = set(eng.c.uid[arc].tolist())
        for _ in range(40):
            eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        alive_arc = [u for u in eng.c.uid[eng.alive].tolist() if u in uid_arc]
        return len(alive_arc), int(((eng.c.origin == 90) & eng.alive).sum())

    n_keep, n_acc = run(True)
    n_lost, _ = run(False)
    assert n_acc > 0 and n_keep > n_lost


def test_new_plate_takes_ocean_units_and_moves_on_its_own(mesh):
    """U28 第二轮：俯冲起始时新增一块板块，把一片洋壳整批转给它，之后按自己的角速度运动（菲律宾海式）。"""
    c, cont = _two_plates(mesh, cap_center=(0.0, 0.0), r=15.0)
    w = {1: np.zeros(3), 2: np.zeros(3)}
    eng = Engine(mesh, c, Params(), 100.0, w)
    patch = (eng.c.plate == 2) & (eng.c.ctype == 0) & (np.abs(mesh.lat) < 15) & (mesh.lon > 100) & (mesh.lon < 130)
    w3 = pole_vector(90.0, 0.0, 0.5)                       # 新板块向东
    assert eng.add_plate(3, w3) and not eng.add_plate(3, w3)
    n = eng.transfer_units(2, 3, np.flatnonzero(patch))
    assert n > 0.8 * patch.sum()
    lon0 = np.degrees(np.arctan2(*eng.positions()[eng.alive & (eng.c.plate == 3)][:, [1, 0]].T)).mean()
    w[3] = w3
    for _ in range(10):
        eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
    lon1 = np.degrees(np.arctan2(*eng.positions()[eng.alive & (eng.c.plate == 3)][:, [1, 0]].T)).mean()
    assert 3.0 < lon1 - lon0 < 6.0                         # 10 Myr × 0.5°/Myr，前缘消耗、后缘新生会让质心略有偏移


def test_ocean_arc_flux_factor_thickens_intraoceanic_arc_faster(mesh):
    """U28 第二轮：同样的岩浆通量摊在窄的洋内弧带上增厚更快（arc_ocean_flux_factor）。"""
    def run(factor):
        c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=15.0)
        p = Params(arc_ocean_flux_factor=factor, arc_ocean_T_max_km=30.0, arc_to_cont_T_km=31.0)
        w = {1: pole_vector(90.0, 0.0, 0.5), 2: np.zeros(3)}
        eng = Engine(mesh, c, p, 100.0, w)
        eng.polarity[1 * 1000 + 2] = 1
        for _ in range(15):
            eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        return float(eng.c.T[eng.alive & (eng.c.plate == 2)].max())
    t1, t3 = run(None), run(3.0)
    assert t3 - 7.0 > 2.5 * (t1 - 7.0) > 0


def test_compressional_arc_thickens_past_magmatic_cap_only_where_allowed(mesh):
    """U28 第三轮：上盘朝海沟推进的洋内弧，构造缩短可以把弧地壳增厚到 arc_compress_T_max_km（吕宋式），岩浆增生仍封顶于
    arc_ocean_T_max_km；不设或不在 arc_compress_pairs 里时两项合起来封顶于 arc_ocean_T_max_km。"""
    def run(compress, pairs=None):
        c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=15.0)
        c.ctype[:] = 0                                   # 两块都是洋壳：洋—洋俯冲
        c.T[:] = 7.0
        p = Params(arc_rate_km_myr=0.2, arc_shortening_km_myr=0.8, arc_ocean_flux_factor=3.0, arc_ocean_T_max_km=30.0,
                   arc_to_cont_T_km=31.0, arc_compress_T_max_km=compress)
        w = {1: np.zeros(3), 2: pole_vector(90.0, 0.0, -0.5)}   # 板块 2 绕北极向西转，在板块 1 东缘压上去（上盘推进）
        eng = Engine(mesh, c, p, 100.0, w)
        eng.polarity[1 * 1000 + 2] = 1                   # 板块 1 下行
        eng.arc_compress_pairs = pairs
        for _ in range(90):                              # 约 60 Myr 到 30 km 上限，之后只有构造缩短一项
            eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        return float(eng.c.T[eng.alive & (eng.c.plate == 2)].max())
    t_off, t_on, t_other = run(None), run(38.0), run(38.0, pairs={5 * 1000 + 6})
    assert t_off <= 30.0 + 1e-9 and t_other <= 30.0 + 1e-9
    assert 32.0 < t_on <= 38.0 + 1e-9


def test_gap_prefer_does_not_grow_the_over_plate_at_a_convergent_boundary(mesh):
    """U28 第二轮：指定归属的新洋壳只给真正张开处；汇聚边界上网格对不齐的小洞归下行板块，上盘不会一次次往外长
    （否则菲律宾海式板块东缘 50 Myr 长出约 600 km，大陆弧前往海里长、海沟离岸越来越远）。"""
    def run(opening_only):
        c, _ = _two_plates(mesh, cap_center=(0.0, 0.0), r=15.0)
        c.ctype[:] = 0
        c.T[:] = 7.0
        w = {1: pole_vector(90.0, 0.0, 0.5), 2: np.zeros(3)}
        eng = Engine(mesh, c, Params(gap_prefer_opening_only=opening_only), 100.0, w)
        eng.polarity[1 * 1000 + 2] = 1                  # 板块 1 俯冲到板块 2 之下
        eng.gap_prefer[1 * 1000 + 2] = 2                # 两块之间张开的新洋壳归板块 2（例如东大洋那样的设定）
        east = (mesh.lon > 20) & (mesh.lon < 60) & (np.abs(mesh.lat) < 20)
        n0 = int((eng.alive & (eng.c.plate == 2) & east[eng.c.node]).sum())
        for _ in range(30):
            eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        pos = eng.positions()
        lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(pos[:, 2], -1, 1)))
        inbox = (lon > 20) & (lon < 60) & (np.abs(lat) < 20)
        return int((eng.alive & (eng.c.plate == 2) & inbox).sum()) - n0
    grow_old, grow_new = run(False), run(True)
    assert grow_new < grow_old
