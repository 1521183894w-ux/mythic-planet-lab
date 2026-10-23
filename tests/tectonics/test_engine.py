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
        w = {1: pole_vector(0.0, 90.0, 0.5), 2: np.zeros(3)}         # 板块 1 向东移动，板块 2 不动：西侧张开
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
