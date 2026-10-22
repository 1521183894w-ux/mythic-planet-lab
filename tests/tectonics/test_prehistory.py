"""前史（1.2b）：省份拼贴、裂解分块、漂移的不变量，以及同一种子可确定性重算（U8）。L5 网格，较粗但够检查结构。"""

import numpy as np
import pytest

from mythic.tectonics.crust import CONT
from mythic.tectonics.engine import graph_components
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.prehistory import PrehistoryParams, build_initial_world


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(5)


@pytest.fixture(scope="module")
def world(mesh, tmp_path_factory):
    return build_initial_world(mesh, seed=3, work_dir=tmp_path_factory.mktemp("ph"))


def test_supercontinent_and_provinces(mesh, world):
    s0 = world.snapshots[0]
    cont = s0["ctype"] == CONT
    area = mesh.area_km2[cont].sum()
    assert area == pytest.approx(PrehistoryParams().cont_area_km2, rel=0.15)
    assert (s0["prov"][cont] > 0).all() and (s0["prov"][~cont] == 0).all()
    assert np.isfinite(s0["t_form"]).all()
    types = {p["type"] for p in world.provinces}
    assert {1, 4} <= types                                     # 有太古宙克拉通，也有古生代增生带
    bounds = {1: (2500, 3600), 2: (1600, 2500), 3: (540, 1600), 4: (420, 540)}
    for p in world.provinces:
        lo, hi = bounds[p["type"]]
        assert lo <= p["age_ma"] <= hi
    # 每个省份在网格上连通（L5 上省份小，只查面积够大的）
    e = mesh.edges
    for p in world.provinces:
        m = s0["prov"] == p["id"]
        if m.sum() < 8:
            continue
        lab = graph_components(mesh.n, e[m[e[:, 0]] & m[e[:, 1]]])
        assert np.unique(lab[m]).size == 1


def test_breakup_into_blocks(mesh, world):
    assert world.events[0]["kind"] == "supercontinent_breakup"
    nb = world.events[0]["n_blocks"]
    assert nb >= 2 and len(world.blocks) == nb
    s0 = world.snapshots[0]
    cont = s0["ctype"] == CONT
    # 裂解时每个陆块在自己的板块上
    assert (s0["plate"][cont] == s0["origin"][cont]).all()
    assert any(b["block_boundary"] for b in world.boundaries)
    for f in world.events[0]["failed_rifts"]:
        assert f["block"] in range(1, nb + 1)


def test_drift_keeps_sphere_covered(mesh, world):
    fin = world.snapshots[-1]
    assert (fin["plate"] >= 0).all()                            # 300 Ma 提交后没有空隙
    assert world.t == 300.0
    assert np.isfinite(fin["elev"]).all()
    # 陆壳只因碰撞拼贴（面积换成厚度）减少、因岛弧增生增加，总量变化不大
    a0 = mesh.area_km2[world.snapshots[0]["ctype"] == CONT].sum()
    a1 = mesh.area_km2[fin["ctype"] == CONT].sum()
    assert a1 == pytest.approx(a0, rel=0.15)


def test_deterministic_rebuild(mesh, world, tmp_path_factory):
    """U8：候选世界要能按"种子 + 参数 + 代码版本"确定性重算。"""
    again = build_initial_world(mesh, seed=3, work_dir=tmp_path_factory.mktemp("ph2"))
    a, b = world.snapshots[-1], again.snapshots[-1]
    for f in ("plate", "ctype", "prov", "origin", "oro"):
        assert (a[f] == b[f]).all(), f
    assert np.allclose(a["elev"], b["elev"])
    assert [e["kind"] for e in world.events] == [e["kind"] for e in again.events]


def test_rift_cut_limits_transform_offsets():
    """U20：分界整理成"洋脊段 + 转换断层"时，任意 gap_km 范围内的累计错开不超过位移减 gap_km，
    刚体漂移后两侧处处至少隔开约 gap_km。不加限制时锯齿分界的长转换断层两侧一直相贴。"""
    from scipy.spatial import cKDTree

    from mythic.tectonics.geometry import DEG, R_KM, km_from_chord, lonlat_to_xyz, rotmat
    from mythic.tectonics.prehistory import monotone_cut

    rng = np.random.default_rng(0)
    lon = rng.uniform(-20, 20, 40000)
    lat = rng.uniform(-20, 20, 40000)
    X = lonlat_to_xyz(lon, lat)
    # 锯齿分界：每 4° 纬度在经度上来回错开 ±6°（约 670 km 的转换断层）
    is_b = lon > 6.0 * np.sign(np.sin(np.radians(lat) * 45.0))
    axis = np.array([0.0, 0.0, 1.0])            # b 相对 a 绕自转轴向东转
    total = 10.0 * DEG                          # 赤道位移约 1110 km
    gap = 400.0

    def min_gap(ct):
        ok, to_b = ct.assign_b(X)
        side_b = np.where(ok, to_b, is_b)
        A, B = X[~side_b], X[side_b] @ rotmat(axis, total).T
        return float(km_from_chord(cKDTree(B).query(A)[0].min()))

    limited = monotone_cut(X, is_b, axis, 1, 2, 100.0, 120.0, 3000.0, total_angle=total, gap_km=gap)
    free = monotone_cut(X, is_b, axis, 1, 2, 100.0, 120.0, 3000.0)
    assert min_gap(limited) > gap - 100.0       # 带宽 100 km 的离散误差
    assert min_gap(free) < 150.0
