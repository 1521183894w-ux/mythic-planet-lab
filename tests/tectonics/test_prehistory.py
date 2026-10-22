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
