"""剧本生成器（1.2c）：几何小工具、分段运动、陆块编组、剧本与规划一致、确定性，以及 L5 上完整跑一遍的不变量。"""

import numpy as np
import pytest

from mythic.tectonics.crust import Params
from mythic.tectonics.geometry import R_KM, chord_from_km, lonlat_to_xyz, rotmat, unit
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.prehistory import build_initial_world
from mythic.tectonics.scenario import (Block, Contact, Dock, Planner, dock_pose, group_adjacent, run_scenario,
                                       slide_to_contact)


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(5)


@pytest.fixture(scope="module")
def world(mesh, tmp_path_factory):
    return build_initial_world(mesh, seed=3, work_dir=tmp_path_factory.mktemp("ph"))


@pytest.fixture(scope="module")
def planned(mesh, world):
    pl = Planner(mesh, world, seed=1)
    return pl, pl.plan(attempts=1)


def _cap(center, radius_km, n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = unit(np.asarray(center, float))
    e1 = unit(np.cross(c, [0.0, 0.0, 1.0]))
    e2 = np.cross(c, e1)
    r = radius_km / R_KM * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    return unit(np.cos(r)[:, None] * c + np.sin(r)[:, None] * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2))


def test_slide_to_contact_matches_gap():
    # 两个半径 500 km 的圆盘，中心相距 20°；绕垂直轴把第二个滑向第一个，接触角 ≈ 20° - 1000 km - 接触半径
    A = _cap(lonlat_to_xyz(0.0, 0.0), 500.0)
    B = _cap(lonlat_to_xyz(20.0, 0.0), 500.0, seed=1)
    h = 50.0
    ct = Contact(A, h)
    ang = slide_to_contact(B, np.array([0.0, 0.0, -1.0]), np.radians(30), ct)
    expect = np.radians(20.0) - 1000.0 / R_KM - 0.6 * h / R_KM
    assert ang is not None and abs(ang - expect) < np.radians(0.6)
    assert slide_to_contact(B, np.array([0.0, 0.0, 1.0]), np.radians(30), ct) is None   # 反方向永远碰不上


def test_dock_pose_segments_and_residual():
    a1, a2 = np.array([0.0, 0.0, 1.0]), unit(np.array([1.0, 0.0, 0.0]))
    d = Dock(1, 100.0, [(300.0, 200.0, a1, 0.4), (200.0, 100.0, a2, 0.2)])
    assert np.allclose(dock_pose(d, 300.0, 0.1, 20.0), np.eye(3))
    assert np.allclose(dock_pose(d, 250.0, 0.1, 20.0), rotmat(a1, 0.2))
    full = rotmat(a2, 0.2) @ rotmat(a1, 0.4)
    assert np.allclose(dock_pose(d, 100.0, 0.1, 20.0), full)
    # 锁定后按最后一段速率保留 10% 的汇聚 20 Myr，之后不再动
    res = rotmat(a2, 0.1 * 0.2 / 100.0 * 20.0) @ full
    assert np.allclose(dock_pose(d, 80.0, 0.1, 20.0), res)
    assert np.allclose(dock_pose(d, 0.0, 0.1, 20.0), res)


def test_group_adjacent_merges_only_touching(mesh):
    x = mesh.xyz
    def block(i, center, r_km):
        idx = np.flatnonzero(x @ unit(center) > np.cos(r_km / R_KM))
        return Block(i, idx, idx, float(mesh.area_km2[idx].sum()), unit(center))
    a = block(1, lonlat_to_xyz(0.0, 0.0), 1500.0)
    b = block(2, lonlat_to_xyz(np.degrees(3000.0 / R_KM), 0.0), 1500.0)   # 与 a 相切（网格上隔约一个间距）
    c = block(3, lonlat_to_xyz(120.0, 0.0), 1500.0)
    g = group_adjacent(mesh, {1: a, 2: b, 3: c}, merge_km=2.0 * mesh.h_km)
    assert set(g) == {1, 3} or set(g) == {2, 3}
    lead = [k for k in g if k != 3][0]
    assert set(g[lead].members) == {1, 2}
    assert g[lead].area_km2 == pytest.approx(a.area_km2 + b.area_km2)
    # U20：给出前史设计阶段的 links 时按它编组（与网格无关），只补并真正贴上（< safety_km）的陆块对
    g2 = group_adjacent(mesh, {1: a, 2: b, 3: c}, links=[[1, 3]], safety_km=0.5 * mesh.h_km)
    assert sorted(sorted(B.members or (k,)) for k, B in g2.items()) == [[1, 3], [2]]
    g3 = group_adjacent(mesh, {1: a, 2: b, 3: c}, links=[], safety_km=2.0 * mesh.h_km)
    assert sorted(sorted(B.members or (k,)) for k, B in g3.items()) == [[1, 2], [3]]


def test_script_matches_plan(planned):
    pl, plan = planned
    s = plan.script
    s.check_complete()
    Fa = lambda t: s.forward_total(plan.anchor, t)
    reopen = {e["block"] for e in plan.breakups if e["kind"] == "reopen"}
    for b, d in plan.docks.items():
        if b == plan.anchor or not d.segs or b in reopen:
            continue
        for t in (250, 180, 120):
            rel = Fa(t).T @ s.forward_total(b, t)
            # 剧本按整 Myr 写，规划的分段端点是 5 Myr 的整数倍，二者应几乎一致
            assert np.abs(rel - pl.pose(b, float(t))).max() < 2e-3, (b, t)
    for b, segs in plan.moves.items():
        for t in (260, 150, 0):
            rel = Fa(t).T @ s.forward_total(b, t)
            assert np.abs(rel - dock_pose(Dock(b, segs[-1][1], segs), float(t), 0.0, 0.0)).max() < 2e-2, (b, t)
    for q, lead in pl.follower_of.items():                     # 编组陆块与领头陆块一起运动
        assert np.allclose(s.forward_total(q, 0), s.forward_total(lead, 0), atol=1e-6)
    # 整体定位：锚定陆块 300 Myr 内匀速完成 G
    assert np.allclose(Fa(0), rotmat(plan.G_axis, plan.G_angle), atol=1e-6)


def test_plan_deterministic(mesh, world, planned):
    _, plan = planned
    again = Planner(mesh, world, seed=1).plan(attempts=1)
    assert again.roles == plan.roles
    for q in plan.script.stages:
        assert np.allclose(again.script.forward_total(q, 0), plan.script.forward_total(q, 0))


def test_run_scenario_invariants(mesh, world, tmp_path):
    run = run_scenario(mesh, world, seed=1, work_dir=tmp_path)
    ts = [s["t"] for s in run.snapshots]
    assert ts[0] == 300 and ts[-1] == 0 and len(ts) == 31        # 每 10 Ma 一帧（U8）
    fin = run.snapshots[-1]
    assert (fin["plate"] >= 0).all() and np.isfinite(fin["elev"]).all()
    # 地壳厚度上限：1.2c 之前孤立弧岛的多余厚度留在原处，可达 600 km。
    # 允许一步内的增厚暂时越过上限（下一次摊平前；L5 上实测最多 +5 km）
    assert max(float(s["T"].max()) for s in run.snapshots) <= Params().T_max_km + 10.0
    assert max(float(s["elev"].max()) for s in run.snapshots) < 9000.0
