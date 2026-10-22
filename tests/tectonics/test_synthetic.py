"""合成试验（tasks/stage1_tectonics.md 验收）："裂谷 -> 洋盆扩张"与"碰撞 -> 造山"结果符合预期。L6 网格（约 110 km）。"""

import numpy as np
import pytest

from mythic.tectonics import synthetic
from mythic.tectonics.crust import CONT
from mythic.tectonics.mesh import load_mesh


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(6)


@pytest.fixture(scope="module")
def rift(mesh, tmp_path_factory):
    return synthetic.rift_spreading(mesh, tmp_path_factory.mktemp("rift") / "rift.rot", rate_deg_myr=0.4, duration=50)


@pytest.fixture(scope="module")
def coll(mesh, tmp_path_factory):
    return synthetic.collision_orogen(mesh, tmp_path_factory.mktemp("coll") / "coll.rot")


def _new_ocean_on_equator(mesh, s):
    sel = (np.abs(mesh.lat) < 1.5) & (s["ctype"] == 0) & (s["age"] < 60) & (mesh.lon > -20) & (mesh.lon < 60)
    return mesh.lon[sel], s["age"][sel]


def test_rift_opens_ocean_of_expected_width(mesh, rift):
    assert [e["kind"] for e in rift.engine.events][0] == "rift"
    lon, _ = _new_ocean_on_equator(mesh, rift.snapshots[-1])
    width = lon.max() - lon.min()
    assert width == pytest.approx(0.4 * 50, abs=1.5)          # 宽度 = 速率 × 时间（±1.5°，约 1.4 个网格）


def test_rift_ocean_ages_symmetric_about_ridge(mesh, rift):
    lon, age = _new_ocean_on_equator(mesh, rift.snapshots[-1])
    ridge = lon[np.argmin(age)]
    mid = 0.5 * (lon.min() + lon.max())
    assert abs(ridge - mid) < 2.0                              # 洋脊在洋盆中线附近
    assert age.min() < 5.0 and 40.0 < age.max() <= 50.0        # 洋脊最新，陆缘处接近裂解以来的时间
    west, east = lon < ridge, lon > ridge
    # 年龄随离洋脊距离单调增大（相关系数）
    for side in (west, east):
        assert abs(np.corrcoef(np.abs(lon[side] - ridge), age[side])[0, 1]) > 0.95


def test_rift_margins_thinned(mesh, rift):
    s = rift.snapshots[-1]
    eq = (np.abs(mesh.lat) < 1.5) & (s["ctype"] == CONT)
    lon_oc, _ = _new_ocean_on_equator(mesh, s)
    margin = eq & (((mesh.lon < lon_oc.min()) & (mesh.lon > lon_oc.min() - 3)) |
                   ((mesh.lon > lon_oc.max()) & (mesh.lon < lon_oc.max() + 3)))
    assert s["T"][margin].min() < 20.0                         # 陆缘减薄
    assert (s["elev"][margin] < 0).any()                       # 陆架、陆坡在水下
    interior = eq & (np.abs(mesh.lon - (lon_oc.min() - 15)) < 3)
    assert np.allclose(s["T"][interior], 35.0)                 # 远离裂谷的大陆内部不受影响


def test_collision_timing_and_orogen(mesh, coll):
    ev = {e["kind"]: e for e in coll.engine.events}
    assert ev["subduction_start"]["subducting"] == 2           # 洋壳在大陆 A 东缘俯冲（安第斯型）
    # 两陆边缘相距 10°（赤道处），以 0.5°/Myr 汇聚：约 20 Myr 后（80 Ma）碰撞
    assert ev["collision_onset"]["t"] == pytest.approx(80.0, abs=3.0)
    s = coll.snapshots[-1]
    assert 55.0 <= s["T"].max() <= 65.5                        # 增厚但不超过厚度上限
    assert 3000.0 <= s["elev"].max() <= 5200.0                 # 高原高程，地球量级
    thick = s["T"] > 45.0
    assert np.abs(mesh.lon[thick] + 5.0).max() < 10.0          # 增厚带在缝合带附近（约 5°W）
    # 下行大陆 B 不应出现岛弧增厚（上盘弧前洋壳被 B 掩覆不算俯冲；1.2a 修正过的问题）
    assert not ((s["origin"] == 2) & (s["oro"] == 1)).any()
    assert s["T"][(s["origin"] == 2) & (s["ctype"] == CONT)].max() <= 35.0 + 1e-6


def test_collision_conserves_continental_volume(coll):
    cl0, cl = coll.engine.commit_log[0], coll.engine.commit_log[-1]
    b = cl["budget"]
    pred = cl0["cont_volume_km3"] + b["arc_added"] + b["arc_converted"] - b["eroded"] - b["rift_thinning"]
    assert cl["cont_volume_km3"] == pytest.approx(pred, rel=1e-6)
