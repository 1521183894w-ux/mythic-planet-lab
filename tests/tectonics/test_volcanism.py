"""U27 火山建造（packages/tectonics/volcanism.py）：热点岛链、慢速板块上成团、剥蚀与沉降、环礁。"""

import numpy as np
import pytest

from mythic.tectonics.crust import Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import lonlat_to_xyz, omega_rotmat, pole_vector
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.volcanism import ATOLL, ISLAND, SEAMOUNT, Hotspot, VolcanoField, VolcanoParams


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(5)


def _ocean(mesh, age=50.0):
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.age[:] = age
    return c


def _run(mesh, w, hs_lat, steps, flux="strong", t0=100.0):
    eng = Engine(mesh, _ocean(mesh), Params(), t0, {1: w})
    vp = VolcanoParams()
    field = VolcanoField([Hotspot("H", "测试", lonlat_to_xyz(0.0, hs_lat), t0, 0.0, vp.flux_km3_myr[flux],
                                  vp.summit_max_m[flux])], vp)
    for _ in range(steps):
        eng.step({1: omega_rotmat(w, 1.0)}, {1: w})
        field.step(eng)
    return eng, field


def test_moving_plate_carries_a_chain_older_downstream(mesh):
    w = pole_vector(90.0, 0.0, 0.5)              # 绕北极转：赤道上向东约 5.6 cm/yr
    eng, field = _run(mesh, w, 0.0, 30)
    st = field.state(eng)
    assert len(st["lon"]) >= 15
    # 越老越往东（板块运动方向）；强热点的火山很快长满，旁边会冒出卫星火山（夏威夷的 Loa、Kea 两列），不严格单调
    assert np.corrcoef(st["t_birth"], st["lon"])[0, 1] > 0.95
    assert st["lon"].max() - st["lon"].min() > 0.8 * 29 * 0.5     # 29 Myr × 0.5°/Myr
    assert (st["kind"][st["t_birth"] - eng.t < 3] == ISLAND).any()   # 热点上正在长的（3 Myr 内诞生的）露出海面


def test_fixed_plate_builds_one_growing_volcano(mesh):
    """板块不动：热点一直喂同一座火山（长满之前）。"""
    eng, field = _run(mesh, np.zeros(3), 20.0, 4, flux="weak")
    st = field.state(eng)
    assert len(st["lon"]) == 1
    assert field.vol[0] == pytest.approx(4 * VolcanoParams().flux_km3_myr["weak"])


@pytest.mark.parametrize("lat, expect", [(0.0, ATOLL), (50.0, SEAMOUNT)])
def test_old_islands_erode_and_sink(mesh, lat, expect):
    """老岛被剥蚀到海面、海底随板块变老继续沉降：热带的由珊瑚礁顶在海面成为环礁，高纬的沉成海山。"""
    w = pole_vector(90.0, 0.0, 0.5)
    eng = Engine(mesh, _ocean(mesh, age=5.0), Params(), 100.0, {1: w})   # 年轻洋壳沉降快
    vp = VolcanoParams()
    field = VolcanoField([Hotspot("H", "测试", lonlat_to_xyz(0.0, lat), 100.0, 95.0, vp.flux_km3_myr["medium"])], vp)
    for _ in range(60):
        eng.step({1: omega_rotmat(w, 1.0)}, {1: w})
        field.step(eng)
    st = field.state(eng)
    assert field.was_island.any()
    oldest = np.argmax(st["t_birth"])
    assert st["kind"][oldest] == expect


def test_full_volcano_on_slow_plate_spawns_a_cluster_and_stays_high(mesh):
    """慢速板块上火山长满后在旁边另起卫星火山（岛群）；顶部不超过热点强弱对应的上限；正在喂的不剥蚀。"""
    vp = VolcanoParams()
    eng = Engine(mesh, _ocean(mesh, age=30.0), Params(), 100.0, {1: np.zeros(3)})
    hs = Hotspot("H", "测试", lonlat_to_xyz(0.0, 30.0), 100.0, 0.0, vp.flux_km3_myr["medium"], vp.summit_max_m["medium"])
    field = VolcanoField([hs], vp)
    for _ in range(40):
        eng.step({1: np.eye(3)}, {1: np.zeros(3)})
        field.step(eng)
    st = field.state(eng)
    assert len(st["lon"]) >= 3
    d = np.degrees(np.arccos(np.clip(np.stack([np.cos(np.radians(st["lat"])) * np.cos(np.radians(st["lon"])),
                                               np.cos(np.radians(st["lat"])) * np.sin(np.radians(st["lon"])),
                                               np.sin(np.radians(st["lat"]))], 1) @ hs.xyz, -1, 1))) * 111.2
    assert (d < vp.spacing_km).all()                                # 都在热点附近（成团，不成链）
    assert (st["summit_m"] <= vp.summit_max_m["medium"] + 1.0).all()
    fed = field.feeding[0]
    assert field.summit_m(eng)[fed] > 0.5 * vp.summit_max_m["medium"]   # 正在喂的那座没有被剥蚀


@pytest.mark.parametrize("own_continent, expect_arc", [(False, True), (True, False)])
def test_arc_volcanoes_grow_on_intraoceanic_upper_plate(mesh, own_continent, expect_arc):
    """U28 岛弧火山：板块 1 向东俯冲到板块 2 之下，板块 2 前缘后方约 120 km 的弧地壳上长出一列弧火山；
    板块 2 自己的陆壳就在弧后（陆缘弧，安第斯式）时不长（只长洋内弧的火山岛）。"""
    c = Crust(mesh.n)
    cen = lonlat_to_xyz(0.0, 0.0)
    c.plate[:] = np.where(mesh.xyz @ cen >= np.cos(np.radians(30.0)), 1, 2)
    c.age[:] = 60.0
    band = (c.plate == 2) & (np.abs(mesh.lat) < 25) & (mesh.lon > 29) & (mesh.lon < 40)
    c.T[band] = 22.0                                                    # 已经增厚的弧地壳（洋壳类型）
    if own_continent:
        cont = band & (mesh.lon > 32.0)
        c.ctype[cont], c.T[cont] = 1, 35.0
    w = {1: pole_vector(90.0, 0.0, 0.5), 2: np.zeros(3)}                # 板块 1 向东约 5.6 cm/yr
    p = Params(arc_ocean_T_max_km=30.0, arc_to_cont_T_km=31.0)
    eng = Engine(mesh, c, p, 100.0, w)
    eng.polarity[1 * 1000 + 2] = 1                                      # 板块 1 下行
    field = VolcanoField([], VolcanoParams(arcs=True, arc_min_cont_km=400.0))   # L5 节点间距约 250 km，阈值放宽到 400 km
    for _ in range(12):
        eng.step({q: omega_rotmat(w[q], 1.0) for q in eng.plates}, w)
        field.step(eng)
    arc = field.alive & (field.src == 1)
    if expect_arc:
        assert arc.sum() >= 3
        assert (field.plate[arc] == 2).all()
    else:
        assert arc.sum() == 0
