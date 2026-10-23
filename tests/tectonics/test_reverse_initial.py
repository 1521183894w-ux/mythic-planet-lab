"""U24：被挤掉的陆缘与初始地壳（packages/tectonics/reverse.py）。"""

import numpy as np

from mythic.metrics.grid import LatLonGrid
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.reverse import PlateHistory, consumed_margin, fill_ocean, initial_crust, pose_from_ops


def _boxes(g):
    LO, LA = np.meshgrid(g.lon, g.lat)
    down = (LO > -10) & (LO < 10) & (LA > -20) & (LA < 0)          # 下行块：赤道以南
    up = (LO > -30) & (LO < 30) & (LA >= 0) & (LA < 30)            # 上盘块：紧贴其北
    return LO, LA, down, up


def test_consumed_margin_is_the_swept_strip_in_front_of_the_down_block():
    g = LatLonGrid(360, 180)
    LO, LA, down, up = _boxes(g)
    # 接触时刻上盘在北面 1000 km（下行块参照系里）：上盘从北往南推进了 1000 km
    R = pose_from_ops(lonlat_to_xyz(0, 15), [{"move": [1000.0, 0.0]}])
    m = consumed_margin(g.xyz, down, up, R, max_km=1500.0)
    assert m.any() and not (m & down).any()
    lat_in = LA[m]
    assert lat_in.min() >= 0 and lat_in.max() <= 10.5                # 只在下行块正前方约 1000 km（9°）内
    lon_in = LO[m]
    assert lon_in.min() > -11 and lon_in.max() < 11                   # 只在下行块的宽度内（两侧推进不算）


def test_consumed_margin_ignores_contacts_across_sea():
    g = LatLonGrid(360, 180)
    LO, LA, down, _ = _boxes(g)
    up = (LO > -30) & (LO < 30) & (LA >= 5) & (LA < 30)               # 中间隔着 5° 的海
    R = pose_from_ops(lonlat_to_xyz(0, 17), [{"move": [1000.0, 0.0]}])
    m = consumed_margin(g.xyz, down, up, R, max_km=1500.0)
    assert not m.any()


def test_initial_crust_reconstructs_rotated_block():
    mesh = load_mesh(5)
    g = LatLonGrid(360, 180)
    LO, LA, down, _ = _boxes(g)
    core = np.where(down, 7, 0)
    ext = np.zeros_like(core)
    h = PlateHistory()
    c0 = lonlat_to_xyz(0, -10)
    h.add(7, 0, "X", [(0, np.eye(3)), (100, pose_from_ops(c0, [{"move": [3000.0, 90.0]}]))])
    f = {"T": np.full(core.shape, 35.0), "t_form": np.full(core.shape, 1000.0), "prov": np.ones(core.shape, int)}
    c, owner, kind = initial_crust(mesh, h, 100.0, [7], core, ext, f, f)
    cont = owner == 7
    lon_c = mesh.lon[cont].mean()
    assert 20 < lon_c < 34                                            # 3000 km 向东（约 27°）
    area = mesh.area_km2[cont].sum()
    expect = g.cell_area_km2[down].sum()
    assert abs(area / expect - 1) < 0.08
    c = fill_ocean(mesh, c, owner, 100.0)
    assert (c.plate == 7).all() and (c.age[~cont] >= 2).all()        # 只有一块板块：洋壳都归它


def test_ocean_between_converging_blocks_goes_with_the_downgoing_plate():
    """U25：汇聚的两块之间的洋壳归下行一方，上盘只留弧前洋壳；其余洋壳归外大洋板块。"""
    from mythic.tectonics.reverse import assign_ocean_plates

    mesh = load_mesh(5)
    lon, lat = mesh.lon, mesh.lat
    plate = np.zeros(mesh.n, np.int64)
    over = (lat > 20) & (lat < 40) & (np.abs(lon) < 20)
    down = (lat < -20) & (lat > -40) & (np.abs(lon) < 20)
    plate[over], plate[down] = 1, 2
    cont = over | down
    out = assign_ocean_plates(mesh, plate, cont, [{"down": [2], "over": [1]}], forearc_km=250.0,
                              corridor_km=6500.0, background=900)

    def at(lo, la):
        return out[int(np.argmin((mesh.xyz @ lonlat_to_xyz(lo, la))[None, :] * -1))]

    assert (out[cont] == plate[cont]).all()
    assert at(0, 0) == 2            # 两块之间：归下行一方
    assert at(0, 18.5) == 1         # 上盘陆缘外 250 km 内：弧前，归上盘
    assert at(0, 60) == 900         # 上盘背后（下行块在同一方向）：外大洋
    assert at(150, 0) == 900        # 远离两块：外大洋


def test_ocean_plate_takes_the_outer_ocean_on_one_side_of_a_block():
    """U26：外大洋里最近的大陆是某块、在它某一侧、纬度在范围内的洋壳另成一块大洋板块；其余仍归外大洋板块。"""
    from mythic.tectonics.reverse import assign_ocean_plates

    mesh = load_mesh(5)
    lon, lat = mesh.lon, mesh.lat
    plate = np.zeros(mesh.n, np.int64)
    blk = (np.abs(lat) < 20) & (np.abs(lon) < 15)
    plate[blk] = 1
    out = assign_ocean_plates(mesh, plate, blk, [], forearc_km=500.0, corridor_km=6500.0, background=900,
                              ocean_plates=[{"id": 910, "near": 1, "side": "west", "lat": [-30, 30]}])

    def at(lo, la):
        return out[int(np.argmax(mesh.xyz @ lonlat_to_xyz(lo, la)))]

    assert at(-40, 0) == 910        # 西侧、纬度带内
    assert at(40, 0) == 900         # 东侧
    assert at(-40, 50) == 900       # 西侧但纬度带外
    assert at(-17, 0) == 1          # 弧前（L5 网格约 220 km，弧前放宽到 500 km）
