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
