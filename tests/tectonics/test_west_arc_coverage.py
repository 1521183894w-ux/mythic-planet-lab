"""只读实体查询必须保持板块覆盖，不能用最近陆地代替海水节点。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np

from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import lonlat_to_xyz, rotmat, xyz_to_lonlat
from mythic.tectonics.mesh import load_mesh


def report_module():
    if "west_arc_c5" in sys.modules:
        return sys.modules["west_arc_c5"]
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("west_arc_c5", root / "scripts/design_west_arc_c5_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_material_probe_matches_engine_after_rotation_and_keeps_ocean():
    mod = report_module()
    mesh = load_mesh(3)
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.T[:] = 7.
    # 横跨 ±180° 的高纬小陆块；其余保持洋壳，旋转后仍应查到同一永久单元。
    cap = mesh.xyz @ lonlat_to_xyz(179., 68.) > np.cos(np.radians(25.))
    c.ctype[cap] = CONT
    c.T[cap] = 35.
    c.origin[cap] = 304
    eng = Engine(mesh, c, Params(), 0., {1: np.zeros(3)})
    eng.R[1] = rotmat(lonlat_to_xyz(70., 25.), np.radians(17.))
    pos = eng.positions()
    lon, lat = xyz_to_lonlat(pos)
    p = {k: getattr(eng.c, k).copy() for k in c.FIELDS}
    p.update(lon=lon, lat=lat)
    direct, fits = mod.material_cover(mesh, p, pos)
    np.testing.assert_array_equal(direct, np.arange(mesh.n))
    fixed, _ = mod.material_cover(mesh, p, mesh.xyz)
    np.testing.assert_array_equal(fixed, eng.cover_fixed(pos))
    assert np.all(c.ctype[direct[~cap]] == 0)
    assert max(f["max_fit_error_km"] for f in fits) < 1e-7


def test_material_probe_resolves_overlaps_like_engine_without_spreading_land():
    mod = report_module()
    mesh = load_mesh(3)
    c = Crust(2 * mesh.n)
    c.node[:] = np.tile(np.arange(mesh.n), 2)
    c.plate[:] = np.repeat([1, 2], mesh.n)
    c.T[:] = 7.
    c.age[:mesh.n] = 20.
    cap = mesh.xyz @ lonlat_to_xyz(-179., 65.) > np.cos(np.radians(20.))
    ix = mesh.n + np.flatnonzero(cap)
    c.ctype[ix] = CONT
    c.T[ix] = 35.
    eng = Engine(mesh, c, Params(), 0., {1: np.zeros(3), 2: np.zeros(3)})
    lon, lat = xyz_to_lonlat(eng.positions())
    p = {k: getattr(eng.c, k).copy() for k in c.FIELDS}
    p.update(lon=lon, lat=lat)
    actual, _ = mod.material_cover(mesh, p, mesh.xyz)
    np.testing.assert_array_equal(actual, eng.cover_fixed())
    assert (c.ctype[actual[cap]] == CONT).all()
    assert (c.ctype[actual[~cap]] == 0).all()
