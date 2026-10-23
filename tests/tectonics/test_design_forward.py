"""U25：正向演化与终图对照的辅助函数（scripts/design_forward.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fwd = _load("design_forward", "design_forward.py")


def test_label_wrap_joins_regions_across_the_dateline():
    m = np.zeros((20, 40), bool)
    m[5:9, :3] = True           # 180°E 一侧
    m[6:10, -2:] = True         # 180°W 一侧，与上面隔着 180° 经线相接
    m[15:17, 18:22] = True      # 另一块，不相接
    lab, n = fwd.label_wrap(m)
    assert n == 2
    assert lab[6, 0] == lab[7, -1] != 0
    assert lab[16, 20] not in (0, lab[6, 0])


def test_coast_mask_is_the_outer_ring_of_land():
    land = np.zeros((10, 10), bool)
    land[2:8, 2:8] = True
    c = fwd.coast_mask(land)
    assert c[2, 2] and c[7, 5] and not c[4, 4]
    assert not (c & ~land).any()
    assert c.sum() == 6 * 6 - 4 * 4


def test_excess_thickness_follows_airy_and_engine_cap():
    """U26 体积平衡：终图高度按 Airy 均衡换算成超出 35 km 的厚度，与引擎同一组常数，封顶于 65 km。"""
    init = _load("design_initial_world", "design_initial_world.py")
    from mythic.tectonics.crust import Crust, CONT, Params, elevation

    p = Params()
    ex = init.excess_thickness_km(np.array([-500.0, 400.0, 2000.0, 4945.0, 8000.0]))
    assert ex[0] == 0 and ex[1] == 0
    assert abs(ex[3] - (p.T_max_km - p.T_cont)) < 0.01 and ex[4] == ex[3]
    c = Crust(1)                                  # 反过来：引擎按这个厚度算出的高程就是 2000 m
    c.ctype[:] = CONT
    c.T[:] = p.T_cont + ex[2]
    assert abs(float(elevation(c, p)[0]) - 2000.0) < 1.0


def test_feature_mask_unions_polygons_and_ridge_bands():
    fm = _load("design_final_map", "design_final_map.py")
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(360, 180)
    land = np.ones(g.shape, bool)
    design = {"plateaus": [{"id": "P", "poly": [[10, 10], [20, 10], [20, 20], [10, 20]]}],
              "mountains": [{"id": "M", "line": [[-40, 0], [-30, 0]], "width_km": 400}]}
    LO, LA = np.meshgrid(g.lon, g.lat)
    m = fm.feature_mask(design, g, land, ["P", "M"])
    assert m[(LO > 11) & (LO < 19) & (LA > 11) & (LA < 19)].all()
    assert m[(np.abs(LA) < 1) & (LO > -39) & (LO < -31)].all()
    assert not m[(np.abs(LA) > 3) & (LO > -39) & (LO < -31)].any()        # 山脊两侧各 200 km（约 1.8°）以外
    assert not fm.feature_mask(design, g, np.zeros(g.shape, bool), ["P", "M"]).any()
