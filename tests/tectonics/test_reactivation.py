"""U33：板内挤压复活（packages/tectonics/reactivation.py）与 history yaml reactivations 的读取（scripts/design_forward.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, ORO_REACT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.reactivation import apply_shortening, line_weight

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(5)


def test_line_weight_is_one_inside_half_width_and_tapers_to_zero():
    w = line_weight(np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0]), 100.0)
    assert w[0] == w[1] == w[2] == 1.0
    assert 0.0 < w[3] < 1.0 and w[4] == pytest.approx(0.0, abs=1e-12) and w[5] == 0.0


def test_shortening_thickens_only_weighted_continental_crust_of_the_plate(mesh):
    c = Crust(mesh.n)
    cen = lonlat_to_xyz(0.0, 0.0)
    cont = mesh.xyz @ cen >= np.cos(np.radians(30.0))
    c.plate[:] = np.where(mesh.xyz[:, 0] > 0.0, 1, 2)
    c.ctype[cont] = CONT
    c.T[cont] = 35.0
    eng = Engine(mesh, c, Params(), 30.0, {1: np.zeros(3), 2: np.zeros(3)})
    w = np.where(mesh.xyz @ cen >= np.cos(np.radians(10.0)), 1.0, 0.0)
    rate = np.log1p(0.5) / 10.0
    for _ in range(10):
        apply_shortening(eng, w, rate, 1.0, plate=1)
    zone = (w > 0) & cont & (c.plate == 1)
    assert np.allclose(eng.c.T[zone], 35.0 * 1.5)                      # 10 Myr 增厚到 1.5 倍
    assert np.allclose(eng.c.T[(w > 0) & cont & (c.plate == 2)], 35.0)  # 别的板块不动
    assert np.allclose(eng.c.T[cont & (w == 0)], 35.0)
    assert (eng.c.T[~cont] == 7.0).all()                                # 洋壳不动
    assert (eng.c.oro[zone] == ORO_REACT).all()
    assert eng.budget["reactivation_added"] > 0


def test_build_reactivations_from_line_and_plateau_features():
    spec = importlib.util.spec_from_file_location("design_forward", ROOT / "scripts" / "design_forward.py")
    fwd = importlib.util.module_from_spec(spec)
    sys.modules["design_forward"] = fwd
    spec.loader.exec_module(fwd)
    from mythic.metrics.grid import LatLonGrid

    g = LatLonGrid(360, 180)
    LO, LA = np.meshgrid(g.lon, g.lat)
    land = (np.abs(LO) < 40) & (np.abs(LA) < 30)
    design = {"mountains": [{"id": "M9", "line": [[-10, 0], [10, 0]], "width_km": 200}],
              "plateaus": [{"id": "P9", "poly": [[-20, 10], [20, 10], [20, 20], [-20, 20]], "ramp_km": 200}]}
    hcfg = {"blocks": [{"code": "B", "id": 101}],
            "reactivations": [{"id": "R1", "feature": "M9", "anchor": "B", "t_start": 25, "strain": 0.9},
                              {"id": "R2", "feature": "P9", "anchor": "B", "t_start": 30, "t_end": 10, "strain": 0.25}]}
    r1, r2 = fwd.build_reactivations(hcfg, design, g, land)
    assert r1["anchor"] == 101 and r1["rate"] == pytest.approx(np.log(1.9) / 25.0)
    assert r1["W"][np.argmin(np.abs(g.lat)), np.argmin(np.abs(g.lon))] == 1.0       # 山脊线上
    assert r1["W"][np.argmin(np.abs(g.lat - 5.0)), np.argmin(np.abs(g.lon))] == 0.0  # 离线约 550 km
    assert r2["t_end"] == 10.0 and r2["rate"] == pytest.approx(np.log(1.25) / 20.0)
    assert r2["W"].max() > 0.9 and r2["W"][np.argmin(np.abs(g.lat)), np.argmin(np.abs(g.lon))] == 0.0


def test_shortening_selects_material_origin():
    """C4：只让原ALB基底缩短，新生裂谷陆壳与同板块其他来源不被冒充为原地体。"""
    from mythic.tectonics.crust import Crust,Params,CONT
    from mythic.tectonics.engine import Engine
    from mythic.tectonics.mesh import load_mesh
    mesh=load_mesh(3);c=Crust(mesh.n);c.plate[:]=1;c.ctype[:]=CONT;c.T[:]=35.
    c.origin[:mesh.n//2]=304;c.origin[mesh.n//2:]=91
    e=Engine(mesh,c,Params(),10.,{1:np.zeros(3)})
    original=e.cont_volume()
    out=apply_shortening(e,np.ones(len(c)),np.log1p(.25),1.,plate=1,origin=304)
    np.testing.assert_allclose(e.c.T[c.origin==304],43.75)
    np.testing.assert_array_equal(e.c.T[c.origin==91],np.full((c.origin==91).sum(),35.))
    assert e.cont_volume()-original==pytest.approx(out["volume_km3"],rel=1e-12)
