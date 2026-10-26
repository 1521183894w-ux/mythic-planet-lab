"""逐边观测须与实际库存变化吻合，并保持原求解器结果不变。"""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.geometry import xyz_to_lonlat


def module():
    name = "b5_coupling_test"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "scripts/design_b5_coupling.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def observer(uids):
    mod = module()
    obs = object.__new__(mod.CouplingObserver)
    obs.uids = np.asarray(uids)
    obs.flow_rows, obs.borrow_rows, obs.borrow_checks = [], [], []
    obs.pending = {}
    obs.R_C_present = np.eye(3)
    obs.flow_error = obs.borrow_error = obs.flow_global_balance = 0.
    return obs


def test_observed_actual_surface_flux_is_neutral_and_closes_for_every_watched_host():
    mesh = load_mesh(3)
    c = Crust(mesh.n)
    c.ctype[:] = CONT
    c.plate[:] = 301
    c.T[:] = 35.
    c.T[0] = 50.
    eng = Engine(mesh, c, Params(erosion_law="linear", tau_oro_myr=1000.), 10., {301: np.zeros(3)})
    control = copy.deepcopy(eng)
    selected = np.r_[0, mesh.edges[np.any(mesh.edges == 0, axis=1)].ravel()]
    obs = observer(np.unique(selected))
    mod = module()
    mod.CouplingObserver.current = obs
    try:
        mod.traced(Engine._surface, "surface")(eng, 1.)
    finally:
        mod.CouplingObserver.current = None
    Engine._surface(control, 1.)
    np.testing.assert_array_equal(eng.c.T, control.c.T)
    assert eng.budget == control.budget
    assert obs.flow_rows and obs.flow_error < 1e-7
    assert obs.flow_global_balance < 1e-7
    center = [r for r in obs.flow_rows if r["target_uid"] == 0]
    assert all(r["volume_into_target_km3"] < 0 for r in center)


def test_simultaneous_holes_expose_shared_donor_oversubscription():
    mesh = load_mesh(2)
    c = Crust(2)
    c.ctype[:] = CONT
    c.plate[:] = 301
    c.T[:] = 40.
    eng = Engine(mesh, c, Params(), 5., {301: np.zeros(3)})
    obs = observer([0, 1])
    area = mesh.area_km2[c.node]
    g = Crust(2)
    g.plate[:] = 301
    g.uid[:] = [100, 101]
    # 两洞各取供体可用量的四分之三，合并扣款超过同一份余额。
    share = np.array([[4.5 * area[0], 0.], [4.5 * area[0], 0.]])
    loc = dict(new=c, g=g, nb=np.array([[0, 1], [0, 1]]), share=share,
        V=share.sum(axis=1), hg=np.ones(2, bool))
    obs.borrow(eng, loc)
    np.subtract.at(c.T, loc["nb"].ravel(), (share / area[loc["nb"]]).ravel())
    obs.check_borrow()
    assert obs.borrow_error < 1e-7
    assert obs.borrow_checks[0]["oversubscribed_donors"] == 1
    assert obs.borrow_checks[0]["max_overdraft_equivalent_km"] == pytest.approx(3.)


def test_material_coordinates_include_final_C_rotation_not_only_initial_frame():
    mesh = load_mesh(2)
    c = Crust(2)
    c.plate[:] = 301
    eng = Engine(mesh, c, Params(), 10., {301: np.zeros(3)})
    obs = observer([0, 1])
    obs.R_C_present = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    expected = np.column_stack(xyz_to_lonlat(mesh.xyz[c.node] @ obs.R_C_present.T))
    np.testing.assert_allclose(obs.coordinates(eng, c.node, c.plate), expected, atol=1e-12)
