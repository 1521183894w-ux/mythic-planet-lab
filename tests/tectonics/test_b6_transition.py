import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import Crust, CONT

spec = importlib.util.spec_from_file_location('transition', Path(__file__).resolve().parents[2] / 'scripts/design_b6_transition.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_partial_capacity_redirects_overflow_and_conserves(monkeypatch):
    e = Engine.__new__(Engine)
    e.c = Crust(4)
    e.c.T[:] = [35., 70., 35., 35.]
    e.c.ctype[:] = CONT
    e.c.plate[:] = 101
    e.alive = np.ones(4, bool)
    e.area = np.ones(4)
    e.pedges = np.array([[0, 1], [1, 2], [2, 3]])
    e.m = SimpleNamespace(h_km=60.)
    e.p = SimpleNamespace(T_cont=35., erosion_law='linear', tau_oro_myr=np.inf,
                          flow_T_km=100., flow_min_T_km=None, kappa_km2_myr=0.,
                          T_max_km=65., arc_to_cont_T_km=31.)
    e.t = 0.
    e.budget = {'eroded': 0., 'arc_converted': 0.}
    monkeypatch.setattr(m.kin, 'total_from', lambda *args: np.eye(3))
    monkeypatch.setattr(m, 'resistance', lambda points: np.array([1., 0., .5, 0.]))
    before = np.dot(e.c.T, e.area)
    m.candidate_surface(e, 1., np.ones((4, 3)))
    assert np.allclose(e.c.T, [35., 65., 40., 35.])
    assert np.isclose(np.dot(e.c.T, e.area), before)

def test_resistance_bounded_and_zero_outside():
    pts = m.lonlat_to_xyz(np.array([0., 103.2, 106., 107.]), np.array([0., 28.8, 31., 25.]))
    w = m.resistance(pts)
    assert np.all((w >= 0.) & (w <= 1.))
    assert w[0] == 0.
    assert w[1] < 1e-8
    assert w[2] > 0.
    assert w[3] > 0.
