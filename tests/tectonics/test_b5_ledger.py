"""实际碰撞分配的只读账本须守住原计算，永久编号可跨提交定位。"""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np

from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.mesh import load_mesh


def module():
    if "b5_evolution_test" not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "scripts/design_b5_evolution.py"
        spec = importlib.util.spec_from_file_location("b5_evolution_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["b5_evolution_test"]


def observer(uids):
    obs = object.__new__(module().Observer)
    obs.uids = np.asarray(uids, np.int64)
    obs.center = 0
    obs.process, obs.timeline, obs.donations = {}, [], []
    obs.resolve_max_error, obs.initial = 0., None
    return obs


def test_collision_reconstruction_matches_actual_multi_recipient_transfer():
    mesh = load_mesh(3)
    c = Crust(mesh.n + 1)
    c.node[:] = np.r_[np.arange(mesh.n), 0]
    c.plate[:] = 1
    c.plate[-1] = 2
    c.ctype[:] = CONT
    c.T[:] = 35.
    c.T[-1] = 20.
    eng = Engine(mesh, c, Params(collision_radius_km=1500.), 5., {1: np.zeros(3), 2: np.zeros(3)})
    eng.polarity[1002] = 2
    control = copy.deepcopy(eng)
    close = np.argsort(np.linalg.norm(mesh.xyz - mesh.xyz[0], axis=1))[:4]
    obs = observer(sorted(close))
    before = obs.stock(eng)
    args = (eng.positions(), np.array([mesh.n]), np.array([1]), np.array([0]))
    obs.record_resolve(Engine._resolve, eng, *args)
    Engine._resolve(control, *args)
    for field in Crust.FIELDS:
        np.testing.assert_array_equal(getattr(eng.c, field), getattr(control.c, field))
    np.testing.assert_array_equal(eng.alive, control.alive)
    assert eng.events == control.events and eng.budget == control.budget
    np.testing.assert_allclose(obs.stock(eng) - before, obs.process["collision"], atol=1e-8)
    assert obs.resolve_max_error < 1e-7
    assert len(obs.donations) == 1 and len(obs.donations[0]["target_uid"]) == 4
    assert (obs.donations[0]["deposited_km3"] > 0).all()


def test_uid_stock_survives_index_compaction_and_marks_not_yet_born():
    mesh = load_mesh(2)
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.T[:] = 35.
    eng = Engine(mesh, c, Params(), 10., {1: np.zeros(3)})
    obs = observer([1, 3, mesh.n + 10])
    expected = obs.stock(eng)
    eng.c = eng.c.gather(np.arange(1, mesh.n))
    eng.alive = np.ones(len(eng.c), bool)
    eng._rebuild()
    np.testing.assert_array_equal(obs.stock(eng), expected)
    assert expected[-1] == 0.
