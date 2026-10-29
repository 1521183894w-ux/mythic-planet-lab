import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import Crust, CONT

spec=importlib.util.spec_from_file_location('halo',Path(__file__).resolve().parents[2]/'scripts/design_b6_halo.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def engine():
    e=Engine.__new__(Engine)
    e.c=Crust(3)
    e.c.T[:]=[35.,85.,35.]
    e.c.ctype[:]=CONT
    e.c.plate[:]=101
    e.alive=np.ones(3,bool)
    e.area=np.ones(3)
    e.pedges=np.array([[0,1],[1,2]])
    e.m=SimpleNamespace(h_km=60.)
    e.p=SimpleNamespace(T_cont=35.,erosion_law='linear',tau_oro_myr=np.inf,
        flow_T_km=200.,flow_min_T_km=None,kappa_km2_myr=0.,
        T_max_km=65.,arc_to_cont_T_km=31.)
    e.t=0.
    e.budget={'eroded':0.,'arc_converted':0.}
    return e


def test_core_and_outward_halo_are_continuous():
    points=m.previous.lonlat_to_xyz(np.array([106.,103.2,102.5,0.]),
                                    np.array([31.,28.8,30.,0.]))
    w,_,_=m.region_weights(points)
    assert w[0] == 1.
    assert np.isclose(w[1],1.)
    assert 0. < w[2] < 1.
    assert w[3] == 0.
    assert np.all((w>=0.) & (w<=1.))


def test_plateau_is_the_only_difference_between_cases(monkeypatch):
    points=m.previous.lonlat_to_xyz(np.array([107.,106.,0.]),np.array([25.,31.,0.]))
    b,p,_=m.region_weights(points)
    monkeypatch.setattr(m,'CASE','basin_only')
    assert np.array_equal(m.resistance(points),b)
    monkeypatch.setattr(m,'CASE','joint')
    assert np.array_equal(m.resistance(points),np.maximum(b,p))
    assert p[0] > b[0]
    assert m.resistance(points)[1] == 1.


def test_repeated_overflow_fills_partial_capacity_but_preserves_core(monkeypatch):
    monkeypatch.setattr(m.previous.kin,'total_from',lambda *args:np.eye(3))
    monkeypatch.setattr(m,'CASE','basin_only')
    ends=[]
    for strength in [.5,1.]:
        monkeypatch.setattr(m,'region_weights',lambda points,w=strength:
            (np.array([w,0.,0.]),np.zeros(3),[np.array([True,False,False]),np.zeros(3,bool)]))
        e=engine()
        initial=float(e.c.T.sum())
        injected=0.
        for _ in range(8):
            # 外部供给并排走外围增量，避免有限试验域全部填满。
            injected+=85.-e.c.T[1]+35.-e.c.T[2]
            e.c.T[1:]=[85.,35.]
            m.candidate_surface(e,1.,np.ones((3,3)))
            assert np.isclose(e.c.T.sum(),initial+injected)
            assert e.c.T.min()>=35.
        ends.append(float(e.c.T[0]))
    assert ends[0] > 60.
    assert np.isclose(ends[1],35.)
