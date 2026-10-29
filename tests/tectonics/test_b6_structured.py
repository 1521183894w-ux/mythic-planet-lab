import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import Crust, CONT

spec=importlib.util.spec_from_file_location('b6_candidate',Path(__file__).resolve().parents[2]/'scripts/design_b6_structured.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_overflow_is_redirected_without_changing_basin_or_total(monkeypatch):
    e=Engine.__new__(Engine)
    e.c=Crust(3)
    e.c.T[:]=[35.,70.,35.]
    e.c.ctype[:]=CONT
    e.c.plate[:]=101
    e.alive=np.ones(3,bool)
    e.area=np.ones(3)
    e.pedges=np.array([[0,1],[1,2]])
    e.m=SimpleNamespace(h_km=60.)
    e.p=SimpleNamespace(T_cont=35.,erosion_law='linear',tau_oro_myr=np.inf,
        flow_T_km=100.,flow_min_T_km=None,kappa_km2_myr=0.,T_max_km=65.,
        arc_to_cont_T_km=31.)
    e.t=0.
    e.budget={'eroded':0.,'arc_converted':0.}
    monkeypatch.setattr(module.kin,'total_from',lambda *args:np.eye(3))
    monkeypatch.setattr(module,'in_sphere_polygon',lambda points,polygon:np.array([True,False,False]))
    pos=np.ones((3,3))
    before=float(np.dot(e.c.T,e.area))
    module.candidate_surface(e,1.,pos)
    assert np.allclose(e.c.T,[35.,65.,40.])
    assert np.isclose(np.dot(e.c.T,e.area),before)
