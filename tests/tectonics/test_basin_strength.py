"""有限阻流对照：保留原算子等价性、解析体积账和输入约束。"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('strength',Path(__file__).resolve().parents[2]/'scripts/design_basin_strength.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def problem():
    return np.array([70.,35.,42.]),np.array([2.,3.,1.]),np.array([[0,1],[1,2]]),np.array([0.,.4])

def test_zero_floor_reproduces_prior_solver_exactly(monkeypatch):
    T,A,E,M=problem();monkeypatch.setattr(m,'FLOOR',0.)
    old=m.original_relax(T,A,E,M,1.,60.,10000.)
    new=m.finite_strength_relax(T,A,E,M,1.,60.,10000.)
    assert np.array_equal(old[0],new[0])
    assert np.array_equal(old[1],new[1])
    assert old[2]==new[2]

def test_finite_floor_opens_blocked_edge_with_analytic_mass_balance(monkeypatch):
    T=np.array([70.,35.]);A=np.array([2.,3.]);E=np.array([[0,1]])
    monkeypatch.setattr(m,'FLOOR',.01)
    new,dv,info=m.finite_strength_relax(T,A,E,np.zeros(1),1.,60.,10000.)
    D=10000.*((70.-40.)/25.)**3*.01
    conductance=D*2/(3*60.**2)*A.mean()
    difference=(T[0]-T[1])/(1+conductance*(1/A[0]+1/A[1]))
    volume=conductance*difference
    assert np.allclose(new,T+np.array([-volume/A[0],volume/A[1]]))
    assert np.allclose(dv,[-volume,volume])
    assert abs(np.dot(new-T,A))<1e-12
    assert T.min()<=new.min()<=new.max()<=T.max()

def test_full_floor_removes_region_contrast(monkeypatch):
    T,A,E,M=problem();monkeypatch.setattr(m,'FLOOR',1.)
    old=m.original_relax(T,A,E,np.ones(2),1.,60.,10000.)
    new=m.finite_strength_relax(T,A,E,M,1.,60.,10000.)
    assert np.array_equal(old[0],new[0])

@pytest.mark.parametrize('floor,mob',[(-.1,0.),(1.1,0.),(np.nan,0.),(.01,-.1),(.01,1.1),(.01,np.nan)])
def test_invalid_coefficient_is_not_hidden(monkeypatch,floor,mob):
    monkeypatch.setattr(m,'FLOOR',floor)
    with pytest.raises(ValueError):
        m.finite_strength_relax(np.array([70.,35.]),np.ones(2),np.array([[0,1]]),np.array([mob]),1.,60.)
