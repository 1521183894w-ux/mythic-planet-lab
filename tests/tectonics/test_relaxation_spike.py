import numpy as np
import pytest
from mythic.tectonics.relaxation_spike import relax_thickness


def test_unequal_area_two_cell_has_known_implicit_solution():
    T=np.array([65.,35.]); A=np.array([2.,5.])
    out,dv,info=relax_thickness(T,A,[[0,1]],[1.],2.,60.,3000.)
    g=3000.*2/(3*60**2)*3.5
    f=2*g*(T[1]-T[0])/(1+2*g*(1/A[0]+1/A[1]))
    assert np.allclose(out,T+np.array([f,-f])/A,atol=1e-12)
    assert np.allclose(dv,[f,-f],atol=1e-12)
    assert abs(info['balance_km3'])<1e-11


def test_barrier_no_tunneling_and_disconnected_cell_unchanged():
    T=np.array([100.,35.,35.,17.])
    out,dv,_=relax_thickness(T,[2.,3.,5.,7.],[[0,1],[1,2]],[0.,1.],50.,60.)
    assert np.array_equal(out,T)
    assert np.array_equal(dv,np.zeros(4))


def test_large_step_preserves_inventory_and_has_no_new_extrema():
    rng=np.random.default_rng(617)
    T=rng.uniform(35.,130.,80); A=rng.uniform(1000.,5000.,80)
    e=np.column_stack([np.arange(79),np.arange(1,80)])
    out,dv,info=relax_thickness(T,A,e,np.ones(79),100.,60.,10000.)
    assert T.min()-1e-10<=out.min()<=out.max()<=T.max()+1e-10
    assert abs(np.dot(out-T,A))<1e-6
    assert np.max(np.abs((out-T)*A-dv))<1e-6
    assert info['active_edges']==79


def test_zero_time_identity_and_permutation_equivariance():
    T=np.array([100.,55.,35.]); A=np.array([2.,3.,5.]); e=np.array([[0,1],[1,2]])
    out,_,_=relax_thickness(T,A,e,[.3,1.],0.,60.)
    assert np.array_equal(out,T)
    out,_,_=relax_thickness(T,A,e,[.3,1.],2.,60.)
    order=np.array([2,0,1]); inv=np.argsort(order)
    alt,_,_=relax_thickness(T[order],A[order],inv[e],[.3,1.],2.,60.)
    assert np.allclose(alt[inv],out,atol=1e-12)


@pytest.mark.parametrize('area,mob,dt',[([1.,0.],[1.],1.),([1.,1.],[-1.],1.),([1.,1.],[1.],-1.)])
def test_invalid_input(area,mob,dt):
    with pytest.raises(ValueError):
        relax_thickness([65.,35.],area,[[0,1]],mob,dt,60.)
