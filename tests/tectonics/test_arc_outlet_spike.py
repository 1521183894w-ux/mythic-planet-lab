from types import SimpleNamespace
import numpy as np
import pytest
from mythic.tectonics.arc_outlet_spike import finite_pair_outlet,apply_arc_outlet,outlet_edges
from mythic.tectonics.crust import Crust,CONT,OCEAN,ORIGIN_ARC,ORO_COLL
from mythic.tectonics.geometry import lonlat_to_xyz


def test_single_pair_matches_analytic_implicit_exchange_without_active_limit():
    T=np.array([85.,7.]);A=np.array([2.,5.])
    out,dv,flux=finite_pair_outlet(T,A,[0],[1],[60.],.05)
    D=10000.*((85-40)/25)**3
    g=D*2/(3*60**2)*3.5
    exact=.05*g*(85-7)/(1+.05*g*(1/2+1/5))
    assert np.allclose(flux,[exact],rtol=1e-12)
    assert np.allclose(out,T+np.array([-exact,exact])/A,atol=1e-12)
    assert abs(np.dot(out-T,A))<1e-11


def test_many_receivers_share_only_available_donor_volume():
    T=np.array([100.]+[7.]*6);A=np.ones(7)
    out,dv,flux=finite_pair_outlet(T,A,[0]*6,np.arange(1,7),np.full(6,60.),1e8)
    assert out[0]>=40.-1e-12
    assert np.isclose(flux.sum(),60.)
    assert out.min()>=T.min() and out.max()<=T.max()
    assert abs(np.dot(out-T,A))<1e-11


def test_many_donors_do_not_create_receiver_overshoot():
    T=np.array([100.]*6+[7.]);A=np.ones(7)
    out,_,_=finite_pair_outlet(T,A,np.arange(6),[6]*6,np.full(6,60.),1e8)
    assert out.min()>=7.-1e-12 and out.max()<=100.+1e-12
    assert abs(np.sum(out-T))<1e-10


def test_zero_dt_and_no_edges_are_identity():
    T=np.array([85.,7.]);A=np.ones(2)
    out,_,_=finite_pair_outlet(T,A,[0],[1],[60.],0.)
    assert np.array_equal(out,T)
    out,_,_=finite_pair_outlet(T,A,[],[],[],1.)
    assert np.array_equal(out,T)


def engine():
    c=Crust(3);c.T[:]=[100.,7.,35.]
    c.ctype[:]=[CONT,OCEAN,CONT];c.origin[:]=[ORIGIN_ARC,0,101]
    c.plate[:]=[920,920,803];c.oro[:]=ORO_COLL
    return SimpleNamespace(c=c,alive=np.ones(3,bool),area=np.ones(3),
        pedges=np.array([[0,1],[1,2]]),t=0.,m=SimpleNamespace(h_km=60.),
        p=SimpleNamespace(flow_T_km=40.,flow_min_T_km=32.5,arc_to_cont_T_km=31.),
        budget={'arc_converted':0.})


def test_native_outlet_preserves_nonarc_continent_and_inventory():
    e=engine();pos=lonlat_to_xyz(np.array([0.,.55,1.1]),np.zeros(3))
    before=e.c.T.copy()
    row=apply_arc_outlet(e,100.,pos,contacts=False)
    assert np.isclose(e.c.T.sum(),before.sum())
    assert e.c.T[0]>=40.
    assert e.c.T[2]==35.
    assert e.c.ctype[1]==CONT and e.c.origin[1]==ORIGIN_ARC
    assert row['converted_cells']==1 and row['contact_transported_km3']==0.


def test_collision_contact_is_local_and_rejects_near_coincident_centers():
    e=engine();e.pedges=np.empty((0,2),int)
    pos=lonlat_to_xyz(np.array([0.,.1,.55]),np.zeros(3))
    s,r,d,k=outlet_edges(e,pos,contacts=True)
    assert np.array_equal(s,[0]) and np.array_equal(r,[2])
    assert k[0]==1
    e.c.oro[0]=0
    assert len(outlet_edges(e,pos,contacts=True)[0])==0


@pytest.mark.parametrize('area,dt',[([0.,1.],1.),([1.,1.],-1.)])
def test_invalid_inputs(area,dt):
    with pytest.raises(ValueError):
        finite_pair_outlet([85.,7.],area,[0],[1],[60.],dt)
