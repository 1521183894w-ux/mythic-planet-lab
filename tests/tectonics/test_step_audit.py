"""用已知球面旋转检验稳健小角度位移率，不把框架标量作真值。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,omega_rotmat,unit
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.material_band import reference_strip,build_network
from mythic.tectonics.step_audit import sphere_log_velocity,map_velocity,network_map,surface_divergence


@pytest.mark.parametrize('angle',[1e-9,1e-7,1e-6,1e-4])
def test_log_map_retains_known_small_rotation(angle):
    x=lonlat_to_xyz(np.array([0.,40.]),np.array([0.,0.]))
    y=x@omega_rotmat(np.array([0.,0.,1.]),angle).T
    actual=sphere_log_velocity(x,y,1.)
    expected=R_KM*angle*np.cross([0.,0.,1.],x)
    np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=1e-12)


def test_zero_map_and_invalid_time_or_antipode_are_explicit():
    x=lonlat_to_xyz(20,30)[None,:]
    np.testing.assert_allclose(sphere_log_velocity(x,x,1.),0.,atol=1e-12)
    for dt in [0.,np.nan,np.inf]:
        with pytest.raises(ValueError):sphere_log_velocity(x,x,dt)
    with pytest.raises(ValueError):sphere_log_velocity(x,-x,1.)


def test_past_mapping_has_correct_forward_sign():
    x=lonlat_to_xyz(0,0)[None,:]
    y=x@omega_rotmat(np.array([0.,0.,1.]),-1e-5).T
    np.testing.assert_allclose(sphere_log_velocity(x,y,-.5),[[0.,2e-5*R_KM,0.]],atol=1e-10)


@pytest.fixture
def rigid_network(tmp_path):
    h=PlateHistory();omega=np.array([0.,0.,5/R_KM])
    for p in [1,2]:h.add(p,0,str(p),[(t,omega_rotmat(omega,-t)) for t in [0,1,2,3]])
    rot=pg.RotationModel(str(h.write_rot(tmp_path/'rigid.rot')))
    strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60)
    return build_network(strip,rot,2,plates=(1,2),tag='rigid')[0]


def test_common_rotation_network_matches_its_input_rigid_rotation(rigid_network):
    snap=rigid_network.topological_snapshot(1.)
    x=lonlat_to_xyz(np.array([-.2,.2]),np.array([-1.,1.]))
    v=map_velocity(snap,x,.001)
    rotation=rigid_network.get_rotation_model().get_rotation(.999,1,1.)
    exact=np.array([(rotation*pg.PointOnSphere(tuple(q))).to_xyz() for q in x])
    expected=sphere_log_velocity(x,exact,.001)
    np.testing.assert_allclose(v,expected,rtol=2e-7,atol=1e-7)
    div=surface_divergence(lambda q:map_velocity(snap,q,.01,centered=True),x)
    np.testing.assert_allclose(div,0.,atol=2e-7)


def test_network_outside_is_rejected_without_fill(rigid_network):
    with pytest.raises(ValueError):network_map(rigid_network.topological_snapshot(1.),lonlat_to_xyz(90,0)[None,:],.9)
