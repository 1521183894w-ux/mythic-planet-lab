"""用独立解析球面速度场检查散度审计，不以框架自己的应变作真值。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,unit
from mythic.tectonics.material_audit import velocity_divergence,cumulative_trapezoid


class AnalyticSnapshot:
    def __init__(self,field):self.field=field
    def get_point_velocities(self,points,**kw):
        return [pg.Vector3D(*v) for v in self.field(np.asarray(points))]


@pytest.mark.parametrize('h',[1.,.1,.01])
def test_spherical_divergence_of_rigid_rotation_is_zero(h):
    x=lonlat_to_xyz(np.array([0.,20.,-40.]),np.array([-20.,30.,60.]))
    w=np.array([.2,-.3,.4])/R_KM
    snapshot=AnalyticSnapshot(lambda q:R_KM*np.cross(w,q))
    np.testing.assert_allclose(velocity_divergence(snapshot,x,probe_km=h),0.,atol=1e-10)


@pytest.mark.parametrize('h',[1.,.1,.01])
def test_curved_surface_gradient_has_known_nonzero_divergence(h):
    x=lonlat_to_xyz(np.array([0.,20.,-40.]),np.array([-20.,30.,60.]))
    axis=unit([.2,-.3,.4]);rate=.02
    snapshot=AnalyticSnapshot(lambda q:rate*R_KM*(axis-(q@axis)[:,None]*q))
    expected=-2*rate*(x@axis)
    np.testing.assert_allclose(velocity_divergence(snapshot,x,probe_km=h),expected,rtol=5e-8,atol=1e-10)


def test_elapsed_time_integration_is_signed_and_matches_affine_rate():
    t=np.array([0.,.25,.5,1.,2.])
    rate=np.column_stack([2*t+3,-t])
    actual=cumulative_trapezoid(rate,t)
    np.testing.assert_allclose(actual,np.column_stack([t*t+3*t,-t*t/2]),atol=1e-14)
    with pytest.raises(ValueError):cumulative_trapezoid(rate,t[::-1])
