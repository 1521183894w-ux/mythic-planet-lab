"""使用可解析的球面矩形映射核对厚度参照误差，不以体积恒等式冒充精度。"""
import numpy as np
import pytest
from scipy.special import roots_legendre
from mythic.tectonics.geometry import R_KM
from mythic.tectonics.material_fv import update_material_mean,thickness_error_integrals


def rectangle_samples(z0,z1,longitude_width):
    # 在z中点分段，使绝对误差的折点也是求积边界。
    x,w=roots_legendre(4);middle=(z0+z1)/2
    z=np.r_[z0+(x+1)/2*(middle-z0),middle+(x+1)/2*(z1-middle)]
    weights=np.r_[w*(middle-z0)/2,w*(z1-middle)/2]*R_KM**2*longitude_width
    return z[None],weights[None]


def test_nonuniform_spherical_map_has_closed_form_l1_and_rms():
    # lon'=lon，z'=z+alpha(1-z²)，J=1-2alpha*z；球面面积元R² dlon dz。
    alpha=.2;z0,z1=-.3,.5;width=.17;h0=34.
    z,w=rectangle_samples(z0,z1,width)
    a0=R_KM**2*width*(z1-z0)
    g=lambda v:v+alpha*(1-v*v)
    moved=R_KM**2*width*(g(z1)-g(z0))
    h=update_material_mean(np.array([h0]),np.array([a0]),np.array([moved]))
    values=thickness_error_integrals(h,np.array([h0]),w,1-2*alpha*z)
    average_j=moved/a0
    expected_l1=abs(alpha)*(z1-z0)/(2*average_j)
    expected_rms=abs(alpha)*(z1-z0)/(np.sqrt(3)*average_j)
    np.testing.assert_allclose(values['l1_volume_error_km3']/(h0*a0),expected_l1,rtol=1e-13)
    np.testing.assert_allclose(np.sqrt(values['square_relative_area_km2']/moved),expected_rms,rtol=1e-13)


def test_uniform_area_change_reproduces_analytic_constant_thickness():
    z,w=rectangle_samples(-.2,.3,.1)
    a0=w.sum();factor=1.7
    h=update_material_mean(np.array([30.]),np.array([a0]),np.array([factor*a0]))
    values=thickness_error_integrals(h,np.array([30.]),w,np.full_like(z,factor))
    np.testing.assert_allclose(h,30./factor)
    assert values['l1_volume_error_km3'][0]<1e-7


def test_nonpositive_geometry_is_rejected():
    with pytest.raises(ValueError,match='材料面积'):
        update_material_mean(np.array([30.]),np.array([10.]),np.array([-10.]))
