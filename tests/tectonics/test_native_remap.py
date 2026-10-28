"""球面一阶矩闭式值及部分覆盖守恒；不以自身体积恒等式验证场精度。"""
import numpy as np
from shapely.geometry import Polygon,box
from mythic.tectonics.geometry import R_KM,unit
from mythic.tectonics.source_clipping import LocalGnomonic
from mythic.tectonics.native_remap import spherical_moments,overlap_integrals


def test_spherical_octant_area_and_first_moment_are_analytic():
    p=LocalGnomonic(unit([1.,1.,1.]))
    triangle=Polygon(p.forward(np.eye(3)))
    area,moment=spherical_moments(triangle,p)
    np.testing.assert_allclose(area,np.pi/2*R_KM**2,rtol=1e-14)
    np.testing.assert_allclose(moment,np.full(3,np.pi/4*R_KM**2),rtol=1e-14)
    # 颠倒环方向不得颠倒物理面积/矩。
    a2,m2=spherical_moments(Polygon(list(triangle.exterior.coords)[::-1]),p)
    np.testing.assert_allclose([a2], [area],rtol=1e-14)
    np.testing.assert_allclose(m2,moment,rtol=1e-14)


def test_partial_coverage_keeps_unmapped_volume_and_linear_integral():
    p=LocalGnomonic(unit([1.,.2,.1]));source=box(-40,-30,40,30)
    left=source.intersection(box(-100,-100,7,100));right=source.difference(left)
    a,m=spherical_moments(source,p);centroid=m/a
    result=overlap_integrals([left,right],[source],p,np.array([35.]),centroid[None],np.array([[5.,-2.,3.]]))
    np.testing.assert_allclose(result['area_km2'].sum(),a,atol=1e-7)
    np.testing.assert_allclose(result['constant_volume_km3'].sum(),a*35,atol=1e-6)
    np.testing.assert_allclose(result['linear_volume_km3'].sum(),a*35,atol=1e-6)
    assert 0<result['constant_volume_km3'][0]<a*35


def test_polygon_hole_is_subtracted_from_area_and_moment():
    p=LocalGnomonic(unit([1.,.2,.1]));outer=box(-40,-30,40,30);hole=box(-3,-5,9,8)
    ao,mo=spherical_moments(outer,p);ah,mh=spherical_moments(hole,p)
    area,moment=spherical_moments(outer.difference(hole),p)
    np.testing.assert_allclose(area,ao-ah,atol=1e-8)
    np.testing.assert_allclose(moment,mo-mh,atol=1e-8)


def test_precision_thin_polygon_has_physical_centroid_and_positive_linear_mass():
    p=LocalGnomonic(unit([1.,.2,.1]));thin=box(100.,30.,110.,30.+1e-10)
    area,moment=spherical_moments(thin,p)
    vertex=p.inverse(np.asarray(thin.exterior.coords)[:-1]);center=unit(vertex.sum(axis=0))
    assert area>0
    assert np.linalg.norm(moment/area)<=1+1e-12
    assert np.linalg.norm(moment/area-center)<=np.linalg.norm(vertex-center,axis=1).max()+1e-12
    gradient=1000*p.east
    volume=20*area+(moment-area*p.center)@gradient
    assert volume>0
