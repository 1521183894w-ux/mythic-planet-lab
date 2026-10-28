"""心射插值的几何、导数与失败行为；不是世界质量验收。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.boundary_transport import GnomonicRotationMap,aligned_network,constrained_connectivity
from mythic.tectonics.geometry import lonlat_to_xyz,R_KM,omega_rotmat,unit
from mythic.tectonics.material_band import reference_strip,spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.reverse import PlateHistory

@pytest.fixture
def field(tmp_path):
    h=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        h.add(plate,0,str(plate),[(t,omega_rotmat(np.array([0.,0.,sign*5/R_KM]),-t)) for t in range(6)])
    rot=pg.RotationModel(str(h.write_rot(tmp_path/'opening.rot')))
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    features=aligned_network(strip,rot,5,plates=(1,2),tag="test")
    model=pg.TopologicalModel(features,rot)
    net=model.topological_snapshot(5.).get_resolved_topologies()[0]
    return GnomonicRotationMap(net,constrained_connectivity(net)),rot,strip

def test_common_rotation_has_exact_unit_jacobian(field):
    m,_,_=field
    x=lonlat_to_xyz(np.array([-.5,0.,.5]),np.array([-1.,0.,1.]))
    r=pg.FiniteRotation(pg.PointOnSphere(40,50),.13)
    moved,j=m.step_with_jacobian(x,{1:r,2:r})
    expected=np.array([(r*pg.PointOnSphere(tuple(p))).to_xyz() for p in x])
    np.testing.assert_allclose(moved,expected,atol=3e-15,rtol=0)
    np.testing.assert_allclose(j,1.,atol=2e-15,rtol=0)

def test_analytic_jacobian_matches_independent_spherical_patch_area(field):
    m,rot,_=field
    centers=unit(m.xyz[m.triangles].sum(axis=1))
    # 只选择离两板块交界面边缘足够远的面中心；微斑块不跨插值导数断面。
    use=np.ptp(m.plate[m.triangles],axis=1)>0
    centers=centers[use][::3]
    rr={p:rot.get_rotation(4.75,p,5.) for p in [1,2]}
    _,analytic=m.step_with_jacobian(centers,rr)
    for radius,tol in [(.1,2e-6),(.01,2e-5)]:
        pp,pt=small_material_patches(centers,radius)
        moved=m.rotation_map(pp,rr)
        numerical=spherical_triangle_area(moved,pt).reshape(-1,4).sum(1)/spherical_triangle_area(pp,pt).reshape(-1,4).sum(1)
        np.testing.assert_allclose(numerical,analytic,atol=tol,rtol=0)

def test_original_material_boundary_is_supported_without_micro_patch(field):
    m,_,strip=field
    m.weights(strip['xyz'][strip['boundary']])

def test_outside_points_are_not_extrapolated(field):
    m,_,_=field
    with pytest.raises(ValueError,match="之外"):
        m.step_with_jacobian(lonlat_to_xyz([30.],[0.]),{1:pg.FiniteRotation(),2:pg.FiniteRotation()})
