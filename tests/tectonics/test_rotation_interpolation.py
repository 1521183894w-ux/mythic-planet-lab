"""局部候选的独立刚体真值、换框架与官方路径复现；不豁免旧失败。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.geometry import lonlat_to_xyz,R_KM,omega_rotmat
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.material_band import reference_strip,build_network,point_array,spherical_triangle_area
from mythic.tectonics.rotation_interpolation import BoundaryRotationMap,rotation_matrix
from mythic.tectonics.material_audit import small_material_patches


@pytest.fixture
def inputs(tmp_path):
    h=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        h.add(plate,0,str(plate),[(t,omega_rotmat(np.array([0.,0.,sign*5/R_KM]),-t))
            for t in range(6)])
    rot=pg.RotationModel(str(h.write_rot(tmp_path/'opening.rot')))
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    _,features=build_network(strip,rot,5,plates=(1,2))
    return rot,features


@pytest.mark.parametrize('angle',[1e-9,1e-7,1e-6,1e-4])
def test_direct_matrix_keeps_small_rotation(angle):
    actual=rotation_matrix(pg.FiniteRotation((90.,0.),angle))
    expected=omega_rotmat(np.array([0.,0.,1.]),angle)
    np.testing.assert_allclose(actual,expected,atol=2e-16,rtol=2e-15)


def test_projected_replica_matches_framework_barycentric_branch(inputs):
    rot,features=inputs
    net=pg.TopologicalModel(features,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    mapper=BoundaryRotationMap(net)
    x=lonlat_to_xyz(np.array([-.2,.2]),np.array([-1.,1.]))
    rr={p:rot.get_rotation(4.75,p,5.) for p in [1,2]}
    actual=mapper.projected_map(x,rr)
    framework=point_array([net.reconstruct_point(tuple(p),4.75,False) for p in x])
    np.testing.assert_allclose(actual,framework,atol=3e-13,rtol=0.)


@pytest.mark.parametrize('anchor',[1,2])
def test_same_physical_mapping_after_time_dependent_frame_change(inputs,anchor):
    rot,features=inputs
    x=lonlat_to_xyz(np.array([-.2,.2]),np.array([-1.,1.]))
    template_net=pg.TopologicalModel(features,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    connectivity=BoundaryRotationMap(template_net).connectivity
    result=[]
    for a in [0,anchor]:
        model=pg.TopologicalModel(features,rot,anchor_plate_id=a)
        net=model.topological_snapshot(5.).get_resolved_topologies()[0]
        mapper=BoundaryRotationMap(net,connectivity=connectivity)
        initial=x@rotation_matrix(rot.get_rotation(5.,a))
        rr={p:rot.get_rotation(4.75,p,5.,anchor_plate_id=a) for p in [1,2]}
        mapped=mapper.rotation_map(initial,rr)
        result.append(mapped@rotation_matrix(rot.get_rotation(4.75,a)).T)
    np.testing.assert_allclose(result[0],result[1],atol=2e-12,rtol=0.)


def test_common_stage_rotation_preserves_exact_position_and_patch_area(inputs):
    rot,features=inputs
    net=pg.TopologicalModel(features,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    mapper=BoundaryRotationMap(net)
    points,tri=small_material_patches(lonlat_to_xyz(np.array([-.2,.2]),np.array([-1.,1.])),1.)
    r=pg.FiniteRotation((42.,-37.),.013)
    result=mapper.rotation_map(points,{1:r,2:r})
    exact=point_array([r*pg.PointOnSphere(tuple(p)) for p in points])
    np.testing.assert_allclose(result,exact,atol=1e-15,rtol=0.)
    a0=spherical_triangle_area(points,tri)
    np.testing.assert_allclose(spherical_triangle_area(result,tri)/a0,1.,atol=2e-8,rtol=0.)


def test_outside_points_are_rejected_instead_of_extrapolated(inputs):
    rot,features=inputs
    net=pg.TopologicalModel(features,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    mapper=BoundaryRotationMap(net)
    with pytest.raises(ValueError,match='之外'):
        mapper.rotation_map(lonlat_to_xyz(90.,0.)[None,:],{1:pg.FiniteRotation(),2:pg.FiniteRotation()})
