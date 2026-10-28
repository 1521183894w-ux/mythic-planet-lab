"""S10索引查询须与S9逐面查询等价，错误支撑仍拒绝。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.rotation_interpolation import BoundaryRotationMap
from mythic.tectonics.rotation_transport import IndexedRotationMap
from mythic.tectonics.geometry import lonlat_to_xyz,R_KM,omega_rotmat
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.material_band import reference_strip,build_network

@pytest.fixture
def network(tmp_path):
    h=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        h.add(plate,0,str(plate),[(t,omega_rotmat(np.array([0.,0.,sign*5/R_KM]),-t)) for t in range(6)])
    rot=pg.RotationModel(str(h.write_rot(tmp_path/'opening.rot')))
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    _,features=build_network(strip,rot,5,plates=(1,2))
    net=pg.TopologicalModel(features,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    return net,rot

def test_index_retains_same_faces_weights_and_finite_map(network):
    net,rot=network
    plain=BoundaryRotationMap(net)
    fast=IndexedRotationMap(net,plain.connectivity)
    rng=np.random.default_rng(3610)
    x=lonlat_to_xyz(rng.uniform(-.8,.8,4096),rng.uniform(-2.8,2.8,4096))
    a,w=plain.weights(x);b,z=fast.weights(x)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(w,z,atol=3e-14,rtol=0.)
    rr={p:rot.get_rotation(4.75,p,5.) for p in [1,2]}
    np.testing.assert_allclose(plain.rotation_map(x,rr),fast.rotation_map(x,rr),atol=2e-15,rtol=0.)

def test_index_reports_outside_and_refuses_mapping(network):
    net,rot=network
    fast=IndexedRotationMap(net)
    x=lonlat_to_xyz(np.array([0.,30.]),np.array([0.,0.]))
    _,flags=fast.support_diagnostics(x)
    np.testing.assert_array_equal(flags['outside'],[False,True])
    with pytest.raises(ValueError,match='之外'):
        fast.rotation_map(x,{1:pg.FiniteRotation(),2:pg.FiniteRotation()})

def test_index_rejects_multiple_strict_interior_faces(network):
    net,_=network
    plain=BoundaryRotationMap(net)
    graph=plain.connectivity+plain.connectivity
    fast=IndexedRotationMap(net,graph)
    from mythic.tectonics.geometry import unit
    x=unit(plain.xyz[plain.triangles[:1]].sum(axis=1))
    with pytest.raises(ValueError,match='重叠'):fast.weights(x)

def test_index_and_original_agree_on_all_mesh_centroid_support(network):
    net,_=network
    plain=BoundaryRotationMap(net)
    fast=IndexedRotationMap(net,plain.connectivity)
    from mythic.tectonics.geometry import unit
    for point in unit(plain.xyz[plain.triangles].sum(axis=1)):
        x=point[None,:]
        try:a,w=plain.weights(x)
        except ValueError as error:
            with pytest.raises(ValueError,match=str(error)):fast.weights(x)
        else:
            b,z=fast.weights(x)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(w,z,atol=3e-14,rtol=0.)
