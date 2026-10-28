"""分片裁切独立几何检查；不放宽既有严格刚体检查。"""
import numpy as np
import pygplates as pg
import pytest
from mythic.tectonics.boundary_transport import GnomonicRotationMap,aligned_network,constrained_connectivity
from mythic.tectonics.geometry import lonlat_to_xyz,R_KM,omega_rotmat
from mythic.tectonics.material_band import reference_strip,spherical_triangle_area
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.piecewise_audit import crossing_reference_faces,split_on_reference_faces

@pytest.fixture
def mapper(tmp_path):
    history=PlateHistory()
    for plate,sign in [(1,-1),(2,1)]:
        history.add(plate,0,str(plate),[(t,omega_rotmat(np.array([0.,0.,sign*5/R_KM]),-t)) for t in range(6)])
    rot=pg.RotationModel(str(history.write_rot(tmp_path/'opening.rot')))
    strip=reference_strip(lonlat_to_xyz(0,-3),lonlat_to_xyz(0,3),200,60)
    feat=aligned_network(strip,rot,5,plates=(1,2),tag='audit_test')
    net=pg.TopologicalModel(feat,rot).topological_snapshot(5.).get_resolved_topologies()[0]
    return GnomonicRotationMap(net,constrained_connectivity(net))

def test_split_preserves_each_parent_spherical_area_and_known_crossing(mapper):
    # 跨多片的单个三角形，不与既有边重合。
    v=lonlat_to_xyz(np.array([-.6,.6,0.]),np.array([-1.7,-1.6,1.7]))
    tri=np.array([[0,1,2]])
    assert crossing_reference_faces(mapper,v,tri)[0]
    before=abs(spherical_triangle_area(v,tri)).sum()
    w,cut,parent,diag=split_on_reference_faces(mapper,v,tri)
    assert len(cut)>1 and np.all(parent==0)
    assert not crossing_reference_faces(mapper,w,cut).any()
    np.testing.assert_allclose(abs(spherical_triangle_area(w,cut)).sum(),before,rtol=1e-10,atol=1e-7)
    assert diag['tiny_subfragment_area_km2']<1e-7

def test_existing_face_is_not_marked_crossing(mapper):
    assert not crossing_reference_faces(mapper,mapper.xyz,mapper.triangles).any()

def test_split_rejects_uncovered_cell(mapper):
    v=lonlat_to_xyz(np.array([-.5,.5,10.]),np.array([-1.,-1.,0.]))
    with pytest.raises(ValueError,match='完整参考面支撑'):
        split_on_reference_faces(mapper,v,np.array([[0,1,2]]))