import numpy as np
import pytest
import shapely
from mythic.tectonics.conforming_edges import shared_boundary
from mythic.tectonics.geometry import unit

def project(v):
    return v[:,:2]/v[:,2,None]

def test_t_junction_has_common_edges_and_no_overlap():
    v=unit(np.array([[0,0,1],[2,0,1],[0,1,1],[1,0,1],[0,-1,1]],float))
    tri=np.array([[0,1,2],[1,3,4],[3,0,4]])
    pool,bounds,incidence,splits=shared_boundary(v,tri,project,segments=8)
    assert len(splits)==1 and splits[0][3]==3
    assert sorted(len(x) for x in incidence.values())==[1,1,1,1,2,2,2]
    polygons=[shapely.Polygon(project(pool[x])) for x in bounds]
    assert all(p.is_valid for p in polygons)
    assert abs(sum(p.area for p in polygons)-shapely.union_all(polygons).area)<1e-12
    assert len(set(bounds[0]) & set(bounds[1])) >= 8
    assert len(set(bounds[0]) & set(bounds[2])) >= 8

def test_rejects_invalid_sampling():
    v=unit(np.array([[0,0,1],[1,0,1],[0,1,1]],float))
    with pytest.raises(ValueError):
        shared_boundary(v,[[0,1,2]],project,segments=0)
