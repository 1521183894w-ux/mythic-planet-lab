"""出生事件与宿主通量的口径回归。"""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from design_b5_sources import birth_balance
from design_b5_sources_report import flow_groups

def test_birth_counts_one_hole_once_with_multiple_donors():
    borrowed,created=birth_balance([100.,80.],[[30.,20.],[10.,0.]])
    np.testing.assert_allclose(borrowed,[50.,10.])
    np.testing.assert_allclose(created,[50.,70.])
    assert sum(created)+sum(borrowed)==180.

def test_birth_does_not_hide_excess_borrowing():
    with pytest.raises(ValueError,match="超过"):
        birth_balance([100.],[[80.,30.]])

@pytest.mark.parametrize("total,shares", [([np.nan],[[0.]]),([1.],[[-1.]]),([1.],[[np.inf]])])
def test_invalid_birth_cannot_enter_audit(total,shares):
    with pytest.raises(ValueError):
        birth_balance(total,shares)

def test_flow_groups_are_signed_whole_host_flows_and_exclude_other_targets():
    births=[{"uid":10,"created_km3":2.},{"uid":11,"created_km3":0.}]
    rows=[{"target_uid":52985,"neighbor_uid":10,"volume_into_target_km3":8.},
          {"target_uid":52985,"neighbor_uid":10,"volume_into_target_km3":-3.},
          {"target_uid":52985,"neighbor_uid":11,"volume_into_target_km3":5.},
          {"target_uid":10,"neighbor_uid":52985,"volume_into_target_km3":-8.}]
    groups=flow_groups(rows,births)
    assert groups["created_birth_host"]=={"incoming_km3":8.,"outgoing_km3":3.,"net_km3":5.}
    assert groups["other"]["net_km3"]==5.
    # 宿主流入8大于出生新增2是允许的，禁止把它命名为新增成分通量。

def test_empty_rift_birth_needs_no_coordinate_query():
    from design_b5_sources import SourceObserver
    obs=SourceObserver.__new__(SourceObserver)
    obs.rift_birth(None,{"g":None,"cg":np.array([False,False])})

def test_rift_birth_records_full_area_volume_without_modifying_crust():
    from types import SimpleNamespace as NS
    from design_b5_sources import SourceObserver
    obs=SourceObserver.__new__(SourceObserver)
    obs.births={}
    obs.coordinates=lambda eng,node,plate:np.array([[18.7,-3.]])
    g=NS(uid=np.array([7]),node=np.array([0]),plate=np.array([301]),origin=np.array([91]),T=np.array([30.]))
    eng=NS(t=10.,m=NS(area_km2=np.array([2.])))
    obs.rift_birth(eng,{"g":g,"cg":np.array([True])})
    assert obs.births[7]["created_km3"]==60.
    assert obs.births[7]["borrowed_km3"]==0.
    assert obs.births[7]["kind"]=="rift_fill"
    np.testing.assert_array_equal(g.T,[30.])
