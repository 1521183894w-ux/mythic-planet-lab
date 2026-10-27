"""守恒展开的物质闭合、共享源、顺序不变和引擎元数据。"""
from types import SimpleNamespace
import numpy as np
import pytest
from mythic.tectonics.stretching import conservative_area_expand as expand
from mythic.tectonics.crust import Crust, CONT, Params, ORIGIN_RIFT
from mythic.tectonics.engine import Engine


def test_uniform_area_doubling_halves_both_columns():
    r = expand([35.], [100.], [100.], np.array([[0]]), [[1.]])
    np.testing.assert_allclose(r["source_thickness"], [17.5])
    np.testing.assert_allclose(r["target_thickness"], [17.5])
    assert r["source_beta"][0] == 2.
    assert r["edge_volume"][0] == 1750.


def test_unequal_areas_shared_sources_conserve_volume_and_ignore_target_order():
    h, a, target = np.array([35., 46., 22.]), np.array([3., 7., 11.]), np.array([5., 9., 13.])
    nb = np.array([[0, 1], [0, 2], [1, 2]])
    w = np.array([[1., 3.], [4., 2.], [7., 5.]])
    r = expand(h, a, target, nb, w)
    assert r["source_thickness"] @ a + r["target_thickness"] @ target == pytest.approx(h @ a, abs=2e-13)
    assert np.all((r["source_thickness"] > 0) & (r["source_thickness"] <= h))
    assert np.sum(r["edge_volume"]) == pytest.approx(r["target_thickness"] @ target)
    np.testing.assert_allclose(np.bincount(r["row"], r["edge_area"]), target)
    order = np.array([2, 0, 1])
    other = expand(h, a, target[order], nb[order], w[order])
    np.testing.assert_allclose(other["source_thickness"], r["source_thickness"], rtol=1e-14)
    np.testing.assert_allclose(other["target_thickness"], r["target_thickness"][order], rtol=1e-14)
    np.testing.assert_array_equal(h, [35., 46., 22.])


def test_duplicate_stencil_entries_are_aggregated_and_unsupported_target_fails():
    r = expand([30., 40.], [3., 5.], [6.], np.array([[0, 0, 1]]), [[1., 2., 3.]])
    direct = expand([30., 40.], [3., 5.], [6.], np.array([[0, 1]]), [[3., 3.]])
    np.testing.assert_array_equal(r["source_thickness"], direct["source_thickness"])
    np.testing.assert_array_equal(r["target_thickness"], direct["target_thickness"])
    with pytest.raises(ValueError, match="没有源材料"):
        expand([35.], [1.], [1.], np.array([[0]]), [[0.]])
    empty = expand([35.], [1.], [], np.empty((0, 1), int), np.empty((0, 1)))
    np.testing.assert_array_equal(empty["source_thickness"], [35.])


def test_engine_transfers_old_material_and_does_not_label_it_new_crust():
    eng = object.__new__(Engine)
    eng.m = SimpleNamespace(area_km2=np.array([100., 200., 100.]))
    eng.budget, eng.t = {}, 50.
    source, g = Crust(2), Crust(1)
    source.plate[:] = [301, 401]
    source.ctype[:] = CONT
    source.T[:] = [35., 40.]
    source.uid[:] = [10, 11]
    source.age[:] = [900., 700.]
    source.prov[:] = [101, 102]
    source.t_form[:] = [1200., 1000.]
    g.node[:] = 2
    g.uid[:] = 12
    eng._expand_rift_area(source, g, np.array([True]), np.array([[0, 1]]),
                          np.array([[.01, .02]]), np.array([301401]))
    assert source.T @ np.array([100., 200.]) + g.T[0] * 100. == pytest.approx(11500.)
    assert g.origin[0] == ORIGIN_RIFT and g.ctype[0] == CONT
    assert g.t_form[0] in [1200., 1000.] and g.oro_t[0] == 50.
    assert eng.budget.get("rift_fill", 0.) == 0.
    assert eng.budget["rift_transferred"] == pytest.approx(g.T[0] * 100.)
    assert abs(eng.rift_area_audit[0]["closure_km3"]) < 1e-10
    assert Params().rift_area_conservative is False
