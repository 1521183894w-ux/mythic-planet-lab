"""分配技术验证的守恒、控制组与边界检查；不替代世界验收。"""
from dataclasses import replace
import numpy as np
import pytest
from spikes.stage1_distribution.study import Settings, geometry, run_case


@pytest.mark.parametrize("kind", ["legacy_cap", "legacy_uncapped", "uniform", "strong_basin", "directional"])
def test_closed_domain_balances_external_reservoir(kind):
    r = run_case(kind, replace(Settings(), duration_myr=3.13))
    s = r["summary"]
    assert abs(s["closure_km3"]) < 1e-5
    assert abs(s["final_volume_km3"] + s["donor_reservoir_final_km3"]
               - s["initial_volume_km3"] - s["donor_reservoir_initial_km3"]) < 1e-5
    assert s["basin_net_input_km3"] + s["corridor_net_input_km3"] + s["remaining_domain_net_input_km3"] == pytest.approx(s["external_input_km3"], abs=1e-5)
    assert np.all(r["source"][~r["core"]] == 0)
    assert np.all(r["thickness"][~r["core"]] == r["initial"][~r["core"]])
    assert r["history"][-1, 0] == pytest.approx(3.13)


def test_material_contrast_one_is_uniform_control():
    s = replace(Settings(), duration_myr=2, viscosity_contrast=1)
    assert np.array_equal(run_case("uniform", s)["thickness"], run_case("strong_basin", s)["thickness"])


def test_direction_factor_one_is_isotropic_control():
    s = replace(Settings(), duration_myr=2, cross_strike_factor=1)
    assert np.array_equal(run_case("strong_basin", s)["thickness"], run_case("directional", s)["thickness"])


@pytest.mark.parametrize("kind", ["uniform", "strong_basin", "directional"])
def test_without_source_satisfies_maximum_principle(kind):
    r = run_case(kind, replace(Settings(), duration_myr=5, source_peak_km_myr=0))
    core = r["core"]
    assert r["thickness"][core].min() >= r["initial"][core].min() - 1e-10
    assert r["thickness"][core].max() <= r["initial"][core].max() + 1e-10
    assert abs(r["summary"]["closure_km3"]) < 1e-5


def test_refinement_preserves_domain_area_and_source_integral():
    totals = []
    for dx in (40, 20):
        _, _, _, core, _, _, _, source, _ = geometry(replace(Settings(), spacing_km=dx))
        assert core.sum() * dx**2 == 4800000
        totals.append(source.sum() * dx**2)
    assert totals[0] == pytest.approx(totals[1], rel=1e-12)


def test_original_cap_control_reproduces_plateau_without_volume_loss():
    s = replace(Settings(), duration_myr=40)
    a, b = run_case("legacy_cap", s), run_case("legacy_uncapped", s)
    assert a["summary"]["area_at_65km_km2"] > 0
    assert a["summary"]["maximum_thickness_ever_km"] <= 65 + 1e-9
    assert b["summary"]["maximum_thickness_ever_km"] > 80
    assert a["summary"]["final_volume_km3"] == pytest.approx(b["summary"]["final_volume_km3"], abs=1e-5)


@pytest.mark.parametrize("change", [
    {"dt_myr": 0}, {"dt_myr": float("nan")}, {"spacing_km": 0},
    {"duration_myr": -1}, {"duration_myr": float("nan")},
    {"source_peak_km_myr": -1}, {"source_peak_km_myr": float("inf")},
    {"viscosity_contrast": 0}, {"cross_strike_factor": 0},
    {"diffusivity_km2_myr": -1},
])
def test_invalid_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        run_case("directional", replace(Settings(), **change))

def test_expanded_domain_keeps_source_and_initial_state_in_original_window():
    a = geometry(Settings())
    b = geometry(replace(Settings(), width_km=4800, height_km=4000))
    _, x, y, core, _, _, _, source, initial = b
    window = core & (np.abs(x) < 1200) & (np.abs(y) < 1000)
    assert np.all(source[core & ~window] == 0)
    assert np.allclose(source[window], a[7][a[3]], rtol=1e-14, atol=1e-14)
    assert np.array_equal(initial[window], a[8][a[3]])
