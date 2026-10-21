"""海岸线与分布工具测试。"""

import numpy as np
import pytest
from _synth import cap, cap_area_km2

from mythic.metrics.coast import (
    box_counting_dimension,
    coastline_development,
    coastline_length_km,
    coastline_segments,
    default_box_sizes_km,
    densify_km,
)
from mythic.metrics.distributions import powerlaw_ccdf_slope, wasserstein, weighted_ks, weighted_quantiles
from mythic.metrics.grid import LatLonGrid

G1024 = LatLonGrid(1024, 512)
G2048 = LatLonGrid(2048, 1024)


def test_polar_cap_coastline_length():
    land = cap(G1024, 90, 0, 30)  # 边界在 60°N，周长 2πR·cos60°
    L = coastline_length_km(G1024, coastline_segments(G1024, land))
    assert L == pytest.approx(2 * np.pi * G1024.radius_km * np.cos(np.radians(60)), rel=0.08)


def test_coastline_crossing_dateline_is_not_lost():
    land = cap(G1024, 0, 180, 15)
    ref = cap(G1024, 0, 0, 15)
    L1 = coastline_length_km(G1024, coastline_segments(G1024, land))
    L2 = coastline_length_km(G1024, coastline_segments(G1024, ref))
    assert L1 == pytest.approx(L2, rel=0.01)


def test_smooth_coast_dimension_near_one():
    land = cap(G2048, 30, 40, 25)
    pts = densify_km(G2048, coastline_segments(G2048, land), step_km=5)
    res = box_counting_dimension(pts, default_box_sizes_km(G2048))
    assert res["D"] == pytest.approx(1.0, abs=0.08)


def test_rough_coast_dimension_above_smooth():
    # 谱合成的分形场取 0 等值线：海岸明显曲折，维数应显著大于 1
    rng = np.random.default_rng(1)
    ny, nx = G2048.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * k ** (-1.4)))
    lat = G2048.lat[:, None]
    land = (field > 0) & (np.abs(lat) < 60)
    pts = densify_km(G2048, coastline_segments(G2048, land), step_km=5)
    res = box_counting_dimension(pts, default_box_sizes_km(G2048))
    assert res["D"] > 1.15


def test_coastline_development_of_disc_close_to_one():
    land = cap(G2048, -20, 60, 20)
    L = coastline_length_km(G2048, coastline_segments(G2048, land))
    dev = coastline_development(L, cap_area_km2(G2048, 20), G2048.radius_km)
    assert dev == pytest.approx(1.0, abs=0.1)


def test_weighted_quantiles_match_numpy_for_equal_weights():
    v = np.random.default_rng(0).normal(size=10001)
    q = weighted_quantiles(v, np.ones_like(v), [0.1, 0.5, 0.9])
    assert np.allclose(q, np.quantile(v, [0.1, 0.5, 0.9]), atol=0.02)


def test_weighted_quantiles_respect_weights():
    q = weighted_quantiles([0.0, 10.0], [3.0, 1.0], [0.5])
    assert q[0] < 5.0


def test_wasserstein_shift_and_ks():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    w = np.ones_like(a)
    assert wasserstein(a, w, a + 3.0, w) == pytest.approx(3.0, abs=1e-9)
    assert weighted_ks(a, w, a, w) == pytest.approx(0.0, abs=1e-12)
    assert weighted_ks(a, w, a + 10.0, w) == pytest.approx(1.0, abs=1e-3)


def test_powerlaw_slope_recovered():
    rng = np.random.default_rng(3)
    b_true = 0.7
    sizes = (1.0 - rng.uniform(size=20000)) ** (-1.0 / b_true)  # Pareto：N(≥s) ∝ s^-0.7
    res = powerlaw_ccdf_slope(sizes, s_min=1.0, s_max=100.0)
    assert res["b"] == pytest.approx(b_true, abs=0.05)
