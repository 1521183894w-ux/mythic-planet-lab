"""剖面须保持球面长度、正交方向，且不能跨空隙连接两个山体。"""
import numpy as np
import pytest

from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, xyz_to_lonlat
from mythic.tectonics.sections import spherical_sections, axis_connected_span


def test_equator_sections_have_analytic_positions_and_extend_ends():
    degree = R_KM * np.pi / 180.
    p, length = spherical_sections([[179., 0.], [-179., 0.]], [-degree, 0., degree, 3 * degree], [-degree, 0., degree])
    lon, lat = xyz_to_lonlat(p)
    np.testing.assert_allclose(lat, np.tile([-1., 0., 1.], (4, 1)), atol=1e-10)
    delta = (lon[:, 1] - [178., 179., 180., -178.] + 180.) % 360. - 180.
    np.testing.assert_allclose(delta, 0., atol=1e-10)
    assert length == pytest.approx(2 * degree)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1), 1., atol=1e-14)


def test_polar_section_has_orthogonal_great_circle_and_requested_distance():
    p, _ = spherical_sections([[0., 70.], [180., 70.]], [1000.], [-250., 0., 250.])
    center = p[0, 1]
    normal = np.cross(lonlat_to_xyz(0., 70.), lonlat_to_xyz(180., 70.))
    tangent = np.cross(normal / np.linalg.norm(normal), center)
    transverse = p[0, 2] - np.cos(250 / R_KM) * center
    assert transverse @ tangent == pytest.approx(0., abs=1e-12)
    distance = np.arccos(np.clip(p[0] @ center, -1., 1.)) * R_KM
    np.testing.assert_allclose(distance[[0, 2]], 250., atol=1e-8)


def test_span_stops_at_uncovered_and_does_not_merge_remote_peak():
    x = np.arange(-40., 41., 10.)
    h = [2000., 2000., np.nan, 1800., 1800., 1800., 500., 2100., 2100.]
    r = axis_connected_span(x, h, 1500.)
    assert r["inside_km"] == [-10., 10.]
    assert r["left_stop"] == "uncovered" and r["right_stop"] == "low"
    assert r["left_bracket_km"] == [-20., -10.]
    assert axis_connected_span(x, h, 2000.)["axis_above"] is False
    full = axis_connected_span(x, np.full(len(x), 1800.), 1500.)
    assert full["left_bracket_km"] is None and full["right_stop"] == "window"


def test_undefined_geometry_and_missing_axis_are_rejected():
    for line in ([[0., 0.], [0., 0.]], [[0., 0.], [180., 0.]]):
        with pytest.raises(ValueError, match="切线"):
            spherical_sections(line, [0.], [0.])
    with pytest.raises(ValueError, match="零点"):
        axis_connected_span([-10., 1., 10.], [1., 2., 1.], 0.)
