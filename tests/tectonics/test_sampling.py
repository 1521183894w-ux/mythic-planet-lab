"""原生取样的旋转、接缝、空隙与显示处方必须分别可核查。"""
import hashlib

import numpy as np
import pytest

from mythic.tectonics.crust import CONT, OCEAN, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import lonlat_to_xyz, rotmat
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.sampling import FIXED_FALLBACK, VIEW_FILL, native_cover, prepare_output, sample_output


@pytest.fixture(scope="module")
def mesh():
    return load_mesh(3)


def fingerprint(eng):
    h = hashlib.sha256()
    for f in Crust.FIELDS:
        h.update(getattr(eng.c, f).tobytes())
    h.update(eng.alive.tobytes())
    for q in eng.plates:
        h.update(eng.R[q].tobytes())
    return h.hexdigest()


def world(mesh, params=None):
    c = Crust(mesh.n)
    c.plate[:] = 1
    c.ctype[:] = CONT
    c.T[:] = 35.
    return Engine(mesh, c, params or Params(), 0., {1: np.zeros(3)})


def test_rotated_native_query_matches_plate_cover_at_dateline_and_pole(mesh):
    eng = world(mesh)
    eng.R[1] = rotmat(lonlat_to_xyz(25., 45.), np.radians(31.))
    p = lonlat_to_xyz(np.array([-180., 180., -179.9, 179.9, 0.]), np.array([68., 68., 89.9, 89.9, 90.]))
    before = fingerprint(eng)
    np.testing.assert_array_equal(native_cover(eng, p, chunk_size=2), eng.covering(1, p))
    assert fingerprint(eng) == before


def test_no_coverage_is_explicit_fixed_fallback_not_new_crust(mesh):
    eng = world(mesh)
    eng.c.plate[:] = np.where(mesh.xyz[:, 0] > 0, 1, 2)
    eng.R[2] = np.eye(3)
    eng.omega[2] = np.zeros(3)
    eng._rebuild()
    eng.R[1] = rotmat([0., 0., 1.], np.radians(18.))
    eng.R[2] = rotmat([0., 0., 1.], np.radians(-18.))
    frame = eng.snapshot()
    frame["dyn"] = np.zeros(mesh.n)
    before = fingerprint(eng)
    ctx = prepare_output(eng, frame)
    result = sample_output(ctx, mesh.xyz, np.zeros(mesh.n), carry_view_fill=False)
    gap = result["owner"] < 0
    assert gap.any() and (~gap).any()
    assert (result["status"][gap] == FIXED_FALLBACK).all()
    assert (result["status"][~gap] == 0).all()
    np.testing.assert_array_equal(result["elev"][gap], frame["elev"][gap])
    assert (eng.c.ctype == CONT).all()
    assert fingerprint(eng) == before


def test_view_fill_is_optional_and_never_changes_material(mesh):
    eng = world(mesh, Params(view_hole_max_cells=4))
    eng.c.ctype[0] = OCEAN
    eng.c.T[0] = 7.
    eng.c.age[0] = 0.
    frame = eng.snapshot()
    frame["dyn"] = np.full(mesh.n, 17.)
    frame["elev"] = frame["elev"] + 17.
    before = fingerprint(eng)
    ctx = prepare_output(eng, frame)
    assert ctx.filled[0]
    p = mesh.xyz[[0, 1]]
    direct = sample_output(ctx, p, np.full(2, 17.), carry_view_fill=False)
    shown = sample_output(ctx, p, np.full(2, 17.), carry_view_fill=True)
    assert direct["ctype"][0] == OCEAN and direct["status"][0] == 0
    assert shown["ctype"][0] == CONT and shown["status"][0] == VIEW_FILL
    assert shown["uid"][0] == -1
    assert eng.c.ctype[0] == OCEAN and fingerprint(eng) == before
    assert shown["elev"][0] == frame["elev"][0]


def test_query_rejects_off_sphere_and_bad_dynamic_terms(mesh):
    eng = world(mesh)
    with pytest.raises(ValueError, match="单位球面"):
        native_cover(eng, [[2., 0., 0.]])
    ctx = prepare_output(eng, eng.snapshot())
    with pytest.raises(ValueError, match="dynamic_m"):
        sample_output(ctx, mesh.xyz[:2], np.zeros(1))
