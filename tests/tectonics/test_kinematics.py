"""旋转与剧本：几何工具、旋转文件往返（剧本 -> PLATES4 -> pyGPlates）、层级组合。"""

import numpy as np
import pytest

from mythic.tectonics.geometry import (axis_angle, lonlat_to_xyz, omega_from_rotmat, omega_rotmat, pole_vector,
                                       rotmat, xyz_to_lonlat)
from mythic.tectonics.kinematics import RotationFileKinematics, StageScript


def test_rotation_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = rng.uniform(0, np.pi)
        a2, g2 = axis_angle(rotmat(ax, ang))
        assert np.allclose(rotmat(a2, g2), rotmat(ax, ang), atol=1e-10)
    w = pole_vector(30, 40, 0.5)
    assert np.allclose(omega_from_rotmat(omega_rotmat(w, 1.0), 1.0), w, atol=1e-12)
    lon, lat = xyz_to_lonlat(lonlat_to_xyz(-179.5, 88.0))
    assert np.isclose(lon, -179.5) and np.isclose(lat, 88.0)


def test_positive_rotation_about_south_pole_moves_west():
    """约定检查：绕南极正向旋转 = 向西。"""
    x = lonlat_to_xyz(0.0, 0.0)
    y = omega_rotmat(pole_vector(-90, 0, 1.0), 1.0) @ x
    assert xyz_to_lonlat(y)[0] < 0


@pytest.fixture(scope="module")
def script_and_file(tmp_path_factory):
    s = StageScript(20)
    s.add_plate(1, 0, "A")
    s.add_plate(2, 1, "B relative to A")
    s.add_plate(3, 1, "C born at 10 Ma")
    s.set_absolute(1, pole_vector(10, 20, 0.3), 20, 0)
    s.set_relative(2, 1, pole_vector(-40, 100, 0.8), 20, 12)
    s.set_relative(2, 1, pole_vector(-40, 100, 0.1), 12, 0)       # 锁定后减速
    s.follow(3, 1, 20, 10)
    s.set_relative(3, 1, pole_vector(60, -30, 0.5), 10, 0, frame_t=10)
    path = s.write_rot(tmp_path_factory.mktemp("rot") / "t.rot")
    return s, RotationFileKinematics(path)


def test_rotation_file_reproduces_script(script_and_file):
    s, kin = script_and_file
    err = max(float(np.abs(kin.stage(q, t) - s.stages[q][t]).max()) for q in (1, 2, 3) for t in range(20, 0, -1))
    assert err < 1e-7


def test_relative_motion_fixed_in_parent_frame(script_and_file):
    """板块 2 相对板块 1 的运动：在板块 1 的参照系里是绕固定欧拉极的匀速旋转。"""
    s, _ = script_and_file
    e = omega_rotmat(pole_vector(-40, 100, 0.8), 1.0)
    for t in (20, 15, 13):
        f1a, f1b = s.forward_total(1, t - 1), s.forward_total(1, t)
        rel = f1a.T @ s.stages[2][t] @ f1b        # 在板块 1 的 t0 参照系中的阶段旋转
        assert np.allclose(rel, e, atol=1e-12)


def test_follow_then_birth(script_and_file):
    """板块 3 诞生前与板块 1 同步运动，旋转文件里也如此。"""
    s, kin = script_and_file
    for t in (20, 15, 11):
        assert np.allclose(kin.stage(3, t), kin.stage(1, t), atol=1e-9)
    assert not np.allclose(kin.stage(3, 5), kin.stage(1, 5), atol=1e-6)
