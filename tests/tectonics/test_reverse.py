"""倒推板块运动（packages/tectonics/reverse.py）：位姿操作、插值、层级组合、速度、旋转文件往返。"""

import numpy as np
import pytest

from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, rotmat
from mythic.tectonics.reverse import PlateHistory, move_matrix, pose_from_ops, quat_to_mat, mat_to_quat, slerp


def _dist_km(a, b):
    return float(np.arccos(np.clip(a @ b, -1, 1)) * R_KM)


def test_move_spin_to_ops():
    c = lonlat_to_xyz(20.0, 10.0)
    R = pose_from_ops(c, [{"move": [1000.0, 90.0]}])
    assert _dist_km(R @ c, c) == pytest.approx(1000.0, abs=1e-6)
    lon_b = np.degrees(np.arctan2((R @ c)[1], (R @ c)[0]))
    assert lon_b > 20.0                                             # 向东走
    S = pose_from_ops(c, [{"spin": 30.0}])
    assert np.allclose(S @ c, c)                                    # 绕质心转，质心不动
    T = pose_from_ops(c, [{"to": [-40.0, -25.0]}])
    assert np.allclose(T @ c, lonlat_to_xyz(-40.0, -25.0))
    # 先移动再自转：自转绕的是移动后的质心
    M = pose_from_ops(c, [{"move": [500.0, 0.0]}, {"spin": 45.0}])
    assert _dist_km(M @ c, c) == pytest.approx(500.0, abs=1e-6)


def test_slerp_midpoint():
    ax = lonlat_to_xyz(30.0, 60.0)
    R = rotmat(ax, np.radians(40.0))
    half = quat_to_mat(slerp(mat_to_quat(np.eye(3)), mat_to_quat(R), 0.5))
    assert np.allclose(half, rotmat(ax, np.radians(20.0)), atol=1e-12)


def _two_plate_history():
    h = PlateHistory()
    c_par = lonlat_to_xyz(0.0, 0.0)
    h.add(1, 0, "P", [(0, np.eye(3)), (100, pose_from_ops(c_par, [{"move": [2000.0, 180.0]}]))])
    c_ch = lonlat_to_xyz(10.0, 0.0)
    h.add(2, 1, "C", [(0, np.eye(3)), (50, pose_from_ops(c_ch, [{"move": [1000.0, 90.0]}])),
                      (100, pose_from_ops(c_ch, [{"move": [1000.0, 90.0]}]))])
    return h, c_par, c_ch


def test_hierarchy_and_speed():
    h, c_par, c_ch = _two_plate_history()
    # 父板块 100 Myr 走 2000 km，匀速 2 cm/yr
    v = h.velocity_cm_yr(1, c_par[None, :], 60.0)
    assert v[0] == pytest.approx(2.0, rel=1e-3)
    # 子板块 0–50 Ma 相对父板块走 1000 km：相对速度 2 cm/yr；50–100 Ma 不动
    assert h.relative_velocity_cm_yr(2, 1, c_ch[None, :], 30.0)[0] == pytest.approx(2.0, rel=1e-3)
    assert h.relative_velocity_cm_yr(2, 1, c_ch[None, :], 80.0)[0] == pytest.approx(0.0, abs=1e-6)
    # 绝对 = 父板块绝对 · 相对
    R = h.absolute(2, 70.0)
    assert np.allclose(R, h.absolute(1, 70.0) @ h.relative(2, 70.0))


def test_keyframes_must_start_at_present():
    h = PlateHistory()
    with pytest.raises(ValueError):
        h.add(1, 0, "x", [(10, np.eye(3))])
    with pytest.raises(ValueError):
        h.add(3, 7, "y", [(0, np.eye(3))])                          # 父板块未定义


def test_rotation_file_roundtrip_with_pygplates(tmp_path):
    pygplates = pytest.importorskip("pygplates")
    from mythic.tectonics.kinematics import fr_to_mat

    h, _, _ = _two_plate_history()
    path = h.write_rot(tmp_path / "h.rot")
    model = pygplates.RotationModel(str(path))
    for pid in (1, 2):
        for t in (0.0, 25.0, 50.0, 70.0, 100.0):
            fr = model.get_rotation(t, pid, 0.0, anchor_plate_id=0)
            assert np.allclose(fr_to_mat(fr), h.absolute(pid, t), atol=1e-6), (pid, t)


def test_move_matrix_is_rotation():
    M = move_matrix(lonlat_to_xyz(100.0, -50.0), 3000.0, 225.0)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12) and np.linalg.det(M) == pytest.approx(1.0)
