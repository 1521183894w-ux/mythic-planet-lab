"""板块运动剧本（路线 B）：每块板块每 1 Myr 的前进阶段旋转 -> PLATES4 旋转文件 -> pyGPlates 逐步给出阶段旋转。

约定：
- 时间以 Ma 计，整数，从 t0 走到 0。"阶段旋转" S_q(t) 把板块 q 从 t 时刻的位置带到 t-1 时刻：x(t-1) = S_q(t) @ x(t)，绝对参照系。
- 旋转文件里的"总旋转" Rot_q(t) 按 PLATES4 惯例把现今（0 Ma）位置带到 t 时刻：Rot_q(0) = I，Rot_q(t) = S_q(t)^T · Rot_q(t-1)。
- 文件按板块层级写：每块板块相对其父板块（000 为绝对参照系）。pyGPlates 读入后按层级组合，GPlates 桌面软件也能打开。

method_class：procedural_constrained（运动由剧本设定）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import axis_angle, lonlat_to_xyz, omega_from_rotmat, omega_rotmat, rotmat, xyz_to_lonlat


class StageScript:
    def __init__(self, t0: int):
        self.t0 = int(t0)
        self.stages: dict[int, dict[int, np.ndarray]] = {}
        self.parent: dict[int, int] = {}
        self.names: dict[int, str] = {}
        self._fwd: dict[int, dict[int, np.ndarray]] = {}

    def add_plate(self, q: int, parent: int = 0, name: str = "") -> None:
        self.stages.setdefault(int(q), {})
        self.parent[int(q)] = int(parent)
        self.names[int(q)] = name or f"plate {q}"

    def _times(self, t_from: int, t_to: int):
        if not (self.t0 >= t_from > t_to >= 0):
            raise ValueError(f"时间段 {t_from} -> {t_to} 不在 {self.t0} -> 0 内或方向不对")
        return range(int(t_from), int(t_to), -1)

    def set_absolute(self, q: int, omega: np.ndarray, t_from: int, t_to: int) -> None:
        """t_from -> t_to 期间绕固定（绝对参照系）欧拉极匀速旋转，omega 为角速度矢量（rad/Myr）。"""
        e = omega_rotmat(np.asarray(omega, float), 1.0)
        for t in self._times(t_from, t_to):
            self.stages[q][t] = e
        self._fwd.clear()

    def follow(self, q: int, ref: int, t_from: int, t_to: int) -> None:
        """t_from -> t_to 期间与板块 ref 一起运动（例如裂解前的新板块、锁定后的碰撞板块）。"""
        for t in self._times(t_from, t_to):
            self.stages[q][t] = self.stages[ref][t]
        self._fwd.clear()

    def set_relative(self, q: int, ref: int, omega_rel: np.ndarray, t_from: int, t_to: int, frame_t: int | None = None) -> None:
        """t_from -> t_to 期间相对板块 ref 匀速旋转。omega_rel（rad/Myr）在 ref 于 frame_t 时刻（默认 t0）的参照系中给出，
        即欧拉极固定在 ref 上。要求 ref 在这段时间的阶段旋转已经设定。"""
        frame_t = self.t0 if frame_t is None else int(frame_t)
        e = omega_rotmat(np.asarray(omega_rel, float), 1.0)
        f0 = self.forward_total(ref, frame_t)
        for t in self._times(t_from, t_to):
            a = self.forward_total(ref, t - 1) @ f0.T
            b = self.forward_total(ref, t) @ f0.T
            self.stages[q][t] = a @ e @ b.T
        self._fwd.clear()

    def forward_total(self, q: int, t: int) -> np.ndarray:
        """板块 q 从 t0 到 t 的累计旋转（x(t) = F @ x(t0)）。"""
        cache = self._fwd.setdefault(q, {self.t0: np.eye(3)})
        if t not in cache:
            last = min(cache)
            r = cache[last]
            for tt in range(last, int(t), -1):
                r = self.stages[q][tt] @ r
                cache[tt - 1] = r
        return cache[int(t)]

    def check_complete(self) -> None:
        for q, st in self.stages.items():
            missing = [t for t in range(self.t0, 0, -1) if t not in st]
            if missing:
                raise ValueError(f"板块 {q} 缺少 {len(missing)} 个时刻的阶段旋转（例如 {missing[:3]}）")

    def totals(self) -> dict[int, dict[int, np.ndarray]]:
        """总旋转（现今 -> t 时刻）：Rot(0) = I，Rot(t) = S(t)^T · Rot(t-1)。"""
        self.check_complete()
        out = {}
        for q, st in self.stages.items():
            rot = {0: np.eye(3)}
            for t in range(1, self.t0 + 1):
                rot[t] = st[t].T @ rot[t - 1]
            out[q] = rot
        return out

    def write_rot(self, path: str | Path) -> Path:
        """写 PLATES4 旋转文件（每块板块每 1 Myr 一条，相对父板块）。"""
        path = Path(path)
        tot = self.totals()
        lines = []
        for q in sorted(self.stages):
            par = self.parent[q]
            for t in range(0, self.t0 + 1):
                rm = tot[q][t] if par == 0 else tot[par][t].T @ tot[q][t]
                ax, ang = axis_angle(rm)
                lon, lat = xyz_to_lonlat(ax)
                if ang < 1e-12:
                    lat, lon = 90.0, 0.0
                lines.append(f"{q:4d} {t:7.1f} {float(lat):11.6f} {float(lon):12.6f} {np.degrees(ang):11.6f} {par:4d} "
                             f"!{self.names[q]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path


def fr_to_mat(fr) -> np.ndarray:
    """pygplates.FiniteRotation -> 3x3 矩阵。"""
    if fr.represents_identity_rotation():
        return np.eye(3)
    lat, lon, ang = fr.get_lat_lon_euler_pole_and_angle_degrees()
    return rotmat(lonlat_to_xyz(lon, lat), np.radians(ang))


class RotationFileKinematics:
    """从旋转文件逐步取阶段旋转（pyGPlates 插值、组合层级）。"""

    def __init__(self, rot_path: str | Path):
        import pygplates

        self.path = Path(rot_path)
        self.model = pygplates.RotationModel(str(self.path))

    def stage(self, q: int, t: int) -> np.ndarray:
        """板块 q 从 t 到 t-1 的阶段旋转（绝对参照系）。"""
        return fr_to_mat(self.model.get_rotation(float(t - 1), int(q), float(t), anchor_plate_id=0))

    def total_from(self, q: int, t_from: float, t_to: float) -> np.ndarray:
        """板块 q 从 t_from 到 t_to 的累计旋转（绝对参照系）。"""
        return fr_to_mat(self.model.get_rotation(float(t_to), int(q), float(t_from), anchor_plate_id=0))

    def step_rotations(self, plates, t: int) -> tuple[dict, dict]:
        dR = {q: self.stage(q, t) for q in plates}
        return dR, {q: omega_from_rotmat(r, 1.0) for q, r in dR.items()}
