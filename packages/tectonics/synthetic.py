"""合成试验（tasks/stage1_tectonics.md 验收："裂谷 → 洋盆扩张"与"碰撞 → 造山"两个示例）。
单元测试（L5）与检查点出图（L7）共用。运动全部走剧本 -> 旋转文件 -> pyGPlates 的正式路径。

裂谷 → 洋盆扩张：赤道上一块大陆（球冠半径 30°）在板块 1 上，另半球是洋壳板块 2。t0 时沿 0° 经线把板块 1 裂开，
东半块成为板块 3，绕北极以 rate °/Myr 向东旋转；板块 1、2 不动。预期：赤道处新洋盆宽度 = rate × 时间；
洋脊在洋盆中线，洋壳年龄从洋脊向两侧陆缘线性增大到裂解以来的时间；陆缘减薄成陆架、陆坡。

碰撞 → 造山：大陆 A（中心 0°N 25°W）在板块 1 上，大陆 B（中心 0°N 25°E）在板块 2 上，两者之间的洋壳属于板块 2。
板块 2 绕北极向西旋转：洋壳先在 A 的东缘俯冲（安第斯型增厚），B 到达后碰撞，之后汇聚减到 20%（锁定）。
预期：碰撞时间 ≈ 两陆间距 / 汇聚速率；缝合带附近地壳增厚到 50 km 以上、高程 3 km 以上；陆壳体积守恒（扣除弧增生与剥蚀）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .crust import CONT, Crust, Params
from .engine import Engine
from .geometry import lonlat_to_xyz, pole_vector, smoothstep
from .kinematics import RotationFileKinematics, StageScript
from .mesh import Mesh

T0 = 100


def _caps(mesh: Mesh, lat, lon, r_deg) -> np.ndarray:
    return mesh.xyz @ lonlat_to_xyz(lon, lat) >= np.cos(np.radians(r_deg))


def _continent(mesh: Mesh, c: Crust, mask: np.ndarray, origin: int, margin_km: float = 250.0):
    from scipy.spatial import cKDTree

    from .geometry import km_from_chord
    d, _ = cKDTree(mesh.xyz[~mask]).query(mesh.xyz[mask])
    c.ctype[mask] = CONT
    c.T[mask] = 22.0 + 13.0 * smoothstep(km_from_chord(d) / margin_km)
    c.origin[mask] = origin


@dataclass
class SynthResult:
    engine: Engine
    snapshots: list
    script: StageScript
    rot_path: Path


def _run(mesh, c, script: StageScript, rot_path: Path, t_end: int, on_commit=None, snap_every: int = 10) -> SynthResult:
    script.write_rot(rot_path)
    kin = RotationFileKinematics(rot_path)
    _, om = kin.step_rotations(sorted(set(c.plate.tolist())), T0)
    eng = Engine(mesh, c, Params(), float(T0), om)
    if on_commit:
        on_commit(eng, kin)
    snaps = [eng.snapshot()]
    t = T0
    while t > t_end:
        dR, om = kin.step_rotations(eng.plates, t)
        committed = eng.step(dR, om)
        t -= 1
        if committed:
            if on_commit:
                on_commit(eng, kin)
            if t % snap_every == 0:
                snaps.append(eng.snapshot())
    return SynthResult(eng, snaps, script, rot_path)


def rift_spreading(mesh: Mesh, rot_path: Path, rate_deg_myr: float = 0.4, duration: int = 50) -> SynthResult:
    c = Crust(mesh.n)
    c.plate[:] = np.where(np.abs(mesh.lon) < 90.0, 1, 2)
    _continent(mesh, c, _caps(mesh, 0.0, 0.0, 30.0), origin=1)
    c.age[c.ctype == 0] = 80.0
    script = StageScript(T0)
    script.add_plate(1, 0, "west block")
    script.add_plate(2, 0, "far ocean")
    script.add_plate(3, 1, "east block")
    script.set_absolute(1, np.zeros(3), T0, 0)
    script.set_absolute(2, np.zeros(3), T0, 0)
    script.set_relative(3, 1, pole_vector(90.0, 0.0, rate_deg_myr), T0, 0)
    rng = np.random.default_rng(0)

    def on_commit(eng, kin):
        if eng.t == T0 and 3 not in eng.plates:
            s1, s2 = lonlat_to_xyz(-10.0, 0.0), lonlat_to_xyz(10.0, 0.0)
            eng.split_plate(1, s1, s2, np.array([0.0, 0.0, 1.0]), 3, rng, p_extra_offset=0.0,
                            omega_new=pole_vector(90.0, 0.0, rate_deg_myr))
    return _run(mesh, c, script, rot_path, T0 - duration, on_commit)


def collision_orogen(mesh: Mesh, rot_path: Path, rate_deg_myr: float = 0.5, lock_after: int = 25,
                     residual: float = 0.2, duration: int = 50) -> SynthResult:
    c = Crust(mesh.n)
    lon = mesh.lon
    c.plate[:] = np.where((lon > -175.0) & (lon < -3.0), 1, 2)
    _continent(mesh, c, _caps(mesh, 0.0, -25.0, 20.0), origin=1)
    _continent(mesh, c, _caps(mesh, 0.0, 25.0, 20.0), origin=2)
    c.age[c.ctype == 0] = 100.0
    script = StageScript(T0)
    script.add_plate(1, 0, "continent A")
    script.add_plate(2, 1, "continent B + ocean")
    script.set_absolute(1, np.zeros(3), T0, 0)
    w = pole_vector(90.0, 0.0, -rate_deg_myr)          # 绕北极负向 = 向西
    script.set_relative(2, 1, w, T0, T0 - lock_after)
    script.set_relative(2, 1, w * residual, T0 - lock_after, 0)
    return _run(mesh, c, script, rot_path, T0 - duration)
