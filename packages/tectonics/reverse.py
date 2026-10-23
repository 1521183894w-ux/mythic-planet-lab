"""倒推板块运动（PROGRESS U22）：从终图出发，按关键时刻的有限旋转描述各刚体块的历史。

做法与地球古地理重建相同：每块板块在若干关键时刻（Ma）相对父板块有一个"总旋转"（把现今位置带到该时刻的位置），
关键时刻之间按四元数球面插值（与 pyGPlates 对 PLATES4 旋转文件的插值方式一致）；绝对位置由父板块链组合而成。

位姿用"操作序列"描述，逐个作用在板块现今位置上（都在父板块的现今参照系里）：
- {move: [距离 km, 方位角 °]}：把板块（当前的）质心沿大圆朝该方位移动；
- {spin: 角度 °}：绕板块（当前的）质心逆时针转（从外向球心看）；
- {to: [经, 纬]}：把质心移到该点（不转）；
- {pole: [经, 纬, 角度 °]}：绕给定欧拉极右手旋转；
- {point_to: [[经, 纬], [经, 纬]]}：把板块上某一点（现今坐标）沿大圆移到目标点（用于热点轨迹等"某点某时在某处"的约束）。
只倒推运动，不倒推地壳过程（CLAUDE.md §4）。method_class：procedural_constrained（运动由设计设定，速度受地球范围约束）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import R_KM, axis_angle, lonlat_to_xyz, rotmat, unit, xyz_to_lonlat


# ---------------------------------------------------------------------------
# 位姿
# ---------------------------------------------------------------------------
def move_matrix(c: np.ndarray, d_km: float, az_deg: float) -> np.ndarray:
    """把单位矢量 c 沿方位角 az（0 = 北，90 = 东）走 d_km 的大圆旋转。"""
    c = unit(c)
    north = unit(np.array([0.0, 0.0, 1.0]) - c[2] * c) if abs(c[2]) < 0.999999 else np.array([1.0, 0.0, 0.0])
    east = np.cross(north, c)
    az = np.radians(az_deg)
    u = np.cos(az) * north + np.sin(az) * east
    return rotmat(np.cross(c, u), d_km / R_KM)


def pose_from_ops(c0: np.ndarray, ops: list) -> np.ndarray:
    """操作序列 -> 旋转矩阵（现今位置 -> 该时刻位置，父板块参照系）。c0 为板块现今质心（单位矢量）。"""
    R = np.eye(3)
    for op in ops or []:
        c = R @ c0
        if "move" in op:
            d, az = op["move"]
            M = move_matrix(c, float(d), float(az))
        elif "spin" in op:
            M = rotmat(c, np.radians(float(op["spin"])))
        elif "to" in op:
            tgt = lonlat_to_xyz(*op["to"])
            ax = np.cross(c, tgt)
            n = np.linalg.norm(ax)
            M = np.eye(3) if n < 1e-12 else rotmat(ax / n, float(np.arctan2(n, c @ tgt)))
        elif "pole" in op:
            lon, lat, ang = op["pole"]
            M = rotmat(lonlat_to_xyz(lon, lat), np.radians(float(ang)))
        elif "point_to" in op:
            # 把板块上某一点（现今坐标，先经过前面的操作）移到目标点，沿大圆
            (lo1, la1), (lo2, la2) = op["point_to"]
            a, b = R @ lonlat_to_xyz(lo1, la1), lonlat_to_xyz(lo2, la2)
            ax = np.cross(a, b)
            n = np.linalg.norm(ax)
            M = np.eye(3) if n < 1e-12 else rotmat(ax / n, float(np.arctan2(n, a @ b)))
        else:
            raise ValueError(f"不认识的位姿操作：{op}")
        R = M @ R
    return R


# ---------------------------------------------------------------------------
# 四元数（插值用）
# ---------------------------------------------------------------------------
def mat_to_quat(R: np.ndarray) -> np.ndarray:
    ax, ang = axis_angle(R)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * ax]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, v = q[0], q[1:]
    s = np.linalg.norm(v)
    if s < 1e-15:
        return np.eye(3)
    return rotmat(v / s, 2 * np.arctan2(s, w))


def slerp(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    if q0 @ q1 < 0:
        q1 = -q1
    c = float(np.clip(q0 @ q1, -1.0, 1.0))
    if c > 0.999999:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    om = np.arccos(c)
    return (np.sin((1 - f) * om) * q0 + np.sin(f * om) * q1) / np.sin(om)


# ---------------------------------------------------------------------------
# 板块历史
# ---------------------------------------------------------------------------
class PlateHistory:
    """各板块的关键帧（相对父板块的总旋转），插值、组合层级、写旋转文件、算速度。"""

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.names: dict[int, str] = {}
        self.keys: dict[int, list[tuple[float, np.ndarray]]] = {}

    def add(self, pid: int, parent: int, name: str, keyframes: list[tuple[float, np.ndarray]]) -> None:
        if parent != 0 and parent not in self.parent:
            raise ValueError(f"板块 {pid} 的父板块 {parent} 还没有定义（父板块要先加）")
        kf = sorted(((float(t), np.asarray(R, float)) for t, R in keyframes), key=lambda x: x[0])
        if not kf or kf[0][0] != 0.0 or not np.allclose(kf[0][1], np.eye(3), atol=1e-9):
            raise ValueError(f"板块 {pid} 的关键帧必须从 0 Ma 的单位旋转开始")
        self.parent[pid], self.names[pid], self.keys[pid] = int(parent), name, kf

    def relative(self, pid: int, t: float) -> np.ndarray:
        kf = self.keys[pid]
        if t <= kf[0][0]:
            return kf[0][1]
        if t >= kf[-1][0]:
            return kf[-1][1]
        for (t0, R0), (t1, R1) in zip(kf[:-1], kf[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                return quat_to_mat(slerp(mat_to_quat(R0), mat_to_quat(R1), f))
        raise AssertionError

    def absolute(self, pid: int, t: float) -> np.ndarray:
        """现今 -> t 时刻的绝对总旋转：父板块的绝对旋转 · 相对父板块的旋转。"""
        R = self.relative(pid, t)
        p = self.parent[pid]
        return R if p == 0 else self.absolute(p, t) @ R

    def positions(self, pid: int, X0: np.ndarray, t: float) -> np.ndarray:
        """板块上现今位置为 X0（n×3）的点在 t 时刻的位置。"""
        return X0 @ self.absolute(pid, t).T

    def velocity_cm_yr(self, pid: int, X0: np.ndarray, t: float, dt: float = 1.0) -> np.ndarray:
        """t 时刻（朝现今方向运动）的绝对速度（cm/yr；1 cm/yr = 10 km/Myr），按 t 与 t - dt 两个位置差分。"""
        a = self.positions(pid, X0, t)
        b = self.positions(pid, X0, max(t - dt, 0.0))
        ang = np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))
        return ang * R_KM / (t - max(t - dt, 0.0)) / 10.0

    def relative_velocity_cm_yr(self, pid: int, ref: int, X0: np.ndarray, t: float, dt: float = 1.0) -> np.ndarray:
        """板块 pid 上的点相对板块 ref 的速度（cm/yr），在 ref 的参照系里差分。"""
        def in_ref(tt):
            return self.positions(pid, X0, tt) @ self.absolute(ref, tt)    # 左乘 ref 绝对旋转的逆
        a, b = in_ref(t), in_ref(max(t - dt, 0.0))
        ang = np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))
        return ang * R_KM / (t - max(t - dt, 0.0)) / 10.0

    def key_times(self) -> list[float]:
        return sorted({t for kf in self.keys.values() for t, _ in kf})

    def write_rot(self, path: str | Path) -> Path:
        """写 PLATES4 旋转文件：每块板块在它自己的关键时刻相对父板块的总旋转（pyGPlates 在关键时刻之间插值）。"""
        lines = []
        for pid in sorted(self.keys):
            for t, R in self.keys[pid]:
                ax, ang = axis_angle(R)
                lon, lat = xyz_to_lonlat(ax)
                if ang < 1e-12:
                    lat, lon = 90.0, 0.0
                lines.append(f"{pid:4d} {t:7.1f} {float(lat):11.6f} {float(lon):12.6f} {np.degrees(ang):11.6f} "
                             f"{self.parent[pid]:4d} !{self.names[pid]}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return path


# ---------------------------------------------------------------------------
# U24：补回碰撞挤掉的地壳、构建初始世界
# ---------------------------------------------------------------------------
def _raster_index(lon, lat, ny: int, nx: int):
    i = np.clip(((90.0 - lat) / 180.0 * ny).astype(int), 0, ny - 1)
    j = np.clip(((lon + 180.0) / 360.0 * nx).astype(int), 0, nx - 1)
    return i, j


def consumed_margin(grid_xyz: np.ndarray, down_mask: np.ndarray, up_mask: np.ndarray, R_up_tc: np.ndarray,
                    max_km: float, step_km: float = 40.0) -> np.ndarray:
    """碰撞时被挤掉的下行块陆缘（下行块的现今参照系、经纬度栅格上的掩膜）。

    R_up_tc：把上盘块的现今位置带到接触时刻 t_c 在下行块参照系里的位置（下行块绝对旋转的逆 · 上盘块绝对旋转）。
    上盘块从 t_c 到现今推进时扫过的、现今被它覆盖的地方，原来是下行块的陆缘（后来被俯冲、拼贴进造山带）：
    取上盘块现今覆盖、t_c 时还没覆盖的格点，沿该处上盘的推进方向往前走，只要一路都在上盘陆地上、max_km 内碰到下行块，
    就算被挤掉的陆缘。中间隔着海（例如残留洋盆）的不算——那里挤掉的是洋壳。"""
    ny, nx = down_mask.shape
    X = grid_xyz.reshape(-1, 3)
    up = up_mask.ravel()
    q = np.flatnonzero(up)
    # t_c 时上盘的覆盖：q 处的点转回上盘现今位置仍在上盘里
    lo, la = xyz_to_lonlat(X[q] @ R_up_tc)
    i, j = _raster_index(lo, la, ny, nx)
    cand = q[~up_mask[i, j]]
    if cand.size == 0:
        return np.zeros_like(down_mask)
    P = X[cand]
    prev = P @ R_up_tc.T                                   # 现今在 q 的上盘点，t_c 时在哪
    d = P - prev
    d = d - np.sum(d * P, axis=1, keepdims=True) * P       # 推进方向（切平面）
    n = np.linalg.norm(d, axis=1)
    ok = n > 1e-9
    P, cand, d = P[ok], cand[ok], d[ok] / n[ok, None]
    hit = np.zeros(cand.size, bool)
    alive = np.ones(cand.size, bool)
    for s in np.arange(step_km, max_km + step_km, step_km):
        a = s / R_KM
        Y = np.cos(a) * P + np.sin(a) * d
        lo, la = xyz_to_lonlat(Y)
        i, j = _raster_index(lo, la, ny, nx)
        newly = alive & down_mask[i, j]
        hit |= newly
        alive &= ~newly & up_mask[i, j]                   # 走出上盘陆地（到了海里）就停
        if not alive.any():
            break
    out = np.zeros(ny * nx, bool)
    out[cand[hit]] = True
    return out.reshape(down_mask.shape)


def initial_crust(mesh, history: PlateHistory, t0: float, block_ids: list, core: np.ndarray, ext: np.ndarray,
                  fields_core: dict, fields_ext: dict):
    """t0 时刻的地壳（固定网格视图）：网格节点转回各块现今位置，落在块里（或块被挤掉的陆缘里）就是该块的陆壳。

    core / ext：现今经纬度栅格上的块编号（ext = 被挤掉的陆缘，可以与别的块的现今范围重叠）；
    fields_core / fields_ext：{"T": 厚度 km, "t_form": 形成年龄 Ma, "prov": 省份编号} 栅格。
    同一节点有几块认领时，现今本体优先于被挤掉的陆缘，再按 block_ids 的顺序。返回 (Crust, 板块编号数组)。"""
    from .crust import CONT, Crust, ORO_NONE

    ny, nx = core.shape
    n = mesh.n
    owner = np.zeros(n, np.int64)
    kind = np.zeros(n, np.int8)                            # 1 = 本体，2 = 被挤掉的陆缘
    src = np.zeros(n, np.int64)
    for pid in block_ids:
        Y = mesh.xyz @ history.absolute(pid, t0)          # R^T x：转回现今位置
        lo, la = xyz_to_lonlat(Y)
        i, j = _raster_index(lo, la, ny, nx)
        flat = i * nx + j
        c_hit = (core[i, j] == pid) & (kind != 1)
        e_hit = (ext[i, j] == pid) & (kind == 0)
        for hit, k in ((c_hit, 1), (e_hit, 2)):
            owner[hit], kind[hit], src[hit] = pid, k, flat[hit]
    c = Crust(n)
    cont = kind > 0
    c.ctype[cont] = CONT
    c.plate[cont] = owner[cont]
    c.origin[cont] = owner[cont]
    for k, f in ((1, fields_core), (2, fields_ext)):
        sel = kind == k
        c.T[sel] = f["T"].ravel()[src[sel]]
        c.t_form[sel] = f["t_form"].ravel()[src[sel]]
        c.prov[sel] = f["prov"].ravel()[src[sel]]
    c.oro[cont] = ORO_NONE
    return c, owner, kind


def fill_ocean(mesh, c, owner: np.ndarray, t0: float, half_rate_km_myr: float = 25.0, max_age: float = 180.0):
    """洋壳：每个洋节点归离它最近的陆块所在的板块；洋底年龄按到板块边界的距离（半扩张速率 2.5 cm/yr），
    边界附近年轻（洋脊）、远离边界的陆缘外侧古老，封顶 180 Myr。"""
    from scipy.spatial import cKDTree

    from .crust import OCEAN
    from .geometry import km_from_chord

    cont = owner > 0
    oc = np.flatnonzero(~cont)
    _, k = cKDTree(mesh.xyz[cont]).query(mesh.xyz[oc])
    plate = owner.copy()
    plate[oc] = owner[np.flatnonzero(cont)[k]]
    e = mesh.edges
    bnd = np.unique(e[plate[e[:, 0]] != plate[e[:, 1]]].ravel())
    d, _ = cKDTree(mesh.xyz[bnd]).query(mesh.xyz[oc])
    age = np.clip(km_from_chord(d) / half_rate_km_myr, 2.0, max_age)
    c.plate = plate.astype(c.plate.dtype)
    c.ctype[oc] = OCEAN
    c.T[oc] = 7.0
    c.age[oc] = age
    c.t_form[oc] = t0 + age
    return c
