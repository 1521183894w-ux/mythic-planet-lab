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
def in_sphere_polygon(x: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """单位矢量 x（n×3）是否落在球面多边形 poly（m×3，顶点按顺序，边为大圆弧）里（U28 第二轮：岛弧系统的板块区域）。
    用以顶点中心为切点的心射投影判断：大圆在心射投影下是直线，边界精确。只适用于顶点离中心不到约 85° 的多边形。"""
    from matplotlib.path import Path as MPath

    x = np.atleast_2d(np.asarray(x, float))
    poly = np.asarray(poly, float)
    c = unit(poly.sum(0))
    e1 = np.cross([0.0, 0.0, 1.0], c)
    e1 = unit(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(c, e1)
    dp = poly @ c
    if (dp <= 0.087).any():
        raise ValueError("多边形太大：有顶点离中心超过约 85°")
    P = np.stack([(poly @ e1) / dp, (poly @ e2) / dp], 1)
    dq = x @ c
    ok = dq > 0.087
    out = np.zeros(len(x), bool)
    if ok.any():
        Q = np.stack([(x[ok] @ e1) / dq[ok], (x[ok] @ e2) / dq[ok]], 1)
        out[ok] = MPath(P).contains_points(Q)
    return out


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


def assign_ocean_plates(mesh, plate: np.ndarray, cont: np.ndarray, rules: list, forearc_km: float,
                        corridor_km: float, background: int, between_cos: float = -0.5,
                        ocean_plates: list | None = None) -> np.ndarray:
    """洋壳归哪块板块（U25）。返回每个节点的板块编号（陆壳节点不变）。

    按"离哪块大陆近就归谁"划分时，两块大陆之间的板块边界落在大洋正中，汇聚时海沟和岛弧出现在大洋中间
    （U25 第一次运行：洋内弧长成 14 百万 km² 的条带状陆地，还先于设计撞上大陆）。地球上汇聚的两块大陆之间，
    洋壳随下行一方运动，海沟在上盘陆缘外不远处（安第斯、苏门答腊的海沟离海岸 100–250 km）。所以：
    - 离大陆 forearc_km 以内的洋壳随该大陆（上盘的弧前，或被动陆缘）；
    - rules（[{down: [板块], over: [板块]}]，按顺序，先匹配的生效）：最近的大陆属于这条规则的一方、另一方最近的大陆
      在它的另一侧（两个方向夹角的余弦小于 between_cos，默认 -0.5 即大于 120°）、两边距离之和不超过 corridor_km 的洋壳，
      归最近的下行板块。（第一版按 90° 判断，洲 A 分到的"A、B 之间"的洋壳一直伸到两洲南边很远，A 拼上 B 后挡在
      A、B 南岸外，南方陆块群的洋壳在大洋中间撞上它，长出洋内弧，洲 C 北缘又撞上这条弧被消耗。）
    - 其余洋壳归外大洋板块 background（不属于任何大陆）；
    - ocean_plates（U26，[{id, near: 板块, side: "west"/"east", lat: [南, 北], lon: [西, 东]}]，按顺序）：外大洋里
      纬度在 lat、经度在 lon（可省；西 > 东 时跨过 180° 经线）范围内，且（写了 near 时）最近的大陆是 near、位于它的
      西侧/东侧（指向它质心的方向与正东方向夹角小于 60° / 与正西方向夹角小于 60°）的洋壳，另成一块大洋板块 id。"""
    from scipy.spatial import cKDTree

    from .geometry import km_from_chord

    out = plate.copy()
    pids = sorted(int(q) for q in np.unique(plate[cont]))
    oc = np.flatnonzero(~cont)
    X = mesh.xyz[oc]
    n = oc.size
    ar = np.arange(n)
    D = np.empty((len(pids), n))
    P = np.empty((len(pids), n, 3))
    for k, q in enumerate(pids):
        idx = np.flatnonzero(cont & (plate == q))
        d, i = cKDTree(mesh.xyz[idx]).query(X)
        D[k], P[k] = km_from_chord(d), mesh.xyz[idx[i]]
    k1 = D.argmin(0)
    d1 = D[k1, ar]
    res = np.where(d1 <= forearc_km, np.asarray(pids)[k1], background)
    done = d1 <= forearc_km

    def tangent(Q):
        v = Q - X
        v = v - np.sum(v * X, axis=1, keepdims=True) * X
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    for r in rules:
        ko = [pids.index(q) for q in r["over"] if q in pids]
        kd = [pids.index(q) for q in r["down"] if q in pids]
        if not ko or not kd:
            continue
        io, idn = D[ko].argmin(0), D[kd].argmin(0)
        do, dd = D[ko][io, ar], D[kd][idn, ar]
        po, pd = P[ko][io, ar], P[kd][idn, ar]
        between = np.sum(tangent(po) * tangent(pd), axis=1) < between_cos
        sel = ~done & between & (do + dd <= corridor_km) & (np.minimum(do, dd) <= d1 + 1e-6)
        res[sel] = np.asarray(pids)[np.asarray(kd)[idn[sel]]]
        done |= sel
    if ocean_plates:
        east = np.cross(np.array([0.0, 0.0, 1.0]), X)
        east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-12)
        lat = np.degrees(np.arcsin(np.clip(X[:, 2], -1, 1)))
        p1 = np.asarray(pids)[k1]
        lon = np.degrees(np.arctan2(X[:, 1], X[:, 0]))
        for op in ocean_plates:
            sel = (res == background) & (lat >= op["lat"][0]) & (lat <= op["lat"][1])
            if op.get("lon"):                  # 经度范围 [西, 东]，西 > 东 时跨过 180° 经线
                w, e = op["lon"]
                sel &= ((lon >= w) & (lon <= e)) if w <= e else ((lon >= w) | (lon <= e))
            if op.get("near") is not None:
                # 在哪一侧按"指向该陆块质心的方向"判断（第一版按指向最近海岸点的方向，海岸凹凸处方向突然翻转，
                # 范围碎成一条条斜条纹，夹在中间的外大洋碎条在正向演化里变成大洋里锯齿状的水下弧脊）
                blk = cont & (plate == int(op["near"]))
                cen = mesh.xyz[blk].sum(0)
                cen = cen / np.linalg.norm(cen)
                te = np.sum(tangent(np.broadcast_to(cen, X.shape)) * east, axis=1)
                sd = op.get("side", "any")
                side = te > 0.5 if sd == "west" else te < -0.5 if sd == "east" else np.ones(n, bool)
                sel &= (p1 == int(op["near"])) & side
            res[sel] = int(op["id"])
    out[oc] = res
    return out


def fill_ocean(mesh, c, owner: np.ndarray, t0: float, half_rate_km_myr: float = 25.0, max_age: float = 180.0):
    """洋壳：每个洋节点归离它最近的陆块所在的板块；洋底年龄按到板块边界的距离（半扩张速率 2.5 cm/yr），
    边界附近年轻（洋脊）、远离边界的陆缘外侧古老，封顶 180 Myr。
    （U25 起板块归属再由 assign_ocean_plates 按俯冲设计改写；年龄仍按这里的划分，相当于洋盆中线是老洋脊。）"""
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
