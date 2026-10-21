"""Spike 02 · 板块运动学：pyGPlates 1.0 与 GPlately 2.0。

要回答的问题（阶段 1 打算"自写规则模拟地壳相互作用 + 用 pyGPlates 记录运动学"）：
1. 能否用程序生成的旋转文件（PLATES4 .rot）描述虚构星球的板块运动，并正确插值、组合（板块层级）？
2. 重建（把现今几何体旋回过去）与速度计算是否与解析值一致？
3. 特征集合能否写出/读回（.gpmlz），以便长期保存世界历史？
4. GPlately 的封装接口能否在离线、虚构模型下使用？地球板块模型库能否联网获取（作参照）？
"""

from __future__ import annotations

import math

import numpy as np
from _common import WORK_DIR, Recorder, import_gplately, savefig

rec = Recorder("02_plate_kinematics")
import pygplates  # noqa: E402

R_KM = 6371.0
work = WORK_DIR / "02_plates"
work.mkdir(parents=True, exist_ok=True)

# ---- 1. 程序生成旋转文件 -----------------------------------------------------------
# 板块 101、301 相对绝对参考系（000）运动；板块 201 相对 101 运动（测试层级组合）。
# 欧拉极固定、角速度恒定，便于和解析值对照。
PLATES = {
    101: dict(pole=(20.0, 40.0), rate=0.35, fixed=0, name="Plate A"),     # 0.35°/Myr
    201: dict(pole=(-30.0, 150.0), rate=0.50, fixed=101, name="Plate B (rel. A)"),
    301: dict(pole=(60.0, -100.0), rate=0.20, fixed=0, name="Plate C"),
}
TIMES = [0, 50, 100, 150, 200]
rot_path = work / "synthetic.rot"
with rec.measure("写 PLATES4 旋转文件（3 块板块，0–200 Ma）") as e:
    lines = []
    for pid, p in PLATES.items():
        for t in TIMES:
            angle = p["rate"] * t
            lines.append(f"{pid:4d} {t:7.1f} {p['pole'][0]:8.3f} {p['pole'][1]:9.3f} {angle:9.3f} {p['fixed']:3d} !{p['name']}")
    rot_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    e["lines"] = len(lines)

with rec.measure("读入 RotationModel") as e:
    model = pygplates.RotationModel(str(rot_path))

with rec.measure("核对：任意时刻的总旋转 = 文件值（含 75 Ma 插值）") as e:
    max_dev = 0.0
    for pid in (101, 301):
        p = PLATES[pid]
        for t in (50.0, 75.0, 200.0):
            rot = model.get_rotation(t, pid, anchor_plate_id=0)
            lat, lon, ang = rot.get_lat_lon_euler_pole_and_angle_degrees()
            expect = p["rate"] * t
            # 欧拉极可能以对跖点 + 负角度表示，统一比较旋转作用在测试点上的结果
            test = pygplates.PointOnSphere(0.0, 0.0)
            got = rot * test
            ref = pygplates.FiniteRotation(pygplates.PointOnSphere(*p["pole"]), math.radians(expect)) * test
            dev = math.degrees(pygplates.GeometryOnSphere.distance(got, ref))
            max_dev = max(max_dev, dev)
    e["max_deviation_deg"] = float(f"{max_dev:.2e}")
    e["status"] = "ok" if max_dev < 1e-6 else "fail"

with rec.measure("核对：层级组合 201 相对 000 = (201 相对 101) ∘ (101 相对 000)") as e:
    t = 120.0
    composed = model.get_rotation(t, 101, anchor_plate_id=0) * model.get_rotation(t, 201, anchor_plate_id=101)
    direct = model.get_rotation(t, 201, anchor_plate_id=0)
    pt = pygplates.PointOnSphere(10.0, 20.0)
    dev = math.degrees(pygplates.GeometryOnSphere.distance(composed * pt, direct * pt))
    e["deviation_deg"] = float(f"{dev:.2e}")
    e["status"] = "ok" if dev < 1e-6 else "fail"

# ---- 2. 特征：每块板块一个多边形（小圆近似）----------------------------------------
CENTERS = {101: (10.0, 20.0), 201: (-25.0, 120.0), 301: (40.0, -110.0)}


def cap_polygon(lat0, lon0, radius_deg, n=72):
    center = pygplates.PointOnSphere(lat0, lon0)
    # 取极点为 center 的小圆：先在北极生成，再旋到 center
    pts = [pygplates.PointOnSphere(90.0 - radius_deg, a) for a in np.linspace(0, 360, n, endpoint=False)]
    to_center = pygplates.FiniteRotation(pygplates.PointOnSphere(90.0, 0.0), center)
    return pygplates.PolygonOnSphere([to_center * p for p in pts])


with rec.measure("创建 3 个板块多边形特征并写 .gpmlz / 读回") as e:
    features = []
    for pid, (la, lo) in CENTERS.items():
        f = pygplates.Feature()
        f.set_geometry(cap_polygon(la, lo, 18.0))
        f.set_reconstruction_plate_id(pid)
        f.set_name(PLATES[pid]["name"])
        features.append(f)
    fc = pygplates.FeatureCollection(features)
    gpml = work / "synthetic_plates.gpmlz"
    fc.write(str(gpml))
    back = pygplates.FeatureCollection(str(gpml))
    e["written_features"] = len(fc)
    e["read_back_features"] = len(back)
    e["file_bytes"] = gpml.stat().st_size
    e["status"] = "ok" if len(back) == len(fc) else "fail"

reconstructed: dict[int, dict[int, np.ndarray]] = {}
with rec.measure("重建 3 个多边形到 0/50/100/150/200 Ma") as e:
    for t in TIMES:
        snap = pygplates.ReconstructSnapshot(fc, model, float(t))
        for rg in snap.get_reconstructed_geometries():
            pid = rg.get_feature().get_reconstruction_plate_id()
            reconstructed.setdefault(pid, {})[t] = np.array(rg.get_reconstructed_geometry().to_lat_lon_array())
    e["snapshots"] = len(TIMES)

# ---- 3. 速度：pyGPlates 与解析值对照 -----------------------------------------------
with rec.measure("板块 101 现今速度：pyGPlates vs 解析（ω·R·sinΔ）") as e:
    pts = [pygplates.PointOnSphere(la, lo) for la in range(-20, 41, 10) for lo in range(0, 41, 10)]
    stage = model.get_rotation(0.0, 101, 1.0, anchor_plate_id=0)  # 1 Myr 阶段旋转
    vel = pygplates.calculate_velocities(pts, stage, 1.0, pygplates.VelocityUnits.cms_per_yr)
    speed = np.array([v.get_magnitude() for v in vel])
    pole = pygplates.PointOnSphere(*PLATES[101]["pole"])
    dist = np.array([pygplates.GeometryOnSphere.distance(p, pole) for p in pts])
    analytic = math.radians(PLATES[101]["rate"]) * R_KM * 1e5 / 1e6 * np.sin(dist)  # cm/yr
    # 测试点里有一个正好在欧拉极上（速度为 0），所以用"绝对误差 / 最大速度"而不是逐点相对误差
    err = np.abs(speed - analytic).max() / analytic.max()
    e["speed_cm_yr_range"] = [round(float(speed.min()), 2), round(float(speed.max()), 2)]
    e["max_err_over_max_speed"] = float(f"{err:.2e}")
    e["status"] = "ok" if err < 1e-3 else "fail"

# 出图用：现今所有板块上的速度箭头（北/东分量）
arrows = []
for pid, (la0, lo0) in CENTERS.items():
    grid = [pygplates.PointOnSphere(la0 + dla, lo0 + dlo) for dla in (-10, 0, 10) for dlo in (-12, 0, 12)]
    stage = model.get_rotation(0.0, pid, 1.0, anchor_plate_id=0)
    v = pygplates.calculate_velocities(grid, stage, 1.0, pygplates.VelocityUnits.cms_per_yr)
    ned = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down(grid, v)
    for p, n in zip(grid, ned):
        la, lo = p.to_lat_lon()
        arrows.append((lo, la, n.get_y(), n.get_x(), pid))

# ---- 4. GPlately ----------------------------------------------------------------
gplately = import_gplately()
with rec.measure("GPlately: PlateReconstruction + Points 重建到 100 Ma", version=gplately.__version__) as e:
    recon = gplately.PlateReconstruction(rotation_model=str(rot_path), static_polygons=str(gpml))
    lons = np.array([20.0, 25.0, 120.0, -110.0])
    lats = np.array([10.0, 12.0, -25.0, 40.0])
    gpts = gplately.Points(recon, lons, lats, time=0)
    rlons, rlats = gpts.reconstruct(100, return_array=True)
    # 与 pyGPlates 直接计算对照
    ref = model.get_rotation(100.0, 101, anchor_plate_id=0).get_inverse() * pygplates.PointOnSphere(10.0, 20.0)
    # GPlately 的 reconstruct 按 static polygons 指定板块号，从 0 Ma 到 100 Ma 用的是 get_rotation(100, pid)
    ref = model.get_rotation(100.0, 101, anchor_plate_id=0) * pygplates.PointOnSphere(10.0, 20.0)
    ref_lat, ref_lon = ref.to_lat_lon()
    e["point0_gplately"] = [round(float(rlats[0]), 4), round(float(rlons[0]), 4)]
    e["point0_pygplates"] = [round(ref_lat, 4), round(ref_lon, 4)]
    e["plate_ids"] = [int(x) for x in gpts.plate_id]
    ok = abs(rlats[0] - ref_lat) < 1e-3 and abs(((rlons[0] - ref_lon + 180) % 360) - 180) < 1e-3
    e["status"] = "ok" if ok else "fail"

with rec.measure("plate-model-manager：联网获取地球板块模型清单") as e:
    import contextlib

    from _common import LOG_DIR

    with contextlib.chdir(LOG_DIR):
        from plate_model_manager import PlateModelManager

        pm = PlateModelManager()
        names = pm.get_available_model_names()
    e["n_models"] = len(names)
    e["examples"] = [n for n in names if any(k in n.lower() for k in ("muller2019", "merdith2021", "muller2022", "zahirovic2022"))][:6]

# ---- 出图 ------------------------------------------------------------------------
import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

colors = {101: "tab:blue", 201: "tab:orange", 301: "tab:green"}
fig = plt.figure(figsize=(12, 6.2))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.gridlines(color="0.85", lw=0.5)
for pid, by_t in reconstructed.items():
    for t, arr in by_t.items():
        alpha = 1.0 - t / 260
        ax.fill(arr[:, 1], arr[:, 0], transform=ccrs.Geodetic(), facecolor=colors[pid], alpha=0.12 if t else 0.35,
                edgecolor=colors[pid], lw=1.6 if t == 0 else 0.8, ls="-" if t == 0 else "--")
        la, lo = arr.mean(axis=0)
        ax.text(lo, la, f"{t}", transform=ccrs.PlateCarree(), fontsize=7, color=colors[pid], ha="center", alpha=alpha)
a = np.array(arrows)
ax.quiver(a[:, 0], a[:, 1], a[:, 2], a[:, 3], transform=ccrs.PlateCarree(), scale=60, width=0.0022, color="k")
ax.set_title("Spike 02 - synthetic rotation model (pyGPlates): plate polygons at 0 Ma (solid) and 50-200 Ma (dashed);\n"
             "arrows = present-day velocities (cm/yr). Plate B moves relative to plate A (hierarchy test).", fontsize=9)
savefig(fig, "02_plate_kinematics.png")
plt.close(fig)

rec.save(pygplates_version=str(pygplates.Version.get_imported_version()), gplately_version=gplately.__version__)
