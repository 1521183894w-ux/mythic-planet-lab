"""U36-S4：真实C/D历史锚定的pyGPlates材料带小试，先出图与来源审计。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
import pygplates as pg
import yaml
from mythic.tectonics.geometry import R_KM, lonlat_to_xyz, omega_rotmat, unit
from mythic.tectonics.kinematics import RotationFileKinematics
from mythic.tectonics.reverse import PlateHistory
from mythic.tectonics.crust import CONT
from mythic.tectonics.sampling import native_cover
from mythic.tectonics.material_band import (build_network, point_array, reconstruct_band,
    reference_strip, spherical_triangle_area)

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache/u36_s4"
DATA = ROOT / "data/worlds/design_v2"
CONFIG = ROOT / "config/design/material_band_u36_s4.yaml"


def source_at_reference(eng, strip):
    centers = unit(strip["xyz"][strip["triangles"]].sum(axis=1))
    owner = native_cover(eng, centers)
    safe = np.maximum(owner, 0)
    known = (owner >= 0) & (eng.c.ctype[safe] == CONT)
    h0 = np.where(known, eng.c.T[safe], np.nan)
    area = abs(spherical_triangle_area(strip["xyz"], strip["triangles"]))
    uid = np.where(known, eng.c.uid[safe], -1)
    plate = np.where(owner >= 0, eng.c.plate[safe], -1)
    sources = {str(p): float(area[plate == p].sum()/area.sum()) for p in np.unique(plate)}
    rails = []
    for side, ids, expected in [("left", strip["grid"][:, 0], 301), ("right", strip["grid"][:, -1], 401)]:
        j = native_cover(eng, strip["xyz"][ids])
        good = j >= 0
        match = good & (eng.c.plate[np.maximum(j, 0)] == expected) & (eng.c.ctype[np.maximum(j, 0)] == CONT)
        rails.append(dict(side=side, expected_plate=expected, point_count=len(j),
                          expected_cont_fraction=float(match.mean()), uncovered_fraction=float((~good).mean())))
    return dict(H0_km=h0, source_uid=uid, source_plate=plate, known=known, area0_km2=area), dict(
        known_cont_area_fraction=float(area[known].sum()/area.sum()),
        source_plate_area_fractions=sources, sampled_original_host_count=int(len(np.unique(uid[known]))),
        H0_known_quantiles_km=np.quantile(h0[known], [0,.5,1]).tolist() if known.any() else [],
        rails=rails,
        source_definition="参考三角形中心的原生覆盖查询；UID为被查询宿主，不是精确裁切宿主后的物质份额，未重分配引擎体积")


def register_contact(eng, endpoints):
    """只在35Ma匹配一次实际C/D交界，保存共轭查询宿主UID，此后不重选。

    在给定段附近找C→D的直接类别转换。无覆盖、洋壳或多重接触不强行连线；
    直线拟合只给局部网络的参考方向，不称作唯一古断层。
    """
    initial=reference_strip(*endpoints,800,30)
    center=initial["xyz"][initial["grid"][:,initial["grid"].shape[1]//2]]
    offsets=np.arange(-400,400.1,5.)
    query=unit(np.cos(offsets/R_KM)[None,:,None]*center[:,None,:]
               +np.sin(offsets/R_KM)[None,:,None]*initial["east"])
    owner=native_cover(eng,query.reshape(-1,3)).reshape(len(center),-1)
    safe=np.maximum(owner,0)
    plate=np.where((owner>=0)&(eng.c.ctype[safe]==CONT),eng.c.plate[safe],-1)
    contacts=[]
    for row in range(len(center)):
        transition=np.flatnonzero((plate[row,:-1]==301)&(plate[row,1:]==401))
        record=dict(along_km=float(initial["along_km"][row]),transition_count=len(transition))
        if len(transition)==1:
            j=int(transition[0])
            record.update(offset_km=float((offsets[j]+offsets[j+1])/2),
                left_uid=int(eng.c.uid[owner[row,j]]),right_uid=int(eng.c.uid[owner[row,j+1]]))
        contacts.append(record)
    valid=[r for r in contacts if r["transition_count"]==1]
    if len(valid)<3:
        raise RuntimeError("35Ma直接C/D接触支持不足，不建立材料带")
    along=np.array([r["along_km"] for r in valid])
    offset=np.array([r["offset_km"] for r in valid])
    fit=np.polyfit(along,offset,1)
    shift=np.polyval(fit,[0,initial["along_km"][-1]])
    registered=unit(np.cos(shift/R_KM)[:,None]*endpoints
                    +np.sin(shift/R_KM)[:,None]*initial["east"])
    return registered,dict(method_class="interpolated",contacts=contacts,
        valid_crossings=len(valid),total_transects=len(center),search_spacing_km=5,
        endpoint_offset_km=shift.tolist(),fit_residual_p90_km=float(np.quantile(abs(np.polyval(fit,along)-offset),.9)),
        note="给定G8段附近35Ma的原生类别转换；5km是查询间距，材料仍L7。拟合不是断层真值，UID是相邻查询宿主，不是全局接触拓扑。")


def synthetic_rotation(case):
    speed = 5/R_KM
    if case == "opening":
        ws = [np.array([0,0,-speed]), np.array([0,0,speed])]
    elif case == "slip":
        ws = [np.array([0,-speed,0]), np.array([0,speed,0])]
    else:
        ws = [np.array([.3,.2,.5])*speed]*2
    h = PlateHistory()
    for plate, w in enumerate(ws, 1):
        h.add(plate, 0, f"control_{plate}", [(0, np.eye(3)), (35, omega_rotmat(w,-35))])
    path = h.write_rot(DEST / f"{case}.rot")
    return pg.RotationModel(str(path))


def main():
    started = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rotation_path = ROOT / "data/worlds/design_v2/history_v2.rot"
    rot = pg.RotationModel(str(rotation_path))
    kin = RotationFileKinematics(rotation_path)
    t0 = cfg["initial_time_ma"]
    ref = point_array([rot.get_rotation(t0,cfg["reference_anchor"])*pg.PointOnSphere(lat,lon)
                       for lon,lat in cfg["reference_line_present_lon_lat"]])
    with (DEST / "control/reference_35Ma.pkl").open("rb") as stream:
        eng = pickle.load(stream)
    design_ref=ref.copy()
    design_strip=reference_strip(*design_ref,cfg["width0_km"],30)
    _,design_source=source_at_reference(eng,design_strip)
    ref,registration=register_contact(eng,ref)
    report = dict(checkpoint="U36-S4", version=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()+"+U36-S4",
        framework_version=pg.__version__, config=cfg, physical_engine_changed=False,
        method_classes={"reference_geometry_width":"user_override", "network_velocity_strain":"interpolated",
                        "reference_thickness":"interpolated", "area_audit":"procedural_constrained"},
        reference_registration=registration,unregistered_source=design_source,
        design_endpoints_xyz=design_ref.tolist(),registered_endpoints_xyz=ref.tolist(),
        controls=[], convergence=[], width_cases=[], guard_cases=[],interpolation_cases=[])
    times = np.arange(t0, -1, -5)
    for width in cfg["width_sensitivity_km"]:
        for dx in (60.,30.,15.):
            strip = reference_strip(*ref,width,dx)
            src, source_report = source_at_reference(eng,strip)
            for dt in (1.,.5,.25):
                model, features = build_network(strip,rot,t0,tag=f"CD_W{width}_dx{int(dx)}")
                timeline, snapshots = reconstruct_band(strip,model,t0,dt,times)
                end = snapshots[0.]
                known, area0, h0 = src["known"],src["area0_km2"],src["H0_km"]
                initial_volume = float(np.sum(area0[known]*h0[known]))
                transported_volume = float(np.sum(end["area_km2"][known]*h0[known]/end["J"][known]))
                framework_volume = float(np.sum(end["area_km2"][known]*h0[known]/end["framework_beta"][known]))
                row = dict(width0_km=width,dx_requested_km=dx,dx_actual_km=strip["actual_dx_km"],
                    dy_actual_km=strip["actual_dy_km"],dt_myr=dt,triangle_count=len(strip["triangles"]),
                    timeline=timeline, initial_known_volume_km3=initial_volume,
                    area_column_volume_residual_km3=transported_volume-initial_volume,
                    framework_known_volume_relative_residual=framework_volume/initial_volume-1,
                    final_known_H_quantiles_km=np.quantile(h0[known]/end["J"][known],[0,.1,.5,.9,1]).tolist(),
                    source=source_report)
                report["convergence"].append(row)
                if dx == 30 and dt == .25:
                    report["width_cases"].append(row)
                    if width == cfg["width0_km"]:
                        report["baseline"] = row
                        arrays = {**strip, **src, "times_ma": times}
                        for k in ("xyz","center_xyz","J","framework_beta","area_km2","outside_centers"):
                            arrays["timeline_"+k] = np.stack([snapshots[float(t)][k] for t in times])
                        np.savez_compressed(DEST / "material_band.npz", **arrays)
                        # GPML特征ID在整个时间序列持续，文件保留可重新读入的拓扑。
                        pg.FeatureCollection(features).write(str(DEST / "material_network.gpml"))
                print("band",width,dx,dt,"J",round(timeline[-1]["area_ratio"],6),"frameworkV",round(row["framework_known_volume_relative_residual"],6),flush=True)
    strip = reference_strip(*ref,cfg["width0_km"],30.)
    for guard in (0.,200.,400.,800.):
        model,_ = build_network(strip,rot,t0,guard_km=guard)
        try:
            timeline,_ = reconstruct_band(strip,model,t0,.25,times)
            report["guard_cases"].append(dict(guard_km=guard,timeline=timeline,status="ran"))
        except (pg.InvalidLatLonError,ValueError) as error:
            report["guard_cases"].append(dict(guard_km=guard,status="failed",error=str(error)))
    for smooth in ("none","barycentric","natural_neighbour"):
        for nn in (False,True):
            model,_=build_network(strip,rot,t0,strain_smoothing=smooth)
            timeline,_=reconstruct_band(strip,model,t0,.25,times,natural_neighbour=nn)
            report["interpolation_cases"].append(dict(strain_smoothing=smooth,natural_neighbor_velocity=nn,timeline=timeline))
    # 真实边界上的法向/走滑均来自已给定法向与真实旋转，不依赖最近点连线。
    from mythic.tectonics.deformation_kinematics import boundary_jump
    center = strip["xyz"][strip["grid"][:,strip["grid"].shape[1]//2]]
    _, omegas = kin.step_rotations([301,401],35)
    vn,vt = boundary_jump(center,np.broadcast_to(strip["east"],center.shape),omegas[301],omegas[401])
    report["initial_boundary_jump"] = dict(normal_km_myr=vn.tolist(),slip_km_myr=vt.tolist())
    # 沿同一段的材料三角形及持久UID只是旁观；不写回eng，不叠加旧B5减薄。
    control_ref=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,30)
    for case in ("opening","slip","rigid"):
        control_rot=synthetic_rotation(case)
        for dt in (1.,.5,.25):
            model,_=build_network(control_ref,control_rot,t0,plates=(1,2),tag=case)
            timeline,snapshots=reconstruct_band(control_ref,model,t0,dt,times)
            report["controls"].append(dict(case=case,dt_myr=dt,timeline=timeline,
                continuous_truth="slip的连续纬向剪切散度为0；rigid面积与形变为0；opening非零正伸展"))
            if dt==.25:
                np.savez_compressed(DEST/f"control_{case}.npz",xyz0=control_ref["xyz"],xyz=snapshots[0.]["xyz"],
                    triangles=control_ref["triangles"],J=snapshots[0.]["J"],framework_beta=snapshots[0.]["framework_beta"])
    report["elapsed_s"]=time.perf_counter()-started
    report["input_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        "config/design/material_band_u36_s4.yaml","data/worlds/design_v2/history_v2.rot","config/design/history_v2_u35_c10_progressive.yaml"]}
    report["limitations"]=["参考线与W0为显式假设，原生覆盖匹配只是核对，未证明共同古断层与物理流变宽度",
        "网格只是网络材料审计采样；初始厚度仍为L7约60km，密集采样不增加物理分辨率",
        "H0/J的闭合是定义性材料账；框架beta与独立几何残差才用于评估时间积分和插值",
        "无断层基底位移、岩石圈热演化、沉积、侵蚀和水位，不输出B5地貌或海陆新结果",
        "网络端部可发生交换；用缓冲长度对照，不把瞬时网络外边界当材料边界"]
    (DATA/"u36_s4_material_band.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("done",report["elapsed_s"],flush=True)


if __name__ == "__main__":
    main()
