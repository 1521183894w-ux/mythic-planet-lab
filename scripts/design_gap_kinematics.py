"""U36-S3：连续解析半球与实际离散覆盖、提交路径的对照。

连续半球有明确法向/面积真值，三种空间分辨率、三种边界相位。
不启用碰撞、弧增厚或地壳流动；提交试验只隔离覆盖/补缝反馈。
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import R_KM, omega_rotmat
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.gap_diagnostics import rift_candidates, margin_normal_diagnostics
from mythic.tectonics.deformation_kinematics import affine_band, integrate_band

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/".cache/u36_s3"
DATA = ROOT/"data/worlds/design_v2"


def build(mesh, case, phase=0):
    q = omega_rotmat(np.array([.31, -.23, .19]), 1.+phase*.71)
    a, b, z = q.T
    c = Crust(mesh.n)
    c.plate[:] = np.where(mesh.xyz @ a >= 0, 1, 2)
    c.ctype[:] = CONT
    c.T[:] = 35.
    c.origin[:] = 1
    speed = 5./R_KM
    if case == "opening":
        omega = {1: -speed*z, 2: speed*z}
    elif case == "slip":
        omega = {1: -speed*a, 2: speed*a}
    else:
        omega = {1: np.array([.7, -.3, .5])*speed, 2: np.array([.7, -.3, .5])*speed}
    eng = Engine(mesh, c, Params(rift_area_conservative=True), 100., omega)
    eng.cont_gap_pairs = {1002: dict(knots=[(0., 35.), (1000., 20.)], holes=False)}
    return eng, (a, b, z), omega


def set_time(eng, omega, elapsed, increment):
    eng.t = 100.-elapsed
    eng.steps_since_commit = increment
    eng.R = {q: omega_rotmat(w, elapsed) for q, w in omega.items()}


def continuous_gap(eng, x, axis):
    # A 初始 n·x>=0；B 初始 n·x<0。旋转后的法向是精确半球边界。
    na, nb = eng.R[1]@axis, eng.R[2]@axis
    return (x@na < 0) & (x@nb >= 0)


def quadrature_gap(eng, axis, count):
    # 与材料网格独立的等面积 Fibonacci 点；全局积分避免只看固定节点的相位锁定。
    index=np.arange(count,dtype=float)+.5
    mu=1.-2.*index/count
    phi=index*(np.pi*(3.-np.sqrt(5.)))
    radius=np.sqrt(1.-mu*mu)
    x=np.column_stack([radius*np.cos(phi),radius*np.sin(phi),mu])
    gap=(eng.covering(1,x)<0)&(eng.covering(2,x)<0)
    truth=continuous_gap(eng,x,axis)
    cell=4*np.pi*R_KM**2/count
    return dict(count=count,weight_km2=cell,gap_area_km2=float(gap.sum()*cell),
                outside_true_gap_km2=float((gap&~truth).sum()*cell),
                true_gap_missed_km2=float((truth&~gap).sum()*cell))


def summarize(mesh, eng, axes, omega, case, phase, elapsed=20.):
    a, b, z = axes
    set_time(eng, omega, elapsed, int(elapsed))
    pos = eng.positions()
    v = eng.cover_fixed(pos)
    gap = v < 0
    truth = continuous_gap(eng, mesh.xyz, a)
    # 两板块穷举查询，检查 cover_fixed 的候选加速是否另有漏覆盖。
    exhaustive = (eng.covering(1, mesh.xyz) < 0) & (eng.covering(2, mesh.xyz) < 0)
    candidates = rift_candidates(eng)
    norms = margin_normal_diagnostics(eng, candidates)
    angle = 10.*elapsed/R_KM if case == "opening" else 0.
    exact_area = 2*R_KM**2*angle
    area = lambda mask: float(mesh.area_km2[mask].sum())
    record = dict(case=case, level=mesh.level, phase=phase, h_km=mesh.h_km,
                  exact_gap_area_km2=exact_area, sampled_true_gap_area_km2=area(truth),
                  discrete_gap_area_km2=area(gap), outside_true_gap_km2=area(gap & ~truth),
                  true_gap_missed_km2=area(truth & ~gap),
                  accelerated_query_mismatches=int(np.sum(gap != exhaustive)),
                  old_opening_candidate_area_km2=float(candidates.get("target_area_km2",np.array([])).sum()),
                  normal_support_area_km2={},
                  independent_quadrature=[quadrature_gap(eng,a,400000)]+([quadrature_gap(eng,a,800000)] if phase==0 else []))
    if norms:
        for key in np.unique(norms["status"]):
            record["normal_support_area_km2"][str(key)] = float(candidates["target_area_km2"][norms["status"] == key].sum())
    # 中部同一物理窗口；km 坐标，不把高密采样冒充更高模型分辨率。
    if phase == 0 and mesh.level == 7:
        xx, yy = np.meshgrid(np.linspace(-450,450,361),np.linspace(-2500,2500,501))
        lon, lat = xx/R_KM, yy/R_KM
        x = (np.cos(lat)*np.sin(lon))[...,None]*a + (np.cos(lat)*np.cos(lon))[...,None]*b + np.sin(lat)[...,None]*z
        xf = x.reshape(-1,3)
        covered = (eng.covering(1,xf)>=0)|(eng.covering(2,xf)>=0)
        true = continuous_gap(eng,xf,a)
        code = np.zeros(len(xf),np.uint8)
        code[~covered & true] = 1
        code[~covered & ~true] = 2
        code[covered & true] = 3
        np.savez_compressed(DEST/f"synthetic_map_{case}.npz",x_km=xx[0],y_km=yy[:,0],
                            code=code.reshape(xx.shape),truth=true.reshape(xx.shape))
    return record


def cadence(mesh, case, interval):
    eng, axes, omega = build(mesh, case)
    initial_volume = eng.cont_volume()
    for elapsed in range(interval,21,interval):
        set_time(eng,omega,elapsed,interval)
        eng.commit()
    audits = getattr(eng,"rift_area_audit",[])
    source_areas, beta, target_areas = [], [], []
    for r in audits:
        source_areas.append(float(r["source_area_km2"].sum())/(np.pi*R_KM))
        beta.extend(r["source_beta"])
        target_areas.extend(r["target_area_km2"])
    return dict(case=case, level=mesh.level, interval_myr=interval,
                total_birth_area_km2=float(sum(target_areas)), transfer_count=len(audits),
                source_equivalent_width_median_km=float(np.median(source_areas)) if source_areas else 0.,
                source_equivalent_width_max_km=max(source_areas,default=0.),
                source_width_definition="每次实际供体面积/连续半球张开半环长πR；slip是比较分母，不是物理变形带",
                beta_p90=float(np.quantile(beta,.9)) if beta else 1.,
                final_min_thickness_km=float(eng.c.T.min()),
                final_min_cont_thickness_km=float(eng.c.T[eng.c.ctype == CONT].min()),
                volume_residual_km3=float(eng.cont_volume()-initial_volume),
                note="仅旋转与commit；不包含相互作用、侵蚀、流动；出生面积不是净伸展")


def main():
    start=time.perf_counter()
    DEST.mkdir(parents=True,exist_ok=True)
    records, cadences = [], []
    for level in (5,6,7):
        mesh=load_mesh(level)
        for phase in (0,1,2):
            for case in ("opening","slip","rigid"):
                eng,axes,omega=build(mesh,case,phase)
                r=summarize(mesh,eng,axes,omega,case,phase)
                records.append(r)
                print("endpoint",level,phase,case,round(r["discrete_gap_area_km2"]),round(r["old_opening_candidate_area_km2"]),flush=True)
        for case in ("opening","slip","rigid"):
            for interval in (1,5,10):
                r=cadence(mesh,case,interval)
                cadences.append(r)
                print("commit",level,case,interval,round(r["total_birth_area_km2"]),round(r["final_min_thickness_km"],3),flush=True)
    bands=[]
    for width in (100.,200.,400.):
        for dx in (40.,20.,10.):
            n=max(1,int(np.ceil(width/dx)))
            for dt in (.5,1.,5.):
                r=integrate_band(width,35.,10.,20.,dt)
                f=affine_band(width,200.,150.,.6)["F"]
                # 独立四边形面积，整带体积（每 km 走向长）。
                edges=np.linspace(0.,width,n+1)
                polygon=np.stack([np.column_stack([edges[:-1],np.zeros(n)]),np.column_stack([edges[1:],np.zeros(n)]),np.column_stack([edges[1:],np.full(n,1000.)]),np.column_stack([edges[:-1],np.full(n,1000.)])],axis=1)@f.T
                area=abs(np.sum(polygon[:,:,0]*np.roll(polygon[:,:,1],-1,axis=1)-polygon[:,:,1]*np.roll(polygon[:,:,0],-1,axis=1)))/2
                bands.append(dict(width0_km=width,dx_requested_km=dx,dx_actual_km=width/n,dt_myr=dt,
                                  polygon_area_km2=float(area),**r))
    report=dict(unit="km, km2, km3, Myr", method_class="procedural_constrained",
                normal_method_class="interpolated", version=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()+"+U36-S3",
                continuous_truth="旋转半球；opening缺口面积=2R²α，α=相对旋转角；slip/rigid=0",
                endpoint=records,cadence=cadences,band= bands,elapsed_s=time.perf_counter()-start)
    (DATA/"u36_s3_synthetic.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("done",report["elapsed_s"],flush=True)


if __name__=="__main__":
    main()
