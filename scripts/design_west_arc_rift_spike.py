"""U35-C2：以 C1 的 38 Ma 真实地壳单元隔离比较事件延迟和转属重采样。
仅运动学重放，不做碰撞、火山、侵蚀或全局填隙；不是完整世界结果。
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
from mythic.tectonics.crust import Crust, CONT, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.geometry import lonlat_to_xyz, xyz_to_lonlat, unit, km_from_chord
from mythic.tectonics.kinematics import RotationFileKinematics
import design_west_arc_report as report

def main():
    mesh = load_mesh(7)
    path = ROOT / "data/worlds/design_v2/u35_c2_rift_input.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {key:np.asarray(values,dtype=raw["dtypes"][key]) for key,values in raw["fields"].items()}
    c = Crust(len(fields["node"]))
    for key in c.FIELDS:
        setattr(c, key, fields[key].copy())
    pos = lonlat_to_xyz(fields["lon"], fields["lat"])
    u, _, vt = np.linalg.svd(mesh.xyz[c.node].T @ pos)
    rotation = (u @ vt).T
    assert np.max(np.linalg.norm(mesh.xyz[c.node] @ rotation.T-pos, axis=1)) < 1e-10
    kin = RotationFileKinematics(ROOT / "data/worlds/design_v2/history_v2.rot")
    expected = pos @ kin.total_from(304, 0., 38.)
    weights = mesh.area_km2[c.node]
    center = unit((expected * weights[:, None]).sum(0))
    initial_volume = float((weights*c.T).sum())
    cases, traces = [], []
    definitions = [(38, True), (38, False), (35, True), (35, False)]
    for when, preserve in definitions:
        eng = Engine(mesh, c.gather(np.arange(len(c))), Params(), 38., {302: np.zeros(3)})
        eng.R[302] = rotation.copy()
        eng._rebuild()
        trace = []
        def detach():
            omega = kin.step_rotations([304], max(when, 1))[1][304]
            moved = np.flatnonzero(eng.alive & (eng.c.plate == 302))
            if preserve:
                eng.rift_units(302, 304, moved, omega)
            else:
                eng.add_plate(304, omega)
                eng.transfer_units(302, 304, moved, kind="terrane_rift")
        if when == 38:
            detach()
        for t in range(38, 0, -1):
            rotations, omega = kin.step_rotations(eng.plates, t)
            for q in eng.plates:
                eng.R[q] = rotations[q] @ eng.R[q]
            eng.omega = omega
            eng.t = float(t-1)
            if t-1 == when and when != 38:
                detach()
            live = eng.alive
            projected = eng.positions()[live] @ kin.total_from(304, 0., float(t-1))
            centroid = unit((projected*eng.area[live, None]).sum(0))
            trace.append({"t_Ma":t-1, "centroid_error_km":float(km_from_chord(np.linalg.norm(centroid-center)))})
        live = eng.alive
        final = eng.positions()[live]
        centroid = unit((final*eng.area[live, None]).sum(0))
        d1 = km_from_chord(cKDTree(expected).query(final)[0])
        d2 = km_from_chord(cKDTree(final).query(expected)[0])
        row = {"event_t_Ma":when, "preserve_frame":preserve, "n":int(live.sum()),
               "area_km2":float(eng.area[live].sum()),
               "volume_km3":float((eng.area[live]*eng.c.T[live]).sum()),
               "centroid_error_km":float(km_from_chord(np.linalg.norm(centroid-center))),
               "symmetric_max_nearest_km":float(max(d1.max(),d2.max()))}
        cases.append(row)
        traces.append(trace)
        row["_final"] = final
    report.setup_fonts()
    design = fwd.yaml.safe_load((ROOT/"config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    built = fwd.build(design, report.GRID)
    E = fwd.design_elevation(design, report.GRID, built)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,4,figsize=(21,5))
    for ax,row in zip(axes,cases):
        report.map_panel(ax,E,built["land"],
            str(row["event_t_Ma"])+" Ma 转属 / "+("保留参照系" if row["preserve_frame"] else "重新采样")+
            "\n单元 "+str(row["n"])+"；质心误差 "+format(row["centroid_error_km"],".2f")+" km",
            extent=(-23,-6,33,43))
        exlo,exla = xyz_to_lonlat(expected)
        lo,la = xyz_to_lonlat(row.pop("_final"))
        ax.scatter(exlo,exla,s=30,facecolors="none",edgecolors="#763d8e",label="38 Ma 按设计运动")
        ax.scatter(lo,la,s=10,c="#d95f02",label="隔离重放")
    axes[0].legend(fontsize=8,loc="upper left")
    fig.suptitle("U35-C2：四组隔离运动学重放，分开事件延迟与重采样",fontsize=15)
    fig.subplots_adjust(bottom=.25,top=.8,wspace=.15)
    fig.text(.01,.02,"真实输入：C1 38 Ma、L7 的29个 ALB 来源单元；同一旋转文件。仅运动学重放，不是完整世界预测。\n"
             "运动 user_override；转属 procedural_constrained；栅格 interpolated；示意高程 visual_only。单位：km、km²、km³；代码 "+fwd.code_version(),fontsize=9)
    report.elevation_legend(fig,.14)
    out = ROOT / "figures/design/v2/forward/20_west_arc_c2_rift_spike.png"
    fig.savefig(out,dpi=125)
    plt.close(fig)
    data = {"input":str(path.relative_to(ROOT)), "source_n":len(c), "source_area_km2":float(weights.sum()),
            "source_volume_km3":initial_volume, "method_class":"procedural_constrained",
            "motion":"user_override", "resolution":"L7", "assumptions":"仅运动学；关闭所有物理改造与填隙，以隔离转属的误差",
            "reference":"沿用 docs/tectonics_method.md 的有限旋转和既有重采样；新增方案继承母块参照系",
            "cases":cases, "traces":traces}
    report.write_json(ROOT/"data/worlds/design_v2/u35_c2_rift_spike.json",data)
    print(json.dumps({k:v for k,v in data.items() if k!="traces"},ensure_ascii=False,indent=2))
if __name__ == "__main__":
    main()
