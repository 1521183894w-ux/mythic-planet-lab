"""U35-C1：从新旧真实运行数据生成对照图与可核查的诊断报表。"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
from design_west_arc_diagnostic import KEY_TIMES
from mythic.figures.style import setup_fonts, shaded_elevation, elevation_cmap_norm
from mythic.metrics.grid import LatLonGrid
from mythic.tectonics.mesh import load_mesh, Raster
from mythic.tectonics.geometry import unit, xyz_to_lonlat, km_from_chord
from mythic.tectonics.kinematics import RotationFileKinematics

CACHE = ROOT / ".cache/u35_c1"
OUT = ROOT / "figures/design/v2/forward"
GRID = LatLonGrid(1024, 512)
EXT = (-26., 6., 30., 44.)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def stamp(fig, extra=""):
    fig.text(.01, .012,
        "U35-C1 · L7 / 1024×512 · 代码 " + fwd.code_version() +
        "\n地壳 procedural_constrained；引擎高程 parameterized_physics；终图高程 visual_only；终图/运动 user_override；栅格 interpolated" +
        ("\n" + extra if extra else ""), fontsize=8, color="#444", va="bottom")


def elevation_legend(fig, y):
    cmap,norm=elevation_cmap_norm()
    ax=fig.add_axes((.65,y,.32,.025))
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax,
                    orientation="horizontal",ticks=[-8000,-4000,0,2000,4000,6000])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("高程 m（配色沿用标准图组；地图叠加阴影）",fontsize=8)

def map_panel(ax, E, land, title, extent=EXT, contour=None):
    x0,x1,y0,y1 = extent
    sx=(GRID.lon>=x0)&(GRID.lon<=x1)
    sy=(GRID.lat>=y0)&(GRID.lat<=y1)
    rgb=shaded_elevation(E, GRID.pixel_km_equator, vert_exag=3., land=land)[...,:3]
    ax.imshow(rgb[np.ix_(sy,sx)], origin="upper",
              extent=(GRID.lon[sx][0], GRID.lon[sx][-1], GRID.lat[sy][-1], GRID.lat[sy][0]),
              interpolation="nearest", aspect="equal")
    if contour is not None and np.any(contour[np.ix_(sy,sx)]):
        ax.contour(GRID.lon[sx], GRID.lat[sy], contour[np.ix_(sy,sx)].astype(float),
                   levels=[.5], colors="#b62471", linewidths=.9)
    ax.set(xlim=(x0,x1),ylim=(y0,y1),title=title,xlabel="经度 °E",ylabel="纬度 °N")
    ax.grid(alpha=.25)
    bar_km=500.
    dx=bar_km/(111.2*np.cos(np.radians(y0+1.)))
    ax.plot([x0+1,x0+1+dx],[y0+.65,y0+.65],color="#202020",lw=2)
    ax.text(x0+1,y0+.9,"500 km",fontsize=8)


def feature_lines(ax, design):
    for group in ("straits","mountains","arcs","islands","gulfs"):
        for q in design.get(group,[]):
            if q["id"] not in ("T1","M31","R17","G28"):
                continue
            a=np.array(q["line"])
            ax.plot(a[:,0],a[:,1],"--",lw=1,color="#743d7f")
            offset={"T1":(-20,-23),"M31":(7,10),"R17":(7,-18),"G28":(-12,5)}[q["id"]]
            ax.annotate(q["id"],xy=a[-1],xytext=offset,textcoords="offset points",
                fontsize=10,color="#52235e",
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none","pad":1},
                arrowprops={"arrowstyle":"-","color":"#743d7f","lw":.6})


def compare_npz(a, b):
    diffs=[]
    with np.load(a) as za, np.load(b) as zb:
        if set(za.files)!=set(zb.files):
            return {"identical":False,"different_keys":sorted(set(za.files)^set(zb.files))}
        for k in za.files:
            x,y=za[k],zb[k]
            equal = x.shape==y.shape and x.dtype==y.dtype
            if equal:
                equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x,y)
            if not equal:
                diffs.append(k)
        return {"identical":not diffs,"arrays_checked":len(za.files),"different_arrays":diffs}


def expected_context(design,hcfg,b,ids,info):
    area=GRID.cell_area_km2
    blocks={}
    for code,blk in info.items():
        mask=ids==blk["id"]
        blocks[blk["id"]]={"code":code,"color":blk["color"],
             "centroid":unit((GRID.xyz[mask]*area[mask][:,None]).sum(0))}
    hist=fwd.build_history(hcfg,{k:v["centroid"] for k,v in blocks.items()})
    return hist,blocks


def main():
    setup_fonts()
    OUT.mkdir(parents=True,exist_ok=True)
    design=yaml.safe_load((ROOT/"config/design/final_map_v2.yaml").read_text(encoding="utf-8"))
    design=fwd.apply_options(design,set())
    hcfg=yaml.safe_load((ROOT/"config/design/history_v2.yaml").read_text(encoding="utf-8"))
    b=fwd.build(design,GRID)
    ids,info=fwd.assign_blocks(hcfg,GRID,b)
    Efin=fwd.design_elevation(design,GRID,b)
    hist,blocks=expected_context(design,hcfg,b,ids,info)
    mesh=load_mesh(7)
    raster=Raster(mesh,512)
    cases={}
    for case in ("previous","latest"):
        with xr.open_dataset(CACHE/case/"world.nc") as ds:
            cases[case]={"E":ds.elevation.values.copy(),"land":ds.land.values>.5}
        cases[case]["summary"]=read_json(CACHE/case/"summary.json")
        cases[case]["obs"]=read_json(CACHE/case/"observations.json")

    neutrality=compare_npz(CACHE/"latest_plain/forward.npz",CACHE/"latest/forward.npz")
    neutrality["previous_extra_ledger"] = compare_npz(CACHE/"previous_no_ledger.npz",CACHE/"previous/forward.npz")
    ja=read_json(CACHE/"latest_plain/forward.json")
    jb=read_json(CACHE/"latest/forward.json")
    neutrality["events_identical"]=ja["events"]==jb["events"]
    neutrality["budget_identical"]=ja["budget"]==jb["budget"]
    if not all(neutrality[k] for k in ("identical","events_identical","budget_identical")) or not neutrality["previous_extra_ledger"]["identical"]:
        write_json(CACHE/"neutrality_failure.json",neutrality)
        raise RuntimeError("观测改变了模拟输出；停止生成本检查点结论")

    baseline=read_json(CACHE/"baseline/forward_v2.json")
    old=cases["previous"]["summary"]
    selected=("comparison","A_engine_0Ma","hypsometry_0Ma","volume_change","budget_km3")
    reproduction={k:baseline[k]==old[k] for k in selected}

    strait=next(q for q in design["straits"] if q["id"]=="T1")
    midpoint=np.mean(np.asarray(strait["line"],float),axis=0)
    j=int(np.argmin(abs(GRID.lon-midpoint[0])))
    i=int(np.argmin(abs(GRID.lat-midpoint[1])))
    def section_width(land):
        if land[i,j]:
            return {"width_km":0.,"center_is_land":True}
        lo=hi=i
        while lo>0 and not land[lo-1,j]: lo-=1
        while hi+1<land.shape[0] and not land[hi+1,j]: hi+=1
        width=(hi-lo+1)*180./GRID.ny*np.pi/180.*6371.
        return {"width_km":float(width),"center_is_land":False,
                "north_water_cell_lat":float(GRID.lat[lo]),"south_water_cell_lat":float(GRID.lat[hi])}
    sections={"longitude":float(GRID.lon[j]),"center_lat":float(GRID.lat[i]),
              "definition":"T1 设计中点所在经线、包含中点的连续水道南北跨度；不是最窄宽度或 A 层海峡指标。",
              "resolution_lat_km":float(180./GRID.ny*np.pi/180.*6371.),
              "design":section_width(b["land"]),
              "previous":section_width(cases["previous"]["land"]),
              "latest":section_width(cases["latest"]["land"])}
    fig,axes=plt.subplots(1,3,figsize=(21,4.5))
    panels=[("终图：设计示意",Efin,b["land"]),
            ("旧设定重算：撕裂断层到 6 Ma",cases["previous"]["E"],cases["previous"]["land"]),
            ("最新配置：撕裂断层到 0 Ma",cases["latest"]["E"],cases["latest"]["land"])]
    for ax,(title,E,land) in zip(axes,panels):
        map_panel(ax,E,land,title,contour=b["land"])
        feature_lines(ax,design)
    for ax,key in zip(axes,("design","previous","latest")):
        ax.plot([GRID.lon[j],GRID.lon[j]],[33,40],":",color="#333",lw=.7)
        ax.text(.02,.98,f"同一截面水道跨度 {sections[key]['width_km']:.0f} km",
                transform=ax.transAxes,va="top",fontsize=9,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.suptitle("西口三方对照：同范围、同分辨率、同高程配色（m）",fontsize=17)
    fig.subplots_adjust(bottom=.18,top=.86,wspace=.16)
    stamp(fig,"紫线为终图海岸；旧图原件保存在 u35_before_c1/；旧设定重算是否复现原摘要见诊断 JSON。")
    elevation_legend(fig,0.11)
    fig.savefig(OUT/"15_west_arc_c1_comparison.png",dpi=130)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(20,8))
    trajectory=[]
    for ax,t in zip(axes.flat,KEY_TIMES):
        with np.load(CACHE/"latest"/f"snapshot_{t:03d}.npz") as snap:
            sea=cases["latest"]["obs"]["snapshot_sea_level_m"][str(t)]
            E=raster(snap["elev"]).astype(float)-sea
        own,_=fwd.raster_at(hist,blocks,ids,GRID,float(t),GRID)
        target=own==info["ALB"]["id"]
        map_panel(ax,E,E>0,f"{t} Ma · 海平面 {sea:+.0f} m",contour=target)
        with np.load(CACHE/"latest"/f"parcels_{t:03d}.npz") as p:
            src=(p["origin"]==info["ALB"]["id"])&(p["ctype"]==1)
            ax.scatter(p["lon"][src],p["lat"][src],s=12,facecolors="none",edgecolors="#ff7f00",linewidths=.8)
            trajectory.append({"t_Ma":t,"source_parcels":int(src.sum()),
                 "source_lat_min":float(p["lat"][src].min()) if src.any() else None,
                 "source_T_p90_km":float(np.percentile(p["T"][src],90)) if src.any() else None,
                 "source_min_raw_elevation_m":float(p["raw_elevation_m"][src].min()) if src.any() else None,
                 "source_min_raw_relative_to_sea_m":float(p["raw_elevation_m"][src].min()-sea) if src.any() else None})
    fig.suptitle("最新配置逐帧证据：紫线＝设计刚体轮廓；橙圈＝原 ALB 来源的实际陆壳单元",fontsize=15)
    fig.subplots_adjust(bottom=.13,top=.92,hspace=.32,wspace=.15)
    stamp(fig,"所有标题均为实际采样年代；背景为固定网格视图，橙圈是原始单元位置，二者不混作同一图层。")
    elevation_legend(fig,0.085)
    fig.savefig(OUT/"16_west_arc_c1_evolution.png",dpi=125)
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for case,color in (("previous","#b04b40"),("latest","#207a9b")):
        rows=[r for r in cases[case]["obs"]["rows"] if r["t_Ma"]<=10]
        t=[r["t_Ma"] for r in rows]
        label="旧设定" if case=="previous" else "最新配置"
        for ax,key,scale,suffix in ((axes[0,0],"alb_cont_volume_km3",1e6,"陆壳总体积"),
                   (axes[0,1],"alb_T_p90_km",1.,"厚度 p90"),
                   (axes[1,0],"alb_original_volume_km3",1e6,"原 ALB 来源宿主上的总体积（含混入）"),
                   (axes[1,1],"collision_deposited_cumulative_km3",1e6,"碰撞累计转入 ALB 的体积（未扣后续剥蚀）")):
            ax.plot(t,[np.nan if r[key] is None else r[key]/scale for r in rows],"-o",ms=3,color=color,label=label)
            ax.set(title=suffix,xlabel="距今 Ma",ylabel="km" if scale==1 else "百万 km³",xlim=(10,0))
            ax.axvline(6,color="#666",ls=":",lw=1)
            ax.grid(alpha=.3);ax.legend()
    fig.suptitle("最后 10 Myr：同一历史输入，仅 WT—ALB 撕裂断层结束时刻不同",fontsize=15)
    fig.tight_layout(rect=(0,.10,1,.94))
    stamp(fig,"总体积按存活单元面积 × 厚度求和；来源标签只识别宿主单元，碰撞转入量按原函数前后差额单独记录。")
    fig.savefig(OUT/"17_west_arc_c1_history.png",dpi=125)
    plt.close(fig)

    delta=cases["latest"]["E"].astype(float)-cases["previous"]["E"].astype(float)
    area=GRID.cell_area_km2
    land_changed=cases["latest"]["land"]!=cases["previous"]["land"]
    changed=np.abs(delta)>0
    global_change={
        "max_abs_elevation_change_m":float(np.abs(delta).max()),
        "changed_area_km2":float(area[changed].sum()),
        "land_mask_changed_area_km2":float(area[land_changed].sum()),
        "area_abs_change_over_1m_km2":float(area[np.abs(delta)>1].sum()),
        "area_abs_change_over_150m_km2":float(area[np.abs(delta)>150].sum()),
    }
    LO,LA=np.meshgrid(GRID.lon,GRID.lat)
    window=(LO>=EXT[0])&(LO<=EXT[1])&(LA>=EXT[2])&(LA<=EXT[3])
    global_change["outside_west_window_max_abs_m"]=float(np.abs(delta[~window]).max())
    fig,axes=plt.subplots(2,1,figsize=(14,9),gridspec_kw={"height_ratios":[2,1]})
    im=axes[0].imshow(delta,extent=(-180,180,-90,90),origin="upper",cmap="RdBu_r",vmin=-1500,vmax=1500,aspect="equal")
    axes[0].contour(GRID.lon,GRID.lat,cases["latest"]["land"].astype(float),levels=[.5],colors="#333",linewidths=.35)
    axes[0].set(title="全球高程变化：最新配置 − 旧设定（m）",xlabel="经度 °E",ylabel="纬度 °N")
    axes[0].grid(alpha=.2)
    fig.colorbar(im,ax=axes[0],shrink=.75,pad=.02,label="高程变化 m（色带截于 ±1500 m）")
    diffs=[]
    old_m={m["id"]:m for m in cases["previous"]["summary"]["comparison"]["mountains"]}
    for m in cases["latest"]["summary"]["comparison"]["mountains"]:
        before=old_m[m["id"]]["engine_p90_m"]
        d=m["engine_p90_m"]-before
        diffs.append({"id":m["id"],"name":m["name"],"previous_p90_m":before,
                      "latest_p90_m":m["engine_p90_m"],"change_m":d})
    axes[1].bar(range(len(diffs)),[r["change_m"] for r in diffs],color="#597e90")
    axes[1].set_xticks(range(len(diffs)),[r["id"] for r in diffs],rotation=50,fontsize=8)
    axes[1].set(ylabel="高程 p90 变化 m",title="所有既有山系窗口逐项对照")
    axes[1].grid(axis="y",alpha=.3)
    fig.tight_layout(rect=(0,.11,1,.98))
    stamp(fig,"未设置新合格线；变化统计用于定位影响范围。主内海连通性与 A/B 层仍按现有指标计算。")
    fig.savefig(OUT/"18_west_arc_c1_global_change.png",dpi=125)
    plt.close(fig)

    kin=RotationFileKinematics(ROOT/"data/worlds/design_v2/history_v2.rot")
    template=(ids==info["ALB"]["id"])&b["land"]
    tip=template&(LA>=36.4)&(LA<=37.2)
    R300=kin.total_from(info["ALB"]["id"],0.,300.)
    with np.load(CACHE/"latest/initial_source.npz") as z:
        initial_xyz=z["xyz"]
        present_xyz=initial_xyz@R300
        initial_lon,initial_lat=xyz_to_lonlat(present_xyz)
        dist,_=cKDTree(initial_xyz).query(GRID.xyz[tip]@R300.T)
        initial_sampling={
            "initial_source_n":len(initial_xyz),
            "initial_source_area_km2":float(z["area"].sum()),
            "template_area_km2":float(area[template].sum()),
            "tip_template_area_km2":float(area[tip].sum()),
            "tip_nearest_initial_source_distance_km":km_from_chord(dist).tolist(),
            "source_projected_to_present_lon":initial_lon.tolist(),
            "source_projected_to_present_lat":initial_lat.tolist(),
        }
    fig,axes=plt.subplots(1,4,figsize=(21,4.8))
    onset=[]
    for ax,t in zip(axes,(38,37,36,35)):
        with np.load(CACHE/"latest"/f"parcels_{t:03d}.npz") as p:
            mask=(p["origin"]==info["ALB"]["id"])&(p["ctype"]==1)
            from mythic.tectonics.geometry import lonlat_to_xyz
            xyz=lonlat_to_xyz(p["lon"][mask],p["lat"][mask])
            present=xyz@kin.total_from(info["ALB"]["id"],0.,float(t))
            lo,la=xyz_to_lonlat(present)
            centroid=unit((present*p["area"][mask][:,None]).sum(0))
            clo,cla=xyz_to_lonlat(centroid)
            record={"t_Ma":t,"source_n":int(mask.sum()),"present_centroid_lon":float(clo),
                    "present_centroid_lat":float(cla),"present_lon_min":float(lo.min()),
                    "present_lon_max":float(lo.max()),"present_lat_min":float(la.min()),
                    "present_lat_max":float(la.max())}
            onset.append(record)
            actual_owner=np.unique(p["plate"][mask]).tolist()
        map_panel(ax,Efin,b["land"],f"{t} Ma：来源单元 {mask.sum()} 个\n所属：{actual_owner}（302=WT，304=ALB）",
                  extent=(-23,-6,33,43),contour=template)
        ax.scatter(lo,la,s=24,facecolors="none",edgecolors="#d95f02",linewidths=1.2)
    fig.suptitle("定位偏差形成的时段：把实际单元按设计刚体运动换算到 0 Ma 坐标",fontsize=15)
    fig.subplots_adjust(bottom=.23,top=.84,wspace=.15)
    stamp(fig,"紫线＝终图 ALB 范围；橙圈＝实际来源单元。只变换坐标检查位置偏差，不逆推地壳过程。"
              "\n38 Ma 仍在母块 WT，35 Ma 才转属 ALB；具体偏移量见诊断 JSON，未在本检查点修改调度。")
    elevation_legend(fig,0.15)
    fig.savefig(OUT/"19_west_arc_c1_rift_timing.png",dpi=125)
    plt.close(fig)
    from mythic.tectonics.geometry import lonlat_to_xyz
    centers=lonlat_to_xyz([r["present_centroid_lon"] for r in onset],
                         [r["present_centroid_lat"] for r in onset])
    for r,center in zip(onset,centers):
        r["shift_from_38Ma_km"]=float(km_from_chord(np.linalg.norm(center-centers[0])))
    report={"baseline_commit":"5d7a5a0ae15f55f8186443ce4fa9b82690b21da9",
            "neutrality":neutrality,"previous_summary_reproduction":reproduction,
            "global_change":global_change,"mountain_changes":diffs,
            "trajectory":trajectory,"initial_sampling":initial_sampling,"onset_38_to_35Ma":onset,"strait_section":sections,
            "collision_transfers":{k:v["obs"]["collision_transfers"] for k,v in cases.items()},
            "material_note":cases["latest"]["obs"]["material_note"],
            "A_previous":cases["previous"]["summary"]["A_engine_0Ma"],
            "A_latest":cases["latest"]["summary"]["A_engine_0Ma"],
            "method_class":"interpolated",
            "note":"诊断统计不新增或修改任何质量指标阈值；旧设定在内存中仅将 WT—ALB 的 to 改回 6。"}
    write_json(ROOT/"data/worlds/design_v2/u35_c1_diagnostic.json",report)
    write_json(CACHE/"neutrality.json",neutrality)
    images=["15_west_arc_c1_comparison.png","16_west_arc_c1_evolution.png",
            "17_west_arc_c1_history.png","18_west_arc_c1_global_change.png","19_west_arc_c1_rift_timing.png"]
    html="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C1 诊断检查点</title><style>body{margin:24px;background:#eee;color:#222;font-family:system-ui}img{max-width:100%;background:white}section{margin:30px 0}a{color:#176b8a}</style><h1>U35-C1：复现与诊断</h1><p>仅复现和诊断，地貌修正另行决定。图中数值来自实际运行。</p>"
    html+="<p><a href='u35_before_c1/14_west_arc.png'>Claude 留下的原始西口图</a> · <a href='../../../stage1/design_v2_forward_v2_0Ma/index.html'>最新标准图组</a></p>"
    for name in images:
        html+="<section><h2>"+name+"</h2><img src='"+name+"'></section>"
    (OUT/"u35_c1.html").write_text(html+"</html>",encoding="utf-8")
    print(json.dumps({"neutrality":neutrality,"old_reproduction":reproduction,"global_change":global_change},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
