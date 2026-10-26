"""U35-C2：裂离修正后的局部、历史与全球对照。只读取已完成运行。"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import design_west_arc_report as base
import design_forward as fwd
from mythic.tectonics.geometry import lonlat_to_xyz, xyz_to_lonlat, unit, km_from_chord
from mythic.tectonics.mesh import load_mesh, Raster
from mythic.tectonics.kinematics import RotationFileKinematics

OUT = ROOT/"figures/design/v2/forward"
CACHES = {"C1":ROOT/".cache/u35_c1/latest", "C2":ROOT/".cache/u35_c2/latest"}
COLORS = {"C1":"#ae5045", "C2":"#237e9a"}
def stamp(fig, note=""):
    fig.text(.01,.015,"U35-C2 · L7 / 1024×512 · 代码 "+fwd.code_version()+
        "\n地壳 procedural_constrained；高程 parameterized_physics；终图/运动 user_override；栅格 interpolated；终图示意高程 visual_only"+
        ("\n"+note if note else ""),fontsize=8,color="#444",va="bottom")
def save(fig, name):
    fig.savefig(OUT/name,dpi=125)
    plt.close(fig)


def neck_sources(cases, raster, grid, column, center):
    """按 Raster 的最近固定节点和各板块参照格网精确追踪断面单元。"""
    cache = CACHES["C2"]
    mesh = raster.mesh if hasattr(raster, "mesh") else load_mesh(7)
    land = cases["C2"]["land"]
    lo = hi = center
    while lo > 0 and not land[lo-1,column]: lo -= 1
    while hi+1 < grid.ny and not land[hi+1,column]: hi += 1
    records = []
    with np.load(cache/"parcels_000.npz") as p:
        rotations = {}
        for i in range(lo,hi+1):
            q = int(cases["C2"]["plate"][i,column])
            if q not in rotations:
                selected = p["plate"] == q
                xyz = lonlat_to_xyz(p["lon"][selected],p["lat"][selected])
                nodes = p["node"][selected]
                u,_,vt = np.linalg.svd(mesh.xyz[nodes].T @ xyz)
                rot = (u @ vt).T
                assert np.max(np.linalg.norm(mesh.xyz[nodes] @ rot.T-xyz,axis=1)) < 1e-9
                rotations[q] = rot
            fixed = int(raster.idx[i,column])
            _,node = mesh.tree.query(mesh.xyz[fixed] @ rotations[q])
            selected = np.flatnonzero((p["plate"]==q)&(p["node"]==node))
            assert selected.size == 1
            k = int(selected[0])
            uid = int(p["uid"][k])
            assert abs(float(p["age"][k])-float(cases["C2"]["ocean_age"][i,column])) < 1e-5
            seen = []
            for t in range(10,-1,-1):
                with np.load(cache/f"parcels_{t:03d}.npz") as z:
                    if np.any(z["uid"]==uid): seen.append(t)
            records.append({"lon":float(grid.lon[column]),"lat":float(grid.lat[i]),
                "fixed_mesh_node":fixed,"plate":q,"source_node":int(node),"uid":uid,
                "assigned_ocean_age_Myr":float(p["age"][k]),"assigned_t_form_Ma":float(p["t_form"][k]),
                "observed_uid_at_times_Ma":seen,"elevation_m":float(cases["C2"]["E"][i,column])})
    return {"method_class":"interpolated","records":records,
            "note":"输出像元并非独立物质单元；UID 首次入表时刻与提交时估算的地质年龄分开记录。该表不证明填隙来自真实张开。"}

def main():
    base.setup_fonts()
    grid = base.GRID
    design = fwd.apply_options(fwd.yaml.safe_load((ROOT/"config/design/final_map_v2.yaml").read_text(encoding="utf-8")),set())
    hcfg = fwd.yaml.safe_load((ROOT/"config/design/history_v2.yaml").read_text(encoding="utf-8"))
    built = fwd.build(design,grid)
    ids, info = fwd.assign_blocks(hcfg,grid,built)
    hist, blocks = base.expected_context(design,hcfg,built,ids,info)
    Efin = fwd.design_elevation(design,grid,built)
    kin = RotationFileKinematics(ROOT/"data/worlds/design_v2/history_v2.rot")
    raster = Raster(load_mesh(7),512)
    cases = {}
    for key,cache in CACHES.items():
        with xr.open_dataset(cache/"world.nc") as ds:
            cases[key] = {"E":ds.elevation.values.copy(),"land":ds.land.values>.5, "ocean_age":ds.tect_ocean_age_myr.values.copy(), "plate":ds.tect_plate.values.copy()}
        cases[key]["summary"] = base.read_json(cache/"summary.json")
        cases[key]["obs"] = base.read_json(cache/"observations.json")
    with np.load(CACHES["C1"]/"parcels_038.npz") as p:
        mask = (p["origin"]==304)&(p["ctype"]==1)
        expected = lonlat_to_xyz(p["lon"][mask],p["lat"][mask])@kin.total_from(304,0.,38.)
        expected_center = unit((expected*p["area"][mask,None]).sum(0))
    trajectories = {}
    for key,cache in CACHES.items():
        rows = []
        for t in range(40,-1,-1):
            with np.load(cache/f"parcels_{t:03d}.npz") as p:
                mask = (p["origin"]==304)&(p["ctype"]==1)
                xyz = lonlat_to_xyz(p["lon"][mask],p["lat"][mask])@kin.total_from(304,0.,float(t))
                center = unit((xyz*p["area"][mask,None]).sum(0))
                rows.append({"t_Ma":t,"n":int(mask.sum()),"centroid_error_km":float(km_from_chord(np.linalg.norm(center-expected_center))),
                             "present_lat_min":float(xyz_to_lonlat(xyz)[1].min())})
        trajectories[key]=rows
    strait=next(q for q in design["straits"] if q["id"]=="T1")
    midpoint=np.mean(strait["line"],axis=0)
    j=int(np.argmin(abs(grid.lon-midpoint[0]))); i=int(np.argmin(abs(grid.lat-midpoint[1])))
    def section(land):
        lo=hi=i
        if land[i,j]:return {"width_km":0.,"center_is_land":True}
        while lo>0 and not land[lo-1,j]:lo-=1
        while hi+1<grid.ny and not land[hi+1,j]:hi+=1
        return {"width_km":float((hi-lo+1)*180/grid.ny*np.pi/180*6371),"center_is_land":False,
                "north_water_cell_lat":float(grid.lat[lo]),"south_water_cell_lat":float(grid.lat[hi])}
    sections={"longitude":float(grid.lon[j]),"center_lat":float(grid.lat[i]),
              "definition":"与 C1 相同经线、包含设计中点的连续水道跨度；不是最窄海峡宽度，也不改变 A 层判定。",
              "design":section(built["land"]),**{key:section(c["land"]) for key,c in cases.items()}}
    fig, axes=plt.subplots(1,3,figsize=(21,5))
    for ax,(key,title,E,land) in zip(axes,[("design","认可终图",Efin,built["land"]),
                ("C1","C1：延迟裂离并重采样",cases["C1"]["E"],cases["C1"]["land"]),
                ("C2","C2：38 Ma 准时、保参照系裂离",cases["C2"]["E"],cases["C2"]["land"])]):
        base.map_panel(ax,E,land,title,contour=built["land"])
        base.feature_lines(ax,design)
        ax.plot([grid.lon[j]]*2,[33,40],":",color="#333",lw=.8)
        ax.text(.02,.98,"同一截面水道跨度 "+format(sections[key]["width_km"],".0f")+" km",
                transform=ax.transAxes,va="top",fontsize=9,bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.suptitle("U35 西口修正：位置归位后，检查山弧、海口和火山岛群",fontsize=16)
    fig.subplots_adjust(bottom=.22,top=.84,wspace=.16)
    base.elevation_legend(fig,.13)
    stamp(fig,"紫线为终图海岸。构造路线、地形公式与指标定义不变；C1 原图保存在 u35_c1/。")
    save(fig,"21_west_arc_c2_comparison.png")
    times=(38,30,10,6,5,0)
    fig,axes=plt.subplots(2,6,figsize=(25,8))
    for row,key in enumerate(("C1","C2")):
        for ax,t in zip(axes[row],times):
            sea=cases[key]["obs"]["snapshot_sea_level_m"][str(t)]
            with np.load(CACHES[key]/f"snapshot_{t:03d}.npz") as p:E=raster(p["elev"]).astype(float)-sea
            own,_=fwd.raster_at(hist,blocks,ids,grid,float(t),grid)
            base.map_panel(ax,E,E>0,key+" / "+str(t)+" Ma",contour=own==304)
            with np.load(CACHES[key]/f"parcels_{t:03d}.npz") as p:
                src=(p["origin"]==304)&(p["ctype"]==1)
                ax.scatter(p["lon"][src],p["lat"][src],s=8,facecolors="none",edgecolors="#ff7f00",linewidths=.7)
    fig.suptitle("同年代演化对照：紫线为设计刚体轮廓；橙圈为真实 ALB 来源单元",fontsize=16)
    fig.subplots_adjust(bottom=.14,top=.91,wspace=.35,hspace=.3)
    base.elevation_legend(fig,.09)
    stamp(fig,"标题均为实际采样年代；背景相对当时海平面。位置点与固定栅格视图分别展示。")
    save(fig,"22_west_arc_c2_evolution.png")
    fig,axes=plt.subplots(2,3,figsize=(20,9))
    for ax,t in zip(axes.flat,times):
        sea=cases["C2"]["obs"]["snapshot_sea_level_m"][str(t)]
        with np.load(CACHES["C2"]/f"snapshot_{t:03d}.npz") as p:E=raster(p["elev"]).astype(float)-sea
        own,_=fwd.raster_at(hist,blocks,ids,grid,float(t),grid)
        base.map_panel(ax,E,E>0,"C2 / "+str(t)+" Ma",contour=own==304)
        with np.load(CACHES["C2"]/f"parcels_{t:03d}.npz") as p:
            src=(p["origin"]==304)&(p["ctype"]==1)
            ax.scatter(p["lon"][src],p["lat"][src],s=14,facecolors="none",edgecolors="#ff7f00",linewidths=.8)
    fig.suptitle("C2 演化放大图：实际采样年代；紫线为设计地体，橙圈为原来源单元",fontsize=16)
    fig.subplots_adjust(bottom=.14,top=.91,wspace=.18,hspace=.3)
    base.elevation_legend(fig,.09)
    stamp(fig,"与图22的C2面板为同一数据；单独放大供审看。背景相对各年代海平面。")
    save(fig,"25_west_arc_c2_evolution_detail.png")

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for key,c in cases.items():
        rows=[r for r in c["obs"]["rows"] if r["t_Ma"]<=10]
        t=[r["t_Ma"] for r in rows]
        for ax,field,scale,title,ylabel in [(axes[0,0],"alb_cont_volume_km3",1e6,"最后10 Myr：ALB陆壳总体积","百万 km³"),
                   (axes[0,1],"alb_T_p90_km",1.,"最后10 Myr：ALB厚度p90","km")]:
            ax.plot(t,[r[field]/scale for r in rows],"-o",ms=3,color=COLORS[key],label=key)
            ax.set(title=title,xlabel="距今 Ma",ylabel=ylabel,xlim=(10,0))
        tr=trajectories[key]
        axes[1,0].plot([r["t_Ma"] for r in tr],[r["centroid_error_km"] for r in tr],color=COLORS[key],label=key)
        axes[1,1].step([r["t_Ma"] for r in tr],[r["n"] for r in tr],color=COLORS[key],label=key,where="post")
    axes[1,0].set(title="统一现今坐标下的位置误差",xlabel="距今 Ma",ylabel="km",xlim=(38,0))
    axes[1,1].set(title="原 ALB 来源单元数",xlabel="距今 Ma",ylabel="个",xlim=(40,0))
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout(rect=(0,.10,1,.95))
    stamp(fig,"误差相对 C1 38 Ma 的同一批真实单元按 ALB 设计运动的位置；38 Ma 前独立运动尚未开始。")
    save(fig,"23_west_arc_c2_history.png")
    delta=cases["C2"]["E"].astype(float)-cases["C1"]["E"].astype(float)
    area=grid.cell_area_km2
    LO,LA=np.meshgrid(grid.lon,grid.lat)
    west=(LO>=base.EXT[0])&(LO<=base.EXT[1])&(LA>=base.EXT[2])&(LA<=base.EXT[3])
    changes={"max_abs_elevation_change_m":float(abs(delta).max()),
             "changed_area_km2":float(area[delta!=0].sum()),
             "area_abs_change_over_150m_km2":float(area[abs(delta)>150].sum()),
             "land_mask_changed_area_km2":float(area[cases["C1"]["land"]!=cases["C2"]["land"]].sum()),
             "outside_west_window_max_abs_m":float(abs(delta[~west]).max()),
             "outside_west_window_area_abs_change_over_1m_km2":float(area[(~west)&(abs(delta)>1)].sum()),
             "outside_west_window_land_mask_changed_area_km2":float(area[(~west)&(cases["C1"]["land"]!=cases["C2"]["land"])].sum())}
    oldm={m["id"]:m for m in cases["C1"]["summary"]["comparison"]["mountains"]}
    mountains=[{"id":m["id"],"name":m["name"],"C1_p90_m":oldm[m["id"]]["engine_p90_m"],
                "C2_p90_m":m["engine_p90_m"],"delta_m":m["engine_p90_m"]-oldm[m["id"]]["engine_p90_m"]}
               for m in cases["C2"]["summary"]["comparison"]["mountains"]]
    fig,axes=plt.subplots(2,1,figsize=(14,9),gridspec_kw={"height_ratios":[2,1]})
    im=axes[0].imshow(delta,extent=(-180,180,-90,90),origin="upper",cmap="RdBu_r",vmin=-1500,vmax=1500,aspect="equal")
    axes[0].contour(grid.lon,grid.lat,cases["C2"]["land"].astype(float),levels=[.5],colors="#333",linewidths=.35)
    axes[0].set(title="全球高程变化：C2 − C1；最大绝对变化 "+format(changes["max_abs_elevation_change_m"],".0f")+" m",xlabel="经度 °E",ylabel="纬度 °N")
    fig.colorbar(im,ax=axes[0],shrink=.75,pad=.02,label="m，色带截于 ±1500")
    axes[1].bar(range(len(mountains)),[m["delta_m"] for m in mountains],color="#597e90")
    axes[1].set_xticks(range(len(mountains)),[m["id"] for m in mountains],rotation=50,fontsize=8)
    axes[1].set(title="既有山系窗口 p90 变化",ylabel="m")
    fig.tight_layout(rect=(0,.11,1,.96))
    stamp(fig,"窗口外最大绝对变化 "+str(changes["outside_west_window_max_abs_m"])+" m；海陆掩膜改变 "+format(changes["land_mask_changed_area_km2"],".0f")+" km²。")
    save(fig,"24_west_arc_c2_global_change.png")
    with np.load(CACHES["C1"]/"forward.npz") as a,np.load(CACHES["C2"]/"forward.npz") as b:
        same, different = [], []
        for k,t in enumerate(a["times"]):
            if t<40:continue
            for field in fwd.FIELDS:
                name=field+"_"+str(k)
                (same if np.array_equal(a[name],b[name],equal_nan=True) else different).append(name)
        pre38={"arrays_checked":len(same)+len(different),"identical":not different,"different_arrays":different}
    from design_strait_pixels import label_wrap
    def cell(lo,la):
        return int(np.argmin(abs(grid.lat-la))),int(np.argmin(abs(grid.lon-lo)))
    sea,ocean=cell(19.,37.),cell(-40.,0.)
    sensitivity={}
    for key,c in cases.items():
        water=~c["land"]
        west_test=(LO>=-30)&(LO<=-5)&(LA>=33)&(LA<=40)
        young=west_test&water&np.isfinite(c["ocean_age"])&(c["ocean_age"]<10)
        masks={"8_connected":(water,True),"4_connected":(water,False),
               "4_connected_remove_young_ocean":(water&~young,False),
               "8_connected_block_west":(water&~west_test,True)}
        sensitivity[key]={}
        for name,(mask,eight) in masks.items():
            lab=label_wrap(mask,eight)
            sensitivity[key][name]=bool(lab[sea]==lab[ocean] and lab[sea]!=0)
    result={"checkpoint":"U35-C2","baseline_commit":"87b0ed4","method_class":"interpolated",
            "units":{"elevation":"m","area":"km2","volume":"km3","time":"Ma"},
            "assumptions":"终图、运动与指标保持既有定义；新增诊断不是质量阈值；参考 docs/tectonics_method.md",
            "pre_event_frames":pre38,"strait_section":sections,"neck_sources":neck_sources(cases,raster,grid,j,i),"connectivity_sensitivity":sensitivity,"global_change":changes,
            "mountain_changes":mountains,"trajectories":trajectories,
            "A":{key:c["summary"]["A_engine_0Ma"] for key,c in cases.items()},
            "last_step":{key:c["obs"]["rows"][-1] for key,c in cases.items()},
            "rift_events":{key:[e for e in base.read_json(CACHES[key]/"forward.json")["events"] if e.get("kind")=="terrane_rift" and e.get("dst")==304] for key in cases}}
    if not pre38["identical"]:raise RuntimeError("38 Ma 前结果已改变，需调查后再交付")
    base.write_json(ROOT/"data/worlds/design_v2/u35_c2_diagnostic.json",result)
    base.write_json(ROOT/"data/worlds/design_v2/u35_c2_observations.json",cases["C2"]["obs"])
    names=["20_west_arc_c2_rift_spike.png","21_west_arc_c2_comparison.png","22_west_arc_c2_evolution.png","23_west_arc_c2_history.png","24_west_arc_c2_global_change.png","25_west_arc_c2_evolution_detail.png"]
    html="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C2 西口收尾</title><style>body{margin:24px;background:#eee;color:#222;font-family:system-ui}img{max-width:100%;background:white}section{margin:30px 0}a{color:#176b8a}</style><h1>U35-C2：西口裂离修正</h1><p>保留终图和运动，修正 ALB 分离时序并消除转属重采样。结果待用户看图，非阶段验收。</p><p><a href='u35_c1.html'>C1诊断</a> · <a href='../../../stage1/design_v2_forward_v2_0Ma/index.html'>最新标准图组</a></p>"
    for name in names:html+="<section><h2>"+name+"</h2><img src='"+name+"'></section>"
    (OUT/"u35_c2.html").write_text(html+"</html>",encoding="utf-8",newline="\n")
    print(json.dumps({"pre_event_frames":pre38,"strait_section":sections,"neck_sources":neck_sources(cases,raster,grid,j,i),"connectivity_sensitivity":sensitivity,"global_change":changes,"last_step":result["last_step"]},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
