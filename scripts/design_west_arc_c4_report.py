"""U35-C4：只读取真实完整运行，输出隔离的候选对照图与验证表。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as f
from design_final_map import densify
import design_west_arc_report as base
from design_west_arc_c3_report import connectivity
from mythic.tectonics.geometry import xyz_to_lonlat,km_from_chord
OUT=ROOT/"figures/design/v2/forward/u35_c4_strain"
CASES={"C2":ROOT/".cache/u35_c2/latest","C4":ROOT/".cache/u35_c4_strain/latest"}
def stamp(fig,note=""):
    fig.text(.01,.012,"U35-C4 候选 · L7 / 1024×512 · "+f.code_version()+
      "\n地壳 procedural_constrained；高程 parameterized_physics；运动/终图 user_override；栅格 interpolated"+
      ("\n"+note if note else ""),fontsize=8,va="bottom")
def save(fig,name):
    fig.savefig(OUT/name,dpi=125);plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True);base.setup_fonts();g=base.GRID
    d=f.apply_options(f.yaml.safe_load((ROOT/"config/design/final_map_v2.yaml").read_text(encoding="utf-8")),set())
    built=f.build(d,g);Ef=f.design_elevation(d,g,built);cases={}
    for key,path in CASES.items():
        with xr.open_dataset(path/"world.nc") as ds:
            cases[key]={"E":ds.elevation.values.copy(),"land":ds.land.values>.5,"age":ds.tect_ocean_age_myr.values.copy()}
        cases[key]["summary"]=base.read_json(path/"summary.json")
        cases[key]["obs"]=base.read_json(path/"observations.json")
    mountains={key:next(r for r in c["summary"]["comparison"]["mountains"] if r["id"]=="M31") for key,c in cases.items()}
    ext=(-19,-9,34,41)
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    for ax,(E,land,title) in zip(axes,[(Ef,built["land"],"认可终图"),(cases["C2"]["E"],cases["C2"]["land"],"C2：M31 p90 982 m"),(cases["C4"]["E"],cases["C4"]["land"],"C4 候选：M31 p90 "+str(mountains["C4"]["engine_p90_m"])+" m")]):
        base.map_panel(ax,E,land,title,extent=ext,contour=built["land"]);base.feature_lines(ax,d)
        ax.contour(g.lon,g.lat,E,levels=[1000,1500,2000],colors="#683f35",linewidths=.65)
    fig.suptitle("保留北侧走滑边界：原有ALB陆壳内部晚期缩短候选",fontsize=15)
    fig.subplots_adjust(bottom=.22,top=.86,wspace=.19);base.elevation_legend(fig,.13)
    stamp(fig,"紫线为认可海岸；等高线为1000/1500/2000 m。候选没有贴高度、改终图或恢复旧位置；尚待用户看图。")
    save(fig,"32_west_arc_c4_comparison.png")
    colors={"C2":"#8c725d","overlap_rejected":"#999999","C4":"#25859b"}
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for key,c in cases.items():
        rows=[r for r in c["obs"]["rows"] if r["t_Ma"]<=10];t=[r["t_Ma"] for r in rows]
        axes[0,0].plot(t,[r["alb_T_p90_km"] for r in rows],"-o",ms=3,c=colors[key],label=key)
        axes[0,1].plot(t,[r["alb_cont_volume_km3"]/1e6 for r in rows],"-o",ms=3,c=colors[key],label=key)
    axes[0,0].set(title="最后10 Myr：ALB陆壳厚度p90",xlabel="距今 Ma",ylabel="km",xlim=(10,0))
    axes[0,1].set(title="内部缩短的体积来源项单独记账",xlabel="距今 Ma",ylabel="百万 km³",xlim=(10,0))
    mt=next(x for x in d["mountains"] if x["id"]=="M31");pts,_=densify(mt["line"]);lon,lat=xyz_to_lonlat(pts)
    distance=np.r_[0,np.cumsum(km_from_chord(np.linalg.norm(np.diff(pts,axis=0),axis=1)))]
    ii=np.clip(np.rint((90-lat)/180*g.ny-.5).astype(int),0,g.ny-1)
    jj=np.clip(np.rint((lon+180)/360*g.nx-.5).astype(int),0,g.nx-1)
    profile={"distance_km":distance.tolist(),"design":Ef[ii,jj].tolist()}
    axes[1,0].plot(distance,Ef[ii,jj],"--",c="#763666",label="认可终图示意")
    for key,c in cases.items():
        h=c["E"][ii,jj];profile[key]=h.tolist();axes[1,0].plot(distance,h,c=colors[key],label=key)
    axes[1,0].set(title="认可山脊线上的最近输出像元高程",xlabel="沿脊距离 km（西→东）",ylabel="m")
    budgets={key:base.read_json(path/"forward.json")["budget"] for key,path in CASES.items()}
    source=budgets["C4"]["reactivation_added"]-budgets["C2"]["reactivation_added"]
    transfers=cases["C4"]["obs"]["collision_transfers"]
    axes[1,1].bar([str(int(r["t_Ma"]))+" Ma 实际碰撞" for r in transfers]+["规定缩短来源项"],[r["donor_volume_km3"]/1e3 for r in transfers]+[source/1e3],color=["#698cac"]*len(transfers)+["#b57b58"])
    axes[1,1].set(title="实际碰撞与规定缩短的来源项分开记账",ylabel="千 km³")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        if ax!=axes[1,1]:ax.legend(fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94));stamp(fig,"C4沿用U33纯剪切，规定20%局部缩短/1.25增厚比，并非求解应变；只选择原ALB宿主域，曲线不是精细DEM。")
    save(fig,"33_west_arc_c4_height_volume.png")
    delta=cases["C4"]["E"].astype(float)-cases["C2"]["E"].astype(float)
    lo,la=np.meshgrid(g.lon,g.lat);west=(lo>=-26)&(lo<=6)&(la>=30)&(la<=44);area=g.cell_area_km2
    change={"max_abs_m":float(abs(delta).max()),"area_over_150m_km2":float(area[abs(delta)>150].sum()),
      "land_change_km2":float(area[cases["C2"]["land"]!=cases["C4"]["land"]].sum()),
      "outside_west_max_abs_m":float(abs(delta[~west]).max()),"outside_west_land_change_km2":float(area[(~west)&(cases["C2"]["land"]!=cases["C4"]["land"])].sum())}
    fig,axes=plt.subplots(1,2,figsize=(17,5))
    im=axes[0].imshow(delta,extent=(-180,180,-90,90),origin="upper",cmap="RdBu_r",vmin=-1000,vmax=1000,aspect="equal")
    axes[0].set(title="C4 − C2 全球高程差",xlabel="经度 °E",ylabel="纬度 °N");fig.colorbar(im,ax=axes[0],shrink=.65,label="m，截于±1000")
    sx=(g.lon>=ext[0])&(g.lon<=ext[1]);sy=(g.lat>=ext[2])&(g.lat<=ext[3])
    im=axes[1].imshow(delta[np.ix_(sy,sx)],extent=(g.lon[sx][0],g.lon[sx][-1],g.lat[sy][-1],g.lat[sy][0]),origin="upper",cmap="RdBu_r",vmin=-1000,vmax=1000,aspect="equal")
    axes[1].contour(g.lon,g.lat,cases["C4"]["land"],levels=[.5],colors="#333",linewidths=.8);axes[1].set(xlim=ext[:2],ylim=ext[2:],title="西口局部变化",xlabel="经度 °E",ylabel="纬度 °N")
    fig.tight_layout(rect=(0,.14,1,.95));stamp(fig,"西口窗口外最大高程差 "+str(change["outside_west_max_abs_m"])+" m；海陆变化 "+str(change["outside_west_land_change_km2"])+" km²。")
    save(fig,"34_west_arc_c4_global_change.png")
    same=[];diff=[]
    with np.load(CASES["C2"]/"forward.npz") as a,np.load(CASES["C4"]/"forward.npz") as b:
        for k,t in enumerate(a["times"]):
            if t<10:continue
            for field in f.FIELDS:
                n=field+"_"+str(k);(same if np.array_equal(a[n],b[n],equal_nan=True) else diff).append(n)
    if diff:raise RuntimeError("晚期接触前已有变化："+str(diff))
    with np.load(CASES["C2"]/"parcels_000.npz") as a,np.load(CASES["C4"]/"parcels_000.npz") as b:
        sa=(a["origin"]==304)&(a["ctype"]==1);sb=(b["origin"]==304)&(b["ctype"]==1)
        ia=np.flatnonzero(sa)[np.argsort(a["uid"][sa])];ib=np.flatnonzero(sb)[np.argsort(b["uid"][sb])]
        geometry={key:np.array_equal(a[key][ia],b[key][ib]) for key in ("uid","node","plate","lon","lat","area")}
    result={"checkpoint":"U35-C4","status":"candidate_pending_user_review","baseline_commit":"209c5ff",
      "method_class":"procedural_constrained","units":{"height":"m","volume":"km3","area":"km2","time":"Ma"},
      "M31":mountains,"internal_shortening_source_km3":source,"budget_km3":budgets,"pre_event":{"arrays_checked":len(same),"identical":not diff},"source_geometry_identical":geometry,
      "connectivity":{key:connectivity(c) for key,c in cases.items()},"global_change":change,"ridge_profile":profile,
      "observations_last":{key:c["obs"]["rows"][-1] for key,c in cases.items()},
      "collision_transfers":{key:c["obs"]["collision_transfers"] for key,c in cases.items()},
      "A":{key:c["summary"]["A_engine_0Ma"] for key,c in cases.items()},
      "limitations":"内部缩短是规定的历史参数，固定面积模型以体积来源项近似，不是连续应变求解；年轻洋壳敏感性与R17实体表达未解决。",
      "candidate_config":"config/design/history_v2_u35_c4_strain.yaml","default_config_unchanged":True}
    base.write_json(ROOT/"data/worlds/design_v2/u35_c4_diagnostic.json",result)
    html="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C4 西口修正候选</title><style>body{margin:24px;background:#eee;font-family:system-ui}img{max-width:100%}</style><h1>U35-C4：山弧修正候选</h1><p>默认结果仍为C2；候选独立保存，等用户看图。山弧增强，年轻洋壳敏感性仍未解决。</p><p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c4_strain/index.html'>候选标准图组与指标</a> · <a href='../u35_c3.html'>C3诊断</a></p>"
    for name in ["32_west_arc_c4_comparison.png","33_west_arc_c4_height_volume.png","34_west_arc_c4_global_change.png"]:html+="<h2>"+name+"</h2><img src='"+name+"'>"
    (OUT/"index.html").write_text(html,encoding="utf-8",newline="\n")
    print(json.dumps({key:result[key] for key in ("M31","pre_event","source_geometry_identical","connectivity","global_change","observations_last")},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
