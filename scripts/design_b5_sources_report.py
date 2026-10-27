"""U35-C11：出生新增、首次碰撞与混合宿主通量的只读报告。"""
from pathlib import Path
import json
import pickle
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as fwd
import design_final_map as final
import design_arc_sections as sec
import design_west_arc_report as base
from design_native_output import dynamic_height
from design_west_arc_c5_report import hashes
from mythic.figures.style import setup_fonts,elevation_cmap_norm
from mythic.figures.netcdf import open_netcdf
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.sampling import prepare_output,sample_output
from mythic.tectonics.sections import spherical_sections
OUT=ROOT/"figures/design/v2/forward/u35_c11"
CACHE=ROOT/".cache/u35_c11"
NAMES={"baseline":"限额：无B5减薄","progressive":"限额：持续减薄β1.5"}

def flow_groups(rows,births,target=52985):
    positive={r["uid"] for r in births if r["created_km3"]>1e-7}
    groups={k:{"incoming_km3":0.,"outgoing_km3":0.} for k in ("created_birth_host","other")}
    for r in rows:
        if r["target_uid"]!=target:continue
        key="created_birth_host" if r["neighbor_uid"] in positive else "other"
        v=r["volume_into_target_km3"]
        groups[key]["incoming_km3" if v>0 else "outgoing_km3"]+=abs(v)
    for g in groups.values():g["net_km3"]=g["incoming_km3"]-g["outgoing_km3"]
    return groups

def save(fig,name,note):
    fig.text(.012,.012,"U35-C11 · "+fwd.code_version()+"\n历史/伸展 user_override；地壳 procedural_constrained；高程 parameterized_physics；查询 interpolated；L7约60km\n"+note,fontsize=8,va="bottom")
    fig.savefig(OUT/name,dpi=130);plt.close(fig)

def neutrality(a,b):
    r=base.compare_npz(a/"forward.npz",b/"forward.npz")
    x,y=base.read_json(a/"forward.json"),base.read_json(b/"forward.json")
    r.update(budget_identical=x["budget"]==y["budget"],events_identical=x["events"]==y["events"])
    with (a/"final_state.pkl").open("rb") as f:ae=pickle.load(f)["engine"]
    with (b/"final_state.pkl").open("rb") as f:be=pickle.load(f)["engine"]
    r["physical_state_hash_identical"]=sec.state_hash(ae)==sec.state_hash(be)
    with open_netcdf(a/"world.nc") as x,open_netcdf(b/"world.nc") as y:
        r["world_arrays_identical"]=set(x.data_vars)==set(y.data_vars) and all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.data_vars)
    if not all(r[k] for k in ("identical","budget_identical","events_identical","physical_state_hash_identical","world_arrays_identical")):raise RuntimeError("来源观测改变了物理")
    return r

def map_scale(ax, km):
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    x=x0+.06*(x1-x0);y=y0+.08*(y1-y0)
    ax.plot([x,x],[y,y+km/111.195],c="#222",lw=2)
    ax.text(x+.03*(x1-x0),y+km/222.39,f"南北{km:g}km",fontsize=8,va="center",
        bbox={"facecolor":"white","edgecolor":"none","alpha":.75})

def main():
    setup_fonts();OUT.mkdir(parents=True,exist_ok=True)
    c7=base.read_json(ROOT/"data/worlds/design_v2/u35_c7_diagnostic.json")
    c10=base.read_json(ROOT/"data/worlds/design_v2/u35_c10_diagnostic.json")
    b=c7["B5"]["design"]
    polygon=final.meander_polygon(b["poly"],final.feature_rng("B5"),b["meander_km"])
    poly=MPath(polygon);p=np.vstack([polygon,polygon[0]])
    inside=lambda xy:bool(poly.contains_point(xy))
    window=lambda xy:16.5<=xy[0]<=21 and -6<=xy[1]<=1
    out={"checkpoint":"U35-C11","base_commit":"3037b3f","physics_changed":False,"method_class":{"history":"user_override","audit":"procedural_constrained","height":"parameterized_physics","query":"interpolated"},
        "neutrality":neutrality(CACHE/"progressive",ROOT/".cache/u35_c11_control/progressive"),"protected_sha256":hashes(),"cases":{},
        "limitations":["出生新增量不等于现存宿主库存","后续通量为整根混合宿主体积，不是新增成分守恒示踪","位置使用洲C现今参照系，非古绝对位置","仅审计rift_fill与hole_created两类出生，未追踪弧岩浆、LIP等其他增厚源；没有修复谷地"]}
    data={};maps={};profiles={}
    lon,lat,xyz,area=sec.local_map((16.5,21,-6,1))
    xx,yy=np.meshgrid(lon,lat);mask=poly.contains_points(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    along=np.arange(-60,532,5);axis,length=spherical_sections([[18.65,-.4],[18.7,-4.6]],along,[0.])
    for case in NAMES:
        path=CACHE/case;s=base.read_json(path/"sources.json");data[case]=s
        births=s["births"];positive={r["uid"] for r in births if r["created_km3"]>1e-7}
        coupling=base.read_json(path/"coupling.json")
        with (path/"final_state.pkl").open("rb") as f:payload=pickle.load(f)
        ctx=prepare_output(payload["engine"],payload["fixed_frame"])
        def query(points):
            shape=points.shape[:-1];flat=points.reshape(-1,3)
            q=sample_output(ctx,flat,dynamic_height(payload,flat),carry_view_fill=False)
            q["elev"][q["owner"]<0]=np.nan
            return {k:v.reshape(shape) for k,v in q.items()}
        q=query(xyz);maps[case]=q;profiles[case]=query(axis)["elev"].ravel()
        valid=mask&(q["owner"]>=0);sel=np.isin(q["uid"],list(positive))
        center=query(lonlat_to_xyz(*c7["B5"]["center_lon_lat"])[None,:])
        meta=base.read_json(path/"forward.json");old=c10["cases"]["new_base" if case=="baseline" else "new_prog"]
        zones={}
        for name,fn in (("design_polygon",inside),("B5_window",window),("outside_window",lambda xy:not window(xy))):
            born=[r for r in births if fn(r["C_frame_lon_lat"])];uids={r["uid"] for r in born}
            collisions=[r for r in s["collision_transfers"] if r["source_uid"] in uids]
            zones[name]={"birth_count":len(born),"created_km3":sum(r["created_km3"] for r in born),"borrowed_km3":sum(r["borrowed_km3"] for r in born),
                "created_by_kind_km3":{kind:sum(r["created_km3"] for r in born if r["kind"]==kind) for kind in ("hole_fill","rift_fill")},
                "surviving_host_count":sum(r["final_host"]["alive"] for r in born),"removed_host_count":sum(not r["final_host"]["alive"] for r in born),
                "whole_host_collision_km3":sum(r["source_whole_host_volume_km3"] for r in collisions),
                "whole_host_collision_into_polygon_km3":sum(t["volume_km3"] for r in collisions for t in r["targets"] if inside(t["C_frame_lon_lat"]))}
        row={"birth_count":len(births),"created_km3":sum(r["created_km3"] for r in births),
            "created_by_kind_km3":{kind:sum(r["created_km3"] for r in births if r["kind"]==kind) for kind in ("hole_fill","rift_fill")},
            "birth_budget_residuals_km3":s["birth_budget_residuals_km3"],"zones":zones,
            "created_budget_residual_km3":s["created_budget_residual_km3"],"collision_distribution_max_residual_km3":s["collision_distribution_max_residual_km3"],
            "C_frame_final_matrix_max_error":s["C_frame_final_matrix_max_error"],"center_flow_whole_host_since36Ma":flow_groups(coupling["flow_edges_since_36Ma"],births),
            "design_polygon":{"area_km2":float(area[mask].sum()),"coverage":float(area[valid].sum()/area[mask].sum()),"height_m":sec.stats(q["elev"][valid]),
                "below_sea_fraction":float(area[valid&(q["elev"]<0)].sum()/area[valid].sum()),"current_positive_birth_host_area_fraction":float(area[valid&sel].sum()/area[valid].sum())},
            "center_height_m":float(center["elev"][0]),"global_budget_km3":meta["budget"],"A":base.read_json(path/"summary.json")["A_engine_0Ma"],
            "C10_budget_max_difference_km3":max(abs(meta["budget"].get(k,0)-old["global_budget_km3"].get(k,0)) for k in set(meta["budget"])|set(old["global_budget_km3"])),
            "C10_center_difference_m":float(center["elev"][0])-old["center_current_host"]["height_m"],
            "birth_time_groups":[{"t_Ma":t,"created_km3":sum(r["created_km3"] for r in births if r["t_Ma"]==t),"B5_window_created_km3":sum(r["created_km3"] for r in births if r["t_Ma"]==t and window(r["C_frame_lon_lat"]))} for t in sorted({r["t_Ma"] for r in births}|set(range(0,301,5)),reverse=True)],
            "near_B5_births":[r for r in births if window(r["C_frame_lon_lat"])]}
        row["long_axis_height_m"]=profiles[case]
        row["long_axis_inside_height_m"]=sec.stats(profiles[case][(along>=0)&(along<=length)])
        row["b_verdict"]=base.read_json(ROOT/("figures/stage1/design_v2_forward_v2_0Ma_u35_c11_"+case)/"metrics.json")["b_verdict"]
        out["cases"][case]=row
    out["long_axis_along_km"]=along
    out["long_axis_length_km"]=length
    out["protected_unchanged"]=hashes()==out["protected_sha256"]==c10["protected_sha256"]
    if not out["protected_unchanged"]:raise RuntimeError("保护文件改变")
    base.write_json(ROOT/"data/worlds/design_v2/u35_c11_diagnostic.json",sec.clean(out))
    fig,axes=plt.subplots(1,2,figsize=(15,8))
    for ax,zone in zip(axes,(None,"B5_window")):
        x=np.arange(2)
        rows=[out["cases"][case] if zone is None else out["cases"][case]["zones"][zone] for case in NAMES]
        hole=np.array([r["created_by_kind_km3"]["hole_fill"] for r in rows])/1e6
        rift=np.array([r["created_by_kind_km3"]["rift_fill"] for r in rows])/1e6
        ax.bar(x,hole,label="hole_created补洞缺额",color="#8b4e8c")
        ax.bar(x,rift,bottom=hole,label="rift_fill整柱新增",color="#4e87a8")
        ax.set(xticks=x,xticklabels=list(NAMES.values()),ylabel="出生新增 百万km³",title="全球两类出生" if zone is None else "B5窗内两类出生")
        ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,.16,1,.94))
    save(fig,"70_two_birth_sources.png","rift_fill整柱新增没有从供体扣料，hole_created仅补洞的借料缺额；两者都不是已求解的材料来源。B5窗=16.5–21°E、6°S–1°N；其余岩浆/厚度来源不在本次出生审计范围。")
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    for row,case in zip(axes,NAMES):
        births=data[case]["births"];r=[b for b in births if b["created_km3"]>1e-7]
        for ax,regional in zip(row,(False,True)):
            sc=ax.scatter([b["C_frame_lon_lat"][0] for b in r],[b["C_frame_lon_lat"][1] for b in r],s=[12+b["created_km3"]/1200 for b in r],c=[b["t_Ma"] for b in r],cmap="viridis",vmin=0,vmax=300,edgecolors="#333",linewidths=.4)
            ax.plot(p[:,0],p[:,1],c="#b74480",lw=1.4)
            ax.set(xlim=(15,23) if regional else (-180,180),ylim=(-10,5) if regional else (-90,90),title=NAMES[case]+("：B5附近" if regional else "：全球显式新增出生"),xlabel="洲C现今参照经度 °E",ylabel="参照纬度 °N");ax.grid(alpha=.2)
            map_scale(ax,100 if regional else 1000)
    fig.subplots_adjust(bottom=.16,top=.94,hspace=.25,wspace=.3,right=.88)
    cax=fig.add_axes([.91,.24,.016,.60]);fig.colorbar(sc,cax=cax,label="出生年代 Ma")
    save(fig,"66_hole_birth_locations.png","仅标新增>1e-7km³的洞；位置随洲C回算到现今参照系，非古经纬度。圆大小是出生新增量，不能理解为存留量。")
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    for case,color in (("baseline","#4e87a8"),("progressive","#8b4e8c")):
        rows=out["cases"][case]["birth_time_groups"]
        for ax,key in zip(axes,("created_km3","B5_window_created_km3")):ax.plot([r["t_Ma"] for r in rows],[r[key]/1e3 for r in rows],"-o",c=color,label=NAMES[case])
    for ax,title,limit in zip(axes,("全球裂谷填充+补洞缺额新增","B5窗口16.5–21°E、6°S–1°N内出生"),(300,40)):
        ax.set(title=title,xlim=(limit,0),xlabel="Ma",ylabel="该次显式新增 千km³");ax.legend();ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,.16,1,.94))
    save(fig,"67_hole_birth_time_and_budget.png","洞只计一次，不按供体行重复计数；hole_created与rift_fill逐项核对实际预算，新增与借入分列于JSON。")
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    for row,case in zip(axes,NAMES):
        births=data[case]["births"];near={r["uid"]:r for r in births if window(r["C_frame_lon_lat"])}
        row[0].plot(p[:,0],p[:,1],c="#b74480",lw=1.5)
        for r in data[case]["collision_transfers"]:
            if r["source_uid"] not in near:continue
            xy=near[r["source_uid"]]["C_frame_lon_lat"]
            for t in r["targets"]:
                if t["volume_km3"]>100:row[0].plot([xy[0],t["C_frame_lon_lat"][0]],[xy[1],t["C_frame_lon_lat"][1]],c="#567b95",alpha=.4,lw=.5+t["volume_km3"]/50000)
        for b in near.values():
            x,y=b["C_frame_lon_lat"];alive=b["final_host"]["alive"]
            row[0].scatter(x,y,s=40,c="#8b4e8c" if alive else "#c08f45",marker="o" if alive else "x")
        row[0].set(xlim=(15,24),ylim=(-10,5),title=NAMES[case]+"：出生宿主首次碰撞去向",xlabel="洲C现今参照经度 °E",ylabel="参照纬度 °N")
        groups=out["cases"][case]["center_flow_whole_host_since36Ma"];x=np.arange(2)
        row[1].bar(x-.18,[groups[k]["incoming_km3"]/1e3 for k in groups],width=.36,label="流入")
        row[1].bar(x+.18,[-groups[k]["outgoing_km3"]/1e3 for k in groups],width=.36,label="流出（负）")
        row[1].set(xticks=x,xticklabels=["带新增出生宿主","其余邻居"],title="UID52985的36→0Ma实际侧向通量",ylabel="累计整根宿主通量 千km³");row[1].legend()
        map_scale(row[0],100)
        for ax in row:ax.grid(alpha=.2)
    fig.tight_layout(rect=(0,.17,1,.94))
    save(fig,"68_birth_host_fate_and_flow.png","紫圆=存活宿主，叉=已移除。连线与柱图均为整根混合宿主体积，不是显式新增成分的守恒路线；不能把它们加到出生新增量上。")
    fig,axes=plt.subplots(1,3,figsize=(18,8))
    cmap,norm=elevation_cmap_norm();cmap=cmap.copy();cmap.set_bad("#888")
    for ax,case in zip(axes[:2],NAMES):
        q=maps[case];im=ax.pcolormesh(lon,lat,q["elev"],cmap=cmap,norm=norm,shading="nearest");ax.plot(p[:,0],p[:,1],c="#b74480")
        uids=[r["uid"] for r in data[case]["births"] if r["created_km3"]>1e-7];sel=np.isin(q["uid"],uids)
        if sel.any():ax.contour(lon,lat,sel,levels=[.5],colors="#fff",linewidths=1.)
        ax.set(title=NAMES[case]+"\n白线=曾带新增的现存宿主",xlabel="经度 °E",ylabel="纬度 °N",aspect=1.)
        map_scale(ax,100)
    for case,color in (("baseline","#4e87a8"),("progressive","#8b4e8c")):axes[2].plot(along,profiles[case],c=color,label=NAMES[case])
    axes[2].axvspan(0,length,color="#deb3ca",alpha=.2);axes[2].axhline(0,c="#333",lw=.7);axes[2].axhline(-400,c="#b74480",ls="--",label="谷底目标")
    axes[2].set(title="设计轴与两端各60km",xlabel="从轴北端向南 km",ylabel="基底高程 m");axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.tight_layout(rect=(0,.24,1,.94))
    cax=fig.add_axes([.09,.16,.47,.022]);fig.colorbar(im,cax=cax,orientation="horizontal",label="原生基底高程 m；灰色无覆盖")
    save(fig,"69_B5_current_hosts_and_axis.png","原生基底高程；灰色无覆盖，白线不表示成分比例。密集查询不提高L7物理分辨率；−400m仅谷底目标，连续谷地仍未实现。")
    html="<!doctype html><meta charset='utf-8'><title>U35-C11补洞来源</title><style>body{font-family:system-ui;max-width:1600px;margin:24px auto;background:#eee}img{max-width:100%}</style><h1>U35-C11：补洞出生与宿主去向</h1><p>只读复跑C10，核对rift_fill整柱新增与hole_created补洞缺额，没有地形修正。出生新增量、后续混合宿主体积、成分存留必须区分；未求解成分守恒示踪。</p>"
    for png in sorted(OUT.glob("*.png")):html+=f"<h2>{png.name}</h2><img src='{png.name}'>"
    for case in NAMES:html+=f"<p><a href='../../../../stage1/design_v2_forward_v2_0Ma_u35_c11_{case}/index.html'>{case}标准图与指标</a></p>"
    (OUT/"index.html").write_text(html,encoding="utf-8")
    print(json.dumps(sec.clean(out),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
