"""U35-C3：实际填隙证据、M31 过程账和完整候选对照。只读取已经完成的运行。"""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as f
import design_west_arc_report as base
from design_strait_pixels import label_wrap
from mythic.tectonics.geometry import lonlat_to_xyz,xyz_to_lonlat,km_from_chord
from mythic.tectonics.kinematics import RotationFileKinematics
from mythic.tectonics.mesh import Raster,load_mesh
OUT=ROOT/"figures/design/v2/forward"
CACHE=ROOT/".cache/u35_c3"
CACHES={"C2":ROOT/".cache/u35_c2/latest","C3":CACHE/"latest"}
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def stamp(fig,note=""):
    fig.text(.01,.012,"U35-C3 · L7 / 1024×512 · 代码 "+f.code_version()+
      "\n地壳 procedural_constrained；高程 parameterized_physics；终图/运动 user_override；栅格 interpolated"+
      ("\n"+note if note else ""),fontsize=8,va="bottom",color="#444")
def save(fig,name):
    fig.savefig(OUT/name,dpi=125)
    plt.close(fig)
def connectivity(c):
    g=base.GRID;lo,la=np.meshgrid(g.lon,g.lat)
    def cell(x,y):return np.argmin(abs(g.lat-y)),np.argmin(abs(g.lon-x))
    sea,oc=cell(19,37),cell(-40,0)
    west=(lo>=-30)&(lo<=-5)&(la>=33)&(la<=40)
    w=~c["land"];young=west&w&np.isfinite(c["age"])&(c["age"]<10)
    ans={}
    for name,mask,eight in [("4_connected",w,False),("8_connected",w,True),
               ("4_connected_remove_young_ocean",w&~young,False),("8_connected_block_west",w&~west,True)]:
        lab=label_wrap(mask,eight);ans[name]=bool(lab[sea]==lab[oc] and lab[sea]!=0)
    return ans
def main():
    base.setup_fonts();g=base.GRID;mesh=load_mesh(7);raster=Raster(mesh,512)
    diag=read(CACHE/"observed/process_diagnostic.json")
    spike=read(CACHE/"commit_spike.json")
    config=f.yaml.safe_load((ROOT/"config/design/history_v2.yaml").read_text(encoding="utf-8"))
    design=f.apply_options(f.yaml.safe_load((ROOT/"config/design/final_map_v2.yaml").read_text(encoding="utf-8")),set())
    built=f.build(design,g);Ef=f.design_elevation(design,g,built)
    cases={}
    for key,p in CACHES.items():
        with xr.open_dataset(p/"world.nc") as ds:
            cases[key]={"E":ds.elevation.values.copy(),"land":ds.land.values>.5,"age":ds.tect_ocean_age_myr.values.copy(),
                        "C":ds.tect_crust_type.values.copy(),"T":ds.tect_crust_km.values.copy()}
        cases[key]["summary"]=read(p/"summary.json");cases[key]["obs"]=read(p/"observations.json")
    neutrality=base.compare_npz(CACHES["C2"]/"forward.npz",CACHE/"observed/forward.npz")
    a=read(CACHES["C2"]/"forward.json");b=read(CACHE/"observed/forward.json")
    neutrality.update(events_identical=a["events"]==b["events"],budget_identical=a["budget"]==b["budget"])
    assert all(neutrality[k] for k in ("identical","events_identical","budget_identical"))
    crit=[x for x in diag["gaps"] if x["t_Ma"]==0 and x["fixed_node"] in (68047,100961)]
    # 实际两次碰撞的分配核：用保存的同年代存活目标位置重算，不改变状态。
    kin=RotationFileKinematics(ROOT/"data/worlds/design_v2/history_v2.rot")
    mt=next(x for x in design["mountains"] if x["id"]=="M31")
    alloc=[]
    for row in diag["contacts"]:
        if not row["consumed"] or row["dst"]!=304:continue
        t=int(row["t_Ma"])
        with np.load(CACHES["C2"]/f"parcels_{t:03d}.npz") as p:
            sel=(p["plate"]==304)&(p["ctype"]==1)
            xyz=lonlat_to_xyz(p["lon"][sel],p["lat"][sel]);area=p["area"][sel]
            d=km_from_chord(np.linalg.norm(xyz-lonlat_to_xyz(row["lon"],row["lat"]),axis=1))
            w=np.where(d<400,(1-(d/400)**2)**2,0.)
            dv=row["volume_km3"]*w*area/(w*area).sum()
            now=xyz@kin.total_from(304,0.,float(t))
            mlon,mlat=xyz_to_lonlat(now)
            dst=f.line_distance_km(SimpleNamespace(shape=(len(now),),xyz=now),mt["line"],np.ones(len(now),bool),float(mt["width_km"])/2)
            inside=np.isfinite(dst)
            record={"t_Ma":t,"donor_uid":row["uid"],"donor_volume_km3":row["volume_km3"],
                "distributed_km3":float(dv.sum()),"volume_inside_M31_node_centers_km3":float(dv[inside].sum()),"targets":[]}
            for k in np.flatnonzero(dv>0):
                record["targets"].append({"uid":int(p["uid"][sel][k]),"lon_present":float(mlon[k]),"lat_present":float(mlat[k]),
                     "added_volume_km3":float(dv[k]),"added_T_km":float(dv[k]/area[k]),"inside_M31":bool(inside[k])})
            alloc.append(record)
    ledger=diag["ledger"]
    totals={pr:sum(x["delta"]["own_volume_km3"] for x in ledger if x["process"]==pr and x["t_Ma"]<10)
            for pr in ("_interact","_arc","_surface")}
    observed=cases["C2"]["obs"]["rows"];v10=next(x["alb_cont_volume_km3"] for x in observed if x["t_Ma"]==10)
    v0=observed[-1]["alb_cont_volume_km3"]
    totals["commits_and_other_km3"]=v0-v10-sum(totals.values())
    totals["net_change_km3"]=v0-v10
    # 26：真实 0 Ma 提交前局部几何及隔离提交验证。
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    ext=(-20,-9,33,40)
    base.map_panel(axes[0,0],cases["C2"]["E"],cases["C2"]["land"],"完整 C2 结果：关键单元是年轻洋壳",extent=ext)
    with np.load(CACHE/"commit_spike_continental_margin.npz") as p:
        base.map_panel(axes[0,1],p["E"],p["E"]>0,"同一个提交前状态：沿用既有陆缘规则",extent=ext)
    with (CACHE/"observed/engine_precommit_000.pkl").open("rb") as fh:e=pickle.load(fh)
    pos=e.positions();lon,lat=xyz_to_lonlat(pos);c=e.c
    for q,col in [(301,"#c37b4d"),(304,"#627cac")]:
        s=e.alive&(c.plate==q)&(lon>ext[0])&(lon<ext[1])&(lat>ext[2])&(lat<ext[3])
        thickness_image=axes[1,0].scatter(lon[s],lat[s],c=np.where(c.ctype[s]==1,c.T[s],np.nan),cmap="viridis",vmin=28,vmax=48,s=22,marker="o" if q==301 else "^",label={301:"C 陆壳（圆）",304:"ALB 陆壳（三角）"}[q])
    for n,row in enumerate(crit,1):
        for ax in axes[0]:
            ax.scatter(row["lon"],row["lat"],s=45,facecolors="none",edgecolors="#d51d52",linewidths=1.5)
            ax.annotate(str(n),(row["lon"],row["lat"]),xytext=(8,5),textcoords="offset points",color="#d51d52")
        inds=[int(np.flatnonzero(c.uid==u)[0]) for u in row["source_uids"]]
        axes[1,0].plot(lon[inds],lat[inds],"--",c="#d51d52",lw=1)
        axes[1,0].plot(row["lon"],row["lat"],"x",c="#d51d52")
        axes[1,0].annotate(str(n)+": "+format(row["separation_cm_yr"],"+.2f")+" cm/yr",(row["lon"],row["lat"]),xytext=(15,-12) if n==1 else (15,18),textcoords="offset points",fontsize=9)
    axes[1,0].set(xlim=ext[:2],ylim=ext[2:],aspect="equal",title="提交前真实陆壳单元与两侧来源连线",xlabel="经度 °E",ylabel="纬度 °N");axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.3)
    fig.colorbar(thickness_image,ax=axes[1,0],shrink=.6,pad=.02,label="陆壳厚度 km")
    names=["100961","68047"];x=np.arange(2)
    for k,(key,col) in enumerate([("original","#9a6c9a"),("continental_margin","#398b9d")]):
        row=next(r for r in spike if r["case"]==key)
        bars=axes[1,1].bar(x+(k-.5)*.35,[-z["elevation_m"] for z in row["critical"]],.35,label={"original":"原始填隙","continental_margin":"同状态陆缘试验"}[key],color=col)
        axes[1,1].bar_label(bars,fmt="%.0f m",padding=3,fontsize=9)
    axes[1,1].set(xticks=x,xticklabels=names,ylabel="水深 m",title="隔离提交验证：两种配置均保持水下");axes[1,1].legend(fontsize=9);axes[1,1].grid(axis="y",alpha=.25)
    fig.suptitle("关键节点：两侧都是陆壳；张开与非张开小洞应分别解释",fontsize=16)
    fig.tight_layout(rect=(0,.13,1,.95));stamp(fig,"正号为按原最近单元连线判据张开，负号为汇聚；它不是平滑地质法向。验证只改变一次提交的配置。")
    save(fig,"26_west_arc_c3_gap_evidence.png")
    # 27：过程账和沿弧分配，额外重算核只诊断已经发生的实际碰撞。
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    times=list(range(9,-1,-1))
    for pr,col,label in [("_interact","#a66550","实际碰撞输入"),("_arc","#3996a2","弧增生"),("_surface","#666666","表面过程")]:
        val=[next(x["delta"]["own_volume_km3"] for x in ledger if x["process"]==pr and x["t_Ma"]==t)/1e3 for t in times]
        axes[0,0].plot(times,np.cumsum(val),"-o",ms=4,color=col,label=label)
    axes[0,0].set(xlim=(10,0),xlabel="距今 Ma，10 Ma 状态之后",ylabel="累计千 km³",title="C2：最后 10 Myr 的实际过程账");axes[0,0].legend();axes[0,0].grid(alpha=.25)
    for key,col in [("C2","#ab775e"),("C3","#337f9d")]:
        rows=[x for x in cases[key]["obs"]["rows"] if x["t_Ma"]<=10]
        axes[0,1].plot([x["t_Ma"] for x in rows],[x["alb_cont_volume_km3"]/1e6 for x in rows],"-o",ms=3,c=col,label=key+(" 候选" if key=="C3" else "")+" ALB 总体积")
    axes[0,1].set(xlim=(10,0),xlabel="距今 Ma",ylabel="百万 km³",title="完整候选：加入陆缘规则后的体积");axes[0,1].legend();axes[0,1].grid(alpha=.25)
    ax=axes[1,0];base.map_panel(ax,cases["C2"]["E"],cases["C2"]["land"],"两次真实碰撞分配核（位置统一到现今）")
    tv={}
    for event in alloc:
        for r in event["targets"]:
            z=tv.setdefault(r["uid"],dict(r));z["total_T_km"]=z.get("total_T_km",0)+r["added_T_km"]
    pts=list(tv.values());im=ax.scatter([r["lon_present"] for r in pts],[r["lat_present"] for r in pts],c=[r["total_T_km"] for r in pts],s=65,cmap="magma",edgecolors="white",lw=.4)
    base.feature_lines(ax,design);fig.colorbar(im,ax=ax,shrink=.7,label="两次碰撞累计加厚 km")
    labels=[str(x["t_Ma"])+" Ma" for x in alloc]
    inside=np.array([x["volume_inside_M31_node_centers_km3"]/1e3 for x in alloc]);total=np.array([x["donor_volume_km3"]/1e3 for x in alloc])
    axes[1,1].bar(labels,inside,label="M31 山脊带内节点",color="#ac7253");axes[1,1].bar(labels,total-inside,bottom=inside,label="山脊带外节点",color="#7998b0")
    axes[1,1].set(ylabel="千 km³",title="400 km 分配核的体积去向");axes[1,1].legend();axes[1,1].grid(axis="y",alpha=.25)
    fig.tight_layout(rect=(0,.13,1,.94));stamp(fig,"M31 内外按同一 width_km/2 山脊距离、目标单元中心统计，是分配诊断，不是新增验收指标；origin 不等于混合岩石成分。")
    save(fig,"27_west_arc_c3_volume_allocation.png")

    # 28：终图、修正前和完整候选；统一色带、范围和比例。
    strait=next(x for x in design["straits"] if x["id"]=="T1");mid=np.mean(strait["line"],axis=0)
    j=int(np.argmin(abs(g.lon-mid[0])));i=int(np.argmin(abs(g.lat-mid[1])))
    def section(land):
        lo=hi=i
        if land[i,j]:return 0.
        while lo>0 and not land[lo-1,j]:lo-=1
        while hi+1<g.ny and not land[hi+1,j]:hi+=1
        return float((hi-lo+1)*np.pi*6371/g.ny)
    sections={"longitude":float(g.lon[j]),"design_km":section(built["land"])}
    sections.update({key:section(c["land"]) for key,c in cases.items()})
    fig,axes=plt.subplots(1,3,figsize=(21,5))
    for ax,(key,E,land,title) in zip(axes,[("design",Ef,built["land"],"认可终图"),("C2",cases["C2"]["E"],cases["C2"]["land"],"C2：位置修正后"),("C3",cases["C3"]["E"],cases["C3"]["land"],"C3 已否决候选：南侧陆缘规则")]):
        base.map_panel(ax,E,land,title,contour=built["land"]);base.feature_lines(ax,design)
        ax.plot([g.lon[j]]*2,[33,40],":",c="#444",lw=.8)
    fig.suptitle("西口完整演化对照：候选造成封口，已撤回",fontsize=16)
    fig.subplots_adjust(bottom=.22,top=.84,wspace=.16);base.elevation_legend(fig,.13)
    stamp(fig,"紫线为认可终图海岸。候选 4/8 连通均断开，A 层仍全过；不采用。正式结果恢复 C2。")
    save(fig,"28_west_arc_c3_comparison.png")
    # 29：原协议的关键年代局部图。
    ids,info=f.assign_blocks(config,g,built);hist,blocks=base.expected_context(design,config,built,ids,info)
    fig,axes=plt.subplots(2,3,figsize=(20,9))
    for ax,t in zip(axes.flat,(38,30,10,6,5,0)):
        sea=cases["C3"]["obs"]["snapshot_sea_level_m"][str(t)]
        with np.load(CACHES["C3"]/f"snapshot_{t:03d}.npz") as p:E=raster(p["elev"]).astype(float)-sea
        own,_=f.raster_at(hist,blocks,ids,g,float(t),g)
        base.map_panel(ax,E,E>0,str(t)+" Ma（实际采样）",contour=own==304)
        with np.load(CACHES["C3"]/f"parcels_{t:03d}.npz") as p:
            sel=(p["origin"]==304)&(p["ctype"]==1);ax.scatter(p["lon"][sel],p["lat"][sel],s=14,facecolors="none",edgecolors="#e86f00",lw=.8)
    fig.suptitle("已否决候选演化：紫线为设计刚体轮廓，橙圈为真实原 ALB 来源单元",fontsize=16)
    fig.subplots_adjust(bottom=.14,top=.91,wspace=.18,hspace=.3);base.elevation_legend(fig,.09)
    stamp(fig,"背景相对当时全球海平面；运动与准时、保参照系裂离保持 C2。")
    save(fig,"29_west_arc_c3_evolution.png")
    lo,la=np.meshgrid(g.lon,g.lat);west=(lo>=-26)&(lo<=6)&(la>=30)&(la<=44)
    delta=cases["C3"]["E"].astype(float)-cases["C2"]["E"].astype(float);area=g.cell_area_km2
    changes={"max_abs_elevation_change_m":float(abs(delta).max()),"area_abs_change_over_150m_km2":float(area[abs(delta)>150].sum()),
      "land_mask_changed_area_km2":float(area[cases["C2"]["land"]!=cases["C3"]["land"]].sum()),
      "outside_west_max_abs_m":float(abs(delta[~west]).max()),"outside_west_area_over_1m_km2":float(area[(~west)&(abs(delta)>1)].sum()),
      "outside_west_land_mask_changed_area_km2":float(area[(~west)&(cases["C2"]["land"]!=cases["C3"]["land"])].sum())}
    old={r["id"]:r for r in cases["C2"]["summary"]["comparison"]["mountains"]}
    mountains=[{"id":r["id"],"C2_p90_m":old[r["id"]]["engine_p90_m"],"C3_p90_m":r["engine_p90_m"],"delta_m":r["engine_p90_m"]-old[r["id"]]["engine_p90_m"]} for r in cases["C3"]["summary"]["comparison"]["mountains"]]
    fig,axes=plt.subplots(2,1,figsize=(14,9),gridspec_kw={"height_ratios":[2,1]})
    im=axes[0].imshow(delta,extent=(-180,180,-90,90),origin="upper",aspect="equal",cmap="RdBu_r",vmin=-1500,vmax=1500)
    axes[0].contour(g.lon,g.lat,cases["C3"]["land"],levels=[.5],colors="#333",linewidths=.35)
    axes[0].set(title="已否决候选 − C2 全球高程差",xlabel="经度 °E",ylabel="纬度 °N");fig.colorbar(im,ax=axes[0],shrink=.75,label="m，截于 ±1500")
    axes[1].bar(range(len(mountains)),[r["delta_m"] for r in mountains]);axes[1].set_xticks(range(len(mountains)),[r["id"] for r in mountains],rotation=50,fontsize=8)
    axes[1].set(title="同一定义下的既有山系 p90 变化",ylabel="m")
    fig.tight_layout(rect=(0,.11,1,.96));stamp(fig,"西口窗口外最大变化 "+str(changes["outside_west_max_abs_m"])+" m；海陆掩膜改变 "+format(changes["land_mask_changed_area_km2"],".0f")+" km²。")
    save(fig,"30_west_arc_c3_global_change.png")
    same=[];different=[]
    with np.load(CACHES["C2"]/"forward.npz") as a,np.load(CACHES["C3"]/"forward.npz") as b:
        for k,t in enumerate(a["times"]):
            if t<40:continue
            for field in f.FIELDS:
                name=field+"_"+str(k);(same if np.array_equal(a[name],b[name],equal_nan=True) else different).append(name)
    assert not different
    # 最终关键节点从完整世界图层取样，不把隔离提交的结果冒充完整演化结果。
    critical_final={}
    for key,c in cases.items():
        critical_final[key]=[]
        for n in (100961,68047):
            ii=int(np.argmin(abs(g.lat-mesh.lat[n])));jj=int(np.argmin(abs(g.lon-mesh.lon[n])))
            assert int(raster.idx[ii,jj])==n
            critical_final[key].append({"fixed_node":n,"sample_lon":float(g.lon[jj]),"sample_lat":float(g.lat[ii]),
                "type":int(c["C"][ii,jj]),"T_km":float(c["T"][ii,jj]),"elevation_m":float(c["E"][ii,jj]),"ocean_age_Myr":float(c["age"][ii,jj]) if np.isfinite(c["age"][ii,jj]) else None})

    # 31：同一对单元间距与失败候选的封口单元历史，不把瞬时符号解释为持续扩张。
    pair_tracks={}
    for n,uids in [(100961,[356769,69978]),(68047,[14459,356769])]:
        rows=[]
        for t in range(10,-1,-1):
            with np.load(CACHES["C2"]/f"parcels_{t:03d}.npz") as p:
                ids=[np.flatnonzero(p["uid"]==u) for u in uids]
                if not all(len(x)==1 for x in ids):continue
                k=np.array([x[0] for x in ids]);xyz=lonlat_to_xyz(p["lon"][k],p["lat"][k])
                rows.append({"t_Ma":t,"pair_distance_km":float(km_from_chord(np.linalg.norm(xyz[0]-xyz[1])))})
        pair_tracks[str(n)]={"uids":uids,"rows":rows}
    blocked_uid=481424;blocking=[]
    for t in range(40,-1,-1):
        with np.load(CACHES["C3"]/f"parcels_{t:03d}.npz") as p:
            k=np.flatnonzero(p["uid"]==blocked_uid)
            if not len(k):continue
            k=int(k[0]);blocking.append({"t_Ma":t,"uid":blocked_uid,**{v:float(p[v][k]) for v in ("T","raw_elevation_m","lon","lat","origin","t_form","oro","oro_t")}})
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for n,entry in pair_tracks.items():
        rows=entry["rows"];axes[0,0].plot([r["t_Ma"] for r in rows],[r["pair_distance_km"] for r in rows],"-o",ms=4,label="节点 "+n+" 的同一对来源单元")
    axes[0,0].set(xlim=(10,0),xlabel="距今 Ma（实际采样）",ylabel="km",title="C2：同一对单元总体靠近；最后一瞬可局部转为分离");axes[0,0].legend(fontsize=8);axes[0,0].grid(alpha=.3)
    axes[0,1].plot([r["t_Ma"] for r in blocking],[r["T"] for r in blocking],"-o",c="#9b5656")
    axes[0,1].set(xlim=(5,0),xlabel="距今 Ma（实际采样）",ylabel="地壳厚度 km",title="失败候选封口单元 UID 481424：5 Ma 入表，2 Ma 增厚")
    axes[0,1].grid(alpha=.3)
    base.map_panel(axes[1,0],cases["C3"]["E"],cases["C3"]["land"],"失败候选：同一单元向西运动并露出海面",extent=(-18,-9,33,40))
    axes[1,0].plot([r["lon"] for r in blocking],[r["lat"] for r in blocking],"-o",c="#d21b52")
    for r in blocking:axes[1,0].annotate(str(r["t_Ma"])+" Ma",(r["lon"],r["lat"]),xytext=(7,5),textcoords="offset points",fontsize=8,color="#9b1742")
    axes[1,1].plot([r["t_Ma"] for r in blocking],[r["raw_elevation_m"] for r in blocking],"-o",c="#9b5656")
    axes[1,1].axhline(0,c="#444",lw=.8)
    axes[1,1].set(xlim=(5,0),xlabel="距今 Ma",ylabel="m，相对 0 Ma 海平面的原始均衡高程",title="地壳厚度跨过海平面；不是浅水仍畅通")
    axes[1,1].grid(alpha=.3)
    fig.suptitle("形成过程检验：局部填成浅水可以，连续演化却会封口",fontsize=16)
    fig.tight_layout(rect=(0,.12,1,.95));stamp(fig,"单元间距不是边界法向开合量；封口单元高程未减去各年代全球海平面，不能冒充古水深。候选已撤回。")
    save(fig,"31_west_arc_c3_failed_candidate_history.png")
    result={"checkpoint":"U35-C3","baseline_commit":"a2d5369","candidate_status":"rejected","production_result":"C2 保持不变","method_class":"procedural_constrained",
        "units":{"time":"Ma","volume":"km3","thickness":"km","elevation":"m","velocity":"cm/yr"},
        "reference":"沿用 U32 陆缘 profile 与 holes 体积借用规则；地球类比见 docs/u35_c3_west_arc.md，数值不作地球反演。",
        "observer_neutrality":neutrality,"pre_event_arrays":{"identical":True,"arrays_checked":len(same)},
        "actual_precommit_critical_gaps":crit,"isolated_commit_spike":spike,
        "baseline_last_10Myr_process_budget":totals,"critical_source_pair_tracks":pair_tracks,"rejected_blocking_uid_history":blocking,"baseline_collision_allocation":alloc,"allocation_note":"按同年代保存的存活目标位置重算既有各向同性400 km分配核，严格闭合供体体积；带内外按M31半宽和目标中心判定，不是逐单元实测应变。",
        "strait_section":sections,"critical_final_samples":critical_final,
        "connectivity_sensitivity":{key:connectivity(c) for key,c in cases.items()},
        "global_change":changes,"mountain_changes":mountains,
        "A":{key:c["summary"]["A_engine_0Ma"] for key,c in cases.items()},
        "last_step":{key:c["obs"]["rows"][-1] for key,c in cases.items()}}
    base.write_json(ROOT/"data/worlds/design_v2/u35_c3_diagnostic.json",result)
    base.write_json(ROOT/"data/worlds/design_v2/u35_c3_process_observations.json",diag)
    base.write_json(ROOT/"data/worlds/design_v2/u35_c3_observations.json",cases["C3"]["obs"])
    names=["26_west_arc_c3_gap_evidence.png","27_west_arc_c3_volume_allocation.png","28_west_arc_c3_comparison.png","29_west_arc_c3_evolution.png","30_west_arc_c3_global_change.png","31_west_arc_c3_failed_candidate_history.png"]
    html="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>U35-C3 西口</title><style>body{margin:24px;background:#eee;color:#222;font-family:system-ui}img{max-width:100%}section{margin:30px 0}a{color:#176b8a}</style><h1>U35-C3：南侧陆缘填隙与山弧过程账</h1><p>候选导致西口封闭，已撤回；正式物理结果保持 C2。本检查点提交诊断与失败证据，未宣布 U35 收尾或阶段通过。</p><p><a href='u35_c2.html'>C2 对照</a> · <a href='../../../stage1/design_v2_forward_v2_0Ma/index.html'>正式结果标准图组与指标</a> · <a href='../../../stage1/design_v2_forward_v2_0Ma_c3_rejected/index.html'>已否决候选标准图组与指标</a></p>"
    for name in names:html+="<section><h2>"+name+"</h2><img src='"+name+"'></section>"
    (OUT/"u35_c3.html").write_text(html+"</html>",encoding="utf-8",newline="\n")
    rejected=ROOT/"figures/stage1/design_v2_forward_v2_0Ma_c3_rejected"
    status={"checkpoint":"U35-C3","status":"rejected","production_result":"C2","reason":"普通4/8连通均断开；A层仍全过，不能据此采用候选。"}
    base.write_json(rejected/"experiment_status.json",status)
    page=rejected/"index.html"
    if page.exists():
        text=page.read_text(encoding="utf-8")
        if 'id="u35-c3-rejected"' not in text:
            import re
            banner='<section id="u35-c3-rejected" style="padding:20px;background:#ffe3e3;color:#752020"><h1>U35-C3 已否决候选</h1><p>南侧陆缘规则在后续碰撞中封住西口。此图组为失败实验留档，正式结果保持 C2。</p><p><a href="../design_v2_forward_v2_0Ma/index.html">查看正式结果</a></p></section>'
            text=re.sub(r"(<body[^>]*>)",lambda m:m.group(1)+banner,text,count=1)
            page.write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps({k:result[k] for k in ("observer_neutrality","baseline_last_10Myr_process_budget","strait_section","critical_final_samples","connectivity_sensitivity","global_change","mountain_changes","last_step")},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
