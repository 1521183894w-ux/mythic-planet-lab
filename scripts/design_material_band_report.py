"""U36-S4：参考接触、固定材料拓扑、带宽与框架误差的可视审计。"""
from __future__ import annotations
import html
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mythic.figures.style import setup_fonts
from mythic.tectonics.material_band import reference_strip
from mythic.tectonics.geometry import R_KM, xyz_to_lonlat

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/worlds/design_v2"
CACHE=ROOT/".cache/u36_s4"
OUT=ROOT/"figures/design/v2/forward/u36_s4"
FIGURES=[]


def save(fig,name,title,note):
    fig.text(.012,.012,"U36-S4 · "+VERSION+" · 历史 design_v2/history_v2 · pyGPlates 1.0.0\n"
        +"参考线/带宽 user_override；接触匹配/网络/初始厚度 interpolated；球面材料面积 procedural_constrained\n"+note,
        fontsize=8,va="bottom")
    fig.savefig(OUT/name,dpi=140)
    plt.close(fig)
    FIGURES.append(dict(file=name,title=title,note=note))


def geographic(ax,xyz,title):
    lon,lat=xyz_to_lonlat(xyz)
    ax.set(title=title,xlabel="经度 / °E",ylabel="纬度 / °N")
    ax.set_aspect(1/np.cos(np.radians(np.mean(lat))))
    ax.grid(alpha=.2)
    return lon,lat


def scale(ax,lat):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    x=x0+.05*(x1-x0); y=y0+.05*(y1-y0)
    dx=100/(R_KM*np.pi/180*np.cos(np.radians(lat)))
    ax.plot([x,x+dx],[y,y],color="#222",lw=3)
    ax.text(x,y+.02*(y1-y0),"100 km",fontsize=8)


def main():
    global VERSION
    setup_fonts();OUT.mkdir(parents=True,exist_ok=True)
    d=json.loads((DATA/"u36_s4_material_band.json").read_text(encoding="utf-8"));VERSION=d["version"]
    a=np.load(CACHE/"material_band.npz")
    tri=a["triangles"];xyz0=a["xyz"];xyz=a["timeline_xyz"][-1];known=a["known"]
    # 同一35Ma控制原生状态，暴露参考线与实际材料来源的错位。
    with (CACHE/"control/reference_35Ma.pkl").open("rb") as stream: eng=pickle.load(stream)
    pos=eng.positions();lon,lat=xyz_to_lonlat(pos)
    strip_lon,strip_lat=xyz_to_lonlat(xyz0)
    use=eng.alive&(lon>strip_lon.min()-2)&(lon<strip_lon.max()+2)&(lat>strip_lat.min()-1)&(lat<strip_lat.max()+1)
    ids=np.flatnonzero(use);categories=np.where(eng.c.plate[ids]==301,0,np.where(eng.c.plate[ids]==401,1,2))
    fig,axs=plt.subplots(1,3,figsize=(14,9))
    cmap=ListedColormap(["#bc8148","#4e90ac","#a9a9a9"])
    axs[0].scatter(lon[ids],lat[ids],c=categories,cmap=cmap,vmin=0,vmax=2,s=15,alpha=.7)
    for endpoints,color,label in [(d["design_endpoints_xyz"],"#8b3cb5","原分区线重建"),(d["registered_endpoints_xyz"],"#202020","35Ma接触匹配线")]:
        s=reference_strip(*np.asarray(endpoints),200,30)
        lo,la=xyz_to_lonlat(s["xyz"][s["grid"][:,s["grid"].shape[1]//2]])
        axs[0].plot(lo,la,color=color,label=label,lw=2)
    for rail in (a["grid"][:,0],a["grid"][:,-1]):axs[0].plot(strip_lon[rail],strip_lat[rail],"k--",lw=1)
    geographic(axs[0],xyz0,"35 Ma：原生C棕/D蓝与参考带");axs[0].legend(fontsize=8);scale(axs[0],strip_lat.mean())
    contacts=d["reference_registration"]["contacts"]
    good=[r for r in contacts if r["transition_count"]==1]
    axs[1].scatter([r["offset_km"] for r in good],[r["along_km"] for r in good],label="直接C→D转换（保存UID）",s=28)
    other=[r for r in contacts if r["transition_count"]!=1]
    axs[1].scatter(np.full(len(other),-205),[r["along_km"] for r in other],marker="x",color="grey",label="无直接唯一转换，未定")
    offsets=d["reference_registration"]["endpoint_offset_km"]
    axs[1].plot(offsets,[0,a["along_km"][-1]],"k--",label="一次拟合后固定")
    axs[1].set(title="只在35Ma登记，不按缝隙逐次重选",xlabel="相对原分区线横向偏移 / km",ylabel="沿段距离 / km")
    axs[1].grid(alpha=.2);axs[1].legend(fontsize=8)
    code=np.where(known,np.where(a["source_plate"]==301,0,1),2)
    pc=axs[2].tripcolor(strip_lon,strip_lat,tri,facecolors=code,cmap=cmap,vmin=0,vmax=2,edgecolors="none")
    geographic(axs[2],xyz0,"参考审计三角形的初始来源")
    fig.colorbar(pc,ax=axs[2],ticks=[0,1,2],shrink=.5).ax.set_yticklabels(["C陆壳","D陆壳","缺值/非陆壳"])
    scale(axs[2],strip_lat.mean());fig.subplots_adjust(bottom=.18,wspace=.42,top=.9)
    save(fig,"81_reference_contact_and_sources.png","参考接触与原生来源","12/28条横切有直接唯一C→D转换；其余保持未定。拟合偏移不等于断层恢复；5km查询间距，原生仍L7约60km。")
    fig,axs=plt.subplots(1,3,figsize=(14,9))
    for ax,points,values,title in [(axs[0],xyz0,a["H0_km"],"35 Ma：原生覆盖初始厚度"),(axs[1],xyz,a["H0_km"]/a["timeline_J"][-1],"0 Ma：材料面积对应H0/J")]:
        lo,la=geographic(ax,points,title)
        ax.set_facecolor("#bdbdbd")
        pc=ax.tripcolor(lo,la,tri,facecolors=np.ma.masked_invalid(values),cmap="viridis",vmin=0,vmax=50)
        fig.colorbar(pc,ax=ax,shrink=.6,label="整柱厚度 / km");scale(ax,la.mean())
    lo,la=geographic(axs[2],xyz,"0 Ma：独立球面材料面积J")
    pc=axs[2].tripcolor(lo,la,tri,facecolors=a["timeline_J"][-1],cmap="magma",vmin=1,vmax=3)
    fig.colorbar(pc,ax=axs[2],shrink=.6,label="J=A/A0 / 无量纲");scale(axs[2],la.mean())
    fig.subplots_adjust(bottom=.18,wspace=.65,top=.91)
    save(fig,"82_material_geometry_and_columns.png","连续材料拓扑和整柱厚度","同一批固定三角形，未补新柱；灰色为初始厚度缺值。H0/J闭合是定义性材料账，不是物理求解验证；不是高程或B5盆底。")
    fig,axs=plt.subplots(1,3,figsize=(14,7.5))
    for r,color in zip(d["width_cases"],["#b36733","#70559e","#2b8693"]):
        ts=[x["t_ma"] for x in r["timeline"]]
        axs[0].plot(ts,[x["area_ratio"] for x in r["timeline"]],"o-",label=f"W0={r['width0_km']} km",color=color)
        q=r["final_known_H_quantiles_km"]
        axs[1].errorbar(r["width0_km"],q[2],yerr=[[q[2]-q[1]],[q[3]-q[2]]],fmt="o",capsize=5,color=color)
        axs[2].bar(r["width0_km"],r["source"]["known_cont_area_fraction"]*100,width=40,color=color)
    axs[0].set(title="同一参考接触与真实C/D速度",xlabel="年代 / Ma",ylabel="材料带面积比 / 无量纲");axs[0].invert_xaxis();axs[0].legend(fontsize=9)
    axs[1].set(title="已知陆壳区：中位与P10–P90",xlabel="明确假设的初始总带宽 / km",ylabel="末态H0/J / km")
    axs[2].set(title="初始陆壳厚度可查询面积",xlabel="初始总带宽 / km",ylabel="占参考带面积 / %",ylim=(0,100))
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.36,top=.88)
    save(fig,"83_width_and_material_history.png","带宽敏感性与材料历史","三种W0都为试验假设，未按目标深度选优；H0来自各自实际覆盖。误差条描述厚度分布，不是置信区间。")
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for dx,color in [(60,"#b36733"),(30,"#70559e"),(15,"#2b8693")]:
        rows=[r for r in d["convergence"] if r["width0_km"]==200 and r["dx_requested_km"]==dx]
        axs[0,0].plot([r["dt_myr"] for r in rows],[r["timeline"][-1]["area_ratio"] for r in rows],"o-",label=f"采样约{dx} km",color=color)
        axs[0,1].plot([r["dt_myr"] for r in rows],[100*r["framework_known_volume_relative_residual"] for r in rows],"o-",label=f"采样约{dx} km",color=color)
    axs[0,0].set(title="200 km带：几何J对采样/时间步",xlabel="时间步 / Myr",ylabel="末态总面积比")
    axs[0,0].ticklabel_format(style="plain",useOffset=False,axis="y")
    axs[0,1].set(title="用框架beta换算厚度的已知区残差",xlabel="时间步 / Myr",ylabel="体积相对残差 / %")
    rows=d["interpolation_cases"]
    smoothing_labels={"none":"无平滑","barycentric":"重心平滑","natural_neighbour":"自然邻平滑"}
    labels=[smoothing_labels[r["strain_smoothing"]]+"\n速度"+("自然邻" if r["natural_neighbor_velocity"] else "重心") for r in rows]
    axs[1,0].bar(np.arange(len(rows)),[100*r["timeline"][-1]["framework_volume_relative_residual"] for r in rows],color="#70559e")
    axs[1,0].set_xticks(np.arange(len(rows)),labels,fontsize=8);axs[1,0].set(title="六种框架插值组合，dt=0.25Myr",ylabel="均匀H0体积相对残差 / %")
    for g in d["guard_cases"]:
        if g["status"]=="ran":axs[1,1].plot(g["guard_km"],g["timeline"][-1]["area_ratio"],"o",color="#2b8693")
        else:axs[1,1].text(g["guard_km"],1.8,"框架失败",rotation=90)
    axs[1,1].set(title="网络端部与参考材料段保持距离",xlabel="两端各延伸缓冲 / km",ylabel="末态材料面积比")
    for ax in axs.flat:ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.subplots_adjust(bottom=.17,wspace=.28,hspace=.48,top=.91)
    save(fig,"84_framework_convergence_and_limits.png","几何与框架标量残差","H0/J的体积闭合不画成成功证据；直接显示框架beta与独立几何不一致。不同插值/时间步未消除此残差，根因尚未定位。")
    fig,axs=plt.subplots(1,3,figsize=(14,7.5))
    names={"opening":"给定正张开","slip":"纯走滑：连续真值J=1","rigid":"共同刚体：连续真值J=1"}
    for ax,case in zip(axs,("opening","slip","rigid")):
        for dt,color in [(1.,"#b36733"),(.5,"#70559e"),(.25,"#2b8693")]:
            rows=next(r["timeline"] for r in d["controls"] if r["case"]==case and r["dt_myr"]==dt)
            ax.plot([r["t_ma"] for r in rows],[(r["area_ratio"]-1)*100 for r in rows],"o-",label=f"dt={dt} Myr",color=color)
        if case!="opening":ax.axhline(0,color="black",ls="--",label="连续真值")
        ax.set(title=names[case],xlabel="年代 / Ma",ylabel="材料面积变化 / %");ax.invert_xaxis();ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=.23,wspace=.35,top=.88)
    save(fig,"85_network_negative_controls.png","框架负对照的实际数值误差","同一200km参考带，半球光滑边界；纯走滑不应真实扩张。ppm级偏差仍公开，不把数值近似写成严格零应变；不是释放阶区试验。")
    manifest=dict(checkpoint="U36-S4",version=VERSION,figures=FIGURES,scientific_limitations=d["limitations"],
        standard_figures="figures/stage1/u36_s4_reference/index.html",framework_version=d["framework_version"],
        metrics_source="figures/stage1/u36_s4_reference/metrics.json")
    (DATA/"u36_s4_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    content='<!doctype html><meta charset="utf-8"><title>U36-S4 参考材料带</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;background:#f5f3ee;color:#242424}img{width:100%}p{max-width:1100px}</style><h1>U36-S4：参考材料带与框架误差</h1><p>独立几何与来源审计；没有新地貌、没有B5沉降或阶段通过结论。</p>'
    for f in FIGURES:content+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/"index.html").write_text(content,encoding="utf-8")
    print("figures",len(FIGURES),flush=True)


if __name__=="__main__":main()
