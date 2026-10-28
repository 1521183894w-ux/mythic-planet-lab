"""S13图件、数据复核与冻结世界一致性；不更改正式世界。"""
from pathlib import Path
import json, hashlib, html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import unit, R_KM
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/worlds/design_v2"
OUT=ROOT/"figures/design/v2/forward/u36_s13"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def write(p,q):p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
def main():
    setup_fonts();OUT.mkdir(exist_ok=True)
    q=read(DATA/"u36_s13_curved_quadrature.json")
    raw=np.load(ROOT/".cache/u36_s13/material_samples.npz")
    points=raw["points"]; jj=raw["J"]
    maximum=0.; minimum=float("inf")
    # 使用独立的矢量三重积/球面扇形表达式复算全部保存的最密边界。
    for step,row in enumerate(q["timeline"]):
        for cell,spec in zip(row["cells"],q["sampling"]):
            c=points[step,spec["center_index"]];p=points[step,spec["boundary_indices"]];r=np.roll(p,-1,axis=0)
            solid=2*np.arctan2(np.cross(p-c,r-c)@c,1+p@c+r@c+np.sum(p*r,axis=1))*R_KM**2
            minimum=min(minimum,float(solid.min()))
            maximum=max(maximum,abs(float(solid.sum())-cell["boundary_area_km2"]["512"]))
            spec64=spec["quadratures"]["64"]
            integral=np.array(spec64["weights_km2"])@jj[step,spec64["indices"]]
            maximum=max(maximum,abs(float(integral)-cell["interior_integral_J_km2"]["64"]))
    assert maximum<1e-7 and minimum>0
    metrics=read(ROOT/"figures/stage1/u36_s13_reference/metrics.json")
    old=read(DATA/"u36_s12_reference_metrics.json")
    clean=lambda z:{k:v for k,v in z["metrics"].items() if k!="timings_s"}
    assert clean(metrics)==clean(old) and metrics["b_verdict"]==old["b_verdict"]
    regions=read(ROOT/"figures/stage1/u36_s13_reference/regions.json")
    assert regions["regions"]==read(DATA/"u36_s12_reference_regions.json")["regions"]
    for name in ["metrics","regions","manifest"]:
        write(DATA/f"u36_s13_reference_{name}.json",read(ROOT/f"figures/stage1/u36_s13_reference/{name}.json"))
    (DATA/"u36_s13_tests.txt").write_text("\n".join(z.rstrip() for z in (ROOT/".cache/u36_s13_tests.log").read_text(encoding="utf8").splitlines())+"\n",encoding="utf8")
    final=q["timeline"][-1]["cells"]; summaries=[]
    for i,z in enumerate(final):
        history=[row["cells"][i] for row in q["timeline"]]
        n0=q["selection"][i]["initial_area_km2"];den=n0*z["center_J"]
        summaries.append(dict(uid=z["uid"],cell_index=z["cell_index"],
             endpoint_residual_percent=z["total_endpoint_center_residual"]*100,
             edge_contribution_percent=z["chord_edge_contribution"]*100,
             center_contribution_using_boundary512_percent=z["center_vs_curved_contribution"]*100,
             center_contribution_using_interior64_percent=(z["interior_integral_J_km2"]["64"]/den-1)*100,
             boundary512_interior64_difference_percent=(z["boundary_area_km2"]["512"]-z["interior_integral_J_km2"]["64"])/n0*100,
             boundary256_512_difference_percent=(z["boundary_area_km2"]["256"]-z["boundary_area_km2"]["512"])/n0*100,
             quadrature32_64_difference_percent=(z["interior_integral_J_km2"]["32"]-z["interior_integral_J_km2"]["64"])/n0*100,
             chord_crossing_sample_times=sum(v["chord_crosses_reference_face"] for v in history),
             maximum_material_other_face_weight_fraction=max(v["quadrature_material_other_face_weight_fraction"] for v in history),
             maximum_all_times_quadrature32_64_difference_percent=max(abs(v["interior_integral_J_km2"]["32"]-v["interior_integral_J_km2"]["64"])/n0*100 for v in history),
             maximum_all_times_boundary512_interior64_difference_percent=max(abs(v["boundary_area_km2"]["512"]-v["interior_integral_J_km2"]["64"])/n0*100 for v in history)))
    write(DATA/"u36_s13_verification.json",dict(checkpoint="U36-S13",method_class="procedural_constrained",
         recalculation_time_count=len(q["timeline"]),maximum_saved_area_recalculation_difference_km2=maximum,
         minimum_sampled_boundary_fan_area_km2=minimum,cell_summary=summaries,
         reference_metrics_identical_except_timings=True,region_windows_identical=True,
         related_tests_passed=44,related_tests_failed=2,old_failures_retained=True,
         protected=q["protected"],global_or_B5_coupled=False,default_world_adopted=False,
         all_sources_or_full_domain_convergence_claimed=False))
    figs=[]
    def save(fig,file,title,note):
        fig.text(.015,.015,"7cc61a8+U36-S13 · design_v2；同一S11运动场 interpolated；给定200km宽度 user_override\n"
                 "曲边/求积诊断 procedural_constrained；图 visual_only；L7约60km，采样细化不增加物理分辨率\n"+note,fontsize=8,va="bottom")
        fig.savefig(OUT/file,dpi=145);plt.close(fig)
        figs.append(dict(file=file,title=title,note=note,sha256=hashlib.sha256((OUT/file).read_bytes()).hexdigest()))
    fig,axes=plt.subplots(4,2,figsize=(16,15))
    for (ax,profile),z,spec in zip(axes,final,q["sampling"]):
        p=points[-1,spec["boundary_indices"]];c=points[-1,spec["center_index"]]
        east=unit(p[0]-c*(p[0]@c));north=np.cross(c,east)
        xy=np.c_[p@east,p@north]/(p@c)[:,None]*R_KM
        chord=xy[np.arange(3)*512]
        ax.plot(*np.r_[xy,xy[:1]].T,label="随材料移动的边界（每边512段）",lw=1.8)
        ax.plot(*np.r_[chord,chord[:1]].T,"--",label="三个终点间的大圆边",lw=1.1)
        ax.plot(0,0,"+",color="black",label="原中心的材料轨迹")
        ax.set(title=f"UID{z['uid']} / 单元{z['cell_index']}；总差{z['total_endpoint_center_residual']*100:+.4f}%",
               xlabel="局部心射东向 / km",ylabel="局部心射北向 / km")
        ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.2)
        for edge in range(3):
            segment=p[edge*512:(edge+1)*512]
            normal=unit(np.cross(p[edge*512],p[((edge+1)%3)*512]))
            deviation=np.arcsin(np.clip(segment@normal,-1,1))*R_KM*1000
            profile.plot(np.arange(512)/512,deviation,label=f"原始边{edge+1}")
        profile.set(title="材料边界离三顶点大圆边的有符号法向距离",xlabel="初始边参数（归一化线性混合，0→1）",ylabel="离大圆边距离 / m")
        profile.grid(alpha=.2);profile.legend(fontsize=7);profile.axhline(0,color="black",lw=.5)
    fig.suptitle("四个材料单元：曲边形状与相对大圆边的法向偏离",y=.98,fontsize=16)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.95),ncol=3,fontsize=9)
    fig.subplots_adjust(bottom=.12,hspace=.85,wspace=.3)
    save(fig,"121_material_curved_edges.png","同一材料的曲边与三顶点直边",
         "按旧误差选取四个诊断单元，不代表全部来源；局部心射图保等比例；右列以米显示曲边法向偏离。")
    fig,axes=plt.subplots(1,2,figsize=(15,8));x=np.arange(4)
    for field,label,color in [("endpoint_residual_percent","原三顶点面积/中心J","#333333"),
                              ("edge_contribution_percent","直边近似贡献","#b36f48"),
                              ("center_contribution_using_boundary512_percent","曲边平均与中心取样贡献","#337f91")]:
        offset={"endpoint_residual_percent":-.24,"edge_contribution_percent":0,"center_contribution_using_boundary512_percent":.24}[field]
        axes[0].bar(x+offset,[s[field] for s in summaries],width=.24,label=label,color=color)
    axes[0].set_xticks(x,[str(z["uid"]) for z in final]);axes[0].set(xlabel="来源UID（各仅一个单元）",ylabel="占初始面积×中心J / %",title="末态误差分解（有符号）")
    axes[0].legend(fontsize=8)
    for i,z in enumerate(final):
        history=[row["cells"][i] for row in q["timeline"]]
        axes[1].plot([row["t_ma"] for row in q["timeline"]],[v["total_endpoint_center_residual"]*100 for v in history],label=f"UID{z['uid']}")
    axes[1].set(xlabel="年代 / Ma",ylabel="原三顶点面积/中心J差 / %",title="四个单元的原误差",xlim=(35,0));axes[1].legend(fontsize=8)
    for ax in axes:ax.axhline(0,color="black",lw=.6);ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.2,wspace=.3)
    save(fig,"122_error_decomposition.png","直边误差与中心取样分开",
         "总差等于两项之和是算术分解；没有逐来源归一化，不作为质量或物理守恒证明。")
    fig,axes=plt.subplots(2,2,figsize=(14,11))
    for ax,z,selection in zip(axes.ravel(),final,q["selection"]):
        a0=selection["initial_area_km2"];b=z["boundary_area_km2"];integrals=z["interior_integral_J_km2"]
        ax.loglog([int(n) for n in b],[max(abs(v-integrals["64"])/a0*100,1e-13) for v in b.values()],"o-",label="曲边面积相对内部64×64")
        ax.loglog([int(n) for n in integrals],[max(abs(v-b["512"])/a0*100,1e-13) for v in integrals.values()],"s-",label="内部积分相对曲边512段")
        ax.set(title=f"UID{z['uid']}：两种独立面积近似",xlabel="每边段数 / 每Duffy轴Gauss阶数（成本不同）",ylabel="相差占初始面积 / %")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=.17,hspace=.3,wspace=.3)
    save(fig,"123_independent_area_sampling.png","曲边几何与内部J积分的采样敏感性",
         "最高两档也只是有限近似；跨片内部积分可不单调，两曲线不是等成本精度排名。")
    fig,axes=plt.subplots(2,2,figsize=(14,11))
    times=[row["t_ma"] for row in q["timeline"]]
    for i,(ax,z) in enumerate(zip(axes.ravel(),final)):
        history=[row["cells"][i] for row in q["timeline"]]
        ax.plot(times,[v["quadrature_material_other_face_weight_fraction"]*100 for v in history],label="内部材料样本位于中心以外分片的初始权重")
        ax.step(times,[100*int(v["chord_crosses_reference_face"]) for v in history],where="mid",alpha=.5,label="旧直边跨片标记（0/100）")
        ax.set(title=f"UID{z['uid']} / 单元{z['cell_index']}",xlabel="年代 / Ma",ylabel="初始面积权重 / %；标记",xlim=(35,0),ylim=(-3,103));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.subplots_adjust(bottom=.16,hspace=.3,wspace=.3)
    save(fig,"124_material_face_history.png","直边跨片与实际材料采样点的分片历史",
         "分片是运动插值分区，不是断层；只检查141个离散时点和有限样本，非连续域证明。")
    write(DATA/"u36_s13_manifest.json",dict(checkpoint="U36-S13",method_class="visual_only",figures=figs))
    page='<meta charset="utf-8"><title>U36-S13 曲边与中心求积</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S13：同一材料的曲边与中心求积</h1><p>限定方法检查点；按误差选取四个单元，未接全球/B5，阶段1未通过。</p><p><a href="../../../../stage1/u36_s13_reference/index.html">冻结世界的29张标准图</a></p>'
    for f in figs:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/"index.html").write_text(page,encoding="utf8")
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
    print("RECALC",maximum,"FIGURES",len(figs),flush=True)
if __name__=="__main__":main()
