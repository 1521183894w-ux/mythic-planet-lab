"""S14实际结果出图和保存列复算；可在完整域运行中先输出四单元图。"""
from pathlib import Path
import argparse,json,hashlib,html,gc,re,time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import unit,R_KM
from mythic.tectonics.material_fv import boundary_areas,thickness_error_integrals
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/worlds/design_v2";CACHE=ROOT/".cache/u36_s14";OUT=ROOT/"figures/design/v2/forward/u36_s14"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def write(p,q):p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
TAGS=["selected_l0_b512_q32","selected_l1_b256_q32","selected_l2_b128_q32"]
def main(selected_only):
    started=time.perf_counter();setup_fonts();OUT.mkdir(exist_ok=True);figs=[]
    rows=[read(DATA/f"u36_s14_{tag}.json") for tag in TAGS]
    uid_order=[z["uid"] for z in rows[0]["source_summary"]]
    def save(fig,file,title,note):
        fig.text(.015,.012,"6be2d4d+U36-S14 · design_v2；同一S11映射 interpolated；200km材料带 user_override\n"
                 "移动材料体积账/误差 procedural_constrained；图 visual_only；L7约60km，细分不提高物理输入分辨率\n"+note,fontsize=8,va="bottom")
        fig.savefig(OUT/file,dpi=145);plt.close(fig)
        figs.append(dict(file=file,title=title,note=note,sha256=hashlib.sha256((OUT/file).read_bytes()).hexdigest()))
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for ax,u in zip(axes.ravel(),uid_order):
        vals=[next(z for z in q["source_summary"] if z["uid"]==u) for q in rows]
        ax.loglog([1,4,16],[z["L1_thickness_error"]*100 for z in vals],"o-",label="曲边面积更新的平均厚度")
        center=[]
        for tag in TAGS:
            a=np.load(CACHE/(tag+".npz"));mask=a["source_uid"]==u;n=32
            h0=a["H0_km"];j=a[f"q{n}_J"][-1];cj=a["center_J"][-1];w=a[f"q{n}_weights_km2"]
            v=thickness_error_integrals(h0/cj,h0,w,j)
            center.append(float(v["l1_volume_error_km3"][mask].sum()/a["V0_km3"][np.searchsorted(a["uids"],u)]*100));a.close()
        ax.loglog([1,4,16],center,"s--",label="同一细分单元的中心取样")
        ax.set(title=f"UID{u}：四反例之一",xlabel="每个原单元的真实子单元数",ylabel="面积积分绝对厚度差 / 初始体积 / %")
        ax.set_xticks([1,4,16],["1","4","16"]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=.18,hspace=.35,wspace=.3)
    save(fig,"125_mean_thickness_refinement.png","守恒平均厚度与场误差的细分对照",
         "四例按旧误差挑选；各档内部参照为32×32节点H0/J；均值守恒并不保证L1场误差优于中心取样。")
    a=np.load(CACHE/(TAGS[-1]+".npz"));initial=a["snapshots"][0];j=a["q32_J"][-1];h0=a["H0_km"];means=a["mean_thickness_km"][-1,1]
    ci=a["pool_center_indices"];bi=a["pool_boundary_indices"];qi=a["q32_pool_indices"];uid=a["source_uid"];old_h=h0/a["center_J"][-1]
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for line,u in enumerate([91203,566]):
        mask=uid==u;index=np.flatnonzero(mask);c=unit(initial[ci[mask]].mean(axis=0))
        east=unit(initial[bi[index[0],0]]-c*(initial[bi[index[0],0]]@c));north=np.cross(c,east)
        def xy(p):return np.stack([p@east,p@north],axis=-1)/(p@c)[...,None]*R_KM
        polygons=[xy(initial[bi[k]]) for k in index]
        reference=1/j[mask];vmin=min(float(reference.min()),float((means[mask]/h0[mask]).min()),float((old_h[mask]/h0[mask]).min()))
        vmax=max(float(reference.max()),float((means[mask]/h0[mask]).max()),float((old_h[mask]/h0[mask]).max()))
        for col,(values,label) in enumerate([(old_h[mask]/h0[mask],"同档中心取样"),(means[mask]/h0[mask],"同档曲边控制体均值")]):
            collection=PolyCollection(polygons,array=values,cmap="viridis",edgecolor="white",linewidth=.3,clim=(vmin,vmax))
            axes[line,col].add_collection(collection);axes[line,col].autoscale_view();axes[line,col].set_title(f"UID{u}：{label}")
        refxy=xy(initial[qi[mask]].reshape(-1,3))
        im=axes[line,2].scatter(refxy[:,0],refxy[:,1],c=reference.ravel(),s=2,cmap="viridis",vmin=vmin,vmax=vmax)
        axes[line,2].set_title(f"UID{u}：独立内部点H0/J")
        for ax in axes[line]:ax.set(xlabel="初始材料心射东向 / km",ylabel="初始材料心射北向 / km");ax.grid(alpha=.2)
        fig.colorbar(im,ax=axes[line].tolist(),label="0Ma厚度 / 初始厚度",fraction=.02,pad=.02)
    fig.subplots_adjust(bottom=.18,hspace=.4,wspace=.35,right=.86)
    save(fig,"126_thickness_field.png","两个跨片反例的厚度场对照",
         "均为16子单元；位置采用初始材料坐标便于对照，纵横显示比例可不同，非0Ma地形或新世界；每行同色标。")
    a.close();del initial,j;gc.collect()
    verification=[]
    if not selected_only:
        full=read(DATA/"u36_s14_full_l0_b64_q8.json")
        all_rows=rows+[full]
        for q in all_rows:
            a=np.load(CACHE/(q["tag"]+".npz"));geom=a["geometry_points"];ci=a["center_indices"];bi=a["boundary_indices"]
            hh=a["mean_thickness_km"];areas=a["scheme_areas_km2"];a0=a["initial_area_km2"];h0=a["H0_km"];known=a["known"]
            uu=a["source_uid"];uids=a["uids"];group=np.searchsorted(uids,uu[known]);v0=a["V0_km3"]
            maximum_geometry=0.;maximum_drift=0.;maximum_saved_timeline=0.;minimum_fan=float("inf")
            saved_j={n:a[f"q{n}_J"] for n in q["quadrature_orders"]};weights={n:a[f"q{n}_weights_km2"] for n in q["quadrature_orders"]}
            qdiff=[];boundary_diff=[]
            for i,t in enumerate(a["times_ma"]):
                calculated,minimum=boundary_areas(geom[i,bi],geom[i,ci]);minimum_fan=min(minimum_fan,minimum)
                maximum_geometry=max(maximum_geometry,float(abs(calculated-areas[i,1]).max()))
                mass=np.bincount(group,weights=(hh[i,1]*areas[i,1])[known],minlength=len(uids))
                maximum_drift=max(maximum_drift,float(abs(mass-v0).max()))
                e_by_n={}
                for n in q["quadrature_orders"]:
                    val=thickness_error_integrals(hh[i,1],h0,weights[n],saved_j[n][i])
                    l1=float(np.nansum(val["l1_volume_error_km3"])/v0.sum())
                    maximum_saved_timeline=max(maximum_saved_timeline,abs(l1-q["timeline"][i][f"q{n}"]["weighted_L1_thickness_error"]))
                    e_by_n[n]=l1
                qdiff.append(abs(e_by_n[q["quadrature_orders"][-1]]-e_by_n[q["quadrature_orders"][-2]]))
                boundary_diff.append(float(np.nansum(abs(hh[i,1]-hh[i,0])*calculated)/v0.sum()))
            assert maximum_geometry<1e-6 and maximum_saved_timeline<1e-12 and maximum_drift<1e-6
            if q["level"]==0:assert q["maximum_archived_center_difference_m"]<1e-4 and q["maximum_archived_center_J_difference"]<1e-10
            verification.append(dict(tag=q["tag"],times_checked=141,maximum_saved_area_difference_km2=maximum_geometry,
                maximum_source_scheme_volume_drift_km3=maximum_drift,maximum_saved_L1_difference=maximum_saved_timeline,
                minimum_fan_area_km2=minimum_fan,maximum_all_times_L1_order_sensitivity=float(max(qdiff)),
                maximum_all_times_boundary_choice_H_sensitivity=float(max(boundary_diff))))
            if q["scope"]=="full":
                n=8;val=thickness_error_integrals(hh[-1,1],h0,weights[n],saved_j[n][-1])
                old=thickness_error_integrals(h0/a["center_J"][-1],h0,weights[n],saved_j[n][-1])
                summed=lambda v:np.bincount(group,weights=v[known],minlength=len(uids))
                mean_l1=summed(val["l1_volume_error_km3"])/v0;old_l1=summed(old["l1_volume_error_km3"])/v0
                final_sources=[dict(z,old_center_L1_thickness_error=float(old_l1[k])) for k,z in enumerate(full["source_summary"])]
                full_extra=dict(sources=final_sources,FV_L1_worse_source_count=int((mean_l1>old_l1).sum()),old_center_L1_worse_source_count=int((old_l1>mean_l1).sum()))
            a.close();del geom,saved_j;gc.collect()
        fig,axes=plt.subplots(1,2,figsize=(15,8))
        for field,label in [("old_endpoint_center_relative_difference","旧三顶点面积×中心厚度"),
                             ("scheme_relative_drift","移动材料自身几何体积账"),
                             ("reference_volume_relative_difference","独立内部面积权重×平均厚度")]:
            y=np.sort([abs(z[field])*100 for z in full_extra["sources"]])
            axes[0].semilogx(np.maximum(y,1e-12),np.arange(1,len(y)+1),label=label)
        axes[0].set(xlabel="每来源末态绝对体积差 / %",ylabel="累计来源数",title="全部77个已知来源，三种账分开");axes[0].legend(fontsize=8)
        axes[1].scatter(old_l1*100,mean_l1*100,s=22,alpha=.75)
        lim=max(old_l1.max(),mean_l1.max())*100*1.05;axes[1].plot([0,lim],[0,lim],"--",c="gray",lw=1)
        axes[1].set(xlabel="中心取样L1厚度误差 / %",ylabel="材料均值L1厚度误差 / %",title=f"全部77来源；均值方案L1较大：{full_extra['FV_L1_worse_source_count']}个")
        for ax in axes:ax.grid(alpha=.2)
        fig.subplots_adjust(bottom=.2,wspace=.3)
        save(fig,"127_all_source_conservation.png","完整来源的守恒与场误差分开",
             "2071材料单元（1931已知厚度/140缺值）；守恒账闭合是方案性质；内部参照仅4/8阶；对数图的0仅绘至1e−12%。")
        fig,axes=plt.subplots(1,2,figsize=(15,8));times=[z["t_ma"] for z in full["timeline"]]
        for field,label in [("old_endpoint_center_source_weighted_absolute_drift","旧三顶点/中心体积账"),
                            ("scheme_source_weighted_absolute_drift","移动控制体自身几何体积账")]:
            axes[0].semilogy(times,np.maximum([z[field]*100 for z in full["timeline"]],1e-12),label=label)
        for n in [4,8]:
            axes[0].semilogy(times,np.maximum([z[f"q{n}"]["reference_volume_source_weighted_absolute_difference"]*100 for z in full["timeline"]],1e-12),label=f"独立内部{n}阶面积的体积差")
            axes[1].plot(times,[z[f"q{n}"]["weighted_L1_thickness_error"]*100 for z in full["timeline"]],label=f"材料均值，内部{n}阶")
        axes[1].plot(times,[z["q8"]["old_center_weighted_L1_thickness_error"]*100 for z in full["timeline"]],"--",label="中心取样，内部8阶")
        axes[0].set(ylabel="逐来源绝对体积差加权 / %",title="全部77来源全时序体积账")
        axes[1].set(ylabel="积分绝对厚度差 / 初始体积 / %",title="账闭合不代表厚度场最优")
        for ax in axes:ax.set(xlabel="年代 / Ma",xlim=(35,0));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.subplots_adjust(bottom=.2,wspace=.3)
        save(fig,"128_full_time_reference.png","全时序独立参照与场误差",
             "有限采样/同一指定运动映射；不证明连续时间、真实流变或完整物理带宽；未知厚度不填零。")
        metrics=read(ROOT/"figures/stage1/u36_s14_reference/metrics.json");old=read(DATA/"u36_s13_reference_metrics.json")
        clean=lambda z:{k:v for k,v in z["metrics"].items() if k!="timings_s"}
        assert clean(metrics)==clean(old) and metrics["b_verdict"]==old["b_verdict"]
        assert read(ROOT/"figures/stage1/u36_s14_reference/regions.json")["regions"]==read(DATA/"u36_s13_reference_regions.json")["regions"]
        for name in ["metrics","manifest","regions"]:
            write(DATA/f"u36_s14_reference_{name}.json",read(ROOT/f"figures/stage1/u36_s14_reference/{name}.json"))
        tests=(ROOT/".cache/u36_s14_tests.log").read_text(encoding="utf8")
        (DATA/"u36_s14_tests.txt").write_text("\n".join(z.rstrip() for z in tests.splitlines())+"\n",encoding="utf8")
        match=re.search(r"(\d+) failed, (\d+) passed in ([\d.]+)s",tests);assert match
        write(DATA/"u36_s14_verification.json",dict(checkpoint="U36-S14",method_class="procedural_constrained",
            recalculations=verification,full_source_summary=full_extra,related_tests_failed=int(match[1]),related_tests_passed=int(match[2]),
            tests_elapsed_s=float(match[3]),reference_metrics_identical_except_timings=True,region_windows_identical=True,
            protected=full["protected"],global_or_B5_coupled=False,physical_width_identified=False,default_C2_unchanged=True,
            scheme_conservation_is_not_independent_accuracy_test=True,report_elapsed_s=time.perf_counter()-started))
        print("VERIFICATION",verification,flush=True);print("FULL",full["timeline"][-1],full_extra["FV_L1_worse_source_count"],flush=True)
    write(DATA/"u36_s14_manifest.json",dict(checkpoint="U36-S14",method_class="visual_only",selected_only=selected_only,figures=figs))
    page='<meta charset="utf-8"><title>U36-S14 移动材料均值</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S14：曲边移动材料体积账与厚度误差</h1><p>方法小试；默认C2，未接全球/B5，阶段1未通过。</p>'
    if selected_only:page+="<p>当前仅四例的三档结果；完整77来源仍运行，尚未提供结论。</p>"
    page+='<p><a href="../../../../stage1/u36_s14_reference/index.html">冻结世界29张标准图</a></p>'
    for f in figs:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/"index.html").write_text(page,encoding="utf8");print("FIGURES",len(figs),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selected-only",action="store_true");args=p.parse_args();main(args.selected_only)
