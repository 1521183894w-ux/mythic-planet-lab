"""U36-S3：运动学真值、网格假张开、供体尺度与真实历史支持度的专项图。"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from mythic.figures.style import setup_fonts
from mythic.tectonics.deformation_kinematics import affine_band
CACHE=ROOT/".cache/u36_s3"
DATA=ROOT/"data/worlds/design_v2"
OUT=ROOT/"figures/design/v2/forward/u36_s3"
STATUS={"opening_supported":("三尺度张开支持（仍待接触史）","#693f96"),
        "closing_supported":("三尺度闭合支持","#d17a21"),
        "shear_unresolved":("剪切下未分辨","#498eae"),
        "unresolved":("边缘/尺度不足，未定","#a2a2a2")}
NAMES={"opening":"解析张开","slip":"纯走滑","rigid":"整体刚体旋转"}
FIGURES=[]


def save(fig,name,title,note):
    fig.text(.012,.014,"U36-S3 · "+VERSION+" · 确定性试验；真实历史 design_v2/history_v2\n"
        +"运动学 procedural_constrained；边缘法向 interpolated；图 visual_only。长度 km / 面积 km² / 厚度 km / 时间 Myr\n"+note,
        fontsize=8,va="bottom")
    fig.savefig(OUT/name,dpi=130)
    plt.close(fig)
    FIGURES.append((name,title,note))


def main():
    global VERSION
    OUT.mkdir(parents=True,exist_ok=True)
    setup_fonts()
    VERSION=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()+"+U36-S3"
    syn=json.loads((DATA/"u36_s3_synthetic.json").read_text(encoding="utf-8"))
    actual=json.loads((DATA/"u36_s3_actual.json").read_text(encoding="utf-8"))
    verify=json.loads((DATA/"u36_s3_verification.json").read_text(encoding="utf-8"))
    endpoints=syn["endpoint"]
    fig,axs=plt.subplots(1,3,figsize=(14,9))
    colors=["#f0eee8","#6c5593","#d78538","#499ca3"]
    labels=["离散覆盖","解析张开与离散空隙重合","解析未张开却有空隙","解析张开被离散覆盖"]
    for ax,case in zip(axs,("opening","slip","rigid")):
        d=np.load(CACHE/f"synthetic_map_{case}.npz")
        ax.imshow(d["code"],origin="lower",extent=[d["x_km"][0],d["x_km"][-1],d["y_km"][0],d["y_km"][-1]],
                  cmap=ListedColormap(colors),vmin=-.5,vmax=3.5,interpolation="nearest",aspect="equal")
        ax.set(xlabel="跨初始边界 / km",ylabel="沿初始边界 / km",ylim=(-1500,1500),title=NAMES[case]+" · L7")
        ax.plot([-380,-180],[-1350,-1350],color="#222",lw=3)
        ax.text(-380,-1300,"200 km",fontsize=9)
        ax.grid(alpha=.2)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc="upper center",ncol=2,bbox_to_anchor=(.5,.98))
    fig.subplots_adjust(top=.84,bottom=.17,wspace=.4)
    save(fig,"76_analytic_gap_masks.png","解析边界与实际离散空隙","同一物理窗口、20 Myr、相对赤道速率10 km/Myr；采样密度只提高观察密度。纯走滑是光滑连续边界零张开对照，非拉分阶区。")
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for ax,case in zip(axs[0],("opening","slip")):
        for field,label,color in [("discrete_gap_area_km2","固定节点空隙","#db7d32"),("old_opening_candidate_area_km2","旧张开出生候选","#7a4d9b"),("independent","独立等面积积分空隙","#367b8b")]:
            hs,means,low,high=[],[],[],[]
            for level in (5,6,7):
                r=[x for x in endpoints if x["level"]==level and x["case"]==case]
                v=[x["independent_quadrature"][0]["gap_area_km2"] if field=="independent" else x[field] for x in r]
                hs.append(r[0]["h_km"]);means.append(np.mean(v));low.append(min(v));high.append(max(v))
            ax.plot(hs,np.array(means)/1e6,"o-",label=label,color=color)
            ax.fill_between(hs,np.array(low)/1e6,np.array(high)/1e6,color=color,alpha=.12)
        truth=next(x["exact_gap_area_km2"] for x in endpoints if x["case"]==case)
        ax.axhline(truth/1e6,color="black",ls="--",label="连续解析真值")
        ax.set(title=NAMES[case]+"：三边界相位范围",xlabel="平均网格边长 / km",ylabel="面积 / 百万 km²")
        ax.invert_xaxis();ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axs[1,0]
    for case,color in [("opening","#7a4d9b"),("slip","#db7d32"),("rigid","#367b8b")]:
        r=[x for x in endpoints if x["case"]==case and x["phase"]==0]
        ax.plot([x["h_km"] for x in r],[(x["independent_quadrature"][1]["gap_area_km2"]-x["independent_quadrature"][0]["gap_area_km2"])/1e3 for x in r],"o-",label=NAMES[case],color=color)
    ax.set(title="独立积分点 40万→80万（相位0）",xlabel="网格边长 / km",ylabel="积分面积变化 / 千 km²")
    ax.invert_xaxis();ax.grid(alpha=.2);ax.legend()
    ax=axs[1,1];ax.axis("off")
    text="固定节点与逐板块穷举覆盖不一致："+str(sum(x["accelerated_query_mismatches"] for x in endpoints))+" 个节点\n\n"
    text+="固定节点与独立积分不是同一观察口径。\nL5固定节点漏掉张开，不能当作没有伸展。\n\n"
    text+="三尺度边缘拟合不是连续真值：\n纯走滑被判为张开支持的候选面积\n"
    text+=f'{sum(x["normal_support_area_km2"].get("opening_supported",0.) for x in endpoints if x["case"]=="slip"):,.1f} km²\n\n'
    text+="三相位样本不足以证明全局阶数收敛。"
    ax.text(.02,.95,text,va="top",fontsize=12,linespacing=1.7)
    fig.subplots_adjust(top=.95,bottom=.16,hspace=.38,wspace=.28)
    save(fig,"77_gap_resolution.png","空隙与旧判据的空间分辨率、积分对照","线为三相位均值，色带为最小—最大；独立积分为全局等面积Fibonacci查询，40万点的同一查询序列用于全部网格。")
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for col,case in enumerate(("opening","slip")):
        for interval,color in [(1,"#bf6f28"),(5,"#7a4d9b"),(10,"#2e8493")]:
            rr=[r for r in syn["cadence"] if r["case"]==case and r["interval_myr"]==interval]
            h=[next(x["h_km"] for x in endpoints if x["level"]==r["level"]) for r in rr]
            axs[0,col].plot(h,[r["source_equivalent_width_median_km"] if r["transfer_count"] else np.nan for r in rr],"o-",color=color,label=f"每{interval} Myr提交")
            axs[1,col].plot(h,[r["final_min_cont_thickness_km"] for r in rr],"o-",color=color,label=f"每{interval} Myr提交")
        axs[0,col].set(title=NAMES[case]+"：原12近邻供体支撑",xlabel="网格边长 / km",ylabel="单次供体等效宽度中位 / km")
        axs[1,col].set(title=NAMES[case]+"：提交后最薄陆壳",xlabel="网格边长 / km",ylabel="仍标陆壳的最小厚度 / km")
        for row in (0,1):
            axs[row,col].invert_xaxis();axs[row,col].grid(alpha=.2);axs[row,col].legend(fontsize=9)
    fig.subplots_adjust(top=.95,bottom=.17,hspace=.35,wspace=.25)
    save(fig,"78_stencil_cadence.png","供体支撑与提交间隔的依赖","等效宽度=每次供体面积/πR，仅为比较算子支撑；slip的πR是统一比较分母，不是实际断层长。L5固定节点漏检，供体支撑记缺值，保留35km不代表无伸展。仅旋转与commit，未启用碰撞/流动；不是完整全球演化。")
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    opening=np.linspace(0,250,101)
    for width,color in [(100.,"#bb752d"),(200.,"#7e54a3"),(400.,"#348291")]:
        axs[0].plot(opening,35*width/(width+opening),label=f"参考带宽 W0={width:.0f} km",color=color)
        vals=[r for r in syn["band"] if r["width0_km"]==width]
        axs[0].scatter([200.]*len(vals),[r["thickness_final_km"] for r in vals],color=color,s=20)
    axs[0].set(title="同一200 km张开，带宽决定厚度",xlabel="累计法向张开 / km",ylabel="整柱平均厚度 / km")
    axs[0].grid(alpha=.2);axs[0].legend()
    ax=axs[1]
    p=np.array([[-100.,-100.],[100.,-100.],[100.,100.],[-100.,100.],[-100.,-100.]])
    for name,op,slip,angle,color in [("参考",0,0,0,"#999999"),("纯走滑 J=1",0,150,0,"#c27830"),("刚体旋转 J=1",0,0,.7,"#3a8c99"),("张开 J=2",200,0,0,"#7b4e9f")]:
        q=p@affine_band(200,op,slip,angle)["F"].T
        ax.plot(q[:,0],q[:,1],label=name,color=color)
    ax.set(aspect="equal",xlabel="x / km",ylabel="y / km",title="面积伸展与剪切、旋转分别记录")
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=.22,wspace=.3)
    save(fig,"79_physical_band_width.png","已给定物理带宽时的面积—厚度关系","W0=100/200/400 km均为敏感性输入，不是从地图反演所得；dx=40/20/10 km、dt=0.5/1/5 Myr的实际单元面积/时间积分重合。不含断层位移、热沉降、沉积或盆地门槛。")
    rows=np.load(CACHE/"actual_candidates.npz")
    xyz=rows["xyz"]
    lon=np.degrees(np.arctan2(xyz[:,1],xyz[:,0]));lat=np.degrees(np.arcsin(xyz[:,2]))
    fig=plt.figure(figsize=(15,10))
    gs=fig.add_gridspec(2,2,height_ratios=[1.15,1])
    ax=fig.add_subplot(gs[0,:])
    for key,(label,color) in STATUS.items():
        use=rows["status"]==key
        ax.scatter(lon[use],lat[use],s=7,c=color,label=label,alpha=.7)
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel="出生时经度 / °E",ylabel="出生时纬度 / °N",title="真实控制历史：原判据选中的出生候选（时间混合，非现今伸展地图）")
    ax.set_xticks(np.arange(-180,181,30));ax.set_yticks(np.arange(-90,91,30));ax.grid(alpha=.25)
    ax.legend(fontsize=9,loc="upper left",ncol=2)
    ax=fig.add_subplot(gs[1,0])
    pairs=sorted(actual["pair"],key=lambda k:actual["pair"][k]["area_km2"],reverse=True)
    bottom=np.zeros(len(pairs))
    for key,(label,color) in STATUS.items():
        vals=np.array([actual["pair"][pair]["status_area_km2"].get(key,0.)/1e6 for pair in pairs])
        ax.bar(np.arange(len(pairs)),vals,bottom=bottom,color=color,label=label);bottom+=vals
    ax.set(xticks=np.arange(len(pairs)),xticklabels=pairs,ylabel="候选出生面积累计 / 百万 km²",xlabel="板块对编码",title="支持度只作诊断，不能作减薄面积")
    ax.tick_params(axis="x",rotation=45);ax.grid(axis="y",alpha=.2)
    ax=fig.add_subplot(gs[1,1])
    for key,(label,color) in STATUS.items():
        use=rows["status"]==key
        ax.scatter(rows["width_proxy_km"][use],rows["stencil_radius_km"][use],s=7,c=color,alpha=.55)
    ax.set(xlabel="原方法两侧原生陆壳最近距离之和 / km",ylabel="原12近邻最大供体距离 / km",title="开口代理宽度与供体支撑是两种量")
    ax.grid(alpha=.2)
    fig.subplots_adjust(top=.94,bottom=.17,hspace=.4,wspace=.25)
    save(fig,"80_actual_gap_support.png","真实候选位置、法向支持度与旧供体支撑","L7约60 km；累计出生位置不能替代材料轨迹。边缘拟合尺度180/360/720 km是沿边界观察尺度，不是跨带W0；仍缺独立接触历史和物理变形带。")
    for name in ("metrics.json","manifest.json","regions.json"):
        src=ROOT/"figures/stage1/u36_s3_reference"/name
        if src.exists():
            shutil.copy2(src,DATA/("u36_s3_reference_"+name))
    source_paths=["packages/tectonics/deformation_kinematics.py","packages/tectonics/gap_diagnostics.py","scripts/design_gap_kinematics.py","scripts/design_gap_actual.py","scripts/design_gap_kinematics_report.py"]
    band_errors=dict(
        max_polygon_area_error_km2=max(abs(r["polygon_area_km2"]-(r["width0_km"]+200)*1000) for r in syn["band"]),
        max_volume_per_length_error_km2=max(abs(r["volume_per_length_km2"]-r["width0_km"]*35) for r in syn["band"]))
    assert band_errors["max_polygon_area_error_km2"] < 1e-8
    assert band_errors["max_volume_per_length_error_km2"] < 1e-8
    manifest=dict(version=VERSION,seed="design_v2/history_v2；解析试验相位0/1/2，无随机种子",
        source_hash_normalization="UTF-8 bytes, CRLF normalized to LF",
        sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest() for p in source_paths},
        layers=[
            dict(name="解析半球",method_class="procedural_constrained",resolution="连续几何；L5/L6/L7离散覆盖；40万/80万独立积分",units="km²",assumptions="刚性半球，光滑边界，解析缺口=2R²α；非全球地质模型"),
            dict(name="给定带宽的有限变形",method_class="procedural_constrained",resolution="实际单元dx40/20/10 km，dt0.5/1/5 Myr",units="W/H km,J无量纲",assumptions="均匀参考厚35km，无沿带散度、无物质源、无断层垂向位移"),
            dict(name="实际历史边缘法向",method_class="interpolated",resolution="L7约60km；拟合180/360/720km",units="km/Myr",assumptions="PCA两侧独立边缘；三尺度支持不是物理真值；诊断余量不是置信区间"),
            dict(name="真实控制历史",method_class="procedural_constrained",resolution="L7；输出1024x512",units="km,km²,km³,Ma",assumptions="既有规定历史user_override，Airy/GDH1 parameterized_physics；仅旁观，未修改地壳")
        ],
        references=["https://www.gplates.org/docs/pygplates/generated/pygplates.StrainRate.html","https://www.gplates.org/docs/pygplates/sample-code/pygplates_reconstruct_crustal_thickness_and_tectonic_subsidence"],
        verification=verify,analytic_validation=band_errors,figures=[dict(file=x[0],title=x[1],note=x[2]) for x in FIGURES])
    (DATA/"u36_s3_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    blocks=["<!doctype html><meta charset='utf-8'><title>U36-S3 空隙与真实伸展</title><style>body{max-width:1500px;margin:30px auto;background:#f4f2ed;color:#252525;font:17px sans-serif}img{width:100%;background:white}section{margin:40px 0}</style>",
            "<h1>U36-S3 · 空隙、法向张开与物理带宽</h1><p>专项验证；真实历史只读观测。三尺度支持不是实际伸展真值，W0尚未由独立几何给定；未接入B5沉降，默认仍C2。</p>",
            "<p><a href='../../../../../stage1/u36_s3_reference/index.html'>本轮重出的控制世界标准图组</a></p>"]
    # 从 forward/u36_s3 回到 figures 需4层。
    blocks[-1]=blocks[-1].replace("../../../../../stage1","../../../../stage1")
    for name,title,note in FIGURES:
        blocks.append("<section><h2>"+html.escape(title)+"</h2><p>"+html.escape(note)+"</p><img src='"+name+"'></section>")
    (OUT/"index.html").write_text("\n".join(blocks),encoding="utf-8")
    print("U36-S3 figures",len(FIGURES),flush=True)


if __name__=="__main__":
    main()
