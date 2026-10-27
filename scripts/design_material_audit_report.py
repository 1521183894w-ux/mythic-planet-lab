"""U36-S5：时间窗口误差分解和原生材料来源裁切的专项出图。"""
from __future__ import annotations
import html
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import xyz_to_lonlat,R_KM

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s5';OUT=ROOT/'figures/design/v2/forward/u36_s5'
FIGURES=[]
LABELS={'error_scalar_integral':'标量时间积分','error_spatial_rate':'应变率与速度梯度','error_temporal_window':'过去1Myr与短窗口','error_velocity_map':'短窗口积分与材料面积'}
COLORS=['#8a8a8a','#bc8148','#7654a4','#34949b']


def save(fig,name,title,note):
    fig.text(.012,.012,'U36-S5 · '+VERSION+' · design_v2/history_v2 · pyGPlates1.0.0 / Shapely2.2.0\n'
        +'给定参考带 user_override；框架速度/应变 interpolated；独立散度与来源裁切 procedural_constrained\n'+note,
        fontsize=8,va='bottom')
    fig.savefig(OUT/name,dpi=140);plt.close(fig);FIGURES.append(dict(file=name,title=title,note=note))


def geographic(ax,xyz,title):
    lo,la=xyz_to_lonlat(xyz);ax.set(title=title,xlabel='经度 / °E',ylabel='纬度 / °N')
    ax.set_aspect(1/np.cos(np.radians(la.mean())));ax.grid(alpha=.2)
    return lo,la


def scale(ax,lat):
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.05*(x1-x0);y=y0+.05*(y1-y0)
    dx=100/(R_KM*np.pi/180*np.cos(np.radians(lat)))
    ax.plot([x,x+dx],[y,y],color='#222',lw=3);ax.text(x,y+.02*(y1-y0),'100 km',fontsize=8)


def main():
    global VERSION
    OUT.mkdir(parents=True,exist_ok=True);setup_fonts()
    VERSION=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S5'
    d=json.loads((DATA/'u36_s5_trajectory.json').read_text(encoding='utf-8'))
    source=json.loads((DATA/'u36_s5_sources.json').read_text(encoding='utf-8'))
    a=np.load(CACHE/'trajectory.npz');old=np.load(ROOT/'.cache/u36_s4/material_band.npz');weights=old['area0_km2'];mean=lambda x:np.average(x,axis=1,weights=weights)
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for key,label,color in [('strain_rate','框架应变率（过去1Myr）','#bc8148'),('past_divergence','速度散度（过去1Myr）','#7654a4'),('instant_divergence','短窗口速度散度','#34949b')]:
        axs[0,0].plot(a['times_ma'],mean(a[key]),label=label,color=color)
    axs[0,0].set(xlim=(35,33),xlabel='年代 / Ma',ylabel='面积变化率 / 1/Myr',title='真实参考段：启动后首2Myr');axs[0,0].legend(fontsize=8)
    for val,label,color in [(np.log(a['J']),'小材料斑块logJ','#222'),(np.log(a['beta']),'框架logβ','#bc8148'),(a['integral_instant_divergence'],'短窗口散度积分','#34949b')]:
        axs[0,1].plot(a['times_ma'],mean(val),label=label,color=color)
    axs[0,1].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='面积对数变化 / 无量纲',title='同一材料中心，35→0Ma');axs[0,1].legend(fontsize=8)
    for k,c in zip(LABELS,COLORS):axs[1,0].plot(a['times_ma'],100*mean(a[k]),color=c,label=LABELS[k])
    axs[1,0].plot(a['times_ma'],100*mean(a['error_total']),'k--',label='总log(β/J)')
    axs[1,0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='对数差 ×100',title='同一权重下可加的四项分解');axs[1,0].legend(fontsize=8)
    ks=list(LABELS);vals=[100*d['window_matched_diagnostic']['weighted_mean_log_components'][k] for k in ks]
    axs[1,1].barh(range(4),vals,color=COLORS);axs[1,1].set_yticks(range(4),[LABELS[k] for k in ks],fontsize=9)
    axs[1,1].axvline(0,color='black',lw=.7);axs[1,1].set(xlabel='初始面积加权对数差 ×100',title='0Ma误差来源：时间滞后占主导')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.16,hspace=.35,wspace=.55,top=.93)
    save(fig,'86_window_error_decomposition.png','同轨迹误差分解','散度为同一框架速度的独立球面差分；四项相加是代数分解，不是四个物理过程。过去1Myr窗口在35Ma启动点存在滞后。')
    fig,axs=plt.subplots(1,3,figsize=(14,8))
    names={'constant':'恒速（覆盖0–50Ma）','step':'35Ma突然启动','ramp':'35→33Ma两级启动'}
    for ax,case in zip(axs,['constant','step','ramp']):
        r=np.load(CACHE/f'control_{case}.npz')
        for key,label,c in [('error_total','log(β/J)','#222'),('error_temporal_window','时间窗口部分','#7654a4'),('error_spatial_rate','空间梯度部分','#bc8148'),('error_velocity_map','短窗积分与位置面积','#34949b'),('error_scalar_integral','标量时间积分','#8a8a8a')]:ax.plot(r['times_ma'],100*np.mean(r[key],axis=1),label=label,color=c)
        ax.axhline(0,color='grey',lw=.6);ax.set(xlim=(35,0),title=names[case],xlabel='年代 / Ma',ylabel='对数差 ×100');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=.21,wspace=.32,top=.9)
    save(fig,'87_onset_history_controls.png','独立启动历史对照','历史覆盖0–50Ma、1Myr关键帧；两级启动为25%→75%→满速。它还暴露较大的短窗积分/点运动差，不能把真实段0.024%推广给所有时变历史；不为指标改启动。')
    fig,axs=plt.subplots(1,3,figsize=(14,8))
    for radius,c in [(1.,'#7654a4'),(.1,'#34949b')]:
        rows=[r for r in d['sensitivity'] if r['patch_km']==radius]
        axs[0].plot([r['dt_myr'] for r in rows],[100*r['summary']['timeline'][-1]['error_total'][2] for r in rows],'o-',color=c,label=f'斑块半径{radius} km')
    axs[0].set(xlabel='积分步长 / Myr',ylabel='末态log(β/J)中位 ×100',title='缩小斑块/时间步仍有残差');axs[0].legend(fontsize=8)
    rows=d['spatial_difference']
    axs[1].semilogx([r['probe_km'] for r in rows],[100*r['summary']['timeline'][-1]['error_velocity_map'][2] for r in rows],'o-',color='#34949b')
    axs[1].set(xlabel='速度空间差分距离 / km',ylabel='短窗积分−logJ中位 ×100',title='独立速度散度的距离敏感性')
    q=d['window_matched_diagnostic'];vs=[q['original_cell_volume_by_framework_relative_residual']*100,q['original_cell_volume_by_short_window_divergence_relative_residual']*100]
    bars=axs[2].bar([0,1],vs,color=['#bc8148','#34949b']);axs[2].set_xticks([0,1],['原框架β','匹配短窗口散度'])
    for b,val in zip(bars,vs):axs[2].text(b.get_x()+b.get_width()/2,val+.02,f'{val:.4f}%',ha='center')
    axs[2].set(ylabel='同一原材料单元体积偏差 / %',title='诊断比较：1.29%→约0.024%')
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22,wspace=.4,top=.89)
    save(fig,'88_difference_and_volume_diagnostics.png','差分敏感性与同单元体积对照','短窗口散度积分仅作诊断，未覆盖框架β或接入全球厚度；仍有离散插值/位置积分差，不声称严格一致或真实流变已求解。')
    with (CACHE/'clipped_sources.pkl').open('rb') as f:p=pickle.load(f)
    material=np.load(CACHE/'clipped_material.npz');tri=material['triangles'];v=material['vertices']
    fig,axs=plt.subplots(1,3,figsize=(14,9))
    lo,la=geographic(axs[0],v,'35Ma：按实际宿主多边形裁切')
    uid=material['source_uid'];plate=material['source_plate'];h0=material['H0_km']
    code=np.where(uid<0,3,np.where(np.isfinite(h0),np.where(plate==301,0,1),2))
    pc=axs[0].tripcolor(lo,la,tri,facecolors=code,cmap=ListedColormap(['#bc8148','#4e90ac','#2b4b76','#bcbcbc']),vmin=0,vmax=3)
    fig.colorbar(pc,ax=axs[0],ticks=[0,1,2,3],shrink=.55).ax.set_yticklabels(['C陆壳','D陆壳','非陆壳','无覆盖'])
    scale(axs[0],la.mean())
    s=source['summary'];keys=['known_cont_area_km2','noncont_area_km2','uncovered_area_km2']
    axs[1].bar(range(3),[s[k]/s['target_area_km2']*100 for k in keys],color=['#bc8148','#2b4b76','#bcbcbc'])
    axs[1].set_xticks(range(3),['已知陆壳','非陆壳','无覆盖']);axs[1].set(title='源面积各自记账',ylabel='占参考带面积 / %',ylim=(0,100));axs[1].grid(alpha=.2)
    rows=sorted([q for q in source['source_hosts'] if q['volume_km3'] is not None],key=lambda q:q['volume_km3'],reverse=True)
    axs[2].plot(np.arange(1,len(rows)+1),np.cumsum([r['volume_km3'] for r in rows])/s['known_cont_volume_km3']*100,color='#7654a4')
    axs[2].set(title='原宿主贡献可追溯至UID',xlabel='按贡献排序的陆壳宿主',ylabel='累计已知体积 / %',ylim=(0,105));axs[2].grid(alpha=.2)
    fig.subplots_adjust(bottom=.2,wspace=.48,top=.9)
    save(fig,'89_native_source_partition.png','原生宿主来源的球面裁切','16702次独立查询与UID分区全一致；非陆壳与无覆盖不补陆壳。物理面积回球面计算，平面心射投影只用于GEOS布尔运算。')
    fig,axs=plt.subplots(1,3,figsize=(14,9))
    for ax,time in zip(axs[:2],[35,0]):
        xyz=material[f'xyz_{time}Ma'];lo,la=geographic(ax,xyz,f'{time}Ma：原宿主材料整柱厚度')
        val=material['H0_km']/material[f'J_{time}Ma'];ax.set_facecolor('#bcbcbc')
        pc=ax.tripcolor(lo,la,tri,facecolors=np.ma.masked_invalid(val),cmap='viridis',vmin=0,vmax=50)
        fig.colorbar(pc,ax=ax,shrink=.55,label='整柱厚度 / km');scale(ax,la.mean())
    rows=source['timeline'];v0=source['triangle_known_volume_km3']
    axs[2].plot([r['t_ma'] for r in rows],[r['known_volume_by_J_km3']/v0 for r in rows],'o-',label='几何J对应柱体积',color='#34949b')
    axs[2].plot([r['t_ma'] for r in rows],[r['known_volume_by_beta_km3']/v0 for r in rows],'o-',label='原框架β对应柱体积',color='#bc8148')
    axs[2].set(xlim=(35,0),title='相同来源下仍能看见β偏差',xlabel='年代 / Ma',ylabel='已知陆壳体积/初始体积');axs[2].grid(alpha=.2);axs[2].legend(fontsize=8)
    fig.subplots_adjust(bottom=.2,wspace=.55,top=.9)
    save(fig,'90_clipped_material_columns.png','裁切后的固定材料与缺值','H0/J闭合是定义性材料账，不是独立物理解；不是高程/B5盆底。86个浮点碎片合计约0.0183m²不求J，保留在原裁切面积账；五时点无翻转/中心越界。')
    manifest=dict(checkpoint='U36-S5',version=VERSION,figures=FIGURES,
        standard_figures='figures/stage1/u36_s5_reference/index.html',physical_engine_changed=False,
        limitations=['时间窗口滞后已定位，空间梯度与轨迹差仍存在','给定参考线/带宽不代表唯一真实断层','无B5新地貌、无全球物理替换'])
    (DATA/'u36_s5_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>U36-S5 运动学与来源审计</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;background:#f5f3ee;color:#242424}img{width:100%}p{max-width:1100px}</style><h1>U36-S5：时间窗口与原生材料来源</h1><p>同轨迹分解和实际来源裁切；没有新全球地形或B5盆底。</p>'
    for f in FIGURES:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8');print('figures',len(FIGURES),flush=True)


if __name__=='__main__':main()
