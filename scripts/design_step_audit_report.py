"""U36-S6：输出可审看的运动学与材料连续性对照图。"""
from pathlib import Path
import html,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import xyz_to_lonlat,R_KM
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s6';OUT=ROOT/'figures/design/v2/forward/u36_s6'
FIGURES=[]

def save(fig,name,title,note):
    fig.text(.012,.012,f'U36-S6 · {VERSION} · design_v2/history_v2 · pyGPlates1.0.0\n给定参考几何 user_override；框架场 interpolated；映射/柱连续性审计 procedural_constrained\n'+note,fontsize=8,va='bottom')
    fig.savefig(OUT/name,dpi=140);plt.close(fig);FIGURES.append(dict(file=name,title=title,note=note))

def axes_grid(axs):
    for ax in np.asarray(axs).flat:ax.grid(alpha=.2)

def main():
    global VERSION
    setup_fonts();OUT.mkdir(parents=True,exist_ok=True)
    VERSION=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S6'
    d=json.loads((DATA/'u36_s6_trajectory.json').read_text(encoding='utf-8'))
    s=json.loads((DATA/'u36_s6_sources.json').read_text(encoding='utf-8'))
    fine=json.loads((DATA/'u36_s6_sources_refined.json').read_text(encoding='utf-8'))
    fig,axs=plt.subplots(1,3,figsize=(15,8));a=d['angle_sweep'];ang=np.array([q['angle_rad'] for q in a])
    axs[0].semilogx(ang,[q['api_speed_km_myr']/q['analytic_speed_km_myr'] for q in a],'o-',label='速度接口 / 解析值',ms=3,color='#ac6850')
    axs[0].semilogx(ang,[q['map_speed_km_myr']/q['analytic_speed_km_myr'] for q in a],label='点位移 / 解析值',color='#277d8e')
    axs[0].set(xlabel='给定旋转角 / rad',ylabel='速度比 / 无量纲',title='赤道已知刚体旋转，窗口1Myr');axs[0].legend(fontsize=8)
    w=d['window_sweep'];wins=[q['window_myr'] for q in w]
    axs[1].semilogx(wins,[q['api_median_km_myr'] for q in w],'o-',label='框架速度接口',color='#ac6850')
    axs[1].semilogx(wins,[q['map_median_km_myr'] for q in w],'o-',label='同一网络点位移率',color='#277d8e')
    axs[1].set(xlabel='速度窗口 / Myr',ylabel='样点速度中位 / km/Myr',title='两级启动：34.5Ma，13中心');axs[1].legend(fontsize=8)
    axs[2].semilogx(wins,[q['zero_api_count'] for q in w],'o-',color='#ac6850');axs[2].set(xlabel='速度窗口 / Myr',ylabel='返回零速度的材料中心 / 个',title='本例0.01Myr及以下返回零',ylim=(-.5,13.5))
    axes_grid(axs);fig.subplots_adjust(bottom=.22,wspace=.34,top=.9)
    save(fig,'91_small_rotation_velocity_floor.png','短窗口小角度截断','1e−6rad旋转仍移动6.371m，represents_identity_rotation为True、速度为零。扫描只给出本机1.0.0实测范围，不声称通用硬阈值。')
    fig,axs=plt.subplots(1,3,figsize=(15,8));r=np.load(CACHE/'ramp_0.125.npz')
    for key,label,c in [('api_short','0.01Myr速度接口','#ac6850'),('api_matched','匹配dt窗口速度接口','#8b6caf'),('map_short','0.01Myr点映射微分','#277d8e')]:
        axs[0].plot(r['times_ma'],100*np.mean(r[key+'_error'],axis=1),label=label,color=c)
    axs[0].set(xlabel='年代 / Ma',ylabel='散度积分 − logJ，均值 ×100',title='两级启动：事件对齐中点求积',xlim=(35,0));axs[0].legend(fontsize=8)
    for case,label in [('constant','恒速'),('step','突启'),('ramp','两级启动'),('turning','变向+走滑'),('rigid','共同刚体')]:
        rows=[q for q in d['controls'] if q['case']==case]
        axs[1].loglog([q['dt_myr'] for q in rows],[max(abs(q['summary']['timeline'][-1]['map_short_error_weighted_mean'])*100,1e-10) for q in rows],'o-',label=label)
    axs[1].set(xlabel='中点求积步长 / Myr',ylabel='末态均值绝对对数差 ×100',title='点映射场与材料面积的步长敏感性');axs[1].legend(fontsize=8)
    rigid=np.load(CACHE/'rigid_0.125.npz');axs[2].plot(rigid['times_ma'],100*(np.mean(rigid['J'],axis=1)-1),color='#ac6850',label='框架网络点映射')
    axs[2].axhline(0,color='#277d8e',label='同输入精确刚体旋转');axs[2].set(xlabel='年代 / Ma',ylabel='材料斑块面积变化 / %',title='反例：共同刚体仍有点映射失真',xlim=(35,0));axs[2].legend(fontsize=8)
    axes_grid(axs);fig.subplots_adjust(bottom=.23,wspace=.36,top=.9)
    save(fig,'92_event_aligned_map_consistency.png','时变运动、步长与刚体反例','所有速度/映射积分共享材料中点。map微分随dt细化接近同映射面积，但共同刚体的面积仍非严格常量；这不是独立物理正确性证明。')
    fig,axs=plt.subplots(1,3,figsize=(15,8));r=np.load(CACHE/'real.npz');old=np.load(ROOT/'.cache/u36_s4/material_band.npz');weights=old['area0_km2'];mean=lambda z:np.average(z,axis=1,weights=weights)
    for key,label,c in [('api_short','短窗口速度接口','#ac6850'),('api_matched','匹配dt速度接口','#8b6caf'),('map_short','点映射微分','#277d8e')]:
        axs[0].plot(r['times_ma'],100*mean(r[key+'_error']),label=label,color=c)
    axs[0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='散度积分 − logJ，加权均值 ×100',title='真实给定参考段：432中心');axs[0].legend(fontsize=8)
    for key,label,c in [('api_short','短窗口API','#ac6850'),('map_short','点映射微分','#277d8e')]:
        axs[1].hist(100*r[key+'_error'][-1],bins=30,weights=weights/weights.sum(),histtype='step',label=label,color=c,lw=1.6)
    axs[1].set(xlabel='末态局部对数差 ×100',ylabel='初始材料面积份额',title='平均接近零仍可能局部异号');axs[1].legend(fontsize=8)
    axs[2].axis('off');end=d['real']['timeline'][-1];lines=['同一真实材料段，dt=0.25Myr','点运动内部步长=0.125Myr','空间探针=0.1km，斑块半径=1km','',f"API短窗误差均值×100：{100*end['api_short_error_weighted_mean']:.6f}",f"匹配窗口误差均值×100：{100*end['api_matched_error_weighted_mean']:.6f}",f"点映射微分误差均值×100：{100*end['map_short_error_weighted_mean']:.6f}",'','两级启动的5个单步对照：','reconstruct_point 与 span 位移相同','只核查指定时刻/样点，不外推全局','框架β及真实物理带宽均未改']
    axs[2].text(.03,.95,'\n'.join(lines),va='top',fontsize=11)
    axes_grid(axs[:2]);fig.subplots_adjust(bottom=.23,wspace=.35,top=.9)
    save(fig,'93_real_segment_map_rates.png','真实参考段的同点映射微分','不直接把S5端点梯形求积与本轮中点结果相减归因。新结果采用事件对齐中点；误差是同一给定点映射内部的一致性。')
    fig,axs=plt.subplots(1,3,figsize=(15,8))
    for item,label,c in [(s,'最长边15km','#ac6850'),(fine,'最长边7.5km','#277d8e')]:
        rows=item['timeline'];t=[q['t_ma'] for q in rows]
        axs[0].plot(t,[100*q['volume_relative_residual'] for q in rows],label=label,color=c)
    axs[0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='有效材料柱体积残差 / %',title='固定全时段有效子集');axs[0].legend(fontsize=8)
    axs[1].plot([q['t_ma'] for q in s['timeline']],[100*q['framework_volume_relative_residual'] for q in s['timeline']],label='原框架β',color='#8b6caf')
    for item,label,c in [(s,'独立中心斑块，15km','#ac6850'),(fine,'独立中心斑块，7.5km','#277d8e')]:axs[1].plot([q['t_ma'] for q in item['timeline']],[100*q['volume_relative_residual'] for q in item['timeline']],label=label,color=c)
    axs[1].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='同来源柱体积残差 / %',title='无总量重归一化、无跨源调平');axs[1].legend(fontsize=8)
    for item,label,c in [(s,'15km','#ac6850'),(fine,'7.5km','#277d8e')]:
        rows=item['timeline'][-1]['source_volumes'];xx=np.array([q['initial_volume_km3'] for q in rows]);yy=np.array([q['volume_km3']/q['initial_volume_km3']-1 for q in rows])*100
        axs[2].scatter(xx,yy,label=label,s=12,color=c,alpha=.7)
    axs[2].set_xscale('log');axs[2].set(xlabel='各来源初始体积 / km³',ylabel='各来源体积残差 / %',title='各来源的有效材料子集');axs[2].legend(fontsize=8)
    axes_grid(axs);fig.subplots_adjust(bottom=.23,wspace=.36,top=.9)
    save(fig,'94_source_continuity_convergence.png','裁切材料的独立柱连续性','H=H0/Jac_center；141时点任一中心/环点越出网络的材料均不求厚度。两档有效子集略异，不能单凭两点下降宣称固定域收敛。')
    a=np.load(CACHE/'local_material_refined.npz');tri=a['triangles'];fig,axs=plt.subplots(1,3,figsize=(15,9))
    for ax,t in zip(axs,[35,20,0]):
        lo,la=xyz_to_lonlat(a[f'xyz_{t}Ma']);h=a[f'H_{t}Ma'];masked=np.ma.masked_invalid(h)
        ax.set_facecolor('#d2d2d2');im=ax.tripcolor(lo,la,tri,facecolors=masked,vmin=0,vmax=50,cmap='viridis',shading='flat')
        ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=f'{t}Ma：给定材料整柱厚度');ax.set_aspect(1/np.cos(np.radians(np.mean(la))));ax.grid(alpha=.2)
        xl=ax.get_xlim();yl=ax.get_ylim();x0=xl[0]+.05*(xl[1]-xl[0]);y0=yl[0]+.05*(yl[1]-yl[0]);dx=100/(R_KM*np.pi/180*np.cos(np.radians(np.mean(la))))
        ax.plot([x0,x0+dx],[y0,y0],lw=3,color='#111');ax.text(x0,y0+.02*(yl[1]-yl[0]),'100km',fontsize=8)
    fig.subplots_adjust(bottom=.19,top=.9,wspace=.32,right=.85);cax=fig.add_axes([.9,.27,.015,.45]);fig.colorbar(im,cax=cax,label='整柱地壳厚度 / km')
    save(fig,'95_local_material_thickness.png','局部材料连续性示意','灰色：非陆壳/无覆盖，以及审计斑块越界的已知边缘材料（另留初始库存）。不是高程/B5；7.5km为审计细分，物理输入L7、给定W0=200km。')
    manifest=dict(checkpoint='U36-S6',version=VERSION,figures=FIGURES,standard_figures='figures/stage1/u36_s6_reference/index.html',
        physical_engine_changed=False,default_world_adopted=False,limitations=['小角度接口截断与时序求积分别核查','共同刚体严格测试失败仍保留','物理宽度/古接触及B5尚未求解'])
    (DATA/'u36_s6_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>U36-S6 单步映射审计</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;background:#f5f3ee;color:#222}img{width:100%}p{max-width:1150px}</style><h1>U36-S6：小角度截断、时序一致性与局部连续性</h1><p>方法检查点。保留严格失败；没有新的全球演化或B5地貌。</p><p><a href="../../../../stage1/u36_s6_reference/index.html">标准图组</a></p>'
    for f in FIGURES:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8');print('figures',len(FIGURES),flush=True)
if __name__=='__main__':main()
