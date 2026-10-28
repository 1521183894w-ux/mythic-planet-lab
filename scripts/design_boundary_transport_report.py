"""S11实际保存数据的专题图，不把条件性柱厚当高程。"""
from pathlib import Path
import json,html,subprocess,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import xyz_to_lonlat,R_KM
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s11'
OUT=ROOT/'figures/design/v2/forward/u36_s11'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def row(scope,level):return read(DATA/f'u36_s11_{scope}_e30_l{level}_dt0.25_a0.json')
def main():
    setup_fonts();OUT.mkdir(parents=True,exist_ok=True)
    ver=read(DATA/'u36_s11_verification.json');full=[row('full',l) for l in range(3)];outer=[row('outer',l) for l in range(3)]
    version=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()+'+U36-S11'
    figs=[dict(file='112_initial_boundary_support.png',title='起点诊断：换投影不能单独修复外圈')]
    def save(fig,name,title,note):
        fig.text(.015,.018,version+' · design_v2；原物理输入L7约60km，细分仅为材料审计\n'
            +'指定200km宽度 user_override；点映射 interpolated；来源与几何审计 procedural_constrained；图 visual_only\n'+note,fontsize=8,va='bottom')
        fig.savefig(OUT/name,dpi=145);plt.close(fig);figs.append(dict(file=name,title=title,note=note))
    a=np.load(CACHE/'full_e30_l2_dt0.25_a0.npz');points=a['points'];J=a['J'];tri=a['triangles'];nv=len(a['vertices'])
    fig,axes=plt.subplots(1,3,figsize=(15,9))
    for ax,t in zip(axes,[35,20,0]):
        i=round((35-t)/.25);lo,la=xyz_to_lonlat(points[i,:nv])
        ax.tripcolor(lo,la,tri,facecolors=np.zeros(len(tri)),cmap=matplotlib.colors.ListedColormap(['.8']))
        im=ax.tripcolor(lo,la,tri,facecolors=np.ma.masked_invalid(a['H0_km']/J[i]),vmin=0,vmax=50,cmap='viridis')
        ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=f'{t}Ma：完整原域条件性柱厚')
        ax.set_facecolor('white');ax.set_aspect(1/np.cos(np.radians(la.mean())));ax.grid(alpha=.2)
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.05*(x1-x0);y=y0+.05*(y1-y0)
        dx=100/(R_KM*np.pi/180*np.cos(np.radians(la.mean())))
        ax.plot([x,x+dx],[y,y],c='black',lw=2);ax.text(x,y+.025*(y1-y0),'100km',fontsize=8)
    fig.subplots_adjust(bottom=.2,right=.9,wspace=.4);fig.colorbar(im,cax=fig.add_axes([.93,.27,.013,.5]),label='整柱厚度 / km')
    save(fig,'113_full_domain_thickness.png','完整来源域与条件性柱厚','包含原外圈；灰色为原非陆壳/无覆盖。没有补值；此图不是盆底、地表、水深或新海岸。')
    fig,axes=plt.subplots(1,3,figsize=(16,8));colors=['#b06b42','#39897d','#586ca6']
    for q,c in zip(full,colors):
        axes[0].plot([z['t_ma'] for z in q['timeline']],[z['volume_relative_residual']*100 for z in q['timeline']],c=c,label=f"{q['triangle_count']}单元")
    axes[0].set(xlabel='年代 / Ma',ylabel='净体积差 / %',xlim=(35,0),title='完整原域：不能只看正负抵消后的净差');axes[0].legend(fontsize=8)
    for qs,name in [(full,'完整域'),(outer,'原外圈')]:
        axes[1].semilogy([1,4,16],[q['timeline'][-1]['weighted_source_absolute_relative_error']*100 for q in qs],'o-',label=name)
        axes[2].semilogy([1,4,16],[abs(q['timeline'][-1]['worst_source_relative_residual'])*100 for q in qs],'o-',label=name)
    axes[1].set(xlabel='每个原有限单元的细分数',ylabel='来源加权绝对差 / %',title='保持每一来源域，真正逐单元细分')
    axes[2].set(xlabel='每个原有限单元的细分数',ylabel='最差来源末态绝对差 / %',title='全域与外圈都保留最差来源')
    for ax in axes:ax.grid(alpha=.2)
    for ax in axes[1:]:ax.set_xticks([1,4,16]);ax.legend()
    fig.subplots_adjust(bottom=.23,wspace=.38)
    save(fig,'114_boundary_refinement.png','完整域和外圈的实际细分结果','有限球面三角形面积乘H0/解析累计J独立求积；不逐UID归一化。15个来源末态误差不单调，不能声称逐源收敛。')
    fig,axes=plt.subplots(1,3,figsize=(16,8))
    for q,c in zip(full,colors):
        s=q['source_summary'];axes[0].scatter([z['initial_volume_km3'] for z in s],[z['final_relative_residual']*100 for z in s],s=15,c=c,label=f"每单元{4**q['uniform_refinement_levels']}份")
    axes[0].set(xscale='log',xlabel='来源初始体积 / km³',ylabel='来源末态体积差 / %',title='全部已知来源逐项保留');axes[0].legend(fontsize=8)
    for ax,u in zip(axes[1:],[448261,79872]):
        for q,c in zip(full,colors):
            b=np.load(CACHE/(q['tag']+'.npz'));k=np.flatnonzero(b['uids']==u)[0]
            ax.plot(b['times_ma'],100*(b['source_volumes_km3'][:,k]/b['V0_km3'][k]-1),c=c,label=f"每单元{4**q['uniform_refinement_levels']}份")
        ax.set(xlim=(35,0),xlabel='年代 / Ma',ylabel='来源体积差 / %',title=f'UID{u}');ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.4)
    save(fig,'115_source_errors.png','旧问题来源与全来源误差','输入插值场已改变，不能与S10同档误差冒称同一物理问题的精度排名；微斑块不再参与正式柱厚计算。')
    fig,axes=plt.subplots(1,3,figsize=(16,8));comp=ver['time_and_frame_comparisons'];labels=['dt减半','换anchor301']
    axes[0].bar(labels,[z['maximum_final_position_difference_m'] for z in comp],color=colors[:2]);axes[0].set(yscale='log',ylabel='末态最大位置差 / m',title='同一全部材料点，不挑框架')
    axes[1].bar(labels,[z['maximum_final_thickness_relative_difference']*100 for z in comp],color=colors[:2]);axes[1].set(yscale='log',ylabel='末态最大柱厚相对差 / %',title='只有两档步长，不声称时间收敛阶')
    for radius,c in [(.1,colors[0]),(.01,colors[1])]:
        rr=[z for z in ver['analytic_derivative_checks'] if z['patch_radius_km']==radius]
        axes[2].semilogy([z['t_ma'] for z in rr],[z['maximum_relative_difference'] for z in rr],'o-',c=c,label=f'{radius*1000:g}m斑块')
    axes[2].set(xlabel='年代 / Ma',ylabel='与解析单步J的最大相对差',title='独立球面面积核对：小斑块有浮点误差底');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.42)
    save(fig,'116_frame_and_derivative.png','参照系、步长和解析导数独立核对','35/20/5Ma各16个混合面内部点；100m与10m斑块结果全部保留。解析导数验证的是指定插值场，不是物理真值。')
    for f in figs:f['sha256']=hashlib.sha256((OUT/f['file']).read_bytes()).hexdigest()
    manifest=dict(checkpoint='U36-S11',method_class='visual_only',version=version,figures=figs)
    (DATA/'u36_s11_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    page='<meta charset="utf-8"><title>U36-S11 完整边界</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S11：完整材料边界与解析面积导数</h1><p>局部方法小试，未接全球/B5，阶段1未通过。</p>'
    for f in figs:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f.get("note",""))}</p>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('FIGURES',len(figs),OUT/'index.html',flush=True)
if __name__=='__main__':main()
