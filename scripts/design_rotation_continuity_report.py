"""U36-S10：只从真实审计输出绘图；条件性柱厚度不称地形。"""
from pathlib import Path
import json,html,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import xyz_to_lonlat,R_KM
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s10'
OUT=ROOT/'figures/design/v2/forward/u36_s10'
def read(name):return json.loads((DATA/name).read_text(encoding='utf8'))
def col(q,uid):return next(z for z in q['source_trajectories'] if z['uid']==uid)
def source_summary(q):
    cols=q['source_trajectories'];v0=np.array([z['initial_volume_km3'] for z in cols])
    v=np.array([z['volume_km3'] for z in cols]);j,k=np.unravel_index(np.argmax(abs(v/v0[:,None]-1)),v.shape)
    return dict(tag=q['tag'],complete=q['complete_35_to_0Ma'],final_net=q['timeline'][-1]['volume_relative_residual'],
        final_source_weighted_absolute=float(np.sum(abs(v[:,-1]-v0))/v0.sum()),
        worst_source_uid=cols[j]['uid'],worst_source_error=float(v[j,k]/v0[j]-1),worst_t_ma=q['timeline'][k]['t_ma'],
        uid448261_final=float(col(q,448261)['volume_km3'][-1]/col(q,448261)['initial_volume_km3']-1),
        elapsed_s=q['elapsed_s'],peak_mib=q['process_peak_working_set_mib'])
def main():
    OUT.mkdir(parents=True,exist_ok=True);setup_fonts()
    trials=[read(f'u36_s10_edge{edge}_patch1_dt0.25_a0.json') for edge in ['30','15','7.5']]
    patch=read('u36_s10_edge7.5_patch0.25_dt0.25_a0.json')
    half=read('u36_s10_edge15_patch1_dt0.125_a0.json')
    frame=read('u36_s10_edge15_patch1_dt0.25_a301.json')
    outer=read('u36_s10_outer_attempt.json')
    old=[read(f'u36_s8_edge{edge}_patch1.json') for edge in ['30','15','7.5']]
    base=trials[1]
    def compare(other):
        with np.load(CACHE/(base['tag']+'.npz')) as a,np.load(CACHE/(other['tag']+'.npz')) as b:
            assert np.array_equal(a['vertices'],b['vertices']) and np.array_equal(a['source_uid'],b['source_uid'])
            return dict(tag=other['tag'],maximum_final_position_difference_m=float(np.max(np.linalg.norm(a['points_0Ma']-b['points_0Ma'],axis=1))*R_KM*1000),
                maximum_final_H_relative_difference=float(np.nanmax(abs(a['H_0Ma']/b['H_0Ma']-1))),
                final_net_residual_difference=other['timeline'][-1]['volume_relative_residual']-base['timeline'][-1]['volume_relative_residual'])
    comparisons=[compare(half),compare(frame)]
    summary=dict(checkpoint='U36-S10',method_class='procedural_constrained',trials=[source_summary(q) for q in trials+[patch,half,frame]],
        comparisons_to_edge15_dt025_anchor0=comparisons,outer_attempt=dict(complete=outer['complete_35_to_0Ma'],failure=outer['failure'],initial_known_volume_km3=outer['initial_known_volume_km3']),
        physical_engine_changed=False,global_or_B5_coupling_performed=False,full_network_boundary_resolved=False,
        time_convergence_order_proven=False,physical_width_km=200,physical_width_class='user_override')
    write_json(DATA/'u36_s10_continuity.json',summary)
    figs=[];version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S10'
    def save(fig,name,title,note):
        fig.text(.015,.015,f'{version} · pyGPlates1.0.0 / SciPy1.18.1 / Shapely2.2.0\n固定参考域/200km宽度 user_override；点映射 interpolated；审计 procedural_constrained；图 visual_only\n'+note,fontsize=8,va='bottom')
        fig.savefig(OUT/name,dpi=140);plt.close(fig);figs.append(dict(file=name,title=title,note=note))
    colors=['#b06b42','#3a887e','#5c679b']
    fig,axs=plt.subplots(1,3,figsize=(16,8))
    for q,o,c in zip(trials,old,colors):
        t=[z['t_ma'] for z in q['timeline']]
        axs[0].plot(t,[100*z['volume_relative_residual'] for z in q['timeline']],c=c,label=f"新 {q['maximum_edge_km']:g}km")
        axs[0].plot([z['t_ma'] for z in o['timeline']],[100*z['volume_relative_residual'] for z in o['timeline']],c=c,ls='--',alpha=.7)
    axs[0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='净体积差 / %',title='同域、同初始来源；虚线为S8旧路径');axs[0].legend(fontsize=8)
    for qs,label,ls in [(trials,'S10候选','-'),(old,'S8旧路径','--')]:
        edge=[q['maximum_edge_km'] for q in qs];err=[]
        for q in qs:
            cols=q['source_trajectories'];v0=np.array([z['initial_volume_km3'] for z in cols]);vf=np.array([z['volume_km3'][-1] for z in cols]);err.append(100*np.sum(abs(vf-v0))/v0.sum())
        axs[1].loglog(edge,err,'o'+ls,label=label)
    axs[1].set(xlabel='最长审计边 / km',ylabel='来源加权绝对体积差 / %',title='空间细分不能只看净误差');axs[1].legend(fontsize=8)
    for q,c in zip(trials,colors):
        z=col(q,448261)
        axs[2].plot([x['t_ma'] for x in q['timeline']],100*(np.array(z['volume_km3'])/z['initial_volume_km3']-1),c=c,label=f"新 {q['maximum_edge_km']:g}km")
    z=col(old[-1],448261);axs[2].plot([x['t_ma'] for x in old[-1]['timeline']],100*(np.array(z['volume_km3'])/z['initial_volume_km3']-1),'k--',label='S8 7.5km')
    axs[2].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='UID448261体积差 / %',title='旧局部平台单独追踪');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.25)
    fig.subplots_adjust(bottom=.22,wspace=.32)
    save(fig,'106_same_source_continuity.png','同一固定域的连续性','原参考分辨率L7约60km；30/15/7.5km只是审计细分。H=H0/J微斑块，体积使用独立有限单元面积。未通过体积守恒定义自证。')
    a=np.load(CACHE/(trials[-1]['tag']+'.npz'));nv=len(a['vertices'])
    fig,axs=plt.subplots(1,3,figsize=(16,9))
    for ax,t in zip(axs,[35,20,0]):
        lo,la=xyz_to_lonlat(a[f'points_{t}Ma'][:nv])
        im=ax.tripcolor(lo,la,a['triangles'],facecolors=np.ma.masked_invalid(a[f'H_{t}Ma']),vmin=0,vmax=50,cmap='viridis')
        ax.set_facecolor('#ccc');ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=f'{t}Ma：条件性整柱厚度')
        ax.set_aspect(1/np.cos(np.radians(la.mean())));ax.grid(alpha=.2)
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.05*(x1-x0);y=y0+.05*(y1-y0);dx=100/(R_KM*np.pi/180*np.cos(np.radians(la.mean())))
        ax.plot([x,x+dx],[y,y],color='#111',lw=3);ax.text(x,y+.02*(y1-y0),'100km',fontsize=8)
    fig.subplots_adjust(bottom=.2,right=.86,wspace=.3);fig.colorbar(im,cax=fig.add_axes([.91,.27,.015,.45]),label='整柱厚度 / km')
    save(fig,'107_conditional_thickness.png','固定来源材料的条件性厚度','灰色是原非陆壳/无覆盖，未补值。此图不是B5地表、盆底、水深或新海岸；无热、断层、沉积与水位求解。')
    fig,axs=plt.subplots(1,3,figsize=(16,8))
    for q,label in [(old[-1],'S8旧路径'),(trials[-1],'S10 1km斑块'),(patch,'S10 0.25km斑块')]:
        cols=q['source_trajectories'];v0=np.array([z['initial_volume_km3'] for z in cols]);vf=np.array([z['volume_km3'][-1] for z in cols])
        axs[0].scatter(v0,100*(vf/v0-1),s=15,label=label)
    axs[0].set_xscale('log');axs[0].set(xlabel='各UID初始体积 / km³',ylabel='末态体积差 / %',title='同一来源逐项核对');axs[0].legend(fontsize=8)
    worst=summary['trials'][3]['worst_source_uid']
    for uid,ax in zip([448261,worst],axs[1:]):
        for q,c in zip(trials,colors):
            z=col(q,uid);ax.plot([x['t_ma'] for x in q['timeline']],100*(np.array(z['volume_km3'])/z['initial_volume_km3']-1),c=c,label=f"{q['maximum_edge_km']:g}km")
        z=col(patch,uid);ax.plot([x['t_ma'] for x in patch['timeline']],100*(np.array(z['volume_km3'])/z['initial_volume_km3']-1),'k:',label='7.5km / 0.25km斑块')
        ax.set(xlim=(35,0),xlabel='年代 / Ma',ylabel='来源体积差 / %',title=f'UID{uid}');ax.legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22,wspace=.35)
    save(fig,'108_source_residuals.png','局部来源与斑块误差','原误差源与新最差来源都保留；单个UID的残差不能被全球或整域净误差抵消。更细斑块仍不提高L7物理分辨率。')
    import pickle
    from mythic.tectonics.source_clipping import polygons
    with (CACHE/'sources.pkl').open('rb') as stream:geometry=pickle.load(stream)
    fig,axs=plt.subplots(1,2,figsize=(14,9))
    for geom,c,label in [(geometry['target'],'#b06b42','原带边缘'),(geometry['domain'],'#3a887e','固定内域')]:
        for j,p in enumerate(polygons(geom)):
            lo,la=xyz_to_lonlat(geometry['projection'].inverse(np.array(p.exterior.coords)));axs[0].plot(lo,la,c=c,label=label if j==0 else None)
    failure_file=CACHE/'outer_attempt_failure.npz'
    if failure_file.exists():
        b=np.load(failure_file);lo,la=xyz_to_lonlat(b['points'][b['bad']]);axs[0].scatter(lo,la,s=10,c='#b72b2b',label='外圈失效查询点')
    x0,x1=axs[0].get_xlim();y0,y1=axs[0].get_ylim();x=x0+.05*(x1-x0);y=y0+.05*(y1-y0);dx=100/(R_KM*np.pi/180*np.cos(np.radians(-20)))
    axs[0].plot([x,x+dx],[y,y],color='#111',lw=3);axs[0].text(x,y+.02*(y1-y0),'100km',fontsize=8)
    axs[0].set(xlabel='经度 / °E',ylabel='纬度 / °N',title='原外圈完整尝试：不删点，不向外补值');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2);axs[0].set_aspect(1/np.cos(np.radians(-20)))
    domain=read('u36_s8_fixed_domain.json')['domain']
    axs[1].bar([0,1],[domain['known_cont_volume_km3'],domain['reserved_outer_known_volume_km3']],color=['#3a887e','#b06b42'])
    axs[1].set_xticks([0,1],['内域已演化审计','外圈只保留初始库存']);axs[1].set(ylabel='已知初始体积 / km³',title='外圈演化仍有阻断')
    text=('失败时点 '+str(outer['failure']['t_ma'])+' Ma；'+outer['failure']['reason']+'；查询计数 '+str(outer['failure'].get('query_counts'))+'；受影响来源初始体积 '+str(outer['failure'].get('affected_known_initial_volume_km3'))+' km³') if outer['failure'] else '完整域支撑检查通过（不等于物理证明）'
    import textwrap
    axs[1].text(.02,.55,'\n'.join(textwrap.wrap(text,50)),transform=axs[1].transAxes,fontsize=8,va='top',bbox=dict(facecolor='white',alpha=.9))
    fig.subplots_adjust(bottom=.23,wspace=.35)
    save(fig,'109_outer_boundary_attempt.png','原外圈支撑与保留库存','库存沿用S8逐UID有符号浮点差，不静默归零。外圈失败不动态删材料，也不把它的初始库存称作已演化体积；S9整网阻断仍保留。')
    fig,axs=plt.subplots(1,2,figsize=(13,8))
    labels=['dt0.25 vs 0.125','anchor0 vs 301']
    axs[0].bar(labels,[q['maximum_final_position_difference_m'] for q in comparisons],color=['#b06b42','#3a887e'])
    axs[0].set_yscale('log');axs[0].set(ylabel='所有材料查询点末态最大位置差 / m',title='同一15km网格：时间与参照系分开')
    axs[1].bar(labels,[100*q['maximum_final_H_relative_difference'] for q in comparisons],color=['#b06b42','#3a887e'])
    axs[1].set_yscale('log');axs[1].set(ylabel='末态整柱厚度最大相对差 / %',title='相同源UID和初始顶点，全部保留')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.4)
    save(fig,'110_time_and_frame.png','同域时间步长与参照系对照','只有两档步长，不能宣称时间收敛阶。换框架差接近浮点精度不代表给定插值场是真实岩石圈运动；不得据B5目标高度调宽度。')

    legacy=read('u36_s10_uid_legacy.json')
    diagnostics=[read(f'u36_s10_uid448261_edge{e:g}_patch{r:g}.json') for r in [.25,.0625] for e in [7.5,1.875,.9375,.46875]]
    summary['single_source_diagnostic']=dict(uid=448261,scope='single_source_only',candidate=[dict(tag=q['tag'],triangle_count=q['triangle_count'],final_residual=q['timeline'][-1]['volume_relative_residual']) for q in diagnostics],legacy=legacy)
    write_json(DATA/'u36_s10_continuity.json',summary)
    fig,axs=plt.subplots(1,3,figsize=(16,8))
    for r,c in [(.25,'#b06b42'),(.0625,'#3a887e')]:
        rows=[q for q in diagnostics if q['patch_radius_km']==r]
        axs[0].loglog([q['triangle_count'] for q in rows],[100*abs(q['timeline'][-1]['volume_relative_residual']) for q in rows],'o-',c=c,label=f'S10 / 斑块{r:g}km')
        rows=[q for q in legacy['rows'] if q['patch_km']==r]
        axs[0].loglog([q['triangle_count'] for q in rows],[100*abs(q['final_residual']) for q in rows],'o--',c=c,label=f'S8 API / 斑块{r:g}km')
    axs[0].set(xlabel='同一来源的三角形个数',ylabel='末态体积差绝对值 / %',title='UID448261：真正细分到原小三角形');axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
    for ax,tag in zip(axs[1:],['uid448261_edge7.5_patch0.25','uid448261_edge0.46875_patch0.25']):
        mesh=np.load(CACHE/(tag+'.npz'));xy=geometry['projection'].forward(mesh['vertices']);xy-=xy.mean(axis=0)
        ax.triplot(xy[:,0],xy[:,1],mesh['triangles'],lw=.6)
        ax.set(xlabel='局部东西坐标 / km',ylabel='局部南北坐标 / km',title=f"同一初始来源：{len(mesh['triangles'])}个单元");ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.35)
    save(fig,'111_UID448261_refinement.png','旧来源平台的有限单元定位','同一约60.069km³初始源；30/15/7.5km阈值均未细分其3.34km原单元。这里只诊断该UID，不推广为全域逐源收敛或真实流变精度。虚线为旧API末态独立重算。')

    write_json(DATA/'u36_s10_manifest.json',dict(checkpoint='U36-S10',version=version,figures=figs,physical_engine_changed=False))
    page='<meta charset="utf-8"><title>U36-S10 固定来源连续性</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S10：同一固定来源域的旋转映射与柱连续性</h1>'
    for q in figs:page+=f'<h2>{html.escape(q["title"])}</h2><img src="{q["file"]}"><p>{html.escape(q["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
