"""U36-S9：投影根因、来源拓扑、条件性候选与整网失败均入图。"""
import hashlib,html,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2'
CACHE=ROOT/'.cache/u36_s9';OUT=ROOT/'figures/design/v2/forward/u36_s9'
NAMES={'rigid':'共同刚体','ramp':'两级启动','turning':'变向+走滑','slip':'沿带错动','real':'真实C/D参考段'}
def main():
    setup_fonts();OUT.mkdir(parents=True,exist_ok=True)
    final=json.loads((DATA/'u36_s9_rotation_maps.json').read_text(encoding='utf8'))
    dynamic=json.loads((DATA/'u36_s9_dynamic_rotation_maps.json').read_text(encoding='utf8'))
    mesh=json.loads((DATA/'u36_s9_mesh_audit.json').read_text(encoding='utf8'))
    version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S9'
    figures=[]
    def save(fig,name,title,note):
        fig.text(.012,.012,f'U36-S9 · {version} · pyGPlates 1.0.0 · 确定性材料采样\n几何/带宽 user_override；两种点映射 interpolated；核对 procedural_constrained；绘图 visual_only\n'+note,fontsize=8,va='bottom')
        fig.savefig(OUT/name,dpi=140);plt.close(fig)
        figures.append(dict(file=name,title=title,note=note))
    fig,axs=plt.subplots(1,2,figsize=(14,8))
    q=next(r for r in final['rows'] if r['case']=='rigid' and r['anchor']==0 and r['dt_myr']==.125)
    t=[r['t_ma'] for r in q['timeline']]
    for key,label,c in [('legacy','官方投影位置插值','#b7714e'),('candidate','固定来源网格+旋转插值','#277e91')]:
        y=[r[key+'_max_exact_area_error']*100 for r in q['timeline']]
        axs[0].semilogy(t,np.maximum(y,1e-12),color=c,label=label)
    axs[0].set(xlabel='年代 / Ma',ylabel='对解析刚体的最大 |J−1| / %',title='相同共同旋转，应严格保面积',xlim=(35,0));axs[0].legend()
    rows=[r for r in final['rows'] if r['case']=='rigid']
    vals=[max(r['timeline'][-1][key+'_max_exact_position_error_km'] for r in rows)*1000 for key in ['legacy','candidate']]
    axs[1].bar([0,1],vals,color=['#b7714e','#277e91']);axs[1].set_xticks([0,1],['官方路径','旋转候选'])
    axs[1].set(yscale='log',ylabel='末态对精确刚体最大位置差 / m',title='三锚定、两档步长的最差值')
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22,wspace=.3)
    save(fig,'102_projection_and_rigid_truth.png','官方投影插值的刚体误差与旋转候选','左图绘图下限1e−12%，只用于显示；非误差容差。官方重心分支独立复现误差逐案保存。候选没有替换原框架beta/速度。')
    fig,axs=plt.subplots(1,2,figsize=(15,8))
    cases=list(NAMES);xs=np.arange(len(cases));width=.24
    for offset,report,key,label,c in [(-1,final,'legacy','官方投影位置','#b7714e'),(0,dynamic,'candidate','旋转但动态连接','#9a81b5'),(1,final,'candidate','旋转且固定来源连接','#277e91')]:
        vals=[];jvals=[]
        for case in cases:
            rr=[q for q in report['rows'] if q['case']==case]
            vals.append(max(q[key+'_max_frame_position_difference_km'] for q in rr)*1000)
            jvals.append(max(q[key+'_max_frame_J_relative_difference'] for q in rr)*100)
        axs[0].bar(xs+offset*width,vals,width,label=label,color=c)
        axs[1].bar(xs+offset*width,jvals,width,label=label,color=c)
    for ax in axs:
        ax.set_xticks(xs,[NAMES[k] for k in cases],rotation=15);ax.set_yscale('log');ax.grid(alpha=.2)
    axs[0].set(ylabel='还原0号框架后的最大位置差 / m',title='全部141/281时点与材料中心的最差值')
    axs[1].set(ylabel='最大参照系面积比差 / %',title='相同输入不应因坐标系改变面积')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,fontsize=10)
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.3)
    save(fig,'103_rotation_and_connectivity.png','旋转插值与来源连接分别检验','五种输入分别比较，未挑最优框架。两档dt=0.25/0.125Myr。首轮动态连接反例完整保留；小面积差只验证这些内部点的运动学客观性。')

    a=np.load(CACHE/'real_0_0.125.npz')
    x=a['candidate_xyz'][0];lon=np.degrees(np.arctan2(x[:,1],x[:,0]));lat=np.degrees(np.arcsin(x[:,2]))
    old,new=a['legacy_J'][-1],a['candidate_J'][-1]
    fig,axs=plt.subplots(1,3,figsize=(15,9))
    for ax,z,title in zip(axs,[old,new,100*(new/old-1)],['官方点映射 J','固定连接旋转候选 J','候选/官方 − 1（%）']):
        if ax is axs[2]:
            lim=max(abs(z.min()),abs(z.max()))
            sc=ax.scatter(lon,lat,c=z,s=14,cmap='RdBu_r',vmin=-lim,vmax=lim)
        else:sc=ax.scatter(lon,lat,c=z,s=14,cmap='viridis',vmin=min(old.min(),new.min()),vmax=max(old.max(),new.max()))
        fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.11,label='%' if ax is axs[2] else '无量纲')
        bar=100/(111.195*np.cos(np.radians(-20.)))
        ax.plot([18.4,18.4+bar],[-23.8,-23.8],color='#222',lw=2)
        ax.text(18.4,-23.68,'100 km（20°S）',fontsize=8)
        ax.set(xlabel='35 Ma材料经度 / °E',ylabel='35 Ma材料纬度 / °N',title=title,aspect='equal');ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.4)
    save(fig,'104_real_material_J.png','真实C/D段的条件性面积比','432个材料中心、35→0Ma、1km环形面积探针，结果画在同一35Ma初始位置便于比较；不是现今地貌、盆底、水深或材料来源重算。')
    fig,axs=plt.subplots(1,3,figsize=(15,8))
    for case,c in [('ramp','#a66a4b'),('real','#277e91')]:
        mr=next(q for q in mesh['rows'] if q['case']==case)
        tt=mesh['sample_times_ma'];vals=[];flips=[];diff=[]
        for t in tt:
            rr=[q for q in mr['timeline'] if q['t_ma']==t]
            vals.append(max(q['dynamic_triangle_symmetric_difference_count'] for q in rr))
            flips.append(max(q['fixed_negative_triangle_count'] for q in rr))
            diff.append(max(abs(q['fixed_mesh_polygon_relative_area_difference']) for q in rr)*100)
        axs[0].plot(tt,vals,color=c,label=NAMES[case])
        axs[1].plot(tt,flips,color=c,label=NAMES[case])
        axs[2].plot(tt,diff,color=c,label=NAMES[case])
    for ax in axs:ax.set(xlabel='年代 / Ma',xlim=(35,0));ax.legend();ax.grid(alpha=.2)
    axs[0].set(ylabel='对anchor0不同的三角形个数',title='动态三角化有换框架连接差')
    axs[1].set(ylabel='固定网格投影负向三角形数',title='整网边缘仍有失败，未删除')
    axs[2].set(ylabel='|网格面积/连续网络面积−1| / %',title='面积覆盖并未严格闭合')
    fig.subplots_adjust(bottom=.22,wspace=.32)
    save(fig,'105_network_boundary_failures.png','候选整网支持范围与边缘失败','统计完整缓冲网络的投影方向，球面有符号面积另记（本轮无负值）；含未被内部材料点采用的边缘细长三角形。逐查询另外拒绝网外、负向及多重内部覆盖；内部采样成功不代表完整域输运成立。')
    manifest=dict(checkpoint='U36-S9',version=version,method_classes=final['method_classes'],figures=figures,
        whole_network_geometry_test_passed=mesh['full_network_geometry_test_passed'],
        raw_inputs=['u36_s9_rotation_maps.json','u36_s9_dynamic_rotation_maps.json','u36_s9_mesh_audit.json'],
        source_reference_world='S3冻结旁观控制数组；S4审计ID固定r8窗口。完整来源与哈希另表',
        physical_engine_changed=False,default_adopted=False)
    (DATA/'u36_s9_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    page='<meta charset="utf-8"><title>U36-S9 旋转插值与来源连接</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S9：内部点客观性候选与整网边缘失败</h1><p>局部插值小试；不是新世界地貌或B5修复。默认C2，阶段1未通过。</p>'
    for q in figures:page+=f'<h2>{html.escape(q["title"])}</h2><img src="{q["file"]}"><p>{html.escape(q["note"])}</p>'
    page+='<p><a href="../../../../stage1/u36_s9_reference/index.html">冻结世界标准图组（29图）</a></p>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('four figures',OUT,flush=True)
if __name__=='__main__':main()
