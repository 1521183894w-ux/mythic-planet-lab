"""S12实际输出的图与独立逐来源复核。"""
from pathlib import Path
import json,hashlib,html,subprocess,gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.geometry import xyz_to_lonlat,R_KM
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s12';OUT=ROOT/'figures/design/v2/forward/u36_s12'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,q):p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    setup_fonts();OUT.mkdir(exist_ok=True)
    rows={m:[read(DATA/f'u36_s12_{m}_l{l}.json') for l in range(3)] for m in ['original','aligned']}
    ver=[]
    for mode,qs in rows.items():
        for level,q in enumerate(qs):
            a=np.load(CACHE/f'{mode}_l{level}.npz');nv=len(a['vertices']);tri=a['triangles'];known=a['known']
            h0=a['H0_km'];uid=a['source_uid'];uids=a['uids'];group=np.searchsorted(uids,uid[known])
            points=a['points'];jj=a['J'];saved=a['source_volumes_km3'];dif=0.;minimum=1.
            signed0=spherical_triangle_area(a['vertices'],tri)
            for i,(pos,j) in enumerate(zip(points,jj)):
                signed=spherical_triangle_area(pos[:nv],tri);minimum=min(minimum,float(np.min(signed/signed0)))
                value=np.bincount(group,weights=abs(signed)[known]*h0[known]/j[known],minlength=len(uids))
                dif=max(dif,float(abs(value-saved[i]).max()))
            assert dif<1e-6 and minimum>0
            ver.append(dict(mode=mode,level=level,times_checked=len(points),maximum_UID_recalculation_difference_km3=dif,
                minimum_finite_cell_area_ratio=minimum,maximum_initial_UID_difference_from_S11_km3=float(abs(a['V0_km3']-np.load(ROOT/f'.cache/u36_s11/full_e30_l{level}_dt0.25_a0.npz')['V0_km3']).max())))
            a.close();del points,jj;gc.collect()
    byuid={}
    for mode,qs in rows.items():
        byuid[mode]={z['uid']:[next(s for s in q['source_summary'] if s['uid']==z['uid'])['final_relative_residual'] for q in qs] for z in qs[0]['source_summary']}
    bad={m:[u for u,e in src.items() if not(abs(e[0])>=abs(e[1])>=abs(e[2]))] for m,src in byuid.items()}
    protected={}
    protected_paths=list(read(DATA/'u36_s11_verification.json')['protected'])+['packages/tectonics/boundary_transport.py','scripts/design_boundary_transport.py','tests/tectonics/test_boundary_transport.py','data/worlds/design_v2/u36_s11_reference.json']
    for p in protected_paths:
        current=subprocess.check_output(['git','hash-object',p],text=True).strip()
        base=subprocess.check_output(['git','rev-parse','c192b52:'+p],text=True).strip()
        protected[p]=current==base
    assert all(protected.values())
    metrics=read(ROOT/'figures/stage1/u36_s12_reference/metrics.json');old=read(DATA/'u36_s11_reference_metrics.json')
    clean=lambda q:{k:v for k,v in q['metrics'].items() if k!='timings_s'}
    assert clean(metrics)==clean(old) and metrics['b_verdict']==old['b_verdict']
    assert read(ROOT/'figures/stage1/u36_s12_reference/regions.json')['regions']==read(DATA/'u36_s11_reference_regions.json')['regions']
    for name in ['metrics','manifest','regions']:write(DATA/f'u36_s12_reference_{name}.json',read(ROOT/f'figures/stage1/u36_s12_reference/{name}.json'))
    tests=(ROOT/'.cache/u36_s12_tests.log').read_text(encoding='utf8')
    (DATA/'u36_s12_tests.txt').write_text('\n'.join(z.rstrip() for z in tests.splitlines())+'\n',encoding='utf8')
    result=dict(checkpoint='U36-S12',method_class='procedural_constrained',mapping_class='interpolated',
        source_error_by_uid=byuid,nonmonotonic_UIDs=bad,recalculations=ver,protected=protected,
        protected_base_commit='c192b52',S11_motion_field_source_files_unchanged=True,
        reference_metrics_identical_except_timings=True,region_windows_identical=True,
        all_tests_passed=False,new_tests_passed=True,old_strict_failures_retained=True,
        global_or_B5_coupled=False,physical_width_identified=False,quality_thresholds_changed=False,default_world_adopted=False,
        limitations=['逐档网格数不同，切片对照同时改变求积点位置和数量，不能单独证明跨片为唯一根因',
            '全部141离散时点支持和有限三角形正向，不证明连续时间或材料曲边精确性',
            '来源绝对误差不是世界质量指标，不以此自动关闭任何严重世界问题'])
    write(DATA/'u36_s12_verification.json',result)
    figs=[];version='c192b52+U36-S12';colors={'original':'#b36f48','aligned':'#337f91'};labels={'original':'原分区','aligned':'起点按分片裁切'}
    def save(fig,file,title,note):
        fig.text(.015,.015,version+' · design_v2；同一S11运动场 interpolated；给定200km宽度 user_override\n'
            +'来源/求积审计 procedural_constrained；图 visual_only；原物理输入L7约60km，细分不增加物理分辨率\n'+note,fontsize=8,va='bottom')
        fig.savefig(OUT/file,dpi=145);plt.close(fig)
        figs.append(dict(file=file,title=title,note=note,sha256=hashlib.sha256((OUT/file).read_bytes()).hexdigest()))
    fig,axes=plt.subplots(1,2,figsize=(15,8))
    for ax,m in zip(axes,['original','aligned']):
        a=np.load(CACHE/f'{m}_l0.npz');lo,la=xyz_to_lonlat(a['vertices']);tri=a['triangles']
        ax.triplot(lo,la,tri,color=colors[m],lw=.3)
        ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=f'{labels[m]}：{len(tri)}有限单元')
        ax.set_aspect(1/np.cos(np.radians(la.mean())));ax.grid(alpha=.2)
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.05*(x1-x0);y=y0+.05*(y1-y0);dx=100/(R_KM*np.pi/180*np.cos(np.radians(la.mean())))
        ax.plot([x,x+dx],[y,y],c='black',lw=2);ax.text(x,y+.02*(y1-y0),'100km',fontsize=8)
    fig.subplots_adjust(bottom=.2,wspace=.3)
    save(fig,'117_initial_partition.png','同一来源域的初始分区对照','只改求积分区，不改运动场。初始跨片1308→0；有限精度微片及逐UID初始库存差另列。')
    fig,axes=plt.subplots(1,3,figsize=(16,8))
    for m,qs in rows.items():
        n=[q['triangle_count'] for q in qs]
        axes[0].loglog(n,[q['timeline'][-1]['weighted_absolute_source_residual']*100 for q in qs],'o-',c=colors[m],label=labels[m])
        axes[1].loglog(n,[max(abs(z['final_relative_residual']) for z in q['source_summary'])*100 for q in qs],'o-',c=colors[m],label=labels[m])
        for level,q in enumerate(qs):
            axes[2].plot([z['t_ma'] for z in q['timeline']],[z['net_relative_residual']*100 for z in q['timeline']],c=colors[m],alpha=.4+.3*level,label=f'{labels[m]} L{level}')
    for ax in axes[:2]:ax.set_xlabel('实际有限材料单元数');ax.legend(fontsize=8)
    axes[0].set(ylabel='来源加权绝对体积差 / %',title='总量抵消不代替来源误差')
    axes[1].set(ylabel='最差来源末态绝对差 / %',title='全部77来源保留')
    axes[2].set(xlabel='年代 / Ma',ylabel='净体积差 / %',xlim=(35,0),title='净差时间序列');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22,wspace=.4)
    save(fig,'118_quadrature_comparison.png','六案独立有限面积—中心厚度对照','相同L号的单元数量不同，不能称同样成本的精度排名；三档不证明收敛阶。')
    selected=bad['original'];values=np.array([[100*byuid[m][u][l] for u in selected] for m in ['original','aligned'] for l in range(3)])
    fig,ax=plt.subplots(figsize=(15,8));vmax=abs(values).max()
    im=ax.imshow(values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
    for i in range(6):
        for j in range(len(selected)):ax.text(j,i,f'{values[i,j]:+.4f}',ha='center',va='center',fontsize=7)
    ax.set_xticks(range(len(selected)),[str(u) for u in selected],rotation=50,ha='right');ax.set_yticks(range(6),[f'{labels[m]} L{l}' for m in ['original','aligned'] for l in range(3)])
    ax.set(title='原15个非单调来源：逐来源有符号末态体积差',xlabel='来源UID')
    fig.colorbar(im,ax=ax,label='有符号末态体积差 / %');fig.subplots_adjust(bottom=.27,left=.14)
    save(fig,'119_nonmonotonic_sources.png','原15个来源逐项核对','完整77来源和两个分区的非单调清单均在JSON保留；不能仅挑改善来源。')
    fig,axes=plt.subplots(1,2,figsize=(15,8));x=np.arange(6);qflat=[q for m in ['original','aligned'] for q in rows[m]]
    exposure=[q['crossing_volume_time_fraction']*100 for q in qflat]
    association=[100*q['crossing_step_absolute_volume_change_km3']/(q['crossing_step_absolute_volume_change_km3']+q['single_face_step_absolute_volume_change_km3']) for q in qflat]
    axes[0].bar(x-.18,exposure,width=.36,label='跨片初始体积×时段份额')
    axes[0].bar(x+.18,association,width=.36,label='跨片组一步绝对体积变差份额')
    axes[0].set(ylabel='份额 / %',title='跨片与一步求积变化的关联');axes[0].legend(fontsize=8)
    for m,qs in rows.items():
        for l,q in enumerate(qs):
            axes[1].scatter([z['crossing_volume_time_fraction']*100 for z in q['source_summary']],[abs(z['final_relative_residual'])*100 for z in q['source_summary']],s=10+8*l,c=colors[m],alpha=.4,label=f'{labels[m]} L{l}')
    axes[0].set_xticks(x,[f'{labels[m]} L{l}' for m in ['original','aligned'] for l in range(3)],rotation=30,ha='right')
    axes[1].set(xlabel='每来源跨片体积×时段份额 / %',ylabel='每来源末态绝对体积差 / %',yscale='log',title='全部来源，关联不能证明唯一成因');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.25,wspace=.3)
    save(fig,'120_crossing_association.png','演化过程中跨片关联','跨片指运动插值分区，不是地质断层；初始对齐不等于后续持续对齐，未每步重切或强制守恒。')
    write(DATA/'u36_s12_manifest.json',dict(checkpoint='U36-S12',method_class='visual_only',version=version,figures=figs))
    page='<meta charset="utf-8"><title>U36-S12 分片求积</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S12：同一分片运动场的材料求积诊断</h1><p>方法检查点；未接全球/B5，阶段1未通过。</p>'
    for f in figs:page+=f'<h2>{html.escape(f["title"])}</h2><img src="{f["file"]}"><p>{html.escape(f["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('NONMONOTONIC',bad,flush=True);print('FIGURES',len(figs),flush=True)
if __name__=='__main__':main()