"""U36-S7：同一物理运动的参照系与点插值客观性对照。"""
import json,time,sys,subprocess,html
from pathlib import Path
import numpy as np,pygplates as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,unit
from mythic.tectonics.kinematics import fr_to_mat
from mythic.tectonics.material_band import build_network,reference_strip,point_array,spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from design_step_audit import control_rotation
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s7';DATA=ROOT/'data/worlds/design_v2';OUT=ROOT/'figures/design/v2/forward/u36_s7'


def main():
    start=time.perf_counter();CACHE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);setup_fonts()
    if '--report-only' in sys.argv:
        report=json.loads((DATA/'u36_s7_frames.json').read_text(encoding='utf-8'));results=report['rows'];arrays=np.load(CACHE/'frames.npz')
    else:
        strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60)
        test=unit(strip['xyz'][strip['triangles']].sum(1))[::8]
        results=[];arrays={}
        for case in ['rigid','ramp','turning','real']:
            if case=='real':
                src=np.load(ROOT/'.cache/u36_s4/material_band.npz');centers=unit(src['xyz'][src['triangles']].sum(1));weights=src['area0_km2']
                rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'));features=pg.FeatureCollection(str(ROOT/'.cache/u36_s4/material_network.gpml'));anchors=[0,301,401]
            else:
                centers=test;weights=np.ones(len(test));rot=control_rotation(case);_,features=build_network(strip,rot,35,plates=(1,2),tag=case);anchors=[0,1,2]
            points,tri=small_material_patches(centers,1.);area0=spherical_triangle_area(points,tri).reshape(-1,4).sum(1)
            for dt in [.25,.125]:
                for natural in [True,False]:
                    basex=None;basej=None
                    for anchor in anchors:
                        # 仅改变坐标框架，最终全部变回原0号锚定框架；不是改变物理板块运动。
                        matrix=fr_to_mat(rot.get_rotation(35.,anchor));q0=points@matrix
                        model=pg.TopologicalModel(features,rot,anchor_plate_id=anchor,topological_snapshot_cache_size=2,
                            default_resolve_topology_parameters=pg.ResolveTopologyParameters(enable_strain_rate_clamping=False,strain_rate_smoothing=pg.StrainRateSmoothing.natural_neighbour))
                        span=model.reconstruct_geometry([tuple(p) for p in q0],35,time_increment=dt,deactivate_points=None,deformation_uses_natural_neighbour_interpolation=natural)
                        times=35-np.arange(round(35/dt)+1)*dt;jlist=[];b=[]
                        for t in times:
                            q=point_array(span.get_geometry_points(float(t)))
                            jlist.append(spherical_triangle_area(q,tri).reshape(-1,4).sum(1)/area0)
                            b.append(np.asarray(span.get_crustal_stretching_factors(float(t)))[:len(centers)])
                        q=q@fr_to_mat(rot.get_rotation(0.,anchor)).T;j=np.array(jlist);beta=np.array(b)
                        if basex is None:basex=q[:len(centers)];basej=j[-1]
                        row=dict(case=case,dt_myr=dt,natural_neighbour=natural,anchor=anchor,
                            J_median=float(np.median(j[-1])),J_min=float(j.min()),J_max=float(j.max()),
                            maximum_position_difference_from_anchor0_km=float(np.max(np.linalg.norm(q[:len(centers)]-basex,axis=1))*R_KM),
                            maximum_area_ratio_difference_from_anchor0=float(np.max(abs(j[-1]/basej-1))),
                            weighted_log_beta_over_J=float(np.average(np.log(beta[-1]/j[-1]),weights=weights)))
                        if case=='rigid':
                            exact=points@fr_to_mat(rot.get_rotation(0.,1,35.)).T
                            row['maximum_position_difference_from_exact_rigid_km']=float(np.max(np.linalg.norm(q-exact,axis=1))*R_KM)
                            row['maximum_area_error_from_exact_rigid']=float(np.max(abs(j[-1]-1)))
                        results.append(row);key=f'{case}_{dt}_{int(natural)}_{anchor}';arrays[key+'_J']=j;arrays[key+'_beta']=beta;arrays[key+'_times']=times;arrays[key+'_final_xyz']=q
                        print(case,dt,natural,anchor,row,flush=True)
        np.savez_compressed(CACHE/'frames.npz',**arrays)
        report=dict(checkpoint='U36-S7',framework_version=pg.__version__,method_class='procedural_constrained',field_class='interpolated',geometry_class='user_override',
            units={'length':'km','time':'Myr','J':'无量纲'},parameters={'initial_time_ma':35,'patch_radius_km':1,'dt_myr':[.25,.125]},
            rows=results,elapsed_s=time.perf_counter()-start,physical_engine_changed=False,
            assumptions=['锚定坐标变换后全部还原到同一0号框架','同一旋转文件、特征和初始物理点；不改变物理带宽','自然邻域/重心插值均为框架现有选项'],
            limitations=['不是独立力学求解','若不同参照系给出不同还原结果，不可选择更好看的参照系宣称修复','不替换全球地壳、B5或默认世界'])
        write_json(DATA/'u36_s7_frames.json',report)
    traces=[]
    for natural in [True,False]:
        for anchor in [0,1,2]:
            key=f'rigid_0.125_{int(natural)}_{anchor}';jj=arrays[key+'_J']
            traces.append(dict(case='rigid',dt_myr=.125,natural_neighbour=natural,anchor=anchor,times_ma=arrays[key+'_times'].tolist(),J_mean=jj.mean(1).tolist(),J_median=np.median(jj,axis=1).tolist(),maximum_local_absolute_area_error=np.max(abs(jj-1),axis=1).tolist()))
    write_json(DATA/'u36_s7_timelines.json',dict(checkpoint='U36-S7',method_class='procedural_constrained',sampling_note='每序列281个实际框架输出时点；不是端点插值',series=traces))
    version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S7';figures=[]
    def save(fig,name,title,note):
        fig.text(.012,.012,f'U36-S7 · {version} · pyGPlates1.0.0\n参考几何 user_override；框架场 interpolated；参照系/面积核对 procedural_constrained\n'+note,fontsize=8,va='bottom');fig.savefig(OUT/name,dpi=140);plt.close(fig);figures.append(dict(file=name,title=title,note=note))
    fig,axs=plt.subplots(1,2,figsize=(14,8))
    for natural,ls in [(True,'-'),(False,'--')]:
        for anchor,c in [(0,'#ad674d'),(1,'#277d8e'),(2,'#8a68a7')]:
            k=f'rigid_0.125_{int(natural)}_{anchor}';axs[0].plot(arrays[k+'_times'],100*(arrays[k+'_J'].mean(1)-1),ls,color=c,label=f'anchor{anchor} '+('自然邻域' if natural else '重心'))
    axs[0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='共同刚体面积变化 / %',title='共同旋转：输入应当严格保面积');axs[0].legend(fontsize=8)
    rows=[q for q in results if q['case']=='rigid' and q['dt_myr']==.125];axs[1].bar(range(len(rows)),[q['maximum_position_difference_from_exact_rigid_km'] for q in rows],color=['#ad674d','#277d8e','#8a68a7']*2)
    axs[1].set_xticks(range(len(rows)),[f"{q['anchor']} / {'NN' if q['natural_neighbour'] else 'BC'}" for q in rows]);axs[1].set(xlabel='锚定板块 / 插值',ylabel='与精确旋转最大位置差 / km',title='变回同一物理框架后比较')
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.2,wspace=.3);save(fig,'96_rigid_frame_objectivity.png','共同刚体与精确参照','NN=自然邻域；BC=重心。锚定板块改变仅改变坐标，不能改变还原后的真实材料运动；零误差刚体特例不证明一般变形正确。')
    fig,axs=plt.subplots(1,3,figsize=(15,8))
    for ax,case in zip(axs,['ramp','turning','real']):
        rows=[q for q in results if q['case']==case and q['dt_myr']==.125]
        ax.bar(range(len(rows)),[q['maximum_position_difference_from_anchor0_km'] for q in rows],color=['#ad674d','#277d8e','#8a68a7']*2)
        ax.set_xticks(range(len(rows)),[f"{q['anchor']}\n{'NN' if q['natural_neighbour'] else 'BC'}" for q in rows],fontsize=8);ax.set(title={'ramp':'两级启动','turning':'变向/走滑','real':'真实参考段'}[case],xlabel='锚定板块 / 插值',ylabel='对同插值anchor0最大位置差 / km');ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.3);save(fig,'97_deforming_frame_differences.png','相同物理运动的参照系差','每组对照共享旋转文件、原始材料点和网络；不同输入史不能相互排名。数值差异不当作物理剪切/真实减薄。')
    fig,axs=plt.subplots(1,2,figsize=(14,8))
    for case,c in [('rigid','#777'),('ramp','#ac6850'),('turning','#8a68a7'),('real','#277d8e')]:
        for natural,ls in [(True,'-'),(False,'--')]:
            rows=[q for q in results if q['case']==case and q['natural_neighbour']==natural and q['anchor']!=0]
            vals=[max(q['maximum_area_ratio_difference_from_anchor0'] for q in rows if q['dt_myr']==dt)*100 for dt in [.25,.125]]
            axs[0].plot([.25,.125],vals,'o'+ls,color=c,label=case+(' NN' if natural else ' BC'))

    axs[0].set(xlabel='时间步 / Myr',ylabel='相对anchor0最大面积比差 / %',title='参照系差是否随时间步减少');axs[0].legend(fontsize=8,ncol=2)
    selected=[q for q in results if q['natural_neighbour'] and q['dt_myr']==.125]
    axs[1].bar(range(len(selected)),[100*q['weighted_log_beta_over_J'] for q in selected],color=np.repeat(['#777','#ac6850','#8a68a7','#277d8e'],3))
    axs[1].set_xticks(range(len(selected)),[f"{q['case']}\n{q['anchor']}" for q in selected],fontsize=7,rotation=50)
    axs[1].set(xlabel='输入史 / 锚定板块（分类）',ylabel='加权log(β/J) ×100',title='原框架标量与同轨迹面积')
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.2,wspace=.35);save(fig,'98_frame_step_and_scalar.png','步长与标量审计','只记录框架数值客观性；真实物理带宽/古接触仍待独立约束。未采用任何参照系或插值方案为全球替换。')
    write_json(DATA/'u36_s7_manifest.json',dict(checkpoint='U36-S7',version=version,figures=figures,physical_engine_changed=False))
    page='<meta charset="utf-8"><title>U36-S7 参照系审计</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S7：同一物理运动的参照系审计</h1>'
    for q in figures:page+=f'<h2>{html.escape(q["title"])}</h2><img src="{q["file"]}"><p>{html.escape(q["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8');print('elapsed',report['elapsed_s'],flush=True)
if __name__=='__main__':main()
