"""U42：有限盆地强度完整历史的固定窗、全球代价及预算对照。"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import setup_fonts,elevation_cmap_norm
from mythic.figures.world import code_version,load,load_earth
from mythic.metrics.grid import LatLonGrid
from mythic.tectonics.mesh import load_mesh,Raster
from design_final_map import apply_options,build,lonlat_mask

def localize_earth_gallery(standard):
    """仅复制画廊用到的公开地球基准图，使只服务figures目录也能加载。"""
    import re,shutil
    page=standard/'index.html'
    if not page.exists():
        return
    html=page.read_text(encoding='utf8')
    for match in set(re.findall(r"(?:\.\./)+data/earth_reference/regions/([A-Za-z0-9_-]+\.png)",html)):
        source=ROOT/'data/earth_reference/regions'/match
        target=standard/'earth_reference'/match
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    html=re.sub(r"(?:\.\./)+data/earth_reference/regions/([A-Za-z0-9_-]+\.png)",r"earth_reference/\1",html)
    page.write_text(html,encoding='utf8',newline='\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',choices=['01','05'],default='05')
    selected=ap.parse_args().review
    out=ROOT/'figures/design/v2/forward/u42_strength';out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/worlds/design_v2/candidates'
    cases={'00':'u42_strength_00','01':'u42_strength_01','05':'u42_strength_05'}
    labels={'00':'U41B复跑：完全阻流','01':'U42A：最低输运1%','05':'U42B：最低输运5%'}
    paths={k:data/c/f'design_v2_forward_v2_0Ma_{c}_1024x512.nc' for k,c in cases.items()}
    z={}
    for k,p in paths.items():
        with open_netcdf(p) as d:z[k]=np.asarray(d.elevation.values,float)
    g=LatLonGrid.from_shape(z['00'].shape);area=g.cell_area_km2
    design=apply_options(yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8')),set())
    features={f['id']:f for key in ['basins','plateaus'] for f in design[key]}
    land=build(design,g)['land']
    masks={k:lonlat_mask(g,features[k]['poly'])&land for k in ['B6','PL8','PL1']}
    runs={k:json.loads((data/c/'run.json').read_text(encoding='utf8')) for k,c in cases.items()}
    reports={k:json.loads((data/c/'forward_v2.json').read_text(encoding='utf8')) for k,c in cases.items()}
    summary=dict(method='u42_comparison_v1',review_candidate=cases[selected],
        method_class={'transport':'procedural_constrained','regions':'user_override','elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
        resolution='L7 about60km; output1024x512',units={'elevation':'m','volume':'km3','area':'km2','time':'Ma'},
        regions={},cases={},control_reproduction={},input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    for fid,mask in masks.items():
        summary['regions'][fid]={k:dict(median_m=float(np.median(v[mask])),
            mean_m=float(np.average(v[mask],weights=area[mask])),
            above4500_fraction=float(area[mask&(v>4500)].sum()/area[mask].sum())) for k,v in z.items()}
    cache=ROOT/'.cache/tectonics/design/candidates'
    fields=['elev','T','age','t_form','oro_t','plate','ctype','origin','oro','prov','dyn']
    with np.load(cache/'u41_contact/forward_v2_v2_L7.npz') as old,np.load(cache/f"{cases['00']}/forward_v2_v2_L7.npz") as new:
        keys=list(old.files);unequal=[k for k in keys if not np.array_equal(old[k],new[k],equal_nan=old[k].dtype.kind in 'biufc')]
        summary['control_reproduction'].update(compared_arrays=len(keys),unequal_arrays=unequal)
    oldrep=json.loads((data/'u41_contact/forward_v2.json').read_text(encoding='utf8'))
    summary['control_reproduction']['budget_identical']=oldrep['budget_km3']==reports['00']['budget_km3']
    summary['control_reproduction']['events_identical']=oldrep['engine_event_counts']==reports['00']['engine_event_counts']
    oldnative=np.load(ROOT/'.cache/u41_outlet/contact/native.npz')
    newnative=np.load(ROOT/'.cache/u42_strength/u42_strength_00/native.npz')
    summary['control_reproduction']['native_compared_arrays']=len(oldnative.files)
    summary['control_reproduction']['native_unequal_arrays']=[k for k in oldnative.files if not np.array_equal(oldnative[k],newnative[k],equal_nan=True)]
    oldnative.close();newnative.close()
    for k in cases:
        r=runs[k];assert r['success']
        diff=z[k]-z['00'];b=reports[k]['budget_km3'];b0=reports['00']['budget_km3']
        row=dict(max_elevation_m=float(z[k].max()),cells_over10km=int((z[k]>10000).sum()),
            changed_area_over100m_km2=float(area[abs(diff)>100].sum()),max_abs_change_m=float(abs(diff).max()),
            land_to_ocean_km2=float(area[(z['00']>0)&(z[k]<=0)].sum()),
            ocean_to_land_km2=float(area[(z['00']<=0)&(z[k]>0)].sum()),
            budget_delta_km3={f:b[f]-b0.get(f,0) for f in b},
            step_count=len(r['step_ledger']),
            max_balance_km3=max(abs(s['balance_km3']) for s in r['step_ledger']),
            max_local_residual_km3=max(s['max_local_residual_km3'] for s in r['step_ledger']),
            max_outlet_balance_km3=max(abs(s['balance_km3']) for s in r['outlet_ledger']),
            pre_outlet_max_T_km=max(s['pre_max_T_km'] for s in r['outlet_ledger']),
            post_outlet_max_T_km=max(s['post_max_T_km'] for s in r['outlet_ledger']),
            elapsed_s=r['elapsed_s'],process_lifetime_peak_wset_bytes=r['process_lifetime_peak_wset_bytes'],
            surface_totals={fid:{f:sum(s['regions'][fid][f] for s in r['step_ledger'])
                for f in ['erosion_removed_km3','lateral_net_in_km3','relaxation_net_in_km3']}
                for fid in ['B6','PL8','other_continent','other_cells']})
        with np.load(cache/f"{cases['00']}/forward_v2_v2_L7.npz") as control,np.load(cache/f'{cases[k]}/forward_v2_v2_L7.npz') as frames:
            ids=np.flatnonzero(control['times']>180)
            row['pre_activation_compared_arrays']=len(ids)*len(fields)
            row['pre_activation_unequal_arrays']=[f'{f}_{i}' for i in ids for f in fields if not np.array_equal(control[f'{f}_{i}'],frames[f'{f}_{i}'],equal_nan=True)]
        # 固定剖面陡阶诊断，不新增或修改质量指标。
        ir=int(np.argmin(abs(g.lat-31.11)));s=(g.lon>=100)&(g.lon<=107)
        row['profile_31N_max_adjacent_jump_m']=float(abs(np.diff(z[k][ir,s])).max())
        row['profile_latitude_deg']=float(g.lat[ir])
        summary['cases'][k]=row
    setup_fonts();cmap,norm=elevation_cmap_norm();extent=(-180,180,-90,90)
    stamp='区域/时期 user_override；输运 procedural_constrained；高程 parameterized_physics；L7约60km；输出1024×512'
    def finish(fig,name):
        fig.text(.5,.004,'设计v2；代码 '+code_version(),ha='center',fontsize=8)
        fig.savefig(out/name,dpi=125,bbox_inches='tight');plt.close(fig)
    def local(ax):
        ax.set_xlim(96,117);ax.set_ylim(20,36);ax.set_aspect(1/np.cos(np.deg2rad(28)))
        ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
        ax.plot([97,97+500/(111.2*np.cos(np.deg2rad(28)))],[20.8,20.8],color='black',lw=2)
        ax.text(99.5,21.2,'500 km（28°N）',ha='center',fontsize=8)
        for fid in ['B6','PL8']:
            xy=np.asarray(features[fid]['poly']);xy=np.vstack([xy,xy[0]])
            ax.plot(*xy.T,color='#cc145c',lw=1)
            ax.text(*xy[:-1].mean(axis=0),fid,color='#a00045',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig,axs=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
    for ax,k in zip(axs.flat,cases):
        im=ax.imshow(z[k],extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
        local(ax);ax.set_title(labels[k]);fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程 / m')
    ax=axs.flat[-1];im=ax.imshow(z[selected]-z['00'],extent=extent,origin='upper',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-600,vcenter=0,vmax=600),interpolation='nearest')
    local(ax);ax.set_title(labels[selected]+' − 控制');fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程变化 / m')
    fig.suptitle('同一300→0Ma历史，仅180Ma后隐式松弛允许盆地有限接收\n'+stamp,fontsize=11);finish(fig,'01_cases.png')
    fig,axs=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    for color,(k,v) in zip(['#777777','#0072b2','#d55e00'],z.items()):
        for ax,lat in zip(axs[:2],[31.,25.]):
            row=int(np.argmin(abs(g.lat-lat)));s=(g.lon>=96)&(g.lon<=117)
            ax.plot(g.lon[s],v[row,s],label=labels[k],color=color)
            ax.set_title(f'{g.lat[row]:.2f}°N最近行');ax.set_xlabel('经度 / °')
        col=int(np.argmin(abs(g.lon-106)));s=(g.lat>=20)&(g.lat<=36)
        axs[2].plot(g.lat[s],v[s,col],label=labels[k],color=color)
    for ax in axs:ax.set_ylabel('高程 / m');ax.grid(alpha=.2)
    axs[2].set_title(f'{g.lon[col]:.2f}°E最近列');axs[2].set_xlabel('纬度 / °');axs[0].legend(fontsize=8)
    fig.suptitle('固定剖面：盆缘落差与盆地被抬高的代价\n'+stamp,fontsize=11);finish(fig,'02_profiles.png')
    fig,axs=plt.subplots(2,4,figsize=(21,9),constrained_layout=True)
    for row,k in enumerate(['01','05']):
        t=np.array([s['t_ma'] for s in runs[k]['step_ledger']])
        for ax,fid in zip(axs[row],['B6','PL8','other_continent','other_cells']):
            for f,label,sign in [('erosion_removed_km3','剥蚀移除',-1),('lateral_net_in_km3','旧侧流净入',1),('relaxation_net_in_km3','隐式松弛净入',1)]:
                ax.plot(t,np.cumsum([sign*s['regions'][fid][f]/1e6 for s in runs[k]['step_ledger']]),label=label)
            ax.set_xlim(180,0);ax.grid(alpha=.2);ax.set_title(labels[k]+' · '+fid);ax.set_xlabel('时间 / Ma');ax.set_ylabel('累计体积 / 百万km³')
    axs[0,0].legend(fontsize=8);fig.suptitle('表层算子分区账：当前移动设计窗口，非完整物源；不含独立U41出口\n'+stamp,fontsize=11);finish(fig,'03_surface_ledger.png')
    raster=Raster(load_mesh(7),512);fig,axs=plt.subplots(3,5,figsize=(22,12),constrained_layout=True)
    for row,k in enumerate(cases):
        with np.load(cache/f'{cases[k]}/forward_v2_v2_L7.npz') as frames:
            sea={s['t']:s['sea_level_m'] for s in reports[k]['sea_level_m']}
            for ax,t in zip(axs[row],[300.,200.,100.,50.,0.]):
                i=int(np.argmin(abs(frames['times']-t)))
                value=raster(frames[f'elev_{i}']).astype(float)-sea[t]
                im=ax.imshow(value,extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
                ax.set_title(labels[k]+f' · {t:g}Ma',fontsize=9);ax.set_xticks([-120,0,120]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.2)
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.025,pad=.07,label='相对当时海面的高程 / m')
    fig.suptitle('真实关键年代：每案同次演化的位置、地壳与海面\n'+stamp,fontsize=11);finish(fig,'04_history.png')
    fig,axs=plt.subplots(2,2,figsize=(17,10),constrained_layout=True)
    for col,k in enumerate(['01','05']):
        im=axs[0,col].imshow(z[k],extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
        axs[0,col].set_title(labels[k]);fig.colorbar(im,ax=axs[0,col],orientation='horizontal',fraction=.04,pad=.07,label='高程 / m')
        im=axs[1,col].imshow(z[k]-z['00'],extent=extent,origin='upper',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-600,vcenter=0,vmax=600),interpolation='nearest')
        axs[1,col].set_title('相对同源控制的全球变化');fig.colorbar(im,ax=axs[1,col],orientation='horizontal',fraction=.04,pad=.07,label='变化 / m')
    for ax in axs.flat:ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
    fig.suptitle('全球代价：固定差值色带；范围外极值记录在报表\n'+stamp,fontsize=11);finish(fig,'05_global_cost.png')
    from mythic.figures.region_panels import region_pair
    from mythic.figures.tectonic_panels import region_tectonics
    from mythic.figures.select_regions import RegionPick
    record=json.loads((ROOT/'figures/stage1/u38_b6_pl8_transition/regions.json').read_text(encoding='utf8'))['regions'][-1]
    pick=RegionPick(key='u38_r8_fixed',title='旧r8固定复查',rule='保留旧r8',earth_ref=None,center=tuple(record['center_lat_lon']),size_km=record['size_km'],reason='旧问题不因抽样关闭')
    world=load(str(paths[selected]));earth=load_earth(world.resolution)
    region_pair(pick,world,earth,record['earth_comparison'],out/'06_previous_r8.png',8)
    region_tectonics(pick,world,out/'07_previous_r8_tectonics.png',8)
    # U41出口前后逐步厚度脉冲，避免以10Myr高程帧替代全部步。
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    for color,k in zip(['#777777','#0072b2','#d55e00'],cases):
        row=runs[k]['outlet_ledger'];t=[s['t_ma'] for s in row]
        ax.plot(t,[s['post_max_T_km'] for s in row],color=color,label=labels[k]+' 出口后')
        ax.plot(t,[s['pre_max_T_km'] for s in row],color=color,ls=':',alpha=.6)
    ax.set_xlim(180,0);ax.set_xlabel('时间 / Ma');ax.set_ylabel('原生陆壳最大厚度 / km');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('181步出口前（虚线）/后（实线）厚度；不是逐步海面相对高程\n'+stamp,fontsize=11);finish(fig,'08_step_peaks.png')
    summary['limits']=['finite mobility hypothesis, not a calibrated rheology','design windows not actual plateau segmentation','saved10Myrframes and postoutlet181steps do not certify continuous history ceiling','r2/oldr8 and material provenance remain unresolved','defaultC2unchanged; stage1notaccepted']
    text=json.dumps(summary,ensure_ascii=False,indent=2)
    (out/'comparison.json').write_text(text,encoding='utf8')
    (data/cases[selected]/'comparison.json').write_text(text,encoding='utf8')
    names=['00_spike.png','01_cases.png','02_profiles.png','03_surface_ledger.png','04_history.png','05_global_cost.png','06_previous_r8.png','07_previous_r8_tectonics.png','08_step_peaks.png']
    html='<!doctype html><meta charset="utf-8"><title>U42有限盆地强度</title><h1>U42：盆缘与有限输运</h1><p>未经标定的参数化候选，默认C2不变；阶段1未通过。固定窗保留r2和旧r8成因问题。</p>'
    for n in names:html+=f'<p><img style="width:100%" src="{n}"></p>'
    html+=f'<p><a href="../../../../stage1/{cases[selected]}/index.html">29张标准图组</a></p>'
    (out/'index.html').write_text(html,encoding='utf8')
    localize_earth_gallery(ROOT/'figures/stage1'/cases[selected])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
