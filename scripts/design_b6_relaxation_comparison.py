"""U40：实际正向世界、机制、剖面、收支与旧问题固定窗对照。"""
from pathlib import Path
import hashlib,json,sys
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
OUT=ROOT/'figures/design/v2/forward/u40_relaxation'
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/worlds/design_v2/candidates'
CANDIDATES={'ring':'u39_b6_pl8_halo','slow':'u40_relax_3000','fast':'u40_relax_10000'}
TITLES={'ring':'U39B：瞬时环形分配','slow':'U40A：有限速率 3000','fast':'U40B：有限速率 10000'}
paths={k:DATA/c/f'design_v2_forward_v2_0Ma_{c}_1024x512.nc' for k,c in CANDIDATES.items()}
# 科学元数据属于复算输出的一部分；不修改地形值。
from netCDF4 import Dataset
for key in ['slow','fast']:
    run=json.loads((DATA/CANDIDATES[key]/'run.json').read_text(encoding='utf8'))
    with Dataset(paths[key],'a') as ds:
        values=np.asarray(ds['elevation'][:])
        ds.setncatts(dict(method_class_elevation='parameterized_physics',
            method_class_transport='procedural_constrained',method_class_region='user_override',
            method_class_sampling='interpolated',transport_method=run['method'],
            transport_kappa_km2_myr=run['kappa_km2_myr'],transport_onset_km=40.,
            transport_scale_km=25.,transport_activation_ma=180.,
            transport_limits='unvalidated frozen graph diffusion; no thermal/stress/rheology solver; no ring/ocean spillover; no uniform thickness clip',
            candidate_status='experimental; not adopted; default C2 unchanged',
            reference='https://academic.oup.com/gji/article/162/2/575/2050588; https://doi.org/10.1130/L114.1',
            physical_ceiling_exceedance_cells_over10km=int((values>10000).sum())))
        ds['elevation'].setncattr('method_class','parameterized_physics')
        if 'U40 after 180Ma' not in ds.getncattr('source'):
            ds.setncattr('source',ds.getncattr('source')+'; U40 after 180Ma replaces ring-collapse by finite-rate experimental graph relaxation; see run.json; not a validated lower-crust solver.')
        if np.any(values>10000):
            v=ds['physical_ceiling_violation'] if 'physical_ceiling_violation' in ds.variables else ds.createVariable('physical_ceiling_violation','i1',('lat','lon'))
            v[:]=(values>10000).astype('i1')
            v.setncatts(dict(method_class='fantasy_exception',description='Model failure above 10km; not accepted as a fantasy exception or adopted world'))
z={}
for k,p in paths.items():
    with open_netcdf(p) as d:z[k]=np.asarray(d.elevation.values,float)
g=LatLonGrid.from_shape(z['ring'].shape);area=g.cell_area_km2
design=apply_options(yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8')),set())
land=build(design,g)['land']
features={f['id']:f for key in ['basins','plateaus'] for f in design[key]}
masks={k:lonlat_mask(g,features[k]['poly']) & land for k in ['B6','PL8']}
summary=dict(method='u40_comparison_v1',
    method_class={'regions':'user_override','transport':'procedural_constrained',
        'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
    resolution='L7 about 60km; output 1024x512',
    units={'elevation':'m','area':'km2','volume':'km3','time':'Ma'},
    regions={},candidates={},input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
for fid,m in masks.items():
    summary['regions'][fid]={k:dict(median_m=float(np.median(v[m])),
        p10_m=float(np.percentile(v[m],10)),p90_m=float(np.percentile(v[m],90)),
        above4500_fraction=float(area[m & (v>4500)].sum()/area[m].sum())) for k,v in z.items()}
setup_fonts();cmap,norm=elevation_cmap_norm();extent=(-180,180,-90,90)
stamp='范围/时期 user_override；输运 procedural_constrained；高程 parameterized_physics；L7约60km；输出1024×512'
def finish(fig,name):
    fig.text(.5,.005,'设计 v2 / 历史 v2；代码 '+code_version(),ha='center',fontsize=8)
    fig.savefig(OUT/name,dpi=125,bbox_inches='tight');plt.close(fig)
def region_axes(ax):
    ax.set_xlim(96,117);ax.set_ylim(20,36);ax.set_aspect(1/np.cos(np.deg2rad(28)))
    ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
    ax.plot([97,97+500/(111.2*np.cos(np.deg2rad(28)))],[20.8,20.8],color='black',lw=2)
    ax.text(99.5,21.2,'500 km（28°N）',ha='center',fontsize=8)
    for fid in ['B6','PL8']:
        xy=np.asarray(features[fid]['poly']);xy=np.vstack([xy,xy[0]])
        ax.plot(xy[:,0],xy[:,1],color='#cc145c',lw=1)
        ax.text(*xy[:-1].mean(axis=0),fid,color='#a00045',
            bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
fig,axs=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
for ax,k in zip(axs.flat,z):
    im=ax.imshow(z[k],extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
    region_axes(ax);ax.set_title(TITLES[k])
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程 / m')
ax=axs.flat[-1]
im=ax.imshow(z['fast']-z['ring'],extent=extent,origin='upper',cmap='RdBu_r',
    norm=TwoSlopeNorm(vmin=-5000,vcenter=0,vmax=5000),interpolation='nearest')
region_axes(ax);ax.set_title('U40B − U39B')
fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程变化 / m')
fig.suptitle('同一300→0Ma历史，只替换180Ma后的溢出算子\n'+stamp,fontsize=11)
finish(fig,'01_cases.png')
fig,axs=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
for color,(k,v) in zip(['#777777','#0072b2','#d55e00'],z.items()):
    for ax,lat in zip(axs[:2],[31.,25.]):
        row=int(np.argmin(abs(g.lat-lat)));s=(g.lon>=96)&(g.lon<=117)
        ax.plot(g.lon[s],v[row,s],label=TITLES[k],color=color)
        ax.set_title(f'{g.lat[row]:.2f}°N 最近像元行');ax.set_xlabel('经度 / °')
    col=int(np.argmin(abs(g.lon-106.)));s=(g.lat>=20)&(g.lat<=36)
    axs[2].plot(g.lat[s],v[s,col],label=TITLES[k],color=color)
for ax in axs:ax.set_ylabel('高程 / m');ax.grid(alpha=.2)
axs[2].set_title(f'{g.lon[col]:.2f}°E 最近像元列');axs[2].set_xlabel('纬度 / °')
axs[0].legend(fontsize=8)
fig.suptitle('固定剖面；范围相同，纵轴如实包含超限结果\n'+stamp,fontsize=11)
finish(fig,'02_profiles.png')
runs={k:json.loads((DATA/c/'run.json').read_text(encoding='utf8')) for k,c in CANDIDATES.items()}
basebudget=json.loads((DATA/CANDIDATES['ring']/'forward_v2.json').read_text(encoding='utf8'))['budget_km3']
for k in ['slow','fast']:
    run=runs[k];assert run['success'];diff=z[k]-z['ring']
    budget=json.loads((DATA/CANDIDATES[k]/'forward_v2.json').read_text(encoding='utf8'))['budget_km3']
    summary['candidates'][k]=dict(
        global_changed_area_over100m_km2=float(area[abs(diff)>100].sum()),
        max_abs_change_m=float(abs(diff).max()),
        land_to_ocean_km2=float(area[(z['ring']>0)&(z[k]<=0)].sum()),
        ocean_to_land_km2=float(area[(z['ring']<=0)&(z[k]>0)].sum()),
        area_over10km_km2=float(area[z[k]>10000].sum()),
        max_elevation_m=float(z[k].max()),
        surface_step_count=len(run['step_ledger']),
        max_abs_transport_balance_km3=max(abs(r['balance_km3']) for r in run['step_ledger']),
        max_local_residual_km3=max(r['max_local_residual_km3'] for r in run['step_ledger']),
        max_native_T_km=max(r['max_T_km'] for r in run['step_ledger']),
        elapsed_s=run['elapsed_s'],process_lifetime_peak_wset_bytes=run['process_lifetime_peak_wset_bytes'],
        budget_delta_vs_u39_km3={f:budget[f]-basebudget.get(f,0) for f in budget},
        surface_totals={r:{f:sum(s['regions'][r][f] for s in run['step_ledger'])
            for f in ['erosion_removed_km3','lateral_net_in_km3','relaxation_net_in_km3']}
            for r in ['B6','PL8','other_continent','other_cells']})
fig,axs=plt.subplots(2,4,figsize=(21,9),constrained_layout=True)
for row,k in enumerate(['slow','fast']):
    run=runs[k];t=np.array([s['t_ma'] for s in run['step_ledger']])
    for ax,r in zip(axs[row],['B6','PL8','other_continent','other_cells']):
        for f,lab,sign in [('erosion_removed_km3','剥蚀移除',-1),
            ('lateral_net_in_km3','原侧流净进入',1),('relaxation_net_in_km3','新松弛净进入',1)]:
            ax.plot(t,np.cumsum([sign*s['regions'][r][f]/1e6 for s in run['step_ledger']]),label=lab)
        ax.set_xlim(180,0);ax.grid(alpha=.2);ax.set_title(TITLES[k]+' · '+r)
        ax.set_xlabel('时间 / Ma');ax.set_ylabel('累计体积 / 百万km³')
axs[0,0].legend(fontsize=8)
fig.suptitle('仅表层算子，按各时刻设计窗口分区；非物源追踪、不含碰撞等来源\n新松弛不向洋壳搬运，旧环形算子可向洋壳溢出；procedural_constrained',fontsize=11)
finish(fig,'03_surface_ledger.png')
mesh=load_mesh(7);raster=Raster(mesh,512)
fig,axs=plt.subplots(2,5,figsize=(22,8),constrained_layout=True)
prefix={}
for row,k in enumerate(['ring','fast']):
    c=CANDIDATES[k]
    with np.load(ROOT/f'.cache/tectonics/design/candidates/{c}/forward_v2_v2_L7.npz') as frames:
        rep=json.loads((DATA/c/'forward_v2.json').read_text(encoding='utf8'))
        sea={r['t']:r['sea_level_m'] for r in rep['sea_level_m']}
        for ax,t in zip(axs[row],[300.,200.,100.,50.,0.]):
            i=int(np.argmin(abs(frames['times']-t)))
            elev=raster(frames[f'elev_{i}']).astype(float)-sea[t]
            im=ax.imshow(elev,extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
            ax.set_title(f'{TITLES[k]} · {t:g}Ma',fontsize=9)
            ax.set_xticks([-120,0,120]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.2)
fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.04,pad=.1,label='相对当时海面高程 / m；超色带值见超限报表')
fig.suptitle('真实关键年代；固定L7场采样，海面来自同次演化\n'+stamp,fontsize=11)
finish(fig,'04_history.png')
with np.load(ROOT/f".cache/tectonics/design/candidates/{CANDIDATES['ring']}/forward_v2_v2_L7.npz") as prior:
    for k in ['slow','fast']:
        with np.load(ROOT/f'.cache/tectonics/design/candidates/{CANDIDATES[k]}/forward_v2_v2_L7.npz') as frames:
            ids=np.flatnonzero(prior['times']>180)
            fields=['elev','T','age','t_form','oro_t','plate','ctype','origin','oro','prov','dyn']
            unequal=[f'{f}_{i}' for i in ids for f in fields if not np.array_equal(prior[f'{f}_{i}'],frames[f'{f}_{i}'],equal_nan=True)]
            summary['candidates'][k]['before_activation_unequal_arrays']=unequal
            summary['candidates'][k]['before_activation_compared_arrays']=len(ids)*len(fields)
fig,axs=plt.subplots(2,2,figsize=(17,10),constrained_layout=True)
for col,k in enumerate(['slow','fast']):
    im=axs[0,col].imshow(z[k],extent=extent,origin='upper',cmap=cmap,norm=norm,interpolation='nearest')
    axs[0,col].set_title(TITLES[k]+'：真实世界')
    fig.colorbar(im,ax=axs[0,col],orientation='horizontal',fraction=.04,pad=.07,label='高程 / m')
    im=axs[1,col].imshow(z[k]-z['ring'],extent=extent,origin='upper',cmap='RdBu_r',interpolation='nearest',
        norm=TwoSlopeNorm(vmin=-5000,vcenter=0,vmax=5000))
    axs[1,col].set_title('相对U39B的全局变化；色带外值见报表')
    fig.colorbar(im,ax=axs[1,col],orientation='horizontal',fraction=.04,pad=.07,label='变化 / m')
for ax in axs.flat:
    ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
fig.suptitle('局部改善的全局代价；不截除不理想区域\n'+stamp,fontsize=11)
finish(fig,'05_global_cost.png')
# 复查旧r8，避免随机窗口变化掩盖旧问题。
from mythic.figures.region_panels import region_pair
from mythic.figures.tectonic_panels import region_tectonics
from mythic.figures.select_regions import RegionPick
record=json.loads((ROOT/'figures/stage1/u38_b6_pl8_transition/regions.json').read_text(encoding='utf8'))['regions'][-1]
pick=RegionPick(key='u38_r8_fixed',title='U38旧问题海岸固定复查',rule='保留旧r8',earth_ref=None,
    center=tuple(record['center_lat_lon']),size_km=record['size_km'],reason='旧问题不因抽样变化关闭')
world=load(str(paths['fast']));earth=load_earth(world.resolution)
region_pair(pick,world,earth,record['earth_comparison'],OUT/'06_previous_r8.png',8)
region_tectonics(pick,world,OUT/'07_previous_r8_tectonics.png',8)
summary['limits']=['two kappa values are sensitivity hypotheses, not calibrated physics',
    'old ring barrier bypass and ocean recipients change with operator replacement',
    'old r2/r8 and material provenance/history issues remain',
    'ceilings refer to diagnostic failure; no fantasy exception accepted']
text=json.dumps(summary,ensure_ascii=False,indent=2)
(OUT/'comparison.json').write_text(text,encoding='utf8')
(DATA/CANDIDATES['fast']/'comparison.json').write_text(text,encoding='utf8')
fig,axs=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
for ax,k in zip(axs,z):
    im=ax.imshow(z[k],extent=extent,origin='upper',cmap='magma',vmin=0,vmax=22000,interpolation='nearest')
    bad=z[k]>10000
    yy,xx=np.nonzero(bad)
    ax.scatter(g.lon[xx],g.lat[yy],s=35,facecolors='none',edgecolors='cyan',lw=1)
    ax.set_xlim(128,150);ax.set_ylim(-15,5);ax.set_aspect(1)
    ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2);ax.set_title(TITLES[k])
    ax.plot([130,130+500/(111.2*np.cos(np.deg2rad(-5)))],[-13,-13],color='cyan',lw=2)
    ax.text(132.2,-12.2,'500 km',color='cyan',ha='center')
fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.04,pad=.12,label='高程 / m（诊断色带0–22000m，不替换标准图色带）')
fig.suptitle('东南岛弧物理上限失败：青圈标出>10km像元；最高21371m\n高程 parameterized_physics；超限格标 fantasy_exception，但未接受为幻想设定；L7约60km',fontsize=11)

finish(fig,'08_ceiling_violations.png')
summary['physical_ceiling']={'max_peak_soft_limit_m':10000,'ceiling_violation_method_class':'fantasy_exception',
    'accepted_as_fantasy':False,'note':'diagnostic model failure, not an adopted world'}
names=['00_spike.png','01_cases.png','02_profiles.png','03_surface_ledger.png','04_history.png',
       '05_global_cost.png','06_previous_r8.png','07_previous_r8_tectonics.png']
text=json.dumps(summary,ensure_ascii=False,indent=2)
(OUT/'comparison.json').write_text(text,encoding='utf8')
(DATA/CANDIDATES['fast']/'comparison.json').write_text(text,encoding='utf8')
names.append('08_ceiling_violations.png')
html='<!doctype html><meta charset="utf-8"><title>U40 分配机制对照</title><h1>U40：有限速率厚度松弛</h1><p>未经标定的机制试验；默认C2不变，阶段1未通过。超物理上限结果为模型失败，不当作已接受的幻想例外。</p>'
for name in names:html+=f'<p><img style="width:100%" src="{name}"></p>'
html+='<p><a href="../../../../stage1/u40_relax_10000/index.html">29张标准图组</a></p>'
(OUT/'index.html').write_text(html,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
