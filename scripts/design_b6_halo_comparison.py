"""U39 同窗、多年代、剖面与表层算子收支对照。运行后出图，保留失败/成功记录。"""
from pathlib import Path
import hashlib
import json
import sys
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
from mythic.figures.world import code_version
from mythic.metrics.grid import LatLonGrid
from mythic.tectonics.mesh import load_mesh,Raster
from design_final_map import apply_options,build,lonlat_mask
import design_b6_halo as method

OUT=ROOT/'figures/design/v2/forward/u39_halo'
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/worlds/design_v2/candidates'
CASES={'control':('.cache/u36_s3/world.nc','原控制'),
       'u37':('data/worlds/design_v2/candidates/u37_b6_structured/design_v2_forward_v2_0Ma_u37_b6_structured_1024x512.nc','U37：B6 硬边'),
       'u38':('data/worlds/design_v2/candidates/u38_b6_pl8_transition/design_v2_forward_v2_0Ma_u38_b6_pl8_transition_1024x512.nc','U38：两区内侧过渡'),
       'basin_only':('data/worlds/design_v2/candidates/u39_b6_halo/design_v2_forward_v2_0Ma_u39_b6_halo_1024x512.nc','U39A：B6 外侧过渡'),
       'joint':('data/worlds/design_v2/candidates/u39_b6_pl8_halo/design_v2_forward_v2_0Ma_u39_b6_pl8_halo_1024x512.nc','U39B：外侧过渡 + PL8')}
CANDIDATES={'basin_only':'u39_b6_halo','joint':'u39_b6_pl8_halo'}
z={}
for key,(path,title) in CASES.items():
    with open_netcdf(ROOT/path) as d:z[key]=np.asarray(d.elevation.values,float)
g=LatLonGrid.from_shape(z['control'].shape)
area=g.cell_area_km2
design=apply_options(yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8')),set())
land=build(design,g)['land']
features={f['id']:f for k in ['basins','plateaus'] for f in design[k]}
masks={k:lonlat_mask(g,features[k]['poly']) & land for k in ['B6','PL8']}
summary={'method_class':{'regions':'user_override','crust':'procedural_constrained','elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
         'method':'u39_comparison_v1','resolution':'L7 about 60 km; output 1024x512',
         'units':{'elevation':'m','area':'km2','volume':'km3','time':'Ma'},
         'input_sha256':{k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for k,(p,_) in CASES.items()},
         'regions':{},'candidates':{}}
for fid,m in masks.items():
    summary['regions'][fid]={k:dict(median_m=float(np.median(v[m])),p10_m=float(np.percentile(v[m],10)),p90_m=float(np.percentile(v[m],90)),
        above4500_fraction=float(area[m & (v>4500)].sum()/area[m].sum())) for k,v in z.items()}
setup_fonts()
cmap,norm=elevation_cmap_norm()
extent=(-180,180,-90,90)
stamp='范围/强度/时间为设计假设；地壳 procedural_constrained；高程 parameterized_physics；L7 约60km；单位m'

def region_axes(ax):
    ax.set_xlim(96,117);ax.set_ylim(20,36)
    ax.set_aspect(1/np.cos(np.deg2rad(28)))
    ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °')
    ax.grid(alpha=.2)
    ax.plot([97,97+500/(111.2*np.cos(np.deg2rad(28)))],[20.8,20.8],color='black',lw=2)
    ax.text(99.5,21.2,'500 km（28°N）',ha='center',fontsize=8)
    for fid in ['B6','PL8']:
        xy=np.asarray(features[fid]['poly']);xy=np.vstack([xy,xy[0]])
        ax.plot(xy[:,0],xy[:,1],color='#cc145c',lw=1)
        ax.text(*xy[:-1].mean(axis=0),fid,color='#a00045',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))

fig,axs=plt.subplots(2,3,figsize=(18,11),constrained_layout=True)
for ax,(key,(_,title)) in zip(axs.flat,CASES.items()):
    im=ax.imshow(z[key],extent=extent,origin='upper',interpolation='nearest',cmap=cmap,norm=norm)
    region_axes(ax);ax.set_title(title)
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程 / m')
ax=axs.flat[-1]
im=ax.imshow(z['joint']-z['control'],extent=extent,origin='upper',interpolation='nearest',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5000,vcenter=0,vmax=5000))
region_axes(ax);ax.set_title('U39B − 原控制')
fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07,label='高程变化 / m')
fig.suptitle('B6 与 PL8：五次真实正向演化同窗对照\n'+stamp,fontsize=12)
fig.text(0.5,-0.02,'设计 v2 / 历史 v2；代码 '+code_version(),ha='center',fontsize=9)
fig.savefig(OUT/'01_cases.png',dpi=130,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
colors=['#777777','#0072b2','#d55e00','#009e73','#cc79a7']
for color,(key,(_,title)) in zip(colors,CASES.items()):
    for ax,lat in zip(axs[:2],[31.,25.]):
        row=int(np.argmin(abs(g.lat-lat)))
        select=(g.lon>=96)&(g.lon<=117)
        ax.plot(g.lon[select],z[key][row,select],label=title,color=color,lw=1.5)
        ax.set_title(f'{g.lat[row]:.2f}°N 最近像元行');ax.set_xlabel('经度 / °')
    col=int(np.argmin(abs(g.lon-106.)))
    select=(g.lat>=20)&(g.lat<=36)
    axs[2].plot(g.lat[select],z[key][select,col],label=title,color=color,lw=1.5)
for ax in axs:
    ax.set_ylabel('高程 / m');ax.set_ylim(-1000,6000);ax.grid(alpha=.2)
axs[2].set_title(f'{g.lon[col]:.2f}°E 最近像元列');axs[2].set_xlabel('纬度 / °')
axs[0].legend(fontsize=8)
fig.suptitle('同位置地形剖面；输出采样分辨率限制，不代表河谷或断层\n'+stamp,fontsize=11)
fig.text(0.5,-0.02,'设计 v2 / 历史 v2；代码 '+code_version(),ha='center',fontsize=9)
fig.savefig(OUT/'02_profiles.png',dpi=130,bbox_inches='tight');plt.close(fig)

runs={}
for key,candidate in CANDIDATES.items():
    run=json.loads((DATA/candidate/'run.json').read_text(encoding='utf8'))
    assert run['success']
    for step in run['step_ledger']:
        residual=step['collapse_balance_km3']-sum(v['collapse_net_in_km3'] for v in step['regions'].values())
        step['regions']['other_cells_residual']={'erosion_removed_km3':0.0,'lateral_net_in_km3':0.0,'collapse_net_in_km3':residual}
    runs[key]=run
    diff=z[key]-z['control']
    summary['candidates'][key]=dict(global_changed_area_over100m_km2=float(area[abs(diff)>100].sum()),
        max_abs_change_m=float(abs(diff).max()),land_to_ocean_km2=float(area[(z['control']>0)&(z[key]<=0)].sum()),
        ocean_to_land_km2=float(area[(z['control']<=0)&(z[key]>0)].sum()),
        surface_step_count=len(run['step_ledger']),max_abs_collapse_closure_km3=max(abs(r['collapse_balance_km3']) for r in run['step_ledger']),
        elapsed_s=run['elapsed_s'],process_lifetime_peak_wset_bytes=run['process_lifetime_peak_wset_bytes'],
        surface_totals={region:{field:sum(r['regions'][region][field] for r in run['step_ledger'])
                      for field in ['erosion_removed_km3','lateral_net_in_km3','collapse_net_in_km3']}
                      for region in ['B6','PL8','other_continent','other_cells_residual']})

fig,axs=plt.subplots(2,4,figsize=(22,9),constrained_layout=True)
for row,(key,run) in enumerate(runs.items()):
    t=np.array([s['t_ma'] for s in run['step_ledger']])
    for ax,region in zip(axs[row],['B6','PL8','other_continent','other_cells_residual']):
        for field,lab,sign in [('erosion_removed_km3','剥蚀移除',-1),('lateral_net_in_km3','侧向流动净进入',1),('collapse_net_in_km3','上限溢出净进入',1)]:
            ax.plot(t,np.cumsum([sign*s['regions'][region][field]/1e6 for s in run['step_ledger']]),label=lab)
        ax.set_xlim(180,0);ax.grid(alpha=.2);ax.set_title(f"{CASES[key][1]} · "+{'other_continent':'其他陆壳','other_cells_residual':'其余单元（差额）'}.get(region,region))
        ax.set_xlabel('时间 / Ma');ax.set_ylabel('累计体积变化 / 百万 km³')
axs[0,0].legend(fontsize=8)
fig.suptitle('仅表层算子收支：按各时刻101板块设计窗口分区，交界优先B6；不含碰撞等来源\n其余单元按全局差减陆壳分区推算，非独立物源账；procedural_constrained',fontsize=11)
fig.text(0.5,-0.02,'设计 v2 / 历史 v2；代码 '+code_version(),ha='center',fontsize=9)
fig.savefig(OUT/'03_surface_ledger.png',dpi=130,bbox_inches='tight');plt.close(fig)

mesh=load_mesh(7);raster=Raster(mesh,512)
history={}
fig,axs=plt.subplots(1,5,figsize=(22,4),constrained_layout=True)
candidate=CANDIDATES['joint']
with np.load(ROOT/f'.cache/tectonics/design/candidates/{candidate}/forward_v2_v2_L7.npz') as frames:
    report=json.loads((DATA/candidate/'forward_v2.json').read_text(encoding='utf8'))
    sea={x['t']:x['sea_level_m'] for x in report['sea_level_m']}
    for ax,t in zip(axs,[300.,200.,100.,50.,0.]):
        k=int(np.argmin(abs(frames['times']-t)))
        elev=raster(frames[f'elev_{k}']).astype(float)-sea[t]
        im=ax.imshow(elev,extent=extent,origin='upper',interpolation='nearest',cmap=cmap,norm=norm)
        ax.set_title(f'{t:g} Ma');ax.set_xticks([-120,0,120]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.2)
        history[str(t)]=dict(snapshot_index=k,sea_level_m=sea[t])
fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.09,pad=.2,label='相对当时海面高程 / m')
fig.suptitle('U39B 真实保存的关键年代：固定L7场最近邻采样；海面来自同次运行（报表取整约0.1m）\n设计运动 user_override；地壳 procedural_constrained；输出 interpolated',fontsize=11,y=1.12)
fig.text(0.5,-0.02,'设计 v2 / 历史 v2；代码 '+code_version(),ha='center',fontsize=9)
fig.savefig(OUT/'04_history.png',dpi=130,bbox_inches='tight');plt.close(fig)
summary['history']=history
summary['residual_note']='other_cells_residual = global collapse volume change minus three continental regions; includes ocean receivers at operator start, inferred algebraically, not independent source tracking'

with np.load(ROOT/'.cache/tectonics/design/candidates/u38_b6_pl8_transition/forward_v2_v2_L7.npz') as prior:
    for key,candidate in CANDIDATES.items():
        with np.load(ROOT/f'.cache/tectonics/design/candidates/{candidate}/forward_v2_v2_L7.npz') as frames:
            indices=np.flatnonzero(prior['times']>180)
            fields=['elev','T','age','t_form','oro_t','plate','ctype','origin','oro','prov','dyn']
            unequal=[f'{f}_{i}' for i in indices for f in fields if not np.array_equal(prior[f'{f}_{i}'],frames[f'{f}_{i}'],equal_nan=True)]
            summary['candidates'][key]['before_activation_unequal_arrays']=unequal
            summary['candidates'][key]['before_activation_compared_arrays']=len(fields)*len(indices)
(OUT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
(DATA/CANDIDATES['joint']/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
html='<!doctype html><meta charset="utf-8"><title>U39 B6/PL8</title><h1>U39：B6核心与外侧过渡</h1><p>两案均为未经标定的设计假设；未采用为默认世界，指标不代替视觉评审。</p>'
for name in ['01_cases.png','02_profiles.png','03_surface_ledger.png','04_history.png']:
    html+=f'<p><img style="width:100%" src="{name}"></p>'
(OUT/'index.html').write_text(html,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

# 固定复查 U38 的旧问题 r8；随机新窗口不替代旧问题。
from mythic.figures.world import load,load_earth
from mythic.figures.region_panels import region_pair
from mythic.figures.tectonic_panels import region_tectonics
from mythic.figures.select_regions import RegionPick
record=json.loads((ROOT/'figures/stage1/u38_b6_pl8_transition/regions.json').read_text(encoding='utf8'))['regions'][-1]
pick=RegionPick(key='u38_r8_fixed',title='U38旧问题海岸固定复查',rule='保留上轮r8窗口',earth_ref=None,
    center=tuple(record['center_lat_lon']),size_km=record['size_km'],reason='旧严重问题不因新随机窗口而关闭')
world=load(str(ROOT/CASES['joint'][0]))
earth=load_earth(world.resolution)
region_pair(pick,world,earth,record['earth_comparison'],OUT/'05_previous_r8.png',8)
region_tectonics(pick,world,OUT/'06_previous_r8_tectonics.png',8)
html=(OUT/'index.html').read_text(encoding='utf8')
for name in ['05_previous_r8.png','06_previous_r8_tectonics.png']:
    html+=f'<p><img style="width:100%" src="{name}"></p>'
(OUT/'index.html').write_text(html,encoding='utf8')
