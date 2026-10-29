"""真实 B6 地形候选的同窗前后对照，不改变世界或阈值。"""
from pathlib import Path
import json, hashlib
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mythic.figures.netcdf import open_netcdf
from mythic.figures.style import elevation_cmap_norm,setup_fonts
from mythic.metrics.grid import LatLonGrid
from design_final_map import apply_options,build,elevation,lonlat_mask

ROOT=Path(__file__).resolve().parents[1];cache=ROOT/'.cache/u37_b6'
before=ROOT/'.cache/u36_s3/world.nc'
after=ROOT/'data/worlds/design_v2/candidates/u37_b6_structured/design_v2_forward_v2_0Ma_u37_b6_structured_1024x512.nc'
with open_netcdf(before) as d: old=np.asarray(d.elevation.values,float)
with open_netcdf(after) as d: new=np.asarray(d.elevation.values,float)
assert old.shape==new.shape==(512,1024)
g=LatLonGrid.from_shape(old.shape)
design=apply_options(yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8')),set())
b=build(design,g);target=elevation(design,g,b)
features={x['id']:x for section in ['basins','plateaus'] for x in design[section]}
diff=new-old;area=g.cell_area_km2
rows={}
for fid in ['B6','PL8']:
 m=lonlat_mask(g,features[fid]['poly']) & b['land']
 rows[fid]=dict(pixel_count=int(m.sum()),target_median_m=float(np.median(target[m])),
      control_median_m=float(np.median(old[m])),candidate_median_m=float(np.median(new[m])),
      change_median_m=float(np.median(diff[m])),area_km2=float(area[m].sum()),
      control_over_4500_fraction=float(area[m&(old>4500)].sum()/area[m].sum()),
      candidate_over_4500_fraction=float(area[m&(new>4500)].sum()/area[m].sum()))
summary=dict(method_class={'designed_region':'user_override','crust':'procedural_constrained',
      'elevation':'parameterized_physics','output_sampling':'interpolated','figures':'visual_only'},
      input_resolution='L7 about 60 km; output 1024x512',units={'elevation':'m','area':'km²'},
      window=[96,117,20,36],regions=rows,
      global_changed_area_over_100m_km2=float(area[np.abs(diff)>100].sum()),
      global_max_abs_change_m=float(np.max(np.abs(diff))),
      land_to_ocean_km2=float(area[(old>0)&(new<=0)].sum()),
      ocean_to_land_km2=float(area[(old<=0)&(new>0)].sum()),
      before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
      after_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
      note='target is schematic; candidate is full run, not default; no claim physical calibration')
out=ROOT/'figures/design/v2/forward/u37_b6';out.mkdir(parents=True,exist_ok=True)
setup_fonts();cmap,norm=elevation_cmap_norm();extent=(-180,180,-90,90)
fig,axes=plt.subplots(1,4,figsize=(20,6),constrained_layout=True)
for ax,z,title in zip(axes,[target,old,new,diff],['终图示意','原控制世界','结构阻流候选','候选 − 控制']):
 if title=='候选 − 控制':
  im=ax.imshow(z,extent=extent,origin='upper',interpolation='nearest',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5000,vcenter=0,vmax=5000))
 else: im=ax.imshow(z,extent=extent,origin='upper',interpolation='nearest',cmap=cmap,norm=norm)
 ax.set_xlim(96,117);ax.set_ylim(20,36);ax.set_aspect(1/np.cos(np.deg2rad(28)))
 for fid in ['B6','PL8']:
  xy=np.asarray(features[fid]['poly']);xy=np.vstack([xy,xy[0]])
  ax.plot(xy[:,0],xy[:,1],color='#cc145c',lw=1.2)
  ax.text(*xy[:-1].mean(axis=0),fid,color='#a00045',bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1))
 ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel('经度 / °')
 if ax is axes[0]:ax.set_ylabel('纬度 / °')
 fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.05,pad=.08,label='高程 / m' if title!='候选 − 控制' else '变化 / m')
fig.suptitle('B6 红层盆地与 PL8 喀斯特高原：真实正向演化候选\n盆地范围与强阻流是假设（user_override）；未直接修改最终高程，物理输入 L7 约 60 km',fontsize=12,y=1.06)
fig.savefig(out/'01_b6_before_after.png',dpi=150,bbox_inches='tight');plt.close(fig)
(cache/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>B6 真实区域候选</title><h1>B6 与 PL8 同窗对照</h1><p>候选未采用；盆地强度和范围由设计给定。终图高程为示意。</p><img style="width:100%" src="01_b6_before_after.png">',encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
