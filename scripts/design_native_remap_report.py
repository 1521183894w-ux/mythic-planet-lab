"""U36-S15实际交叠、厚度与覆盖损失出图；不改世界数据。"""
from pathlib import Path
import json,hashlib,subprocess,time
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from mythic.figures.style import setup_fonts
from mythic.tectonics.source_clipping import LocalGnomonic,polygons
from mythic.tectonics.native_remap import spherical_moments
from design_native_remap import write,clean
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s15';DATA=ROOT/'data/worlds/design_v2';OUT=ROOT/'figures/design/v2/forward/u36_s15'
TAGS=['35Ma_b64','0Ma_b64','0Ma_b32']


def geometry(a,prefix):
    blob=a[prefix+'_wkb_bytes'].tobytes();o=a[prefix+'_wkb_offsets']
    return [shapely.from_wkb(blob[o[k]:o[k+1]]) for k in range(len(o)-1)]


def collection(geoms,values=None,**kwargs):
    paths=[];colors=[]
    for i,g in enumerate(geoms):
        for poly in polygons(g):
            poly=shapely.orient_polygons(poly)
            rings=[]
            for ring in [poly.exterior,*poly.interiors]:
                xy=np.array(ring.coords);codes=np.full(len(xy),MPath.LINETO,dtype=np.uint8);codes[0]=MPath.MOVETO;codes[-1]=MPath.CLOSEPOLY
                rings.append(MPath(xy,codes))
            paths.append(PathPatch(MPath.make_compound_path(*rings)))
            if values is not None:colors.append(values[i])
    return PatchCollection(paths,array=np.array(colors) if values is not None else None,**kwargs)


def main():
    start=time.perf_counter();setup_fonts();OUT.mkdir(exist_ok=True)
    arrays=[np.load(CACHE/(tag+'.npz')) for tag in TAGS];rows=[json.loads((DATA/f'u36_s15_{tag}.json').read_text(encoding='utf8')) for tag in TAGS]
    figrows=[];base=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    def save(fig,name,title,note):
        fig.text(.012,.012,base+'+U36-S15 · design_v2；S11映射 interpolated；200km材料带 user_override\n'
          +'球面交叠/有限线性重建 procedural_constrained；图 visual_only；真实L7宿主约60km；非全球/B5耦合\n'+note,fontsize=8,va='bottom')
        fig.savefig(OUT/name,dpi=145);plt.close(fig)
        figrows.append(dict(file=name,title=title,note=note,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
    def map_axes(ax,a,title):
        p=LocalGnomonic(a['projection_center_xyz']);xy=p.forward(a['source_boundary_xyz'].reshape(-1,3));lo=xy.min(0);hi=xy.max(0)
        ax.set(xlim=(lo[0]-10,hi[0]+10),ylim=(lo[1]-10,hi[1]+10),xlabel='局部心射东向 / km',ylabel='局部心射北向 / km',title=title,aspect='equal');ax.grid(alpha=.15)
        # 坐标单位即km；另给200km标尺，避免形变显示掩盖尺度。
        ax.plot([lo[0]+20,lo[0]+220],[lo[1]+15]*2,color='black',lw=2);ax.text(lo[0]+120,lo[1]+22,'200 km',ha='center',fontsize=8)
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    for col,k in enumerate([0,1]):
        a=arrays[k];src=geometry(a,'source');target=geometry(a,'target');known=a['known']
        ax=axes[0,col];ax.add_collection(collection(src,known.astype(float),cmap='cividis',clim=(0,1),edgecolor='none'))
        ax.add_collection(collection(target,facecolor='none',edgecolor='black',linewidth=.3))
        ax.add_collection(collection([g for g,u in zip(target,a['target_uid']) if u<0],facecolor='#dd5555',edgecolor='none',alpha=.7))
        map_axes(ax,a,f"{rows[k]['t_ma']} Ma：蓝=未知，黄=已知，红=无原生宿主")
    a=arrays[1];target=geometry(a,'target');fraction=np.divide(a['target_known_area_km2'],a['target_area_km2'],out=np.zeros(len(target)),where=a['target_area_km2']>0)
    native=a['target_uid']>=0;shown=[g for g,u in zip(target,native) if u];gap=[g for g,u in zip(target,native) if not u]
    c=collection(shown,fraction[native],cmap='viridis',clim=(0,1),edgecolor='white',linewidth=.3);axes[1,0].add_collection(c);map_axes(axes[1,0],a,'0 Ma：已知面积 / 带内宿主裁片面积（带外未填）');fig.colorbar(c,ax=axes[1,0],label='面积分数')
    count=np.sum(a['target_source_known_area_km2']>0,axis=1)
    c=collection(shown,count[native],cmap='viridis',edgecolor='white',linewidth=.3);axes[1,1].add_collection(c);map_axes(axes[1,1],a,'0 Ma：带内每宿主的正面积来源分量数');fig.colorbar(c,ax=axes[1,1],label='原始正面积计数，含浮点碎片')
    for ax in axes[1]:ax.add_collection(collection(gap,facecolor='#dd5555',edgecolor='none',alpha=.7))
    fig.subplots_adjust(bottom=.15,hspace=.27,wspace=.24)
    save(fig,'129_native_coverage.png','真实宿主覆盖、未知与多来源','原生宿主只显示材料带内裁片；孔洞/细碎片另据WKB与面积账审计，色块不代表完整宿主厚度。')
    a=arrays[1];p=LocalGnomonic(a['projection_center_xyz']);points=a['q8_points_xyz'];xy=p.forward(points.reshape(-1,3));j=a['q8_J'];known=a['known'];reference=a['source_H0_km'][:,None]/j;t=a['q8_target_cell'];safe=np.maximum(t,0);group=np.searchsorted(a['uids'],np.maximum(a['source_uid'],a['uids'][0]))
    fields=[reference,a['source_mean_thickness_km'][:,None]+np.einsum('cni,ci->cn',points-a['source_centroid_xyz'][:,None],a['source_gradients_km']),a['target_linear_mean_km'][safe],a['target_source_linear_mean_km'][safe,group[:,None]]]
    labels=['内部H0/J参照（同一指定映射）','材料单元有限线性重建','每个真实宿主单一已知均值','保留来源标签的宿主分量均值']
    low=float(np.nanmin(reference));high=float(np.nanmax(reference));fig,axes=plt.subplots(2,2,figsize=(15,11));valid=known[:,None]&(t>=0)&(a['target_uid'][safe]>=0)
    for ax,field,label in zip(axes.ravel(),fields,labels):
        mask=(valid&np.isfinite(field)).ravel();im=ax.scatter(xy[mask,0],xy[mask,1],c=field.ravel()[mask],s=1.8,cmap='viridis',vmin=low,vmax=high,rasterized=True)
        map_axes(ax,a,label)
    fig.subplots_adjust(bottom=.16,wspace=.22,hspace=.3,right=.88);cb=fig.add_axes([.91,.25,.016,.5]);fig.colorbar(im,cax=cb,label='厚度 / km（四图同色标）')
    save(fig,'130_native_thickness.png','同一覆盖范围的厚度信息损失','仅画有真实宿主且H0已知的相同内部8×8节点；未知/无宿主留空；分量均值需要来源标签，当前引擎并未采用。')
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for row,style in zip(rows[1:],['o-','s--']):
        for method,color in [('constant','#2b76ad'),('linear','#b75a23')]:
            vals=np.sort([abs(z[method]['total_minus_material_volume_km3']) for z in row['source_summary']]);axes[0,0].plot(np.maximum(vals,1e-14),np.arange(1,len(vals)+1)/len(vals),style,color=color,ms=3,label=row['tag']+' '+method)
    axes[0,0].set(xscale='log',xlabel='逐来源 |交叠转移+未覆盖库存−材料库存| / km³',ylabel='来源累计分数',title='零相对边界通量账与交叠几何，非物理真值');axes[0,0].legend(fontsize=8)
    source=rows[1]['source_summary'];x=np.arange(len(source));fractions=[z['constant']['unmapped_volume_km3']/z['material_polygon_volume_km3']*100 for z in source]
    axes[0,1].bar(x,fractions,color='#bc554c');axes[0,1].set(xlabel='来源UID排序（全部77源）',ylabel='未转入真实宿主的已知体积 / %',title='无原生宿主库存单列，未以最近邻填洞')
    for k,(row,color) in enumerate(zip(rows[:2],['#315f93','#b76530'])):
        axes[1,0].bar(np.arange(3)+k*.3,[row['known_area_km2'],row['unknown_area_km2'],row['uncovered_native_area_km2']],width=.3,label=str(row['t_ma'])+' Ma',color=color)
    axes[1,0].set_xticks(np.arange(3)+.15,['已知材料','未知厚度材料','无原生宿主（可与前两项重叠）']);axes[1,0].set(ylabel='面积 / km²',title='未知≠零体积；无宿主≠未知来源');axes[1,0].legend()
    axes[1,1].bar([0,1,2],[r['source_partition_area_residual_km2'] for r in rows],color=['#777','#315f93','#b76530']);axes[1,1].set_xticks([0,1,2],TAGS);axes[1,1].set(ylabel='Σ材料多边形面积−并集面积 / km²',title='保存边界离散后存在微小重叠，未归一化消除')
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.subplots_adjust(bottom=.19,hspace=.4,wspace=.25)
    save(fig,'131_native_inventory.png','逐来源交叠转移、未覆盖与边界重叠','曲边/宿主分片残差保留；对数图中零差仅显示为1e−14km³。重叠面积不是物理碰撞或通量，不据此修正库存。')
    methods=['material_constant','material_linear','host_constant','host_linear','host_source_constant','host_source_linear']
    names=['材料常数','材料线性','宿主单值\n常数转移','宿主单值\n线性转移','来源分量\n常数转移','来源分量\n线性转移']
    fig,axes=plt.subplots(1,2,figsize=(15,7));r4,r8=rows[1]['audit'];xx=np.arange(len(methods))
    vals=[r8[m+'_host_covered_L1']*100 for m in methods];axes[0].bar(xx,vals,color=['#5f89ad']*2+['#bc554c']*2+['#529075']*2)
    axes[0].set_xticks(xx,names,fontsize=8);axes[0].set(ylabel='同一已知且有真实宿主范围的L1 / %',title='体积账可保留，宿主单值仍损失厚度信息')
    for k,m in enumerate(methods):axes[0].text(k,vals[k]+.02,f'{vals[k]:.3f}',ha='center',fontsize=9)
    axes[1].bar(xx,[abs(r8[m+'_host_covered_L1']-r4[m+'_host_covered_L1'])*100 for m in methods],color='#c99c4d')
    axes[1].set_xticks(xx,names,fontsize=8);axes[1].set(ylabel='4/8阶L1差 / 百分点',title='间断宿主边界的有限采样敏感性，不是严格误差界')
    for ax in axes:ax.grid(alpha=.15)
    fig.subplots_adjust(bottom=.24,wspace=.28)
    save(fig,'132_native_field_loss.png','来源混合与宿主内平均的误差对照','所有柱共用同一有宿主参照库存作分母；未覆盖库存另外列账。首轮将未覆盖区并成伪宿主的图口径已弃用，原始试算留缓存。')
    # 从WKB独立布尔关系核准重叠，不修改保存多边形。
    overlaps=[]
    for tag,a in zip(TAGS,arrays):
        src=geometry(a,'source');projection=LocalGnomonic(a['projection_center_xyz']);pairs=shapely.STRtree(src).query(src,predicate='intersects');pairs=pairs[:,pairs[0]<pairs[1]];items=[]
        for i,j in pairs.T:
            area=spherical_moments(src[i].intersection(src[j]),projection)[0]
            if area>0:items.append(dict(cell_i=int(i),cell_j=int(j),area_km2=area,uid_i=int(a['source_uid'][i]),uid_j=int(a['source_uid'][j])))
        overlaps.append(dict(tag=tag,positive_pair_count=len(items),pairwise_overlap_area_sum_km2=sum(v['area_km2'] for v in items),largest_pairs=sorted(items,key=lambda v:v['area_km2'],reverse=True)[:10]))
    verification=dict(checkpoint='U36-S15',method_class='procedural_constrained',overlaps=overlaps,source_refinement_not_performed=True)
    paths=list(json.loads((DATA/'u36_s14_verification.json').read_text(encoding='utf8'))['protected'])+['packages/tectonics/material_fv.py','scripts/design_material_fv.py','scripts/design_material_fv_report.py','tests/tectonics/test_material_fv.py']
    protected={p:subprocess.check_output(['git','hash-object',p],text=True).strip()==subprocess.check_output(['git','rev-parse','b098170:'+p],text=True).strip() for p in paths}
    assert all(protected.values());verification['protected']=protected;verification['protected_baseline_commit']='b098170'
    if (ROOT/'figures/stage1/u36_s15_reference/metrics.json').exists():
        def prune(q):
            if isinstance(q,dict):return {k:prune(v) for k,v in q.items() if k!='generated' and 'timing' not in k.lower() and 'time' not in k.lower() and 'elapsed' not in k.lower()}
            if isinstance(q,list):return [prune(v) for v in q]
            return q
        old=json.loads((ROOT/'figures/stage1/u36_s14_reference/metrics.json').read_text(encoding='utf8'));new=json.loads((ROOT/'figures/stage1/u36_s15_reference/metrics.json').read_text(encoding='utf8'))
        verification['standard_metrics_equal_excluding_time']=prune(old)==prune(new)
        a0=json.loads((ROOT/'figures/stage1/u36_s14_reference/regions.json').read_text(encoding='utf8'));a1=json.loads((ROOT/'figures/stage1/u36_s15_reference/regions.json').read_text(encoding='utf8'));verification['all_region_windows_equal']=a0['regions']==a1['regions']
        for name in ['manifest','metrics','regions']:(DATA/f'u36_s15_reference_{name}.json').write_bytes((ROOT/f'figures/stage1/u36_s15_reference/{name}.json').read_bytes())
    write(DATA/'u36_s15_verification.json',verification)
    write(DATA/'u36_s15_manifest.json',dict(checkpoint='U36-S15',method_class='visual_only',figures=figrows,elapsed_s=time.perf_counter()-start))
    cards=''.join('<h2>'+r['title']+'</h2><img style="max-width:100%" src="'+r['file']+'"><p>'+r['note']+'</p>' for r in figrows)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>U36-S15 局部宿主重映射</title><h1>U36-S15：局部方法检查点，阶段1未通过</h1>'+cards,encoding='utf8')
    print('REPORT_FINISHED',time.perf_counter()-start,verification,flush=True)
if __name__=='__main__':main()
