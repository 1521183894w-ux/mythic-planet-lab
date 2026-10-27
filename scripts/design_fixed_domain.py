"""U36-S8：同一固定内域的来源、边界支持与柱连续性空间对照。"""
from pathlib import Path
import sys,json,pickle,time,gc,subprocess,html
import numpy as np,pygplates as pg,shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mythic.figures.style import setup_fonts
from mythic.tectonics.geometry import unit,xyz_to_lonlat,R_KM
from mythic.tectonics.material_band import point_array,spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.source_clipping import triangulate_sources,spherical_geometry_area
from mythic.tectonics.crust import CONT
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1];CACHE=ROOT/'.cache/u36_s8';DATA=ROOT/'data/worlds/design_v2';OUT=ROOT/'figures/design/v2/forward/u36_s8'


def compact_trial(result):
    """同一来源不重复写141次常量；保存可逐时重算的列式体积数据。"""
    if 'source_trajectories' in result:return result
    timeline=result['timeline'];uids=[q['uid'] for q in timeline[0]['source_rows']];columns=[]
    for uid in uids:
        seq=[next(q for q in row['source_rows'] if q['uid']==uid) for row in timeline]
        initial=seq[0]['initial_volume_km3'];assert all(q['initial_volume_km3']==initial for q in seq)
        columns.append(dict(uid=uid,initial_volume_km3=initial,volume_km3=[q['volume_km3'] for q in seq]))
    for j,row in enumerate(timeline):
        assert abs(sum(q['volume_km3'][j] for q in columns)-row['volume_km3'])<1e-6
        if j not in [0,len(timeline)-1]:row.pop('source_rows')
    result['source_trajectories']=columns
    result['storage_note']='source_trajectories与timeline共用141时间坐标；源初始体积不重复写，无数据抽样删减'
    return result


def build_domain():
    with (ROOT/'.cache/u36_s5/clipped_sources.pkl').open('rb') as f:raw=pickle.load(f)
    p=raw['projection'];band=np.load(ROOT/'.cache/u36_s4/material_band.npz');target=shapely.Polygon(p.forward(band['xyz'][band['boundary']]))
    domain=target.buffer(-2.,join_style=2);pieces=[]
    for old in raw['pieces']:
        hit=old['geometry'].intersection(domain)
        if hit.is_empty:continue
        q=dict(old);q['geometry']=hit;q['area_km2']=spherical_geometry_area(hit,p)
        q['volume_km3']=q['area_km2']*q['T_km'] if q['ctype']==CONT else None;pieces.append(q)
    area=spherical_geometry_area(domain,p);known=sum(q['area_km2'] for q in pieces if q['ctype']==CONT);volume=sum(q['volume_km3'] for q in pieces if q['ctype']==CONT)
    inventory=[]
    for old in raw['pieces']:
        if old['ctype']!=CONT:continue
        inner=next((q for q in pieces if q['uid']==old['uid']),None)
        inventory.append(dict(uid=old['uid'],original_volume_km3=old['volume_km3'],inner_volume_km3=inner['volume_km3'] if inner else 0.,reserved_outer_volume_km3=old['volume_km3']-(inner['volume_km3'] if inner else 0.)))
    report=dict(reserved_outer_inventory_note='逐UID外圈为原裁切体积减内域体积的有符号浮点差；负微差保留为几何数值残差，不是物理负库存，不静默归零',negative_reserved_uid_count=sum(q['reserved_outer_volume_km3']<0 for q in inventory),minimum_reserved_signed_difference_km3=min(q['reserved_outer_volume_km3'] for q in inventory),outer_source_inventory=inventory,method_class='procedural_constrained',reference_geometry_class='user_override',
        definition='原35Ma参考带在同一心射投影中内缩2km；仅为数值审计域，未改变200km物理假设带宽',
        fixed_domain_wkb_sha256=__import__('hashlib').sha256(domain.wkb).hexdigest(),target_area_km2=area,
        known_cont_area_km2=known,known_cont_volume_km3=volume,
        noncont_area_km2=sum(q['area_km2'] for q in pieces if q['uid']>=0 and q['ctype']!=CONT),
        uncovered_area_km2=sum(q['area_km2'] for q in pieces if q['uid']<0),
        area_partition_residual_km2=sum(q['area_km2'] for q in pieces)-area,
        reserved_outer_known_volume_km3=raw['summary']['known_cont_volume_km3']-volume,
        reserved_outer_known_area_km2=raw['summary']['known_cont_area_km2']-known,
        original_known_volume_km3=raw['summary']['known_cont_volume_km3'])
    with (CACHE/'domain.pkl').open('wb') as f:pickle.dump(dict(pieces=pieces,projection=p,domain=domain,original_target=target),f)
    return pieces,p,report


def run_trial(pieces,projection,edge,radius):
    start=time.perf_counter();v,tri,source,fragments=triangulate_sources(pieces,projection,max_edge_km=edge,return_diagnostics=True)
    h0=np.array([pieces[i]['T_km'] if pieces[i]['ctype']==CONT else np.nan for i in source]);uid=np.array([pieces[i]['uid'] for i in source]);known=np.isfinite(h0)
    centers=unit(v[tri].sum(1));pp,pt=small_material_patches(centers,radius);points=np.r_[v,pp]
    rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'));model=pg.TopologicalModel(str(ROOT/'.cache/u36_s4/material_network.gpml'),rot,topological_snapshot_cache_size=2)
    span=model.reconstruct_geometry([tuple(q) for q in points],35,time_increment=.25,deactivate_points=None)
    signed0=spherical_triangle_area(v,tri);ar0=abs(signed0);p0=spherical_triangle_area(pp,pt).reshape(-1,4).sum(1);initial=float(np.sum(ar0[known]*h0[known]));timeline=[];saved={};ever=np.zeros(len(tri),bool)
    for t in 35-np.arange(141)*.25:
        q=point_array(span.get_geometry_points(float(t)));vv=q[:len(v)];patch=q[len(v):];signed=spherical_triangle_area(vv,tri);area=abs(signed);j=signed/signed0
        jac=spherical_triangle_area(patch,pt).reshape(-1,4).sum(1)/p0;h=h0/jac
        loc=model.topological_snapshot(float(t)).get_point_locations([tuple(z) for z in patch]);outside=np.array([z.not_located_in_resolved_topology() for z in loc]);bad=outside[:len(centers)]|outside[len(centers):].reshape(-1,4).any(1);ever|=bad
        beta=np.asarray(span.get_crustal_stretching_factors(float(t)))[len(v):len(v)+len(centers)]
        source_rows=[]
        for u in np.unique(uid[known]):
            mask=known&(uid==u);v0=float(np.sum(ar0[mask]*h0[mask]));vt=float(np.sum(area[mask]*h[mask]));source_rows.append(dict(uid=int(u),initial_volume_km3=v0,volume_km3=vt,relative_residual=vt/v0-1))
        timeline.append(dict(t_ma=float(t),volume_km3=float(np.sum(area[known]*h[known])),volume_relative_residual=float(np.sum(area[known]*h[known])/initial-1),
            framework_volume_relative_residual=float(np.sum(area[known]*h0[known]/beta[known])/initial-1),
            outside_patch_count=int(bad.sum()),inverted_cells=int(np.sum(j<=0)),inverted_patches=int(np.sum(jac<=0)),
            known_thickness_quantiles_km=np.quantile(h[known],[0,.1,.5,.9,1]).tolist(),source_rows=source_rows))
        if t in [35.,20.,0.]:saved.update({f'xyz_{int(t)}Ma':vv,f'H_{int(t)}Ma':h})
    tag=f'edge{edge:g}_patch{radius:g}';np.savez_compressed(CACHE/(tag+'.npz'),vertices=v,triangles=tri,H0_km=h0,source_uid=uid,**saved)
    try:
        import psutil
        peak=getattr(psutil.Process().memory_info(),'peak_wset',None);peak=peak/2**20 if peak else None
    except ImportError:peak=None
    result=dict(tag=tag,maximum_edge_km=edge,patch_radius_km=radius,triangle_count=len(tri),known_triangle_count=int(known.sum()),
        initial_known_volume_km3=initial,numerical_fragments=fragments,all_known_domain_evaluated=not ever[known].any(),ever_outside_patch_count=int(ever.sum()),
        timeline=timeline,elapsed_s=time.perf_counter()-start,process_peak_working_set_mib=peak)
    result=compact_trial(result)
    write_json(DATA/f'u36_s8_{tag}.json',result);print(tag,'triangles',len(tri),'outside',int(ever.sum()),'final',timeline[-1]['volume_relative_residual'],'seconds',result['elapsed_s'],flush=True)
    del span,model;gc.collect();return result


def plot(report):
    setup_fonts();figs=[];version=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'+U36-S8';trials=report['trials'];domain=report['domain']
    def save(fig,name,title,note):
        fig.text(.012,.012,f'U36-S8 · {version} · pyGPlates1.0.0 / Shapely2.2.0\n原参考带 user_override；框架场 interpolated；固定域/柱连续性审计 procedural_constrained\n'+note,fontsize=8,va='bottom');fig.savefig(OUT/name,dpi=140);plt.close(fig);figs.append(dict(file=name,title=title,note=note))
    with (CACHE/'domain.pkl').open('rb') as f:p=pickle.load(f)
    fig,axs=plt.subplots(1,2,figsize=(14,9));from mythic.tectonics.source_clipping import polygons
    for geom,col,label in [(p['original_target'],'#b87354','原参考带外缘'),(p['domain'],'#277d8e','固定审计域（投影内缩2km）')]:
        for k,poly in enumerate(polygons(geom)):
            lo,la=xyz_to_lonlat(p['projection'].inverse(np.asarray(poly.exterior.coords)));axs[0].plot(lo,la,color=col,label=label if k==0 else None)
    axs[0].set(xlabel='经度 / °E',ylabel='纬度 / °N',title='35Ma：三档细分共享完全相同的域');axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12));axs[0].set_aspect(1/np.cos(np.radians(-20)))
    xl=axs[0].get_xlim();yl=axs[0].get_ylim();xx=xl[0]+.05*(xl[1]-xl[0]);yy=yl[0]+.05*(yl[1]-yl[0]);dx=100/(R_KM*np.pi/180*np.cos(np.radians(-20)))
    axs[0].plot([xx,xx+dx],[yy,yy],lw=3,color='#111');axs[0].text(xx,yy+.02*(yl[1]-yl[0]),'100km',fontsize=8)
    vals=[domain['known_cont_volume_km3'],domain['reserved_outer_known_volume_km3']];axs[1].bar([0,1],vals,color=['#277d8e','#b87354']);axs[1].set_xticks([0,1],['固定内域已知陆壳','外圈保留初始库存']);axs[1].set(ylabel='初始地壳柱体积 / km³',title='固定域之外保留来源库存')
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.35);save(fig,'99_fixed_source_domain.png','固定源域与保留外圈','2km是审计保护余量，不是物理变形带宽、断层宽度或新地壳；所有面积/体积回球面计算，不以投影面积冒充km²。')
    fig,axs=plt.subplots(1,3,figsize=(15,8))
    for q in trials:
        label=f"边{q['maximum_edge_km']:g}km / 斑块{q['patch_radius_km']:g}km";t=[z['t_ma'] for z in q['timeline']]
        axs[0].plot(t,[100*z['volume_relative_residual'] for z in q['timeline']],label=label)
    axs[0].set(xlim=(35,0),xlabel='年代 / Ma',ylabel='固定域柱体积残差 / %',title='同一源域，无动态剔除');axs[0].legend(fontsize=8)
    rows=[q for q in trials if q['patch_radius_km']==1.];axs[1].loglog([q['maximum_edge_km'] for q in rows],[abs(q['timeline'][-1]['volume_relative_residual'])*100 for q in rows],'o-');axs[1].set(xlabel='最长审计边 / km',ylabel='末态体积相对差绝对值 / %',title='同一物理输入的空间细分')
    for q in [trials[0],trials[-1]]:
        rows=q['timeline'][-1]['source_rows'];axs[2].scatter([z['initial_volume_km3'] for z in rows],[100*z['relative_residual'] for z in rows],s=15,label=q['tag'])
    axs[2].set_xscale('log');axs[2].set(xlabel='各UID初始体积 / km³',ylabel='各UID末态体积差 / %',title='来源分别核对，不能只看净残差');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.23,wspace=.35);save(fig,'100_fixed_domain_continuity.png','固定域柱连续性与空间细分','H=H0/Jac_center，体积使用独立有限三角形面积；仍是给定运动下的柱连续性假设。不是流变、真实侧向流动或完整三维物质解。')
    a=np.load(CACHE/'edge7.5_patch1.npz');fig,axs=plt.subplots(1,3,figsize=(15,9))
    for ax,t in zip(axs,[35,20,0]):
        lo,la=xyz_to_lonlat(a[f'xyz_{t}Ma']);im=ax.tripcolor(lo,la,a['triangles'],facecolors=np.ma.masked_invalid(a[f'H_{t}Ma']),vmin=0,vmax=50,cmap='viridis');ax.set_facecolor('#ccc');ax.set(xlabel='经度 / °E',ylabel='纬度 / °N',title=f'{t}Ma：固定材料整柱厚度');ax.set_aspect(1/np.cos(np.radians(np.mean(la))));ax.grid(alpha=.2)
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();xx=x0+.05*(x1-x0);yy=y0+.05*(y1-y0);dx=100/(R_KM*np.pi/180*np.cos(np.radians(np.mean(la))));ax.plot([xx,xx+dx],[yy,yy],lw=3,color='#111');ax.text(xx,yy+.02*(y1-y0),'100km',fontsize=8)
    fig.subplots_adjust(bottom=.22,wspace=.32,right=.85);fig.colorbar(im,cax=fig.add_axes([.9,.27,.015,.45]),label='整柱厚度 / km')
    save(fig,'101_fixed_material_thickness.png','固定域材料厚度','灰色为原非陆壳/无覆盖，不补陆壳。域外原材料保留库存；图为条件性整柱厚度，不是地表高程、水深、B5或新海岸。')
    write_json(DATA/'u36_s8_manifest.json',dict(checkpoint='U36-S8',version=version,figures=figs,physical_engine_changed=False))
    page='<meta charset="utf-8"><title>U36-S8 固定域连续性</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style><h1>U36-S8：固定源域与材料连续性</h1>'
    for q in figs:page+=f'<h2>{html.escape(q["title"])}</h2><img src="{q["file"]}"><p>{html.escape(q["note"])}</p>'
    (OUT/'index.html').write_text(page,encoding='utf-8')

def main():
    CACHE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if '--compact-only' in sys.argv:
        report=json.loads((DATA/'u36_s8_fixed_domain.json').read_text(encoding='utf-8'));old_hash=report['domain']['fixed_domain_wkb_sha256'];_,_,report['domain']=build_domain();assert report['domain']['fixed_domain_wkb_sha256']==old_hash
        trials=[]
        for row in report['trials']:
            path=DATA/f"u36_s8_{row['tag']}.json";result=compact_trial(json.loads(path.read_text(encoding='utf-8')));write_json(path,result);trials.append({k:v for k,v in result.items() if k!='source_trajectories'})
        report['trials']=trials;write_json(DATA/'u36_s8_fixed_domain.json',report)
    elif '--plot-only' in sys.argv:report=json.loads((DATA/'u36_s8_fixed_domain.json').read_text(encoding='utf-8'))
    else:
        start=time.perf_counter();pieces,p,domain=build_domain();trials=[run_trial(pieces,p,edge,radius) for edge,radius in [(30.,1.),(15.,1.),(7.5,1.),(7.5,.25)]]
        report=dict(checkpoint='U36-S8',method_class='procedural_constrained',framework_version=pg.__version__,domain=domain,trials=[{k:v for k,v in q.items() if k!='source_trajectories'} for q in trials],
            reference_width_km=200,dt_myr=.25,domain_unchanged_between_trials=True,physical_engine_changed=False,elapsed_s=time.perf_counter()-start,
            limitations=['固定域只是数值审计子域，不是整个真实变形带','参照系误差仍保留','无物理带宽/古接触/热/断层/水位求解，未接B5或全球'])
        write_json(DATA/'u36_s8_fixed_domain.json',report)
    plot(report)
if __name__=='__main__':main()
