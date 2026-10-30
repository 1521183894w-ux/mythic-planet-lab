"""U42：同一冻结原生地壳、仅输运20Myr的小试；不是完整地史。"""
from pathlib import Path
import hashlib,json,pickle,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_basin_strength as trial
from mythic.tectonics.crust import CONT
from mythic.tectonics.geometry import xyz_to_lonlat
from mythic.figures.style import setup_fonts
from mythic.figures.world import code_version
source=ROOT/'.cache/u41_outlet/contact/final_state.pkl'
with source.open('rb') as f:e=pickle.load(f)
c=e.c;pos=e.positions();lon,lat=xyz_to_lonlat(pos)
ids=np.flatnonzero(e.alive & (c.ctype==CONT)&(c.plate==101))
wb,wp,_=trial.base.halo.region_weights(pos[ids]);w=np.zeros(len(c));w[ids]=np.maximum(wb,wp)
i,j=e.pedges.T
cont=e.alive & (c.ctype==CONT)
ok=cont[i]&cont[j]&(np.minimum(c.T[i],c.T[j])>=e.p.flow_min_T_km)
# 冻结同一局部计算域，三案一致；边界无通量，不用于世界结论。
domain=cont&(lon>=94)&(lon<=120)&(lat>=18)&(lat<=40)
ok &= domain[i]&domain[j]
edges=e.pedges[ok];mob=1-np.maximum(w[edges[:,0]],w[edges[:,1]])
out=ROOT/'figures/design/v2/forward/u42_strength';out.mkdir(parents=True,exist_ok=True)
setup_fonts();fig,axs=plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
select=domain&(np.abs(lat-31.11)<.4);order=np.flatnonzero(select);order=order[np.argsort(lon[order])]
axs[0].plot(lon[order],c.T[order],':',color='black',label='原生初态')
rows=[]
for floor,color in zip([0.,.01,.05],['#777777','#0072b2','#d55e00']):
    trial.FLOOR=floor;T=c.T.copy();balances=[];residuals=[]
    for step in range(20):
        T,dv,info=trial.finite_strength_relax(T,e.area,edges,mob,1.,e.m.h_km)
        balances.append(info['balance_km3']);residuals.append(info['max_local_residual_km3'])
    axs[0].plot(lon[order],T[order],color=color,label=f'最低流动率 {floor:g}')
    s=domain&(w==1)
    rows.append(dict(floor=floor,steps=20,domain_cells=int(domain.sum()),eligible_edges=len(edges),
        max_balance_km3=max(abs(v) for v in balances),max_residual_km3=max(residuals),
        basin_core_net_in_km3=float(np.dot((T-c.T)[s],e.area[s])),
        basin_core_median_T_km=float(np.median(T[s]))))
    axs[1].plot(lon[order],T[order]-c.T[order],color=color,label=f'最低流动率 {floor:g}')
for ax in axs:
    ax.set_xlim(96,117);ax.set_xlabel('经度 / °');ax.set_ylabel('地壳厚度 / km')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[0].set_title('31.11°N附近原生单元；20步后')
axs[1].set_title('相对同一冻结初态的厚度变化')
fig.suptitle('只比较有限阻流机制：冻结0Ma位置、边与物源，20×1Myr；局部外边界无通量\nprocedural_constrained；非正向地史；L7约60km；原生厚度km',fontsize=11)
fig.text(.5,.005,'种子/设计v2；代码 '+code_version()+'；500km约5.25°经度（31°N）',ha='center',fontsize=8)
fig.savefig(out/'00_spike.png',dpi=125,bbox_inches='tight');plt.close(fig)
report=dict(method='u42_frozen_native_strength_spike_v1',method_class='procedural_constrained',
    resolution='L7 about60km',units={'T':'km','volume':'km3','time':'Myr'},
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    assumptions=['same0Ma native domain and edges; source and motion frozen;20oneMyrsteps',
                 'local exterior boundary closed; not a world reconstruction'],rows=rows)
(out/'spike.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2))
