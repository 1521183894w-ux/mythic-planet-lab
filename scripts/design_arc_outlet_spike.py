"""U41 控制原生末态的一步出口小试；只作方法对照，不冒充完整演化。"""
from pathlib import Path
import json,pickle,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
from mythic.tectonics.arc_outlet_spike import apply_arc_outlet
from mythic.tectonics.geometry import xyz_to_lonlat
from mythic.tectonics.crust import CONT
from mythic.figures.style import setup_fonts
from mythic.figures.world import code_version

def main():
    setup_fonts()
    out=ROOT/'figures/design/v2/forward/u41_outlet';out.mkdir(parents=True,exist_ok=True)
    states={};stats={}
    for mode in ['control','native','contact']:
        with (ROOT/'.cache/u41_outlet/control/final_state.pkl').open('rb') as f:e=pickle.load(f)
        pos=e.positions();before=e.c.T.copy()
        if mode=='control':row=None
        else:row=apply_arc_outlet(e,1.,pos,contacts=mode=='contact')
        lo,la=xyz_to_lonlat(pos)
        states[mode]=(e,lo,la,row)
        stats[mode]=dict(max_T_km=float(e.c.T[e.alive&(e.c.ctype==CONT)].max()),
            change_balance_km3=float(np.dot(e.c.T-before,e.area)),outlet=row)
    fig,axs=plt.subplots(1,3,figsize=(19,6),constrained_layout=True)
    for ax,(mode,(e,lo,la,row)) in zip(axs,states.items()):
        sel=e.alive&(lo>122)&(lo<147)&(la>-10)&(la<2)
        im=ax.scatter(lo[sel],la[sel],c=e.c.T[sel],s=15,cmap='magma',vmin=7,vmax=175)
        if row:
            uidmap={int(u):i for i,u in enumerate(e.c.uid)}
            for edge in row['edges']:
                s=uidmap[edge['source_uid']];r=uidmap[edge['target_uid']]
                if sel[s]:
                    ax.plot([lo[s],lo[r]],[la[s],la[r]],color='#00bce3' if edge['kind']==0 else '#36ec62',lw=.7)
        ax.set_xlim(122,147);ax.set_ylim(-10,2);ax.set_aspect(1)
        ax.set_title({'control':'原生控制末态','native':'单步：同板块薄壳出口','contact':'单步：再加邻近陆壳接触'}[mode])
        ax.set_xlabel('经度 / °');ax.set_ylabel('纬度 / °');ax.grid(alpha=.2)
        ax.plot([123,127.5],[-9,-9],color='cyan',lw=2);ax.text(125,-8.5,'约500km',color='cyan',ha='center')
    fig.colorbar(im,ax=axs,orientation='horizontal',fraction=.05,pad=.15,label='原生地壳厚度 / km')
    fig.suptitle('U40控制0Ma原生单元，一步1Myr候选对照；青线=同板块，绿线=有限距离接触假设\n非完整世界结果；procedural_constrained；约60km；代码 '+code_version(),fontsize=11)
    fig.savefig(out/'00_native_spike.png',dpi=125,bbox_inches='tight');plt.close(fig)
    summary=dict(method_class='procedural_constrained',method='u41_native_one_step_spike_v1',
        resolution='native L7 about60km',units={'thickness':'km','volume':'km3','time':'Myr'},
        assumptions=['same frozen actual U40B 0Ma engine state; one1Myr outlet step, no plate motion or new sources',
            'contact approximation excludes center separations below0.5h and above1.5h'],
        limits=['not a full forward result','not physical calibration','volume only, no density/composition transfer'],
        cases=stats)
    (out/'spike.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    prior=np.load(ROOT/'.cache/tectonics/design/candidates/u40_relax_10000/forward_v2_v2_L7.npz')
    new=np.load(ROOT/'.cache/tectonics/design/candidates/u41_control/forward_v2_v2_L7.npz')
    unequal=[]
    for k in prior.files:
        a,b=prior[k],new[k]
        equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
        if not equal:unequal.append(k)
    summary['control_rerun_compared_array_count']=len(prior.files)
    summary['control_rerun_unequal_arrays']=unequal
    (out/'spike.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
    print('control arrays',len(prior.files),'unequal',unequal)
    print({k:{'max_T_km':v['max_T_km'],'balance_km3':v['change_balance_km3'],
        'transported_km3':v['outlet']['transported_km3'] if v['outlet'] else 0} for k,v in stats.items()})
if __name__=='__main__':main()
