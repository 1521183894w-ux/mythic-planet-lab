"""U40 一维机制对照；procedural_constrained，非真实世界或物理校准。"""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
from mythic.tectonics.relaxation_spike import relax_thickness
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import Crust,CONT
from mythic.figures.style import setup_fonts
from mythic.figures.world import code_version

def main():
    setup_fonts()
    x=np.arange(-1800,1801,60.)
    T=35+10*np.exp(-(x/450)**2);A=np.full(len(x),3600.)
    edges=np.column_stack([np.arange(len(x)-1),np.arange(1,len(x))])
    cases={}
    for name,kappa in [('ring',None),('finite_3000',3000.),('finite_10000',10000.)]:
        c=Crust(len(x));c.T[:]=T;c.ctype[:]=CONT;c.plate[:]=101
        eng=Engine.__new__(Engine);eng.c=c;eng.alive=np.ones(len(x),bool)
        eng.area=A;eng.pedges=edges;eng.p=SimpleNamespace(T_max_km=65.,arc_to_cont_T_km=31.)
        maxerr=0.;net=0.
        for t in range(80):
            supply=1.2*np.exp(-(x/180)**2);c.T+=supply;net+=float(np.dot(supply,A))
            if kappa is None:Engine._collapse(eng,eng.alive)
            else:
                c.T[:],dv,info=relax_thickness(c.T,A,edges,np.ones(len(edges)),1.,60.,kappa)
                maxerr=max(maxerr,abs(info['balance_km3']))
        cases[name]=dict(thickness_km=c.T.tolist(),
            volume_error_km3=float(np.dot(c.T-T,A)-net),max_step_error_km3=maxerr)
    out=ROOT/'figures/design/v2/forward/u40_relaxation';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,5));ax.plot(x,T,'--',color='gray',label='初始地壳')
    labels={'ring':'原环形分配 + 65km上限','finite_3000':'有限速率 3000 km²/Myr','finite_10000':'有限速率 10000 km²/Myr'}
    for k,v in cases.items():ax.plot(x,v['thickness_km'],label=labels[k])
    ax.axhline(65,color='gray',lw=.7);ax.set_xlabel('距供给中心 / km');ax.set_ylabel('地壳厚度 / km')
    ax.grid(alpha=.2);ax.legend();ax.set_title('同一80Myr高斯供给：只替换分配算子（封闭一维链，无剥蚀）')
    fig.text(.5,.01,'procedural_constrained；网格60km；体积km³；设计 v2；代码 '+code_version(),ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/'00_spike.png',dpi=135);plt.close(fig)
    summary=dict(method_class='procedural_constrained',method='u40_1d_operator_spike_v1',
        units={'thickness':'km','distance':'km','time':'Myr','volume':'km3'},resolution_km=60,
        duration_myr=80,cases=cases,x_km=x.tolist(),
        assumptions=['closed synthetic 1D chain; identical Gaussian supply; no erosion/barrier'],
        limits=['not a real world or physical validation','start-step frozen coefficients each 1Myr'])
    (out/'spike.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print({k:{f:v[f] for f in ['volume_error_km3','max_step_error_km3']} for k,v in cases.items()})
if __name__=='__main__':main()
