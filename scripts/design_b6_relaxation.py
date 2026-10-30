"""U40：U39联合案仅180Ma后把环形溢出改为有限速率厚度松弛。

范围、时期、强度为设计假设；不修改默认引擎或终态高程。
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_b6_halo as halo
import design_forward as fwd
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import CONT
from mythic.tectonics.relaxation_spike import relax_thickness

KAPPA=3000.
ledger=[]
original=Engine._surface


def candidate_surface(self,dt,pos=None):
    if self.t>180.:return original(self,dt,pos)
    c,p=self.c,self.p
    pos=self.positions() if pos is None else pos
    cont=self.alive & (c.ctype==CONT)
    ids=np.flatnonzero(cont & (c.plate==101))
    future=pos[ids] @ halo.previous.kin.total_from(101,float(self.t),0.).T
    wb,wp,inside=halo.region_weights(future)
    weight=np.zeros(len(c)); weight[ids]=np.maximum(wb,wp)
    masks={}
    for name,local in zip(('B6','PL8'),inside):
        mask=np.zeros(len(c),bool);mask[ids[local]]=True;masks[name]=mask
    masks['PL8'] &= ~masks['B6']
    masks['other_continent']=cont & ~masks['B6'] & ~masks['PL8']
    masks['other_cells']=self.alive & ~cont
    step=dict(t_ma=float(self.t),regions={})
    ex=np.where(cont,np.maximum(c.T-p.T_cont,0.),0.)
    if p.erosion_law=='relief':loss=np.minimum(self.erosion_rate(pos)*dt,ex)
    elif p.erosion_law=='linear':loss=ex*dt/p.tau_oro_myr
    else:raise ValueError('unknown erosion law')
    c.T-=loss
    self.budget['eroded']+=float(np.dot(loss,self.area))
    i,j=self.pedges.T
    eligible=cont[i] & cont[j]
    if p.flow_min_T_km is not None:
        eligible &= np.minimum(c.T[i],c.T[j])>=p.flow_min_T_km
    old=eligible & (np.maximum(c.T[i],c.T[j])>p.flow_T_km)
    dv=np.zeros(len(c))
    if old.any():
        a,b=i[old],j[old]
        coef=p.kappa_km2_myr*dt*2/(3*self.m.h_km**2)
        flux=coef*(c.T[b]-c.T[a])*.5*(self.area[a]+self.area[b])
        flux*=1-.99*np.maximum(weight[a],weight[b])
        dv=np.bincount(a,flux,minlength=len(c))-np.bincount(b,flux,minlength=len(c))
        c.T+=dv/self.area
    # 真实有限边流；没有虚拟占容、跨障碍跳跃、洋壳接收或统一截顶。
    e=self.pedges[eligible]
    mobility=1-np.maximum(weight[e[:,0]],weight[e[:,1]])
    before=c.T.copy()
    new,transport,info=relax_thickness(c.T,self.area,e,mobility,dt,self.m.h_km,KAPPA)
    c.T[:]=new
    if abs(info['balance_km3'])>.01 or info['max_local_residual_km3']>1e-4:
        raise RuntimeError(f'relaxation residual: {info}')
    if new.min()<before.min()-1e-8 or new.max()>before.max()+1e-8:
        raise RuntimeError('relaxation new extremum')
    for name,mask in masks.items():
        step['regions'][name]=dict(cells=int(mask.sum()),area_km2=float(self.area[mask].sum()),
            erosion_removed_km3=float(np.dot(loss[mask],self.area[mask])),
            lateral_net_in_km3=float(dv[mask].sum()),
            relaxation_net_in_km3=float(transport[mask].sum()))
    step.update(info)
    step.update(max_T_km=float(c.T[cont].max()),
        area_above70km_km2=float(self.area[cont & (c.T>70)].sum()))
    ledger.append(step)
    if int(self.t)%10==0:print('U40',self.t,info,'max_T',step['max_T_km'],flush=True)


def main():
    global KAPPA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kappa',type=float,choices=[3000.,10000.],default=3000.)
    KAPPA=ap.parse_args().kappa
    candidate=f'u40_relax_{int(KAPPA)}'
    cache=ROOT/'.cache/u40_relaxation'/candidate;cache.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();success=False
    sources=[Path(__file__),ROOT/'packages/tectonics/relaxation_spike.py',
        ROOT/'scripts/design_b6_halo.py',ROOT/'scripts/design_b6_transition.py',
        ROOT/'packages/tectonics/engine.py',ROOT/'scripts/design_forward.py',
        ROOT/'config/design/final_map_v2.yaml',
        ROOT/'config/design/history_v2_u35_c10_progressive.yaml',
        ROOT/'data/worlds/design_v2/history_v2.rot',
        ROOT/'.cache/tectonics/design/initial_v2_v2_L7.pkl']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    Engine._surface=candidate_surface;saved=sys.argv[:]
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
        '--candidate',candidate,'--no-figures']
    try:
        fwd.main();success=True
    finally:
        Engine._surface=original;sys.argv=saved
        report=dict(success=success,candidate=candidate,method='u40_frozen_graph_relaxation_v1',
            method_class={'region':'user_override','transport':'procedural_constrained',
                'elevation':'parameterized_physics','sampling':'interpolated','figures':'visual_only'},
            resolution='L7 about 60 km; output 1024x512',
            units={'thickness':'km','time':'Ma','volume':'km3','diffusivity':'km2/Myr','elevation':'m'},
            kappa_km2_myr=KAPPA,onset_km=40.,scale_km=25.,
            assumptions=['U39 joint weights, 180Ma activation, old erosion/lateral flow retained',
                'only ring-collapse operator replaced after 180Ma; prior steps unchanged',
                'frozen start-step cubic thickness response; backward Euler via SciPy spsolve',
                'edge mobility=1-max(region weights); no blocked-edge tunneling or ocean reception',
                'no thickness clipping; only existing eligible continental edges'],
            limits=['not calibrated channel flow; no thermal, stress, rheology or flexure solver',
                'graph uses existing mean 60km spacing, not exact spherical dual face flux',
                'ring barrier bypass and ocean conversion removed as part of operator replacement',
                'thin/disconnected closed domains may retain excessive thickness',
                'regional net flux is not provenance; old rift source/world issues remain'],
            references=['https://academic.oup.com/gji/article/162/2/575/2050588',
                'https://doi.org/10.1130/L114.1',
                'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html'],
            input_sha256=hashes,step_ledger=ledger,elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (cache/'run.json').write_text(text,encoding='utf8')
        if success:
            dest=ROOT/'data/worlds/design_v2/candidates'/candidate
            (dest/'run.json').write_text(text,encoding='utf8')
        print('U40_END',candidate,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
