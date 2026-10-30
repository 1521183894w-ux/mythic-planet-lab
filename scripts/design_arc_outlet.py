"""U41：捕获U40控制的超厚岛弧原生单元与算子输入，后续比较出口候选。

只读诊断不改变控制物理；数据procedural_constrained，图visual_only。
"""
from pathlib import Path
import argparse,hashlib,json,pickle,sys,time
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_b6_relaxation as previous
import design_forward as fwd
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import CONT,OCEAN
from mythic.tectonics.geometry import xyz_to_lonlat

CACHE=ROOT/'.cache/u41_outlet'
MODE='control'
rows=[]
original_surface=Engine._surface
original_interact=Engine._interact
original_arc=Engine._arc
original_simulate=fwd.simulate


def observed_interact(self,pos):
    before=self.c.T.copy()
    result=original_interact(self,pos)
    self._u41_collision_dT=self.c.T-before
    return result


def observed_arc(self,pos):
    before=self.c.T.copy()
    result=original_arc(self,pos)
    self._u41_arc_dT=self.c.T-before
    return result


def observed_surface(self,dt,pos=None):
    before=self.c.T.copy()
    previous.candidate_surface(self,dt,pos)
    if self.t>180.:return
    c=self.c
    pos=self.positions() if pos is None else pos
    lon,lat=xyz_to_lonlat(pos)
    select=self.alive & (c.ctype==CONT) & (c.T>65) & (lon>120)&(lon<155)&(lat>-15)&(lat<5)
    if not select.any():return
    i,j=self.pedges.T
    alive_edge=self.alive[i]&self.alive[j]
    eligible=alive_edge&(c.ctype[i]==CONT)&(c.ctype[j]==CONT)
    if self.p.flow_min_T_km is not None:
        eligible &= np.minimum(c.T[i],c.T[j])>=self.p.flow_min_T_km
    degree=np.bincount(self.pedges[eligible].ravel(),minlength=len(c))
    all_degree=np.bincount(self.pedges[alive_edge].ravel(),minlength=len(c))
    for n in np.flatnonzero(select):
        rows.append(dict(t_ma=float(self.t),uid=int(c.uid[n]),plate=int(c.plate[n]),
            origin=int(c.origin[n]),event=int(c.oro[n]),lon=float(lon[n]),lat=float(lat[n]),
            T_km=float(c.T[n]),area_km2=float(self.area[n]),eligible_degree=int(degree[n]),
            alive_native_degree=int(all_degree[n]),collision_dT_km=float(self._u41_collision_dT[n]),
            arc_dT_km=float(self._u41_arc_dT[n]),surface_dT_km=float(c.T[n]-before[n])))


def captured_simulate(*args,**kwargs):
    result=original_simulate(*args,**kwargs)
    eng,frames,vframes=result
    with (CACHE/MODE/'final_state.pkl').open('wb') as f:
        pickle.dump(eng,f,protocol=5)
    np.savez_compressed(CACHE/MODE/'native.npz',positions=eng.positions(),alive=eng.alive,
        area_km2=eng.area,edges=eng.pedges,h_km=eng.m.h_km,
        **{k:getattr(eng.c,k) for k in eng.c.FIELDS})
    return result


def main():
    global MODE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['control'],default='control')
    MODE=ap.parse_args().mode
    out=CACHE/MODE;out.mkdir(parents=True,exist_ok=True)
    candidate='u41_'+MODE
    previous.KAPPA=10000.
    Engine._interact=observed_interact;Engine._arc=observed_arc
    Engine._surface=observed_surface;fwd.simulate=captured_simulate
    saved=sys.argv[:];start=time.perf_counter();success=False
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
        '--candidate',candidate,'--no-figures']
    try:
        fwd.main();success=True
    finally:
        Engine._surface=original_surface;Engine._interact=original_interact
        Engine._arc=original_arc;fwd.simulate=original_simulate;sys.argv=saved
        report=dict(success=success,candidate=candidate,mode=MODE,
            method_class={'transport':'procedural_constrained','region':'user_override',
                'elevation':'parameterized_physics','sampling':'interpolated'},
            method='u41_native_outlet_capture_v1',resolution='L7 about60km; output1024x512',
            units={'T':'km','area':'km2','time':'Ma','operator_dT':'km'},
            assumptions=['same U40B dynamics; callbacks observe interact/arc/surface thickness deltas',
                'watched region120-155E,15S-5N; store only alive continental T>65km'],
            limits=['operator increments are not material provenance','selection omits earlier subthreshold history'],
            rows=rows,step_ledger=previous.ledger,
            elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (out/'run.json').write_text(text,encoding='utf8',newline='\n')
        if success:(ROOT/'data/worlds/design_v2/candidates'/candidate/'run.json').write_text(text,encoding='utf8',newline='\n')
        print('U41_END',MODE,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
if __name__=='__main__':main()
