"""U42：U41B完全阻流改为有限强度；procedural_constrained，默认不变。"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import psutil
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_arc_outlet_trial as trial
import design_arc_outlet as capture
import design_b6_relaxation as base
from mythic.tectonics.engine import Engine
FLOOR=.01
original_relax=base.relax_thickness

def finite_strength_relax(thickness,area,edges,mobility,dt_myr,h_km,kappa_km2_myr=10000.):
    """m_new=floor+(1-floor)*m；不将流动系数比称为测量黏度。"""
    if not np.isfinite(FLOOR) or not 0<=FLOOR<=1:
        raise ValueError('invalid floor')
    m=np.asarray(mobility,float)
    if not np.isfinite(m).all() or np.any((m<0)|(m>1)):
        raise ValueError('invalid original mobility')
    return original_relax(thickness,area,edges,FLOOR+(1-FLOOR)*m,dt_myr,h_km,kappa_km2_myr)

def main():
    global FLOOR
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--floor',type=float,choices=[0.,.01,.05],default=.01)
    FLOOR=ap.parse_args().floor
    candidate=f'u42_strength_{int(round(FLOOR*100)):02d}'
    cache=ROOT/'.cache/u42_strength'/candidate
    cache.mkdir(parents=True,exist_ok=True)
    capture.MODE=candidate;capture.CACHE=ROOT/'.cache/u42_strength'
    base.KAPPA=10000.;trial.CONTACTS=True
    base.relax_thickness=finite_strength_relax
    base.candidate_surface=trial.surface_with_outlet
    Engine._surface=capture.observed_surface
    Engine._interact=capture.observed_interact;Engine._arc=capture.observed_arc
    capture.fwd.simulate=capture.captured_simulate
    sources=[Path(__file__).resolve()]+[ROOT/p for p in [
        'scripts/design_arc_outlet_trial.py','scripts/design_arc_outlet.py',
        'scripts/design_b6_relaxation.py','scripts/design_b6_halo.py',
        'scripts/design_b6_transition.py','scripts/design_forward.py',
        'packages/tectonics/engine.py','packages/tectonics/relaxation_spike.py',
        'packages/tectonics/arc_outlet_spike.py','packages/tectonics/borrowing.py',
        'config/design/history_v2_u35_c10_progressive.yaml','config/design/final_map_v2.yaml',
        'data/worlds/design_v2/history_v2.rot','.cache/tectonics/design/initial_v2_v2_L7.pkl']]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    saved=sys.argv[:];start=time.perf_counter();success=False
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
        '--candidate',candidate,'--no-figures']
    try:
        capture.fwd.main()
        nc=ROOT/f'data/worlds/design_v2/candidates/{candidate}/design_v2_forward_v2_0Ma_{candidate}_1024x512.nc'
        with Dataset(nc,'a') as ds:
            z=np.asarray(ds['elevation'][:])
            ds.setncatts(dict(method_class_elevation='parameterized_physics',
                method_class_transport='procedural_constrained',method_class_region='user_override',
                method_class_sampling='interpolated',transport_method='u42_finite_strength_v1',
                transport_mobility_floor=FLOOR,transport_activation_ma=180.,
                transport_limits='finite mobility hypothesis; inherited U41 approximate contact outlet',
                candidate_status='experimental; default C2 unchanged',
                physical_ceiling_exceedance_cells_over10km=int((z>10000).sum())))
            ds['elevation'].setncattr('method_class','parameterized_physics')
            if np.any(z>10000):
                v=ds.createVariable('physical_ceiling_violation','i1',('lat','lon'))
                v[:]=(z>10000).astype('i1')
                v.setncatts(dict(method_class='fantasy_exception',description='Unaccepted model failure'))
        success=True
    finally:
        base.relax_thickness=original_relax;base.candidate_surface=trial.BASE
        Engine._surface=capture.original_surface;Engine._interact=capture.original_interact
        Engine._arc=capture.original_arc;capture.fwd.simulate=capture.original_simulate
        sys.argv=saved
        report=dict(success=success,candidate=candidate,method='u42_finite_strength_v1',
            method_class={'transport':'procedural_constrained','regions':'user_override',
                'elevation':'parameterized_physics','sampling':'interpolated'},
            resolution='L7 about60km; output1024x512',
            units={'T':'km','volume':'km3','elevation':'m','time':'Ma'},
            mobility_floor=FLOOR,kappa_km2_myr=10000.,
            assumptions=['same U41B motion/source/erosion/explicit flow/contact outlet',
                'only implicit continental mobility changes after180Ma',
                'm_new=floor+(1-floor)*m_old; same eligible edges and250km masks',
                'floor0/.01/.05 are sensitivity hypotheses; no height clipping or end editing'],
            limits=['not calibrated viscosity; old explicit flow retains1percent floor',
                'no stress/heat/density/composition/flexure/deposition solver',
                'U41 approximate contact; full rift/material source problem remains',
                'r2/oldr8 causal evidence and plateau interior remain unresolved'],
            references=['https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JB005457',
                'https://academic.oup.com/gji/article/162/2/575/2050588',
                'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html'],
            input_sha256=hashes,step_ledger=base.ledger,outlet_ledger=trial.outlet_ledger,
            elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (cache/'run.json').write_text(text,encoding='utf8',newline='\n')
        if success:(ROOT/'data/worlds/design_v2/candidates'/candidate/'run.json').write_text(text,encoding='utf8',newline='\n')
        print('U42_END',candidate,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
if __name__=='__main__':main()
