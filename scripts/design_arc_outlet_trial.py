"""U41：原生薄壳出口与碰撞接触出口的完整世界对照，默认世界不变。"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import psutil
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'packages'),str(ROOT/'scripts')]
import design_arc_outlet as capture
import design_forward as fwd
from mythic.tectonics.engine import Engine
from mythic.tectonics.arc_outlet_spike import apply_arc_outlet
BASE=capture.previous.candidate_surface
CONTACTS=False
outlet_ledger=[]


def surface_with_outlet(self,dt,pos=None):
    BASE(self,dt,pos)
    if self.t<=180.:
        pre=float(self.c.T[self.alive & (self.c.ctype==1)].max())
        row=apply_arc_outlet(self,dt,pos,contacts=CONTACTS)
        row['pre_max_T_km']=pre
        row['post_max_T_km']=float(self.c.T[self.alive & (self.c.ctype==1)].max())
        outlet_ledger.append(row)


def main():
    global CONTACTS
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['native','contact'],default='native')
    mode=ap.parse_args().mode;CONTACTS=mode=='contact'
    candidate='u41_'+mode
    capture.MODE=mode
    out=capture.CACHE/mode;out.mkdir(parents=True,exist_ok=True)
    capture.previous.KAPPA=10000.
    capture.previous.candidate_surface=surface_with_outlet
    Engine._interact=capture.observed_interact;Engine._arc=capture.observed_arc
    Engine._surface=capture.observed_surface;fwd.simulate=capture.captured_simulate
    saved=sys.argv[:];start=time.perf_counter();success=False
    sources=[Path(__file__),ROOT/'scripts/design_arc_outlet.py',
        ROOT/'packages/tectonics/arc_outlet_spike.py',
        ROOT/'packages/tectonics/borrowing.py',ROOT/'packages/tectonics/relaxation_spike.py',
        ROOT/'scripts/design_b6_relaxation.py',ROOT/'scripts/design_b6_halo.py',
        ROOT/'scripts/design_b6_transition.py',ROOT/'scripts/design_forward.py',
        ROOT/'packages/tectonics/engine.py',
        ROOT/'config/design/history_v2_u35_c10_progressive.yaml',
        ROOT/'config/design/final_map_v2.yaml',ROOT/'data/worlds/design_v2/history_v2.rot',
        ROOT/'.cache/tectonics/design/initial_v2_v2_L7.pkl']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
        '--candidate',candidate,'--no-figures']
    try:
        fwd.main()
        nc=ROOT/f'data/worlds/design_v2/candidates/{candidate}/design_v2_forward_v2_0Ma_{candidate}_1024x512.nc'
        with Dataset(nc,'a') as ds:
            z=np.asarray(ds['elevation'][:])
            ds.setncatts(dict(method_class_elevation='parameterized_physics',
                method_class_transport='procedural_constrained',method_class_region='user_override',
                method_class_sampling='interpolated',transport_method='u41_finite_arc_outlet_v1',
                transport_mode=mode,transport_kappa_km2_myr=10000.,
                transport_immobile_floor_km=40.,transport_activation_ma=180.,
                candidate_status='experimental; default C2 unchanged',
                transport_limits='finite paired exchange and shared donor limit; volume only, no density/composition mixture; optional approximate proximity contact, not exact common face',
                physical_ceiling_exceedance_cells_over10km=int((z>10000).sum())))
            ds['elevation'].setncattr('method_class','parameterized_physics')
            ds.setncattr('source',ds.getncattr('source')+'; U41 finite juvenile-arc outlet trial, no upper thickness clip; see run.json; no validated stress/thermal solver.')
            if np.any(z>10000):
                v=ds.createVariable('physical_ceiling_violation','i1',('lat','lon'));v[:]=(z>10000).astype('i1')
                v.setncatts(dict(method_class='fantasy_exception',description='Unaccepted model ceiling violation'))
        success=True
    finally:
        Engine._surface=capture.original_surface;Engine._interact=capture.original_interact
        Engine._arc=capture.original_arc;fwd.simulate=capture.original_simulate
        capture.previous.candidate_surface=BASE;sys.argv=saved
        report=dict(success=success,candidate=candidate,mode=mode,
            method='u41_finite_arc_outlet_v1',
            method_class={'outlet':'procedural_constrained','regions':'user_override',
                'elevation':'parameterized_physics','sampling':'interpolated'},
            resolution='L7 about60km; output1024x512',
            units={'thickness':'km','time':'Ma','volume':'km3','distance':'km','elevation':'m'},
            assumptions=['same U40B erosion and finite continental relaxation; outlet added after each surface step since180Ma',
                'only juvenile arc-origin continental donors above existing40km flow onset',
                'native thin/ocean receivers on same plate; no coordinate window restriction in dynamics',
                'contact case adds current-distance0.5-1.5h continental receivers across plates only for collided arc donors with no eligible native continental edge',
                'exclude near-coincident centers; proximity does not prove shared material boundary',
                'frozen two-cell implicit proposals, convex row scaling, existing shared donor inventory limit',
                '40km immobile donor floor, not upper thickness/elevation cap',
                'ocean receivers convert at existing31km proxy; continental recipient labels do not track mixing fractions'],
            limits=['not calibrated rheology; no stress, heat, density or full composition transfer',
                'finite distance contact approximates collisional coupling, not exact Voronoi shared face',
                'floor40km is a hypothesis reusing flow onset; no physical calibration',
                'material edge logs are this operator only, not full history provenance',
                'prior r2/r8, plateau, rift source and stage delivery issues remain'],
            input_sha256=hashes,step_ledger=capture.previous.ledger,outlet_ledger=outlet_ledger,
            watched_rows=capture.rows,elapsed_s=time.perf_counter()-start,
            process_lifetime_peak_wset_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        text=json.dumps(report,ensure_ascii=False,indent=2)
        (out/'run.json').write_text(text,encoding='utf8',newline='\n')
        if success:(ROOT/'data/worlds/design_v2/candidates'/candidate/'run.json').write_text(text,encoding='utf8',newline='\n')
        print('U41_END',mode,success,report['elapsed_s'],report['process_lifetime_peak_wset_bytes'],flush=True)
if __name__=='__main__':main()
