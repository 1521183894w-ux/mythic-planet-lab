"""U36-S10：从完整控制演化保存35Ma原宿主，重建同一S8固定材料域。"""
from pathlib import Path
import json,pickle,subprocess,sys,time,hashlib
import numpy as np
import pygplates as pg
import shapely
from mythic.tectonics.engine import Engine
from mythic.tectonics.material_band import reference_strip,build_network
from mythic.tectonics.source_clipping import clip_native_sources
from mythic.figures.netcdf import open_netcdf
from design_west_arc_report import compare_npz
from design_arc_sections import state_hash
import design_forward as fwd

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/u36_s10'
DATA=ROOT/'data/worlds/design_v2'

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    original_step=Engine.step
    final={}
    original_simulate=fwd.simulate
    def observe(eng,*args,**kwargs):
        if eng.t==35:
            with (CACHE/'reference_35Ma.pkl').open('wb') as stream:
                pickle.dump(eng,stream,protocol=5)
        return original_step(eng,*args,**kwargs)
    def capture(*args,**kwargs):
        result=original_simulate(*args,**kwargs)
        final['hash']=state_hash(result[0])
        return result
    argv=sys.argv
    Engine.step=observe;fwd.simulate=capture
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml','--candidate','u36_s10_source_control','--no-figures']
    try:fwd.main()
    finally:Engine.step=original_step;fwd.simulate=original_simulate;sys.argv=argv
    ref=ROOT/'.cache/u36_s3'
    current=ROOT/'.cache/tectonics/design/candidates/u36_s10_source_control/forward_v2_v2_L7'
    check=compare_npz(ref/'forward.npz',current.with_suffix('.npz'))
    old,new=[json.loads(p.read_text(encoding='utf8')) for p in [ref/'forward.json',current.with_suffix('.json')]]
    check.update(budget_identical=old['budget']==new['budget'],events_identical=old['events']==new['events'])
    with (ref/'final_state.pkl').open('rb') as stream:check['native_state_identical']=final['hash']==state_hash(pickle.load(stream)['engine'])
    world=DATA/'candidates/u36_s10_source_control/design_v2_forward_v2_0Ma_u36_s10_source_control_1024x512.nc'
    with open_netcdf(ref/'world.nc') as a,open_netcdf(world) as b:
        check['world_arrays_identical']=set(a.data_vars)==set(b.data_vars) and all(np.array_equal(a[k].values,b[k].values,equal_nan=True) for k in a.data_vars)
    assert check['identical'] and all(check[k] for k in ['budget_identical','events_identical','native_state_identical','world_arrays_identical'])
    with (CACHE/'reference_35Ma.pkl').open('rb') as stream:eng=pickle.load(stream)
    old=json.loads((DATA/'u36_s4_material_band.json').read_text(encoding='utf8'))
    strip=reference_strip(*np.asarray(old['registered_endpoints_xyz']),200.,30.)
    pieces,p,summary=clip_native_sources(eng,strip['xyz'][strip['boundary']])
    target=shapely.Polygon(p.forward(strip['xyz'][strip['boundary']]))
    domain=target.buffer(-2.,join_style=2)
    inner=[]
    for piece in pieces:
        hit=piece['geometry'].intersection(domain)
        if not hit.is_empty:inner.append(dict(piece,geometry=hit))
    expected=json.loads((DATA/'u36_s8_fixed_domain.json').read_text(encoding='utf8'))['domain']
    digest=hashlib.sha256(domain.wkb).hexdigest()
    assert digest==expected['fixed_domain_wkb_sha256']
    from mythic.tectonics.source_clipping import spherical_geometry_area
    from mythic.tectonics.crust import CONT
    volumes={z['uid']:spherical_geometry_area(z['geometry'],p)*z['T_km'] for z in inner if z['ctype']==CONT}
    inventory={z['uid']:z['inner_volume_km3'] for z in expected['outer_source_inventory']}
    difference=max(abs(v-inventory[u]) for u,v in volumes.items())
    assert difference<1e-6
    rot=pg.RotationModel(str(DATA/'history_v2.rot'))
    _,features=build_network(strip,rot,35,plates=(301,401),tag='S10_real')
    pg.FeatureCollection(features).write(str(CACHE/'network.gpml'))
    with (CACHE/'sources.pkl').open('wb') as stream:
        pickle.dump(dict(pieces=inner,original_pieces=pieces,projection=p,domain=domain,target=target),stream,protocol=5)
    check.update(checkpoint='U36-S10',method_class='procedural_constrained',fixed_domain_wkb_sha256=digest,maximum_UID_initial_volume_difference_km3=difference,original_clip_summary=summary,elapsed_s=time.perf_counter()-begin,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),note='完整旁观控制重跑对照S3；原网络特征ID重新建立，不冒称S4 GPML字节相同')
    (DATA/'u36_s10_source_rebuild.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
    print('PREPARED',check,flush=True)
if __name__=='__main__':main()
