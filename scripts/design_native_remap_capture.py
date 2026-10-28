"""U36-S15：完整旁观控制复跑以保存真实0Ma原生宿主，不改变默认世界。"""
from pathlib import Path
import sys,pickle,time,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import design_forward as fwd
from design_west_arc_report import compare_npz
from design_arc_sections import state_hash
from mythic.figures.netcdf import open_netcdf
CACHE=ROOT/'.cache/u36_s15';CACHE.mkdir(exist_ok=True);start=time.perf_counter()
original=fwd.simulate
seen={}
def capture(*args,**kwargs):
 result=original(*args,**kwargs)
 seen['native_state_sha256']=state_hash(result[0])
 with (CACHE/'reference_0Ma.pkl').open('wb') as stream:pickle.dump(result[0],stream,protocol=5)
 return result
fwd.simulate=capture
sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml','--candidate','u36_s15_source_control','--no-figures']
try:fwd.main()
finally:fwd.simulate=original
ref=ROOT/'.cache/u36_s4/control'
current=ROOT/'.cache/tectonics/design/candidates/u36_s15_source_control/forward_v2_v2_L7'
check=compare_npz(ref/'forward.npz',current.with_suffix('.npz'))
old,new=[json.loads(p.read_text(encoding='utf8')) for p in [ref/'forward.json',current.with_suffix('.json')]]
check.update(budget_identical=old['budget']==new['budget'],events_identical=old['events']==new['events'])
world=ROOT/'data/worlds/design_v2/candidates/u36_s15_source_control/design_v2_forward_v2_0Ma_u36_s15_source_control_1024x512.nc'
with open_netcdf(ref/'world.nc') as a,open_netcdf(world) as b:
 check['world_arrays_identical']=set(a.data_vars)==set(b.data_vars) and all(np.array_equal(a[k].values,b[k].values,equal_nan=True) for k in a.data_vars)
assert check['identical'] and all(check[k] for k in ['budget_identical','events_identical','world_arrays_identical'])
check.update(checkpoint='U36-S15',method_class='procedural_constrained',elapsed_s=time.perf_counter()-start,**seen,note='原始最终原生状态旧缓存缺失，完整旁观控制复跑，未替换默认世界；旧最终原生状态无可比文件，不宣称逐字核对')
(CACHE/'control_verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
shutil.move(str(world),str(CACHE/'control_world.nc'))
shutil.move(str(world.parent/'forward_v2.json'),str(CACHE/'control_world_report.json'))
world.parent.rmdir()
print('CONTROL_CAPTURED',check,flush=True)
