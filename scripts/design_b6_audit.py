from pathlib import Path
import json, numpy as np
old=np.load('.cache/u36_s3/forward.npz')
new=np.load('.cache/tectonics/design/candidates/u37_b6_structured/forward_v2_v2_L7.npz')
t=old['times'];assert np.array_equal(t,new['times'])
different=[]
for i in np.flatnonzero(t>180):
 for name in ['elev','T','age','t_form','oro_t','plate','ctype','origin','oro','prov','dyn']:
  key=f'{name}_{i}'
  if not np.array_equal(old[key],new[key],equal_nan=True):different.append(key)
a=json.loads(Path('.cache/u36_s3/forward.json').read_text(encoding='utf8'))
b=json.loads(Path('.cache/tectonics/design/candidates/u37_b6_structured/forward_v2_v2_L7.json').read_text(encoding='utf8'))
r=json.loads(Path('.cache/u37_b6/run.json').read_text(encoding='utf8'))
out=dict(pre_activation_sampled_times=t[t>180].tolist(),pre_activation_different_arrays=different,
 budget_control=a['budget'],budget_candidate=b['budget'],
 budget_differences={k:b['budget'][k]-v for k,v in a['budget'].items()},
 collapse_balance_sampled_max_km3=max(abs(x['collapse_balance_km3']) for x in r['diagnostic_rows']),
 note='前史仅比较每10Ma保存的11个格网字段；体积预算字段中剥蚀等是不同过程，不能直接将差额视为单一物质去向')
Path('.cache/u37_b6/audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
