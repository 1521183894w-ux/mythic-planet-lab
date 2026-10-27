"""U36-S6：归档真实检查、冻结世界与保护内容一致性。"""
import hashlib,json,re,subprocess,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2'

def main():
    old=json.loads((DATA/'u36_s5_verification.json').read_text(encoding='utf-8'))
    protected={}
    paths=list(old['protected'])+['packages/tectonics/material_audit.py','packages/tectonics/source_clipping.py','tests/tectonics/test_material_audit.py','tests/tectonics/test_source_clipping.py']
    for path in paths:
        actual=subprocess.check_output(['git','hash-object',path],cwd=ROOT,text=True).strip()
        original=subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT,text=True).strip()
        protected[path]=dict(git_object=actual,identical_to_base=actual==original)
    assert all(q['identical_to_base'] for q in protected.values())
    folder=ROOT/'figures/stage1/u36_s6_reference'
    for key in ['metrics','manifest','regions']:shutil.copy2(folder/(key+'.json'),DATA/f'u36_s6_reference_{key}.json')
    current=json.loads((folder/'metrics.json').read_text(encoding='utf-8'));prior=json.loads((DATA/'u36_s5_reference_metrics.json').read_text(encoding='utf-8'))
    metrics={k:v for k,v in current['metrics'].items() if k!='timings_s'};previous={k:v for k,v in prior['metrics'].items() if k!='timings_s'}
    parity=dict(metrics_identical_except_timings=metrics==previous,b_verdict_identical=current['b_verdict']==prior['b_verdict'])
    assert all(parity.values())
    source_checks=[]
    baseline=json.loads((DATA/'u36_s5_sources.json').read_text(encoding='utf-8'))['summary']['known_cont_volume_km3']
    for suffix in ['', '_refined']:
        s=json.loads((DATA/f'u36_s6_sources{suffix}.json').read_text(encoding='utf-8'))
        a=np.load(ROOT/f'.cache/u36_s6/local_material{suffix}.npz');supported=a['supported']
        assert s['initial_volume_km3']>0 and s['supported_initial_volume_km3']>0
        assert abs(s['initial_volume_km3']/baseline-1)<1e-7
        assert abs(s['supported_initial_volume_km3']+s['unsupported_known_initial_volume_km3']-s['initial_volume_km3'])<1e-6
        assert all(q['inverted_triangles']==0 and q['inverted_patches']==0 for q in s['timeline'])
        assert all(np.isnan(a[f'H_{t}Ma'][~supported]).all() for t in [35,30,20,10,0])
        assert all(np.isfinite(a[f'H_{t}Ma'][supported]).all() for t in [35,30,20,10,0])
        source_checks.append(dict(maximum_edge_km=s['maximum_edge_km'],positive_initial_inventory=True,
          initial_inventory_identical_to_s5_within_relative_1e_7=True,unsupported_stock_kept_separately=True,
          unsupported_thickness_is_missing=True,no_recorded_inversions=True,
          supported_initial_volume_km3=s['supported_initial_volume_km3'],unsupported_known_initial_volume_km3=s['unsupported_known_initial_volume_km3']))
    log=(ROOT/'.cache/u36_s6_tests.log').read_text(encoding='utf-8-sig')
    summary=re.findall(r'(?m)^([^\n]*\b(?:passed|failed|skipped)\b[^\n]*\bin [\d.]+s)\s*$',log)[-1].strip()
    clean=log.replace(str(ROOT),'<workspace>').replace(ROOT.as_posix(),'<workspace>')
    clean=re.sub(r'pytest-of-[^/\\\s\x27]+','pytest-of-<local-user>',clean)
    (DATA/'u36_s6_tests.txt').write_text('\n'.join(line.rstrip() for line in clean.splitlines())+'\n',encoding='utf-8')
    png=list((ROOT/'figures/design/v2/forward/u36_s6').glob('*.png'))+list(folder.rglob('*.png'));assert len(png)==34
    result=dict(checkpoint='U36-S6',protected=protected,all_protected_identical=True,control_metrics=parity,
      excluded_comparison_fields=['generated','metrics.timings_s'],source_checks=source_checks,
      control_world_source='复用S4冻结NC；本轮未重跑全球演化',control_world_nc_sha256=hashlib.sha256((ROOT/'.cache/u36_s4/control/world.nc').read_bytes()).hexdigest(),
      test_summary=summary,test_all_passed=False,png_count=len(png),
      png_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in png},
      default_world_adopted=False,quality_thresholds_changed=False)
    assert result['control_world_nc_sha256']==old['control_world_nc_sha256']
    (DATA/'u36_s6_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary,'protected',len(protected),'metrics',parity,'figures',len(png),flush=True)
if __name__=='__main__':main()
