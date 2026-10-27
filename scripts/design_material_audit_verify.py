"""U36-S5：原始测试、保护文件与同一控制世界图件的结果归档。"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2'


def main():
    old=json.loads((DATA/'u36_s4_verification.json').read_text(encoding='utf-8'))
    paths=list(old['protected_git_objects'])+['packages/tectonics/material_band.py','tests/tectonics/test_material_band.py']
    protected={}
    for p in paths:
        current=subprocess.check_output(['git','hash-object',p],cwd=ROOT,text=True).strip()
        expected=subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=ROOT,text=True).strip()
        protected[p]=dict(git_object=current,identical_to_base=current==expected)
    assert all(v['identical_to_base'] for v in protected.values())
    folder=ROOT/'figures/stage1/u36_s5_reference'
    for name in ['metrics','manifest','regions']:shutil.copy2(folder/(name+'.json'),DATA/('u36_s5_reference_'+name+'.json'))
    current=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    previous=json.loads((DATA/'u36_s4_reference_metrics.json').read_text(encoding='utf-8'))
    current_metrics={k:v for k,v in current['metrics'].items() if k!='timings_s'}
    previous_metrics={k:v for k,v in previous['metrics'].items() if k!='timings_s'}
    parity=dict(metrics_identical_except_timings=current_metrics==previous_metrics,b_verdict_identical=current['b_verdict']==previous['b_verdict'])
    assert all(parity.values())
    log=(ROOT/'.cache/u36_s5_tests.log').read_text(encoding='utf-8-sig')
    summary=re.findall(r'(?m)^([^\n]*\b(?:passed|failed|skipped)\b[^\n]*\bin [\d.]+s)\s*$',log)[-1].strip()
    log=log.replace(str(ROOT),'<workspace>').replace(ROOT.as_posix(),'<workspace>')
    log=re.sub(r'pytest-of-[^/\\\s\x27]+','pytest-of-<local-user>',log)
    (DATA/'u36_s5_tests.txt').write_text('\n'.join(line.rstrip() for line in log.splitlines())+'\n',encoding='utf-8')
    figures=list((ROOT/'figures/design/v2/forward/u36_s5').glob('*.png'))+list(folder.rglob('*.png'))
    assert len(figures)==34
    result=dict(checkpoint='U36-S5',protected=protected,all_protected_identical=True,control_metrics=parity,
        excluded_comparison_fields=['generated','metrics.timings_s'],
        control_world_source='复用S4只读控制NC；本轮没有新300→0Ma演化、没有新全球物理状态',
        control_world_nc_sha256=hashlib.sha256((ROOT/'.cache/u36_s4/control/world.nc').read_bytes()).hexdigest(),
        test_summary=summary,test_all_passed=re.search(r'\b[1-9]\d* failed',summary) is None,
        png_count=len(figures),png_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},
        default_world_adopted=False,quality_thresholds_changed=False)
    (DATA/'u36_s5_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary,'metrics identical',parity,'figures',len(figures),flush=True)


if __name__=='__main__':main()
