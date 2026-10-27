"""局部方法检查点的冻结世界、保护文件、真实测试与图件归档。"""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--special-count',type=int,required=True);a=p.parse_args();u=a.unit
    prior=json.loads((DATA/'u36_s6_verification.json').read_text(encoding='utf-8'));paths=list(prior['protected'])+['packages/tectonics/step_audit.py','tests/tectonics/test_step_audit.py','scripts/design_step_audit.py','scripts/design_step_audit_report.py','scripts/design_step_audit_verify.py']
    if u=='u36_s8':paths += ['scripts/design_frame_audit.py','scripts/design_frame_inputs.py','scripts/verify_local_checkpoint.py']
    protected={}
    for path in paths:
        expected=subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT,text=True).strip();actual=subprocess.check_output(['git','hash-object',path],cwd=ROOT,text=True).strip();protected[path]=dict(git_object=actual,identical_to_base=actual==expected)
    assert all(q['identical_to_base'] for q in protected.values())
    folder=ROOT/f'figures/stage1/{u}_reference'
    for key in ['metrics','manifest','regions']:shutil.copy2(folder/(key+'.json'),DATA/f'{u}_reference_{key}.json')
    old=json.loads((DATA/'u36_s6_reference_metrics.json').read_text(encoding='utf-8'));new=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    assert {k:v for k,v in old['metrics'].items() if k!='timings_s'}=={k:v for k,v in new['metrics'].items() if k!='timings_s'} and old['b_verdict']==new['b_verdict']
    raw=(ROOT/f'.cache/{u}_tests.log').read_text(encoding='utf-8-sig');summary=re.findall(r'(?m)^([^\n]*\b(?:passed|failed|skipped)\b[^\n]*\bin [\d.]+s)\s*$',raw)[-1].strip()
    clean=raw.replace(str(ROOT),'<workspace>').replace(ROOT.as_posix(),'<workspace>');clean=re.sub(r'pytest-of-[^/\\\s\x27]+','pytest-of-<local-user>',clean)
    (DATA/f'{u}_tests.txt').write_text('\n'.join(q.rstrip() for q in clean.splitlines())+'\n',encoding='utf-8')
    figs=list((ROOT/f'figures/design/v2/forward/{u}').glob('*.png'))+list(folder.rglob('*.png'));assert len(figs)==29+a.special_count
    sha=hashlib.sha256((ROOT/'.cache/u36_s4/control/world.nc').read_bytes()).hexdigest();assert sha==prior['control_world_nc_sha256']
    result=dict(checkpoint=u,protected=protected,protected_identical=True,metrics_identical_except_timings=True,b_verdict_identical=True,
        reused_world='S4冻结控制世界；本轮未重跑全球演化',world_nc_sha256=sha,test_summary=summary,test_all_passed='failed' not in summary,
        png_count=len(figs),png_sha256={q.relative_to(ROOT).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in figs},
        quality_thresholds_changed=False,default_world_adopted=False)
    (DATA/f'{u}_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(u,summary,'figures',len(figs),'protected',len(protected),flush=True)
if __name__=='__main__':main()
