"""归档S9真实运行/测试、保护内容、冻结世界、图和科学范围。"""
from pathlib import Path
import hashlib,json,re,shutil,subprocess
import numpy as np
import scipy,pygplates as pg
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    prior=json.loads((DATA/'u36_s8_verification.json').read_text(encoding='utf8'))
    paths=sorted(set(prior['protected'])|{'scripts/design_fixed_domain.py','scripts/design_fixed_domain_verify.py',
        'tests/tectonics/test_material_band.py','tests/tectonics/test_step_audit.py',
        'scripts/design_rotation_interpolation_verify.py'}-{'scripts/design_rotation_interpolation_verify.py'})
    protected={}
    for p in paths:
        current=subprocess.check_output(['git','hash-object',p],cwd=ROOT,text=True).strip()
        baseline=subprocess.check_output(['git','rev-parse','77a58ca:'+p],cwd=ROOT,text=True).strip()
        protected[p]=dict(identical_to_base=current==baseline,git_object=current)
    assert all(q['identical_to_base'] for q in protected.values())
    out=ROOT/'figures/stage1/u36_s9_reference'
    for name in ['metrics','manifest','regions']:
        shutil.copy2(out/(name+'.json'),DATA/f'u36_s9_reference_{name}.json')
    old=json.loads((DATA/'u36_s8_reference_metrics.json').read_text(encoding='utf8'))
    new=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    strip=lambda d:{k:v for k,v in d['metrics'].items() if k!='timings_s'}
    assert strip(old)==strip(new) and old['b_verdict']==new['b_verdict']
    regions=json.loads((DATA/'u36_s8_reference_regions.json').read_text(encoding='utf8'))
    current=json.loads((out/'regions.json').read_text(encoding='utf8'))
    assert regions['regions']==current['regions']
    for raw,target in [('u36_s9_tests.log','u36_s9_tests.txt'),('u36_s9_initial_tests.log','u36_s9_initial_test_failure.txt')]:
        text=(ROOT/'.cache'/raw).read_text(encoding='utf-8-sig')
        text=text.replace(str(ROOT),'<workspace>').replace(ROOT.as_posix(),'<workspace>')
        text=re.sub(r'pytest-of-[^/\\\s\x27]+','pytest-of-<local-user>',text)
        (DATA/target).write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf8')
    log=(DATA/'u36_s9_tests.txt').read_text(encoding='utf8')
    summary=re.findall(r'(?m)^([^\n]*\b(?:passed|failed)\b[^\n]*\bin [\d.]+s)\s*$',log)[-1].strip()
    rows=json.loads((DATA/'u36_s9_rotation_maps.json').read_text(encoding='utf8'))['rows']
    mesh=json.loads((DATA/'u36_s9_mesh_audit.json').read_text(encoding='utf8'))
    assert len(rows)==30 and all(q['minimum_selected_triangle_det']>0 for q in rows)
    figs=list((ROOT/'figures/design/v2/forward/u36_s9').glob('*.png'))+list(out.rglob('*.png'))
    assert len(figs)==33
    timestep=[]
    for case in ['rigid','ramp','turning','slip','real']:
        with np.load(ROOT/f'.cache/u36_s9/{case}_0_0.25.npz') as aa,np.load(ROOT/f'.cache/u36_s9/{case}_0_0.125.npz') as bb:
            timestep.append(dict(case=case,maximum_final_center_position_difference_m=float(np.max(np.linalg.norm(aa['candidate_xyz'][-1]-bb['candidate_xyz'][-1],axis=-1))*6371000),maximum_final_J_relative_difference=float(np.max(abs(aa['candidate_J'][-1]/bb['candidate_J'][-1]-1)))))
    with np.load(ROOT/'.cache/u36_s9/real_0_0.125.npz') as aa:
        compare=dict(checkpoint='U36-S9',case='real',anchor=0,dt_myr=.125,legacy_final_J=aa['legacy_J'][-1].tolist(),candidate_final_J=aa['candidate_J'][-1].tolist(),candidate_over_legacy_J_minus_one_quantiles=np.quantile(aa['candidate_J'][-1]/aa['legacy_J'][-1]-1,[0,.1,.5,.9,1]).tolist(),timestep=timestep,note='同一432中心的末态J；不是独立物理真值，两个步长不证明收敛阶')
    (DATA/'u36_s9_comparison.json').write_text(json.dumps(compare,ensure_ascii=False,indent=2),encoding='utf8')
    assert all(q.get('final_time_support_checked') for q in rows)
    report=dict(checkpoint='U36-S9',base_commit='77a58ca',protected=protected,
        protected_identical=True,metrics_identical_except_timings=True,b_verdict_identical=True,region_windows_identical=True,reference_metadata_note='来源保留S3真实代码版本，生成时间更新；不冒称S4元数据逐字相同',
        tests_summary=summary,test_all_passed="failed" not in summary,
        original_strict_framework_failures_retained=all('FAILED tests/tectonics/'+name in log for name in ['test_step_audit.py::test_common_rotation_network_matches_its_input_rigid_rotation','test_material_band.py::test_common_rotation_stays_within_spherical_sampling_tolerance']),
        complete_network_geometry_test_passed=mesh["full_network_geometry_test_passed"],
        point_maps_checked=len(rows),all_queried_points_supported_without_removal=all(q["sampled_outside_count"]==0 and q["sampled_inverted_count"]==0 and q["minimum_selected_triangle_det"]>0 and q["final_time_support_checked"] for q in rows),
        prior_source_UID_platform_resolved=False,global_or_B5_coupling_performed=False,
        png_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in figs},
        input_provenance=json.loads((DATA/'u36_s9_reference_source.json').read_text(encoding='utf8')),
        quality_thresholds_changed=False,default_world_adopted=False)
    (DATA/'u36_s9_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    evidence=dict(framework='pyGPlates'+pg.__version__+' (existing Windows environment)',scipy_version=scipy.__version__,upstream_tag='PyGPlates-1.0.0',
        upstream_commit='27d48c91f24d7db9f6a212e7957d8c08064e5d9b',
        source_file='src/app-logic/ResolvedTriangulationNetwork.cc',
        point_map_lines=[625,769],velocity_lines=[934,986],
        references=['https://github.com/GPlates/GPlates/blob/27d48c91f24d7db9f6a212e7957d8c08064e5d9b/src/app-logic/ResolvedTriangulationNetwork.cc#L625',
        'https://www.gplates.org/docs/pygplates/generated/pygplates.ResolvedTopologicalNetwork.html',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html'],
        copied_upstream_code_into_project=False,new_solver_installed=False,
        conclusion='局部内部点客观性改善；不是唯一物理插值、完整网络/框架修复或全球输运')
    (DATA/'u36_s9_method_sources.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(summary,'protected',len(protected),'figures',len(figs),flush=True)
if __name__=='__main__':main()
