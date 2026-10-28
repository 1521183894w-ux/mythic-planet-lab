"""S10实际输出归档核对；不改变质量阈值，也不忽略旧严格失败。"""
from pathlib import Path
import hashlib,json,re,subprocess,shutil
import numpy as np
from mythic.tectonics.material_band import spherical_triangle_area
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/worlds/design_v2';CACHE=ROOT/'.cache/u36_s10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    prior=json.loads((DATA/'u36_s9_verification.json').read_text(encoding='utf8'))
    paths=sorted(set(prior['protected'])|{'packages/tectonics/rotation_interpolation.py','scripts/design_rotation_interpolation.py','scripts/design_rotation_mesh_audit.py','tests/tectonics/test_rotation_interpolation.py'})
    protected={}
    for p in paths:
        current=subprocess.check_output(['git','hash-object',p],cwd=ROOT,text=True).strip()
        base=subprocess.check_output(['git','rev-parse','1336b4a:'+p],cwd=ROOT,text=True).strip()
        protected[p]=dict(identical_to_base=current==base,git_object=current)
    assert all(q['identical_to_base'] for q in protected.values())
    standard=ROOT/'figures/stage1/u36_s10_reference'
    for name in ['metrics','manifest','regions']:
        shutil.copy2(standard/(name+'.json'),DATA/f'u36_s10_reference_{name}.json')
    old=json.loads((DATA/'u36_s9_reference_metrics.json').read_text(encoding='utf8'))
    new=json.loads((standard/'metrics.json').read_text(encoding='utf8'))
    strip=lambda q:{k:v for k,v in q['metrics'].items() if k!='timings_s'}
    assert strip(old)==strip(new) and old['b_verdict']==new['b_verdict']
    assert json.loads((DATA/'u36_s9_reference_regions.json').read_text(encoding='utf8'))['regions']==json.loads((standard/'regions.json').read_text(encoding='utf8'))['regions']
    checks=[]
    for path in sorted(DATA.glob('u36_s10_edge*.json')):
        q=json.loads(path.read_text(encoding='utf8'));a=np.load(CACHE/(q['tag']+'.npz'))
        assert q['complete_35_to_0Ma'] and q['all_points_preserved']
        assert len(q['timeline'])==round(35/q['dt_myr'])+1
        assert q['minimum_selected_projected_det']>0
        v0=a['V0_km3'];vv=a['source_volumes_km3']
        assert np.max(abs(vv.sum(axis=1)-np.array([z['volume_km3'] for z in q['timeline']])))<1e-6
        independent=[]
        for t in [35,20,0]:
            pp=a[f'points_{t}Ma'];nv=len(a['vertices'])
            ar=abs(spherical_triangle_area(pp[:nv],a['triangles']))
            initial_patch=spherical_triangle_area(a['points_35Ma'][nv:],a['patch_triangles']).reshape(-1,4).sum(axis=1)
            jac=spherical_triangle_area(pp[nv:],a['patch_triangles']).reshape(-1,4).sum(axis=1)/initial_patch
            h=a['H0_km']/jac
            error=float(np.nanmax(abs(h/a[f'H_{t}Ma']-1)))
            assert error<2e-8
            volume=float(np.sum(ar[a['known']]*h[a['known']]))
            j=int(round((35-t)/q['dt_myr']))
            assert abs(volume/vv[j].sum()-1)<2e-9
            independent.append(dict(t_ma=t,H_relative_recalculation_difference=error,volume_relative_recalculation_difference=float(volume/vv[j].sum()-1)))
        oldtag=f"u36_s8_edge{q['maximum_edge_km']:g}_patch{q['patch_radius_km']:g}.json"
        source_difference=None
        if (DATA/oldtag).exists():
            oldq=json.loads((DATA/oldtag).read_text(encoding='utf8'))
            lookup={z['uid']:z['initial_volume_km3'] for z in oldq['source_trajectories']}
            assert {z['uid'] for z in q['source_trajectories']}==set(lookup)
            source_difference=max(abs(z['initial_volume_km3']-lookup[z['uid']]) for z in q['source_trajectories'])
            assert source_difference<1e-6
        checks.append(dict(tag=q['tag'],time_count=len(q['timeline']),initial_source_volume_difference_from_S8_km3=source_difference,independent_saved_state_recalculation=independent))
    text=(ROOT/'.cache/u36_s10_tests.log').read_text(encoding='utf8')
    for p in [ROOT,Path.home()]:
        text=text.replace(str(p),'<workspace>' if p==ROOT else '<local-user>').replace(p.as_posix(),'<workspace>' if p==ROOT else '<local-user>')
    text=re.sub(r'pytest-of-[^/\\\s\x27]+','pytest-of-<local-user>',text)
    (DATA/'u36_s10_tests.txt').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n',encoding='utf8')
    summary=re.findall(r'(?m)^([^\n]*\b(?:passed|failed)\b[^\n]*\bin [\d.]+s)\s*$',text)[-1].strip()
    assert '95 passed' in summary and '2 failed' in summary
    assert all('FAILED tests/tectonics/'+n in text for n in ['test_step_audit.py::test_common_rotation_network_matches_its_input_rigid_rotation','test_material_band.py::test_common_rotation_stays_within_spherical_sampling_tolerance'])
    figures=list((ROOT/'figures/design/v2/forward/u36_s10').glob('*.png'))+list(standard.rglob('*.png'))
    assert len(figures)==35
    source=json.loads((DATA/'u36_s10_source_rebuild.json').read_text(encoding='utf8'))
    parity=json.loads((DATA/'u36_s10_S9_parity.json').read_text(encoding='utf8'))
    assert source['identical'] and source['maximum_UID_initial_volume_difference_km3']==0.
    assert parity['maximum_center_position_difference_km']<1e-8
    outer=json.loads((DATA/'u36_s10_outer_attempt.json').read_text(encoding='utf8'))
    report=dict(checkpoint='U36-S10',base_commit='1336b4a',method_class='procedural_constrained',protected=protected,
        protected_identical=True,metrics_identical_except_timings=True,b_verdict_identical=True,region_windows_identical=True,
        tests_summary=summary,test_all_passed=False,old_strict_failures_retained=True,checks=checks,
        all_full_inner_trials_preserve_all_material=True,full_network_boundary_resolved=False,
        outer_attempt_completed=outer['complete_35_to_0Ma'],outer_failure=outer['failure'],
        first_rebuilt_connectivity_outputs_retained=True,reference_connectivity_explicitly_uses_archived_S9=True,
        source_and_input_parity=parity,global_or_B5_coupling_performed=False,quality_thresholds_changed=False,default_world_adopted=False,
        png_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in figures},
        limits='柱连续性是给定运动下的条件性模型；体积误差不等于独立物理验证，单源细化不证明全域逐源收敛')
    write_json(DATA/'u36_s10_verification.json',report)
    print(summary,'protected',len(protected),'figures',len(figures),'inner trials',len(checks),flush=True)
if __name__=='__main__':main()
