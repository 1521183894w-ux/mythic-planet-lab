"""U36-S8：固定域/逐UID库存/141时点来源轨迹的独立归档核对。"""
from pathlib import Path
import json,pickle
import shapely
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/worlds/design_v2'
def main():
    report=json.loads((D/'u36_s8_fixed_domain.json').read_text(encoding='utf-8'));domain=report['domain'];checks=[]
    assert abs(sum(q['inner_volume_km3'] for q in domain['outer_source_inventory'])-domain['known_cont_volume_km3'])<1e-6
    assert abs(sum(q['reserved_outer_volume_km3'] for q in domain['outer_source_inventory'])-domain['reserved_outer_known_volume_km3'])<1e-6
    assert abs(domain['known_cont_volume_km3']+domain['reserved_outer_known_volume_km3']-domain['original_known_volume_km3'])<1e-6
    with (ROOT/'.cache/u36_s8/domain.pkl').open('rb') as f:geometry=pickle.load(f)
    bounds=geometry['domain'].bounds;rng=np.random.default_rng(3608);xy=rng.uniform(bounds[:2],bounds[2:],(10000,2));xy=xy[shapely.contains_xy(geometry['domain'],xy[:,0],xy[:,1])]
    expected=np.full(len(xy),-999999,dtype=np.int64)
    for piece in geometry['pieces']:expected[shapely.contains_xy(piece['geometry'],xy[:,0],xy[:,1])]=piece['uid']
    assert np.all(expected!=-999999)
    for trial in report['trials']:
        q=json.loads((D/f"u36_s8_{trial['tag']}.json").read_text(encoding='utf-8'));cols=q['source_trajectories'];times=np.array([z['t_ma'] for z in q['timeline']]);assert len(times)==141
        volumes=np.array([z['volume_km3'] for z in cols]);initial=np.array([z['initial_volume_km3'] for z in cols]);total=np.array([z['volume_km3'] for z in q['timeline']])
        assert volumes.shape[1]==141 and np.max(abs(volumes.sum(0)-total))<1e-6
        assert q['all_known_domain_evaluated'] and q['ever_outside_patch_count']==0
        assert all(z['inverted_cells']==0 and z['inverted_patches']==0 for z in q['timeline'])
        per_uid={z['uid']:z['inner_volume_km3'] for z in domain['outer_source_inventory']}
        source_difference=max(abs(z['initial_volume_km3']-per_uid[z['uid']]) for z in cols)
        worst=np.unravel_index(np.argmax(abs(volumes/initial[:,None]-1)),volumes.shape)
        mesh=np.load(ROOT/f".cache/u36_s8/{q['tag']}.npz");projected=geometry['projection'].forward(mesh['vertices']);polys=shapely.polygons(projected[mesh['triangles']]);tree=shapely.STRtree(polys);pairs=tree.query(shapely.points(xy),predicate='within')
        actual=np.full(len(xy),-999999,dtype=np.int64);actual[pairs[0]]=mesh['source_uid'][pairs[1]];mismatch=int(np.sum(actual!=expected));assert mismatch==0
        checks.append(dict(initial_source_query_count=len(xy),initial_source_query_seed=3608,initial_source_uid_mismatch_count=mismatch,maximum_source_error_uid=cols[worst[0]]['uid'],maximum_source_error_t_ma=float(times[worst[1]]),maximum_source_error_initial_volume_km3=float(initial[worst[0]]),tag=q['tag'],all_141_sampled_times_inside=True,no_sampled_inversions=True,
            initial_volume_difference_from_fixed_polygons_km3=q['initial_known_volume_km3']-domain['known_cont_volume_km3'],
            maximum_source_initial_volume_difference_km3=source_difference,source_volume_sum_max_error_km3=float(np.max(abs(volumes.sum(0)-total))),
            maximum_source_absolute_relative_error=float(np.max(abs(volumes/initial[:,None]-1))),
            final_weighted_source_absolute_error=float(np.sum(abs(volumes[:,-1]-initial))/initial.sum()),
            final_net_volume_relative_error=q['timeline'][-1]['volume_relative_residual'],process_peak_working_set_mib=q['process_peak_working_set_mib']))
    result=dict(checkpoint='U36-S8',fixed_domain_hash=domain['fixed_domain_wkb_sha256'],outer_stock_partition_closed=True,outer_stock_partition_note='初始内外分账的代数恒等核对，非动态物理守恒证明',outer_uid_stock_all_nonnegative=all(q['reserved_outer_volume_km3']>=0 for q in domain['outer_source_inventory']),negative_outer_signed_differences=[dict(uid=q['uid'],signed_difference_km3=q['reserved_outer_volume_km3']) for q in domain['outer_source_inventory'] if q['reserved_outer_volume_km3']<0],checks=checks,
        limits='只核对固定数值内域；141采样时点不代表时点之间已逐点检查，不是物理成因验证')
    (D/'u36_s8_domain_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
