"""U36-S3：在既有控制历史中旁观裂谷候选；不改变正式引擎。"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as fwd
from design_west_arc_c5_report import hashes
from design_west_arc_report import compare_npz
from design_arc_sections import state_hash
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import CONT, ORIGIN_RIFT
from mythic.tectonics.gap_diagnostics import rift_candidates, margin_normal_diagnostics
from mythic.figures.netcdf import open_netcdf

DEST=ROOT/".cache/u36_s3"
DATA=ROOT/"data/worlds/design_v2"


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    original_commit=Engine.commit
    original_simulate=fwd.simulate
    captures=[]
    parity=[]
    commits=[]
    def observe(eng):
        uid_start=eng.next_uid
        budget_start=eng.budget.get("rift_fill",0.)
        rows=rift_candidates(eng)
        if rows and len(rows["gap"]):
            normals=margin_normal_diagnostics(eng,rows)
            c=eng.c
            eligible=(c.ctype[rows["neighbor"]]==CONT)&(
                (c.plate[rows["neighbor"]]==rows["pair"][:,None]//1000)|
                (c.plate[rows["neighbor"]]==rows["pair"][:,None]%1000))
            support=np.max(np.where(eligible,rows["distance_km"],0.),axis=1)
            record={k:rows[k].copy() for k in ("xyz","pair","target_area_km2","width_proxy_km")}
            record.update(t_ma=np.full(len(rows["gap"]),eng.t),stencil_radius_km=support,**normals)
            captures.append(record)
            commits.append(dict(t_ma=eng.t,target_count=len(rows["gap"]),raw_gap_count=rows["raw_gap_count"],
                target_area_km2=float(rows["target_area_km2"].sum())))
        result=original_commit(eng)
        # 直接核对真实出生宿主，排除只读重建与Engine去重/归属差异。
        c=eng.c
        born=(c.uid>=uid_start)&(c.ctype==CONT)&(c.origin==ORIGIN_RIFT)&(c.t_form==eng.t)
        got=np.sort(c.plate[born].astype(np.int64)*eng.m.n+c.node[born])
        expected=np.sort(rows["target_owner"].astype(np.int64)*eng.m.n+rows["target_node"]) if rows else np.array([],np.int64)
        if not np.array_equal(got,expected):
            raise RuntimeError(f"候选重建与实际出生宿主不一致：{eng.t} Ma, got={len(got)}, expected={len(expected)}")
        actual_volume=float(np.sum(c.T[born]*eng.m.area_km2[c.node[born]]))
        budget_delta=eng.budget.get("rift_fill",0.)-budget_start
        if not np.isclose(actual_volume,budget_delta,rtol=1e-11,atol=1e-7):
            raise RuntimeError("实际出生体积与rift_fill增量不闭合")
        parity.append(dict(t_ma=eng.t,count=int(born.sum()),actual_volume_km3=actual_volume,
                           budget_delta_km3=budget_delta,host_identity_equal=True))
        return result
    def capture(*a,**kw):
        bound=inspect.signature(original_simulate).bind(*a,**kw)
        bound.apply_defaults()
        eng,frames,vframes=result=original_simulate(*a,**kw)
        with (DEST/"final_state.pkl").open("wb") as stream:
            pickle.dump(dict(engine=eng,fixed_frame=frames[-1],volcanoes=vframes[-1],
                 plumes=bound.arguments.get("plumes"),rifts=bound.arguments.get("rifts"),
                 rot=str(bound.arguments["rot"])),stream,protocol=5)
        return result
    Engine.commit=observe
    fwd.simulate=capture
    old_argv=sys.argv
    sys.argv=["design_forward.py","--design","v2","--history","v2",
              "--history-config","config/design/history_v2_u35_c10_progressive.yaml",
              "--candidate","u36_s3_observed","--no-figures"]
    start=time.perf_counter()
    try:
        fwd.main()
    finally:
        Engine.commit=original_commit
        fwd.simulate=original_simulate
        sys.argv=old_argv
    elapsed=time.perf_counter()-start
    (DEST/"observation_run.json").write_text(json.dumps(dict(elapsed_s=elapsed,commits=commits,host_parity=parity)),encoding="utf-8")
    for suffix in (".npz",".json"):
        shutil.copy2(ROOT/f".cache/tectonics/design/candidates/u36_s3_observed/forward_v2_v2_L7{suffix}",DEST/("forward"+suffix))
    shutil.copy2(ROOT/"data/worlds/design_v2/candidates/u36_s3_observed/design_v2_forward_v2_0Ma_u36_s3_observed_1024x512.nc",DEST/"world.nc")
    np.savez_compressed(DEST/"actual_candidates.npz",**{k:(captures[0][k] if k=="scales_km" else np.concatenate([r[k] for r in captures])) for k in captures[0]})
    return summarize()


def summarize():
    run_meta=json.loads((DEST/"observation_run.json").read_text(encoding="utf-8"))
    elapsed,commits=run_meta["elapsed_s"],run_meta["commits"]
    with np.load(DEST/"actual_candidates.npz") as arrays:
        all_rows={k:arrays[k].copy() for k in arrays.files}
    np.savez_compressed(DEST/"actual_candidates.npz",**all_rows)
    area=all_rows["target_area_km2"]
    status=all_rows["status"]
    summary=dict(candidate_count=len(area),cumulative_candidate_area_km2=float(area.sum()),
        area_definition="实际commit中按宿主去重的旧裂谷出生候选面积累计；不是现今净伸展面积",
        diagnostic_classification="两侧边缘三尺度拟合的一致性，仅支持度；无独立接触历史，不能确认为真实伸展",
        scales_km=all_rows["scales_km"].tolist(),
        host_parity=run_meta.get("host_parity"),
        status={str(k):dict(count=int(np.sum(status==k)),area_km2=float(area[status==k].sum()),
                           area_fraction=float(area[status==k].sum()/area.sum())) for k in np.unique(status)},
        stencil_radius_quantile_km=np.quantile(all_rows["stencil_radius_km"],[.1,.5,.9,1.]).tolist(),
        width_proxy_quantile_km=np.quantile(all_rows["width_proxy_km"],[.1,.5,.9,1.]).tolist(),
        commits=commits,elapsed_s=elapsed,run_note=run_meta.get("elapsed_note"),forward_log_reported_seconds=run_meta.get("forward_log_reported_seconds"),
        raw_file=".cache/u36_s3/actual_candidates.npz",
        raw_sha256=hashlib.sha256((DEST/"actual_candidates.npz").read_bytes()).hexdigest())
    pair_summary={}
    for pair in np.unique(all_rows["pair"]):
        use=all_rows["pair"]==pair
        pair_summary[str(pair)]=dict(count=int(use.sum()),area_km2=float(area[use].sum()),
             status_area_km2={str(s):float(area[use&(status==s)].sum()) for s in np.unique(status)})
    summary["pair"]=pair_summary
    baseline=ROOT/".cache/u36_s2/control"
    compatibility=compare_npz(baseline/"forward.npz",DEST/"forward.npz")
    old_meta=json.loads((baseline/"forward.json").read_text(encoding="utf-8"))
    new_meta=json.loads((DEST/"forward.json").read_text(encoding="utf-8"))
    compatibility.update(budget_identical=old_meta["budget"]==new_meta["budget"],
                         events_identical=old_meta["events"]==new_meta["events"])
    with (baseline/"final_state.pkl").open("rb") as stream:
        old=pickle.load(stream)
    with (DEST/"final_state.pkl").open("rb") as stream:
        new=pickle.load(stream)
    compatibility["native_state_identical"]=state_hash(old["engine"])==state_hash(new["engine"])
    with open_netcdf(baseline/"world.nc") as aa,open_netcdf(DEST/"world.nc") as bb:
        compatibility["world_elevation_identical"]=np.array_equal(aa.elevation.values,bb.elevation.values)
    protected=hashes()
    import subprocess
    for p in ("packages/tectonics/engine.py","packages/tectonics/crust.py","packages/tectonics/kinematics.py","config/metrics.yaml"):
        protected[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    compatibility["protected_worktree_sha256"]=protected
    compatibility["protected_identical_to_head"]={p:subprocess.check_output(["git","hash-object",p],cwd=ROOT)==subprocess.check_output(["git","rev-parse","HEAD:"+p],cwd=ROOT) for p in protected}
    compatibility["protected_comparison"]="Git clean过滤后的对象哈希；工作区CRLF与仓库LF分开记录，不比较不同换行的原始字节"
    assert all(compatibility[k] for k in ("identical","budget_identical","events_identical","native_state_identical","world_elevation_identical"))
    assert all(compatibility["protected_identical_to_head"].values())
    (DATA/"u36_s3_actual.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (DATA/"u36_s3_verification.json").write_text(json.dumps(compatibility,ensure_ascii=False,indent=2),encoding="utf-8")
    print("U36-S3 actual",json.dumps(summary["status"],ensure_ascii=False),elapsed,flush=True)


if __name__=="__main__":
    summarize() if "--report-only" in sys.argv else run()
