"""U35-C3 小规模验证：同一个真实 0 Ma 提交前状态，只改变 C–ALB 既有填隙配置。"""
from pathlib import Path
import sys,pickle,json,argparse,shutil,uuid
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as f
import design_west_arc_report as base
from mythic.tectonics.crust import CONT
from mythic.tectonics.mesh import Raster
OUT=ROOT/".cache/u35_c3"
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-candidate",action="store_true",help="完整复现已否决候选；配置只在内存增补，结果归档 .cache/u35_c3/latest，结束后恢复原共用结果。")
    args=ap.parse_args()
    if args.full_candidate:
        import design_west_arc_diagnostic as observer
        original=f.yaml
        backup=OUT/("formal_backup_"+uuid.uuid4().hex[:10])
        paths=[ROOT/"data/worlds/design_v2/forward_v2.json",ROOT/"data/worlds/design_v2/design_v2_forward_v2_0Ma_1024x512.nc"]
        paths += [ROOT/(".cache/tectonics/design/forward_v2_v2_L7"+ext) for ext in (".npz",".json")]
        paths += [p for p in (ROOT/"figures/design/v2/forward").glob("*.png") if p.name[:2].isdigit() and int(p.name[:2])<=14]
        saved=[]
        for path in paths:
            if not path.exists():continue
            rel=path.relative_to(ROOT);dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dst);saved.append((dst,path))
        def effective_config(stream):
            cfg=original.safe_load(stream)
            if isinstance(cfg,dict) and "forward" in cfg and "blocks" in cfg:
                rules=cfg["forward"].setdefault("pull_apart",[])
                assert not any(set(r["pair"])=={"C","ALB"} for r in rules)
                rules.extend(base.read_json(ROOT/"data/worlds/design_v2/u35_c3_candidate_patch.json")["append"])
                base.write_json(OUT/"candidate_effective_history.json",cfg)
            return cfg
        f.yaml=SimpleNamespace(safe_load=effective_config)
        previous=sys.argv
        sys.argv=["design_west_arc_diagnostic.py","--case","latest","--output-root",".cache/u35_c3"]
        try:return observer.main()
        finally:
            f.yaml=original;sys.argv=previous
            for src,dst in saved:shutil.copy2(src,dst)
            print("候选实验结束，共用正式结果已从备份恢复：",backup.relative_to(ROOT),flush=True)
    rows=[]
    for case in ("original","continental_margin"):
        with (OUT/"observed/engine_precommit_000.pkl").open("rb") as h:e=pickle.load(h)
        if case!="original":
            e.cont_gap_pairs[301304]={"knots":[(0,32.),(300,31.4),(460,18.)],"holes":True}
            e.cont_gap_since[301304]=38.
        before=dict(e.budget)
        e.commit()
        s=e.snapshot()
        raster=Raster(e.m,512)
        E=raster(s["elev"]); C=raster(s["ctype"])
        np.savez_compressed(OUT/f"commit_spike_{case}.npz",E=E,C=C,T=raster(s["T"]))
        rec={"case":case,"budget_delta":{k:float(v-before.get(k,0)) for k,v in e.budget.items() if v!=before.get(k,0)},"critical":[]}
        for n in (100961,68047):
            i=int(e.cover_fixed(e.positions())[n])
            rec["critical"].append({"node":n,"uid":int(e.c.uid[i]),"type":int(e.c.ctype[i]),"T_km":float(e.c.T[i]),"elevation_m":float(s["elev"][n])})
        rows.append(rec)
    (OUT/"commit_spike.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
