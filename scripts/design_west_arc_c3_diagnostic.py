"""U35-C3：只读追踪西口补网前开合与 ALB 地壳过程收支。
原模拟函数每次只调用一次；不修改世界状态或运行参数。
"""
from pathlib import Path
import sys, json, pickle
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import design_forward as f
import design_west_arc_diagnostic as old
from mythic.tectonics.crust import CONT, ORIGIN_RIFT
from mythic.tectonics.geometry import xyz_to_lonlat, km_from_chord, R_KM
from mythic.tectonics.volcanism import VolcanoField
OUT=ROOT/".cache/u35_c3/observed"
OUT.mkdir(parents=True,exist_ok=True)
ledger=[]; gaps=[]; contacts=[]
def mass(e):
    c=e.c; a=e.area
    own=e.alive&(c.plate==304)&(c.ctype==CONT)
    src=e.alive&(c.origin==304)&(c.ctype==CONT)
    return {"own_volume_km3":float((c.T[own]*a[own]).sum()),
            "source_volume_km3":float((c.T[src]*a[src]).sum()),
            "own_n":int(own.sum()),"source_n":int(src.sum())}
def main():
    old.OUT=ROOT/".cache/u35_c3"
    originals={name:getattr(f.Engine,name) for name in ("_interact","_arc","_surface","commit","_resolve")}
    original_volc=VolcanoField.step
    for name in ("_interact","_arc","_surface"):
        fn=originals[name]
        def observe(e,*args,_fn=fn,_name=name,**kwargs):
            if e.t>10:return _fn(e,*args,**kwargs)
            before=mass(e); b=dict(e.budget)
            result=_fn(e,*args,**kwargs)
            after=mass(e)
            ledger.append({"t_Ma":float(e.t),"process":_name,
                "delta":{k:after[k]-before[k] for k in before},
                "budget_delta":{k:float(v-b.get(k,0)) for k,v in e.budget.items() if v!=b.get(k,0)}})
            return result
        setattr(f.Engine,name,observe)
    def resolve(e,pos,U,Q,B):
        if e.t>15:return originals["_resolve"](e,pos,U,Q,B)
        c=e.c
        sel=e.alive[U]&(c.ctype[U]==CONT)&(c.ctype[B]==CONT)&((c.plate[U]==304)|(Q==304))
        rows=[]
        for k in np.flatnonzero(sel):
            a,b=int(U[k]),int(B[k]);p,q=int(c.plate[a]),int(Q[k])
            u=pos[a]-pos[b]
            length=np.linalg.norm(u)
            vn=float(np.cross(e.omega[p]-e.omega[q],pos[a])@u/max(length,1e-12)*R_KM/10)
            lo,la=xyz_to_lonlat(pos[a])
            pair=min(p,q)*1000+max(p,q)
            rule=e.no_collision_pairs.get(pair)
            rows.append({"uid":int(c.uid[a]),"host_uid":int(c.uid[b]),"src":p,"dst":q,"lon":float(lo),"lat":float(la),
                "T_km":float(c.T[a]),"volume_km3":float(c.T[a]*e.area[a]),
                "nearest_normal_separation_cm_yr":vn,
                "suppression_active":bool(rule and rule[1]-1e-9<=e.t<=rule[0]+1e-9),
                "_idx":a})
        result=originals["_resolve"](e,pos,U,Q,B)
        for row in rows:
            row["consumed"]=not bool(e.alive[row.pop("_idx")])
            row["t_Ma"]=float(e.t);contacts.append(row)
        return result
    f.Engine._resolve=resolve
    def commit(e):
        if e.t<=10:
            if e.t==0:
                with (OUT/"engine_precommit_000.pkl").open("wb") as out:pickle.dump(e,out,protocol=5)
            pos=e.positions(); keep=np.flatnonzero(e.alive)
            missing=e.cover_fixed(pos)<0
            nodes=np.flatnonzero(missing&(e.m.lon>-26)&(e.m.lon<6)&(e.m.lat>30)&(e.m.lat<44))
            if nodes.size:
                d,k=cKDTree(pos[keep]).query(e.m.xyz[nodes],k=min(12,len(keep)))
                plates=e.c.plate[keep[k]]
                other=plates!=plates[:,:1];has=other.any(1);first=other.argmax(1)
                for i,node in enumerate(nodes):
                    a=int(keep[k[i,0]]); b=int(keep[k[i,first[i]]])
                    p,q=int(e.c.plate[a]),int(e.c.plate[b])
                    u=pos[a]-pos[b]; un=max(np.linalg.norm(u),1e-12)
                    vn=float(np.cross(e.omega[p]-e.omega[q],e.m.xyz[node])@u/un*R_KM/10)
                    record={"t_Ma":float(e.t),"fixed_node":int(node),"lon":float(e.m.lon[node]),"lat":float(e.m.lat[node]),
                         "plate_pair":[p,q],"has_two_plates":bool(has[i]),"separation_cm_yr":vn,
                         "source_uids":[int(e.c.uid[a]),int(e.c.uid[b])],
                         "source_types":[int(e.c.ctype[a]),int(e.c.ctype[b])],
                         "source_T_km":[float(e.c.T[a]),float(e.c.T[b])],
                         "source_origin":[int(e.c.origin[a]),int(e.c.origin[b])],
                         "nearest_distances_km":[float(km_from_chord(d[i,0])),float(km_from_chord(d[i,first[i]]))],
                         "gap_rule":str(e.cont_gap_pairs.get(min(p,q)*1000+max(p,q)))}
                    gaps.append(record)
        return originals["commit"](e)
    f.Engine.commit=commit
    def volc_step(field,e):
        result=original_volc(field,e)
        if abs(e.t-10)<1e-9:
            with (OUT/"engine_010.pkl").open("wb") as out:pickle.dump(e,out,protocol=5)
        return result
    VolcanoField.step=volc_step
    previous=sys.argv
    sys.argv=["design_forward.py","--design","v2","--history","v2","--no-figures"]
    try:
        rc=f.main()
        old.archive_run("observed")
        payload={"checkpoint":"U35-C3","baseline_commit":"a2d5369","method_class":"procedural_constrained",
            "units":{"time":"Ma","volume":"km3","thickness":"km","velocity":"cm/yr"},
            "ledger":ledger,"gaps":gaps,"contacts":contacts,
            "note":"最近单元连线法向仅是原算法实际判据；不等于已平滑的地质边界法向。原方法每次调用一次。"}
        (OUT/"process_diagnostic.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8",newline="\n")
        print(json.dumps({"ledger_rows":len(ledger),"gap_rows":len(gaps),"contact_rows":len(contacts)},ensure_ascii=False))
        return rc
    finally:
        for name,fn in originals.items():setattr(f.Engine,name,fn)
        VolcanoField.step=original_volc
        sys.argv=previous
if __name__=="__main__":raise SystemExit(main())
