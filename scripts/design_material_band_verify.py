"""归档U36-S4不改默认世界、实际材料身份与图件的可核对证据。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/worlds/design_v2"


def main():
    protected=["config/design/final_map_v2.yaml","config/design/history_v2.yaml",
        "config/design/history_v2_u35_c4_strain.yaml","config/design/history_v2_u35_c10_progressive.yaml",
        "config/metrics.yaml","data/worlds/design_v2/history_v2.rot","data/worlds/design_v2/forward_v2.json",
        "packages/tectonics/engine.py","packages/tectonics/crust.py","packages/tectonics/kinematics.py"]
    clean={p:subprocess.check_output(["git","hash-object",p],cwd=ROOT,text=True).strip() for p in protected}
    expected={p:subprocess.check_output(["git","rev-parse","HEAD:"+p],cwd=ROOT,text=True).strip() for p in protected}
    assert clean==expected
    with np.load(ROOT/".cache/u36_s4/material_band.npz") as data:
        source_uid=data["source_uid"]
        identity=dict(vertex_count=len(data["xyz"]),triangle_count=len(data["triangles"]),
            snapshots=len(data["times_ma"]),new_material_cells=0,
            initial_query_host_count=len(np.unique(source_uid[source_uid>=0])),
            note="固定审计三角形编号/连接关系不变；源UID是初始查询对应，尚非宿主精确裁切的物质运输")
    paths=list((ROOT/"figures/design/v2/forward/u36_s4").glob("*.png"))+list((ROOT/"figures/stage1/u36_s4_reference").rglob("*.png"))
    test_log=(ROOT/".cache/u36_s4_tests.log").read_text(encoding="utf-8-sig")
    summaries=re.findall(r"(?m)^([^\n]*\b(?:passed|failed|skipped|xfailed)\b[^\n]*\bin [\d.]+s)\s*$",test_log)
    if not summaries:
        raise RuntimeError("缺少本轮pytest实际结束汇总，不生成验证报告")
    report=dict(protected_git_objects=clean,all_protected_identical_to_head=True,material_identity=identity,
        png_count=len(paths),png_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        test_summary=summaries[-1].strip(),
        test_all_passed=re.search(r"\b[1-9]\d* failed\b",summaries[-1]) is None,
        tested_quality_thresholds_changed=False,default_world_adopted=False,
        diagram_note="专项是小试材料带，不是新世界DEM；标准组为旁观控制世界")
    assert report["png_count"]==34
    (DATA/"u36_s4_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # 日志仅消除本机临时目录用户标识；不更改命令结果、数值或测试栈。
    for source,target in [(".cache/u36_s4_tests.log","u36_s4_tests.txt"),(".cache/u36_s4_initial_failure.log","u36_s4_initial_test_failure.txt")]:
        file=ROOT/source
        if file.exists():
            log=file.read_text(encoding="utf-8-sig")
            log=log.replace(str(ROOT),"<workspace>").replace(ROOT.as_posix(),"<workspace>")
            log=re.sub(r"pytest-of-[^/\\'\s]+","pytest-of-<local-user>",log)
            log="\n".join(line.rstrip() for line in log.splitlines())+"\n"
            (DATA/target).write_text(log,encoding="utf-8")
    print(json.dumps(dict(protected=len(clean),png_count=len(paths),identity=identity),ensure_ascii=False))


if __name__=="__main__":main()
