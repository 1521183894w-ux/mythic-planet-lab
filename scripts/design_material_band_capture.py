"""U36-S4：只读保留35 Ma原生材料，完整控制重跑核对旁观无干扰。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
from design_arc_sections import state_hash
from design_west_arc_report import compare_npz
from mythic.tectonics.engine import Engine
from mythic.figures.netcdf import open_netcdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / ".cache/u35_c10/progressive")
    args = parser.parse_args()
    dest = ROOT / ".cache/u36_s4/control"
    dest.mkdir(parents=True, exist_ok=True)
    original_step, original_simulate = Engine.step, fwd.simulate
    def observe(eng, *a, **kw):
        if eng.t == 35:
            with (dest / "reference_35Ma.pkl").open("wb") as stream:
                pickle.dump(eng, stream, protocol=5)
        return original_step(eng, *a, **kw)
    final = {}
    def capture(*a, **kw):
        result = original_simulate(*a, **kw)
        final["hash"] = state_hash(result[0])
        return result
    old_argv = sys.argv
    Engine.step, fwd.simulate = observe, capture
    sys.argv = ["design_forward.py", "--history-config", "config/design/history_v2_u35_c10_progressive.yaml",
                "--candidate", "u36_s4_control", "--no-figures"]
    start = time.perf_counter()
    try:
        fwd.main()
    finally:
        Engine.step, fwd.simulate, sys.argv = original_step, original_simulate, old_argv
    for suffix in (".npz", ".json"):
        shutil.copy2(ROOT / f".cache/tectonics/design/candidates/u36_s4_control/forward_v2_v2_L7{suffix}", dest / ("forward" + suffix))
    shutil.copy2(ROOT / "data/worlds/design_v2/candidates/u36_s4_control/design_v2_forward_v2_0Ma_u36_s4_control_1024x512.nc", dest / "world.nc")
    ref = args.reference_dir
    comparison = compare_npz(ref / "forward.npz", dest / "forward.npz")
    old, new = [json.loads(p.read_text(encoding="utf-8")) for p in (ref / "forward.json", dest / "forward.json")]
    comparison.update(budget_identical=old["budget"] == new["budget"], events_identical=old["events"] == new["events"])
    with (ref / "final_state.pkl").open("rb") as stream:
        comparison["native_state_identical"] = final["hash"] == state_hash(pickle.load(stream)["engine"])
    with open_netcdf(ref / "world.nc") as a, open_netcdf(dest / "world.nc") as b:
        comparison["world_arrays_identical"] = set(a.data_vars) == set(b.data_vars) and all(
            np.array_equal(a[k].values, b[k].values, equal_nan=True) for k in a.data_vars)
    assert comparison["identical"] and all(comparison[k] for k in ("budget_identical", "events_identical", "native_state_identical", "world_arrays_identical"))
    comparison["elapsed_s"] = time.perf_counter() - start
    comparison["reference"] = "U35-C10 progressive（共享供体限额、原经验裂谷厚度、持续局部减薄）"
    (ROOT / "data/worlds/design_v2/u36_s4_control_verification.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print("control verification", comparison, flush=True)


if __name__ == "__main__":
    main()
