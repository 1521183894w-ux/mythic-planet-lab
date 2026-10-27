"""U36-S2：隔离运行真实历史的面积展开候选与原规则对照。

method_class=procedural_constrained；不更改默认世界。大状态仅写.cache，可重算。
"""
from __future__ import annotations
import argparse
import inspect
from pathlib import Path
import pickle
import shutil
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=("control", "conservative"), required=True)
    args = ap.parse_args()
    case = args.case
    dest = ROOT / ".cache/u36_s2" / case
    dest.mkdir(parents=True, exist_ok=True)
    original_config = ROOT / "config/design/history_v2_u35_c10_progressive.yaml"
    history = yaml.safe_load(original_config.read_text(encoding="utf-8"))
    if case == "conservative":
        history["forward"]["engine_params"]["rift_area_conservative"] = True
    config = ROOT / "config/design" / ("history_v2_u36_s2_" + case + ".yaml")
    if case == "control":
        config = original_config
    else:
        text = original_config.read_text(encoding="utf-8")
        anchor = "    hole_borrow_shared_budget: true"
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + "\n    rift_area_conservative: true")
        config.write_text(text, encoding="utf-8")
    original_simulate = fwd.simulate

    def capture(*a, **kw):
        bound = inspect.signature(original_simulate).bind(*a, **kw)
        bound.apply_defaults()
        eng, frames, vframes = result = original_simulate(*a, **kw)
        with (dest / "final_state.pkl").open("wb") as stream:
            pickle.dump(dict(engine=eng, fixed_frame=frames[-1], volcanoes=vframes[-1],
                             plumes=bound.arguments.get("plumes"),
                             rifts=bound.arguments.get("rifts"),
                             rot=str(bound.arguments["rot"])), stream, protocol=5)
        return result

    fwd.simulate = capture
    old_argv = sys.argv
    candidate = "u36_s2_" + case
    sys.argv = ["design_forward.py", "--design", "v2", "--history", "v2",
                "--history-config", str(config), "--candidate", candidate, "--no-figures"]
    start = time.perf_counter()
    try:
        rc = fwd.main()
        for suffix in (".npz", ".json"):
            shutil.copy2(ROOT / f".cache/tectonics/design/candidates/{candidate}/forward_v2_v2_L7{suffix}",
                         dest / ("forward" + suffix))
        world_dir = ROOT / "data/worlds/design_v2/candidates" / candidate
        shutil.copy2(world_dir / "forward_v2.json", dest / "summary.json")
        shutil.copy2(world_dir / f"design_v2_forward_v2_0Ma_{candidate}_1024x512.nc", dest / "world.nc")
        print(f"[U36-S2] {case} 完整运行结束 {time.perf_counter()-start:.2f}s", flush=True)
        return rc
    finally:
        sys.argv, fwd.simulate = old_argv, original_simulate


if __name__ == "__main__":
    raise SystemExit(main())
