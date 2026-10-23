"""原型库 × 终图核对表（PROGRESS U22）：原型库改名、增删原型或终图删改要素后，核对表必须同步更新。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dac = _load("design_archetype_check", "design_archetype_check.py")


def test_v2_check_covers_every_archetype():
    lib = dac.parse_library()
    chk, dsg = dac.load("v2")
    assert dac.validate(lib, chk, dac.design_features(dsg)) == []
    assert len(chk["entries"]) == len(lib) >= 100


def test_validation_catches_missing_entries_and_bad_features():
    lib = dac.parse_library()
    chk, dsg = dac.load("v2")
    broken = {**chk, "entries": [dict(e) for e in chk["entries"][1:]]}      # 漏掉第一个原型
    broken["entries"][0]["features"] = ["不存在的要素"]
    broken["entries"][1]["status"] = "maybe"
    errors = dac.validate(lib, broken, dac.design_features(dsg))
    assert any("没有出现在核对表里" in e for e in errors)
    assert any("不在终图里" in e for e in errors)
    assert any("不是" in e for e in errors)


def test_design_options_replace_features():
    import yaml

    dfm = _load("design_final_map", "design_final_map.py")
    raw = yaml.safe_load((ROOT / "config" / "design" / "final_map_v2.yaml").read_text(encoding="utf-8"))
    base = dfm.apply_options(raw, set())
    big = dfm.apply_options(raw, {"bigA"})
    ids = lambda d, k: {it["id"] for it in d[k]}                              # noqa: E731
    assert "P1" in ids(base, "peninsulas") and "P1b" not in ids(base, "peninsulas")
    assert "P1b" in ids(big, "peninsulas") and "P1" not in ids(big, "peninsulas")
    assert "MA2" in ids(big, "masses") and "MA2" not in ids(base, "masses")
