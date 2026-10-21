"""奇景覆盖度核对（0.5b）的一致性检查：原型库改名或删原型后，归类文件必须同步更新。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("archetype_coverage", ROOT / "scripts" / "archetype_coverage.py")
ac = importlib.util.module_from_spec(_spec)
sys.modules["archetype_coverage"] = ac  # dataclass 需要能在 sys.modules 里找到所在模块
_spec.loader.exec_module(ac)


def test_library_parses_all_tables():
    lib = ac.parse_library()
    assert len(lib) >= 100
    assert "超长碰撞造山带" in lib and lib["超长碰撞造山带"].level == "A/B"
    assert lib["熔岩管洞"].level == "C"
    assert lib["钙华台地与钙华坝湖"].is_new and not lib["丹霞"].is_new


def test_every_site_mapped_to_existing_archetypes():
    errors = ac.validate(ac.parse_library(), ac.load_sites(), ac.load_map())
    assert errors == []


def test_validation_catches_bad_names_and_gaps():
    lib, sites, mapping = ac.parse_library(), ac.load_sites(), ac.load_map()
    broken = [dict(e) for e in mapping[1:]]          # 漏掉第一个遗产地
    broken[0]["archetypes"] = ["不存在的原型"]          # 写错原型名
    errors = ac.validate(lib, sites, broken)
    assert any("没有归类" in e for e in errors)
    assert any("找不到「不存在的原型」" in e for e in errors)
