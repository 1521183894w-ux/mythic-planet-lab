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


def test_iugs_sites_mapped_to_archetypes_or_minerals():
    """0.5c：IUGS 地质遗产地每处都已归类，原型名和矿床类型名都能在两份文档中找到。"""
    lib, minerals = ac.parse_library(), ac.parse_minerals()
    sites, mapping = ac.load_iugs_sites(), ac.load_map(ac.IUGS_MAP)
    assert len(sites) >= 200
    assert ac.validate_iugs(lib, minerals, sites, mapping) == []
    assert lib["泥火山与油气渗漏"].is_new_c and not lib["泥火山与油气渗漏"].is_new
    assert "锂（盐湖卤水）" in minerals and "汞" in minerals   # 去掉了"0.5c 新增"标记


def test_iugs_validation_catches_unknown_mineral_and_empty_entry():
    lib, minerals = ac.parse_library(), ac.parse_minerals()
    sites, mapping = ac.load_iugs_sites(), ac.load_map(ac.IUGS_MAP)
    broken = [dict(e) for e in mapping]
    broken[0] = {**broken[0], "archetypes": [], "minerals": []}       # 既无原型也无矿床
    broken[1] = {**broken[1], "minerals": ["不存在的矿"]}
    errors = ac.validate_iugs(lib, minerals, sites, broken)
    assert any("既没有原型也没有矿床类型" in e for e in errors)
    assert any("找不到「不存在的矿」" in e for e in errors)
