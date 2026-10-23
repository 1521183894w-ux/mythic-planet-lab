"""原型库 × 终图核对表：校验并生成文档（PROGRESS U22）。

输入：
  config/design/archetype_check_<版本>.yaml   逐条核对（状态、终图要素编号、候选位置、说明、确认阶段）
  config/design/final_map_<版本>.yaml         终图设计（含所有选项的要素，用来核对要素编号）
  docs/archetype_library.md                   原型库（解析方法与 scripts/archetype_coverage.py 相同）
输出：
  docs/design_archetype_check_<版本>.md       核对表文档（整份自动生成）

校验（任何一条不满足即退出码 1）：原型库的每个原型都出现且只出现一次；状态是四种之一；
features 里的编号都是终图里的要素；placed / added 至少有一个要素；candidate 至少有一个要素或候选位置。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\design_archetype_check.py --design v2
    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\design_archetype_check.py --design v2 --check
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("placed", "added", "candidate", "independent")

_spec = importlib.util.spec_from_file_location("archetype_coverage", ROOT / "scripts" / "archetype_coverage.py")
_ac = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("archetype_coverage", _ac)
_spec.loader.exec_module(_ac)
parse_library = _ac.parse_library


def load(design: str) -> tuple[dict, dict]:
    chk = yaml.safe_load((ROOT / "config" / "design" / f"archetype_check_{design}.yaml").read_text(encoding="utf-8"))
    dsg = yaml.safe_load((ROOT / "config" / "design" / f"final_map_{design}.yaml").read_text(encoding="utf-8"))
    return chk, dsg


def design_features(dsg: dict) -> dict[str, str]:
    """终图里所有要素：编号 -> 名称（不管选项开关，全部算在内）。"""
    out = {}
    for v in dsg.values():
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and "id" in it:
                    out[it["id"]] = it.get("name", it["id"])
    return out


def validate(lib: dict, chk: dict, feats: dict[str, str]) -> list[str]:
    errors = []
    seen = Counter(e["archetype"] for e in chk["entries"])
    for name in lib:
        if seen[name] == 0:
            errors.append(f"原型「{name}」没有出现在核对表里")
    for name, n in seen.items():
        if name not in lib:
            errors.append(f"核对表里的「{name}」在原型库中找不到")
        if n > 1:
            errors.append(f"「{name}」出现了 {n} 次")
    for e in chk["entries"]:
        st = e.get("status")
        if st not in STATUSES:
            errors.append(f"「{e['archetype']}」的状态 {st!r} 不是 {STATUSES} 之一")
        for f in e.get("features", []):
            if f not in feats:
                errors.append(f"「{e['archetype']}」引用的要素 {f} 不在终图里")
        if st in ("placed", "added") and not e.get("features"):
            errors.append(f"「{e['archetype']}」状态为 {st}，但没有写终图要素")
        if st == "candidate" and not (e.get("features") or e.get("at")):
            errors.append(f"「{e['archetype']}」是候选地点，但没有要素也没有位置")
    return errors


def render(lib: dict, chk: dict, feats: dict[str, str], design: str) -> str:
    st_name = chk["statuses"]
    entries = chk["entries"]
    order = {name: i for i, name in enumerate(lib, 1)}
    by_name = {e["archetype"]: e for e in entries}
    cnt = Counter(e["status"] for e in entries)
    num = {e["archetype"]: i for i, e in enumerate(entries, 1)}
    lines = [
        f"# 原型库 × 终图 {design} 核对表",
        "",
        "> 本文件由 `scripts/design_archetype_check.py` 从 `config/design/archetype_check_"
        f"{design}.yaml` 生成，不要手改。图：`figures/design/{design}/06_archetype_stage.png`"
        "（编号与下表相同）。",
        "> 目的：终图定稿前，逐条看原型库（`docs/archetype_library.md`）的每个原型，终图有没有给它留出成因条件、在哪里"
        "（PROGRESS U22，用户 2026-10-23 要求）。",
        "",
        "## 汇总",
        "",
        f"原型库共 {len(lib)} 个原型：",
        "",
    ]
    for s in STATUSES:
        lines.append(f"- **{st_name[s]}** {cnt[s]} 个")
    lines += [
        "",
        "状态的含义：",
        "",
        "- **终图已安排**：v1 里已经有提供成因条件的要素。",
        "- **v2 新增**：核对时发现缺、按成因补进终图 v2 的。",
        "- **候选地点**：终图给出了大尺度条件（地形、纬度、岩性、海陆位置），但形态要等后续阶段的过程算出来才能确认"
        "（例如沙漠要等气候、大三角洲要等河网）；表中给出最可能出现的地方。",
        "- **不依赖终图**：尺度远小于终图（C 级叙事标注）、在合适的地方普遍出现、随机事件，或行星级现象；终图不需要为它做安排。",
        "",
        "确认阶段：1 构造演化 / 1.2e 岩性 / 2 地形与水系 / 3 气候、冰川与海洋 / C 只作叙事标注。",
        "",
        "## 逐条核对",
        "",
    ]
    section = None
    for name, a in lib.items():
        if a.section != section:
            section = a.section
            lines += ["", f"### {section}", "", "| 编号 | 原型 | 等级 | 状态 | 终图上的位置（要素） | 说明 | 确认阶段 |",
                      "|---|---|---|---|---|---|---|"]
        e = by_name[name]
        where = "、".join(f"{f} {feats[f]}" for f in e.get("features", []))
        if e.get("at"):
            pts = "；".join(f"{lo:g}°, {la:g}°" for lo, la in e["at"])
            where = f"{where}；候选位置 {pts}" if where else f"候选位置 {pts}"
        lines.append(f"| {num[name]} | {name} | {a.level} | {st_name[e['status']]} | {where or '—'} | "
                     f"{e.get('note', '')} | {e.get('stage', '')} |")
    lines.append("")
    assert len(order) == len(entries)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="v2")
    ap.add_argument("--check", action="store_true", help="只校验，不写文档")
    args = ap.parse_args()
    lib = parse_library()
    chk, dsg = load(args.design)
    feats = design_features(dsg)
    errors = validate(lib, chk, feats)
    for e in errors:
        print("错误：", e)
    if errors:
        return 1
    cnt = Counter(e["status"] for e in chk["entries"])
    print(f"{len(lib)} 个原型全部核对；" + "，".join(f"{chk['statuses'][s]} {cnt[s]}" for s in STATUSES))
    if not args.check:
        out = ROOT / "docs" / f"design_archetype_check_{args.design}.md"
        out.write_text(render(lib, chk, feats, args.design), encoding="utf-8", newline="\n")
        print(f"已写 {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
