"""奇景覆盖度核对（阶段 0.5b）：校验 UNESCO 遗产地 → 原型 的归类，并生成统计。

输入：
  data/earth_reference/whc_vii_viii.csv        遗产地清单（scripts/fetch_whc_sites.py 生成）
  data/earth_reference/whc_archetype_map.yaml  人工归类
  docs/archetype_library.md                    原型库（解析表格第一列为"原型"的所有表）
输出：
  docs/archetype_coverage.md 中 <!-- AUTO:BEGIN --> 与 <!-- AUTO:END --> 之间的内容（其余部分手写，不动）

校验（任何一条不满足即退出码 1）：
  - CSV 中每个遗产地都已归类，且只归类一次；
  - 归类文件中的编号都在 CSV 里（标了 extra: true 的补充核对条目除外）；
  - 每个原型名都能在原型库中找到；每个遗产地至少一个原型。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\archetype_coverage.py          # 校验并更新文档
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\archetype_coverage.py --check  # 只校验
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "docs" / "archetype_library.md"
SITES_CSV = ROOT / "data" / "earth_reference" / "whc_vii_viii.csv"
MAP_YAML = ROOT / "data" / "earth_reference" / "whc_archetype_map.yaml"
DOC = ROOT / "docs" / "archetype_coverage.md"
BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"
NEW_MARK = "0.5b 新增"


@dataclass
class Archetype:
    name: str
    section: str   # 所在小节标题，例如 "5.9 地球历史与深部构造（0.5b 新增）"
    level: str     # 表达等级首字母 A / B / C；第 1–4 节表格没有等级列，记为 "A/B"
    level_text: str

    @property
    def is_new(self) -> bool:
        return NEW_MARK in self.section


def parse_library(path: Path = LIBRARY) -> dict[str, Archetype]:
    """解析原型库中所有首列为"原型"的表格。"""
    out: dict[str, Archetype] = {}
    section = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            section = ln.lstrip("#").strip()
        cells = [c.strip() for c in ln.strip().strip("|").split("|")] if ln.startswith("|") else []
        if cells and cells[0] == "原型":
            level_col = next((k for k, c in enumerate(cells) if c.startswith("等级")), None)
            i += 2  # 跳过表头与分隔行
            while i < len(lines) and lines[i].startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                name = row[0].replace("**", "").strip()
                if level_col is not None and level_col < len(row):
                    text = row[level_col]
                    level = text[0] if text[:1] in ("A", "B", "C") else "?"
                else:
                    text, level = "A 或 B（第 1–4 节默认）", "A/B"
                if name in out:
                    raise ValueError(f"原型名重复：{name}")
                out[name] = Archetype(name, section, level, text)
                i += 1
            continue
        i += 1
    return out


def load_sites(path: Path = SITES_CSV) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["whc_id"]: r for r in csv.DictReader(f)}


def load_map(path: Path = MAP_YAML) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for e in data:
        e["id"] = str(e["id"])
    return data


def validate(lib: dict[str, Archetype], sites: dict[str, dict], mapping: list[dict]) -> list[str]:
    errors = []
    counts = Counter(e["id"] for e in mapping)
    errors += [f"编号 {k} 归类了 {n} 次" for k, n in counts.items() if n > 1]
    mapped = set(counts)
    errors += [f"遗产地 {k}（{sites[k]['name_en']}）没有归类" for k in sorted(set(sites) - mapped, key=int)]
    for e in mapping:
        if e["id"] not in sites and not e.get("extra"):
            errors.append(f"编号 {e['id']} 不在遗产地清单中（如为补充核对条目，请标 extra: true）")
        if not e.get("archetypes"):
            errors.append(f"编号 {e['id']} 没有原型")
        for a in e.get("archetypes") or []:
            if a not in lib:
                errors.append(f"编号 {e['id']}：原型库中找不到「{a}」")
    return errors


def render(lib: dict[str, Archetype], sites: dict[str, dict], mapping: list[dict]) -> str:
    core = [e for e in mapping if not e.get("extra")]
    extra = [e for e in mapping if e.get("extra")]
    n = len(core)
    n_vii = sum(sites[e["id"]]["vii"] == "1" for e in core)
    n_viii = sum(sites[e["id"]]["viii"] == "1" for e in core)
    primary = Counter(e["archetypes"][0] for e in core)   # 统计只算正式清单，补充核对条目不计入
    anyuse: dict[str, list[str]] = defaultdict(list)
    for e in core:
        for a in e["archetypes"]:
            anyuse[a].append(e["name"])
    level_primary = Counter(lib[e["archetypes"][0]].level for e in core)
    need_new = [e for e in core if any(lib[a].is_new for a in e["archetypes"])]
    only_new = [e for e in core if all(lib[a].is_new for a in e["archetypes"])]
    primary_new = [e for e in core if lib[e["archetypes"][0]].is_new]

    out = []
    out.append("> 本节由 `scripts/archetype_coverage.py` 生成，请勿手改。\n")
    out.append("### 总览\n")
    out.append(f"- 核对的遗产地：**{n}** 处（标准 vii {n_vii} 处、viii {n_viii} 处，两者兼有 "
               f"{n_vii + n_viii - n} 处）；另有补充核对 {len(extra)} 处（见逐项表末尾）。")
    out.append(f"- 全部归类完成：每处至少对应一个原型，共用到 **{len(anyuse)}** 个原型"
               f"（原型库现有 {len(lib)} 个，其中 0.5b 新增 {sum(a.is_new for a in lib.values())} 个）。")
    out.append(f"- 用到 0.5b 新增原型的遗产地 {len(need_new)} 处；其中主原型是新增原型的 {len(primary_new)} 处，"
               f"**只能**用新增原型描述的 {len(only_new)} 处。也就是说，补充前的原型库有 {len(only_new)} 处遗产地完全归不进去，"
               f"另有 {len(primary_new) - len(only_new)} 处只能描述其次要景观。")
    lv = "；".join(f"{k} 级 {v} 处" for k, v in sorted(level_primary.items()))
    out.append(f"- 按主原型的表达等级：{lv}（\"A/B\" 指第 1–4 节没有单列等级的原型）。\n")

    out.append("### 各原型被引用的次数\n")
    out.append("| 原型 | 所在节 | 等级 | 作主原型 | 全部引用 | 遗产地举例 |")
    out.append("|---|---|---|---|---|---|")
    for a in sorted(anyuse, key=lambda x: (-len(anyuse[x]), x)):
        m = re.match(r"[\d.]+", lib[a].section)
        sec = m.group(0).strip(".") if m else lib[a].section
        ex = "、".join(anyuse[a][:4]) + ("等" if len(anyuse[a]) > 4 else "")
        tag = "（新）" if lib[a].is_new else ""
        out.append(f"| {a}{tag} | {sec} | {lib[a].level} | {primary.get(a, 0)} | {len(anyuse[a])} | {ex} |")
    unused = [a for a in lib if a not in anyuse]
    out.append("")
    out.append(f"原型库中**没有** UNESCO（vii/viii）遗产地对应的原型 {len(unused)} 个："
               + "、".join(unused) + "。这些原型仍然有效（多为海底、天外、尺度太小或未单独列入遗产名录的景观），"
               "说明原型库的范围超出了 UNESCO 名单。\n")

    out.append("### 逐项归类\n")
    out.append("主原型加粗；标\"新\"的为 0.5b 新增原型。")
    out.append("")
    out.append("| 编号 | 遗产地 | 标准 | 原型 | 依据 |")
    out.append("|---|---|---|---|---|")
    for e in sorted(core, key=lambda x: int(x["id"])) + extra:
        crit = sites[e["id"]]["criteria"] if e["id"] in sites else "见依据"
        arch = []
        for k, a in enumerate(e["archetypes"]):
            s = f"{a}（新）" if lib[a].is_new else a
            arch.append(f"**{s}**" if k == 0 else s)
        name = e["name"] + ("（补充）" if e.get("extra") else "")
        out.append(f"| {e['id']} | {name} | {crit} | {'；'.join(arch)} | {e.get('note', '')} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不更新文档")
    args = ap.parse_args()

    lib = parse_library()
    sites = load_sites()
    mapping = load_map()
    errors = validate(lib, sites, mapping)
    if errors:
        print("校验失败：")
        for e in errors:
            print("  -", e)
        return 1
    print(f"校验通过：{len(sites)} 处遗产地全部归类，原型库 {len(lib)} 个原型")
    if args.check:
        return 0

    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{DOC.relative_to(ROOT)} 中缺少 {BEGIN} / {END} 标记")
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + BEGIN + "\n" + render(lib, sites, mapping) + END + tail, encoding="utf-8")
    print(f"已更新 {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
