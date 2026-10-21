"""奇景覆盖度核对（阶段 0.5b、0.5c）：校验两份外部清单 → 原型 的归类，并生成统计。

输入：
  data/earth_reference/whc_vii_viii.csv              UNESCO 世界遗产（标准 vii/viii，scripts/fetch_whc_sites.py 生成）
  data/earth_reference/whc_archetype_map.yaml        人工归类（0.5b）
  data/earth_reference/iugs_geoheritage_sites.csv    IUGS 地质遗产地（scripts/fetch_iugs_sites.py 生成）
  data/earth_reference/iugs_archetype_map.yaml       人工归类（0.5c），可另列矿床类型
  docs/archetype_library.md                          原型库（解析表格第一列为"原型"的所有表）
  docs/mineral_resources.md                          矿产资源（解析表格第一列为"矿床"或"资源"的所有表）
输出：
  docs/archetype_coverage.md 中 <!-- AUTO:BEGIN -->…<!-- AUTO:END -->（UNESCO）与
  <!-- AUTO-IUGS:BEGIN -->…<!-- AUTO-IUGS:END -->（IUGS）之间的内容（其余部分手写，不动）

校验（任何一条不满足即退出码 1）：
  - CSV 中每个遗产地都已归类，且只归类一次；
  - 归类文件中的编号都在 CSV 里（UNESCO 标了 extra: true 的补充核对条目除外）；
  - 每个原型名都能在原型库中找到；UNESCO 每处至少一个原型；
  - IUGS 每处至少一个原型或一个矿床类型，矿床类型名都能在矿产资源文档中找到。

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
IUGS_CSV = ROOT / "data" / "earth_reference" / "iugs_geoheritage_sites.csv"
IUGS_MAP = ROOT / "data" / "earth_reference" / "iugs_archetype_map.yaml"
MINERALS = ROOT / "docs" / "mineral_resources.md"
DOC = ROOT / "docs" / "archetype_coverage.md"
BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"
IUGS_BEGIN, IUGS_END = "<!-- AUTO-IUGS:BEGIN -->", "<!-- AUTO-IUGS:END -->"
NEW_MARK = "0.5b 新增"
NEW_MARK_C = "0.5c 新增"


@dataclass
class Archetype:
    name: str
    section: str   # 所在小节标题，例如 "5.9 地球历史与深部构造（0.5b 新增）"
    level: str     # 表达等级首字母 A / B / C；第 1–4 节表格没有等级列，记为 "A/B"
    level_text: str

    @property
    def is_new(self) -> bool:
        """0.5b 新增（UNESCO 核对时补充）。"""
        return NEW_MARK in self.section

    @property
    def is_new_c(self) -> bool:
        """0.5c 新增（IUGS 核对时补充）。"""
        return NEW_MARK_C in self.section

    @property
    def tag(self) -> str:
        return "（新）" if self.is_new else "（新·0.5c）" if self.is_new_c else ""


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


def mineral_key(name: str) -> str:
    """矿产资源表第一列去掉"0.5c 新增"标记，例如"锂（盐湖卤水，0.5c 新增）"→"锂（盐湖卤水）"。"""
    return name.replace("**", "").replace("（0.5c 新增）", "").replace("，0.5c 新增", "").strip()


def parse_minerals(path: Path = MINERALS) -> dict[str, str]:
    """解析矿产资源文档中首列为"矿床"或"资源"的表格，返回 {名称: 所在小节}。"""
    out: dict[str, str] = {}
    section = ""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            section = ln.lstrip("#").strip()
        cells = [c.strip() for c in ln.strip().strip("|").split("|")] if ln.startswith("|") else []
        if cells and cells[0] in ("矿床", "资源"):
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                name = mineral_key(lines[i].strip().strip("|").split("|")[0])
                if name in out:
                    raise ValueError(f"矿床名重复：{name}")
                out[name] = section
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


def load_iugs_sites(path: Path = IUGS_CSV) -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["wp_id"]: r for r in csv.DictReader(f)}


def validate_iugs(lib: dict[str, Archetype], minerals: dict[str, str], sites: dict[str, dict],
                  mapping: list[dict]) -> list[str]:
    errors = []
    counts = Counter(e["id"] for e in mapping)
    errors += [f"IUGS 编号 {k} 归类了 {n} 次" for k, n in counts.items() if n > 1]
    errors += [f"IUGS 遗产地 {sites[k]['no']}（{sites[k]['name_en']}）没有归类"
               for k in sorted(set(sites) - set(counts), key=lambda k: int(sites[k]["no"]))]
    for e in mapping:
        if e["id"] not in sites:
            errors.append(f"IUGS 编号 {e['id']} 不在遗产地清单中")
        if not e.get("archetypes") and not e.get("minerals"):
            errors.append(f"IUGS 编号 {e['id']} 既没有原型也没有矿床类型")
        for a in e.get("archetypes") or []:
            if a not in lib:
                errors.append(f"IUGS 编号 {e['id']}：原型库中找不到「{a}」")
        for m in e.get("minerals") or []:
            if m not in minerals:
                errors.append(f"IUGS 编号 {e['id']}：矿产资源文档中找不到「{m}」")
    return errors


def _sec(a: Archetype) -> str:
    m = re.match(r"[\d.]+", a.section)
    return m.group(0).strip(".") if m else a.section


def render_iugs(lib: dict[str, Archetype], minerals: dict[str, str], sites: dict[str, dict], mapping: list[dict],
                whc_mapping: list[dict]) -> str:
    rows = sorted(mapping, key=lambda e: int(sites[e["id"]]["no"]))
    n = len(rows)
    batches = Counter(sites[e["id"]]["batch"] for e in rows)
    cats = Counter(sites[e["id"]]["category"] for e in rows)
    with_arch = [e for e in rows if e.get("archetypes")]
    with_min = [e for e in rows if e.get("minerals")]
    only_min = [e for e in rows if not e.get("archetypes")]
    primary = Counter(e["archetypes"][0] for e in with_arch)
    anyuse: dict[str, list[str]] = defaultdict(list)
    for e in with_arch:
        for a in e["archetypes"]:
            anyuse[a].append(e["name"])
    minuse: dict[str, list[str]] = defaultdict(list)
    for e in with_min:
        for m in e["minerals"]:
            minuse[m].append(e["name"])
    record = primary.get("化石产地与地层剖面", 0)
    need_c = [e for e in with_arch if any(lib[a].is_new_c for a in e["archetypes"])]
    primary_c = [e for e in with_arch if lib[e["archetypes"][0]].is_new_c]
    only_c = [e for e in with_arch if all(lib[a].is_new_c for a in e["archetypes"])]
    need_b = [e for e in with_arch if any(lib[a].is_new for a in e["archetypes"])]
    level_primary = Counter(lib[e["archetypes"][0]].level for e in with_arch)
    used_both = set(anyuse) | {a for e in whc_mapping for a in e["archetypes"]}
    never = [a for a in lib if a not in used_both]

    out = []
    out.append("> 本节由 `scripts/archetype_coverage.py` 生成，请勿手改。\n")
    out.append("### 总览\n")
    out.append(f"- 核对的遗产地：**{n}** 处（" + "、".join(f"{k} {v} 处" for k, v in batches.items()) + "）。按 IUGS 分类："
               + "、".join(f"{k} {v}" for k, v in cats.most_common()) + "。")
    out.append(f"- 全部归类完成：{len(with_arch)} 处对应原型，共用到 **{len(anyuse)}** 个原型（原型库现有 {len(lib)} 个，其中 0.5b 新增 "
               f"{sum(a.is_new for a in lib.values())} 个、0.5c 新增 {sum(a.is_new_c for a in lib.values())} 个）；"
               f"{len(with_min)} 处对应 `docs/mineral_resources.md` 的矿床类型，其中 {len(only_min)} 处是纯矿床、没有景观原型。")
    out.append(f"- 主原型是\"化石产地与地层剖面\"的 {record} 处：IUGS 名单近一半看重的是地质记录（化石、地层界线、不整合），而不是景观。")
    out.append(f"- 用到 0.5c 新增原型的遗产地 {len(need_c)} 处；其中主原型是 0.5c 新增原型的 {len(primary_c)} 处，**只能**用 0.5c 新增原型描述的 "
               f"{len(only_c)} 处。也就是说，0.5c 之前的原型库有 {len(only_c)} 处归不进去，另有 {len(primary_c) - len(only_c)} 处只能描述其次要特征。")
    out.append(f"- 用到 0.5b 新增原型的遗产地 {len(need_b)} 处：0.5b 为 UNESCO 名单补充的原型，在第二份清单上同样用得上。")
    lv = "；".join(f"{k} 级 {v} 处" for k, v in sorted(level_primary.items()))
    out.append(f"- 按主原型的表达等级：{lv}。")
    out.append(f"- 两份清单合计用到 {len(used_both)} 个原型；两份清单都没有引用的原型 {len(never)} 个：" + "、".join(never)
               + "。这些多为海底、天外、生态、尺度太小或地球上罕见的景观，仍然有效。\n")

    out.append("### 各原型被引用的次数（IUGS）\n")
    out.append("| 原型 | 所在节 | 等级 | 作主原型 | 全部引用 | 遗产地举例 |")
    out.append("|---|---|---|---|---|---|")
    for a in sorted(anyuse, key=lambda x: (-len(anyuse[x]), x)):
        ex = "、".join(anyuse[a][:4]) + ("等" if len(anyuse[a]) > 4 else "")
        out.append(f"| {a}{lib[a].tag} | {_sec(lib[a])} | {lib[a].level} | {primary.get(a, 0)} | {len(anyuse[a])} | {ex} |")
    out.append("")
    out.append("### 矿床类型被引用的次数（对应 `docs/mineral_resources.md`）\n")
    out.append("| 矿床类型 | 所在节 | 引用 | 遗产地 |")
    out.append("|---|---|---|---|")
    for m in sorted(minuse, key=lambda x: (-len(minuse[x]), x)):
        tag = "（0.5c 新增）" if m in NEW_MINERALS else ""
        out.append(f"| {m}{tag} | {minerals[m]} | {len(minuse[m])} | {'、'.join(minuse[m])} |")
    out.append("")
    out.append("### 逐项归类\n")
    out.append("序号按发布日期排列（与 CSV 的 no 列一致）；主原型加粗；标\"新\"的为 0.5b 新增原型，标\"新·0.5c\"的为 0.5c 新增原型。")
    out.append("")
    out.append("| 序号 | 遗产地 | 批次 | IUGS 分类 | 原型 | 矿床 | 依据 |")
    out.append("|---|---|---|---|---|---|---|")
    for e in rows:
        s = sites[e["id"]]
        arch = [f"**{a}{lib[a].tag}**" if k == 0 else f"{a}{lib[a].tag}" for k, a in enumerate(e.get("archetypes") or [])]
        out.append(f"| {s['no']} | {e['name']}（{s['name_en']}） | {s['batch']} | {s['category']} | {'；'.join(arch) or '—'} | "
                   f"{'、'.join(e.get('minerals') or []) or '—'} | {e.get('note', '')} |")
    return "\n".join(out) + "\n"


# 0.5c 在矿产资源文档中新增的矿床类型（文档中带"0.5c 新增"标记，解析时已去掉）
NEW_MINERALS = {"汞", "铁氧化物—磷灰石铁矿", "沉积型锰", "锂（盐湖卤水）", "紫水晶、玛瑙晶洞", "青金石", "祖母绿"}


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
               f"（原型库现有 {len(lib)} 个，其中 0.5b 新增 {sum(a.is_new for a in lib.values())} 个、"
               f"0.5c 新增 {sum(a.is_new_c for a in lib.values())} 个）。")
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
        out.append(f"| {a}{lib[a].tag} | {sec} | {lib[a].level} | {primary.get(a, 0)} | {len(anyuse[a])} | {ex} |")
    unused = [a for a in lib if a not in anyuse]
    out.append("")
    out.append(f"原型库中**没有** UNESCO（vii/viii）遗产地对应的原型 {len(unused)} 个："
               + "、".join(unused) + "。这些原型仍然有效（多为海底、天外、尺度太小或未单独列入遗产名录的景观），"
               "说明原型库的范围超出了 UNESCO 名单。\n")

    out.append("### 逐项归类\n")
    out.append("主原型加粗；标\"新\"的为 0.5b 新增原型，标\"新·0.5c\"的为 0.5c 新增原型（作为次要景观补进来的）。")
    out.append("")
    out.append("| 编号 | 遗产地 | 标准 | 原型 | 依据 |")
    out.append("|---|---|---|---|---|")
    for e in sorted(core, key=lambda x: int(x["id"])) + extra:
        crit = sites[e["id"]]["criteria"] if e["id"] in sites else "见依据"
        arch = []
        for k, a in enumerate(e["archetypes"]):
            s = f"{a}{lib[a].tag}"
            arch.append(f"**{s}**" if k == 0 else s)
        name = e["name"] + ("（补充）" if e.get("extra") else "")
        out.append(f"| {e['id']} | {name} | {crit} | {'；'.join(arch)} | {e.get('note', '')} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不更新文档")
    args = ap.parse_args()

    lib = parse_library()
    minerals = parse_minerals()
    sites = load_sites()
    mapping = load_map()
    iugs_sites = load_iugs_sites()
    iugs_map = load_map(IUGS_MAP)
    errors = validate(lib, sites, mapping) + validate_iugs(lib, minerals, iugs_sites, iugs_map)
    if errors:
        print("校验失败：")
        for e in errors:
            print("  -", e)
        return 1
    print(f"校验通过：UNESCO {len(sites)} 处、IUGS {len(iugs_sites)} 处全部归类；原型库 {len(lib)} 个原型，矿床类型 {len(minerals)} 个")
    if args.check:
        return 0

    text = DOC.read_text(encoding="utf-8")
    for b, e in ((BEGIN, END), (IUGS_BEGIN, IUGS_END)):
        if b not in text or e not in text:
            print(f"{DOC.relative_to(ROOT)} 中缺少 {b} / {e} 标记")
            return 1
    text = _replace_block(text, BEGIN, END, render(lib, sites, mapping))
    text = _replace_block(text, IUGS_BEGIN, IUGS_END, render_iugs(lib, minerals, iugs_sites, iugs_map, mapping))
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"已更新 {DOC.relative_to(ROOT)}")
    return 0


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + "\n" + body + end + tail


if __name__ == "__main__":
    sys.exit(main())
