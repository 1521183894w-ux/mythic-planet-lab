"""从 Wikidata 获取 UNESCO 世界遗产中符合标准 (vii) 或 (viii) 的遗产地清单（阶段 0.5b 奇景覆盖度核对用）。

为什么用 Wikidata：UNESCO 官网（whc.unesco.org）的名单下载接口有人机验证，脚本无法也不应绕过；
Wikidata 记录了每个遗产地的 UNESCO 编号（P757）和入选标准（P2614），许可证 CC0。

处理规则：
- 只保留带 UNESCO 编号的条目；编号形如 "156"、"156bis"、"1234rev"，主体数字相同视为同一遗产地。
- 编号带 "-001" 这类后缀的是系列遗产的组成部分，不单列。
- 同一遗产地有多个 Wikidata 条目时，合并其标准、国家，名称取编号最"干净"的条目。
- 完整性自检：统计所有带编号遗产地中缺少入选标准的比例，写进输出的元数据。

输出：data/earth_reference/whc_vii_viii.csv 与同名 .meta.json（来源、查询时间、统计）。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\fetch_whc_sites.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "earth_reference" / "whc_vii_viii.csv"
OUT_META = OUT_CSV.with_suffix(".meta.json")
ENDPOINT = "https://query.wikidata.org/sparql"
# Wikimedia 要求脚本带可识别的 User-Agent（https://meta.wikimedia.org/wiki/User-Agent_policy）
UA = "mythic-planet-lab/0.1 (https://github.com/1521183894w-ux/mythic-planet-lab) research script"

CRITERIA = {  # Wikidata 中 UNESCO 入选标准 (i)–(x) 的条目
    "Q23038972": "i", "Q23038976": "ii", "Q23038977": "iii", "Q23038978": "iv", "Q23038979": "v",
    "Q23038980": "vi", "Q23038981": "vii", "Q23038983": "viii", "Q23038985": "ix", "Q23038986": "x",
}
ROMAN_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
ID_RE = re.compile(r"^(\d+)([a-z]*)$")  # 主体数字 + 可选 bis/ter/rev 等；带 "-" 的组成部分不匹配

Q_SITES = """
SELECT ?item ?id ?crit ?en ?zh ?countryEn ?coord WHERE {
  ?item wdt:P757 ?id .
  OPTIONAL { ?item wdt:P2614 ?crit . }
  OPTIONAL { ?item rdfs:label ?en . FILTER(LANG(?en) = "en") }
  OPTIONAL { ?item rdfs:label ?zh . FILTER(LANG(?zh) = "zh") }
  OPTIONAL { ?item wdt:P17 ?country . ?country rdfs:label ?countryEn . FILTER(LANG(?countryEn) = "en") }
  OPTIONAL { ?item wdt:P625 ?coord . }
}
"""


def sparql(query: str) -> list[dict]:
    for attempt in range(1, 4):
        try:
            r = requests.get(ENDPOINT, params={"query": query, "format": "json"},
                             headers={"User-Agent": UA}, timeout=(30, 300))
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except requests.RequestException as exc:
            print(f"查询失败（第 {attempt} 次）：{exc}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError("Wikidata 查询多次失败")


def val(b: dict, k: str) -> str | None:
    return b[k]["value"] if k in b else None


def main() -> int:
    t0 = time.time()
    rows = sparql(Q_SITES)
    print(f"Wikidata 返回 {len(rows)} 行（{time.time() - t0:.0f} s）", flush=True)

    sites: dict[str, dict] = defaultdict(lambda: {"criteria": set(), "countries": set(), "items": {}})
    for b in rows:
        m = ID_RE.match(val(b, "id").strip().lower())
        if not m:
            continue  # 组成部分或格式异常
        base = m.group(1)
        qid = val(b, "item").rsplit("/", 1)[1]
        s = sites[base]
        it = s["items"].setdefault(qid, {"id": val(b, "id").strip(), "en": None, "zh": None, "coord": None})
        if len(val(b, "id").strip()) < len(it["id"]):  # 一个条目可有多个编号（如 951 与 951bis），记最短的
            it["id"] = val(b, "id").strip()
        for k in ("en", "zh", "coord"):
            if it[k] is None and val(b, k):
                it[k] = val(b, k)
        c = val(b, "crit")
        if c and c.rsplit("/", 1)[1] in CRITERIA:
            s["criteria"].add(CRITERIA[c.rsplit("/", 1)[1]])
        if val(b, "countryEn"):
            s["countries"].add(val(b, "countryEn"))

    total = len(sites)
    no_crit = sorted(k for k, s in sites.items() if not s["criteria"])
    selected = {k: s for k, s in sites.items() if s["criteria"] & {"vii", "viii"}}
    print(f"带 UNESCO 编号的遗产地 {total} 个，其中缺入选标准 {len(no_crit)} 个；符合 (vii)/(viii) 的 {len(selected)} 个",
          flush=True)

    # 没有英文名的条目，取任意语言的名称兜底（例如 1728 默恩崖只有丹麦语等标签）
    no_en = sorted({q for s in selected.values() for q, it in s["items"].items() if not it["en"]})
    if no_en:
        q = ("SELECT ?item (SAMPLE(?l) AS ?any) WHERE { VALUES ?item { "
             + " ".join(f"wd:{x}" for x in no_en) + " } ?item rdfs:label ?l . } GROUP BY ?item")
        fallback = {val(b, "item").rsplit("/", 1)[1]: val(b, "any") for b in sparql(q)}
        for s in selected.values():
            for qid, it in s["items"].items():
                if not it["en"] and fallback.get(qid):
                    it["en"] = fallback[qid]

    def pick(s: dict, key: str) -> str:
        """取编号最"原始"（最短，如 951 而不是 951bis）的条目；并列的多个条目名称全部列出。"""
        cand = [it for it in s["items"].values() if it[key]]
        if not cand:
            return ""
        shortest = min(len(it["id"]) for it in cand)
        return " / ".join(sorted({it[key] for it in cand if len(it["id"]) == shortest}))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["whc_id", "name_en", "name_zh", "criteria", "vii", "viii", "countries", "lat", "lon",
                    "wikidata"])
        for base in sorted(selected, key=int):
            s = selected[base]
            lat = lon = ""
            coord = pick(s, "coord").split(" / ")[0]
            m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", coord)
            if m:
                lon, lat = f"{float(m.group(1)):.3f}", f"{float(m.group(2)):.3f}"
            crit = [c for c in ROMAN_ORDER if c in s["criteria"]]
            w.writerow([base, pick(s, "en"), pick(s, "zh"), " ".join(crit), int("vii" in crit),
                        int("viii" in crit), "; ".join(sorted(s["countries"])), lat, lon,
                        " ".join(sorted(s["items"]))])

    meta = {
        "source": "Wikidata Query Service (https://query.wikidata.org/)，属性 P757 世界遗产编号、P2614 入选标准",
        "license": "CC0 1.0（Wikidata 结构化数据）",
        "query_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "query": Q_SITES.strip(),
        "rows_returned": len(rows),
        "sites_with_whc_id": total,
        "sites_missing_criteria": len(no_crit),
        "sites_missing_criteria_ids": no_crit,
        "sites_vii_or_viii": len(selected),
        "sites_vii": sum("vii" in s["criteria"] for s in selected.values()),
        "sites_viii": sum("viii" in s["criteria"] for s in selected.values()),
        "notes": "UNESCO 官网名单下载接口有人机验证，未使用。Wikidata 是社区维护数据，可能与 UNESCO 官方名单有出入，见 docs/archetype_coverage.md 的核对说明。",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写出 {OUT_CSV.relative_to(ROOT)}（{len(selected)} 行）与元数据，用时 {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
