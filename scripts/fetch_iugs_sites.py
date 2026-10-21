"""获取 IUGS（国际地质科学联合会）地质遗产地名单（阶段 0.5c：原型库的第二份外部核对清单）。

为什么要第二份清单：0.5b 用 UNESCO 世界遗产（标准 vii、viii）核对原型库，192 处全部归类。但 UNESCO 名单只收录
各国申报、又被列入名录的地方，许多著名地貌不在上面（达纳基尔洼地、乌尤尼盐沼、死海、理查特结构等）。
IUGS 地质遗产地由地质学家按科学价值评选（2022 年第一批 100 处，2024 年第二批 100 处），专门针对地质与地貌，
正好补上这块。

数据来源：IUGS 国际地质遗产委员会网站（https://iugs-geoheritage.org）的 WordPress 公开接口
（/wp-json/wp/v2/geoheritage_sites），只取名称、分类、发布日期与链接这些事实信息，不保存网站的描述文字和图片。

批次按发布日期判断：2022 年发布的是第一批 100 处，2024 年发布的是第二批 100 处，其他年份单列（例如 2026 年新增）。

输出：data/earth_reference/iugs_geoheritage_sites.csv 与同名 .meta.json（来源、查询时间、统计）。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python scripts\\fetch_iugs_sites.py
"""

from __future__ import annotations

import csv
import html
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "earth_reference" / "iugs_geoheritage_sites.csv"
OUT_META = OUT_CSV.with_suffix(".meta.json")
ENDPOINT = "https://iugs-geoheritage.org/wp-json/wp/v2/geoheritage_sites"
UA = "mythic-planet-lab/0.1 (https://github.com/1521183894w-ux/mythic-planet-lab) research script"
CATEGORY_ZH = {
    "history-of-geosciences": "地学史",
    "igneous-and-metamorphic-petrology": "火成岩与变质岩",
    "tectonics": "构造",
    "stratigraphy-and-sedimentology": "地层与沉积",
    "paleontology": "古生物",
    "volcanology": "火山",
    "mineralogy": "矿物与矿床",
    "geomorphology-and-active-geological-processes": "地貌与活动地质过程",
    "impact-structures-and-extraterrestrial-rocks": "撞击构造与地外岩石",
}
BATCH = {"2022": "第一批（2022）", "2024": "第二批（2024）"}


def fetch_all() -> tuple[list[dict], int]:
    rows, page, total = [], 1, None
    while True:
        r = requests.get(ENDPOINT, params={"per_page": 100, "page": page, "_fields": "id,slug,title,link,date,class_list"},
                         headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("X-WP-Total", 0))
        batch = r.json()
        rows += batch
        if page >= int(r.headers.get("X-WP-TotalPages", 1)) or not batch:
            return rows, total
        page += 1


def main() -> int:
    rows, total = fetch_all()
    if len(rows) != total:
        print(f"警告：接口报告 {total} 处，实际取到 {len(rows)} 处")
    rows.sort(key=lambda r: (r["date"], r["id"]))
    out = []
    for no, r in enumerate(rows, 1):
        cats = [c.removeprefix("categories-") for c in r.get("class_list", []) if c.startswith("categories-")]
        year = r["date"][:4]
        out.append({
            "no": no, "wp_id": r["id"], "batch": BATCH.get(year, f"{year} 年新增"),
            "category": " / ".join(CATEGORY_ZH.get(c, c) for c in cats),
            "name_en": " ".join(html.unescape(r["title"]["rendered"]).replace("\xa0", " ").split()),
            "published": r["date"][:10], "url": r["link"],
        })
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    meta = {"source": ENDPOINT, "homepage": "https://iugs-geoheritage.org", "queried": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_sites": len(out), "n_reported_by_api": total, "by_batch": dict(Counter(o["batch"] for o in out)),
            "by_category": dict(Counter(o["category"] for o in out)),
            "note": "只保存名称、分类、发布日期与链接（事实信息）；网站文字与图片版权归 IUGS，未保存。"}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"写出 {OUT_CSV.relative_to(ROOT)}：{len(out)} 处；批次 {meta['by_batch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
