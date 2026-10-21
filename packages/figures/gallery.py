"""标准图组的静态画廊页 index.html（review_protocol §2.1）。只用相对路径，整个目录可直接在浏览器中打开。"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path


CSS = """
:root { --bg:#f6f5f2; --card:#ffffff; --ink:#1f2328; --muted:#5b6370; --line:#d9d6cf; --good:#1e7b4a; --bad:#b03a2e; --accent:#2c5d8a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 "Microsoft YaHei","PingFang SC",system-ui,sans-serif; }
header { padding:24px 32px 12px; border-bottom:1px solid var(--line); background:#fff; }
h1 { margin:0 0 6px; font-size:24px; }
h2 { margin:32px 0 12px; font-size:19px; border-left:4px solid var(--accent); padding-left:10px; }
.meta { color:var(--muted); font-size:13px; }
.note { background:#fff8e6; border:1px solid #f0d9a8; padding:10px 14px; border-radius:6px; font-size:13px; margin-top:10px; }
nav { position:sticky; top:0; background:#fffdf8ee; border-bottom:1px solid var(--line); padding:8px 32px; z-index:2; font-size:14px; }
nav a { margin-right:18px; color:var(--accent); text-decoration:none; }
main { padding:0 32px 48px; max-width:1680px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(560px,1fr)); gap:18px; }
.grid.wide { grid-template-columns:1fr; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.card img { width:100%; display:block; border-bottom:1px solid var(--line); }
.card .cap { padding:10px 14px; font-size:13.5px; }
.card .cap b { font-size:14.5px; }
.card .cap .why { color:var(--muted); margin-top:4px; }
table { border-collapse:collapse; background:#fff; font-size:13.5px; margin:6px 0 4px; }
th, td { border:1px solid var(--line); padding:5px 10px; text-align:left; vertical-align:top; }
th { background:#efede8; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.ok { color:var(--good); font-weight:600; white-space:nowrap; }
.no { color:var(--bad); font-weight:600; white-space:nowrap; }
.scroll { overflow-x:auto; max-width:100%; }
td.num { white-space:nowrap; }
.thumbs { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }
.thumbs a { text-decoration:none; color:var(--ink); font-size:12.5px; }
.thumbs img { width:100%; border:1px solid var(--line); border-radius:4px; }
footer { color:var(--muted); font-size:12px; padding:16px 32px 32px; }
@media (max-width:700px) { main, header, nav { padding-left:16px; padding-right:16px; } .grid { grid-template-columns:1fr; } }
"""


def _e(s) -> str:
    return html.escape(str(s))


def _flag(ok) -> str:
    return '<span class="ok">达标</span>' if ok else '<span class="no">未达标</span>'


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return _e(v)


CHECK_NAMES = {"area_in_range": "面积", "ratio_to_second_ok": "唯一性", "straits_ok": "海峡数",
               "bordering_subcontinents_ok": "接壤亚大陆数", "elongation_ok": "形状（近圆）"}


def a_table(metrics) -> str:
    A = metrics["A"]
    rows = []
    rows.append(("海洋面积占比", _fmt(A["ocean_fraction"]["value"]), f"{A['ocean_fraction']['target']}", A["ocean_fraction"]["pass"]))
    mc = A["main_continent"]
    rows.append(("主大陆占陆地面积", _fmt(mc["share_of_land"]), f"{mc['target']}", mc["pass"]))
    sc = A["subcontinents"]
    rows.append(("亚大陆数", str(sc["count"]), f"{sc['target']}", sc["pass"]))
    mis = A["main_inland_sea"]
    if mis.get("exists"):
        m = mis["main"]
        v = f"{m['area_km2'] / 1e6:.2f} 百万 km²；通外洋海峡 {m['straits_to_world_ocean']} 条；为第二大内部海域的 " \
            f"{(mis['ratio_to_second'] or float('inf')):.1f} 倍"
        geo = mis.get("geometry")
        if geo:
            v += (f"；接壤亚大陆 {geo['n_bordering_subcontinents']} 个（沿岸占比 "
                  + "、".join(f"{s['border_share']:.0%}" for s in geo["subcontinent_border_shares"] if s["border_share"] >= 0.01)
                  + f"）；主轴长宽比 {geo['aspect_ratio']:.2f}")
        failed = [k for k, ok in mis.get("checks", {}).items() if not ok]
        if failed:
            v += "；未过：" + "、".join(CHECK_NAMES.get(k, k) for k in failed)
    else:
        v = "无"
    rows.append(("主内海", v, "1–5 百万 km²；1–3 条海峡；≥ 第二大的 2 倍；至少接壤 2 个亚大陆（各占沿岸 ≥ 15%）；主轴长宽比 ≥ 2",
                 mis.get("pass", False)))
    lat = A["main_continent_latitude"]
    rows.append(("主大陆南北端纬度", f"{lat['north_deg']:.1f}° / {lat['south_deg']:.1f}°",
                 f"北端 ≥ {lat['target']['north_min']}°，南端 ≤ {lat['target']['south_max']}°", lat["pass"]))
    pc = A["polar_continent"]
    rows.append(("极地大陆", "；".join(f"{p['area_km2'] / 1e6:.1f} 百万 km²，质心 {p['centroid_lat_deg']:.0f}°" for p in pc["found"]) or "无",
                 "质心 |纬度| ≥ 70°，面积 5–15 百万 km²", pc["pass"]))
    oo = A["open_ocean"]
    rows.append(("远洋无陆圆直径", f"{oo['land_free_circle_diameter_km']:.0f} km", f"≥ {oo['target_min_diameter_km']} km", oo["pass"]))
    body = "".join(f"<tr><td>{_e(a)}</td><td>{b}</td><td>{_e(c)}</td><td>{_flag(d)}</td></tr>" for a, b, c, d in rows)
    return f"<table><tr><th>A 层硬约束</th><th>值</th><th>目标</th><th>结果</th></tr>{body}</table>"


METRIC_NAMES = {"fractal_dimension": "分形维数 D", "coastline_development": "海岸发育系数", "peninsula_index": "半岛指数",
                "bay_index": "海湾指数", "land_fraction_beyond_1000km": "离海 > 1000 km 的陆地比例",
                "distance_to_sea_median_km": "距海距离中位数（km）"}


def metric_label(k: str) -> str:
    base, _, r = k.partition("@")
    return METRIC_NAMES[base] + (f"（半径 {r} km）" if r else "")


def unit_label(u: str, name: str | None = None) -> str:
    if name and name != u:
        return name
    return "主大陆" if u == "main_continent" else f"陆块 {u.removeprefix('unit_')}"


def b_table(verdict: dict, reference: dict) -> str:
    """B1/B2 判定表（U11：与地球大陆级陆块本体同类比较）。"""
    units = verdict["units"]
    cols = list(units)
    head = ("<tr><th>指标</th><th>地球合格线<br><span class='meta'>大陆级陆块本体中最平滑者</span></th>"
            "<th>主大陆目标<br><span class='meta'>地球亚非欧本体</span></th>"
            + "".join(f"<th>{_e(unit_label(u, units[u].get('name')))}<br><span class='meta'>"
                      f"{units[u]['area_km2'] / 1e6:.1f} 百万 km²</span></th>" for u in cols) + "</tr>")
    target = reference.get("target_main_continent") or {}
    rows = []
    for k, line in reference["pass_line"].items():
        nd = 0 if k.startswith("distance_to_sea") else 3
        sign = "≤" if line["direction"] == "max" else "≥"
        line_name = reference["units"][line["unit"]]["name"]
        cells = []
        for u in cols:
            r = units[u]["metrics"][k]
            tag = ('<span class="meta">参考</span>' if r["pass"] is None else
                   ('<span class="ok">达标</span>' if r["pass"] else '<span class="no">未达标</span>'))
            cells.append(f'<td class="num">{_fmt(r["value"], nd)} {tag}</td>')
        ref_only = units[cols[0]]["metrics"][k]["pass"] is None if cols else False
        rows.append(f"<tr><td>{_e(metric_label(k))}{'（只作参考）' if ref_only else ''}</td>"
                    f"<td class='num'>{sign} {_fmt(line['value'], nd)}<br><span class='meta'>{_e(line_name)}</span></td>"
                    f"<td class='num'>{_fmt(target.get(k), nd)}</td>{''.join(cells)}</tr>")
    summary = "；".join(f"{unit_label(u, units[u].get('name'))} {units[u]['n_pass']}/{units[u]['n_judged']} 项达标"
                       for u in cols)
    note = ("<p class='meta'>判定方式（U11，2026-10-21 用户同意）：地球参照与世界用同一种切分（陆块单元，连通本体），"
            "世界每个大陆级陆块的各项不低于地球大陆级陆块本体中的最小值（距海距离类不高于最大值）。"
            "海岸发育系数随陆块面积增大，只作参考。地球亚非欧本体是主大陆的评分目标，不是硬门槛。"
            f"本世界：{_e(summary)}。</p>")
    return f"<div class='scroll'><table>{head}{''.join(rows)}</table></div>{note}"


def _spread(m, key):
    v = m["B1"].get("coast_complexity_spread", {}).get(key)
    return float("nan") if v is None else v


def c_table(metrics, earth_metrics) -> str:
    def row(name, f, nd=2):
        return f"<tr><td>{_e(name)}</td><td class='num'>{_fmt(f(metrics), nd)}</td><td class='num'>{_fmt(f(earth_metrics), nd)}</td></tr>"
    rows = [
        row("陆地平均高程（m）", lambda m: m["C"]["hypsometry"]["mean_land_m"], 0),
        row("海洋平均深度（m）", lambda m: m["C"]["hypsometry"]["mean_ocean_m"], 0),
        row("陆地高程中位数（m）", lambda m: m["C"]["hypsometry"]["quantiles_land_m"]["0.5"], 0),
        row("陆地坡度中位数（度）", lambda m: m["C"]["slope"]["quantiles_land_deg"]["0.5"]),
        row("陆地坡度 90% 分位（度）", lambda m: m["C"]["slope"]["quantiles_land_deg"]["0.9"]),
        row("山系数", lambda m: m["C"]["mountains"]["n_systems"], 0),
        row("长度 > 1000 km 的山系数", lambda m: m["C"]["mountains"]["n_longer_than_min_length"], 0),
        row("山地占陆地比例", lambda m: m["C"]["mountains"]["mountain_land_fraction"], 3),
        row("最长山系（km）", lambda m: m["C"]["mountains"]["longest_km"], 0),
        row("面积加权主轴长宽比（成带程度）", lambda m: m["C"]["mountains"]["area_weighted_aspect_ratio"]),
        row("成带山系（主轴长宽比 ≥ 3）的面积占比", lambda m: m["C"]["mountains"]["elongated_area_fraction"]),
        row("低矮山地一级：长 > 1000 km 的山系数", lambda m: m["C"]["mountains"]["low_tier"]["n_longer_than_min_length"], 0),
        row("海岸复杂度 p10（只作统计，U18）", lambda m: _spread(m, "p10")),
        row("海岸复杂度 p50", lambda m: _spread(m, "p50")),
        row("海岸复杂度 p90", lambda m: _spread(m, "p90")),
        row("海岸复杂度变异系数（越小 = 各处越一样）", lambda m: _spread(m, "cv")),
        row("较平直海岸窗口占比（复杂度 < 1.5）", lambda m: _spread(m, "straight_fraction")),
        row("岛屿数", lambda m: m["B1"]["islands"]["n_islands"], 0),
        row("Korčak 指数 b", lambda m: m["B1"]["islands"]["korcak"]["b"]),
        row("大陆架占海洋比例", lambda m: m["B1"]["shelf_fraction_of_ocean"], 3),
    ]
    return ("<table><tr><th>C 层与其他</th><th>世界</th><th>地球（同分辨率）</th></tr>" + "".join(rows) + "</table>")


def _card(fig: dict, out_dir: Path) -> str:
    why = f"<div class='why'>{_e(fig['why'])}</div>" if fig.get("why") else ""
    return (f"<div class='card'><a href='{_e(fig['file'])}' target='_blank'><img loading='lazy' src='{_e(fig['file'])}' "
            f"alt='{_e(fig['title'])}'></a><div class='cap'><b>{_e(fig['title'])}</b>{why}</div></div>")


def write_gallery(out_dir: Path, world, stage: int, metrics: dict, earth_metrics: dict, verdict: dict, reference: dict,
                  figures: list[dict],
                  regions: list[dict], atlas_dir: Path) -> Path:
    sections = []
    groups = [("global", "全球与极地", "grid"), ("maps", "指标分布图", "grid"), ("charts", "与地球对比", "grid wide"),
              ("regions", "区域放大图（世界 | 地球对照）", "grid")]
    for key, title, cls in groups:
        cards = "".join(_card(f, out_dir) for f in figures if f["group"] == key)
        sections.append(f"<section id='{key}'><h2>{_e(title)}</h2><div class='{cls}'>{cards}</div></section>")

    warn = metrics.get("resolution_warnings") or []
    warn_html = ("<ul class='meta'>" + "".join(f"<li>{_e(w)}</li>" for w in warn) + "</ul>") if warn else "<p class='meta'>无</p>"

    atlas_html = ""
    idx = atlas_dir / "index.json"
    if idx.exists():
        try:
            rel = Path(os.path.relpath(atlas_dir, out_dir)).as_posix()
        except ValueError:  # 输出目录与图集不在同一个盘（Windows），只能用绝对地址
            rel = atlas_dir.resolve().as_uri()
        items = json.loads(idx.read_text(encoding="utf-8"))["regions"]
        thumbs = "".join(f"<a href='{rel}/{_e(r['file'])}' target='_blank'><img loading='lazy' src='{rel}/{_e(r['file'])}' "
                         f"alt='{_e(r['name'])}'>{_e(r['name'])}</a>" for r in items)
        atlas_html = (f"<section id='atlas'><h2>地球区域参照图集（{len(items)} 区）</h2>"
                      "<p class='meta'>每张左为与世界同分辨率（2048×1024），右为 ETOPO 2 km 真实细节。"
                      "由 scripts/build_region_atlas.py 生成；图片不进 Git，新机器上需先运行该脚本。</p>"
                      f"<div class='thumbs'>{thumbs}</div></section>")

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>标准图组 · {_e(world.display_name)}</title><style>{CSS}</style></head>
<body>
<header>
<h1>标准图组 · 阶段 {stage} · {_e(world.display_name)}（{_e(world.world_id)}）</h1>
<div class="meta">格网 {_e(world.resolution)}（赤道像元约 {world.grid.pixel_km_equator:.1f} km） · 种子 {_e(world.seed)} ·
代码 {_e(world.code_version)} · 高程 method_class：{_e(world.method_label)} · 生成 {time.strftime('%Y-%m-%d %H:%M')}</div>
<div class="meta">数据：{_e(world.source)}</div>
<div class="note">本页汇总标准图组与指标，供评审与用户看图。指标达标不等于阶段完成：完成须同时满足指标、world-reviewer 评审无未处理的严重问题、
用户在 PROGRESS.md 写下“通过”（CLAUDE.md 第 2 节）。看图时请重点看区域放大图与旁边的地球对照。</div>
</header>
<nav><a href="#summary">指标</a><a href="#global">全球</a><a href="#maps">指标分布</a><a href="#charts">对比</a>
<a href="#regions">区域放大</a>{'<a href="#atlas">地球图集</a>' if atlas_html else ''}</nav>
<main>
<section id="summary"><h2>指标摘要</h2>
{a_table(metrics)}
<h3>B1 海岸复杂度、B2 距海距离：与地球大陆级陆块本体同类比较</h3>
{b_table(verdict, reference)}
<h3>C 层与其他</h3>
{c_table(metrics, earth_metrics)}
<h3>分辨率提示</h3>{warn_html}
<p class="meta">完整数值：<a href="metrics.json">metrics.json</a>；区域选取：<a href="regions.json">regions.json</a>；
图件清单：<a href="manifest.json">manifest.json</a></p>
</section>
{''.join(sections)}
{atlas_html}
</main>
<footer>由 scripts/make_figures.py 生成。指标定义见 docs/metrics_definitions.md；出图规范见 docs/review_protocol.md §2。</footer>
</body></html>
"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8", newline="\n")
    return path
