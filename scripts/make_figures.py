"""标准图组管线（docs/review_protocol.md §2）：对一个世界生成标准图组、指标 JSON 与 index.html 画廊。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_figures.py                 # 地球 2048x1024
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_figures.py --world earth:1024x512
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\make_figures.py --world data/worlds/xxx.nc

输出：figures/stage<N>/<world_id>/（不进 Git，可由"世界数据 + 代码版本"重建）
  01–08 全球图、极地图与指标分布图；09–12 与同分辨率地球的对比图；regions/r1–r8 区域放大图（世界 | 地球对照）；
  metrics.json（全部指标）、regions.json（8 个区域的选取依据）、manifest.json（图件清单）、index.html（画廊）。

世界文件格式见 packages/figures/world.py。对比用的地球格网必须与世界同分辨率（512x256 / 1024x512 / 2048x1024）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from mythic.figures import charts, maps
from mythic.figures.gallery import write_gallery
from mythic.figures.region_panels import earth_reference, region_pair
from mythic.figures.select_regions import CoastAnalysis, select_regions
from mythic.figures.style import setup_fonts
from mythic.figures.world import EARTH_REF, ROOT, blinded, load, load_earth, mirrored, to_jsonable
from mythic.metrics.compute import compute_all, load_config
from mythic.metrics.verdict import b_verdict, b_verdict_flat

EARTH_RESOLUTIONS = ["512x256", "1024x512", "2048x1024"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", default="earth", help='"earth"、"earth:<分辨率>" 或世界 netCDF 文件路径')
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", default=None, help="输出目录（默认 figures/stage<N>/<world_id>）")
    ap.add_argument("--blind", default=None, metavar="名称",
                    help="盲评版（阶段 0.7 尺子校准）：图中世界只叫这个名称，隐去来源、种子与 method_class；须同时给 --out")
    ap.add_argument("--mirror-shift-deg", type=float, default=None,
                    help="（仅盲评用）先东西镜像再向东平移若干度：保距变换，不改变任何指标，只降低一眼认出地图的可能")
    args = ap.parse_args()
    t_all = time.perf_counter()
    setup_fonts()
    cfg = load_config()

    world = load(args.world)
    if args.mirror_shift_deg is not None:
        world = mirrored(world, args.mirror_shift_deg)
    if args.blind:
        if not args.out:
            raise SystemExit("--blind 必须同时给 --out")
        log(f"盲评版：{world.world_id}（method_class {world.method_class}）-> 显示为 {args.blind}")
        world = blinded(world, args.blind, Path(args.out).name)
    out = Path(args.out) if args.out else ROOT / "figures" / f"stage{args.stage}" / world.world_id
    (out / "regions").mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list((out / "regions").glob("*.png")):
        old.unlink()  # 只删本管线生成的图，避免旧图混进新图组
    log(f"世界 {world.world_id}（{world.resolution}，method_class {world.method_label}，代码 {world.code_version}）-> {out}")

    t0 = time.perf_counter()
    metrics, fields = compute_all(world.elevation, world.land, cfg=cfg, return_fields=True, **world.compute_kwargs())
    log(f"世界指标 {time.perf_counter() - t0:.1f} s")
    if world.is_earth:
        earth, em, ef = world, metrics, fields
    else:
        if world.resolution not in EARTH_RESOLUTIONS:
            raise SystemExit(f"世界分辨率 {world.resolution} 没有对应的地球基准格网（可用：{EARTH_RESOLUTIONS}）")
        t0 = time.perf_counter()
        earth = load_earth(world.resolution)
        em, ef = compute_all(earth.elevation, earth.land, cfg=cfg, return_fields=True, **earth.compute_kwargs())
        log(f"同分辨率地球指标 {time.perf_counter() - t0:.1f} s")

    # B1/B2 判定（U11）：与同分辨率地球大陆级陆块本体同类比较，参照取自基准文件
    base = json.loads((EARTH_REF / f"baselines_{world.resolution}.json").read_text(encoding="utf-8"))
    reference = base.get("landmass_unit_reference")
    if reference is None:
        raise SystemExit("地球基准文件缺少 landmass_unit_reference，请先运行 scripts/compute_earth_baselines.py")
    verdict = (b_verdict_flat(reference["units"], reference, cfg) if world.is_earth
               else b_verdict(metrics["by_region"], reference, cfg))
    log("B1/B2 判定：" + "；".join(f"{u} {v['n_pass']}/{v['n_judged']}" for u, v in verdict["units"].items()))

    er_cfg = yaml.safe_load((ROOT / "config" / "earth_regions.yaml").read_text(encoding="utf-8"))
    earth_regions = {r["id"]: r for r in er_cfg["regions"]}
    sizes = {k: float(r.get("size_km", er_cfg["default_size_km"])) for k, r in earth_regions.items()}
    t0 = time.perf_counter()
    picks, ca = select_regions(world.world_id, world.grid, world.land, metrics, fields, cfg, sizes)
    if world.is_earth:
        earth_ca, earth_picks = ca, {p.key: p for p in picks}
    else:
        ep, earth_ca = select_regions(earth.world_id, earth.grid, earth.land, em, ef, cfg, sizes)
        earth_picks = {p.key: p for p in ep}
    log(f"区域选取 {time.perf_counter() - t0:.1f} s：" + "；".join(
        f"{p.key}={'缺' if p.missing else f'{p.center[0]:.1f},{p.center[1]:.1f}'}" for p in picks))

    figures: list[dict] = []

    def add(group, file, title, why="", method=""):
        figures.append({"group": group, "file": file, "title": title, "why": why,
                        "method_class": method or f"由高程/海陆掩膜派生（输入 {world.method_label}）"})

    t0 = time.perf_counter()
    maps.global_elevation(world, out / "01_elevation_global.png")
    add("global", "01_elevation_global.png", "高程（等距圆柱投影）", "分层设色 + 山体阴影；海平面处色带断开。",
        world.method_label)
    maps.polar_elevation(world, out / "02_elevation_north_polar.png", north=True)
    add("global", "02_elevation_north_polar.png", "北极区高程（等面积方位投影）", "避免等距圆柱投影的高纬拉伸掩盖问题。",
        world.method_label)
    maps.polar_elevation(world, out / "03_elevation_south_polar.png", north=False)
    add("global", "03_elevation_south_polar.png", "南极区高程（等面积方位投影）", "", world.method_label)
    maps.landmass_units(world, metrics, fields, cfg, out / "04_landmass_units.png")
    add("maps", "04_landmass_units.png", "陆块单元、亚大陆与主内海（A 层）", "A 层硬约束的空间结构与达标情况。")
    maps.coast_morphology(world, metrics, fields, out / "05_coast_morphology.png")
    add("maps", "05_coast_morphology.png", "半岛与海湾（B1）", "半径 100 km 开运算削掉的陆地与闭运算填充的海。")
    maps.distance_to_sea(world, metrics, fields, out / "06_distance_to_sea.png")
    add("maps", "06_distance_to_sea.png", "距海距离（B2）")
    maps.slope(world, fields, metrics, out / "07_slope.png")
    add("maps", "07_slope.png", "坡度")
    maps.mountain_systems(world, metrics, fields, out / "08_mountain_systems.png")
    add("maps", "08_mountain_systems.png", "山系（C 层，v1 定义）")
    log(f"全球图与指标图 {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    charts.hypsometry(world, metrics, em, out / "09_hypsometry.png")
    add("charts", "09_hypsometry.png", "高程曲线与高程分布（与地球对比）")
    charts.slope_distance_mountains(world, metrics, fields, em, ef, earth, out / "10_slope_distance_mountains.png")
    add("charts", "10_slope_distance_mountains.png", "坡度、距海距离、山系长度（与地球对比）")
    charts.coast_vs_earth(world, metrics, em, earth, cfg, verdict, reference, out / "11_coast_vs_earth.png")
    add("charts", "11_coast_vs_earth.png", "海岸盒计数、岛屿面积分布、各大陆级陆块的 B1/B2 判定（与地球陆块本体同类比较）")
    charts.coast_complexity_spread(world, ca, metrics, earth, earth_ca, em, out / "12_coast_complexity_spread.png")
    add("charts", "12_coast_complexity_spread.png", "海岸复杂度的空间分布（与地球对比，只作统计）",
        "B1 只量海岸复杂度的多少；这张图看它在空间上怎么分布：哪里碎、哪里直（U18(a)）。")
    log(f"对比图 {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    region_records = []
    for n, p in enumerate(picks, 1):
        ref = earth_reference(p, world, earth_regions, earth_ca, earth_picks)
        f = f"regions/{p.key}.png"
        region_pair(p, world, earth, ref, out / f, n)
        add("regions", f, f"区域 {n} · {p.title}", (p.missing and f"未找到：{p.missing}") or p.reason,
            f"{world.method_label}（世界高程）")
        region_records.append({**p.as_dict(), "earth_comparison": {**ref, "center": list(ref["center"])}})
    log(f"区域放大图 {time.perf_counter() - t0:.1f} s")

    meta = {"world_id": world.world_id, "display_name": world.display_name, "resolution": world.resolution,
            "seed": world.seed, "code_version": world.code_version, "method_class_elevation": world.method_label,
            "source": world.source, "stage": args.stage, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_version": cfg.get("version")}
    (out / "metrics.json").write_text(json.dumps(to_jsonable({**meta, "b_verdict": verdict, "metrics": metrics}), ensure_ascii=False, indent=1),
                                      encoding="utf-8", newline="\n")
    (out / "regions.json").write_text(json.dumps(to_jsonable({**meta, "regions": region_records}), ensure_ascii=False,
                                                 indent=1), encoding="utf-8", newline="\n")
    (out / "manifest.json").write_text(json.dumps({**meta, "figures": figures}, ensure_ascii=False, indent=1),
                                       encoding="utf-8", newline="\n")
    page = write_gallery(out, world, args.stage, metrics, em, verdict, reference, figures, region_records,
                         EARTH_REF / "regions")
    log(f"完成：{len(figures)} 张图，画廊 {page}，总用时 {time.perf_counter() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
