"""区域放大图（review_protocol §2.2）：世界窗口与地球对照窗口并排，同比例尺、同分辨率。

地球对照用与世界相同分辨率的地球格网现场裁剪（不是直接贴图集 PNG），所以窗口边长、像元大小两边完全一致。
图集 PNG（data/earth_reference/regions/）另有 ETOPO 2 km 细节版，可在画廊中对照。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .regional import draw_elevation_panel, laea_window, sample_global  # noqa: E402
from .select_regions import CoastAnalysis, RegionPick, random_coast_center  # noqa: E402
from .style import elevation_cmap_norm, stamp  # noqa: E402


def _window(world, center, size_km):
    px = world.grid.pixel_km_equator
    lat, lon, ext = laea_window(center[0], center[1], size_km, px)
    elev = sample_global(world.elevation, lat, lon, order=0)
    land = sample_global(world.land.astype("float32"), lat, lon, order=0)
    return elev, land, lat, lon, ext, px


def earth_reference(pick: RegionPick, world, earth_regions: dict, earth_ca: CoastAnalysis) -> dict:
    """地球对照窗口：按角色取 config/earth_regions.yaml 中的区域；随机海岸则在地球上同样随机抽一段。"""
    if pick.earth_ref:
        r = earth_regions[pick.earth_ref]
        return {"id": r["id"], "name": r["name"], "center": tuple(r["center"]),
                "note": f"config/earth_regions.yaml：{r['id']}（评审对照：{'；'.join(r.get('review_role', []))}）"}
    center, how = random_coast_center(earth_ca, world.world_id, "r8-earth")
    return {"id": "random", "name": "地球上随机抽取的一段海岸", "center": center, "note": how}


def region_pair(pick: RegionPick, world, earth_world, ref: dict, path, index: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    fig.subplots_adjust(left=0.01, right=0.9, top=0.92, bottom=0.17, wspace=0.04)
    size = pick.size_km
    if pick.missing:
        axes[0].text(0.5, 0.5, f"未找到\n\n{pick.missing}", ha="center", va="center", fontsize=14, color="#c0392b",
                     transform=axes[0].transAxes)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"{world.display_name}：本规则没有可选的区域", fontsize=9)
    else:
        e, l_, la, lo, ext, px = _window(world, pick.center, size)
        draw_elevation_panel(axes[0], e, l_, la, lo, ext, px,
                             f"{world.display_name}（{world.resolution}，约 {px:.1f} km 像元）· 中心 "
                             f"{pick.center[0]:.1f}°, {pick.center[1]:.1f}°", pick.center)
    e, l_, la, lo, ext, px = _window(earth_world, ref["center"], size)
    draw_elevation_panel(axes[1], e, l_, la, lo, ext, px,
                         f"地球对照：{ref['name']}（同为 {earth_world.resolution}）· 中心 {ref['center'][0]:.1f}°, "
                         f"{ref['center'][1]:.1f}°", ref["center"])
    cmap, norm = elevation_cmap_norm()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes((0.915, 0.27, 0.012, 0.55)))
    cb.set_label("高程 / 水深（m）；海平面处色带断开")
    fig.suptitle(f"区域 {index} · {pick.title}  ——  {size:g} km 见方，两边同比例尺、同分辨率（局部等面积方位投影）",
                 fontsize=12)
    extra = [f"选取规则：{pick.rule}。", f"选取结果：{pick.missing or pick.reason}",
             f"地球对照：{ref['note']}"]
    stamp(fig, world, f"{world.method_class}（世界高程）；地球为 interpolated（ETOPO 2022 重采样）", extra)
    fig.savefig(path, dpi=110)
    plt.close(fig)
