"""U34：主内海西口与 20°E 一线的逐格海陆图（评审 stage1_2026-10-25_4 第 2、3 条）。

    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\design_strait_pixels.py

读标准世界文件（scripts/design_forward.py 导出的 0 Ma 世界，带 tect_* 构造图层），检查 A 层的两项连通是否依赖对角接触：
- 主内海经西口通外洋：按 4 连通（只认上下左右相邻）是否仍连通；去掉西口的年轻洋壳（< 10 Myr）后是否仍连通；
- 洲 C 与洲 D、洲 B 是否按 4 连通相连；20°E 一线的水是连通的水道还是一串封闭的小水体。
A 层本身用 8 连通（packages/metrics），这里只做核对，不改指标定义。
输出 figures/design/v2/forward/13_strait_pixels.png（forward 图组的附图）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
from mythic.figures.style import setup_fonts  # noqa: E402

WORLD = ROOT / "data/worlds/design_v2/design_v2_forward_v2_0Ma_1024x512.nc"
OUT = ROOT / "figures/design/v2/forward/13_strait_pixels.png"


def label_wrap(mask: np.ndarray, eight: bool) -> np.ndarray:
    """连通标记，经度环绕；eight=False 为 4 连通。"""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int) if eight else None)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ny = mask.shape[0]
    for i in range(ny):
        for di in ((-1, 0, 1) if eight else (0,)):
            j = i + di
            if 0 <= j < ny and lab[i, 0] and lab[j, -1]:
                ra, rb = find(lab[i, 0]), find(lab[j, -1])
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(k) for k in range(n + 1)])
    return roots[lab]


def main() -> int:
    setup_fonts()
    ds = xr.open_dataset(WORLD)
    land = ds["land"].values >= 0.5
    lat, lon = ds["lat"].values, ds["lon"].values
    age = ds["tect_ocean_age_myr"].values
    ctype = ds["tect_crust_type"].values
    LO, LA = np.meshgrid(lon, lat)

    def cell(lo, la):
        return int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))

    sea, ocean = cell(19.0, 37.0), cell(-40.0, 0.0)
    water = ~land
    west = (LO >= -30) & (LO <= -5) & (LA >= 33) & (LA <= 40)
    young = west & water & np.isfinite(age) & (age < 10)
    res = {}
    for name, w in (("8 连通", label_wrap(water, True)), ("4 连通", label_wrap(water, False)),
                    ("4 连通、西口年轻洋壳当陆地", label_wrap(water & ~young, False))):
        res[name] = bool(w[sea] == w[ocean] and w[sea] != 0)
    blocked = water.copy()
    blocked[west] = False
    res["8 连通、挡住西口（别处还有没有出口）"] = bool((lambda w: w[sea] == w[ocean] and w[sea] != 0)(label_wrap(blocked, True)))
    L4 = label_wrap(land, False)
    cont = {k: int(L4[cell(*v)]) for k, v in {"洲 C": (-5.0, 10.0), "洲 D": (40.0, 0.0), "洲 B": (90.0, 55.0)}.items()}
    for k, v in res.items():
        print(f"主内海 -> 外洋（{k}）：{'连通' if v else '不连通'}")
    print(f"陆地 4 连通的分量编号：{cont}")

    W4 = label_wrap(water, False)
    kind = np.zeros(land.shape, int)                 # 0 陆地
    kind[water] = 4                                   # 封闭小水体（既不连外洋也不连主内海）
    kind[water & (W4 == W4[ocean])] = 1               # 与外洋 4 连通的水（含主内海，它们经西口相连）
    kind[(kind == 1) & (ctype == 1)] = 2               # 其中的水下陆壳
    kind[young] = 3                                   # 西口的年轻洋壳（< 10 Myr）
    cmap = ListedColormap(["#d9c9a3", "#2f6db5", "#8fc1e8", "#d62728", "#7a3e9d"])
    names = ["陆地", "与外洋相连的水（洋壳）", "与外洋相连的水（水下陆壳）", "西口的年轻洋壳（< 10 Myr）", "封闭小水体（不通外洋）"]

    fig = plt.figure(figsize=(16, 10))
    for k, (lo0, lo1, la0, la1, title) in enumerate((
            (-26, -4, 32, 41, "主内海西口（逐格，1024x512 约 32 × 39 km）"),
            (14, 26, -36, 36, "20°E 一线（洲 C、D 之间）"))):
        J = np.flatnonzero((lon >= lo0) & (lon <= lo1))
        I = np.flatnonzero((lat >= la0) & (lat <= la1))
        dl = 180.0 / len(lat) / 2
        ext = (lon[J[0]] - dl, lon[J[-1]] + dl, lat[I[-1]] - dl, lat[I[0]] + dl)
        ax = fig.add_axes((0.04, 0.17, 0.6, 0.74) if k == 0 else (0.69, 0.06, 0.28, 0.86))
        ax.imshow(kind[np.ix_(I, J)], extent=ext, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest",
                  aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="lower left",
               bbox_to_anchor=(0.04, 0.02), ncol=3, fontsize=9)
    txt = "；".join(f"{k}：{'连通' if v else '不连通'}" for k, v in res.items())
    fig.text(0.04, 0.115, "主内海 -> 外洋 " + txt, fontsize=9)
    fig.text(0.04, 0.095, f"洲 C、D、B 的陆地按 4 连通是否同一片：{'是' if len(set(cont.values())) == 1 else '否'}（A 层的连通与海峡判定都用 8 连通，"
                          "这里只核对，不改指标定义）", fontsize=9)
    fig.suptitle("U34 西口与 20°E 的逐格海陆（评审 stage1_2026-10-25_4 第 2、3 条）", fontsize=13)
    fig.savefig(OUT, dpi=110)
    print(f"写 {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
