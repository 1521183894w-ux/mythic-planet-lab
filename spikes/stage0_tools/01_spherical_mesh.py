"""Spike 01 · 球面三角网格：stripy 与 scipy（ConvexHull / SphericalVoronoi / cKDTree）对比。

要回答的问题（为阶段 1 构造模拟选网格方案）：
1. 在 1 万 ~ 65 万顶点（地球尺度约 220 km ~ 28 km 间距）下，建网耗时和内存是多少？
2. 两种方案的三角剖分是否一致？
3. 网格 -> 经纬度栅格（2048x1024，标准出图分辨率）的映射要多久？
4. 能否在网格邻接关系上做板块分区这类图运算？（出一张演示图）
"""

from __future__ import annotations

import heapq

import numpy as np
from _common import Recorder, available_gb, savefig

R_EARTH_KM = 6371.0
rec = Recorder("01_spherical_mesh")

import stripy  # noqa: E402
from scipy.spatial import ConvexHull, SphericalVoronoi, cKDTree  # noqa: E402


def lonlat_grid(nx: int, ny: int):
    lon = np.linspace(-np.pi, np.pi, nx, endpoint=False) + np.pi / nx
    lat = np.linspace(np.pi / 2, -np.pi / 2, ny, endpoint=False) - np.pi / (2 * ny)
    return np.meshgrid(lon, lat)


def to_xyz(lon, lat):
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def smooth_field(lon, lat):
    """球面上处处光滑的测试函数（用三维坐标定义）。
    注意：不能用 sin(3*lat)*cos(2*lon) 这类函数，它在两极随经度变化、不连续，会让插值误差虚高。"""
    x, y, z = to_xyz(np.ravel(lon), np.ravel(lat)).T
    return np.sin(3 * x) + y * z + 0.5 * np.cos(2 * z)


def adjacency_csr(simplices: np.ndarray, n: int):
    """由三角形列表构造顶点邻接（CSR 格式）。"""
    e = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    order = np.argsort(e[:, 0], kind="stable")
    e = e[order]
    indptr = np.searchsorted(e[:, 0], np.arange(n + 1))
    return indptr, e[:, 1]


summary: dict = {"levels": {}}
GX, GY = lonlat_grid(2048, 1024)
grid_xyz = to_xyz(GX.ravel(), GY.ravel())

meshes = {}
for level in (5, 6, 7, 8):
    n_expected = 10 * 4**level + 2
    spacing_km = np.sqrt(4 * np.pi * R_EARTH_KM**2 / n_expected) * 1.07
    tag = f"L{level} ({n_expected:,} 点, ~{spacing_km:.0f} km)"
    lev: dict = {"n_vertices": n_expected, "spacing_km": round(spacing_km, 1)}

    with rec.measure(f"stripy 建网 {tag}", n=n_expected) as e:
        mesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=level, tree=True)
        e["n_triangles"] = int(mesh.simplices.shape[0])
        lev["stripy_build_s"] = None
    lev["stripy_build_s"] = rec.items[-1]["seconds"]
    meshes[level] = mesh

    with rec.measure(f"邻接表（CSR）{tag}") as e:
        indptr, nbrs = adjacency_csr(mesh.simplices, mesh.npoints)
        deg = np.diff(indptr)
        e["degree_min_max"] = [int(deg.min()), int(deg.max())]

    with rec.measure(f"stripy 最近顶点 -> 2048x1024 栅格 {tag}") as e:
        _, nearest = mesh.nearest_vertices(GX.ravel(), GY.ravel(), k=1)
        e["queries"] = int(GX.size)
    lev["stripy_nearest_grid_s"] = rec.items[-1]["seconds"]

    if level <= 7:
        sub_x, sub_y = lonlat_grid(1024, 512)
        field = smooth_field(mesh.lons, mesh.lats)
        with rec.measure(f"stripy 线性插值 -> 1024x512 {tag}") as e:
            zi, err = mesh.interpolate(sub_x.ravel(), sub_y.ravel(), field, order=1)
            abs_err = np.abs(zi - smooth_field(sub_x, sub_y))
            e["max_abs_err"] = float(f"{abs_err.max():.2e}")
            e["mean_abs_err"] = float(f"{abs_err.mean():.2e}")
        with rec.measure(f"stripy 梯度（球面）{tag}") as e:
            dlon, dlat = mesh.gradient_lonlat(field)
            e["finite"] = bool(np.isfinite(dlon).all() and np.isfinite(dlat).all())

    xyz = np.column_stack([mesh.x, mesh.y, mesh.z])
    with rec.measure(f"scipy ConvexHull（球面 Delaunay）{tag}") as e:
        hull = ConvexHull(xyz)
        e["n_triangles"] = int(hull.simplices.shape[0])
        if level <= 6:
            a = np.sort(np.sort(mesh.simplices, axis=1), axis=0)
            b = np.sort(np.sort(hull.simplices, axis=1), axis=0)
            same = {tuple(t) for t in np.sort(mesh.simplices, axis=1)} == {tuple(t) for t in np.sort(hull.simplices, axis=1)}
            e["same_triangles_as_stripy"] = bool(same)
    lev["scipy_hull_s"] = rec.items[-1]["seconds"]

    if available_gb() > 3.0 or level <= 7:
        with rec.measure(f"scipy SphericalVoronoi + 面积 {tag}") as e:
            sv = SphericalVoronoi(xyz, radius=1.0)
            sv.sort_vertices_of_regions()
            areas = sv.calculate_areas()
            e["area_sum_over_4pi"] = round(float(areas.sum() / (4 * np.pi)), 6)
            e["area_max_over_min"] = round(float(areas.max() / areas.min()), 3)
        lev["scipy_voronoi_s"] = rec.items[-1]["seconds"]
    else:
        rec.skip(f"scipy SphericalVoronoi {tag}", f"可用内存 {available_gb():.1f} GB < 3 GB")

    with rec.measure(f"scipy cKDTree 建树+查询 2048x1024 {tag}") as e:
        tree = cKDTree(xyz)
        _, nn = tree.query(grid_xyz, k=1, workers=-1)
        e["agree_with_stripy"] = round(float(np.mean(nn == nearest.ravel())), 6)
    lev["scipy_kdtree_s"] = rec.items[-1]["seconds"]
    summary["levels"][level] = lev
    if level < 8:
        del hull
    print(f"  可用内存 {available_gb():.1f} GB")

# ---- 演示：在 L6 网格上随机"长出"板块（加权 Dijkstra 区域生长）--------------------
level = 6
mesh = meshes[level]
indptr, nbrs = adjacency_csr(mesh.simplices, mesh.npoints)
rng = np.random.default_rng(20261021)
n_plates = 14
with rec.measure("演示：L6 网格上 14 块板块区域生长") as e:
    seeds = rng.choice(mesh.npoints, n_plates, replace=False)
    plate = np.full(mesh.npoints, -1, dtype=int)
    cost = np.full(mesh.npoints, np.inf)
    # 每块板块的"生长速度"略有不同，得到大小不一的板块。
    # 首次运行用 sigma=0.6，结果一块板块占了 80% 球面（地球最大的太平洋板块约占 20%），改为 0.25。
    growth = rng.lognormal(0.0, 0.25, n_plates)
    noise = rng.lognormal(0.0, 0.6, mesh.npoints)  # 每个顶点的通行代价，制造不规则边界
    heap = []
    for p, s in enumerate(seeds):
        cost[s] = 0.0
        heapq.heappush(heap, (0.0, int(s), p))
    while heap:
        c, v, p = heapq.heappop(heap)
        if plate[v] != -1:
            continue
        plate[v] = p
        for w in nbrs[indptr[v]:indptr[v + 1]]:
            if plate[w] == -1:
                nc = c + noise[w] / growth[p]
                if nc < cost[w]:
                    cost[w] = nc
                    heapq.heappush(heap, (nc, int(w), p))
    sizes = np.bincount(plate, minlength=n_plates) / mesh.npoints
    e["plate_area_fractions_sorted"] = [round(float(s), 3) for s in np.sort(sizes)[::-1]]

import cartopy.crs as ccrs  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_, nearest6 = mesh.nearest_vertices(GX.ravel(), GY.ravel(), k=1)
plate_grid = plate[nearest6.ravel()].reshape(GX.shape)

coarse = meshes[5]
fig = plt.figure(figsize=(13, 5.2))
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(30, 25))
ax1.set_global()
lon_d, lat_d = np.degrees(mesh.lons), np.degrees(mesh.lats)
ax1.scatter(lon_d, lat_d, c=plate, s=1.2, cmap="tab20", transform=ccrs.PlateCarree(), vmin=0, vmax=19)
small = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=3)
for tri in small.simplices:
    ring = np.append(tri, tri[0])
    ax1.plot(np.degrees(small.lons[ring]), np.degrees(small.lats[ring]), color="k", lw=0.25,
             transform=ccrs.Geodetic())
ax1.set_title(f"L6 icosahedral mesh ({mesh.npoints:,} vertices), plates; black: L3 mesh edges", fontsize=9)
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())
ax2.set_global()
ax2.imshow(plate_grid, extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap="tab20",
           vmin=0, vmax=19, interpolation="nearest")
ax2.set_title("Same plates rasterized to 2048x1024 via nearest vertex (Robinson)", fontsize=9)
fig.suptitle("Spike 01 - spherical mesh + plate partition demo (region growing on mesh adjacency)", fontsize=10)
savefig(fig, "01_spherical_mesh_plates.png")
plt.close(fig)

rec.save(**summary)
