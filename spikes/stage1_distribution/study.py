"""U36-S1：固定局部网格上的地壳分配小试，不替换正式世界。

仅复用 Landlab LinearDiffuser 的数值组件求解扩散类比；
地学方法标签为 procedural_constrained，不能称完整下地壳力学。
坐标 km，厚度 km，时间 Myr，扩散系数 km²/Myr。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.special import erf
from landlab import RasterModelGrid
from landlab.components import LinearDiffuser
from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine

METHOD_VERSION = "u36-s1-v1"
REFERENCES = [
    "https://doi.org/10.1130/0091-7613(2000)28<703:TOBTEM>2.0.CO;2",
    "https://doi.org/10.1029/2007JB005457",
    "https://landlab.csdms.io/generated/api/landlab.components.diffusion.diffusion.html",
]


@dataclass(frozen=True)
class Settings:
    spacing_km: float = 40.0
    width_km: float = 2400.0
    height_km: float = 2000.0
    duration_myr: float = 40.0
    dt_myr: float = 0.25
    diffusivity_km2_myr: float = 12000.0
    viscosity_contrast: float = 1000.0
    cross_strike_factor: float = 0.25
    source_peak_km_myr: float = 1.8


def geometry(settings=Settings()):
    """2400×2000km合成域；初始结构与输入都不读取终图高度。"""
    dx = settings.spacing_km
    nx = int(round(settings.width_km / dx)) + 2
    ny = int(round(settings.height_km / dx)) + 2
    if not np.isclose((nx - 2) * dx, settings.width_km) or not np.isclose((ny - 2) * dx, settings.height_km):
        raise ValueError("网格间距必须整除2400与2000km")
    grid = RasterModelGrid((ny, nx), xy_spacing=dx)
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    x, y = grid.x_of_node - settings.width_km / 2 - dx / 2, grid.y_of_node - settings.height_km / 2 - dx / 2
    core = np.zeros(grid.number_of_nodes, dtype=bool)
    core[grid.core_nodes] = True
    # 结构区为合成的继承强盆地；北向框只作检查窗，定向系数作用于全域。
    basin_shape = ((x - 220.0) / 230.0) ** 2 + (y / 420.0) ** 2
    basin_weight = np.clip((1.15 - basin_shape) / 0.3, 0.0, 1.0)
    basin = (basin_shape < 0.7) & core
    corridor = (np.abs(x + 350) < 250) & (y > 450) & (y < 850) & core
    source = settings.source_peak_km_myr * np.exp(-0.5 * (
        ((x + 420) / 160) ** 2 + (y / 360) ** 2))
    source[(~core) | (np.abs(x) >= 1200) | (np.abs(y) >= 1000)] = 0.0
    # 保持40/20km格网上相同的有效面积与连续高斯积分输入量；闭边界置于单元面。
    fx = 0.5 * (erf((1200 + 420) / (2**0.5 * 160)) - erf((-1200 + 420) / (2**0.5 * 160)))
    fy = erf(1000 / (2**0.5 * 360))
    input_rate = settings.source_peak_km_myr * 2 * np.pi * 160 * 360 * fx * fy
    if input_rate > 0:
        source *= input_rate / (source.sum() * dx**2)
    initial = np.full(grid.number_of_nodes, 35.0)
    # 盆地初始只比参考柱厚0.33km（约450m）；不设置最终目标或高度钳制。
    initial += 0.33 * basin_weight
    return grid, x, y, core, basin_weight, basin, corridor, source, initial


def link_diffusivity(grid, basin_weight, settings, structured=False, directional=False):
    """先在节点定义相对流动性，再用调和均值跨越强弱界面。

    定向试验假定全域构造走向为y方向，x方向系数减小。
    比值为假设，不把地学文献中的黏度直接当已校准的扩散系数。
    """
    mobility = (settings.viscosity_contrast ** (-basin_weight)
                if structured else np.ones(grid.number_of_nodes))
    i, j = grid.nodes_at_link.T
    links = 2 * mobility[i] * mobility[j] / (mobility[i] + mobility[j])
    if directional:
        horizontal = np.abs(grid.x_of_node[j] - grid.x_of_node[i]) > 0
        links[horizontal] *= settings.cross_strike_factor
    return links * settings.diffusivity_km2_myr


def _legacy_adapter(grid, core, initial, spacing, capped):
    """调用原Engine._surface/_collapse；只适配为等面积四邻接局部图。

    不调用Engine.step，不含真实板块碰撞，不能称全球旧世界的复跑。
    """
    eng = Engine.__new__(Engine)
    eng.c = Crust(grid.number_of_nodes)
    eng.c.T[:] = initial
    eng.c.ctype[:] = CONT
    eng.c.plate[:] = 1
    eng.alive = core.copy()
    edges = grid.nodes_at_link
    eng.pedges = edges[core[edges[:, 0]] & core[edges[:, 1]]]
    eng.area = np.full(grid.number_of_nodes, spacing ** 2)
    eng.p = Params(tau_oro_myr=np.inf, kappa_km2_myr=300.0,
                   T_max_km=65.0 if capped else 1e6)
    eng.m = SimpleNamespace(h_km=spacing)
    eng.budget = {"eroded": 0.0, "arc_converted": 0.0}
    eng.t = 40.0
    return eng


def run_case(kind, settings=Settings()):
    """相同输入下的闭边界材料分配。新增体积是外部碰撞供给，不是凭空造壳。"""
    if kind not in {"legacy_cap", "legacy_uncapped", "uniform", "strong_basin", "directional"}:
        raise ValueError(kind)
    if (not np.isfinite(settings.viscosity_contrast) or settings.viscosity_contrast < 1
            or not 0 < settings.cross_strike_factor <= 1
            or not np.isfinite(settings.diffusivity_km2_myr)
            or settings.diffusivity_km2_myr <= 0
            or not np.isfinite(settings.duration_myr) or settings.duration_myr < 0
            or not np.isfinite(settings.dt_myr) or settings.dt_myr <= 0
            or not np.isfinite(settings.spacing_km) or settings.spacing_km <= 0
            or not np.isfinite(settings.source_peak_km_myr) or settings.source_peak_km_myr < 0
            or not np.isfinite(settings.width_km) or settings.width_km <= 0
            or not np.isfinite(settings.height_km) or settings.height_km <= 0):
        raise ValueError("无效的试验参数")
    grid, x, y, core, bw, basin, corridor, source, initial = geometry(settings)
    z = grid.add_field("topographic__elevation", initial.copy(), at="node")
    eng = None
    kd = link_diffusivity(grid, bw, settings, kind in {"strong_basin", "directional"}, kind == "directional")
    if kind.startswith("legacy"):
        eng = _legacy_adapter(grid, core, initial, settings.spacing_km, kind == "legacy_cap")
        z = eng.c.T
    else:
        diffuser = LinearDiffuser(grid, linear_diffusivity=kd, method="simple", deposit=True)
    area = settings.spacing_km ** 2
    v0 = float(z[core].sum() * area)
    time = 0.0
    added = 0.0
    history = []
    snapshots = {}
    peak_ever = float(z[core].max())
    while time < settings.duration_myr - 1e-12:
        dt = min(settings.dt_myr, settings.duration_myr - time)
        z += source * dt
        added += float(source.sum() * dt * area)
        if eng is not None:
            eng._surface(dt)
        else:
            diffuser.run_one_step(dt)
        time += dt
        peak_ever = max(peak_ever, float(z[core].max()))
        height = 400 + (z - 35.0) * 151.5
        volume = float(z[core].sum() * area)
        history.append([time, float(np.median(height[basin])), float(np.median(height[corridor])),
                        float(height[core].max()), volume - v0 - added])
        for stamp in (10, 20, 40):
            if abs(time - stamp) < 1e-9:
                snapshots[str(stamp)] = z.copy()
    h = 400 + (z - 35.0) * 151.5
    dv = (z - initial) * area
    summary = {
        "kind": kind, "active_area_km2": float(core.sum() * area),
        "initial_volume_km3": v0, "external_input_km3": added,
        "donor_reservoir_initial_km3": float(source.sum() * area * settings.duration_myr),
        "donor_reservoir_final_km3": float(source.sum() * area * settings.duration_myr) - added,
        "final_volume_km3": float(z[core].sum() * area),
        "closure_km3": float(z[core].sum() * area) - v0 - added,
        "basin_median_m": float(np.median(h[basin])),
        "corridor_median_m": float(np.median(h[corridor])),
        "maximum_m": float(h[core].max()),
        "maximum_thickness_ever_km": peak_ever,
        "area_at_65km_km2": float(np.count_nonzero(core & np.isclose(z, 65.0, atol=1e-6, rtol=0)) * area),
        "basin_net_input_km3": float(dv[basin].sum()),
        "basin_direct_source_km3": float(source[basin].sum() * area * settings.duration_myr),
        "basin_net_lateral_km3": float(dv[basin].sum() - source[basin].sum() * area * settings.duration_myr),
        "corridor_net_input_km3": float(dv[corridor].sum()),
        "corridor_direct_source_km3": float(source[corridor].sum() * area * settings.duration_myr),
        "corridor_net_lateral_km3": float(dv[corridor].sum() - source[corridor].sum() * area * settings.duration_myr),
        "remaining_domain_net_input_km3": float(dv[core & ~basin & ~corridor].sum()),
    }
    if history:
        summary["max_abs_closure_km3"] = float(np.max(np.abs(np.asarray(history)[:, -1])))
    return {
        "grid": grid, "x": x, "y": y, "core": core, "basin": basin, "corridor": corridor,
        "basin_weight": bw, "source": source, "initial": initial, "thickness": z.copy(),
        "height": h, "diffusivity": kd, "history": np.asarray(history), "snapshots": snapshots,
        "summary": summary,
    }
