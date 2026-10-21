"""Spike 05 · 地貌演化：Fastscape（fastscapelib-fortran 内核）与 Landlab。

要回答的问题（阶段 2 区域侵蚀与沉积选型）：
1. 两者在相同参数（抬升 + 河流侵蚀 SPL + 坡面扩散）、相同初始地形下，结果是否一致？
2. 每步耗时与网格规模的关系；台式机上 1024x1024 能否承受？
3. 输出地形能否直接供下游水文工具（spike 06）使用？

重要发现（2026-10-21 实测）：Python 封装层 fastscape 0.1.0 依赖的 xarray-simlab 0.5.0 与当前环境不兼容
（调用了 zarr 3 中已删除的 zarr.MemoryStore，以及 numpy 2.x 中已删除的 np.in1d），xs.run() 直接报错。
因此这里绕过 xarray-simlab，按 fastscape basic_model 的过程顺序直接驱动 fastscapelib-fortran 内核
（单流向汇流 -> 河流侵蚀 SPL 与线性扩散在同一地表上计算侵蚀量 -> 一并扣除；固定边界不抬升）。

统一参数：dx = 500 m，抬升 U = 1 mm/yr，K = 1e-5，m = 0.4，n = 1，D = 0.01 m²/yr，
dt = 10 kyr，200 步 = 2 Myr，四周边界固定高程（开放出口）。
"""

from __future__ import annotations

import numpy as np
from _common import WORK_DIR, Recorder, available_gb, savefig

rec = Recorder("05_landscape_evolution")
import fastscape  # noqa: E402
import fastscapelib_fortran as fsf  # noqa: E402
import landlab  # noqa: E402
import xsimlab as xs  # noqa: E402
from fastscape.models import basic_model  # noqa: E402
from fastscape.processes.flow import _flow_accumulate_sd  # noqa: E402
from landlab import RasterModelGrid  # noqa: E402
from landlab.components import (  # noqa: E402
    FastscapeEroder,
    FlowAccumulator,
    LinearDiffuser,
    PriorityFloodFlowRouter,
)

work = WORK_DIR / "05_lem"
work.mkdir(parents=True, exist_ok=True)

DX = 500.0
U, K, M, N, D = 1e-3, 1e-5, 0.4, 1.0, 1e-2
DT, NSTEPS = 1e4, 200


def initial_surface(n: int, seed: int = 42) -> np.ndarray:
    """与 fastscape 的 RandomInitialSurface 相同：0–1 m 均匀白噪声。"""
    return np.random.RandomState(seed).rand(n, n)


def run_fastscape(z0: np.ndarray, nsteps: int):
    """直接驱动 fastscapelib-fortran 内核，复现 fastscape basic_model 的一步。"""
    ny, nx = z0.shape
    nn = nx * ny
    fsf.fastscape_init()
    fsf.fastscape_set_nx_ny(nx, ny)
    fsf.fastscape_setup()
    fsf.fastscape_set_xl_yl((nx - 1) * DX, (ny - 1) * DX)
    fsf.fastscape_set_bc(1111)  # 四边固定高程
    ctx = fsf.fastscapecontext
    ctx.dt = DT
    ctx.kf = np.full(nn, K)
    ctx.kfsed = -1.0
    ctx.m = M
    ctx.n = N
    ctx.g1 = 0.0
    ctx.g2 = 0.0
    ctx.kd = np.full(nn, D)
    ctx.kdsed = -1.0
    mask = np.ones((ny, nx))
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0.0
    uplift = (U * DT * mask).ravel()
    z = z0.ravel().astype("float64").copy()
    area = np.zeros(nn)
    for _ in range(nsteps):
        z = z + uplift
        ctx.h = z
        fsf.flowroutingsingleflowdirection()
        stack = ctx.stack.astype("int64") - 1  # Fortran 1 起 -> Python 0 起
        receivers = ctx.rec.astype("int64") - 1
        area = np.full(nn, DX * DX)
        _flow_accumulate_sd(area, stack, receivers)
        ctx.a = area
        ctx.h = z
        fsf.streampowerlawsingleflowdirection()
        e_channel = z - ctx.h
        ctx.h = z
        fsf.diffusion()
        e_hill = z - ctx.h
        z = z - e_channel - e_hill
    fsf.fastscape_destroy()
    return z.reshape(ny, nx), area.reshape(ny, nx)


def try_xsimlab_wrapper():
    """确认 fastscape 的 xarray-simlab 封装层在当前环境下能否运行（预期失败，记录原因）。"""
    times = np.arange(3) * DT
    ds_in = xs.create_setup(
        model=basic_model,
        clocks={"time": times, "out": [times[0], times[-1]]},
        master_clock="time",
        input_vars={
            "grid__shape": [32, 32],
            "grid__length": [31 * DX, 31 * DX],
            "boundary__status": "fixed_value",
            "uplift__rate": U,
            "spl__k_coef": K,
            "spl__area_exp": M,
            "spl__slope_exp": N,
            "diffusion__diffusivity": D,
            "init_topography__seed": 1,
        },
        output_vars={"topography__elevation": "out"},
    )
    ds_in.xsimlab.run(model=basic_model)


def run_landlab(z0: np.ndarray, nsteps: int, router: str = "priorityflood"):
    """router='priorityflood'：PriorityFloodFlowRouter（基于 richdem，D8 + 填洼）；
    router='depressionfinder'：FlowAccumulator + DepressionFinderAndRouter（纯 Python，慢）。"""
    n = z0.shape[0]
    mg = RasterModelGrid((n, n), xy_spacing=DX)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = z0.ravel()  # 与 Fastscape 用同一份初始地形
    # 四周开放、高程固定（Landlab 默认边界即为固定值）
    if router == "priorityflood":
        fa = PriorityFloodFlowRouter(mg, flow_metric="D8", depression_handler="fill")
    else:
        fa = FlowAccumulator(mg, flow_director="D8", depression_finder="DepressionFinderAndRouter")
    sp = FastscapeEroder(mg, K_sp=K, m_sp=M, n_sp=N)
    ld = LinearDiffuser(mg, linear_diffusivity=D)
    for _ in range(nsteps):
        z[mg.core_nodes] += U * DT
        fa.run_one_step()
        sp.run_one_step(DT)
        ld.run_one_step(DT)
    return z.reshape(n, n).copy(), mg.at_node["drainage_area"].reshape(n, n).copy()


summary = {"fastscape_version": fastscape.__version__, "landlab_version": landlab.__version__,
           "params": dict(dx=DX, U=U, K=K, m=M, n=N, D=D, dt=DT, nsteps=NSTEPS)}

# ---- 0. 封装层是否可用 ---------------------------------------------------------------
with rec.measure("fastscape（xarray-simlab 封装层）能否运行") as e:
    import xsimlab

    e["xsimlab_version"] = xsimlab.__version__
    try_xsimlab_wrapper()
rec.items[-1]["expected"] = "fail：xarray-simlab 0.5.0 与 zarr 3 / numpy 2.x 不兼容"

# ---- 1. 256x256、2 Myr：两者对比 ---------------------------------------------------
n = 256
z0 = initial_surface(n)
with rec.measure(f"Fastscape 内核 {n}x{n}，{NSTEPS} 步") as e:
    z_fs, a_fs = run_fastscape(z0, NSTEPS)
    e["mean_elev_m"] = round(float(z_fs.mean()), 1)
    e["max_elev_m"] = round(float(z_fs.max()), 1)
t_fs = rec.items[-1]["seconds"]

with rec.measure(f"Landlab {n}x{n}，{NSTEPS} 步（PriorityFlood D8 填洼 + FastscapeEroder + LinearDiffuser）") as e:
    z_ll, a_ll = run_landlab(z0, NSTEPS, router="priorityflood")
    e["mean_elev_m"] = round(float(z_ll.mean()), 1)
    e["max_elev_m"] = round(float(z_ll.max()), 1)
t_ll = rec.items[-1]["seconds"]

# 首次运行用 DepressionFinderAndRouter 时，整个脚本 10 分钟没跑完；这里只测 5 步量化它有多慢
with rec.measure(f"Landlab {n}x{n}，5 步（DepressionFinderAndRouter，纯 Python 洼地路由）") as e:
    run_landlab(z0, 5, router="depressionfinder")
summary_dfr_s_per_step = round(rec.items[-1]["seconds"] / 5, 3)

with rec.measure("两者结果对比") as e:
    core = (slice(1, -1), slice(1, -1))
    diff = z_ll[core] - z_fs[core]
    e["mean_elev_ratio_ll_over_fs"] = round(float(z_ll[core].mean() / z_fs[core].mean()), 3)
    e["corr_elevation"] = round(float(np.corrcoef(z_ll[core].ravel(), z_fs[core].ravel())[0, 1]), 3)
    la, lf = np.log10(a_ll[core] + 1), np.log10(a_fs[core] + 1)
    e["corr_log_drainage_area"] = round(float(np.corrcoef(la.ravel(), lf.ravel())[0, 1]), 3)
    e["rms_diff_m"] = round(float(np.sqrt((diff**2).mean())), 1)
    e["speedup_fastscape"] = round(t_ll / max(t_fs, 1e-9), 1)
np.save(work / "dem_fastscape_256.npy", z_fs)

# 稳态理论对照：河道 S = (U/K)^(1/n) · A^(-m/n)，即 log10 S = 2 - 0.4 log10 A。
# 首次运行在 2 Myr 时做这个检验，拟合指数为 +0.26（符号都不对）：当时中央仍是未被侵蚀的抬升平台
# （最高点 = U·t = 2000 m），地形远未达到稳态，且用了中心差分坡度。改为跑到稳态、用 D8 下游坡度检验。


def d8_downstream_slope(z: np.ndarray) -> np.ndarray:
    """每个节点到 8 邻域中最低点的最大下降坡度（D8 最陡下降）。"""
    best = np.zeros_like(z)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.roll(np.roll(z, dy, 0), dx, 1)
            s = (z - nb) / (DX * np.hypot(dy, dx))
            best = np.maximum(best, s)
    return best


with rec.measure("Fastscape 内核 256x256 跑到稳态：2000 步 = 20 Myr") as e:
    z_mid, _ = run_fastscape(z0, 1000)
    z_ss, a_ss = run_fastscape(z0, 2000)
    e["mean_elev_10Myr_m"] = round(float(z_mid.mean()), 1)
    e["mean_elev_20Myr_m"] = round(float(z_ss.mean()), 1)
    e["max_elev_20Myr_m"] = round(float(z_ss.max()), 1)
    e["steady"] = bool(abs(z_ss.mean() / z_mid.mean() - 1) < 0.01)

with rec.measure("稳态检验：河道 log10 S = 截距 + 指数 · log10 A") as e:
    s = d8_downstream_slope(z_ss)
    inner = np.zeros_like(z_ss, dtype=bool)
    inner[5:-5, 5:-5] = True  # 去掉边界附近（固定高程出口的落差）
    sel = inner & (a_ss > 1e7) & (s > 0)
    coef = np.polyfit(np.log10(a_ss[sel]), np.log10(s[sel]), 1)
    e["fitted_exponent"] = round(float(coef[0]), 3)
    e["fitted_intercept"] = round(float(coef[1]), 3)
    e["theory"] = "指数 -0.4，截距 log10(U/K) = 2.0"
    e["n_channel_nodes"] = int(sel.sum())
    e["status"] = "ok" if abs(coef[0] + 0.4) < 0.05 and abs(coef[1] - 2.0) < 0.3 else "fail"
np.save(work / "dem_fastscape_256_steady.npy", z_ss)
sa_points = (np.log10(a_ss[sel]), np.log10(s[sel]), coef)

# ---- 2. 规模扩展：每步耗时 ----------------------------------------------------------
scaling = {}
for size in (512, 1024):
    steps = 10
    need_gb = size * size * 8 * 60 / 1e9  # 粗估：约 60 个双精度场
    if available_gb() < need_gb + 1.0:
        rec.skip(f"规模测试 {size}x{size}", f"可用内存 {available_gb():.1f} GB 不足（需约 {need_gb:.1f} GB）")
        continue
    z0s = initial_surface(size)
    with rec.measure(f"Fastscape 内核 {size}x{size}，{steps} 步") as e:
        run_fastscape(z0s, steps)
    scaling[f"fastscape_{size}_s_per_step"] = round(rec.items[-1]["seconds"] / steps, 3)
    with rec.measure(f"Landlab（PriorityFlood）{size}x{size}，{steps} 步") as e:
        run_landlab(z0s, steps, router="priorityflood")
    scaling[f"landlab_{size}_s_per_step"] = round(rec.items[-1]["seconds"] / steps, 3)
scaling["fastscape_256_s_per_step"] = round(t_fs / NSTEPS, 4)
scaling["landlab_256_s_per_step"] = round(t_ll / NSTEPS, 4)
scaling["landlab_256_depressionfinder_s_per_step"] = summary_dfr_s_per_step
summary["scaling"] = scaling

# ---- 出图 --------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

ls = LightSource(azdeg=315, altdeg=40)
ext = (0, n * DX / 1e3, 0, n * DX / 1e3)
vmax = max(z_fs.max(), z_ll.max())
fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
for ax, z, title in ((axes[0, 0], z_fs, f"Fastscape ({t_fs:.1f} s)"), (axes[0, 1], z_ll, f"Landlab ({t_ll:.1f} s)")):
    rgb = ls.shade(z, cmap=plt.get_cmap("gist_earth"), vert_exag=2, blend_mode="soft", vmin=0, vmax=vmax, dx=DX, dy=DX)
    ax.imshow(rgb, extent=ext)
    ax.set_title(f"{title}: elevation after 2 Myr, max {z.max():.0f} m", fontsize=9)
    ax.set_xlabel("km")
for ax, a, title in ((axes[1, 0], a_fs, "Fastscape"), (axes[1, 1], a_ll, "Landlab")):
    im = ax.imshow(np.log10(a + 1), extent=ext, cmap="Blues", vmin=5.4, vmax=9.5)
    ax.set_title(f"{title}: log10 drainage area (m²)", fontsize=9)
    ax.set_xlabel("km")
fig.colorbar(im, ax=axes[1, :], shrink=0.7)
fig.suptitle("Spike 05 - same parameters & initial surface: uplift 1 mm/yr, SPL K=1e-5 m=0.4 n=1, D=0.01 m²/yr, "
             "256x256 @ 500 m", fontsize=10)
savefig(fig, "05_landscape_evolution.png")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
rgb = ls.shade(z_ss, cmap=plt.get_cmap("gist_earth"), vert_exag=2, blend_mode="soft", dx=DX, dy=DX)
ax1.imshow(rgb, extent=ext)
ax1.set_title(f"Fastscape kernel at steady state (20 Myr), max {z_ss.max():.0f} m", fontsize=9)
ax1.set_xlabel("km")
la_, ls_, cf = sa_points
ax2.scatter(la_, ls_, s=2, alpha=0.3, label="channel nodes (A > 1e7 m², interior)")
xx = np.linspace(la_.min(), la_.max(), 10)
ax2.plot(xx, 2.0 - 0.4 * xx, "r-", lw=2, label="theory: log S = 2 - 0.4 log A")
ax2.plot(xx, cf[1] + cf[0] * xx, "k--", lw=1.5, label=f"fit: log S = {cf[1]:.2f} {cf[0]:+.3f} log A")
ax2.set_xlabel("log10 drainage area (m²)")
ax2.set_ylabel("log10 D8 downstream slope")
ax2.legend(fontsize=8)
ax2.set_title("Steady-state slope-area relation vs stream-power theory", fontsize=9)
savefig(fig, "05_steady_state_check.png")
plt.close(fig)

rec.save(**summary)
