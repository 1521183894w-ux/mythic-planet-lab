# 科学工具快速验证（阶段 0.4）

> 测试日期：2026-10-21；机器：台式机 HANS600452（i5-12600KF，16 GB 单通道内存，测试时可用内存约 3.7–4.4 GB，后台程序较多）。
> 环境：`environment.win-64.lock.txt`（Python 3.12.15，conda-forge）。
> 每项结论都来自实际运行：脚本在 `spikes/stage0_tools/`，图在 `figures/stage0/tools/`，原始数值在 `figures/stage0/tools/results/*.json`（`figures/` 不进 Git，重跑脚本即可再生成）。
> 运行方式：`powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u spikes\stage0_tools\01_spherical_mesh.py`（其余同理，06 依赖 05 的输出）。

标签含义：
- **required_now**：后续阶段的核心依赖，已在环境中验证可用，保留。
- **optional_later**：有用，但不是近期必需；或需要额外条件（WSL、补丁）才能用。
- **not_feasible_here**：在本项目当前环境下不能用。

---

## 1. 结论总表

| 工具 | 版本 | 许可证 | 用途 | 结论 | 一句话理由 |
|---|---|---|---|---|---|
| **pyGPlates** | 1.0.0 | GPL-2.0-only | 板块运动学：旋转模型、重建、速度、GPML 存档 | **required_now** | 旋转插值、层级组合、速度与解析值完全一致（误差 ≤ 1.4e-6） |
| GPlately | 2.0.0 | GPL-2.0+ | pyGPlates 的封装、出图、地球板块模型下载 | optional_later | 与 pyGPlates 结果一致；对虚构星球不是必需，地球参照模型可联网获取 |
| **stripy** | 2.3.3 | LGPL-2.0+ | 球面三角网格、邻接、插值、梯度 | **required_now** | 65 万点建网 1.4 s；线性插值二阶收敛 |
| **scipy.spatial** | 1.18.1 | BSD-3 | cKDTree、SphericalVoronoi（单元面积） | **required_now** | 与 stripy 三角剖分完全一致；KD 树查询比 stripy 快 4–7 倍 |
| **xarray + zarr v3 / netCDF4** | 2026.9 / 3.4.1 / 1.7.4 | Apache / MIT / MIT | 世界数据与构造历史存储 | **required_now** | 数值与 `method_class` 等元数据（含中文）无损往返；按时间追加、惰性读取可用 |
| **matplotlib + cartopy** | 3.11.2 / 0.26.0 | PSF / BSD-3 | 标准出图 | **required_now** | Robinson/正射/区域投影正常；Natural Earth 下载落在工作区内 |
| **scipy.ndimage + scikit-image** | 1.18.1 / 0.26.0 | BSD-3 | 连通区域、距离、海岸线、形态学 | **required_now** | 2048x1024 上每项 < 0.4 s；但有三个必须处理的坑（见 §3.5） |
| **richdem** | 2.4.3 | GPL-3.0-only | 洼地填充、D8 流向、汇水面积 | **required_now** | 与 Fastscape 内核相关 0.99；2048x1024 0.5 s；Landlab 的 PriorityFlood 路由也依赖它 |
| pysheds | 0.5 | GPL-3.0+ | 同上 | optional_later | 与 numpy 2.4+ 不兼容（调用已删除的 `np.in1d`），需补丁才能运行；比 richdem 慢约 7 倍 |
| **Fastscape 内核**（fastscapelib-fortran） | 2.8.4 | GPL-3.0-only | 区域地貌演化（隐式 SPL + 扩散） | **required_now**（阶段 2 核心） | 稳态坡度—面积关系与理论吻合（指数 -0.398 vs -0.4，截距 2.001 vs 2.0）；比 Landlab 快约 40 倍 |
| fastscape（xarray-simlab 封装层） | 0.1.0 / xsimlab 0.5.0 | BSD-3 | 上述内核的高层接口 | **not_feasible_here** | xarray-simlab 0.5.0 使用 zarr 3 已删除的 `zarr.MemoryStore` 和 numpy 2 已删除的 `np.in1d`，`run()` 直接报错 |
| Landlab | 2.10.1 | MIT | 区域地貌演化组件库 | optional_later | 结果与 Fastscape 一致（平均高程比 0.998），但慢 40 倍；组件丰富，特殊过程（滑坡、岩性层等）可按需使用 |
| goSPL | 2026.7.14（PyPI） | GPL-3.0 | 全球球面地貌演化 | optional_later（需 WSL） | 依赖 PETSc + MPI，官方源码构建只说明 Linux；Windows 原生不可行。阶段 2 全球侵蚀的候选 |
| ExoPlaSim | 3.4.2（PyPI） | GPL-2.0 | 三维大气环流 | optional_later（需 WSL） | 需 gfortran + OpenMPI；官方说明 Windows 10 经 WSL 可用。阶段 3 候选 |

**许可证提示**：核心工具大多是 GPL（pyGPlates 为 GPL-2.0-only；richdem、fastscapelib-fortran 为 GPL-3.0-only）。本项目的代码只是在运行时调用这些库，个人创作使用不受影响。若将来要给本仓库代码声明一个开源许可证，需要先核对兼容性（GPL-2.0-only 与 GPL-3.0-only 互不兼容）。这不是法律意见，届时再单独评估。

---

## 2. 对后续阶段的直接影响（重要发现汇总）

1. **球面网格可以支撑阶段 1。** stripy 二十面体网格 L7（16.4 万点，约 60 km）全部操作在 1–2 s 内，L8（65.5 万点，约 30 km）建网 1.4 s，进程峰值约 1.6 GB。阶段 1 的多种子筛选建议用 L7，选中的世界再用 L8 复算。
2. **0.5 的指标代码必须处理三件事**（用地球数据实测确认，不处理就会算错）：
   - 经度 ±180° 环绕：不处理时地球陆块数 686，处理后 683，欧亚大陆的楚科奇部分会被算成独立陆块；
   - 面积按 cos(纬度) 加权：不加权陆地占 33.1%，加权后 28.75%（地球公认约 29%）；
   - 距海岸距离用球面大圆距离：平面欧氏距离把欧亚"离海最远点"算成 2762 km 且位置偏离；球面算法得 2486 km @ 45.8°N 88.7°E，与文献 ≈2510 km @ 46.3°N 86.7°E 吻合（误差约 1%）。
3. **海岸线长度依赖分辨率。** 2048x1024 下地球海岸线 52.1 万 km，这个数只能和同分辨率的地球基准比，不能和文献里的海岸线长度直接比。0.5 的海岸复杂度指标要在同一格网上和地球基准对照。
4. **阶段 2 的侵蚀用 Fastscape 内核，直接调用 Fortran 接口。** 它的 Python 封装层（xarray-simlab）已不可用，本次已写出等价的驱动循环（见 `05_landscape_evolution.py` 的 `run_fastscape`），并用稳态理论验证正确。
5. **河网要在侵蚀后的地形上算，湖泊和内流盆地要显式处理。** 在侵蚀演化后的地形上 richdem / pysheds / Fastscape 三者汇水面积几乎一致；但在未经侵蚀、有 30% 格点需要填洼的合成地形上，三者相关只有 0.34–0.49——平地与洼地中的流向取决于算法。奇幻星球一定有内流盆地和大湖，阶段 2 必须明确选一种处理方式并记录。
6. **构造历史的存储量需要提前规划。** 30 片 1024x512 的历史（float32）压缩后 58 MB。阶段 1 若每 1 Ma 存一帧、L7 网格、多个属性，单个种子可达数百 MB，几十个种子会超出磁盘预算。建议筛选阶段每 10 Ma 存一帧，候选世界再按"种子 + 参数 + 代码版本"确定性重算出 1 Ma 历史（见 `docs/tectonics_survey.md` §5）。

---

## 3. 各工具详细结果

### 3.1 球面网格：stripy vs scipy（`01_spherical_mesh.py`）

图：`figures/stage0/tools/01_spherical_mesh_plates.png`（L6 网格上用加权区域生长"长出"14 块板块，再映射到 2048x1024 栅格）。

| 级别 | 顶点数 | 间距（地球半径） | stripy 建网 | scipy ConvexHull | scipy SphericalVoronoi | 网格→2048x1024 最近点：stripy / cKDTree |
|---|---|---|---|---|---|---|
| L5 | 10,242 | ~239 km | 0.015 s | 0.08 s | 0.10 s | 0.91 s / 0.14 s |
| L6 | 40,962 | ~119 km | 0.06 s | 0.35 s | 0.47 s | 0.99 s / 0.15 s |
| L7 | 163,842 | ~60 km | 0.28 s | 0.78 s | 2.0 s | 1.14 s / 0.22 s |
| L8 | 655,362 | ~30 km | 1.43 s | 3.28 s | 8.5 s | 1.28 s / 0.32 s |

- stripy 与 scipy ConvexHull 的三角形集合完全相同（L5、L6 逐个比对）；三角形数 = 2N−4。
- 顶点度数 5–6（12 个五度点来自二十面体顶点），Voronoi 单元面积最大/最小 = 1.36，面积和 = 4π。
- 插值精度（光滑测试函数，线性插值到 1024x512）：最大误差 L5 1.8e-3 → L6 4.5e-4 → L7 1.1e-4，每加密一级约缩小 4 倍（二阶收敛）。
  - 首次测试用了 sin(3·lat)·cos(2·lon)，误差虚高到 1.7。原因是该函数在两极随经度变化、本身不连续，不是 stripy 的问题；已换成用三维坐标定义的光滑函数。
- 板块区域生长演示：14 块板块面积占比 25.7%、16.0%、11.3% …… 0.3%，跨日界线的板块连续无裂缝。首次参数（生长速度差异过大）导致一块板块占 80%，已调小。**这只是网格图运算的演示，不是构造模拟。**
- 结论：网格与邻接用 stripy（自带梯度、平滑、插值）；大量最近点查询用 scipy cKDTree；需要单元面积时用 SphericalVoronoi。

### 3.2 板块运动学：pyGPlates / GPlately（`02_plate_kinematics.py`）

图：`figures/stage0/tools/02_plate_kinematics.png`（三块虚构板块 0–200 Ma 的位置与现今速度箭头）。

- 用程序生成 PLATES4 `.rot` 文件（3 块板块，板块 201 相对板块 101 运动），pyGPlates 读入后：
  - 任意时刻（含 75 Ma 插值）的总旋转与解析值偏差 0；
  - 层级组合 (201→000) = (101→000)∘(201→101) 偏差 0；
  - 速度与解析值 ω·R·sinΔ 的最大偏差为最大速度的 1.4e-6（0–3.23 cm/yr）。
- 特征集合写出为 `.gpmlz`（3 个多边形 4.5 KB）并读回一致；重建到 5 个年代正常。
- GPlately 2.0 的 `PlateReconstruction` + `Points` 用同一虚构模型重建到 100 Ma，结果与直接用 pyGPlates 计算一致（1.0407°N, 29.1644°E）。
- plate-model-manager 能联网列出 33 个地球板块模型（含 Müller 2019、Merdith 2021、Müller 2022、Zahirovic 2022），可作为构造统计的地球参照。
- 未测：拓扑闭合的板块边界（`TopologicalModel`）。用程序构造拓扑特征较复杂，留到阶段 1.1 原型时再测。
- 注意：GPlately 导入时会在当前目录建 `gplately.log`、`ptt.log`；统一用 `_common.import_gplately()` 先切到 `.cache/logs` 再导入（已实测日志落在 `.cache/logs`）。

### 3.3 存储：xarray + zarr / netCDF（`03_storage.py`）

| 测试 | 结果 |
|---|---|
| 2048x1024 两变量（12.6 MB 原始）→ zarr v3 | 7.8 MB，0.06 s；数值与元数据完全一致 |
| 同上 → netCDF4（zlib 4） | 7.1 MB，0.2 s；完全一致 |
| `method_class`、方法、版本、单位、中文假设说明、中文全局属性 | 两种格式都无损往返 |
| 球面网格 65.5 万节点 × 2 变量 → zarr | 14.7 MB，0.11 s，一致 |
| 按时间追加 30 片 1024x512（模拟 300 Myr、每 10 Myr 一片） | 58 MB，0.8 s；惰性打开后只读 150 Ma 一片，数值正确 |

- 提示：zarr 3 写出时会警告"合并元数据（consolidated metadata）尚未进入 zarr v3 规范"。不影响本项目使用。
- 结论：世界数据与构造历史用 zarr（支持追加、按块惰性读取）；需要和其他软件交换时导出 netCDF。

### 3.4 出图与地球陆地数据（`04_mapping_morphology.py`）

图：`figures/stage0/tools/04_landmasses_distance.png`、`04_projections_regional.png`。

- cartopy 自动下载的 Natural Earth 文件落在 `.cache/xdg-data/cartopy/`（工作区内）。
- 绘制耗时：Robinson 两联图（2048x1024 栅格 + 50m 海岸线）含保存 3.3 s；正射 + 地中海区域图 1.0 s（数据已缓存时）。
- 地中海区域图上，2048x1024 陆地掩膜（约 20 km 像素）与 Natural Earth 50m 海岸线吻合良好，爱琴海岛屿可见——这正是后面"区域放大图 + 地球对照"所需的分辨率下限。

### 3.5 形态学与指标运算（同上脚本）

| 运算（2048x1024） | 耗时 | 结果 / 注意事项 |
|---|---|---|
| 连通区域（8 连通）+ 经度环绕合并 | 0.01 s | 683 个陆块（不环绕 686）。最大陆块面积（百万 km²）：79.55（亚欧非）、37.99（美洲）、12.07（南极）、7.62（澳大利亚）、2.15（格陵兰），与公认值相符 |
| 陆地比例 | — | 不加权 33.05%，按 cos(纬度) 加权 28.75% |
| 距海岸距离：平面 EDT | 0.14 s | 2762 km @ 48.1°N 94.5°E（错误） |
| 距海岸距离：球面（海岸像素 KD 树） | 0.33 s | 2486 km @ 45.8°N 88.7°E（文献 ≈2510 km @ 46.3°N 86.7°E） |
| `find_contours` 海岸线 + 球面长度 | 0.09 s | 1172 条，52.1 万 km（依赖分辨率） |
| 开运算 disk(2)：被削掉的细窄陆地 | 0.04 s | 1.78%（半岛/细长陆地指标的雏形）。`skimage.morphology.binary_opening` 在 0.26 已弃用，改用 `scipy.ndimage.binary_opening` |
| 盒计数分形维数（全球海岸像素，2–64 px） | 0.02 s | 1.31（等距圆柱投影下，高纬有畸变；0.5 中需按区域、在等面积或局部投影下计算） |
| regionprops 紧凑度 4πA/P² | 0.05 s | 像素空间计算有投影畸变，0.5 中需改为球面面积/周长 |

### 3.6 地貌演化：Fastscape 内核 vs Landlab（`05_landscape_evolution.py`）

图：`figures/stage0/tools/05_landscape_evolution.png`（2 Myr 对比）、`05_steady_state_check.png`（稳态检验）。

统一参数：dx = 500 m，抬升 1 mm/yr，K = 1e-5，m = 0.4，n = 1，D = 0.01 m²/yr，dt = 10 kyr，四边固定高程，同一份初始随机地形。

| 测试 | Fastscape 内核 | Landlab（PriorityFlood + FastscapeEroder + LinearDiffuser） |
|---|---|---|
| 256x256，200 步（2 Myr） | 3.8 s | 148.6 s（约 40 倍） |
| 每步耗时 512x512 | 0.11 s | 3.0 s |
| 每步耗时 1024x1024 | 0.47 s | 24.6 s（另一次运行 12.3 s，后台负载影响大） |
| 2 Myr 后平均高程 | 1550.0 m | 1547.5 m |
| 高程相关 / 汇水面积 log 相关 | — | 0.955 / 0.625 |

- 两者宏观一致（平均高程比 0.998）；河网细节不同（相关 0.625），因为初始地形是白噪声，汇流方向的平局处理不同就会分岔，属正常。
- Landlab 的纯 Python 洼地路由 `DepressionFinderAndRouter` 在 256x256 上每步 1.4 s；首次运行用它跑 1024x1024，10 分钟没跑完。应使用 `PriorityFloodFlowRouter`（依赖 richdem）。
- **物理正确性检验**：Fastscape 内核跑 2000 步（20 Myr）后，内部河道节点的 D8 下游坡度与汇水面积拟合为 log S = 2.001 − 0.398 log A，理论值（S = (U/K)·A^(−m/n)）为 2.0 − 0.4 log A。
  - 首次在 2 Myr 时做这个检验得到 +0.26（符号都不对）：当时中央还是未被侵蚀的抬升平台（最高点 = U·t = 2000 m），且用了中心差分坡度。已改为跑到稳态、用 D8 下游坡度。
  - 20 Myr 时河道已满足稳态关系，但整体平均高程在 10→20 Myr 间仍变化 2.5%（分水岭与坡面仍在调整），按 1% 标准未判为完全稳态。
- 封装层问题：`fastscape` 0.1.0 依赖的 xarray-simlab 0.5.0 不兼容 zarr 3（`zarr.MemoryStore`）和 numpy 2.x（`np.in1d`）。已按 fastscape `basic_model` 的过程顺序直接驱动 fastscapelib-fortran 内核。
- 备选：fastscapelib（C++ 新版，conda-forge 0.3.0，2025-06）有维护中的 Python 绑定，未在本次测试；阶段 2 若需要更灵活的网格/边界（例如周期边界）再评估。

### 3.7 水文：richdem vs pysheds（`06_hydrology.py`）

图：`figures/stage0/tools/06_hydrology_compare.png`（稳态地形上三者的汇水面积）。

| 测试 | richdem | pysheds（加补丁） | Fastscape 内核（参照） |
|---|---|---|---|
| 256x256 稳态地形，最大汇水面积 | 1937.8 km² | 1937.0 km² | 1935.5 km² |
| 与 Fastscape 的 log 汇水面积相关 | 0.990 | 1.000 | — |
| 河道（>10 km²）像素重合度 | 0.976 | 1.000 | — |
| 2048x1024 合成地形耗时 | 0.49 s | 3.46 s | 0.45 s |

- pysheds 0.5 在 9 处调用 `np.in1d`（numpy 2.4 起删除），不加补丁时 `accumulation` 等直接报错。测试中临时用 `np.isin` 补上；项目代码不应依赖这种补丁。
- 合成地形（多尺度噪声，未经侵蚀）上三者 log 汇水面积相关只有 0.34–0.49：29.6% 的格点被填洼抬高，这些格点上相关只有 0.24；未填格点上 0.55。推测其余差异来自未侵蚀地形上分水岭不清、坡度相同时的取舍不同（未单独验证）。
- richdem 的 `FlowAccumulation` 输出的是上游格点数（含自身），乘以格点面积才是汇水面积；它会向标准输出打印大量文献信息，需要重定向。

---

## 4. 未安装工具的调研

### 4.1 goSPL（全球尺度地貌演化）

| 项 | 内容 | 来源 |
|---|---|---|
| 最新版本 | 2026.7.14（2026-07-10 发布）；GitHub 仓库 2026-10-13 仍有提交 | PyPI、GitHub 提交记录 |
| 许可证 | GPL-3.0 | PyPI |
| 能力 | 全球非结构化球面网格（区域可加密），河流侵蚀（流功率律）、坡面扩散、内陆洼地 priority-flood、海洋沉积、双岩性沉积与地层、可选挠曲均衡；由板块运动、古地形、古气候驱动；官方称全球分辨率 5–100 km | CSDMS 模型页、goSPL 技术指南 |
| 依赖 | MPI、PETSc ≥ 3.21、并行 HDF5；Python 3.11/3.12；PyPI 无二进制包，需现场编译 Fortran 扩展 | goSPL 安装文档 |
| 平台 | 官方安装文档只说明 Linux 源码构建，另有 Docker 镜像与 conda `geodels` 频道；未提及 Windows 原生（CSDMS 页面列出 Windows，但安装文档未给出 Windows 路线） | goSPL 安装文档、CSDMS |
| 代表性应用 | Salles 等 2023（Science）用它重建过去 1 亿年地球地貌（二手报道称约 10 km 分辨率，未核实原文） | 新闻报道 |
| 在本项目的可行性 | Windows 原生不可行；WSL2 + conda/Docker 可行性高。台式机 16 GB 内存下全球 25–50 km 网格量级预计可承受，未实测 | — |
| 建议 | 阶段 2 全球侵蚀的首选候选（它是唯一成熟的球面全球 LEM，且能吃板块运动作为输入）。区域细节用 Fastscape 内核。WSL 启用后在阶段 2 开头做 spike | — |

### 4.2 ExoPlaSim（三维大气环流）

| 项 | 内容 | 来源 |
|---|---|---|
| 最新版本 | 3.4.2（2025-01-02 发布）；GitHub 仓库 2026-09-19 仍有提交 | PyPI、GitHub 提交记录 |
| 许可证 | GPL-2.0 | PyPI |
| 依赖 | GNU C/Fortran 编译器、OpenMPI（官方强烈推荐 GNU + OpenMPI）；`pip install exoplasim`，首次使用时现场编译 | ExoPlaSim 文档 |
| 平台 | Linux（Ubuntu 22.04 测试）；**Windows 10 经 WSL 可用**；macOS 需 Xcode | ExoPlaSim 文档 |
| 分辨率与速度 | T21（64x32 格点、10 层）为主力；论文给出：笔记本 4 线程 T21 跑 1 模式年约 3.75 分钟；T42、16 线程约 15–16 分钟/年；并行线程数上限为纬度格点数 | Paradise 等 2022（MNRAS 511, 3272），arXiv:2107.07685 |
| 平衡时间 | T21 水球模型需 50–200 模式年达到能量平衡（判据：每十年漂移 < 0.5 W/m²） | 同上 |
| 可设参数 | 自定义陆地掩膜 `landmap` 与地形 `topomap`（模型分辨率下的 `.sra` 文件）、地形缩放 `orography`、自转周期、黄赤交角、偏心率、恒星通量/温度、重力、半径、气压、混合层深度、冰川、植被 | ExoPlaSim API 文档 |
| 限制 | 海洋是 50 m 混合层"板状海洋"，**没有动态洋流热输送**；海冰无法完全关闭 | Paradise 等 2022 |
| 在本项目的可行性 | 需 WSL。按论文笔记本速度估算，T21 跑 100–200 年 spin-up 约 6–12 小时一个情景（未实测，台式机 6 大核预计同量级） | 估算 |
| 建议 | 阶段 3 的大气环流候选。由于无洋流热输送，洋流需另行参数化并标注 `parameterized_physics`。WSL 启用后在阶段 3 开头做 spike，先测台式机实际速度 | — |

---

## 5. 测试中修正过的错误（如实记录）

| 问题 | 原因 | 处理 |
|---|---|---|
| stripy 插值误差虚高到 1.7 | 测试函数在两极不连续 | 换用三维坐标定义的光滑函数 |
| 板块演示一块板块占 80% | 生长速度差异参数过大 | 调小参数，结果 25.7% 起递减 |
| 板块速度核对报 NaN | 测试点恰好在欧拉极上（速度 0），相对误差除零 | 改用"最大偏差 / 最大速度" |
| 地图上澳大利亚与美洲同色 | 按标签号取余配色撞色（计算本身正确） | 改为按面积排名配色 |
| 稳态检验指数 +0.26 | 2 Myr 时未达稳态，且用中心差分坡度 | 跑到 20 Myr，用 D8 下游坡度 |
| 05 脚本 10 分钟未跑完 | Landlab 纯 Python 洼地路由在 1024x1024 上过慢 | 改用 PriorityFloodFlowRouter |
| 曾误以为 gplately 日志写到了仓库根目录 | 误读了命令输出（实际是 `ls .cache/logs` 的结果） | 逐步追踪确认日志都在 `.cache/logs` |

---

## 6. 参考与来源

- goSPL：<https://github.com/Geodels/gospl>、<https://gospl.readthedocs.io>、<https://csdms.colorado.edu/wiki/Model:Gospl>；Salles, Mallard & Zahirovic (2020), *JOSS* 5(56), 2804, doi:10.21105/joss.02804。
- ExoPlaSim：<https://github.com/alphaparrot/ExoPlaSim>、<https://exoplasim.readthedocs.io>；Paradise et al. (2022), *MNRAS* 511, 3272（arXiv:2107.07685）。
- 欧亚大陆离海最远点：Garcia-Castellanos & Lombardo (2007), *Scottish Geographical Journal* 123(3), 227–233。
- Natural Earth：<https://www.naturalearthdata.com>（公有领域）。
