# 阶段 0 · 环境、工具验证与地球基准

> 先读 `CLAUDE.md` 和 `PROGRESS.md`。本阶段不生成任何虚构世界。
> 目标：搭好可在两台电脑上重建的环境；验证关键科学工具；**用地球数据建立质量基准，并证明这套指标和评审能识别出"圆润世界"**。

---

## 为什么这个阶段很重要

上一轮失败的直接原因，是没有任何东西能告诉开发者"这个世界不够好"。本阶段要先造好这把"尺子"，并且用一个故意做坏的世界检验它：**如果尺子量不出坏世界，后面所有阶段都没有意义。**

---

## 工作单元

### 0.1 环境审计（只读）

- 检查并记录：Windows 版本、CPU、内存、GPU 与显存、D 盘剩余空间、已安装的 Python / Node / Git / 编译器、WSL 状态、虚拟化是否开启、常用端口占用。
- 输出 `docs/setup_report_<机器名>.md`。两台电脑各做一次（用户换机时提醒他运行）。
- **不安装任何东西**，只给出建议清单。

### 0.2 Git 与项目骨架

- 初始化 Git 仓库，写好 `.gitignore`（已有初稿，按需补充）。
- 建立目录骨架（参考下方"目录约定"），首次提交。
- 远程仓库：给出创建私有仓库（GitHub 或 Gitee，由用户选择）和 `gh auth login`（或 SSH 密钥）的分步说明，**由用户本人完成登录**。完成后推送。
- 在 `使用说明.md` 中补充实际的同步命令。

### 0.3 Python 环境

- 环境必须完全位于工作区内（例如 `environment/`），可整目录删除重建。
- 地理空间库（GDAL、rasterio、cartopy、pyproj 等）在 Windows 上用 pip 安装经常出问题。**建议评估 micromamba（conda-forge 源）**：它是单个可执行文件，环境前缀可以放在工作区内。也可以与 uv 对比，择优选用并说明理由。
- 选择与科学包兼容的 Python 版本（不盲目用最新），锁定依赖版本（例如 `environment.yml` + 锁文件）。
- 编写 `scripts/setup_env.ps1`：在新机器上一键重建环境。在当前机器上实际运行一遍验证。

### 0.4 科学工具快速验证（spike）

对每个工具做"安装 → 导入 → 跑一个最小样例 → 记录结果"，并标注 `required_now` / `optional_later` / `not_feasible_here`。结果写入 `docs/tool_evaluation.md`。

| 工具 | 用途 | 本阶段要求 |
|---|---|---|
| pyGPlates / GPlately | 板块旋转、重建、记录 | 必测 |
| stripy 或其他球面网格方案 | 球面三角网格（构造模拟的基础） | 必测，可与 scipy 的球面 Voronoi 方案对比 |
| xarray + zarr / netCDF | 数据存储 | 必测 |
| matplotlib + cartopy | 标准出图 | 必测 |
| scipy.ndimage / scikit-image | 形态学运算、指标计算 | 必测 |
| richdem 或 pysheds | 洼地填充、流向、流域 | 必测 |
| Landlab | 区域侵蚀与沉积 | 必测 |
| Fastscape | 地貌演化 | 尽量测 |
| goSPL | 全球尺度地貌演化（球面网格，可能需要 WSL/Linux） | 只调研文档，记录可行性，不安装 |
| ExoPlaSim | 3D 大气环流 | 只调研文档与 WSL 需求，不安装 |

同时做一份**构造模拟方法调研**，写入 `docs/tectonics_survey.md`，至少覆盖：
- Cortial 等人的 *Procedural Tectonic Planets*（Computer Graphics Forum, 2019）；
- 其他可参考的开源板块模拟项目；
- 用 pyGPlates 记录运动学、自写规则处理地壳相互作用的混合方案。

调研目的：为阶段 1 选型提供依据。只调研，不实现。

### 0.5 质量指标代码 + 地球基准

- 在 `packages/metrics/` 实现 `quality_metrics.md` 中 A、B1、C 层里**仅需高程数据**的指标（海陆比、连通性、海岸复杂度各项、距海距离、高程曲线、坡度分布等）。每个指标配单元测试。
- 下载 NOAA ETOPO 2022 的低分辨率版本（例如 60 角秒，下载前报告文件大小）和 Natural Earth 必要图层。来源、版本、日期、许可证写入 `data/external_manifests/`。
- 重采样到 512×256、1024×512、2048×1024 三级格网，计算全部指标，存为 `data/earth_reference/baselines_<分辨率>.json`。
- 按大陆分别计算海岸复杂度（亚非欧、北美、南美、澳洲等），得到**地球的合理范围**。
- 生成地球区域参照图集（地中海、爱琴海、挪威海岸、东南亚群岛、东非裂谷、加勒比、波罗的海等，以及 `archetype_library.md` 中各原型的地球例子所在区域），存放在 `data/earth_reference/regions/`。

### 0.5b 奇景覆盖度核对

- 按 `docs/archetype_library.md` 第 8 节，对照 UNESCO 世界自然遗产名单（重点是标准 vii 与 viii），把每个遗产地的主要景观归入原型库中的某个原型。
- 归不进去的，按原型库格式补充新原型，并标注表达等级（A/B/C）。
- 结果写入 `docs/archetype_coverage.md`，新增原型清单交用户过目。

### 0.6 标准出图管线

- 实现 `scripts/make_figures`，按 `review_protocol.md` 第 2 节生成标准图组和 `index.html` 画廊。
- 用地球数据跑一遍。

### 0.7 尺子校准：故意做一个坏世界

这是本阶段最关键的验收项。

- 生成一个**"圆润世界"对照样本**：海陆比 71/29、一块连通大陆占 80%、有一个内海，但海岸平滑、没有半岛和群岛、山脉是随机噪声斑块。也就是上一轮 Codex 的典型产出。
- 对地球、圆润世界各跑一遍指标和标准图组。
- **期望结果**：
  - 圆润世界**通过 A 层硬约束**（证明硬约束不代表质量）；
  - 圆润世界**在 B1 海岸复杂度上明显不达标**；
  - 地球自身落在所有 B、C 层基准范围内。
- 调用 `world-reviewer` 分别评审地球和圆润世界（不告诉它哪个是地球）。**期望**：它能指出圆润世界的严重问题，且不对地球提出严重问题。
- 如果指标或评审没能区分两者，**先修尺子**，不能进入阶段 1。

---

## 交付物

- 可重建的工作区环境与 `scripts/setup_env.ps1`
- Git 仓库（已推送到用户的远程仓库）
- `docs/setup_report_<机器名>.md`、`docs/tool_evaluation.md`、`docs/tectonics_survey.md`
- `packages/metrics/` 及测试
- 地球基准数据与区域参照图集
- 标准出图管线与画廊
- 尺子校准报告：`reviews/stage0_calibration.md`
- 奇景覆盖度核对：`docs/archetype_coverage.md`

## 验收

- [ ] 环境能用脚本在当前机器上从零重建（记录耗时与占用空间）
- [ ] 所有指标有单元测试且通过
- [ ] 地球基准已建立，地球自身落在基准范围内
- [ ] 圆润世界通过硬约束、但被 B1 指标和评审员同时识别为不合格
- [ ] 用户看过地球图组与圆润世界图组，在 `PROGRESS.md` 写下"通过"

## 本阶段明确不做

- 不做任何虚构世界的构造模拟
- 不做三维界面、前端、后端 API
- 不安装 WSL、ExoPlaSim、goSPL（只调研）

## 磁盘预算

本阶段工作区总占用目标 **≤ 10 GB**。下载单个文件超过 2 GB 前必须征得用户同意。

---

## 目录约定（初稿，可在本阶段调整并记录理由）

```
工作区/
├─ CLAUDE.md  PROGRESS.md  使用说明.md  .gitignore
├─ .claude/agents/
├─ config/            # metrics.yaml、生成参数等
├─ docs/
├─ tasks/
├─ packages/
│  ├─ metrics/        # 质量指标与原型识别
│  ├─ tectonics/      # 阶段 1
│  ├─ terrain/        # 阶段 2
│  ├─ hydrology/      # 阶段 2
│  ├─ climate/        # 阶段 3
│  ├─ ocean/          # 阶段 3（含潮汐）
│  ├─ astronomy/      # 阶段 3
│  ├─ ecology/        # 阶段 3
│  └─ world_schema/   # 数据结构与存取
├─ scripts/
├─ tests/
├─ data/
│  ├─ earth_reference/
│  ├─ external_manifests/
│  └─ worlds/         # 生成的世界（大部分不进 Git）
├─ figures/           # 标准图组（不进 Git，可重建）
├─ reviews/           # 评审报告（进 Git）
├─ environment/       # Python 环境、WSL 等（不进 Git）
└─ .cache/            # 可清理缓存（不进 Git）
```
