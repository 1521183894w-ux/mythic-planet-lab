# U40：把瞬时环形溢出改为有限速率厚度松弛

2026-10-30；基线 `37e33a2`。按用户“拉取最新，然后继续”，用本机已启用的代理核对后，远端与本地 main 一致。保留原有未跟踪的 `.codex/` 和 `scripts/design_template.py`。

本单元是未通过的机制实验检查点。默认 C2 不变，阶段 1 未通过。

## 问题、方法和边界

U39B 已同时保住 B6 低地和 PL8 较低高原，但留下齐高台面、盆缘陡阶。本轮先出一维对照图，再运行两案完整 300→0 Ma 世界：共同保留同一初始缓存、运动历史、U39B 的 B6/PL8 权重、180 Ma 激活、剥蚀与原有弱侧流；仅在 180 Ma 后把瞬时截顶并逐圈填满的算子换为有限速率厚度松弛。

新算子在现有原生陆壳邻接边上计算：
`D_ij = kappa * [max(max(T_i,T_j)-40, 0)/25]^3 * mobility_ij`，
`mobility_ij = 1-max(w_i,w_j)`。按现有陆架保护条件筛边；质量矩阵采用实际单元面积，边导度复用既有约 60 km 网格系数；以 SciPy `spsolve` 解一次冻结系数的后向欧拉步。常数零空间平移减轻大时间步的消减误差；没有末态截顶、裁零或事后归一化。每步重新核对成对边流与厚度体积变化。

两档 kappa 为 3000 和 10000 km²/Myr，代表未标定的输运敏感性。约 15 km 通道、黏度约 10¹⁸–10²⁰ Pa·s 的文献范围仅提供量级背景；这里的 `T-40` **不是**求解出的真实通道厚度。文献也明确显示温度、流变和边界会限制远距离通道流动，不能由曲线变顺就认定正确。本机沿用已有 SciPy 1.18.1（BSD 许可），没有新增环境依赖；CPU/Windows 已实跑，不宣称它是网上最新版本。

范围、强度、板块归属和激活年代为 `user_override`；厚度输运为 `procedural_constrained`；Airy/GDH1 高程为 `parameterized_physics`；输出采样为 `interpolated`；图为 `visual_only`。方法名 `u40_frozen_graph_relaxation_v1`；L7 约 60 km，输出 1024×512；厚度 km、体积 km³、时间 Ma、高程 m。未求解应力、温度、真实流变、挠曲、沉积或水位。

与旧算子的差别不止时间尺度：旧环形分配能跳过保留接收容量的区域，并在陆壳不足时溢出到洋壳；新算子沿边传输，核心边完全不透过，也没有洋壳接收。因此本轮是整个分配算子的替换对照，不能把所有变化归因于“有限时间”一个因素。

## 实际地图结果

[专题画廊](../figures/design/v2/forward/u40_relaxation/index.html)有 9 张图；[U40B 标准图组](../figures/stage1/u40_relax_10000/index.html)有 29 张。专题包括合成剖面、真实世界同窗图、固定剖面、表层收支、真实年代、全球代价、旧 r8 固定复查，以及明确显示超高岛弧的诊断色带。

| 设计窗口 | U39B | U40A：3000 | U40B：10000 |
|---|---:|---:|---:|
| B6 像元中位高程 m | 448 | 441 | 441 |
| B6 >4500m 面积占比 | 0% | 0% | 0% |
| PL8 像元中位高程 m | 963 | 851 | 851 |
| PL8 >4500m 面积占比 | 18.2% | 0% | 0% |
| PL8 90分位高程 m | 4874 | 1220 | 1240 |

PL8 原来的高台残块明显减少，B6 保留低地；但约 31.1°N 剖面仍有很陡的高原—低地过渡，高原主体反而超过原来的 4945 m。两新案均在东南岛弧产生 13 个 >10 km 像元，合计约 1.983 万 km²，最高 **21370.5 m**，约在 142.91°E、5.80°S；原生保存场最大厚度约 **173.42 km**。两案这一极值相同，本轮将扩散系数从3000提高到10000，未消除这些受限位置的积料。

该超限作为模型失败保存。NC 中超限格标有 `fantasy_exception`，但**没有接受为幻想设定**，也没有采用为世界结果。标准高程色带会饱和，因此专题 08 用 0–22000 m 的诊断色带直接揭示问题；没有更改标准色带。

相对 U39B，全球高程变化 >100 m 的面积分别约 **1307.63 / 1772.45 万 km²**；陆转海约 3.367 万 km²，海转陆约 0.586 / 0.728 万 km²。改动涉及全球，不能只展示 PL8 的局部改善。累计剥蚀相对 U39B 分别增加约 1176.89 万 km³、减少约 196.92 万 km³；岛弧新增均减少 5.53 万 km³，复活增厚、裂谷减薄和旧补洞预算也出现联动变化。逐项差值见 comparison.json，不能把这些预算差当成完整物源归因。

U40B 的 A 层 7/7；主大陆及极地大陆 B1/B2 各 9/9，远方大陆 8/9，失败项仍是 100 km 海湾指数。细线连接半径 25 km 小于约 39 km 格距的警告保留。指标通过不足以抵消高原和岛弧的严重问题。

## 数值检查与实际运行

- 两案激活前 300–190 Ma 的 12 帧、各 11 数组，均与 U39B 完全相同，共各 132 数组；180 Ma 起结果才分离。
- 每案 181 个新算子步骤。全域体积变化最大绝对账差约 2.47e−8 / 3.56e−8 km³；逐格厚度变化与成对边流的最大差约 4.64e−9 / 9.84e−9 km³。账闭合仅是数值检查，不代表科学过程正确。
- B6 核心新松弛净接收均为 0；PL8 累计约 4107 / 183313 km³。分区随各时刻 101 板块上的设计范围查询，不是物源标签。四分区本轮直接统计，不使用“其余单元差额”冒充独立来源；新算子没有洋壳接收。账本仅含表层算子，不含碰撞等其他来源。
- 两案含出数实际耗时 **262.68 / 262.01 秒**，并发执行；各独立模拟进程生命周期峰值 **842.08 / 842.68 MiB**，包括初始化、演化和世界输出，末尾 run.json 序列化未另测，整体并发峰值未测。标准 29 图实际 **35 秒**。
- 相关 **60 项检查通过，8 条既有 Landlab 警告**。包括不等面积双单元解析解、大时间步守恒与无新极值、障碍与孤立单元、置换等价性，以及既有 B6/引擎/分配试验回归。未跑全套；旧严格失败未重新验证，不能视为已解决。
- 一维合成试验三案用相同 80 Myr 高斯供给，不含剥蚀；原环形分配形成齐平台，新方案形成连续剖面但更慢方案更高。它不是球面或实际下地壳的物理验证。
- 首次大时间步测试因消减误差约 1.006e−6 km³ 略超 1e−6 门槛而失败；采用常数平移后通过，未放宽测试门槛。早期出图脚本的命令长度和标题转义错误已修复后实跑。失败输出保留本机缓存，不报告为成功。

运行输入源码哈希与当前文件均匹配；38 图哈希已冻结于 review_packet.json/audit.json。NC 仅补充方法/超限标签，地形值未改；原始大缓存与 NC 不进 Git。

## 复算与交接

项目环境与 U39B 初始/对照缓存就绪后：

    scripts/run.ps1 python scripts/design_relaxation_spike.py
    scripts/run.ps1 python scripts/design_b6_relaxation.py --kappa 3000
    scripts/run.ps1 python scripts/design_b6_relaxation.py --kappa 10000
    scripts/run.ps1 python scripts/design_b6_relaxation_comparison.py
    scripts/run.ps1 python scripts/make_figures.py --world data/worlds/design_v2/candidates/u40_relax_10000/design_v2_forward_v2_0Ma_u40_relax_10000_1024x512.nc --stage 1 --out figures/stage1/u40_relax_10000
    scripts/run.ps1 python -m pytest tests/tectonics/test_relaxation_spike.py tests/tectonics/test_b6_halo.py tests/tectonics/test_b6_transition.py tests/tectonics/test_b6_structured.py tests/tectonics/test_engine.py tests/tectonics/test_distribution_spike.py -q

当前电脑仍有 PowerShell/CET 与受限执行环境启动问题，实际以命令提示符和项目 Python 运行，临时入口 .cache/run_project.py 复用了 scripts/env_vars.ps1 的 D 盘环境变量及已有系统代理。没有修改系统设置、安装软件或新增 C 盘项目缓存。

下一行动应围绕真实世界改进：先分别处理陆壳内部有限搬运与岛弧/薄壳/孤立域的真实邻接和材料出口，再决定高原流变响应；同时复查盆缘陡阶。不得只重加统一截顶掩盖超高岛弧，也不得靠扩大系数或渐变带宽度宣布修复。旧 r2/r8 海岸成因、裂谷材料来源、岩性与完整历史交付仍未解决。

## 独立评审

独立报告 [reviews/stage1_2026-10-30_1.md](../reviews/stage1_2026-10-30_1.md) 已归档。评审员逐张实际看38图并核对全部图片SHA；未读源码、旧评审或开发者自评，未复算指标/原生网格。列3项严重（岛弧超限、r2局部复杂岸成因、旧r8固定窗）与6项一般问题；不建议作为验收或可采用世界提交，只允许明确标为失败诊断检查点展示。报告原文保持，仅追加回应；原文前缀已核对。旧盆缘、高原、物源与其他历史问题不因本轮分级变化关闭。

## 参考

- Clark 等（2005），[Dynamic topography produced by lower crustal flow against rheological strength heterogeneities bordering the Tibetan Plateau](https://academic.oup.com/gji/article/162/2/575/2050588)：下地壳流动与强盆地边界的成因背景，非本项目公式的验证。
- [Limit of channel flow in orogenic plateaux](https://doi.org/10.1130/L114.1)（2010）：温度、流变、边界与通道远距离搬运的限制。
- [SciPy 官方 spsolve 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html)：成熟稀疏线性系统求解器；本项目只实现试验算子的组装。
