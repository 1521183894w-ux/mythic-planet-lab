# U38：B6 与 PL8 渐变阻流敏感性试验

2026-10-29；基线 f16328d。**仅作实验检查点，不采用为默认世界，阶段 1 未通过。**

## 方法与边界

延续 U37，在真实 300→0 Ma 正向演化中，把终图 B6 和 PL8 范围假设为随 101 号板块移动的阻流区，从 180 Ma 起生效。每个区域的权重在其内部从边界零值向内按 250 km 平滑上升；流动最多减少 99%，环形溢出接收容量按权重暂时预留后再恢复。此处两块区域形状、运动、起始时间与阻力均是人为假设，不是已证实的地质历史。区域标签为 user_override，地壳分配为 procedural_constrained；高程是 parameterized_physics，输出 interpolated，图是 visual_only。网格 L7 约 60 km，输出 1024×512；单位为 km、km³、Ma 和 m。原引擎、上限、均衡与剥蚀背景见 docs/tectonics_method.md 和 U37 交接；这不是应力或沉积求解器。

## 真实对比

同一窗口、三次完整正向运行：[专题画廊](../figures/design/v2/forward/u38_transition/index.html)；[标准图组](../figures/stage1/u38_b6_pl8_transition/index.html)。

| 区域 | 原控制中位 | U37 硬边候选 | U38 渐变候选 |
|---|---:|---:|---:|
| B6 | 4945 m | 452 m | 4794 m |
| PL8 | 4945 m | 4945 m | 1012 m |

B6 高于 4500 m 的面积比例依次为 70.0%、5.7%、65.4%；PL8 依次为 52.5%、55.5%、23.6%。这些是既定设计窗口内的统计，并不代表地质真实性。图上 PL8 有较宽降高区但西北侧仍陡、内部平滑；B6 仍被高原覆盖，不能称盆地。U38 同时更改了两处区域及边界过渡，不能单凭这一组试验归因于其中哪一项。高程变化超过 100 m 的全球面积较控制约 127.31 万 km²，最大变化约 4531 m；正高程海陆分类无变化。对照 U37，累计剥蚀少约 74.02 万 km³、岛弧新增少约 2722 km³；这些只是预算变化，不是完整物质去向解释。既有裂谷材料来源问题仍未关闭。

A 层 7/7；主大陆及极地大陆 B1/B2 各 9/9，远方大陆 8/9。指标不代替图像评审；约 25 km 判定半径小于输出格距，细连接有分辨率限制。

## 核验与复算

实跑模拟 231 秒，含出数及对照总 259.33 秒；标准图组 40 秒，内存峰值未测。激活前 300–190 Ma 12 帧、每帧 11 项数组与 U37 相等；每个作用步骤检查溢出分配账差绝对值不超过 0.01 km³，19 个 10 Ma 抽样时点最大约 1.91e−6 km³。仅说明该算子账，不能证明其他材料来源或物理合理性。新增的部分容量分配守恒及边界权重测试，与 U37、引擎及分配小试相关测试均通过；8 条 Landlab 警告，未运行全套测试。原始 run.json 中 basin_* 字段在本次指 B6 与 PL8 联合区域；此标签已在脚本中改名，原运行记录保留不篡改。

项目环境和初始世界缓存就绪后，运行：

    scripts/run.ps1 python scripts/design_b6_transition.py
    scripts/run.ps1 python scripts/design_b6_transition_comparison.py
    scripts/run.ps1 python scripts/make_figures.py --world data/worlds/design_v2/candidates/u38_b6_pl8_transition/design_v2_forward_v2_0Ma_u38_b6_pl8_transition_1024x512.nc --stage 1 --out figures/stage1/u38_b6_pl8_transition
    scripts/run.ps1 python -m pytest tests/tectonics/test_b6_transition.py tests/tectonics/test_b6_structured.py tests/tectonics/test_engine.py tests/tectonics/test_distribution_spike.py

对照脚本需要 .cache/u36_s3/world.nc 和 U37 本机 NC；NC 和大缓存不进 Git。记录、指标、图和脚本已留档。试验时默认沙箱初始化失败、PowerShell 报 CET 兼容错误，使用命令提示符和项目 Python 实际运行；没有修改系统设置。

## 独立评审与后续

独立报告见 reviews/stage1_2026-10-29_4.md，严重问题 4 项：B6 仍不成盆地、随机海岸 r8 的复杂成因未讲通、巨大平顶高原、正式阶段评审材料不完整。评审只建议作为实验检查点。此前 r2 成因疑虑也未因本次区域抽样变化或分级变化自动关闭。

下一步应在盆地核心维持适当阻流的同时研究盆缘过渡，并保留 PL8 方向的改善；两处机制分开对照，检查地形剖面、被排挤的地壳及全球代价。现有两个候选均未采用。
