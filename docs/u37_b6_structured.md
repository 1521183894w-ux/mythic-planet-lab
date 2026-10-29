# U37：B6 红层盆地真实世界修复候选

2026-10-29；基线 `1a24f29`。按用户“请继续”，从局部材料方法检查转向真实地貌。**这是候选，尚未替换默认 C2，不是阶段 1 通过。**

## 尝试与约束

针对原控制世界中 B6 红层盆地被 4945 m 齐高平台覆盖的问题，用真实 300→0 Ma 演化试一个继承强盆地：B6 的终图多边形随 101 号板块运动，从 180 Ma 起视为阻挡侧向地壳分配的区域。终图区域和开始时间是 `user_override`，不是已经证明的古地质历史；0.01 的流动倍率是试验参数，不是测得的岩石黏度。

候选减少涉及盆地单元的侧向通量，并让厚度尚低于上限的盆地单元不接收超厚地壳的环形溢出；直接碰撞、剥蚀及其他世界过程继续运行。溢出的体积按旧算法分配到其他有容量的单元，不在终态直接降低高程。原有 65 km 统一上限与环形分配仍在，因此不把候选称为力学求解器或最终方案。碰撞直接增厚仍可进入盆地；强分区边界没有被证明是现实地质结构。

## 实际结果：有进展，也有代价

[真实同窗图](../figures/design/v2/forward/u37_b6/01_b6_before_after.png)依次为终图示意、原控制、候选及候选减控制。终图高度仅为示意，不是物理真值。

| 同一设计窗口 | 原控制中位高程 | 候选中位高程 | 现象 |
|---|---:|---:|---|
| B6 红层盆地 | 4945 m | 452 m | 恢复低盆地角色；边界贴近设计多边形，过于整齐 |
| PL8 喀斯特高原 | 4945 m | 4945 m | 没有恢复较低高原；高于 4500 m 的面积占比约 52.5%→55.5% |

B6 设计窗口中高于 4500 m 的面积占比约 70.0%→5.7%；候选对最终高程并未直接画入目标值，但这不能消除按目标多边形人为设定强盆地所带来的形态偏置。图中盆地呈近矩形、边缘陡直，不能将它当作自然形成的完整地貌。输出是 1024×512，物理 L7 分辨率约 60 km，不表示细尺度地形。

全球海陆正高程分类没有像元改变；高程变化超过 100 m 的面积约 95.68 万 km²，最大绝对变化约 4628 m。对照覆盖全球，不能只凭 B6 的中位数判断无副作用。候选 A 层 7/7，主大陆和极地 B 层各 9/9，远方大陆 8/9；指标不是视觉验收，也未修改定义或阈值。

## 材料与验证范围

真实控制和候选在激活前 300–190 Ma 十二个保存时点、每时点 11 个格网字段相同。候选 181 个作用步骤均检查环形分配前后材料体积差绝对值不超过 0.01 km³；19 个十百万年采样点中最大绝对差约 1.91e−6 km³。这只检查该算子账，不证明全球所有材料来源合理。候选累计剥蚀比控制增加约 60.95 万 km³，岛弧新增约多 902 km³；不能把盆地降低的全部体积直接归结到某一个去向。既有约 1.020 亿 km³ 裂谷整柱新增来源问题仍未解决。

完整候选运行实际约 247.37 秒；图组 33 秒；峰值内存未测。相关引擎与既有分配测试 47 项通过、8 条 Landlab 警告；另有一项针对性测试确认盆地不接收溢出时其他单元接收且总量不变。没有运行完整测试集，也没有声称旧共同刚体严格失败已解决。

候选标准图：[画廊](../figures/stage1/u37_b6_structured/index.html)，指标和实际输出在 `data/worlds/design_v2/candidates/u37_b6_structured/`；原生世界 NC 约 16 MB 保留本地、不进 Git，可按代码重算。区域范围 `user_override`，地壳候选 `procedural_constrained`，高程 `parameterized_physics`，输出 `interpolated`，图 `visual_only`。版本、参数、输入哈希及引用见 `run.json`、`comparison.json` 和 `audit.json`。

## 复算

在项目既有 D 盘环境和初始世界缓存准备好后：

```powershell
scripts/run.ps1 python scripts/design_b6_structured.py
scripts/run.ps1 python scripts/design_b6_comparison.py
scripts/run.ps1 python scripts/design_b6_audit.py
scripts/run.ps1 python scripts/make_figures.py --world data/worlds/design_v2/candidates/u37_b6_structured/design_v2_forward_v2_0Ma_u37_b6_structured_1024x512.nc --stage 1 --out figures/stage1/u37_b6_structured
scripts/run.ps1 python -m pytest tests/tectonics/test_b6_structured.py tests/tectonics/test_engine.py tests/tectonics/test_distribution_spike.py
```

下一步应让继承强区的边缘和地壳分配过渡更自然，同时直接改善 PL8，并复查被排挤的材料及附近山系；不要根据 B6 数字达标直接采用。历史、岩性、复杂海岸和阶段其余严重问题仍未完成。


## 独立评审与回应

独立报告 reviews/stage1_2026-10-29_3.md 实际查看 29 张标准图及专题图，保留 3 项严重问题：B6 近矩形与陡直盆缘、PL8 极高平顶、r2 复杂岸细部成因不足。评审认可局部研究检查点，不建议作为阶段成果验收。开发者接受并保留这些问题；下一步直接改善盆缘过渡与 PL8，同步检查材料代价和全球影响，默认 C2 不变。原评审正文仅追加回应，追加前 SHA-256 为 c6cf9aebd367c0f58e15ebf56af4817dcd6fe5e35651f4efe7c11b7b2882951e。
