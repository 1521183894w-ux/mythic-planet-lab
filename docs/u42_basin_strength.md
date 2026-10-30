# U42：盆缘陡阶与有限强度对照

2026-10-30；基线 `12ad950`，按用户“拉起最新，然后继续”。`git pull --ff-only`实际返回 Already up to date；保留原有未跟踪文件。默认C2不变，阶段1未通过。本单元只推进盆缘，不声称处理了r2/固定旧r8的湾岬成因。

## 问题与方法

U41B固定剖面约31.11°N、103°E的陡阶附近，真实原生边的旧隐式流动系数最低为0；相邻单元存在很大的厚度差。控制中仍有有限的旧显式侧流，因此“完全阻流”仅指U40新增的隐式松弛，不是所有过程都禁止输运。强盆地可在地球上产生陡边，陡峭本身不等于错误；本轮检验的是人为零流动边界是否锁住可见陡阶。

先冻结同一U41B原生末态，在相同局部域内仅做20×1Myr输运小试，位置、源项与外边界冻结；这不是额外20Myr地史。之后完整运行300→0Ma三案，仅180Ma后将隐式松弛的边系数从`m`改为：

```
m_new = floor + (1 - floor) * m
floor = 0 / 0.01 / 0.05
```

最低系数是未标定的敏感性假设，不能称为实测黏度比。既有B6外侧250km和PL8内侧250km权重、旧显式侧流、侵蚀、源项、运动、U41B近邻接触出口均相同；不改终态高度、不截顶、不加噪声。复用已安装的SciPy稀疏求解，未新装依赖或求解器。

方法`u42_finite_strength_v1`；输运`procedural_constrained`，设计范围与180Ma时刻`user_override`，Airy/GDH1高程`parameterized_physics`，输出`interpolated`，图`visual_only`。L7约60km，输出1024×512；厚度km、高程m、体积km³、时间Ma。没有应力、热、流变、密度/组分、挠曲或沉积求解；体积闭合不等于完整质量守恒。继承U41的近邻接触假设和裂谷物源旧问题。

## 实际地图结果

[专题9图](../figures/design/v2/forward/u42_strength/index.html)；[5%候选标准29图及指标](../figures/stage1/u42_strength_05/index.html)。三案使用相同固定窗及剖面，旧r8中心22.2°N、28.2°W继续保留。

| 结果 | 控制0 | 1% | 5% |
|---|---:|---:|---:|
| 31.11°N固定行最大相邻像元落差/m | 2444.97 | 2328.44 | 1935.15 |
| B6中位/m | 441.16 | 441.16 | 441.16 |
| B6面积加权均高/m | 449.11 | 451.20 | 459.37 |
| PL8中位/m | 850.73 | 850.73 | 850.73 |
| 全球最高/m | 8074.87 | 8074.84 | 8074.73 |
| 全球相对控制变化>100m面积/km² | 0 | 1327.33 | 23812.60 |
| 最大绝对高程变化/m | 0 | 136.74 | 555.52 |
| 累计剥蚀相对控制/km³ | 0 | -2006 | -9469 |

最大像元落差是固定100–107°E行的诊断，**不是新增质量指标，也不是分辨率独立的物理坡度**。5%案降低约20.9%，但三条剖面仍十分接近，图上依然有陡边与大片平缓低地；只能说局部缓和，不能宣称自然盆缘已经形成。PL8没有被抬回大高原，B6也未整体填高。PL1设计窗依然不是独立分割的高原本体。

两案正高程海陆分类逐像元同控制，陆转海/海转陆面积均0；0Ma均无>10km像元。181个出口后步骤的最大原生厚度三案均93.48km，出口前为106.59km；不将这些厚度统计冒充每一步相对当时海面的全球高程，不声称三亿年全程上限已验证。

## 输运与代价

1%/5%案181步隐式松弛进入B6的分区净量约651.02/4010.30km³，控制0；PL8净量约183009.96/181955.18km³。这些是随板块移动设计窗口的表层算子账，不是完整物源历史；不含独立U41出口。预算JSON中的剥蚀数值按原接口取整，不冒充高精度全历史闭合。

两案松弛每步体积残差最大4.75e-08/4.68e-08km³，局部方程残差最大1.08e-08/7.54e-09km³；出口残差不超过1.75e-10km³。有限系数没有造出新物源，侵蚀响应仍随地形变化联动。

## 实际核对与性能

- 零下限控制与归档U41B：352个保存数组、16个原生数组完全相同；预算与事件计数相同。两候选激活前各132个数组同控制。
- 当前全部运行时输入SHA与实际源码/配置/初始缓存一致，见候选audit.json。
- 77项相关测试实际通过，8条既有Landlab警告；其中9项新检查覆盖控制逐数组等价、成对解析解与体积守恒、统一系数和非法输入。旧严格失败未重跑，不宣称全套测试通过。
- 控制/1%/5%总运行时间318.83/318.46/317.97s；独立进程生命期峰值847.25/846.87/848.23MiB。三进程并行，整体峰值未测；不能拿单进程值代替整体。
- 5%候选标准29图实际49s；A7/7，主大陆和极地大陆B各9/9、远方大陆8/9，100km海湾指标仍失败；25km细连接判定低于约39km格距的警告保留。指标定义与阈值未改。
- 9专题+29标准共38张图及SHA清单已冻结。独立评审归档后在下方追加状态，阶段未通过。

## 复现

在项目环境已准备好、U41B原生小试缓存存在时：

```powershell
scripts/run.ps1 python scripts/design_basin_strength_spike.py
scripts/run.ps1 python scripts/design_basin_strength.py --floor 0
scripts/run.ps1 python scripts/design_basin_strength.py --floor 0.01
scripts/run.ps1 python scripts/design_basin_strength.py --floor 0.05
scripts/run.ps1 python scripts/make_figures.py --world data/worlds/design_v2/candidates/u42_strength_05/design_v2_forward_v2_0Ma_u42_strength_05_1024x512.nc --stage 1 --out figures/stage1/u42_strength_05
scripts/run.ps1 python scripts/design_basin_strength_comparison.py --review 05
scripts/run.ps1 python -m pytest tests/tectonics/test_basin_strength.py tests/tectonics/test_arc_outlet_spike.py tests/tectonics/test_relaxation_spike.py tests/tectonics/test_b6_halo.py tests/tectonics/test_b6_transition.py tests/tectonics/test_b6_structured.py tests/tectonics/test_engine.py tests/tectonics/test_distribution_spike.py -q
```

本机PowerShell/Codex沙箱启动故障，实际通过cmd与已有`.cache/project_python.py`在D盘项目环境运行同一Python脚本，没有修改系统设置。换机先复建环境、初始缓存与U41B原生状态；NC/NPZ/pickle均可重算且不提交Git。JSON方法、结果与评审图保留，不能仅拉Git就假设大型缓存也存在。

## 参考与下一步

[Cook & Royden 2008](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JB005457)与[Clark等2005](https://academic.oup.com/gji/article/162/2/575/2050588)支持强度差异会影响高原边缘与流动路径，但没有标定本项目的floor值、立方响应或图形边界。[SciPy官方spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html)仅作为成熟线性求解工具，不提供地质真实性担保。

有限阻流只带来局部变化，不能继续用调floor代替构造层级。下一单元转向r2一段具体湾—岬—海槽的实际演化与结构，直接处理地图成因；盆缘仍保留为未关闭问题，后续需区分真实强块接触、地壳层结构与表层侵蚀的作用。

本地预览仅服务figures目录。标准画廊用到的51张公开地球参照缩略图由既有地球图集复制到earth_reference/，不重复提交Git；换机复建地球区域图集后，最后运行本轮comparison脚本即可复制。两个画廊实际HTTP检查89个PNG地址均成功，其中38张为本轮评审图，其余为地球参照；38张与Git暂存逐字一致。

## 独立评审归档

[独立报告](../reviews/stage1_2026-10-30_3.md)已原样归档，开发者只在末尾追加回应。评审员实际看38/38图、核对SHA38/38一致；未读源码、PROGRESS、旧评审或本交接。报告列2类严重、3项一般：盆缘台阶仍严重；r2与固定旧r8两处岸因缺口合并计为另一类严重，不能与上轮逐岸段数量直接比较。串珠岛列、光滑内陆、远方海湾不足/海底尖峰保留。评审不建议请求阶段通过，可展示局部机制检查点。

报告在子agent最终回复交付后由主会话原样写盘，原文长度与raw/LF SHA、仅追加前缀核对已记audit.json。默认C2不变，阶段1未通过。下一单元优先处理r2具体湾—岬—海槽的形成顺序和实际地貌，盆缘与旧r8均不关闭。
