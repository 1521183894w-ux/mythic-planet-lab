# U36-S12：同一分片运动场的求积对照

日期：2026-10-28；基线c192b52。接续用户“拉取最新，然后继续”及“请继续‘”。本轮只研究S11同一运动场的材料求积；默认C2，未接全球/B5，阶段1未通过。

最细档起点对齐用了85343单元，来源加权绝对体积差仍0.00242246%；原分区33136单元为0.00226713%。起点对齐没有显示出效率优势，不能作通用修复，也没有证明分片导数跳变是15个非单调来源的唯一原因。

## 对照如何保持同一问题

复用S11同一两板块旋转、200km假设带宽、心射网络与归档连接、完整来源域。D盘本机已有S5/S8裁切与内域缓存，已按WKB哈希核对，重建S11所需输入；没有新全球演化。S11八案全部时间行和来源摘要与原归档相等，补齐34图及独立评审见u36_s11_device_rebuild.json和reviews/stage1_2026-10-28_6.md。

原有限三角形沿35Ma参考面的边实际裁切，GEOS约束三角剖分保留交片边界；物理面积回球面计算。原2071单元中1308个起点跨面，裁切为5344单元后起点跨面为0。最初逐UID库存最大差3.39365e-7km³；它是求积几何的浮点差，不是新物质。其后每个单元真实分4/16份，仍保留每个来源。裁切同时改变单元数和采样位置，这不是等成本的单因素试验。

沿实际中心轨迹累计S11解析面积因子J，H=H0/J；每个时点独立以有限球面三角形面积×中心厚度求体积。没有逐UID归一化。有限边仍取大圆弦段，不等同精确追踪变形后的曲边。所有141离散时点支持检查和有限单元正向检查通过，不证明连续时间或真实流变正确。

## 实际结果

| 分区 | 实际单元数 | 末态净差% | 来源加权绝对差% | 最差来源/有符号差% |
|---|---:|---:|---:|---|
| 原分区 L0 | 2071 | +0.00174510 | 0.03939753 | 448261 / +0.27381779 |
| 原分区 L1 | 8284 | -0.00060033 | 0.01001040 | 21809 / -0.09360755 |
| 原分区 L2 | 33136 | -0.00017028 | 0.00226713 | 448261 / +0.01711360 |
| 起点裁切 L0 | 5344 | -0.00356822 | 0.02050257 | 448261 / +0.27381779 |
| 起点裁切 L1 | 21339 | -0.00125909 | 0.00790763 | 448261 / +0.06845439 |
| 起点裁切 L2 | 85343 | -0.00034694 | 0.00242246 | 448261 / +0.01711358 |

原分区15个来源末态误差不随三档单调下降；这15个在起点裁切三档中均呈单调，但完整77来源又出现另一组16个非单调，不能称全部来源问题解决；完整77来源及原15个逐项对照都保留。UID448261三档最差误差约0.273818%→0.0684544%→0.0171136%，起点裁切没有消除它。不能用总净差或多数来源改善代替逐源结论。

按起点有限三角形是否越过中心所在的运动插值面，分组记录一步体积变化。原粗档跨面体积×时段份额65.416%，贡献73.301%的一步绝对体积变差；原细档为21.806%/52.210%。裁切后仍会跨面，三档份额约43.924%/22.176%/11.204%。这种关联不证明唯一因果：同片组也有误差，起点对齐没有保持后续持续对齐。一步有符号变化和末初体积差相等只是望远镜相消的算术核对，不当独立守恒验证。

## 微片、库存与复核

裁切产生低于既有1e-9km²面积精度底的微片，合计8.0974e-8km²（约0.081m²），逐父单元/UID保留面积及已知初始库存约2.99476e-6km³；这个排除量与已审计库存的净差口径不同，不能强行设为同一个数。更细分后又产生低于同一精度底的微片：L1/L2已知初始库存分别6.36781e-7/1.44382e-6km³，逐UID原值保留在precision_fragment_ledger；只不求不稳定有限面积比，不冒充物质耗尽，也不写成严格同一有效库存。相对同档S11的逐UID初始库存最大差约3.39e-7/4.45e-7/1.01e-6km³。原多边形、S11来源账与既有微片账不改，非陆壳与无覆盖保留缺值。

六案全部141时点逐UID复算与保存列相同；这是从保存点、累计J和H0重新求和，不是独立重新求解运动场。最小有限单元面积比均为正。不过精度底附近仍有已保留微片面积比约0.56–0.61，初始面积约1.5e-9km²；这些局部比值受浮点影响，不能把正面积或整体误差小当作局部演化可靠。34项保护代码/配置/默认摘要同c192b52，含S11映射源码、运行入口、测试和归档连接；没有改指标或默认世界。

相关测试范围实际41通过/2旧严格失败，1.27s；3项新增几何检查通过。旧material_band与step_audit的共同旋转断言保留，测试范围与S11的46项不同。全局测试未新增全量运行。

六案实际合计1381.95s，最大单案进程峰值1147.67MiB。原分区诊断首轮反复解压NPZ历史列增加耗时，代码已改成只读一次，未将上述实际耗时换成优化后的估计；没有重跑计时认证。标准29图另实际64s，工作单元整体峰值未测。

## 图件与复算

- 专题117–120：figures/design/v2/forward/u36_s12/index.html，同域网格、六案误差、原15来源、跨面关联。
- 标准29图：figures/stage1/u36_s12_reference/index.html，实际由S4冻结NC重出；指标剔除计时后、B判定与全部抽查窗口同S11。A7/7，主/极地B9/9，远方8/9；没有新世界或新海岸。
- 数值：data/worlds/design_v2/u36_s12_{partition,method,original_l0,original_l1,original_l2,aligned_l0,aligned_l1,aligned_l2,verification,manifest}.json，NPZ在.cache/u36_s12。

先按S11交接重建同一来源输入和原三档NPZ（须核对归档连接及来源域，不静默重择连接），在项目环境内运行：

```powershell
scripts/run.ps1 python scripts/design_piecewise_audit.py --prepare
foreach ($mode in @('original','aligned')) {
  foreach ($level in @(0,1,2)) {
    scripts/run.ps1 python scripts/design_piecewise_audit.py --mode $mode --level $level
  }
}
scripts/run.ps1 python scripts/make_figures.py --world .cache/u36_s4/control/world.nc --stage 1 --out figures/stage1/u36_s12_reference
scripts/run.ps1 python scripts/design_piecewise_audit_report.py
```

report会读.cache/u36_s12_tests.log；测试命令为pytest tests/tectonics/test_piecewise_audit.py tests/tectonics/test_boundary_transport.py tests/tectonics/test_material_band.py tests/tectonics/test_step_audit.py tests/tectonics/test_rotation_interpolation.py tests/tectonics/test_rotation_transport.py -ra --tb=short。2旧失败照实保留，不能据此写成全绿。

方法元数据见u36_s12_method.json。参考域/200km宽度为user_override，S11阶段旋转/SLERP为interpolated，裁切/求积诊断为procedural_constrained，图为visual_only；单位km、km²、km³、Ma/Myr，原物理输入L7约60km，细分不增加物理分辨率。实际复用pyGPlates1.0.0、SciPy1.18.1、Shapely2.2.0/GEOS3.14.1；没有新依赖安装。2026-10-28核对[Shapely约束剖分](https://shapely.readthedocs.io/en/stable/reference/shapely.constrained_delaunay_triangles.html)和[SciPy SLERP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)官方说明，只支持接口使用，不替本项目物理验证。

最初原细档在S11依赖NPZ生成前调用，保留.cache/u36_s12_original_l2_dependency_failure.log，依赖就绪后已完成。自动审批服务曾因用量上限未执行暂存/提交命令；用户继续后按正常审批成功重试，没有绕过审批。所有新增文件、缓存和计算在D盘既有工作树，没有系统设置或新大型下载。

下一步应先区分“同片内有限曲边/中心求积”与“跨片时间迁移”的误差，再决定是否需要分片求积或保守重映射的小试。本轮起点对齐不采用为全球修复；物理带宽/古接触、真实基底—热—沉积—水位与B5仍待独立证据。齐高地形、远方B1、r2细岸及阶段缺项均不因此关闭。独立评审已归档 `reviews/stage1_2026-10-28_7.md`：逐张看33图、六案各141时点全部77来源独立复算，最大保存体积差约5.82e-11km³；保留3项世界严重问题，只建议方法检查点。其对UID448261的核查显示六案跨片时段份额均为零而误差非零，支持跨片并非全部误差唯一成因。原稿前缀校验见 `u36_s12_review_integrity.json`，仅追加回应，阶段1未通过。
