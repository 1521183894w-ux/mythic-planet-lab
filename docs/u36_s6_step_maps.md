# U36-S6：小角度速度截断、事件对齐映射与局部材料连续性（2026-10-27）

接续 `2e64d67`，按用户“拉取最新，然后继续”；拉取实际返回 Already up to date。**本轮确认：S5两级启动短窗口反例的一项重要来源是框架将微小阶段旋转识别为恒等旋转，速度返回零而点映射仍移动。用点映射微分和事件对齐求积可显著缩小内部一致性误差，但共同刚体反例、边缘材料支持范围和物理带宽仍未闭合，不能接成全球或B5修复。**

## 已实际测到的区别

同一个 `pyGPlates1.0.0`，给定49个角度（1e−9—1e−3 rad）的已知赤道刚体旋转。1e−6rad旋转实际位移6.371m，`represents_identity_rotation()`返回True，`calculate_velocities`返回0；3.1623e−6rad一档已返回非零。49档扫描给出实际过渡区间，**不是公布一个跨版本硬阈值**。点坐标变化以`atan2(|x×y|,x·y)`计算，保留这一小角度；不通过框架角度反解再把小角度置零。

两级启动的34.5Ma同一13中心，窗口0.025Myr时全部非零，0.01/0.005/0.001Myr时13个全部为零；同网络点位移率中位始终约0.833351km/Myr。即使0.025—1Myr未截断，速度接口与点映射仍有小的插值差，不能只换成大窗口便称严格一致。

S5的端点梯形求积与本轮中点求积口径不同，因此不能把两轮的末态数字直接相减并全归因于截断。新审计把时间步对齐1Myr旋转关键帧，在**相同材料中点**比较：0.01Myr中心速度窗口、匹配dt的中心速度窗口、同窗口的点映射对数位移率。位置仍由框架推进；没有自己按新速度重建运动或改框架β。

真实参考段为同一432中心、W0=200km，求积dt=0.25Myr、框架位置内部步长0.125Myr、球面空间探针0.1km、材料斑块半径1km。0Ma初始面积加权的“散度积分−logJ”×100：短窗速度接口 -0.025262，匹配窗口 -0.024321，点映射微分 -0.000157。局部误差仍有异号；加权均值小不证明每处都准确。两级启动5个指定时刻的0.125Myr单步，`reconstruct_point`与时间序列点位置逐点相同；只核查这组样点/时刻。

## 五种输入历史及不可忽略的反例

恒速、35Ma突启、35→33Ma两级启动、增加方向变化/走滑、共同刚体输入，各测dt=0.25/0.125/0.0625Myr。前四种的运动不同，不作“最优历史”排名；这只是适用范围检查，不是为终图选更好看的运动。所有案显式覆盖0—50Ma、1Myr关键帧。

| 输入史 | dt0.25，末态均值对数差×100 | dt0.125 | dt0.0625 |
|---|---:|---:|---:|
|constant|-0.00055480|-0.00013725|-0.00003426|
|step|-0.00055480|-0.00013725|-0.00003426|
|ramp|-0.00051753|-0.00012800|-0.00003196|
|turning|-0.00055758|-0.00013952|-0.00003626|
|rigid|-0.00000509|-0.00000509|-0.00000509|

上表是**同一框架点映射场内部**的散度积分与材料面积，不是独立物理解的误差。共同刚体时，框架网络点映射的末态J中位约0.999876579，最大面积偏差约−0.0147588%；同输入精确刚体旋转J约1（本次最大约4e−12）。它不随积分dt消失。新严格测试也确认网络点位移率与同输入精确旋转不严格相符；**保留失败，不放宽断言，不用内部一致性替刚体正确性背书**。原S4严格β检查同样逐字保留。

正式轨迹审计（含15组运动/步长案、角度/窗口扫描与刚体基准）实际约62.09s。仅测量既有框架；没有把点映射微分包装成岩石圈流变求解器。

## 同一真实来源内的局部柱连续性

继承S5裁切的78原生宿主、77已知陆壳UID和无覆盖分账。初始已知陆壳体积约5554757.5574km³，与原裁切账相对差约1.5e−11。网格最长边15km和7.5km只是审计细分，**物理输入仍为L7，带宽仍为给定值**。

每个源三角形保持UID、H0；独立跟踪其中心附近的1km小斑块，用局部Jacobian推演`H=H0/Jac_center`。核对体积用有限源三角形的面积，因此没有直接用同一三角形的`H0/J_same_cell`制造恒等闭合。也没有跨UID调平、总量归一化或将灰缺值补成陆壳。柱连续性仍是不可压缩柱体的**运动学假设**，不是独立三维物质/流变解。

初次全带尝试在细化后体积残差反增，15km约+0.107888%、7.5km约+0.318907%。实际核查发现1km斑块的环点越出网络，网外部分的处理污染了Jacobian。正式结果在35→0Ma的**全部141采样时点**检查中心及四环点，固定保留全时段均在网内的材料子集；其余已知材料保留初始库存、厚度标缺值。采样时点检查不等于两个时点之间绝无越界。

| 来源审计 | 最长边15km | 最长边7.5km |
|---|---:|---:|
| 审计三角形数 | 8209 | 32815 |
| 已知陆壳三角形 | 7670 | 30662 |
| 全时段可评估陆壳三角形 | 7633 | 30421 |
| 因斑块越界而不评估的已知源面积/km² | 219.1323 | 638.3999 |
| 相应初始库存/km³ | 7794.0510 | 23000.3270 |
| 有效子集初始库存/km³ | 5546963.5064 | 5531757.2304 |
| 0Ma有效子集体积相对差/% | -0.00041803 | -0.00021268 |
| 本次实际用时/s | 50.97 | 208.65 |

两档有效子集不同，**不能仅凭残差下降称完整固定域收敛**。未评估库存不是被消耗或流失；它与S5原来的非陆壳1.3219%、无覆盖4.5009%是不同问题，均留账。141时点记录无源三角形或中心斑块翻转；没有高程、断层基底、热沉降、水位或B5新地貌。

## 检查、世界与出图

- 相关检查实际 **82通过、2失败，20.76s**。失败为原S4共同刚体β检查和新增同输入精确刚体点运动检查；没有跳过、xfail或调阈值。旧断言/模块逐字保留。16项保护Git内容相同。
- 本轮未重跑全球300→0Ma，复用S4冻结NC；标准管线实际36s重出29图。排除生成时间/计时字段后，所有世界指标与B判定同S5；A7/7、主大陆/极地B9/9、远方B8/9。远方100km海湾指数仍未过；齐高高地、r2细岸、B5等旧问题不自动关闭。
- 新专项图91—95：`figures/design/v2/forward/u36_s6/index.html`。标准图：`figures/stage1/u36_s6_reference/index.html`。图95色条遮挡已按实际看图修正并重出，最新34图哈希在`u36_s6_verification.json`。
- 原始结果：`data/worlds/design_v2/u36_s6_trajectory.json`、`u36_s6_sources.json`、`u36_s6_sources_refined.json`、`u36_s6_tests.txt`；全部轨迹数组和材料缓存为`.cache/u36_s6/`，不入Git。
- 所有新计算/文件在D盘既有隔离工作树；复用主工作区现有环境，未安装新依赖、下载大型数据或修改系统设置。本轮内存峰值未测，不用耗时冒充内存证据。

## 复算与方法标记

普通已建环境的项目工作区可运行（隔离工作树本轮以主环境解释器、工作树PYTHONPATH运行相同Python入口）：

```powershell
scripts/run.ps1 python -u scripts/design_step_audit.py
scripts/run.ps1 python -u scripts/design_step_audit.py --part sources --source-edge-km 7.5
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u36_s4/control/world.nc --stage 1 --out figures/stage1/u36_s6_reference
scripts/run.ps1 python -u scripts/design_step_audit_report.py
scripts/run.ps1 python -m pytest tests/tectonics/test_step_audit.py tests/tectonics/test_material_audit.py tests/tectonics/test_source_clipping.py tests/tectonics/test_material_band.py tests/tectonics/test_deformation_kinematics.py tests/tectonics/test_gap_diagnostics.py tests/tectonics/test_kinematics.py tests/tectonics/test_stretching.py tests/tectonics/test_engine.py -rA
scripts/run.ps1 python scripts/design_step_audit_verify.py
```

核验脚本读取`.cache/u36_s6_tests.log`，换机复算需将上述pytest实际输出保存到该文件；需先按S4/S5交接重建35Ma状态、GPML、材料裁切与冻结控制NC。未提交大型缓存。

2026-10-27核实官方[点映射接口](https://www.gplates.org/docs/pygplates/generated/pygplates.ResolvedTopologicalNetwork.html)、[网络速度窗口接口](https://www.gplates.org/docs/pygplates/generated/pygplates.TopologicalSnapshot.html)、[有限旋转与恒等旋转](https://www.gplates.org/docs/pygplates/generated/pygplates.FiniteRotation.html)。API默认值不替代本轮实测小角度扫描；本轮未改内部公式。方法标签：参考线/宽度`user_override`，框架场`interpolated`，独立映射微分/连续性审计`procedural_constrained`；没有新增`numerical_physics`层。

下一单元先处理**共同刚体输入的点映射失真和完整固定材料域的边界支持**，在精确刚体参照、变向运动、同一来源/同一域下核对之后再谈局部耦合。物理带宽与古接触仍须独立约束，不能从数值供体距离或这次审计细分反推。真实热/断层/水位成因与B5、r2细岸需要各自证据。默认仍C2，阶段1未通过。

独立报告已归档 `reviews/stage1_2026-10-28_1.md`：实际逐张看34图、哈希相符，保留真实伸展/地貌成因与r2细岸两项严重问题，仅建议方法检查点。原稿24530字节与SHA256已核对，开发者仅追加回应，证明见`u36_s6_review_integrity.json`。
