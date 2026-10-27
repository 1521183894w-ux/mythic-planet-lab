# U36-S4：真实C/D运动锚定的参考材料带（2026-10-27）

接续最新提交`faaa277`与U36-S3。本单元用已有 **pyGPlates 1.0.0变形网络**，在南裂谷海中段建立一次登记的参考材料带，区分接触身份、物理带宽假设、连续材料面积与网格补缝。已经实际运行并出图，但**框架伸展标量与独立面积有未解释残差，严格刚体检查仍失败；尚不能接入B5或替换全球裂谷规则。默认仍C2，阶段1未通过。**

## 本次做了什么

选用认可的C/D分区边界、亦为G8中段的`(20.6°E,21°S)→(19.2°E,14°S)`，时间35→0Ma。既有控制历史完整旁观复跑，在35Ma保存原生状态；没有用S2过深候选作初始条件，也没有改变运动或再次叠加局部B5减薄。

直接把终图分区线转到35Ma，并不等于那时的实际材料接触。原200km条带中，约91%参考面积查询归D，C侧边界仅一半查询属于预期C陆壳。随后只在35Ma、原线附近横切匹配：28条横切中12条发现唯一直接C陆壳→D陆壳转换，保存两侧查询宿主UID；另外16条未定，没有跨无覆盖区强行补接触。一次线性拟合将两端分别移约−152.44/−47.46km，P90拟合残差20.45km。**这是局部插值登记，不是唯一古断层恢复、完整共轭陆缘或全球接触拓扑。**

登记后200km带两侧各28个边界点均查询到预期的C/D陆壳。带内94.213%面积有可查询陆壳厚度，剩余5.787%为无覆盖或非陆壳，保持缺值；没有补成35km厚柱。基准包含252顶点、432审计三角形，初始查询涉及68个宿主，连接关系与材料编号贯穿时间序列。查询UID表示被取样的初始宿主，**未精确裁切原宿主面积，也未把它们的完整体积重新分配给新网格**，不能据此宣布引擎成分运输已经完成。

两条网络侧边分别固定到真实301/401板块，速度和应变插值、材料点积分均调用pyGPlates。没有自写流变求解器，没有应变率截断，没有额外指数型裂谷减薄。两端延伸400km作为网络缓冲，参考材料段单独追踪；瞬时网络边界不当成材料边界。初始法向张开约4.812–5.899km/Myr、走滑约7.582km/Myr，来自给定参考法向与既有运动，不是最近点连线。

## 带宽仍是假设，不能由刚体运动唯一确定

200km为明确的试验参考总宽度，并用100/400km对照。它代表区域窄裂谷的百公里量级假设，尚无真实流变约束，没有按B5的−400m目标选优。

| W0/km | 末态材料面积比 | 已知陆壳区H0/J中位/km | 初始陆壳厚度缺值面积 |
|---:|---:|---:|---:|
| 100 | 2.874056 | 12.465 | 11.574% |
| 200 | 1.936781 | 18.383 | 5.787% |
| 400 | 1.468198 | 24.400 | 3.043% |

同一运动下不同带宽给出不同减薄，证明面积支撑必须明确，不能由“最近12供体”决定。表中H0/J是整柱厚度，**不是地面高程、断层基底位移或湖水深度**。没有热沉降、挠曲、沉积、侵蚀、水位或B5盆底输出。

## 几何审计与尚未解释的框架偏差

实际完成27组带宽×采样×时间步对照（W0=100/200/400km，采样60/30/15km，dt=1/0.5/0.25Myr），另有4组端部缓冲、6组速度/应变插值组合和9组解析负/正对照。网络及审计约79.14s，不是一次全球模拟耗时。每案检查35、30、25、20、15、10、5、0Ma八个时点；**不把八时点无越界写成所有内部子步已检查**。

- 27组八时点均未见三角形翻转或中心离开网络。三角形面积总和与独立跟踪多边形面积约1e−12相对量级一致。H0/J账本最大绝对残差约1.86e−9km³；后者是定义性闭合，不能单独证明速度场或厚度演化正确。
- 200km基准末态单元J范围1.8313–2.0422。框架自带伸展因子beta与独立J的相对偏差P90约1.840%；若以H0/beta计算已知厚度区，总体积约偏大1.290%，均匀初始厚度积分偏大1.308%。两种口径分开，不是隐蔽新增材料。
- 减小时间步、细化采样和切换六种框架速度/应变插值组合未消除约百分之一残差，**根因未定位**。不能把“成熟框架”当作忽略独立检查的理由，也不能把H0/J闭合当成框架beta已验证。
- 不设端部缓冲时总面积比约1.91117；200/400/800km缓冲为1.936773/1.936781/1.936780。本段结果对足够远的端部较稳定，但不证明其他裂谷段、弯曲/释放阶区或时变接触都适用。
- 光滑边界纯走滑的连续真值J=1，dt=0.25Myr实际面积误差约−62.80ppm；共同刚体约+1.756ppm。偏差很小也没有写成严格零。它们不对应S3那种走滑网格缝隙造成的大量假裂谷，却仍是框架数值近似的边界。

本轮method_class：参考位置/带宽`user_override`；原生接触登记、速度/应变插值和初始厚度取样`interpolated`；有符号球面面积与固定材料账`procedural_constrained`。没有图层标成`numerical_physics`。输入、分辨率、单位、假设、版本与哈希见配置及结果JSON。

## 验证、标准世界与测试失败

控制完整300→0Ma实际339.69s；与既有C10持续控制的352保存数组、预算、事件、最终原生状态和全部世界数组一致。十个终图/正式历史/候选参照/指标/旋转/默认摘要/核心引擎保护文件的Git clean对象与基点HEAD相同。新代码不接入Engine，网络结果只在隔离试验目录保存。

完整标准管线实际66s重出29图。A层7/7；主大陆/极地B项各9/9，远方大陆8/9，`bay_index@100=0.01285125<0.01345213`仍失败。主内海约445.89万km²，3亚大陆；这只是相同控制世界的复核，没有新世界质量改善。

**相关测试67通过、1失败，12.49s，未声称全通过。** 失败为新材料带的共同刚体框架beta严格检查，局部最大偏差约1.47e−7。第一次运行还在1e−9面积断言处失败（实际3.39e−9）；确认框架网络插值并非直接施加解析刚体矩阵后，将面积数值检查写成0.1ppm，并另查解析位置与框架标量。后者仍失败，原始失败和最终运行均归档；没有继续放宽它让检查变绿，也没有修改任何世界质量指标阈值。长时/较快刚体负对照的ppm级偏差另见图85，不能由短时单测代判。

日志只替换本机临时目录中的用户标识并移除行尾空白，数值和测试结果原样保留。没有新依赖安装、系统设置修改或大型下载；既有解释器与全部新计算/大缓存放D盘工作树，未测本轮峰值内存。C盘当前检出用于Git同步。

## 看图、文件与复算

- 专项画廊：`figures/design/v2/forward/u36_s4/index.html`，图81参考接触与来源、82固定材料几何/厚度、83带宽、84框架残差/缓冲、85负对照。
- 本轮29张控制标准图：`figures/stage1/u36_s4_reference/index.html`。
- 小报表：`data/worlds/design_v2/u36_s4_{material_band,control_verification,verification,manifest}.json`、`u36_s4_reference_{metrics,manifest,regions}.json`、`u36_s4_tests.txt`。
- 材料顶点、三角形、源UID、初始缺值、八帧几何/J/beta及可读回GPML网络：`.cache/u36_s4/`。大型状态不进Git，可按初始缓存和代码重算。

标准D盘项目环境中：

```powershell
scripts/run.ps1 python -u scripts/design_material_band_capture.py --reference-dir .cache/u35_c10/progressive
scripts/run.ps1 python -u scripts/design_material_band.py
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u36_s4/control/world.nc --stage 1 --out figures/stage1/u36_s4_reference
scripts/run.ps1 python -u scripts/design_material_band_report.py
scripts/run.ps1 python -m pytest tests/tectonics/test_material_band.py tests/tectonics/test_deformation_kinematics.py tests/tectonics/test_gap_diagnostics.py tests/tectonics/test_kinematics.py tests/tectonics/test_stretching.py tests/tectonics/test_engine.py -rA
scripts/run.ps1 python scripts/design_material_band_verify.py
```

控制前置缓存仍为U24初始世界及现有旋转；首次换机按C10交接重建控制参照。本机实际通过D盘工作树的忽略启动器复用已有解释器，未复制环境。`design_material_band_verify.py`读取本地pytest日志的实际结束汇总，不替代再次执行pytest。

## 方法依据与下一单元

2026-10-27实际核实官方：[pyGPlates安装与Windows/Linux支持](https://www.gplates.org/docs/pygplates/pygplates_getting_started)、[GPlates源仓库及GPLv2许可](https://github.com/GPlates/GPlates)、[TopologicalModel材料重建接口](https://www.gplates.org/docs/pygplates/generated/pygplates.TopologicalModel.html)、[变形网络伸展/沉降示例](https://www.gplates.org/docs/pygplates/sample-code/pygplates_reconstruct_crustal_thickness_and_tectonic_subsidence)、[应变插值参数](https://www.gplates.org/docs/pygplates/generated/pygplates.ResolveTopologyParameters.html)。官方当前文档/发行信息为1.0.0，本机实测亦1.0.0；Windows运行已经验证，Linux/WSL本轮没有实际运行。示例中的沉降接口未在本轮调用，不能据文档存在就宣称已求解沉降。

下一步先用**同一材料点处的速度梯度/散度时间积分**对照beta、局部面积与球面几何，隔离旋转文件插值、应变平滑与材料面积口径；并将原宿主按参考带精确裁切、明确初始空隙去向。未定位百分之一偏差前不接B5。接触未定的16条横切、带宽的流变约束、弯曲边界/释放阶区和时变接触仍是未完成项；本轮不关闭S3成因门槛、r2海岸、B5、齐高台面、重复岛形或M31端部等旧问题。

独立评审已保存到`reviews/stage1_2026-10-27_7.md`，实际逐张查看全部34图，本轮范围内保留2项严重：材料接触/物理带宽/框架标量尚未闭合；r2典型细岸仍缺成因证据。报告明确保留67通过/1失败，只建议方法检查点；接受全部结论，回应仅在原报告末尾追加，未关闭其他旧问题或宣布阶段完成。
