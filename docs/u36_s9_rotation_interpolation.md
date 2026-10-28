# U36-S9：投影点插值、来源连接与内部材料客观性候选

日期：2026-10-28。按用户“拉取最新，然后继续”，通过本机已有代理快进至 `77a58ca`，接续S8。本轮是局部方法检查点；默认C2，全球地壳与B5未替换，阶段1未通过。

## 本轮结论

**给定相同运动时，内部点不应因观察坐标系而改变变形。本轮给出了能满足这组对照的局部候选，但没有得到可用于完整网络输运的修复。**

1. 核对安装版pyGPlates1.0.0对应官方标签源码：其重心点映射先旋转邻近顶点，再投影至同一Lambert等积方位平面，插值平面位置后反投影。有限球面刚体运动在该平面上一般不是仿射运动，因而此操作不能严格保留共同刚体旋转。独立重现该分支，与真实API单步结果的最大差约1.02e−10 km。
2. 只把插值对象改为有限旋转还不够：首轮仍用每时刻新三角网，两级启动换框架差最高约5.64m，新增严格测试失败。不同观察框架会在近退化配置中选出不同连接，原结果和3失败/90通过日志均保留。
3. 将35Ma官方网格按边界来源身份固定为同一参考连接，再用同一LAEA重心权重、pyGPlates阶段旋转和SciPy SLERP，五类输入的内部采样材料在三框架还原后接近数值精度。参考连接是给定模型的一部分；没有根据结果挑一个框架。**该候选改变了给定插值场，不证明它是唯一或真实的岩石圈运动。**
4. 整网几何检查失败：边缘细长三角形与连续网络边界存在面积差，部分**投影三角形**方向转负；本次球面有符号三角面积没有负值，两种几何量不可混称真实材料翻转。所有原值保留。内部点支持通过不能代替整个网络或物质边界闭合。

## 同口径实际对照

共同刚体、两级启动、变向+走滑、沿带错动、真实C/D参考段；每类3锚定×2步长（0.25/0.125Myr），共30组，每组同时跑官方重心路径和固定连接旋转候选。另有相同30组动态连接首轮。共同旋转有独立精确刚体真值；其他输入主要验证坐标客观性，不以内部一致性当物理真值。

两侧运动仍由既有历史给定。35→0Ma共有141/281时点；合成13中心，真实432中心，每中心配4个1km环点。每步检查网内支撑、被采用投影三角形正向及没有多个严格内部覆盖；0Ma末态另核。无材料点因失败被删除。三框架均还原到0号坐标，比较全部采样时点和中心的最差值：

| 输入史 | 官方路径最大框架位置差/m | 固定连接旋转候选/m |
|---|---:|---:|
|共同刚体|4.455864|1.175e−8|
|两级启动|27.727768|9.202e−9|
|变向+走滑|19.505174|8.921e−9|
|沿带错动|9.433494|8.022e−9|
|真实C/D参考段|122.500678|2.605e−7|

真实段候选最大框架面积比差约1.441e−9；共同刚体对精确旋转的最大位置差约1.302e−8m、最大面积误差约6.738e−12。这里只说明本组内部点已没有先前量级的框架污染，不能把多位小数解释为真实地球精度。

## 没有被小框架差解决的事

- **时间和插值模型仍有代价。** 真实段anchor0的dt0.25→0.125末态中心位置最大差约4.427m，面积比最大相对差6.239e−5，只有两档，未证明时间收敛阶。候选与官方路径真实段的最大点位差约2.875km；dt0.125的末态432中心J相对差范围约−1.537%至+1.592%，中位+0.04995%。客观性改善不等于“候选J已被独立物理验证”，也不意味着这2.875km都可算作框架误差。
- **整网支持未成立。** 同一参考网格的投影方向检查在两级启动/变向案最多8个、真实段最多20个三角形转负；球面负面积为0。网格与连续边界的面积相对差最大约1.325e−5（0.001325%，含缓冲段）。官方网格本身按三角形中心归类的结果也另记。未删这些边界三角形、未改检查阈值，`full_network_geometry_test_passed=false`。本轮查询点都没有采用上述失效投影单元，不能推广为完整原参考带及边界输运。
- **S8的UID448261来源平台、原外圈库存未演化、古接触和物理带宽仍未修复。** 本轮没有在新映射上重新计算整域来源/柱厚度，也没有把S8旧映射的连续性误差搬来为新候选背书。
- **原框架beta和速度没有替换。** 91通过/2失败，原两项严格测试仍失败；新增9项全部通过。首轮新增参照系反例的断言未放宽，改为使用明确的固定来源连接后通过。
- 无断层基底位移、热沉降、沉积侵蚀、水位或新海岸；B5、r2细岸成因、齐高高原及阶段缺项仍保留。

## 图、数据与冻结世界

专项画廊：`figures/design/v2/forward/u36_s9/index.html`，四图102–105分别展示刚体真值、两种插值/两种连接的框架差、真实段J、整网边界失败。图104画在35Ma相同初始材料位置，是条件性面积比，不是现今高程/盆底。

标准图：`figures/stage1/u36_s9_reference/index.html`，实际35s重出29图。本机没有S4大缓存，复用S3冻结控制数组；S3/S4之前已各自与同一C10控制核对。本轮将S3 NC的全部物理数组原样复制，只给审计输入赋S4的world_id以固定r8抽查窗口，新增明确来源说明，保留真实S3代码版本。源与审计输入的不同文件SHA、数组逐项相同证据见`u36_s9_reference_source.json`；没有冒称S4的NC字节哈希。

既有指标除计时外、B判定、8个区域窗口均与S8相同：A7/7，主/极地B9/9，远方8/9。26项旧代码/配置/默认摘要Git规范内容相同，未改指标或正式引擎。测试原输出与首轮失败日志已消除本机用户路径后归档；数值、断言和堆栈未改。

小报表在`data/worlds/design_v2/u36_s9_*.json`，完整轨迹在D盘忽略目录`.cache/u36_s9/`。最终30组实际187.16s，进程峰值162.38MiB（pyGPlates1.0.0、SciPy1.18.1）；原始计时/进程峰值在`u36_s9_rotation_maps.json`；不是另一台机器性能认证。复用现有Windows CPU环境，无新求解器或依赖安装、系统变更及大数据下载。

## 复算与下一步

```powershell
scripts/run.ps1 python -u scripts/design_rotation_interpolation.py --connectivity-mode dynamic
scripts/run.ps1 python -u scripts/design_rotation_interpolation.py
scripts/run.ps1 python -u scripts/design_rotation_mesh_audit.py
scripts/run.ps1 python scripts/design_rotation_reference.py
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u36_s9/reference_world.nc --stage 1 --out figures/stage1/u36_s9_reference
scripts/run.ps1 python scripts/design_rotation_interpolation_report.py
scripts/run.ps1 python -m pytest tests/tectonics/test_rotation_interpolation.py tests/tectonics/test_step_audit.py tests/tectonics/test_material_audit.py tests/tectonics/test_source_clipping.py tests/tectonics/test_material_band.py tests/tectonics/test_deformation_kinematics.py tests/tectonics/test_gap_diagnostics.py tests/tectonics/test_kinematics.py tests/tectonics/test_stretching.py tests/tectonics/test_engine.py -rA
scripts/run.ps1 python scripts/design_rotation_interpolation_verify.py
```

验证脚本需本轮真实pytest输出保存到`.cache/u36_s9_tests.log`；首轮动态连接报告及失败日志也保留。本轮程序默认输出固定连接版本；`u36_s9_dynamic_rotation_maps.json`保存首轮证据，未把它改写成已通过。换机若无S3冻结NC，先按S3交接复算；真实网络从已归档35Ma参考端点和相同C/D旋转重新建立，不把新GPML身份冒称原S4文件字节相同。

方法分类：参考几何/200km带宽为`user_override`；两侧阶段旋转及两种点映射为`interpolated`；来源连接、客观性/几何检查为`procedural_constrained`；图为`visual_only`。审计1km环点及30km沿轨采样不提高原L7约60km的物理分辨率。单位km、m、Ma/Myr、无量纲J；不标为数值流变求解。

下一步应先在**同一固定内部材料域**上接入候选的原宿主裁切/来源与柱连续性，对照原S8的UID平台，并建立原外圈的有效边界处理；不要扩大到全球或依据B5目标高程调宽度。边缘投影支持、物理带宽、古接触与厚度到基底—热—沉积—水位的证据仍是接入门槛。

依据已实际核对的[pyGPlates1.0.0官方投影点映射源码](https://github.com/GPlates/GPlates/blob/27d48c91f24d7db9f6a212e7957d8c08064e5d9b/src/app-logic/ResolvedTriangulationNetwork.cc#L625)、[官方点映射接口](https://www.gplates.org/docs/pygplates/generated/pygplates.ResolvedTopologicalNetwork.html)、[SciPy SLERP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)。只读取对应源码，没有编译、修改或复制第三方源码进入项目代码。

独立评审结果另在`reviews/stage1_2026-10-28_4.md`，开发者只在原报告后追加回应。阶段1未通过。

独立评审已归档：33图和60份原始轨迹均实际核对，1项严重世界问题（r2典型细岸成因）保留，整网几何与物理接入另列阻断。仅建议局部方法检查点，不建议世界或阶段验收；原稿完整性见`u36_s9_review_integrity.json`，开发者仅追加回应。
