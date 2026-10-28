# U36-S10：同一归档旋转映射的固定来源域与柱连续性

日期：2026-10-28。按用户“拉取最新，然后继续”，通过本机已有系统代理成功拉取，基线为 `1336b4a`。本轮为阶段1局部方法检查点；默认C2，未接全球/B5，阶段1未通过。

## 结论与输入身份

**S9归档候选可以跟踪S8同一固定内域全部审计点，但没有改善同档柱连续性求积精度。旧UID448261平台的一个根因已定位：三档最长边阈值均未细分到这个小来源。外圈仍无法完整演化。**

完整300→0Ma旁观控制实际复跑284.66s，352数组、预算、事件、最终原生状态及世界数组均与S3控制一致。重建35Ma原宿主裁切，内域WKB哈希与S8相同，全部UID初始多边形体积差为0；四种对应细分的初始三角形来源体积也与S8相同。控制世界数组相同不能代替局部材料候选有效。

首轮本机重新选35Ma连接，与S9归档有48个三角形对称差。首轮六个内域案、外圈及八个单源案的JSON、NPZ、日志全部保留在 `u36_s10_rebuilt_*.json` 和 `.cache/u36_s10/rebuilt_connectivity/`。最终读取S9归档来源连接，仅把名称前缀对应到S10，全部重跑；没有按体积误差挑连接。具体浮点/退化择边机制本轮未继续定位。

加速只用Shapely STRtree筛包围盒，沿用原LAEA重心权重、1e−9接受条件和网内/正向/无严格内部重叠检查，不向外延拓。4项新增测试核对4096内部点、全部三角形重心的接受/拒绝等价、越界及重叠拒绝。另与S9同一432中心、1km微斑块及141时点原始轨迹核对：最大中心位置差7.6263e−12km，J相对差1.0686e−9。只确认同一候选与实现等价，不代表物理真值。

## 同一固定内域的实际结果

下表均为最终S9归档连接。200km假设带宽、S8原内缩2km审计域不变。全部材料顶点、中心、环点保留；每个141/281离散时点及末态检查支持，不动态剔除。非陆壳/无覆盖仍为缺值。

H=H0/微斑块J，体积独立使用有限来源三角形球面面积；没有除以同一有限单元面积比来强迫闭合。

| 最大边km / 斑块km / dtMyr / anchor | 末态净体积差% | 来源加权绝对差% | 全时序最差UID / 误差% |
|---|---:|---:|---|
| 30 / 1 / .25 / 0 | −0.00679859 | 0.03345517 | 71754 / −0.2550574 |
| 15 / 1 / .25 / 0 | −0.00202805 | 0.00718260 | 448261 / +0.0944502 |
| 7.5 / 1 / .25 / 0 | −0.00043713 | 0.00167359 | 448261 / +0.0944502 |
| 7.5 / .25 / .25 / 0 | −0.00060055 | 0.00201297 | 79872 / −0.1933015 |
| 15 / 1 / .125 / 0 | −0.00202793 | 0.00720019 | 448261 / +0.0944072 |
| 15 / 1 / .25 / 301 | −0.00202805 | 0.00718260 | 448261 / +0.0944502 |

S8旧路径同档7.5km/1km的净差约−0.00006151%、来源绝对差0.000917%。**客观性改善没有自动提高柱连续性求积精度。** 微斑块缩至0.25km反而让部分来源误差变大，不能声称全来源均收敛。

15km案步长减半，所有查询点末态最大位置差4.5399m、厚度最大相对差0.007681%；只有两档，不证明时间收敛阶。锚定0→301还原后，末态最大位置差4.5248e−8m、厚度最大相对差6.3636e−10。小框架差不能证明插值场是真实岩石圈运动。

六个内域案合计507.82s，材料审计进程峰值515.08MiB；首轮计时另存。完整旁观控制未保存其进程峰值，不能把515.08MiB当整个工作单元峰值；本机计时不认证另一台电脑。

## UID448261平台的有限单元定位

该来源约60.069km³，原三角形最长边3.336km。30/15/7.5km阈值均留下同一个三角形，因此三档误差相同不代表已完成该来源的细分收敛。

保持同一来源多边形，明确细分成1/4/16/64个单元，以0.25/0.0625km微斑块跑八个候选案；S8原API另独立重算九个末态。原7.5km/1km案与S8归档误差差约−3.43e−11，确认复建一致。以0.25km微斑块为例：

| 同一来源三角形数 | 新候选末态体积差% | S8旧API末态体积差% |
|---|---:|---:|
| 1 | +0.0944381 | +0.0287200 |
| 4 | +0.0236092 | +0.0071806 |
| 16 | +0.0059028 | +0.0017949 |
| 64 | +0.0014764 | +0.0004489 |

这支持旧平台至少有明确的“有限材料单元求积未细化”成分；不能推广为全域尤其UID79872均收敛。几百米是审计细分，物理输入仍是L7约60km。没有逐UID归一化或事后补体积。

## 原外圈实际尝试失败

起始35Ma已有91个查询在网络外、125个无受支持投影三角形，两类可重叠，不能相加当失效材料数。触及的已知有限单元初始体积约78148.609km³，只是审计影响量，不是已经损失的材料。

外圈约143259.782km³已知初始库存继续按S8逐UID有符号差保留，含微小负浮点裁切账差，不静默归零。未删除失败点、外推、生成新壳或宣称外圈已演化。S9整网投影方向/边界覆盖阻断保留；本轮未建立有效的完整边界输运。

## 图、测试和范围

- 专题画廊：`figures/design/v2/forward/u36_s10/index.html`。106同域连续性、107条件性柱厚、108来源、109外圈、110步长/参照系、111单来源细分。
- 标准图：`figures/stage1/u36_s10_reference/index.html`，实际34s重出29图。输入来自本轮完整旁观控制，只用S4审计ID固定r8窗口，实际来源代码/哈希保留，物理数组同旧控制。指标及窗口同S9：A7/7、主/极地B9/9、远方B8/9。
- 实际测试95通过/2失败，15.92s；两项旧共同刚体严格失败原样保留，未改断言。30项正式代码/配置/默认摘要Git规范内容与1336b4a相同。
- 保存35/20/0Ma全部材料点、微斑块、连接、H0及UID，独立重算J/H/体积。35图哈希及核对见`u36_s10_verification.json`；测试原输出见`u36_s10_tests.txt`。
- 独立评审已归档`reviews/stage1_2026-10-28_5.md`。r2细岸、齐高高地、物理古接触/带宽及阶段缺项未因此关闭。

## 复算入口

先从项目根目录使用既有环境，无新依赖安装：

```powershell
scripts/run.ps1 python scripts/design_rotation_sources_prepare.py
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 30
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 15
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 7.5
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 7.5 --radius .25
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 15 --dt .125
scripts/run.ps1 python scripts/design_rotation_continuity.py --edge 15 --anchor 301
scripts/run.ps1 python scripts/design_rotation_continuity.py --outer
foreach ($r in @(.25, .0625)) {
  foreach ($e in @(7.5, 1.875, .9375, .46875)) {
    scripts/run.ps1 python scripts/design_rotation_continuity.py --uid 448261 --edge $e --radius $r
  }
}
scripts/run.ps1 python scripts/design_rotation_continuity_support.py
scripts/run.ps1 python scripts/design_rotation_continuity_report.py
```

准备脚本需S3控制缓存，缺失先按S3交接复算。support脚本的parity部分需S9真实轨迹`.cache/u36_s9/real_0_0.25.npz`，缺失只能先跑`--part legacy`，不得冒称完成旧轨迹对照；本机实际存在并已核对。标准图固定窗口处理见`u36_s10_reference_source.json`，不能换随机ID后宣称同窗。测试输出保存到`.cache/u36_s10_tests.log`后运行verify脚本。

标签：固定参考域/200km宽度为`user_override`；阶段旋转、LAEA权重与SLERP映射为`interpolated`；裁切、来源求积、索引与审计为`procedural_constrained`；图为`visual_only`。pyGPlates1.0.0、SciPy1.18.1、Shapely2.2.0/GEOS3.14.1；单位km、km²、km³、Ma/Myr。继承S9两板块、无内部刚块限制，不标为数值物理。

依据沿用S9已核实的[官方点映射源码](https://github.com/GPlates/GPlates/blob/27d48c91f24d7db9f6a212e7957d8c08064e5d9b/src/app-logic/ResolvedTriangulationNetwork.cc#L625)、[pyGPlates接口](https://www.gplates.org/docs/pygplates/generated/pygplates.ResolvedTopologicalNetwork.html)、[SciPy SLERP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)。未选用/安装新求解器。

下一单元建议先研究保留完整材料边界、不向外延拓的支撑表示，再按各来源真正细化有限单元求积，检查UID79872的微斑块敏感性。不能只缩小整体最长边阈值就宣布每个小来源收敛。物理带宽、古接触及厚度到基底—热—沉积—水位证据仍是全球/B5接入门槛。

原始NPZ布局见 data/worlds/design_v2/u36_s10_npz_schema.json。特别注意：patch_triangles使用微斑块局部索引，source_index不是uids索引；逐来源复算应使用source_uid与known归组。

独立评审结论：实际查看35图并完成所列原始数组独立复算，保留3项严重（完整外圈未演化、平顶/内部平滑地形、远方大陆B1不达标），仅建议局部方法检查点，不建议世界/阶段验收。原文已逐字保存，开发者仅追加回应；完整性见u36_s10_review_integrity.json。旧r2细岸问题未因此关闭。
