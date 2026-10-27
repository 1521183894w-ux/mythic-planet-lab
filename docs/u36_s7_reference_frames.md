# U36-S7：共同刚体与参照系客观性（2026-10-28）

接续`2390e92`，这是用户睡前授权的第一个追加行动单元。**共同刚体伪变形可在随动坐标中消失，但一般伸展/变向运动变回同一物理框架后仍不一致；本轮确认的是数值参照系依赖，没有选一个“更好看”的框架替换全球求解。**

## 实际对照

同一初始物理点、旋转历史和网络，测试共同刚体、两级启动、变向+走滑、真实C/D参考段。每种各3个锚定框架、自然邻域/重心两种框架现有插值、dt0.25/0.125Myr，共48案；结果全部变回0号框架比较。控制13材料中心，真实段432中心，1km局部面积斑块，35→0Ma。锚定板块ID是分类，不是连续物理参数。

- 共同刚体，dt0.125：anchor0点映射对精确旋转最大位置差约5.509m、最大面积伪变形约0.014759%；anchor1/2的还原位置差约3.2e−11/1.0e−10km，J约1。两种插值均如此。随动框架把共同运动变成零运动，这个特例不验证一般变形。
- 两级启动变回同框架后，相对anchor0最大位置差约24.11m；变向/走滑约26.54m；真实段anchor301/401约61.79/134.05m。真实段最大面积比差约0.00528%/0.02368%。这些不是实际地壳位移或独立真实减薄。
- 时间步0.25→0.125Myr均未消除参照系差；本轮48行配对中两种插值的六个数值汇总字段逐项相同，只说明这些样本未测出差异，不证明一般插值方法等价。真实段框架β/J的加权对数差×100约−1.299（anchor0）/−1.246（301/401），时间窗口旧问题仍在；不能将0.024%级差称为1.3%问题的全部解释。

## 独立评审要求后补跑的输入等价检查

评审指出仅说“同一旋转文件”不足以在禁读源码条件下独立确认同一物理输入，已补跑并归档`u36_s7_input_equivalence.json`。12组输入的初始点坐标往返最大约1.62e−12km；35/34/20/0Ma边界变回0框架后的双向点集差最大约3.19e−7km，边界源feature ID集合全部相同。0—50Ma每1Myr核对总旋转还原，矩阵最大分量差约7.95e−10，同时保存两板块相对旋转矩阵核对。核对只覆盖列出的4几何时点和51旋转时点，不声称检查了所有连续时刻。

## 检查与范围

48案实际78.70s；32张图（专项96–98+标准29）已生成。标准图实际31s，复用S4冻结NC，本轮未重跑全球演化；除生成时间/计时外，指标及B判定同S6：A7/7、主/极地B9/9、远方B8/9。相关检查实际82通过/2失败，17.58s，原两项严格失败保留；21项保护内容与基点相同，没有修改原指标、框架β、引擎或默认C2。

画廊：`figures/design/v2/forward/u36_s7/index.html`，标准图`figures/stage1/u36_s7_reference/index.html`。原始数据`u36_s7_frames.json`、`u36_s7_input_equivalence.json`、`u36_s7_tests.txt`、`u36_s7_verification.json`，完整数组`.cache/u36_s7/frames.npz`不提交。图96使用的6条共同刚体序列另存`u36_s7_timelines.json`，每条281个实际框架输出时点，非端点连线；这是按独立评审要求补齐的可入Git时序证据。

方法标签：给定参考几何`user_override`，框架场`interpolated`，参照系/面积审计`procedural_constrained`；单位km/Myr、km、无量纲J。复用pyGPlates1.0.0，官方[TopologicalModel锚定框架说明](https://www.gplates.org/docs/pygplates/generated/pygplates.TopologicalModel.html)。没有新依赖、系统设置或大型下载，所有新文件/计算在D盘；本轮未测内存峰值。

```powershell
scripts/run.ps1 python -u scripts/design_frame_audit.py
scripts/run.ps1 python -u scripts/design_frame_inputs.py
scripts/run.ps1 python -u scripts/design_frame_audit.py --report-only
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u36_s4/control/world.nc --stage 1 --out figures/stage1/u36_s7_reference
scripts/run.ps1 python scripts/verify_local_checkpoint.py --unit u36_s7 --special-count 3
```

验证脚本需`.cache/u36_s7_tests.log`，相关pytest命令同S6且本轮已实际重跑。控制历史复用S6三种旋转构造，换机需先重建S4输入。仅为明确给定运动下的数值误差，不是地质唯一历史、流变解或B5高程。

下一单元按用户授权，固定同一内部材料域做来源与连续性空间对照，保留原外圈库存；不靠不断剔除更多材料把数字变小。参照系误差、物理带宽/古接触、边界成因和r2典型细岸另留证据门槛。阶段1仍未通过。独立报告`reviews/stage1_2026-10-28_2.md`已逐张实看32图并核对哈希，补证输入等价/281时序后仍保留2严重（真实变形到地貌成因、r2细岸），只建议方法检查点；原稿21028字节保持，开发者仅追加回应，证明见`u36_s7_review_integrity.json`。
