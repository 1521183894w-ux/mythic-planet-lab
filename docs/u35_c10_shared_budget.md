# U35-C10：共享供体限额与B5反事实（2026-10-27）

接续`0145060`，是用户所要求连续两个行动单元中的第二单元。新增可选的共享供体总余额约束，完整运行限额基准、限额持续伸展和关闭限额的旧持续情景；不改β、时间、位置、刚体运动或质量指标。**限额候选消除了共享余额超扣，但没有形成B5谷地；旧持续候选接近−444m的局部低段对这项数值规则敏感。两个新候选均未采用，默认仍C2，U35与阶段1未通过。**

## 规则与控制变量

旧规则先逐洞按最近12个供体的34km以上柱体积提出请求。新规则汇总同一次提交的所有洞，按供体计算总请求；超过该供体同一份前态余额时，将所有请求同比缩小。它不按洞的处理顺序争抢、不向其他供体反复讨料，每个洞的获得量不超过原请求。未超额的请求保持原数值。

新增`Params.hole_borrow_shared_budget=False`，仅两个隔离配置设True；这是受过程约束的离散材料规则修正，不是新的流变或伸展求解器。34km是借料规则门槛，不是地球普适物理下限。供体不足时，仍按既有规则把缺额显式记作`hole_created`并填足规定新洞；因此必须同时报告新增量，不能把限额称为完整物理质量守恒，也不能把缺额当成求解出来的沉积或岩浆来源。

配置`history_v2_u35_c10_baseline.yaml`与C4、`history_v2_u35_c10_progressive.yaml`与C8持续候选语义上各只相差这一个布尔选项。后者仍为35→0Ma、总β1.5、半宽40km、相同轴线及对数权重的`user_override`。两个完整候选40Ma及以前各297个保存数组与各自旧情景完全一致。

## 数值约束修复与地貌结果不能混为一谈

完整300→0Ma的供体超额分配，在两个限额候选中均降到浮点舍入量：合计小于4.5×10⁻¹¹km³、最严重单体小于2.1×10⁻¹⁵km等效厚度，没有超过1e−6km³的超额提交。逐洞记录与实际库存变化的重建残差小于3×10⁻¹¹km³，21个原始追踪宿主的库存闭合残差小于7.3×10⁻¹¹km³。

| 同一B5设计多边形，原生查询 | 旧基准 | 限额基准 | 旧持续 | 限额持续 |
|---|---:|---:|---:|---:|
| 定位中心高程 m | 1487.94 | 1594.10 | 1040.03 | 1024.11 |
| 多边形中位高程 m | 1487.94 | 1594.10 | 866.55 | 857.35 |
| 多边形最低高程 m | −211.43 | −211.43 | −444.02 | −219.02 |
| 低于海面的覆盖面积比例 | 0.5651% | 0.5651% | 7.5690% | 0.5651% |
| 全时段显式补洞新增体积 km³ | 259284.51 | 461201.45 | 421424.50 | 624731.14 |

约30080.93km²的实际设计轮廓在四案中均100%原生覆盖；灰色周边空隙继续保持未知，不跨接为海或陆。限额持续比旧持续中心低15.92m、中位低9.19m，但设计范围内低于海面的面积大幅减少；因此不能用定位点略降宣布谷形整体改善。新增沿设计轴线与两端各60km的纵剖面：限额持续在轴线目标段查询高程最低约456.85m、中位约863.34m，仍主要是高地和台阶，未展示连续低谷。图中的−400m仅为谷底目标，不要求整块多边形都铺平到该高度。

旧规则最严重供体的位置已在关闭限额的兼容运行中记录：35Ma最大者UID40911、来源/所属301，换到洲C现今参照系约18.954°E、22.291°S；10Ma最大者UID458680、来源91、所属301，约18.793°E、2.977°S，位于B5附近。后者与旧持续低段对照说明目标低点对共享借料约束敏感；并不证明所有低地都由该缺陷造成。5/0Ma最严重宿主位置与来源也保存在JSON，不按新增UID编号跨候选强行对齐。

## 全局代价与保留的骨架

与各自同伸展的旧借料情景比较，而非混用C6原生输出：

| 实际变化 | 限额基准−旧基准 | 限额持续−旧持续 |
|---|---:|---:|
| 最大绝对高程差 m | 1413.20 | 1413.20 |
| 绝对高程差>1m面积 km² | 1228661 | 1189011 |
| 其中B5窗外面积 km² | 1034793 | 1008842 |
| 海陆掩膜翻转面积 km² | 16018.33 | 19070.01 |
| 显式补洞新增体积的增加 km³ | 201916.95 | 203306.63 |
| M31窗最大高程差 m | 0 | 0 |

变化沿C/D交界周围分布，包含B5以外的南方段；不是全球到处均匀受损。图65用黑线标示>1m足迹，不能只凭固定色带看起来浅就忽略；这个1m是描述阈值，没有预设成严重破坏或代价可接受线。跨时间变化足迹、现存体积、新增体积和旧规则累计超额量是不同概念，不能混用。

两新案A层均7/7，主大陆/极地B项各9/9，远方大陆8/9，海湾项0.0128512504仍低于0.0134521341。主内海约446万km²、3亚大陆等骨架保留。西口4/8邻接连通，阻断西口后均不连通；移除年轻地壳仍断，这个成因反事实未解决。M31、高原台面、重复岛形和北岸具体湾岬成因没有因本轮数值约束自动修复。

## 实际运行、材料与复现

- 三次完整运行：限额基准225.62s、限额持续228.43s、旧持续兼容228.46s。关闭新选项后，352个保存数组、预算、事件、原生字段、存活掩膜、旋转与世界数组均与C9持续情景相同。
- **64项相关测试通过，9.95s**，覆盖共享供体约束、洞顺序无关、无短缺完全不变、同一洞重复供体和非法输入，以及原有构造/出图相关验证。未声称全套测试通过，未测本轮内存峰值，未安装新依赖。
- 五张新对照图61–65与画廊：`figures/design/v2/forward/u35_c10/index.html`；两新案各29张完整标准图，生成分别约32/33s，均退出0。
- 数值与来源：`data/worlds/design_v2/u35_c10_{diagnostic,provenance}.json`；两个候选摘要另存。完整状态、NC、逐边/逐洞账本与旧持续兼容输出在`.cache/u35_c10/`，不将大型数据提交Git。
- 原正式历史、终图、C4配置、指标、旋转与默认摘要6文件仍保持C8/C9保护哈希，未替换默认世界。

```powershell
scripts/run.ps1 python -u scripts/design_b5_shared_budget.py --case baseline --history-config config/design/history_v2_u35_c10_baseline.yaml
scripts/run.ps1 python -u scripts/design_b5_shared_budget.py --case progressive --history-config config/design/history_v2_u35_c10_progressive.yaml
scripts/run.ps1 python -u scripts/design_b5_shared_budget.py --case legacy --history-config config/design/history_v2_u35_c8_b5_progressive.yaml
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u35_c10/baseline/world.nc --stage 1 --out figures/stage1/design_v2_forward_v2_0Ma_u35_c10_baseline
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u35_c10/progressive/world.nc --stage 1 --out figures/stage1/design_v2_forward_v2_0Ma_u35_c10_progressive
scripts/run.ps1 python -u scripts/design_b5_shared_budget_report.py
scripts/run.ps1 python -m pytest tests/figures tests/tectonics/test_reactivation.py tests/tectonics/test_design_forward.py tests/tectonics/test_west_arc_coverage.py tests/tectonics/test_sampling.py tests/tectonics/test_sections.py tests/tectonics/test_rifts.py tests/tectonics/test_b5_ledger.py tests/tectonics/test_b5_coupling.py tests/tectonics/test_borrowing.py
```

本机仍用D盘既有解释器，经`.cache/local_run.ps1`从当前检出加载代码。限额是候选对照，默认False；后续要处理补洞材料来源与体积/面积表示，不能依靠旧借穿产生的低点验收谷地，也不能只靠加大β完成目标。

## 独立评审回应

`reviews/stage1_2026-10-27_2.md`已保存。world-reviewer实际逐张检查63张新PNG，并复看增加变化边界后的图65，建议作为方法修复与形态代价的比较检查点提交。报告判定共享供体重复透支在这两个候选审计范围内已处理，仍有五项严重形态/成因问题：B5、高原台面、北岸具体湾岬成因、重复岛形、M31拼接与端部；显式新增来源与传播足迹另列为能力边界。接受全部结论。

这不代表默认世界已经修复共享透支；默认选项仍False，旧世界及候选的审计范围必须分开。后续应保留供体共享余额约束，处理补洞材料来源、实际面积与覆盖表示，再检验沿轴连续谷底、边坡和两端过渡；不能为保住旧低点而恢复超扣。没有用户看图通过记录，不采用最终B5地貌，不关闭U35或阶段1。
