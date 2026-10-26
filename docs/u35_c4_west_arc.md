# U35-C4 西口山弧候选与换机交接（2026-10-26）

本次按用户“继续推进”实施；用户随后要求“我要换设备了，提交并推送”，因此在当前结果处交接。**这是已提交独立评审的修正候选，尚未采用为默认世界，U35 和阶段 1 均未通过。** 基线为 `209c5ff`（C3；正式地形仍是 C2）。

## 实际改动与结果

新增独立配置 `config/design/history_v2_u35_c4_strain.yaml`，沿用 U33 的板内纯剪切近似，只为 M31 原 ALB 宿主来源域增加 10–0 Ma 的规定缩短：最大增厚比 1.25，对应理想纯剪切下 20% 水平收缩。新增可选来源域过滤，防止新转入的其他宿主单元一起被加厚。默认配置、已纠正运动、终图和指标阈值均未修改。

完整正向演化已真实运行。M31 同窗口 p90：C2 **982 m → 候选 2092 m**，认可终图栅格为 2073 m（设计名义高度 2800 m）。原有 29 个 ALB 来源单元的 UID、节点、板块、经纬度、面积完全一致；10 Ma 及以前的 330 个保存数组一致。海陆掩膜变化为 0 km²，西口外窗口（−26–6°E、30–44°N 之外）最大高程差 0 m；全球最大高程差 1371.79 m，变化超过 150 m 的面积 66581.04 km²。普通 4/8 连通均畅通；把小于 10 Myr 洋壳视作陆地仍断开。A 层 7 项通过；远方大陆海湾指标仍 0.012851 < 0.013452，不能验收。

只缩窄实际碰撞体积扩散核、并恢复晚期 WT 碰撞的另一候选，完整运行仅到 1462 m，且增加海岸坑洞，已否决。其配置及试验脚本留在本机 .cache/u35_c4，不作为有效配置提交。不能为补山高恢复错误位置或错误接触。

## 科学含义与限制

本候选是倒推的非唯一构造历史参数，**应变没有由刚体速度自动求出**。原 M31 带最终宽约 130 km；162.5 km 收至 130 km 对应 32.5 km 缩短，只是选参尺度核对，不是模型测得的缩短。北向约 100 km 再靠近也不能独立验证这 32.5 km。

模型单元面积保持不变，增厚以外加体积来源项近似；本次新增规定缩短来源 **478091.9978 km³**，实际碰撞输入仍是 **223039.6576 km³**，两者分开记账。不能声称重分配了碰撞输入或实现连续体质量守恒。来源域代表宿主来源，不代表单元内所有混合岩石的成分。

运动和终图、规定的缩短历史为 `user_override`；构造演化为 `procedural_constrained`；Airy/GDH1 等参数化高程为 `parameterized_physics`；输出栅格为 `interpolated`。L7、约 55 km 节点尺度，输出 1024×512；厚度 km，高程 m，体积 km³，时间 Ma。未求解河流侵蚀、精细断层应变或连续地幔动力学。高度改善不等于山弧已经自然：沿认可山脊仍有长平顶和东端骤落，R17 的实体表达也未补齐。

地球参照只支持同时存在拉张与挤压、以及晚期反转的可能，不给出本候选 0.25 的数值依据：[Martínez-García 等（2017），阿尔沃兰海伸展转挤压](https://doi.org/10.1002/2017TC004489)；[Platt 等（2003），弧形造山带运动学](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001TC001321)。

## 图、测试与评审

- `figures/design/v2/forward/u35_c4_strain/index.html`：32 三方山弧图、33 最后 10 Myr 厚度/体积与沿脊曲线、34 全球差异。
- `figures/stage1/design_v2_forward_v2_0Ma_u35_c4_strain/index.html`：29 张标准图、指标和区域构造层。
- `data/worlds/design_v2/u35_c4_diagnostic.json`：数值证据；`u35_c4_provenance.json`：命令、资源、哈希、无干扰复算。
- `reviews/stage1_2026-10-26_4.md`：独立 world-reviewer 原文。评审只看简报、图和指标，开发者回应见下方。
- 实际运行 61 项相关测试全部通过（18.73 s，峰值 RSS 0.204 GB）。完整观测运行 298.76 s，峰值 RSS 0.9 GB；标准图组 38.49 s。全套前史已知失败沿用 C3 记录，本轮没有声称全套通过。
- 图表脚本第一次经监视器输出时遇到 GBK 控制台编码错误；直接将输出记录到 D 盘日志后重跑，真实退出码 0。

自动审批曾拒绝覆盖共用正式世界及图件，理由是尚未验证并认可的候选不应直接提升为正式结果。已改用 `--candidate` 将缓存、世界、摘要与图目录全部隔离，且禁止候选模式写共用 shelf_active。默认输出仍保留 C2。本次提交同步独立候选，采用与否等用户看图；没有静默提升。

## 换机后从哪里接

先拉取最新、读本文件和独立评审；已提交的图、指标和配置可直接查看。环境、.cache、.nc 不进入 Git，新机器需要复算时运行（所有命令在项目根目录，Windows PowerShell）：

```powershell
# 先按项目既有 setup_env 流程确认本机环境，再生成默认运动及初始世界缓存
scripts/run.ps1 python -u scripts/design_history.py --design v2 --history v2
scripts/run.ps1 python -u scripts/design_initial_world.py --design v2 --history v2
# 基线也独立运行，供对照脚本读取，避免覆盖共用结果
scripts/run.ps1 python -u scripts/design_west_arc_diagnostic.py --case latest --output-root .cache/u35_c2 --candidate u35_c4_baseline --no-figures
scripts/run.ps1 python -u scripts/design_west_arc_diagnostic.py --case latest --output-root .cache/u35_c4_strain --history-config config/design/history_v2_u35_c4_strain.yaml --candidate u35_c4_strain --no-figures
scripts/run.ps1 python -u scripts/design_west_arc_c4_report.py
scripts/run.ps1 python -u scripts/make_figures.py --world .cache/u35_c4_strain/latest/world.nc --stage 1 --out figures/stage1/design_v2_forward_v2_0Ma_u35_c4_strain
```

接续重点：先由用户判断当前山弧改善是否可接受；继续处理沿脊平顶和东端落海、关键年轻填隙单元的实体地壳表达。不得把 p90 接近终图当成山系连续性通过，也不得把洋龄估计当作真实海底扩张证据。本轮按用户换机指令停下。

## 独立评审回应

独立评审保留5项严重问题，建议独立候选提交用户审看，不建议U35收尾。逐项回应：

1. #1：接受山弧高度已有改善、形态仍未完成。长平顶与东端落海保留为下一轮工作，不以p90关闭。
2. #2：东方山盆同高台面是既有问题，本轮窗口外变化0，未处理。
3. #3：r2湾岬的局部伸展与断块证据仍缺，本轮未改其地形，不用复杂度数字替代成因。
4. #4：东南岛列形态层次仍不足，本轮未处理。
5. #5：B5洼地仍约1688 m，本轮未处理。
6. #6一般问题：普通海口畅通并非已封闭；关键年轻洋壳成因、R17实体仍未闭合。

候选world_id改变使r8抽样换窗，评审已按新窗记录，不能直接沿用C2坐标。用户提前换机，评审未逐张复核标准图09–11、13，最后重绘图33亦未复核；不声称完整全球验收。

最终无观测独立完整复算已结束：306.77 s，峰值RSS 0.849 GB；与观测候选352个保存数组及预算一致，10个世界数据数组一致。5个共用默认文件SHA256保持不变。详见来源JSON。
