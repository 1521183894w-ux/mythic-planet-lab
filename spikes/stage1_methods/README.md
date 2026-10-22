# 阶段 1.1 构造方案原型（spike）

两条候选路线各一个最小原型，在同一个情景、同一套地壳规则下对比，供用户选定方案（PROGRESS U9）。
结论与对比见 `docs/stage1_method_comparison.md`。

## 运行

```
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u spikes\stage1_methods\run_all.py --level 7 --seeds 0-7 --workers 4
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u spikes\stage1_methods\figures.py --level 7 --seeds 0-7
```

台式机上 16 个运行约 5 分钟，出图约 2 分钟。输出：
- 图：`figures/stage1/methods/01_setup.png` … `08_diagnostics.png`
- 每个运行的摘要：`figures/stage1/methods/results/run_<路线><种子>_L7.json`；判定汇总 `evals_L7.json`
- 路线 B 的旋转文件（PLATES4 格式，可用 GPlates 桌面软件打开）：`figures/stage1/methods/rotation_files/`
- 快照：`.cache/spikes/stage1/runs/*.npz`

以上都不进 Git，可按"种子 + 代码版本"重算。

## 文件

| 文件 | 内容 |
|---|---|
| `common.py` | 路径、球面几何、旋转、stripy 网格缓存、栅格化 |
| `engine.py` | 共用地壳引擎：板块参照系中的地块单元、俯冲（极性、海沟、岛弧/安第斯型增厚）、碰撞（体积拼贴）、洋壳生成与年龄—水深、造山带衰减与地壳流动、裂谷（洋脊—转换断层阶梯）、地体转移 |
| `scenario.py` | 共用情景：两块大陆（球冠拼成）、三块板块、基准运动与种子扰动 |
| `route_a.py` | 路线 A：板片拉力、碰撞后地体转移、泊松随机裂解 |
| `route_b.py` | 路线 B：运动学规划（只做刚体旋转，按 A 层主内海代码核对）→ PLATES4 旋转文件 → pyGPlates 逐步给出阶段旋转 |
| `evaluate.py` | 用阶段 0 的 A 层代码判定是否形成主内海（正式 / 宽松两档） |
| `runlog.py` | 板块速度、示踪点（古纬度历史）、快照 |
| `run_all.py`、`figures.py` | 批量运行、出图 |

## 表示方式的教训（写进阶段 1.2 的设计）

1. **不能定期把所有板块重新取样到固定网格。** 第一版每 5 Myr 这样做：位移小于半个网格间距的运动会被整个取整吞掉
   （L6 上低于约 1.2 cm/yr 的相对运动永远不动），碰撞后 15% 的残余汇聚因此被冻结；快速运动也有随机游走式漂移
   （100 Myr 后大陆质心偏 128 km、与精确旋转的重合度 IoU 0.81）。
   现在每块板块保留自己的参照格网和累计旋转、内部永不重新取样，刚体旋转 100 Myr 后与精确结果逐节点一致（IoU 1.0），
   慢速板块（0.05°/Myr）也正确移动。
2. **把一块地壳换到另一块板块的格网时要"收集"，不能"散射"。** 两套格网互相旋转，逐个把单元塞进最近节点会造成重复与空洞，
   空洞随后被填成洋壳，出现莫尔条纹。现在对目标格网的每个节点取覆盖它的源单元，体积前后比 0.999–1.000。
3. **裂谷阶梯要跟随总体走向**：每段都随机左右错开会变成锯齿。
