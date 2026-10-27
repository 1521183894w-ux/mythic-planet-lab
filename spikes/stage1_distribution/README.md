# U36-S1 地壳分配方法小试

合成固定网格上的方法验证，不能用作真实世界修复结果。详见 [交接说明](../../docs/u36_s1_distribution_spike.md)。

在项目根目录，用现有D盘环境运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python spikes/stage1_distribution/run.py
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python -m pytest tests/tectonics/test_distribution_spike.py tests/tectonics/test_engine.py
```

输出：

- 五张图和画廊：`figures/design/v2/distribution_spike/`。
- 参数、方法标签、单位、假设、参考文献、版本、主对照/敏感性/细化数据和保护哈希：`data/worlds/design_v2/distribution_spike/report.json`。
- 完整数组：同目录`arrays.npz`（可重算，不提交）。
- 独立评审：`reviews/stage1_2026-10-27_4.md`。

`study.py`复用Landlab LinearDiffuser和已有Engine原规则；`run.py`执行同输入对照并出图。需要已安装Landlab 2.10.1的项目环境，不下载新依赖。图件注明基底Git提交，报表额外记录本试验脚本SHA256，以识别未提交改动。

C11真实世界的参照标准图需本地已有C11持续候选NC：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/make_figures.py --world data/worlds/design_v2/candidates/u35_c11_progressive/design_v2_forward_v2_0Ma_u35_c11_progressive_1024x512.nc --stage 1 --out figures/stage1/u36_s1_reference_c11
```

输入NC不随本检查点打包；本机已实际运行，指标副本保存在`data/worlds/design_v2/distribution_spike/reference_metrics.json`。换机未保留NC时，先按C11交接复算候选，不能把新小试数组冒充全球世界。
