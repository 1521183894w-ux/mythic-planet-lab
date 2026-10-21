# spikes/

小规模技术验证（spike）脚本。每个子目录对应一个工作单元，脚本独立可运行，结论写在 `docs/` 中。

| 目录 | 工作单元 | 结论文档 |
|---|---|---|
| `stage0_tools/` | 0.4 科学工具快速验证 | `docs/tool_evaluation.md` |

运行：

```
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u spikes\stage0_tools\01_spherical_mesh.py
```

- 图写到 `figures/stage0/tools/`，数值结果写到 `figures/stage0/tools/results/*.json`，中间文件写到 `.cache/spikes/`。这些都不进 Git，重跑即可再生成。
- `06_hydrology.py` 依赖 `05_landscape_evolution.py` 的输出。
- spike 代码是一次性的验证代码，不是项目代码；正式实现放在 `packages/` 下。
