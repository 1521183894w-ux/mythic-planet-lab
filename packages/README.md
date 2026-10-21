# packages/

项目自己的 Python 包，按功能分目录。打包方式（`pyproject.toml` 等）在 0.3 确定 Python 环境后再定。

| 目录 | 内容 | 开始阶段 |
|---|---|---|
| `metrics/` | 质量指标与原型识别 | 0 |
| `tectonics/` | 板块构造模拟 | 1 |
| `terrain/` | 地形与侵蚀 | 2 |
| `hydrology/` | 河网与流域 | 2 |
| `climate/` | 气候 | 3 |
| `ocean/` | 海洋（含潮汐） | 3 |
| `astronomy/` | 轨道、自转、月亮 | 3 |
| `ecology/` | 生物群系 | 3 |
| `world_schema/` | 世界数据结构与存取 | 0–1 |

每个数据图层都要带 `method_class` 标签和方法说明，见 `CLAUDE.md` 第 4 节。
