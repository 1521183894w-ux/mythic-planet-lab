# packages/

项目自己的 Python 包，按功能分目录，以 `mythic` 包名可编辑安装（`pyproject.toml`，`import mythic.metrics`）。

| 目录 | 内容 | 开始阶段 |
|---|---|---|
| `metrics/` | 质量指标与原型识别 | 0 |
| `figures/` | 标准图组：世界读取（`world.py`，世界文件格式见其文档）、区域自动选取、全球/极地/指标图、与地球对比图、区域放大图、画廊 | 0 |
| `tectonics/` | 板块构造模拟 | 1 |
| `terrain/` | 地形与侵蚀 | 2 |
| `hydrology/` | 河网与流域 | 2 |
| `climate/` | 气候 | 3 |
| `ocean/` | 海洋（含潮汐） | 3 |
| `astronomy/` | 轨道、自转、月亮 | 3 |
| `ecology/` | 生物群系 | 3 |
| `world_schema/` | 世界数据结构与存取（阶段 0 先用 `figures/world.py` 的最小格式：elevation + 可选 land + method_class 等属性） | 0–1 |

每个数据图层都要带 `method_class` 标签和方法说明，见 `CLAUDE.md` 第 4 节。
