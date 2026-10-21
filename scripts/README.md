# scripts/

命令行入口：环境重建（`setup_env.ps1`）、出图（`make_figures`）、数据下载等。

## 环境相关脚本

| 脚本 | 用途 |
|---|---|
| `setup_env.ps1` | 一键重建 Python 环境（默认按锁文件；`-FromSpec` 重新求解并更新锁文件；`-Force` 覆盖重建；`-CleanCache` 冷启动） |
| `run.ps1` | 在环境中运行一条命令，例如 `scripts\run.ps1 python -m pytest` |
| `activate.ps1` | 打开一个已激活环境的 PowerShell 窗口，给人手动操作 |
| `env_vars.ps1` | 以上三个脚本共用的路径与缓存变量（不单独运行） |
| `check_env.py` | 环境自检：导入、GDAL/PROJ 冒烟测试、缓存位置检查 |
| `disk_usage.py` | 统计目录真实磁盘占用（硬链接只算一次） |

## 地球基准相关脚本（阶段 0.5）

| 脚本 | 用途 |
|---|---|
| `download_earth_data.py` | 按 `data/external_manifests/` 下载地球原始数据（约 490 MB），分段并行、断线续传、SHA256 校验；已有文件跳过 |
| `build_earth_reference.py` | ETOPO 2022 + Natural Earth -> 三级格网（高程、陆地比例、湖泊、大洲），约 30 秒 |
| `compute_earth_baselines.py` | 用指标代码计算地球基准，写 `data/earth_reference/baselines_*.json`，约 20 秒 |
| `build_region_atlas.py` | 地球区域参照图集（51 区，两联图），约 2 分钟 |

指标定义与地球基准值见 `docs/metrics_definitions.md`。

## 标准出图管线（阶段 0.6）

| 脚本 | 用途 |
|---|---|
| `make_figures.py` | 对一个世界生成标准图组（docs/review_protocol.md §2）：全球与极地高程、A 层陆块与主内海、半岛海湾、距海距离、坡度、山系、与同分辨率地球的对比图、8 张自动选取的区域放大图（各配同比例尺地球对照），以及 `metrics.json`、`regions.json`、`manifest.json` 和 `index.html` 画廊。地球 2048x1024 约 45 秒 |

```
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u scripts\make_figures.py                      # 地球
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -u scripts\make_figures.py --world 世界文件.nc    # 虚构世界
```

- 输出在 `figures/stage<N>/<world_id>/`（不进 Git，可重建）。用浏览器打开其中的 `index.html` 看图。
- 世界文件格式见 `packages/figures/world.py`：`elevation`（m）+ 可选 `land`，全局属性 `method_class_elevation` 必填。
- 对比用的地球格网必须与世界同分辨率（512x256 / 1024x512 / 2048x1024），新机器上需先建好地球格网（见上一节）。
- 8 张区域图的选取规则见 `packages/figures/select_regions.py` 开头的说明。

盲评用法（阶段 0.7）：`--blind "世界 A" --out figures/stage0/calib_world_A` 隐去名称、来源、种子与 method_class；`--mirror-shift-deg 100` 先把世界东西镜像再平移（保距变换，指标不变）。

## 尺子校准（阶段 0.7）

| 脚本 | 用途 |
|---|---|
| `make_rounded_world.py` | 生成"圆润世界"对照样本（故意做坏：光滑解析几何 + 噪声斑块山，能通过 A 层），写 `data/worlds/rounded_control/`（nc 不进 Git，`params.json` 进 Git），约 2 分钟 |
| `make_noise_world.py` | 生成第二个对照样本"噪声海岸世界"（圆润世界的大格局 + 多尺度分形噪声海岸，method_class visual_only；故意调参让 A 层和 B1 都通过），写 `data/worlds/noise_control/`（nc 不进 Git，`params.json` 进 Git），约 2 分钟。`--explore` 只试几组噪声参数 |
| `calibration_metrics.py` | 地球与两个对照样本的指标对比、B1 判定，写 `reviews/stage0_calibration_data.json`，约 1 分钟 |
| `coast_origin_diagnostics.py` | 探索性诊断（不是正式指标）：海岸复杂度的空间异质性、与纬度和地形的相关、远洋岛屿、大尺度半岛海湾指数、山系形态，比较地球与两个对照样本，写 `reviews/stage0_noise_diagnostics_<分辨率>.json`，2048x1024 约 40 秒 |

校准结论见 `reviews/stage0_calibration.md`。

## 奇景覆盖度核对（阶段 0.5b、0.5c）

| 脚本 | 用途 |
|---|---|
| `fetch_whc_sites.py` | 从 Wikidata 获取 UNESCO 世界遗产中符合标准 (vii)/(viii) 的遗产地，写 `data/earth_reference/whc_vii_viii.csv`，约 10 秒 |
| `fetch_iugs_sites.py` | 0.5c：从 IUGS 地质遗产委员会网站的公开接口获取地质遗产地名单（只取名称、分类、日期、链接），写 `data/earth_reference/iugs_geoheritage_sites.csv`，约 5 秒 |
| `archetype_coverage.py` | 校验两份清单 → 原型的归类（`whc_archetype_map.yaml`、`iugs_archetype_map.yaml`，矿床类型对照 `docs/mineral_resources.md`），生成 `docs/archetype_coverage.md` 的两个统计附录；`--check` 只校验 |

详细说明见 `docs/environment.md`。

## 编写约定

- PowerShell 脚本必须兼容 **Windows PowerShell 5.1**（两台电脑都没有 PowerShell 7）。
- `.ps1` 文件里如有中文，必须存为 **UTF-8 带 BOM**。PowerShell 5.1 会把不带 BOM 的文件按系统代码页（GBK）读取，中文会变成乱码，甚至导致语法错误。
- 运行方式：`powershell -ExecutionPolicy Bypass -File scripts\xxx.ps1`（只在本进程放开执行策略，不改系统设置）。
- 所有下载、缓存、临时文件写到工作区内的 `.cache/`，不写 C 盘。
