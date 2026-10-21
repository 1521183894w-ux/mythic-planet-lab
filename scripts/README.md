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
| `build_earth_reference.py` | ETOPO 2022 + Natural Earth -> 三级格网（高程、陆地比例、湖泊、大洲），约 30 秒 |
| `compute_earth_baselines.py` | 用指标代码计算地球基准，写 `data/earth_reference/baselines_*.json`，约 20 秒 |
| `build_region_atlas.py` | 地球区域参照图集（51 区，两联图），约 2 分钟 |

指标定义与地球基准值见 `docs/metrics_definitions.md`。

详细说明见 `docs/environment.md`。

## 编写约定

- PowerShell 脚本必须兼容 **Windows PowerShell 5.1**（两台电脑都没有 PowerShell 7）。
- `.ps1` 文件里如有中文，必须存为 **UTF-8 带 BOM**。PowerShell 5.1 会把不带 BOM 的文件按系统代码页（GBK）读取，中文会变成乱码，甚至导致语法错误。
- 运行方式：`powershell -ExecutionPolicy Bypass -File scripts\xxx.ps1`（只在本进程放开执行策略，不改系统设置）。
- 所有下载、缓存、临时文件写到工作区内的 `.cache/`，不写 C 盘。
