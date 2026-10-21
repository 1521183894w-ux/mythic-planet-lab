# Python 环境说明

> 工作单元 0.3。本文记录环境方案的选型理由、目录布局、使用方法和实测数据。
> 环境脚本：`scripts/setup_env.ps1`；规格：`environment.yml`；锁文件：`environment.win-64.lock.txt`。

---

## 1. 方案选型：micromamba + conda-forge

### 1.1 结论

用 **micromamba 2.9.0（单个 exe）+ conda-forge 频道**，Python **3.12**。环境、包缓存、临时文件全部在工作区内。

### 1.2 依据（2026-10-21 实测）

**uv（PyPI 路线）小实验**：用 uv 0.13.4 对任务书所需的 26 个包逐个做"只用预编译 wheel"的依赖解析（目标 Python 3.12、Windows x64，不安装）：

| 结果 | 包 |
|---|---|
| 有 Windows wheel（24 个，且可共同解析，闭包 99 个包） | numpy、scipy、pandas、xarray、zarr、netCDF4、rasterio、pyproj、shapely、geopandas、cartopy、matplotlib、scikit-image、pygplates、gplately、stripy、landlab、fastscape、pysheds、pyyaml、jinja2、pillow、pooch、pytest |
| **没有 wheel，无法安装** | **gdal**（Python 接口）、**richdem**（PyPI 最新版停在 2018 年，只有源码包） |
| **能装但不能用** | **fastscape**：PyPI 上的依赖清单里没有它的 Fortran 计算核心（fastscapelib-fortran 不在 PyPI），装完运行时缺核心 |

**micromamba（conda-forge 路线）**：同一组包（加上 fastscapelib-f2py）一次求解成功，351 个包、下载 648 MB，全部是最新版本。

**选择 conda-forge 的理由**：
1. 三个 PyPI 上缺失或残缺的包（gdal、richdem、fastscape 核心）在 conda-forge 上都有 Windows 版本。
2. rasterio、pyproj、cartopy、pygplates、gdal 在 conda-forge 上**共用同一套 GDAL 3.13 / PROJ 9.9 动态库和数据**；而 PyPI wheel 各自打包一份 GDAL/PROJ，版本和 proj.db 可能不一致，容易出现隐蔽的坐标系问题。
3. 以后阶段需要的 Fortran/C 编译型科学包，conda-forge 都提供预编译版本，台式机上不需要装编译器。
4. micromamba 是单个 exe，放在 `environment/bin/` 下即可，不需要系统级安装，不修改 PATH、注册表或用户配置。

**没选 uv 的代价**：conda 求解比 uv 慢（本次求解约 1 分钟），环境体积更大。对本项目可以接受。

### 1.3 Python 版本

所有必测包在 conda-forge win-64 上同时可用的区间是 3.12–3.13（下限由 zarr、rasterio、numpy 最新版决定，上限由 landlab 决定）。选 3.12：交集最宽、最成熟。

### 1.4 精简

- 用 `matplotlib-base` 而不是 `matplotlib`：后者会额外装 Qt6 + PySide6 交互界面（共 5 个包、约 100 MB），本项目只需要后台出图。
- 环境里仍有 Qt5（qt-main 5.15），由 pygplates 依赖带入。

---

## 2. 目录布局

| 位置 | 内容 | 进 Git |
|---|---|---|
| `environment.yml` | 环境规格（人工维护，只约束 Python 主版本和包名） | 是 |
| `environment.win-64.lock.txt` | 锁文件：每个包的下载地址 + md5（由脚本生成） | 是 |
| `environment/bin/micromamba.exe` | micromamba（脚本自动下载并校验 SHA256） | 否 |
| `environment/env/` | Python 环境本体 | 否 |
| `environment/mamba/` | micromamba 根目录（元数据、频道索引缓存） | 否 |
| `environment/setup_log.txt`、`env_report.json` | 重建日志、自检报告 | 否 |
| `.cache/mamba-pkgs/` | 包缓存（与环境之间是硬链接，可删除，删除后重建会重新下载） | 否 |
| `.cache/tmp/` 等 | 临时文件和各库缓存（见第 4 节） | 否 |

**删除整个环境**：直接删除 `environment\` 和 `.cache\` 两个目录即可，不留系统痕迹。

---

## 3. 使用方法

所有命令在 `D:\mythic-planet-lab` 下的 PowerShell 中运行。

| 场景 | 命令 |
|---|---|
| 新机器上首次建环境 | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` |
| 环境坏了，重建 | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Force` |
| 增删包后（改完 `environment.yml`） | `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -FromSpec -Force`，然后提交更新后的锁文件 |
| 在环境里运行一条命令 | `powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -m pytest` |
| 打开一个已激活环境的窗口 | `powershell -NoExit -ExecutionPolicy Bypass -File scripts\activate.ps1` |
| 只做自检 | `powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python scripts\check_env.py` |

说明：
- 两台电脑的执行策略都是默认的 Restricted，`-ExecutionPolicy Bypass` 只对这一个进程放开，不改系统设置。
- 默认按锁文件安装，**不重新求解**，所以两台电脑装出的包完全相同。只有 `-FromSpec` 会重新求解并改写锁文件。
- micromamba 运行时带 `--no-rc`，不读取用户目录下的 `.condarc`，也不受 `D:\Anaconda3` 影响。

---

## 4. 环境变量与缓存位置

`scripts/env_vars.ps1` 只在当前进程内设置以下变量，保证不写 C 盘：

| 变量 | 指向 | 原默认位置 |
|---|---|---|
| `MAMBA_ROOT_PREFIX` | `environment\mamba` | 用户目录下（C 盘） |
| `CONDA_PKGS_DIRS` | `.cache\mamba-pkgs` | 根目录下 `pkgs` |
| `TEMP` / `TMP` | `.cache\tmp` | `C:\Users\...\AppData\Local\Temp` |
| `PIP_CACHE_DIR` | `.cache\pip` | `%LOCALAPPDATA%\pip\cache` |
| `MPLCONFIGDIR` | `.cache\matplotlib` | `%USERPROFILE%\.matplotlib` |
| `NUMBA_CACHE_DIR` | `.cache\numba` | 包目录内 `__pycache__` |
| `PROJ_USER_WRITABLE_DIRECTORY` | `.cache\proj` | `%LOCALAPPDATA%\proj` |
| `XDG_DATA_HOME` | `.cache\xdg-data` | `%USERPROFILE%\.local\share`（cartopy 数据目录基于它） |
| `XDG_CACHE_HOME` | `.cache\xdg-cache` | — |
| `WIN_PD_OVERRIDE_LOCAL_APPDATA` | `.cache\platformdirs\local` | `%LOCALAPPDATA%`（platformdirs 的用户缓存目录；pooch、gplately 的下载缓存基于它） |
| `WIN_PD_OVERRIDE_APPDATA` | `.cache\platformdirs\roaming` | `%APPDATA%` |
| `MAMBA_USE_SHARDED_REPODATA=false` | — | 见第 5 节第 1 条 |
| `PYTHONUTF8=1` | — | 系统代码页 GBK，强制 Python 用 UTF-8 |
| `PYTHONNOUSERSITE=1` | — | 不加载 `%APPDATA%\Python` 下的用户级包 |
| `CONDA_PREFIX`、`GDAL_DATA`、`PROJ_DATA`、`GDAL_DRIVER_PATH`、`PATH` | 环境内对应目录 | 模拟 conda 激活 |
| `HTTPS_PROXY`、`HTTP_PROXY`、`NO_PROXY` | **仅在未设置时**取 Windows 系统代理（WinINET），如 `http://127.0.0.1:7890` | 不设置（micromamba 只认环境变量，不认系统代理） |

`check_env.py` 会逐项核对这些位置是否在工作区内，不在就判为失败。

关于代理：conda-forge 在两台机器上直连都不通。台式机把代理变量写在了用户环境里；笔记本没有写（只开了 FlClash 的系统代理），在用户自己打开的终端里运行 `setup_env.ps1` 会连不上 conda-forge。因此 `env_vars.ps1` 在没有 `HTTPS_PROXY` 时读取系统代理补上（2026-10-21 加入，已在清空变量的子进程里实测）。已有变量时不动。Python 的 `requests`（地球数据下载脚本）在没有环境变量时本来就会读系统代理。

---

## 5. 已发现的坑与处理（2026-10-21 实测）

1. **micromamba 2.9 的分片索引缓存写到 C 盘。** 首次求解后发现 `%LOCALAPPDATA%\conda\pkgs\cache\shards` 下多了 1621 个文件（10 MB）：分片索引（sharded repodata）的逐包缓存不遵守 `CONDA_PKGS_DIRS`。
   处理：设 `MAMBA_USE_SHARDED_REPODATA=false`，改用完整索引。实测关闭后 C 盘无新文件，求解出的 351 个包与开启时**完全相同**。代价：求解从约 1 分钟变为约 2 分钟，`.cache\mamba-pkgs\cache` 里多约 480 MB 的索引 JSON（可随时删除）。只有 `-FromSpec` 需要求解，按锁文件安装不下载索引。误写到 C 盘的那份缓存（当天新建的 `%LOCALAPPDATA%\conda\pkgs`）已删除；同级已有的 `conda\conda` 目录早于本项目，未动。
2. **gplately 一导入就在当前目录建 `gplately.log` 和 `ptt.log`。** 来自它自带的 `logging_config.yaml`，文件名写死、无配置项。
   处理：`.gitignore` 忽略 `*.log`；`check_env.py` 导入 gplately 时先切到 `.cache\logs`。以后项目代码导入 gplately 也要这样做（阶段 0.4 会封装成一个统一的导入函数）。
3. **gplately 的下载缓存默认在 `%LOCALAPPDATA%\gplately`。** 它用 `pooch.os_cache()`，底层是 platformdirs。
   处理：platformdirs 4.13 支持 `WIN_PD_OVERRIDE_*` 覆盖变量，已设置，自检确认指向工作区。
4. **micromamba 会在 `%USERPROFILE%\.conda\environments.txt` 里登记环境路径**（追加一行文字）。这个文件在本项目之前就存在（登记着 `D:\Anaconda3`），micromamba 没有关闭登记的选项。只有一行文字，不是项目文件也不是缓存，保留不处理。
5. **PowerShell 5.1 读取不带 BOM 的脚本会按 GBK 解码。** 所有 `.ps1` 存为 UTF-8 带 BOM，`tests/test_repo_conventions.py` 会检查。

---

## 6. 实测（台式机 HANS600452，2026-10-21）

| 场景 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| 首次按规格求解并安装 | `setup_env.ps1 -FromSpec` | 5.5 分钟（求解约 1 分钟 + 下载 648 MB + 安装） | 351 个包；生成锁文件；自检发现 gplately 缓存在 C 盘（已修复，见第 5 节） |
| **从零重建（冷启动）** | `setup_env.ps1 -Force -CleanCache` | **5.6 分钟**（删除旧环境和包缓存后，按锁文件重新下载并安装） | 自检 49 项全部通过 |
| 有缓存时重建 | `setup_env.ps1 -Force` | 0.5 分钟 | 自检 49 项全部通过 |

磁盘占用（`scripts/disk_usage.py`，硬链接只算一次）：

| 目录 | 文件数 | 真实占用 |
|---|---|---|
| `.cache\mamba-pkgs`（包缓存：压缩包 + 解压内容） | 69,707 | 3.73 GB |
| `environment\`（环境，绝大部分是包缓存的硬链接） | 65,000 | 另加 0.15 GB |
| **合计** | | **3.88 GB** |

- 删掉包缓存后，环境本身仍占 3.13 GB（硬链接的数据仍保留在环境中），下次重建需重新下载。
- 本阶段磁盘预算 10 GB，目前整个工作区约 4.0 GB。
- 重建过程中扫描了 C 盘用户目录：除 Claude Code 自身的日志外，没有新写入。
- 网络：走本机代理，conda-forge 下载约 3 MB/s（见环境审计报告）。

## 7. 实测（笔记本 Ezreal，2026-10-21）

新机器第一次安装（没有任何包缓存，等同冷启动），在 Claude Code 会话里运行：

| 场景 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| **新机器首次安装（冷启动）** | `setup_env.ps1` | **11.6 分钟**（19:17:49 开始，下载 micromamba 11 秒，按锁文件下载并安装 351 个包约 11 分钟） | 自检 50 项全部通过（比第 6 节多 1 项：0.5 新增的 `mythic.metrics` 导入检查）；`pytest` 40 项全部通过 |

- 比台式机（5.6 分钟）慢一倍，主要是下载：conda-forge 经代理时出现 4 次 "Download error (56) Failure when receiving data from the peer"，micromamba 自动重试后成功。安装日志里还有 micromamba 常规的"文件已存在"提示（例如 `xorg-libx11` 与 pip 的 `__pycache__` 重叠），不影响结果。
- 磁盘占用与台式机**完全相同**：`.cache\mamba-pkgs` 69,707 个文件 3.73 GB，`environment\` 另加 0.15 GB，合计 **3.88 GB**。
- 加上地球原始数据（466 MB）和图集后，整个工作区真实占用 4.40 GB（`disk_usage.py`，硬链接只算一次）。
- C 盘检查：扫描安装开始后 `%LOCALAPPDATA%`、`%APPDATA%`、`.conda`、`.cache`、`.local` 下的新写入，与项目有关的只有 `%USERPROFILE%\.conda\environments.txt`（38 字节，本机原先没有这个文件，是 micromamba 新建的，见第 5 节第 4 条）；其余都是 Edge、WPS、Claude 等应用自己的写入。
- 代理：本次运行时会话里已有 `HTTPS_PROXY`；用户自己开的终端里可能没有，`env_vars.ps1` 已加系统代理回退（第 4 节）。
