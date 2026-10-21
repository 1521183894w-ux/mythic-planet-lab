# 环境审计报告 · 台式机（HANS600452）

> 工作单元：0.1 环境审计（只读，未安装任何东西）
> 审计时间：2026-10-21 12:15–12:28（UTC+8）
> 审计方式：PowerShell 5.1 / Git Bash 下的只读查询（CIM/WMI、注册表、`nvidia-smi`、`Get-NetTCPConnection`、`curl` 连通性探测、conda-forge 与 PyPI 公共 API 查询）。下文所有数字均来自实际命令输出。
> 另一台电脑（笔记本）需要单独做一次，输出 `docs/setup_report_<笔记本机器名>.md`。

---

## 1. 结论摘要

| 项目 | 状态 | 对项目的影响 |
|---|---|---|
| CPU / 内存 / 磁盘 | i5-12600KF（10 核 16 线程），16 GB 单条内存，D 盘剩余 381 GB | 满足要求；内存是最紧的资源（见 §3） |
| GPU | GT 1030，2 GB，驱动 582.66 | 台式机只走 CPU 计算；三维渲染需按 GT 1030 设计 |
| Python | **系统内没有可用 Python**（PATH 上没有；`D:\Anaconda3` 存在但未接入 PATH） | 干净起点，适合在工作区内建独立环境 |
| 科学包可得性 | conda-forge 上所有必测包都有 win-64 二进制 | 支持 0.3 选 micromamba + conda-forge |
| Git | 2.55.0 + Git LFS 3.7.1；**未设置 user.name / user.email**；无 SSH 密钥；无 `gh` | 0.2 需要用户提供提交身份并自行登录远程 |
| WSL | **功能未启用**，无已注册发行版；BIOS 虚拟化已开启 | 以后启用需用户以管理员执行；本阶段不装 |
| Windows 补丁 | 停在 2021-10（19044.1288） | 影响新版 WSL 的安装方式（见 §6） |
| 网络 | 本机代理 FlClash（127.0.0.1:7890）已设为 HTTP(S)_PROXY；**GitHub 直连超时**，其他源直连/代理均可 | 下载 GitHub 资源必须走代理 |
| 端口 | 8000、5173 被**旧 Codex 项目**占用 | 建议用户关闭旧项目（见 §8） |

---

## 2. 操作系统

| 项 | 值 |
|---|---|
| 版本 | Windows 10 企业版 LTSC（EnterpriseS），21H2 |
| 内部版本 | 10.0.19044.**1288**，64 位 |
| 系统语言 | zh-CN；时区 China Standard Time |
| 代码页 | 系统 ANSI 代码页 **936（GBK）**；控制台当前 65001（UTF-8） |
| 安装日期 | 2026-08-24（用 2021 年的镜像装的） |
| 已装补丁 | 最新为 KB5006670 等，日期均为 2021-10-06；Windows Update 服务为"手动/已停止" |
| PowerShell | 仅 Windows PowerShell 5.1.19041（无 PowerShell 7） |
| 执行策略 | 所有作用域均为 Undefined（即默认 Restricted）；脚本需用 `-ExecutionPolicy Bypass` 在进程级运行，不需改系统设置 |
| PowerShell 配置文件 | 四个 profile 均不存在（没有 conda init 之类的自动激活） |
| 长路径支持 | `LongPathsEnabled = 0`（未开启 260 字符以上路径） |
| 电源方案 | 卓越性能 |

备注：Windows 10 Enterprise LTSC 2021 的官方支持预计于 2027-01 结束（来自微软生命周期政策的已有知识，本次未在线核实），与本项目短期工作无冲突。

## 3. 硬件

### CPU
- 12th Gen Intel Core i5-12600KF：**6 个性能核 + 4 个能效核，共 10 核 16 线程**，基准频率 3.7 GHz。
- 虚拟化：VT-x 固件已开启（`VirtualizationFirmwareEnabled = True`），支持 SLAT。
- 审计时 CPU 负载 72%（后台程序较多，见下）。

### 内存
- **16 GB，单条**（BANK 0，Colorful DDR4-3200，16 GB）→ **单通道运行**。
- 页面文件：`C:\pagefile.sys`，系统自动管理，当前分配 7.1 GB；提交上限约 23.4 GB。
- **审计时可用内存只有约 1.0 GB**，已提交 19.7 GB。主要占用：STM32CubeIDE（1.3 GB）、ChatGPT（多进程合计约 1.9 GB）、Claude（0.6 GB）、微信、Chrome、Edge、Codex，以及正在运行的旧 Codex 项目。
- 结论：CLAUDE.md 规定的"峰值约 12 GB"在当前后台负载下**做不到**。重计算时需要用户先关掉大型应用；后续所有重计算脚本应在启动时检查可用内存并给出提示。

### GPU
| 项 | 值 |
|---|---|
| 型号 | NVIDIA GeForce GT 1030（Pascal，计算能力 6.1） |
| 显存 | 2048 MiB（审计时已用 617 MiB，被桌面与应用占用） |
| 驱动 | 582.66（WDDM），驱动报告 CUDA 13.0 |
| PCIe | Gen3 ×4 |
| 当前分辨率 | **1440 × 900** |
| CUDA 工具包 | 未安装（无 `nvcc`） |

结论：台式机不做 GPU 计算，全部走 CPU 多核。三维界面（阶段 5）要能在 2 GB 显存、1440×900 屏幕上流畅使用。
未确定：本卡是 GDDR5 版还是 DDR4 版（两者显存带宽差约 3 倍，影响阶段 5 的渲染预算），阶段 5 前再测。

### 磁盘
| 盘符 | 文件系统 | 容量 | 剩余 |
|---|---|---|---|
| C: | NTFS | 477 GB | 366.1 GB |
| D:（卷标"软件"） | NTFS | 476.9 GB | **381.3 GB** |

- C 盘和 D 盘是**同一块** Lexar ARES 1 TB NVMe SSD 的两个分区（MBR），读写性能相同。
- 工作区当前占用约 90 KB。本阶段 10 GB 预算充足。

## 4. 已安装的软件与工具链

| 工具 | 位置 | 版本 | 备注 |
|---|---|---|---|
| Python | — | — | PATH 上**没有** `python`/`py`/`pip`/`conda`/`mamba`/`micromamba`/`uv`/`pixi` |
| Anaconda3 | `D:\Anaconda3` | Python 3.14.6，占 11.2 GB | 不在 PATH，无子环境；**本项目不使用**（保持环境自包含） |
| Node.js | `D:\nodejs` | v26.8.1，npm 11.19.0 | 阶段 5 前用不到 |
| Git | `C:\Program Files\Git` | 2.55.0.windows.5 | 系统级安装（属于正常软件安装） |
| Git LFS | 随 Git | 3.7.1 | 以后保留大型主世界数据时可用 |
| Git Credential Manager | 随 Git | — | `credential.helper=manager`，可浏览器登录 GitHub，不需要 `gh` |
| GitHub CLI `gh` | — | 未安装 | |
| OpenSSH 客户端 | 系统自带 | 8.1p1 | `~/.ssh` 不存在（无密钥） |
| CMake | `D:\CMake` | 4.4.2 | |
| Visual Studio 2022 Community | `C:\Program Files\...` | 17.2，MSVC 14.32.31326 | 含 C++ 组件；Windows SDK 10.0.19041 |
| Fortran 编译器 | — | 无 | conda-forge 提供预编译包，Windows 上不需要；ExoPlaSim 将来在 WSL 里编译 |
| make / ninja / gcc / clang | — | 无 | |
| .NET SDK | 系统 | 6.0.301 | 与本项目无关 |
| Docker / VirtualBox / VMware | — | 无 | 启用 WSL2 不会和其他虚拟机软件冲突 |
| curl | 系统 7.55.1；Git 自带 8.21.0 | | 脚本里优先用 Git 自带的新版 |
| VS Code | `D:\Microsoft VS Code` | 1.93.1 | |
| 安全软件 | — | 火绒（HipsDaemon/HipsTray）+ Windows Defender 实时防护 | 可能拖慢环境解压、拦截新下载的可执行文件 |

### Git 配置现状
- 全局：只有 `safe.directory=*`；**没有 user.name / user.email**。
- 系统级：`core.autocrlf=true`、`core.symlinks=false`、`init.defaultbranch=master`。
- 影响：0.2 需加 `.gitattributes` 固定换行规则（避免两台机器来回改换行），新仓库用 `main` 分支。

## 5. Python 版本与科学包可得性（只查询，未安装）

查询 conda-forge（`api.anaconda.org`）与 PyPI 的公开元数据，取各包最新版本的 Windows 构建：

| 包 | conda-forge 最新（win-64/noarch） | 覆盖的 Python | PyPI Windows wheel |
|---|---|---|---|
| pygplates | 1.0.0（2026-10） | 3.9–3.15 | 1.0.0，cp38–cp313 |
| gplately | 2.0.0（noarch） | ≥3.9 | 2.0.0（纯 Python） |
| stripy | 2.3.3（2025-10） | 3.9–3.14 | 2.3.3，cp37–cp312 |
| landlab | 2.10.1（2025-08） | **3.11–3.13** | 2.11.0，cp311–cp314 |
| fastscape | 0.1.0（noarch） | ≥3 | 0.1.0（纯 Python） |
| fastscapelib-f2py（Fastscape 的 Fortran 核心） | 2.8.4 | 3.8–3.14 | — |
| fastscapelib（新 C++ 版） | 0.3.0 | 待确认 | cp310–cp313 |
| richdem | 2.4.3（2026-10） | 3.10–3.15 | **0.3.4（2018 年），无 Windows wheel** |
| pysheds | 0.5 | 3.9–3.14 | 0.5（纯 Python） |
| cartopy | 0.26.0 | 3.11–3.15 | |
| gdal | 3.13.3 | 3.10–3.15 | |
| rasterio | 1.5.2 | 3.12–3.15 | |
| xarray | 2026.9.0（noarch） | ≥3.11 | |
| zarr | 3.4.1（noarch） | **≥3.12** | |
| netCDF4 | 1.7.4 | 3.10–3.14 | |
| scikit-image | 0.26.0 | 3.11–3.14 | |
| scipy / numpy | 1.18.1 / 2.5.4 | 3.12–3.15 | |
| goSPL | conda-forge 上没有 | — | 符合预期（Linux/MPI 工具，阶段 0 只调研） |

结论：
- 所有包同时可用的 Python 区间是 **3.12–3.13**（下限由 zarr、rasterio、numpy 最新版决定，上限由 conda-forge 上的 landlab 决定）。
- **初步建议 Python 3.12**（交集最宽、最成熟）；0.3 中用实际依赖求解结果确认。
- richdem 只能从 conda-forge 装；GDAL/rasterio/cartopy 在 Windows 上也以 conda-forge 最稳。这支持任务书提出的 **micromamba + conda-forge** 方案。0.3 仍会和 uv 做对比。

## 6. WSL 与虚拟化

| 项 | 状态 |
|---|---|
| 固件虚拟化（VT-x） | **已开启** |
| SLAT / VM 监视模式扩展 | 支持 |
| `Microsoft-Windows-Subsystem-Linux` | Disabled |
| `VirtualMachinePlatform` | Disabled |
| Hyper-V 全部组件 | Disabled |
| 当前是否运行 Hypervisor | 否 |
| 已注册发行版 | 无（无 Lxss 注册表项） |
| `wsl.exe` | 仅系统自带存根，`--status`/`--version` 只输出帮助文本 |

两条可选路线（**都不在阶段 0 执行**，到时由用户以管理员身份操作）：
1. **不更新 Windows，走"手动安装"**：微软官方文档（learn.microsoft.com/windows/wsl/install-manual，本次实际抓取核对）写明 x64 系统需 1903 / 18362.1049 以上，本机 19044 满足；步骤为启用两个可选功能 → 重启 → 安装 WSL2 Linux 内核更新包 → `wsl --set-default-version 2` → 用 `wsl --import` 把发行版放到 `environment/wsl/`。文档也专门提到 LTSC 系统没有应用商店时可直接下载发行版。
2. **先装 Windows 累积更新，再装新版 WSL（GitHub 上的 MSI 包）**：功能更新、内核更新更及时。本机补丁停在 2021 年，新版 WSL 可能要求较新的累积更新（我知道的说法是需要 KB5020030 / 19044.2311 以上，**本次未能从官方页面核实**，到时再确认）。

ExoPlaSim 只需要普通的 WSL2 + Ubuntu + gfortran，两条路线都能满足。

## 7. 网络

- 代理：FlClash 在 127.0.0.1:7890；用户环境变量已设 `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:7890`、`NO_PROXY=localhost,127.0.0.1,::1,.local`；WinINET 代理开启；WinHTTP 为直连。
- 连通性（`curl` 取 1 字节探测，2026-10-21 12:2x）：

| 目标 | 直连 | 走代理 |
|---|---|---|
| conda-forge（conda.anaconda.org） | 可达 0.8 s | 可达 1.0 s |
| micromamba 下载接口（micro.mamba.pm） | 可达（307 跳转） | 可达 |
| PyPI | 可达 0.3 s | 可达 2.8 s |
| **GitHub** | **15 s 超时** | 可达 0.5 s |
| Gitee | 可达 6.1 s | 可达 1.4 s |
| NOAA NCEI（ETOPO 页面） | 可达 | 可达 |
| Natural Earth | 可达 | 可达 |
| 清华 TUNA conda-forge 镜像 | 可达 0.2 s | 可达 |

- 下载速度抽样：conda-forge 走代理约 3.3 MB/s；TUNA 镜像直连约 2.9 MB/s。
- 结论：环境脚本要沿用现有代理变量；GitHub 上的资源（发行包、部分 GPlately 数据）必须走代理；conda-forge 可直接用官方源，TUNA 作为备用。
- 另：GitHub API 匿名访问已被限流（这个出口 IP 共享额度），脚本中不要依赖匿名调用 GitHub API。

## 8. 端口与正在运行的旧项目

监听中的端口（节选）：

| 端口 | 进程 | 说明 |
|---|---|---|
| **8000** | python（uvicorn `apps.api.main:app`，工作目录 `D:\一个世界观`） | **旧 Codex 项目的后端，正在运行** |
| **5173** | node（vite，`D:\一个世界观\apps\web`） | **旧 Codex 项目的前端，正在运行** |
| 7890、1053、47890 | FlClash | 代理 |
| 41337 | STM32CubeIDE | |
| 14013–14023 | 微信 | |
| 135、139、445、5040、5357、496xx | Windows 系统服务 | |

空闲的常用开发端口：3000、5000、5432、6006、8080、8888、9000。

- 旧项目的 Python/Node 来自 `C:\Users\Administrator\.cache\codex-runtimes\`（Codex 自带运行时）。本次**没有读取**旧项目内容，也没有动它。
- `使用说明.md` 已建议把旧项目归档，避免被当作参考。建议用户关闭这两个进程（释放内存和端口），并把 `D:\一个世界观` 移到归档位置。本项目今后的开发端口避开 8000 和 5173。

## 9. C 盘现状

项目文件和环境目前都不在 C 盘。以下是**审计前就已存在**的用户级缓存，与本项目无关，未改动：

| 路径 | 大小 | 来源 |
|---|---|---|
| `%LOCALAPPDATA%\npm-cache` | 1.2 GB | npm 默认缓存 |
| `%APPDATA%\npm` | 1.1 GB | npm 全局包 |
| `%USERPROFILE%\.cache` | 1.4 GB | 主要是 Codex 运行时 |
| `%LOCALAPPDATA%\pip\cache` | 17 MB | pip 默认缓存 |
| `%USERPROFILE%\.conda\environments.txt` | 很小 | 记录了 `D:\Anaconda3` |

需要注意：很多工具默认把缓存写到 C 盘（`TEMP`/`TMP` 都在 `C:\Users\Administrator\AppData\Local\Temp`，npm 缓存在 C 盘，matplotlib/cartopy/pooch 的缓存默认也在用户目录）。0.3 的环境脚本必须把这些全部重定向到工作区（见 §10）。

本次审计自己的临时文件（几个探测脚本和下载的两页微软文档，共约 0.2 MB）放在 Claude Code 会话临时目录（`C:\Users\Administrator\AppData\Local\Temp\claude\...\scratchpad`），不是项目文件，可随时删除。

## 10. 建议清单（本单元不执行，0.2/0.3 实施，需要时征得用户同意）

### 不需要用户操作、我会在后续单元做的
1. **0.2**：`git init -b main`；加 `.gitattributes`（文本统一 LF，`*.png`/`*.nc` 等标为二进制），抵消系统级 `autocrlf=true`；提交身份写在仓库级配置，不改全局。
2. **0.3**：micromamba 单文件放在 `environment/bin/`；`MAMBA_ROOT_PREFIX` 和包缓存放在工作区内；环境前缀用短路径（如 `environment/envs/mpl`），绕开 260 字符路径限制。
3. **0.3**：环境激活脚本统一设置：`PYTHONUTF8=1`（系统代码页是 GBK，避免读写中文文档时乱码）、`TEMP`/`TMP` 指向 `.cache/tmp`、`PIP_CACHE_DIR`、`MPLCONFIGDIR`、cartopy 数据目录、GPlately/pooch 缓存目录全部指向工作区。各库缓存变量在 0.3/0.4 逐个实测确认。
4. **0.3**：`setup_env.ps1` 只用 PowerShell 5.1 语法；用 `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` 运行，不改系统执行策略。
5. **0.3**：Python 初步定为 3.12，锁定版本（`environment.yml` + 锁文件）。
6. 并行计算：默认工作线程数在 0.4/0.5 中实测（6 大核 + 4 小核，超过 12 线程后收益可能很小），重计算脚本启动时检查可用内存。
7. 以后用 Node 时（阶段 5），在项目 `.npmrc` 里把 npm 缓存指到 `.cache/npm`。

### 需要用户决定或本人操作的
| # | 事项 | 何时需要 | 建议 |
|---|---|---|---|
| U1 | 关闭并归档旧 Codex 项目 `D:\一个世界观`（现在正在运行，占用 8000/5173 端口和内存） | 尽早 | ~~建议关闭并移到归档目录~~ 用户仍在使用，决定保留（2026-10-21）；实测约占 220 MB，只在大内存计算时临时关闭 |
| U2 | 提供 Git 提交用的名字和邮箱（只写进本仓库配置） | 0.2 | |
| U3 | 远程仓库选 GitHub 还是 Gitee | 0.2 | GitHub：必须走代理，可用自带的 Git Credential Manager 在浏览器登录，不必装 `gh`。Gitee：直连可用。两者都行，看你更常用哪个 |
| U4 | 重计算时关闭大型后台程序（STM32CubeIDE、ChatGPT、浏览器等） | 0.5 起 | 当前可用内存约 1 GB，达不到 12 GB 预算 |
| U5 | 是否在火绒/Defender 中把 `D:\mythic-planet-lab\environment` 和 `.cache` 加入信任 | 0.3 前（可选） | 能明显加快环境安装；这是安全设置，由你决定、你来操作。不加也能用，只是慢一些 |
| U6 | 是否安装 Windows 累积更新 | 启用 WSL 前（阶段 0 之后） | 不更新也能走 WSL2 手动安装路线；更新后可用新版 WSL |
| U7 | （可选硬件）再加一条 16 GB DDR4-3200 | 任意时候 | 变成 32 GB 双通道，内存容量和带宽都改善；不加也能按 12 GB 预算工作 |
| U8 | （可选）开启 Windows 长路径支持 | 一般不需要 | 我会用短路径避开；只有遇到具体报错再考虑 |
