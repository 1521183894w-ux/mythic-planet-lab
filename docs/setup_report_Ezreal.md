# 环境审计报告 · 笔记本（Ezreal）

> 工作单元：0.1 环境审计（只读，审计本身未安装任何东西）
> 审计时间：2026-10-21 19:10–19:25（UTC+8）
> 审计方式：PowerShell 5.1 / Git Bash 下的只读查询（CIM/WMI、注册表、`nvidia-smi`、`vswhere`、`Get-NetTCPConnection`、`curl` 连通性探测）。下文所有数字均来自实际命令输出。
> 台式机的报告见 `docs/setup_report_HANS600452.md`，本报告格式与之对应，重点写两台机器的差异。
> 审计后在本仓库设置了提交身份（仅本仓库配置，U2 已定的身份），并按交接计划运行了 `setup_env.ps1`，结果见 `docs/environment.md` §6。

---

## 1. 结论摘要

| 项目 | 状态 | 对项目的影响 |
|---|---|---|
| CPU / 内存 / 磁盘 | i7-12700H（14 核 20 线程），16 GB DDR5-4800 双通道，D 盘剩余 99 GB | 计算能力强于台式机；但性能预算仍以台式机为下限 |
| GPU | RTX 3060 Laptop，6 GB，驱动 572.61，计算能力 8.6 | 可以在这台机器上试 GPU 加速（只作可选项，必须有 CPU 回退） |
| Python | 系统里有多个旧 Python（3.6–3.13）和 uv，**均不使用**；PATH 上的 `python` 是 Windows 应用商店别名 | 项目环境自包含，不受影响 |
| Git | 2.55.0 + LFS 3.7.1，Git Credential Manager；全局没有提交身份 | 已在本仓库设置 U2 身份；**推送前用户需在这台机器登录一次 GitHub** |
| WSL | **未安装**；虚拟化固件已开启；Windows 11 26H2、补丁是最新的（2026-10-18） | 将来启用 WSL 可直接走 `wsl --install` 新路线，不需要先打补丁（台式机的 U6 在这里不存在） |
| 网络 | FlClash 系统代理 127.0.0.1:7890；**conda-forge、raw.githubusercontent、NOAA 数据服务器直连不通**，走代理都通；GitHub 主站直连可达 | **代理环境变量没有写进用户/系统环境**，用户自己开的终端里可能没有，见 §7 |
| 端口 | 8000、5173 空闲；旧 Codex 项目目录 `D:\一个世界观` 在本机也存在但未运行 | 无冲突 |
| 安全软件 | 联想杀毒（火绒内核）实时防护；Defender 实时防护关闭 | 可能拖慢环境解压，见 U5 |

---

## 2. 操作系统

| 项 | 值 |
|---|---|
| 版本 | Windows 11 家庭中文版（CoreCountrySpecific），26H2 |
| 内部版本 | 10.0.26300.**9711**，64 位 |
| 系统语言 | zh-CN；时区 China Standard Time |
| 代码页 | 系统 ANSI 代码页 **936（GBK）**；控制台 65001（UTF-8）——与台式机相同，`PYTHONUTF8=1` 照样需要 |
| 安装日期 | 2025-06-01 |
| 已装补丁 | 最新 KB5121794、KB5127648、KB5127694（2026-10-18）——**系统是最新的** |
| PowerShell | 仅 Windows PowerShell 5.1.26100（无 PowerShell 7） |
| 执行策略 | CurrentUser 为 RemoteSigned，其他未定义；项目脚本仍统一用 `-ExecutionPolicy Bypass` 在进程级运行 |
| PowerShell 配置文件 | 四个 profile 均不存在 |
| 长路径支持 | `LongPathsEnabled = 1`（已开启；台式机未开启，所以项目仍按短路径设计） |
| 电源方案 | 野兽模式（联想 Legion 自带方案） |
| 用户目录 | `C:\Users\15211`（"文档"重定向到 OneDrive） |

## 3. 硬件

整机：联想 Legion Y9000P IAH7H（型号 82RF，BIOS J2CN40WW）。

### CPU
- 12th Gen Intel Core i7-12700H：**6 个性能核 + 8 个能效核，共 14 核 20 线程**，基准频率 2.3 GHz。
- 虚拟化：VT-x 固件已开启，支持 SLAT 与 VM 监视模式扩展。
- 对比台式机（i5-12600KF，10 核 16 线程，基准 3.7 GHz）：核数更多，单核持续频率受笔记本功耗/散热限制。两台机器的计时不能直接比较，**性能预算仍以台式机为准**。

### 内存
- **16 GB，两条 8 GB Samsung DDR5-4800（M425R1GB4BB0-CQKOL），双通道**。容量与台式机相同，带宽明显更高（台式机是单条 DDR4-3200 单通道）。
- 页面文件：手动设置，`C:\pagefile.sys` 19 GB + `D:\pagefile.sys` 2.9 GB；提交上限约 37.7 GB。
- 审计时可用内存约 **5.7 GB**。主要占用：Claude（12 个进程合计约 2.3 GB）、Edge（约 1.2 GB）、内存压缩（1.1 GB）、ChatGPT（约 0.9 GB）。
- 结论：比台式机宽裕（台式机审计时约 1 GB），但做 12 GB 级别的重计算前仍需关掉浏览器和 ChatGPT 等。

### GPU
| 项 | 值 |
|---|---|
| 型号 | NVIDIA GeForce RTX 3060 Laptop GPU（Ampere，计算能力 8.6） |
| 显存 | 6144 MiB（审计时已用 1444 MiB） |
| 驱动 | 572.61（WDDM），驱动报告支持 CUDA 12.8 |
| 功耗上限 | 139 W |
| PCIe | 最高 Gen4 ×16 |
| 显示 | **2560 × 1600，165 Hz** |
| 核显 | 设备管理器里只看到独显（可能在 BIOS/Legion Zone 里设成了独显直连），不影响项目 |
| CUDA 工具包 | 已装 **CUDA 11.1**（`nvcc` 在 PATH，`CUDA_PATH` 已设置），版本较旧 |

结论：可以在这台机器上评估 GPU 加速（例如阶段 2 的侵蚀、阶段 5 的渲染），但只能是可选项，必须保留 CPU 回退，并在台式机 GT 1030 上验证。若以后用 GPU 库，优先选 conda-forge 自带 CUDA 运行时的包，不依赖系统里这份 11.1 工具包。阶段 5 的界面要同时适配 2560×1600（本机）和 1440×900（台式机）两种屏幕。

### 磁盘
| 盘符 | 文件系统 | 容量 | 剩余 | 所在硬盘 |
|---|---|---|---|---|
| C: | NTFS | 300 GB | 85 GB | SK hynix 512 GB NVMe（GPT） |
| D:（新加卷） | NTFS | 174.7 GB | **99.3 GB** | 与 C 盘同一块 SK hynix |
| E:（新加卷） | NTFS | 953.9 GB | 251.8 GB | 致态 TiPlus7100s 1 TB NVMe（MBR） |

- 工作区按 CLAUDE.md 放在 D 盘，审计时约 0.8 MB（只有 Git 仓库内容）。本阶段 10 GB 预算充足。
- E 盘空间更大。若将来 D 盘不够（例如阶段 1 多种子历史），可以讨论把 `data/worlds/` 放到 E 盘，但这要改 CLAUDE.md 的约定，届时由用户决定。

## 4. 已安装的软件与工具链

| 工具 | 位置 | 版本 | 备注 |
|---|---|---|---|
| Python（PATH 上的 `python`） | `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe` | — | 0 字节的应用商店别名，不是真的 Python |
| Python（py 启动器登记） | `D:\Python`（3.13）、`D:\新建文件夹`（3.11、3.6）、`D:\python3.9.13`、`D:\Python-3.7.6`、`C:\...\Python38` | 3.6–3.13 | **本项目不使用**（环境自包含） |
| uv | `C:\Users\15211\.local\bin\uv.exe` | 0.12.13 | 本项目不使用 |
| conda / mamba / micromamba | — | 无 | micromamba 由 `setup_env.ps1` 下载到 `environment\bin\` |
| Node.js | — | 无 | 阶段 5 前用不到；D 盘有 `.pnpm-store`，与本项目无关 |
| Git | `D:\Git` | 2.55.0.windows.3 | |
| Git LFS | 随 Git | 3.7.1 | |
| Git Credential Manager | 随 Git | — | `credential.helper=manager` |
| GitHub CLI `gh` | — | 未安装 | 不需要 |
| OpenSSH 客户端 | 系统自带 | 9.5p2 | `~/.ssh` 不存在 |
| Visual Studio Community | `C:\Program Files\Microsoft Visual Studio\18`、`\2022` | MSVC 14.51 与 14.43 | 两个版本并存 |
| CMake / make / ninja / gcc / gfortran | — | PATH 上都没有 | conda-forge 提供预编译包，Windows 上不需要 |
| CUDA 工具包 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1` | 11.1.105 | 见 §3 |
| .NET SDK | 系统 | 6.0.400、10.0.401 | 与本项目无关 |
| Docker / 虚拟机软件 | — | 无 | |
| curl | 系统与 Git 自带 | 均为 8.21.0 | |
| VS Code | `%LOCALAPPDATA%\Programs\Microsoft VS Code` | — | |
| 安全软件 | — | 联想杀毒（Lenovo Anti-Virus powered by Huorong）；Windows Defender 实时防护**关闭** | 火绒内核可能拖慢环境解压 |

### Git 配置现状
- 系统级（`D:\Git\etc\gitconfig`）：`core.autocrlf=true`、`core.symlinks=false`、`init.defaultbranch=master`、`credential.helper=manager`——与台式机相同，仓库里的 `.gitattributes` 已固定 LF，不受影响。
- 全局（`C:\Users\15211\.gitconfig`）：只有 `safe.directory`，**没有 user.name / user.email**，没有 `http.proxy`。
- 本仓库：克隆时没有提交身份。审计后已按 U2 设置 `user.name=1521183894w-ux`、`user.email=1521183894w@gmail.com`、`core.quotepath=false`（只写进 `.git/config`，不改全局）。
- GitHub 推送凭据：本机是否已登录过 GitHub 无法只读确认（GCM 存在 Windows 凭据管理器里）。第一次推送时如弹出登录窗口，需要用户本人完成授权。

## 5. Python 与科学包

科学包的 Windows 可得性已在台式机审计中查过（conda-forge 上都有 win-64 二进制），两台机器按同一份锁文件安装，这里不重复查询。

## 6. WSL 与虚拟化

| 项 | 状态 |
|---|---|
| 固件虚拟化（VT-x） | **已开启** |
| SLAT / VM 监视模式扩展 | 支持 |
| `wsl --status` | "未安装适用于 Linux 的 Windows 子系统。可通过运行 `wsl.exe --install` 进行安装。" |
| 可选功能 `Microsoft-Windows-Subsystem-Linux` / `VirtualMachinePlatform` | 查询需要管理员权限，未查（只读审计不提权） |
| 已注册发行版 | 无（无 Lxss 注册表项） |
| 当前是否运行 Hypervisor | 否；基于虚拟化的安全（VBS）未运行 |

与台式机不同：本机是 Windows 11 26H2 且补丁最新，**可以直接用新版 WSL**（`wsl --install --no-distribution` 安装 WSL 本体，再用 `wsl --import` 把发行版放到 `environment/wsl/`）。家庭版支持 WSL2（不需要 Hyper-V 完整功能）。仍然要用户以管理员身份执行，时间按 U10（阶段 2/3 前）。

## 7. 网络

- 代理：FlClash 在 127.0.0.1:7890，**系统代理（WinINET）已开启**，指向 127.0.0.1:7890；WinHTTP 为直连。
- **环境变量**：当前 Claude Code 会话的进程里有 `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:7890`、`NO_PROXY=localhost,127.0.0.1,::1,.local`，但**用户级和系统级环境变量（注册表）里都没有**——它是从启动 Claude 的父进程继承来的。用户自己打开的 PowerShell 窗口里很可能没有这些变量。
  - 影响：micromamba（libcurl）只认环境变量，不认系统代理；在没有变量的窗口里运行 `setup_env.ps1` 会连不上 conda-forge。Python 的 `requests`（地球数据下载）在 Windows 上没有环境变量时会读系统代理，不受影响。
  - 处理：在 `scripts/env_vars.ps1` 里加了回退——没有 `HTTPS_PROXY` 而系统代理开启时，把系统代理设为**本进程**的代理变量（不改用户/系统设置）。详见 `docs/environment.md` §4。
- 连通性（`curl` 取 1 字节探测，2026-10-21 19:2x）：

| 目标 | 直连 | 走代理 |
|---|---|---|
| conda-forge（conda.anaconda.org） | **不通**（立即失败） | 可达 2.8 s |
| micromamba 下载接口（micro.mamba.pm） | 可达（307 跳转） | 可达 |
| PyPI | 可达 0.3 s | 可达 0.5 s |
| GitHub 主站 | 可达 0.7 s | 可达 2.8 s |
| raw.githubusercontent.com | **不通** | 可达 3.9 s |
| NOAA NCEI 网页 | 可达 | 可达 |
| NOAA 数据服务器（ngdc.noaa.gov） | **15 s 超时** | 可达 1.4 s |
| Natural Earth（naciscdn.org） | 可达 | 可达 |
| 清华 TUNA conda-forge 镜像 | 可达 1.4 s | 可达 0.5 s |

- 结论：与台式机一样，项目脚本一律走代理。注意本机是 conda-forge 直连不通、GitHub 主站直连可通，和台式机正好不同，不能凭一台机器的经验判断另一台。

## 8. 端口

| 端口 | 进程 | 说明 |
|---|---|---|
| 7890、1053、47890 | FlClash | 代理 |
| 12900 | LZTray | 联想 Legion Zone |
| 49801 | jhi_service | Intel 动态应用加载器 |
| 135、445、1001、5040、7680、496xx | 系统服务 | |

- **8000、5173 空闲**。旧 Codex 项目目录 `D:\一个世界观`（以及 `D:\一个世界观.zip`）在本机也存在，但没有在运行。本次没有读取它。
- 空闲的常用开发端口：3000、5000、5432、6006、8080、8888、9000。

## 9. C 盘现状

项目文件和环境都不在 C 盘。以下是**审计前就已存在**的用户级缓存，与本项目无关，未改动：

| 路径 | 大小 |
|---|---|
| `C:\Users\15211\.cache` | 2.91 GB |
| `%LOCALAPPDATA%\pip\cache` | 0.33 GB |
| `%LOCALAPPDATA%\uv\cache` | 0.30 GB |

本次审计的临时文件在 Claude Code 会话临时目录（`%LOCALAPPDATA%\Temp\claude\...`），不是项目文件。环境重建后对 C 盘新写入的检查见 `docs/environment.md` §6。

## 10. 建议清单

### 不需要用户操作、已经或将由开发者处理的
1. 本仓库提交身份已设置（U2）。
2. `env_vars.ps1` 增加系统代理回退（§7）。
3. 两台机器的计时只比量级，性能预算以台式机为准；GPU 相关评估只在本机做，并保留 CPU 回退。

### 需要用户决定或本人操作的
| # | 事项 | 何时需要 | 建议 |
|---|---|---|---|
| U13 | 在本机登录一次 GitHub（推送需要） | 本机第一次推送时 | 运行 `git -C D:/mythic-planet-lab push`，弹窗里选 "Sign in with your browser"；如果之前登录过就不会弹窗 |
| U5 | 火绒（联想杀毒）是否信任 `D:\mythic-planet-lab\environment` 和 `.cache` | 可选 | 同台式机：不加也能用，本机冷启动 11.6 分钟，主要慢在下载而不是解压，目前没有必要 |
| U10 | WSL 启用方式 | 阶段 2/3 前 | 本机系统已是最新，可直接 `wsl --install --no-distribution`（管理员）；台式机仍按 U6 决定 |
| U14 | 将来 D 盘不够时是否允许用 E 盘 | 阶段 1 多种子运行时再看 | D 盘剩 99 GB，目前够用 |

（编号与 `PROGRESS.md` 的"问题与待决事项"一致。）
