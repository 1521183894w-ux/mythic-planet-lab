# 项目进度（PROGRESS）

> 每次开工先读本文件，每个检查点结束时更新。
> "用户评审"部分只由用户填写；用户在对话中口述时，开发者可以**逐字转录**并注明"转录自对话"，但不得改写、概括或删改。

---

## 当前状态

- 当前阶段：阶段 0 · 环境、工具验证与地球基准
- 当前工作单元：0.4 已完成（工具验证 + 构造方法调研）；下一步 0.5 质量指标代码 + 地球基准
- 当前使用的机器：台式机（HANS600452）
- 最近一次提交：见 `git log -1`；已推送到 `origin/main`（https://github.com/1521183894w-ux/mythic-planet-lab）

## 已完成

（按时间倒序，每条一行：日期 · 工作单元 · 简述 · 产出路径）

- 2026-10-21 · 0.4 科学工具快速验证 · 6 个 spike 全部实际运行并看图核对：stripy 网格 L8（65.5 万点）建网 1.4 s；pyGPlates 旋转/层级/速度与解析值一致；zarr/netCDF 元数据（含中文）无损；地球陆地数据验证经度环绕、面积加权、球面距海距离（欧亚离海最远点 2486 km，文献 ≈2510 km）；Fastscape 内核稳态坡度—面积关系与理论吻合（-0.398 vs -0.4），比 Landlab 快约 40 倍；richdem 与 Fastscape 汇水面积相关 0.99。发现 fastscape 封装层（xarray-simlab）与 pysheds 不兼容 numpy 2.x/zarr 3。goSPL、ExoPlaSim 只调研（均需 WSL）。构造方法调研读了 Cortial 2019 全文，列出两条候选路线 · `docs/tool_evaluation.md`、`docs/tectonics_survey.md`、`spikes/stage0_tools/`、图在 `figures/stage0/tools/`
- 2026-10-21 · 0.3 Python 环境 · uv 小实验（26 个包中 gdal、richdem 无 Windows wheel，fastscape 缺 Fortran 核心）后选定 micromamba 2.9.0 + conda-forge + Python 3.12；351 个包按 md5 锁定；`setup_env.ps1` 台式机从零重建（清空缓存、重新下载）5.6 分钟，有缓存 0.5 分钟，真实占用 3.88 GB；自检 49 项全过，pytest 2 项通过；修复 micromamba 分片缓存与 gplately 缓存写 C 盘的问题 · `environment.yml`、`environment.win-64.lock.txt`、`scripts/setup_env.ps1` 等、`docs/environment.md`
- 2026-10-21 · 0.2 推送与收尾 · 用户在终端完成 GCM 授权并推送，核对远程 `main` 与本地一致；台式机已保存 GitHub 凭据；按用户要求把 CLAUDE.md 第 6 节改为公开仓库，并补充"不得提交凭据和未同意公开的个人信息"；使用说明改正公开仓库下的克隆/登录步骤 · `CLAUDE.md`、`使用说明.md`
- 2026-10-21 · 0.2 Git 与项目骨架（本地部分）· `git init -b main`；补充 `.gitignore`，新增 `.gitattributes`（仓库与工作区统一 LF）；建立目录骨架并附说明；`使用说明.md` 补充 GitHub 建仓、登录、克隆、日常同步命令 · `.gitattributes`、`config/`、`packages/`、`scripts/`、`data/`、`使用说明.md`
- 2026-10-21 · 0.1 环境审计（台式机）· 只读审计硬件、系统、工具链、WSL、网络、端口、C 盘缓存，查询 conda-forge/PyPI 上关键科学包的 Windows 可得性，未安装任何东西 · `docs/setup_report_HANS600452.md`

## 进行中

- 笔记本尚未做：0.1 环境审计、0.3 环境重建实测（用户换到笔记本时提醒）

## 问题与待决事项

（需要用户决定的事情放在这里，并在汇报中提醒用户）

详见 `docs/setup_report_HANS600452.md` 第 10 节。

- U1 已定：旧 Codex 项目 `D:\一个世界观` 用户仍在使用，保留、不归档（2026-10-21）。实测其三个进程共占约 220 MB 内存、端口 8000/5173；它在工作区之外，本项目不读取它，开发端口避开 8000/5173。仅在跑大内存计算时请用户临时关闭。
- U2 已定：提交身份 `1521183894w-ux <1521183894w@gmail.com>`（仅本仓库配置），已有本地提交已改为此署名。
- U3 已定：远程仓库用 GitHub `1521183894w-ux/mythic-planet-lab`，**公开仓库**。开发者指出该仓库为公开、与 CLAUDE.md 第 6 节"私有远程仓库"不一致后，用户明确决定保持公开（2026-10-21，用户原话："我确认就用公开仓库，直接推送。"）。按 CLAUDE.md 开头的优先级规则，以用户明确指示为准；随后用户要求把 CLAUDE.md 第 6 节改为公开仓库，已于同日修改。后果：仓库内所有文档与提交记录（含提交邮箱）公开可见；今后不要把不想公开的内容（例如未定稿的私人设定、账号信息）放进仓库。
- U4 审计时可用内存约 1 GB（2026-10-21 下午复测约 2 GB）；0.5 起的重计算需要用户先关闭大型后台程序。
- U5（可选，已不急）是否把 `environment/`、`.cache/` 加入火绒/Defender 信任。不加时台式机从零重建也只要 5.6 分钟，目前没有必要。
- U6（阶段 0 之后）启用 WSL 前，决定是否先装 Windows 累积更新（系统补丁停在 2021-10）。
- U7（可选硬件）加一条 16 GB 内存组成双通道。
- U8（阶段 1 开始时）构造历史存储方案：每 1 Ma 存一帧会让几十个种子超出 10 GB 磁盘预算。建议筛选阶段每 10 Ma 存一帧，候选世界再确定性重算 1 Ma 历史（见 `docs/tectonics_survey.md` §5）。
- U9（阶段 1.1）构造方案由用户在看过两条路线的原型对比图后选定。调研的初步倾向是路线 B（pyGPlates 管运动学 + 自写地壳规则），但不是决定。
- U10（阶段 2/3 前）启用 WSL：goSPL（全球侵蚀候选）和 ExoPlaSim（大气环流候选）都只能在 WSL 里运行。

## 已知局限

（诚实记录目前方法的不足，例如"降水为参数化近似"）

- 台式机 GT 1030 显存是 GDDR5 版还是 DDR4 版未确定，阶段 5 前再测。
- 新版 WSL（GitHub MSI 包）在 Win10 上要求的最低补丁版本未能从官方页面核实。
- 锁文件只有 win-64 一份；将来 WSL 里的环境（ExoPlaSim）需要单独的 linux-64 规格与锁文件。
- gplately 导入时会在当前目录建 `gplately.log`/`ptt.log`（无法配置）。目前靠 `.gitignore` 和自检时切换目录处理；0.4 起项目代码应通过统一的导入函数使用 gplately。
- micromamba 关闭分片索引后，`-FromSpec` 求解会在 `.cache\mamba-pkgs\cache` 留下约 480 MB 索引文件（可删除）。
- Claude Code 的后台任务在会话结束时会被终止（本单元两次遇到）。超过 10 分钟的长任务需要另想办法（例如由用户在终端运行，或拆分）。
- fastscape 的 Python 封装层（xarray-simlab 0.5.0）与 zarr 3 / numpy 2.x 不兼容；阶段 2 直接调用 fastscapelib-fortran 内核（spike 05 已有可用的驱动循环）。
- pysheds 0.5 与 numpy 2.4+ 不兼容（`np.in1d`）；水文用 richdem。
- 0.4 的计时在后台程序较多时测得，同一测试两次运行可差一倍（Landlab 1024x1024：12.3 vs 24.6 s/步），只能看量级。
- Bird 2003、Parsons & Sclater 1977 的引用尚未核对原文，阶段 1 使用前核对。

## 阶段完成记录

| 阶段 | 用户确认日期 | 备注 |
|---|---|---|
| 0 | | |
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |

---

## 用户评审

<!-- 格式：
### <日期> 阶段 N
结论：通过 / 不通过 / 有条件通过
意见：
- ...
-->
