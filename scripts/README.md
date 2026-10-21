# scripts/

命令行入口：环境重建（`setup_env.ps1`）、出图（`make_figures`）、数据下载等。

## 编写约定

- PowerShell 脚本必须兼容 **Windows PowerShell 5.1**（两台电脑都没有 PowerShell 7）。
- `.ps1` 文件里如有中文，必须存为 **UTF-8 带 BOM**。PowerShell 5.1 会把不带 BOM 的文件按系统代码页（GBK）读取，中文会变成乱码，甚至导致语法错误。
- 运行方式：`powershell -ExecutionPolicy Bypass -File scripts\xxx.ps1`（只在本进程放开执行策略，不改系统设置）。
- 所有下载、缓存、临时文件写到工作区内的 `.cache/`，不写 C 盘。
