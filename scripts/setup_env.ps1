<#
.SYNOPSIS
  一键重建本项目的 Python 环境。环境、包缓存、临时文件全部位于工作区内。

.DESCRIPTION
  默认按锁文件 environment.win-64.lock.txt 精确安装（不重新求解），两台电脑得到完全相同的包。

  参数：
    -FromSpec    按 environment.yml 重新求解并安装，然后更新锁文件（增删包后使用）
    -Force       环境已存在时先删除再重建
    -CleanCache  同时清空包缓存 .cache\mamba-pkgs（模拟全新机器的冷启动，会重新下载约 650 MB）
    -SkipCheck   跳过安装后的自检（scripts\check_env.py）

  整个环境可以直接删除 environment\ 目录后重建，不影响系统。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -FromSpec -Force
#>
param(
    [switch]$FromSpec,
    [switch]$Force,
    [switch]$CleanCache,
    [switch]$SkipCheck
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # PowerShell 5.1 的下载进度条会让下载慢很多倍
# 输出被重定向到文件时按 UTF-8 写（系统默认代码页是 GBK，否则日志里的中文是乱码）
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }

# micromamba 版本固定，升级时同时更新版本号和 SHA256（来自官方发布页的 .sha256 文件）
$MambaVersion = '2.9.0-0'
$MambaSha256  = 'A6D804394B2418991C4E29562853EAACE2F2CE9D9DA661A98E74E02E8DBB44B0'
$MambaUrl     = "https://github.com/mamba-org/micromamba-releases/releases/download/$MambaVersion/micromamba-win-64"

. (Join-Path $PSScriptRoot 'env_vars.ps1')

$Spec = Join-Path $MythicRoot 'environment.yml'
$Lock = Join-Path $MythicRoot 'environment.win-64.lock.txt'
$Log  = Join-Path $MythicRoot 'environment\setup_log.txt'

function Write-Step([string]$Message) {
    $line = "[setup_env $(Get-Date -Format 'HH:mm:ss')] $Message"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

# 任何一步失败都记入 setup_log.txt，然后终止（退出码非 0）
trap {
    if ($Log -and (Test-Path (Split-Path $Log))) {
        Add-Content -Path $Log -Value "[setup_env $(Get-Date -Format 'HH:mm:ss')] 失败：$($_.Exception.Message)" -Encoding UTF8
    }
    break
}

function Test-Sha256([string]$File, [string]$Expected) {
    (Get-FileHash -Path $File -Algorithm SHA256).Hash -eq $Expected.ToUpper()
}

$stopwatch = [Diagnostics.Stopwatch]::StartNew()
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Write-Step "开始。工作区：$MythicRoot；模式：$(if ($FromSpec) { '按规格求解' } else { '按锁文件' })"

# ---- 1. micromamba ----------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Split-Path $MythicMamba) | Out-Null
if ((Test-Path $MythicMamba) -and (Test-Sha256 $MythicMamba $MambaSha256)) {
    Write-Step "micromamba $MambaVersion 已存在，SHA256 校验通过"
} else {
    Write-Step "下载 micromamba $MambaVersion ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $tmp = "$MythicMamba.download"
    $request = @{ Uri = $MambaUrl; OutFile = $tmp; UseBasicParsing = $true }
    if ($env:HTTPS_PROXY) { $request.Proxy = $env:HTTPS_PROXY }   # GitHub 在本机需要走代理
    Invoke-WebRequest @request
    if (-not (Test-Sha256 $tmp $MambaSha256)) {
        Remove-Item $tmp -Force
        throw "micromamba 下载文件的 SHA256 不符，已删除。请检查网络或代理后重试。"
    }
    Move-Item $tmp $MythicMamba -Force
    Write-Step "micromamba 下载完成，SHA256 校验通过"
}

# ---- 2. 处理已有环境与缓存 --------------------------------------------------
if (Test-Path $MythicEnvPrefix) {
    if (-not $Force) { throw "环境已存在：$MythicEnvPrefix。如需重建请加 -Force。" }
    Write-Step "删除现有环境 $MythicEnvPrefix"
    Remove-Item $MythicEnvPrefix -Recurse -Force
}
if ($CleanCache -and (Test-Path $env:CONDA_PKGS_DIRS)) {
    Write-Step "清空包缓存 $env:CONDA_PKGS_DIRS"
    Remove-Item $env:CONDA_PKGS_DIRS -Recurse -Force
    New-Item -ItemType Directory -Force -Path $env:CONDA_PKGS_DIRS | Out-Null
}

# ---- 3. 创建环境 ------------------------------------------------------------
$createArgs = @('create', '--no-rc', '-y', '-p', $MythicEnvPrefix)
if ($FromSpec) {
    Write-Step "按 environment.yml 求解并创建环境 ..."
    & $MythicMamba @createArgs -f $Spec --channel-priority strict
} else {
    if (-not (Test-Path $Lock)) { throw "找不到锁文件 $Lock。首次请使用 -FromSpec 生成。" }
    Write-Step "按锁文件精确创建环境 ..."
    & $MythicMamba @createArgs -f $Lock
}
if ($LASTEXITCODE -ne 0) { throw "micromamba create 失败（退出码 $LASTEXITCODE）" }
Write-Step "环境创建完成"

# ---- 4. 更新锁文件（仅 -FromSpec）------------------------------------------
if ($FromSpec) {
    $exported = & $MythicMamba env export --no-rc -p $MythicEnvPrefix --explicit
    if ($LASTEXITCODE -ne 0) { throw "导出锁文件失败（退出码 $LASTEXITCODE）" }
    $header = @(
        "# 由 scripts/setup_env.ps1 -FromSpec 生成，请勿手改。规格见 environment.yml。",
        "# 生成时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm')；micromamba $MambaVersion；平台 win-64"
    )
    $text = ((@($header) + @($exported)) -join "`n") + "`n"
    [IO.File]::WriteAllText($Lock, $text, (New-Object System.Text.UTF8Encoding $false))
    $count = @($exported | Where-Object { $_ -match '^https?://' }).Count
    Write-Step "锁文件已更新：$Lock（$count 个包）"
}

# ---- 5. 安装本项目代码（packages/ -> import mythic），可编辑模式 ---------------
. (Join-Path $PSScriptRoot 'env_vars.ps1')   # 环境刚建好，重新设置 PATH 与 GDAL/PROJ 变量
Write-Step "以可编辑模式安装本项目代码（pyproject.toml，包名 mythic）..."
Push-Location $MythicRoot
try {
    & $MythicPython -m pip install -e . --no-deps --no-build-isolation --quiet
    if ($LASTEXITCODE -ne 0) { throw "安装本项目代码失败（退出码 $LASTEXITCODE）" }
} finally { Pop-Location }

# ---- 6. 自检 ----------------------------------------------------------------
if (-not $SkipCheck) {
    Write-Step "运行自检 scripts\check_env.py ..."
    & $MythicPython (Join-Path $PSScriptRoot 'check_env.py')
    if ($LASTEXITCODE -ne 0) { throw "自检未通过（退出码 $LASTEXITCODE），详见上方输出与 environment\env_report.json" }
}

# ---- 7. 耗时与占用 ----------------------------------------------------------
$stopwatch.Stop()
Write-Step ("完成。耗时 {0:N1} 分钟" -f $stopwatch.Elapsed.TotalMinutes)
# 环境目录里大部分文件是包缓存的硬链接，按目录直接相加会重复计算，所以用 disk_usage.py 去重统计
$usage = & $MythicPython (Join-Path $PSScriptRoot 'disk_usage.py') $env:CONDA_PKGS_DIRS (Join-Path $MythicRoot 'environment')
foreach ($line in $usage) { Write-Step "磁盘占用：$line" }
Write-Step "在环境中运行命令：powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python 脚本.py"
