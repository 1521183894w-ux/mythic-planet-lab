# 本项目的环境变量（被 setup_env.ps1、run.ps1、activate.ps1 点引用，不单独运行）。
#
# 作用：让 Python 环境、包缓存、临时文件和各个库的数据缓存全部落在工作区内，不写 C 盘。
# 所有设置只影响当前进程，不修改系统或用户级环境变量。
# 新增会写缓存的库时，在这里补充对应变量，并在 scripts/check_env.py 中加入位置检查。

$MythicRoot      = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MythicEnvPrefix = Join-Path $MythicRoot 'environment\env'
$MythicMamba     = Join-Path $MythicRoot 'environment\bin\micromamba.exe'
$MythicPython    = Join-Path $MythicEnvPrefix 'python.exe'
$MythicCache     = Join-Path $MythicRoot '.cache'

$env:MYTHIC_ROOT = $MythicRoot

# 代理：micromamba（libcurl）只认 HTTP(S)_PROXY 环境变量，不认 Windows 系统代理。
# 笔记本 Ezreal 上这两个变量没有写进用户/系统环境（2026-10-21 审计），用户自己开的终端里可能没有，
# 而 conda-forge 在两台机器上都必须走代理。没有变量、系统代理（WinINET）又已开启时，
# 把系统代理设为本进程的代理变量。只影响当前进程，不改用户或系统设置。
if (-not $env:HTTPS_PROXY) {
    try {
        $inet = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
        if ($inet.ProxyEnable -eq 1 -and $inet.ProxyServer) {
            # 形如 "127.0.0.1:7890"，或按协议分开写的 "http=host:port;https=host:port;socks=..."
            $server = [string]$inet.ProxyServer
            if ($server -match '(?:^|;)https=([^;]+)')    { $server = $Matches[1] }
            elseif ($server -match '(?:^|;)http=([^;]+)') { $server = $Matches[1] }
            elseif ($server -match '=')                   { $server = $null }   # 只有 socks 等，不处理
            if ($server) {
                if ($server -notmatch '^\w+://') { $server = "http://$server" }
                $env:HTTPS_PROXY = $server
                if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $server }
                if (-not $env:NO_PROXY)   { $env:NO_PROXY = 'localhost,127.0.0.1,::1,.local' }
            }
        }
    } catch { }
}

# micromamba：根目录（配置、元数据）与包缓存
$env:MAMBA_ROOT_PREFIX = Join-Path $MythicRoot 'environment\mamba'
$env:CONDA_PKGS_DIRS   = Join-Path $MythicCache 'mamba-pkgs'
# micromamba 2.9 的"分片索引"缓存不遵守 CONDA_PKGS_DIRS，会写到 %LOCALAPPDATA%\conda\pkgs（C 盘），
# 2026-10-21 实测确认。关闭分片、改用完整索引后，缓存全部落在 CONDA_PKGS_DIRS 内，求解结果相同。
$env:MAMBA_USE_SHARDED_REPODATA = 'false'

# 临时文件（默认在 C:\Users\...\AppData\Local\Temp）
$env:TEMP = Join-Path $MythicCache 'tmp'
$env:TMP  = $env:TEMP

# Python 本身
$env:PYTHONUTF8       = '1'   # 系统代码页是 GBK(936)，强制 UTF-8，避免读写中文文档出错
$env:PYTHONNOUSERSITE = '1'   # 不加载 %APPDATA%\Python 下的用户级包
$env:PIP_CACHE_DIR    = Join-Path $MythicCache 'pip'

# 各库的缓存与数据目录
$env:MPLCONFIGDIR                 = Join-Path $MythicCache 'matplotlib'
$env:NUMBA_CACHE_DIR              = Join-Path $MythicCache 'numba'
$env:PROJ_USER_WRITABLE_DIRECTORY = Join-Path $MythicCache 'proj'
$env:XDG_DATA_HOME                = Join-Path $MythicCache 'xdg-data'    # cartopy 的 Natural Earth 下载目录基于它
$env:XDG_CACHE_HOME               = Join-Path $MythicCache 'xdg-cache'

# platformdirs（pooch、gplately 等很多库用它决定缓存/数据目录）在 Windows 上默认用
# %LOCALAPPDATA% 和 %APPDATA%（都在 C 盘）。platformdirs >= 4.x 支持以下覆盖变量。
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = Join-Path $MythicCache 'platformdirs\local'
$env:WIN_PD_OVERRIDE_APPDATA       = Join-Path $MythicCache 'platformdirs\roaming'

# 本项目的运行日志目录（gplately 导入时会在当前目录建 gplately.log/ptt.log，见 docs/environment.md）
$MythicLogs = Join-Path $MythicCache 'logs'

foreach ($d in @($env:MAMBA_ROOT_PREFIX, $env:CONDA_PKGS_DIRS, $env:TEMP, $env:PIP_CACHE_DIR,
                 $env:MPLCONFIGDIR, $env:NUMBA_CACHE_DIR, $env:PROJ_USER_WRITABLE_DIRECTORY,
                 $env:XDG_DATA_HOME, $env:XDG_CACHE_HOME,
                 $env:WIN_PD_OVERRIDE_LOCAL_APPDATA, $env:WIN_PD_OVERRIDE_APPDATA, $MythicLogs)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# 环境已建好时：模拟 conda 激活（PATH 与 GDAL/PROJ 数据目录），重复点引用不会重复添加
if (Test-Path $MythicPython) {
    $lib = Join-Path $MythicEnvPrefix 'Library'
    $env:CONDA_PREFIX = $MythicEnvPrefix
    $env:GDAL_DATA    = Join-Path $lib 'share\gdal'
    $env:PROJ_DATA    = Join-Path $lib 'share\proj'
    $plugins = Join-Path $lib 'lib\gdalplugins'
    if (Test-Path $plugins) { $env:GDAL_DRIVER_PATH = $plugins }

    $prepend = @(
        $MythicEnvPrefix,
        (Join-Path $lib 'mingw-w64\bin'),
        (Join-Path $lib 'usr\bin'),
        (Join-Path $lib 'bin'),
        (Join-Path $MythicEnvPrefix 'Scripts'),
        (Join-Path $MythicEnvPrefix 'bin'),
        (Join-Path $MythicRoot 'environment\bin')
    )
    $current = $env:PATH -split ';' | Where-Object { $_ -and ($prepend -notcontains $_) }
    $env:PATH = (@($prepend) + @($current)) -join ';'
}
