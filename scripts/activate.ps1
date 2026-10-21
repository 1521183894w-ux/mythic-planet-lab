# 打开一个已激活项目环境的 PowerShell 窗口（给人手动操作用）：
#
#   powershell -NoExit -ExecutionPolicy Bypass -File scripts\activate.ps1
#
# 之后在这个窗口里直接输入 python、pytest 即可。关闭窗口即失效，不影响系统设置。

. (Join-Path $PSScriptRoot 'env_vars.ps1')

if (-not (Test-Path $MythicPython)) {
    Write-Warning "项目环境不存在：$MythicEnvPrefix。请先运行 scripts\setup_env.ps1。"
}

function global:prompt { "(mythic) PS $($executionContext.SessionState.Path.CurrentLocation)> " }
Set-Location $MythicRoot
Write-Host "已激活项目环境：$MythicEnvPrefix" -ForegroundColor Green
