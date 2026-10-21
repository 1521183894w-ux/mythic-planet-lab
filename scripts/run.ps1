# 在项目环境中运行一条命令（不需要先激活）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python scripts\check_env.py
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 python -m pytest
#
# 注意：这里故意不写 param()，用 $args 原样接收所有参数，这样 "-m" 之类的参数不会被
# PowerShell 当成本脚本的参数名。

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env_vars.ps1')

if (-not (Test-Path $MythicPython)) {
    Write-Error "项目环境不存在：$MythicEnvPrefix。请先运行 scripts\setup_env.ps1。"
    exit 1
}

$command = @($args)
if ($command.Count -eq 0) { $command = @('python') }
$exe  = $command[0]
$rest = @()
if ($command.Count -gt 1) { $rest = $command[1..($command.Count - 1)] }

& $exe @rest
exit $LASTEXITCODE
