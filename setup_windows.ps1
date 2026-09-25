$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = $null
foreach ($version in @('3.12', '3.11', '3.13', '3.14')) {
    & py -$version -c 'import sys; print(sys.executable)' *> $null
    if ($LASTEXITCODE -eq 0) {
        $python = "-$version"
        break
    }
}

if (-not $python) {
    Write-Host '未找到受支持的 64 位 Python（推荐 3.12）。' -ForegroundColor Red
    Write-Host '请从 https://www.python.org/downloads/windows/ 安装 Python 3.12 后重新运行本脚本。'
    exit 1
}

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py $python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

Write-Host ''
Write-Host '安装完成。请打开电脑版微信中的天天象棋，再双击“诊断.bat”。' -ForegroundColor Green
