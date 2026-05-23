<#
run_all.ps1
-----------
Launcher per run_all.py.

- Setta HADOOP_HOME e aggiunge C:\hadoop\bin al PATH (richiesto da Spark
  su Windows per winutils.exe / hadoop.dll).
- Usa il Python del venv locale (.venv\Scripts\python.exe).
- Inoltra eventuali argomenti a run_all.py (es. il worker: "23" o "ruben").

Uso:
    .\run_all.ps1                # default
    .\run_all.ps1 23             # worker ass3_23-05.py
    .\run_all.ps1 ruben          # worker ass3_ruben.py
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$env:HADOOP_HOME = "C:\hadoop"
if (-not ($env:Path -split ";" | Where-Object { $_ -ieq "C:\hadoop\bin" })) {
    $env:Path = "C:\hadoop\bin;" + $env:Path
}

$python = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv non trovato: $python" }

Write-Host "[launcher] HADOOP_HOME=$env:HADOOP_HOME"
Write-Host "[launcher] python      =$python"
Write-Host "[launcher] script      =run_all.py $Args"

& $python -u (Join-Path $here "run_all.py") @Args
exit $LASTEXITCODE
