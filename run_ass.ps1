<#
run_ass.ps1
-----------
Launcher per lanciare uno dei worker (ass3_23-05.py / ass3_ruben.py)
direttamente con un preset, senza passare per run_all.py.

- Setta HADOOP_HOME e aggiunge C:\hadoop\bin al PATH.
- Setta le env var BENCH_* lette dallo script.
- Usa il Python del venv locale.

Uso:
    # sanity (default) su ass3_23-05.py
    .\run_ass.ps1

    # preset medium su ass3_23-05.py
    .\run_ass.ps1 -Preset medium

    # preset full memory-optimized
    .\run_ass.ps1 -Preset full

    # cambiare worker
    .\run_ass.ps1 -Worker ruben -Preset sanity

    # preset custom
    .\run_ass.ps1 -NValues "1000,5000,20000" -MemOpt false -Suffix "_custom"
#>

[CmdletBinding(DefaultParameterSetName = "Preset")]
param(
    [ValidateSet("23", "ruben")]
    [string]$Worker = "23",

    [Parameter(ParameterSetName = "Preset")]
    [ValidateSet("sanity", "medium", "full")]
    [string]$Preset = "sanity",

    [Parameter(ParameterSetName = "Custom")]
    [string]$NValues,

    [Parameter(ParameterSetName = "Custom")]
    [ValidateSet("true", "false")]
    [string]$MemOpt,

    [Parameter(ParameterSetName = "Custom")]
    [string]$Suffix
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$workers = @{
    "23"    = "ass3_23-05.py"
    "ruben" = "ass3_ruben.py"
}
$script = Join-Path $here $workers[$Worker]
if (-not (Test-Path $script)) { throw "Worker script non trovato: $script" }

# preset -> (n_values, mem_opt, suffix)
$presets = @{
    "sanity" = @("1000,10000",                              "false", "_sanity")
    "medium" = @("10000,50000,100000",                      "false", "_medium")
    "full"   = @("10000,50000,100000,500000,1000000",       "true",  "_full")
}

if ($PSCmdlet.ParameterSetName -eq "Preset") {
    $p = $presets[$Preset]
    $NValues = $p[0]; $MemOpt = $p[1]; $Suffix = $p[2]
} else {
    if (-not $NValues) { throw "-NValues richiesto nel set Custom" }
    if (-not $MemOpt)  { $MemOpt = "false" }
    if (-not $Suffix)  { $Suffix = "_custom" }
}

# winutils env
$env:HADOOP_HOME = "C:\hadoop"
if (-not ($env:Path -split ";" | Where-Object { $_ -ieq "C:\hadoop\bin" })) {
    $env:Path = "C:\hadoop\bin;" + $env:Path
}

# bench env
$env:BENCH_N_VALUES = $NValues
$env:BENCH_MEM_OPT  = $MemOpt
$env:BENCH_SUFFIX   = $Suffix

$python = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv non trovato: $python" }

Write-Host "[launcher] HADOOP_HOME    =$env:HADOOP_HOME"
Write-Host "[launcher] python         =$python"
Write-Host "[launcher] script         =$script"
Write-Host "[launcher] BENCH_N_VALUES =$env:BENCH_N_VALUES"
Write-Host "[launcher] BENCH_MEM_OPT  =$env:BENCH_MEM_OPT"
Write-Host "[launcher] BENCH_SUFFIX   =$env:BENCH_SUFFIX"
Write-Host ""

& $python -u $script
exit $LASTEXITCODE
