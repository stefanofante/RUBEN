<#
install_winutils.ps1
--------------------
Installs winutils.exe + hadoop.dll for the Hadoop version used by the
PySpark in the current Python environment, then sets HADOOP_HOME and
updates PATH (both for the current session and persistently at User scope).

Usage:
    # from the project root, with the venv active:
    powershell -ExecutionPolicy Bypass -File .\install_winutils.ps1

    # or, to force a specific Hadoop version:
    powershell -ExecutionPolicy Bypass -File .\install_winutils.ps1 -HadoopVersion 3.3.6
#>

[CmdletBinding()]
param(
    [string]$HadoopVersion = "",
    [string]$InstallDir    = "C:\hadoop",
    [string]$PythonExe     = ""
)

$ErrorActionPreference = "Stop"

# ---------- 1) detect python ----------
if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        "python"
    )
    foreach ($c in $candidates) {
        try {
            $null = & $c --version 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonExe = $c; break }
        } catch { }
    }
}
if (-not $PythonExe) { throw "No python interpreter found. Pass -PythonExe explicitly." }
Write-Host "[winutils] python = $PythonExe"

# ---------- 2) detect hadoop version from pyspark ----------
if (-not $HadoopVersion) {
    $detect = @'
import importlib, os, re, sys
try:
    import pyspark
except Exception as e:
    print("ERR:", e); sys.exit(2)
spark_home = os.path.dirname(pyspark.__file__)
jars = os.path.join(spark_home, "jars")
ver = None
if os.path.isdir(jars):
    for f in os.listdir(jars):
        m = re.match(r"hadoop-client-api-(\d+\.\d+\.\d+)\.jar$", f)
        if m:
            ver = m.group(1); break
        m = re.match(r"hadoop-common-(\d+\.\d+\.\d+)\.jar$", f)
        if m:
            ver = m.group(1)
print("PYSPARK:", pyspark.__version__)
print("HADOOP:", ver or "unknown")
'@
    $out = & $PythonExe -c $detect
    Write-Host $out
    $line = $out | Where-Object { $_ -like "HADOOP:*" } | Select-Object -First 1
    if ($line) { $HadoopVersion = ($line -split ":")[1].Trim() }
    if (-not $HadoopVersion -or $HadoopVersion -eq "unknown") {
        Write-Warning "Could not detect Hadoop version from PySpark. Falling back to 3.3.6."
        $HadoopVersion = "3.3.6"
    }
}
Write-Host "[winutils] hadoop version = $HadoopVersion"

# ---------- 3) pick available release in cdarlint/winutils ----------
# Known available tags in https://github.com/cdarlint/winutils (subset).
$known = @(
    "3.4.1","3.4.0","3.3.6","3.3.5","3.3.4","3.3.2","3.3.1","3.3.0",
    "3.2.4","3.2.3","3.2.2","3.2.1","3.2.0","3.1.2","3.1.0","3.0.0"
)
function Pick-Release([string]$wanted, [string[]]$pool) {
    if ($pool -contains $wanted) { return $wanted }
    # match major.minor, take highest patch <= wanted patch, else highest of that minor
    $w = $wanted.Split(".")
    $sameMinor = $pool | Where-Object {
        $p = $_.Split("."); $p[0] -eq $w[0] -and $p[1] -eq $w[1]
    } | Sort-Object { [version]$_ } -Descending
    if ($sameMinor.Count -gt 0) { return $sameMinor[0] }
    # fallback: highest version in pool with same major
    $sameMajor = $pool | Where-Object { $_.Split(".")[0] -eq $w[0] } |
                 Sort-Object { [version]$_ } -Descending
    if ($sameMajor.Count -gt 0) { return $sameMajor[0] }
    return $pool[0]
}
$release = Pick-Release $HadoopVersion $known
if ($release -ne $HadoopVersion) {
    Write-Host "[winutils] exact release not in repo; using nearest: $release"
}

# ---------- 4) download ----------
$binDir = Join-Path $InstallDir "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

$baseUrl = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-$release/bin"
$files   = @("winutils.exe", "hadoop.dll")

foreach ($f in $files) {
    $dest = Join-Path $binDir $f
    $url  = "$baseUrl/$f"
    Write-Host "[winutils] downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    if (-not (Test-Path $dest) -or (Get-Item $dest).Length -lt 1024) {
        throw "Download of $f failed or file too small."
    }
}
Write-Host "[winutils] files installed in $binDir"

# ---------- 5) env vars ----------
# session (current shell)
$env:HADOOP_HOME = $InstallDir
if (-not ($env:Path -split ";" | Where-Object { $_ -ieq $binDir })) {
    $env:Path += ";$binDir"
}

# persistent (User scope)
[Environment]::SetEnvironmentVariable("HADOOP_HOME", $InstallDir, "User")
$oldUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($oldUserPath -split ";" | Where-Object { $_ -ieq $binDir })) {
    $newUserPath = if ([string]::IsNullOrEmpty($oldUserPath)) { $binDir } else { "$oldUserPath;$binDir" }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    Write-Host "[winutils] PATH (User) updated."
} else {
    Write-Host "[winutils] PATH (User) already contains $binDir."
}

# ---------- 6) smoke test ----------
Write-Host ""
Write-Host "[winutils] smoke test:"
& (Join-Path $binDir "winutils.exe") ls $InstallDir
Write-Host ""
Write-Host "[winutils] DONE. HADOOP_HOME=$env:HADOOP_HOME"
Write-Host "[winutils] In new PowerShell sessions, env vars are picked up automatically."
