# Checkers Alpha launcher (run by start.bat in the project root).
#
# Builds the UI when it's missing or older than its source, starts the
# server (python serve.py, port 8020) in this window, and opens the
# browser once it answers. Closing the window stops the server.
#
# If a server is already running: one started after the latest change to
# the Python code is reused (the browser just opens); an older one would
# serve stale code, so it's stopped, with its window, and replaced.
# Set CHECKERS_NO_BROWSER=1 to skip opening the browser (for testing).

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = 'C:\Users\biz\miniconda3\envs\alpha_checkers2\python.exe'
$port = 8020
$url = "http://127.0.0.1:$port"
$browser = -not $env:CHECKERS_NO_BROWSER
$Host.UI.RawUI.WindowTitle = 'Checkers Alpha'

# Whether the server answers.
function Test-Server {
    try {
        Invoke-WebRequest -UseBasicParsing "$url/api/players" `
            -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

# The process listening on the port, or $null.
function Get-Listener {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $conn) { return $null }
    return Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($conn.OwningProcess)"
}

# When the Python code the server runs last changed.
function Get-CodeTime {
    $files = Get-ChildItem server, core -Recurse -Filter *.py -File
    $files += Get-Item serve.py
    return ($files | Measure-Object LastWriteTime -Maximum).Maximum
}

# Stop a server and the launcher windows it runs in (its PowerShell and
# start.bat parents), so no stale window is left behind.
function Stop-Server($proc) {
    $ids = @($proc.ProcessId)
    $parent = $proc.ParentProcessId
    while ($parent) {
        $p = Get-CimInstance Win32_Process -Filter "ProcessId = $parent"
        if (-not $p -or $p.CommandLine -notmatch 'start\.(ps1|bat)') {
            break
        }
        $ids += $p.ProcessId
        $parent = $p.ParentProcessId
    }
    foreach ($id in $ids) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    for ($i = 0; $i -lt 20 -and (Get-Listener); $i++) {
        Start-Sleep -Milliseconds 250
    }
}

# Build the UI if it's missing or any source file is newer than the build
# (the server reads it from disk, so a running server picks it up).
$built = Get-Item 'ui\dist\index.html' -ErrorAction SilentlyContinue
$stale = -not $built
if ($built) {
    $sources = 'ui\src', 'ui\public', 'ui\index.html'
    $stale = [bool](Get-ChildItem $sources -Recurse -File |
        Where-Object { $_.LastWriteTime -gt $built.LastWriteTime })
}
if ($stale) {
    Write-Host 'Building the UI...'
    Push-Location ui
    try {
        if (-not (Test-Path node_modules)) {
            npm install --no-fund --no-audit
        }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'the UI build failed' }
    } finally {
        Pop-Location
    }
}

# A server already running: reuse it if current, replace it if stale.
$running = Get-Listener
if ($running) {
    if ($running.CommandLine -notmatch 'serve\.py') {
        Write-Host "Port $port is taken by another program:"
        Write-Host "  $($running.CommandLine)"
        Write-Host 'Close it (or see E:\Projects\.claude for the port list).'
        exit 1
    }
    if ($running.CreationDate -gt (Get-CodeTime) -and (Test-Server)) {
        if ($browser) { Start-Process $url }
        exit 0
    }
    Write-Host 'The running server is older than the code: restarting it.'
    Stop-Server $running
}

# Open the browser as soon as the server answers, without waiting here.
if ($browser) {
    Start-Job -ArgumentList $url -ScriptBlock {
        param($url)
        for ($i = 0; $i -lt 240; $i++) {
            try {
                Invoke-WebRequest -UseBasicParsing "$url/api/players" `
                    -TimeoutSec 2 | Out-Null
                Start-Process $url
                break
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } | Out-Null
}

Write-Host ''
Write-Host "  Checkers Alpha: $url"
Write-Host '  Close this window to stop the server.'
Write-Host ''
& $python serve.py
