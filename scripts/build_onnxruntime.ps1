Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

<#
.SYNOPSIS
Download and build ONNX Runtime v1.22.0 into target/onnxruntime.

.DESCRIPTION
Optional operator trim config can be provided with -OpsConfig or OPS_CONFIG.

.PARAMETER OpsConfig
Path to the operator config file to be passed to --include_ops_by_config

.PARAMETER Help
Show help message

.EXAMPLE
.\build_onnxruntime.ps1 -OpsConfig handpose_estimation_mediapipe/required_operators.config

.NOTES
Environment overrides:
  OUT_DIR     Where to place downloads and sources (default: <repo>/target/onnxruntime)
  ORT_VERSION ONNX Runtime tag to fetch (default: 1.22.0)
  OPS_CONFIG  Operator config path passed to --include_ops_by_config
#>

[CmdletBinding()]
param(
    [string]$OpsConfig,
    [switch]$Help
)

function Show-Usage {
    Get-Help $MyInvocation.MyCommand.Path
    exit 0
}

if ($Help) {
    Show-Usage
}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { Join-Path $root "target\onnxruntime" }
$OrtVersion = if ($env:ORT_VERSION) { $env:ORT_VERSION } else { "1.22.0" }
if (-not $OpsConfig -and $env:OPS_CONFIG) { $OpsConfig = $env:OPS_CONFIG }

$SrcDir = Join-Path $OutDir "onnxruntime-$OrtVersion"
$Archive = Join-Path $OutDir "onnxruntime-$OrtVersion.zip"

# Convert to absolute path if relative
if ($OpsConfig) {
    if (-not [System.IO.Path]::IsPathRooted($OpsConfig)) {
        $OpsConfig = Join-Path $root $OpsConfig
    }
    if (-not (Test-Path $OpsConfig)) {
        Write-Error "Specified ops config not found: $OpsConfig"
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $SrcDir)) {
    Write-Host "Downloading ONNX Runtime v$OrtVersion sources..."
    Invoke-WebRequest "https://github.com/microsoft/onnxruntime/archive/refs/tags/v$OrtVersion.zip" -OutFile $Archive
    Expand-Archive -Path $Archive -DestinationPath $OutDir -Force
}

# Fix Eigen SHA1 hash mismatch in deps.txt
$DepsFile = Join-Path $SrcDir "cmake\deps.txt"
if (Test-Path $DepsFile) {
    Write-Host "Fixing Eigen SHA1 hash in deps.txt..."
    $content = Get-Content $DepsFile -Raw
    $content = $content -replace '5ea4d05e62d7f954a46b3213f9b2535bdd866803', '51982be81bbe52572b54180454df11a3ece9a934'
    Set-Content -Path $DepsFile -Value $content -NoNewline
}

Push-Location $SrcDir

$buildArgs = @(
    "--config", "Release",
    "--parallel",
    "--skip_tests",
    "--use_full_protobuf",
    "--cmake_extra_defines", "CMAKE_C_FLAGS=-fPIC", "CMAKE_CXX_FLAGS=-fPIC"
)

if ($OpsConfig) {
    Write-Host "Using operator config at $OpsConfig"
    $buildArgs += @("--minimal_build", "--include_ops_by_config", $OpsConfig)
}

cmd /c ".\build.bat $($buildArgs -join ' ')"
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "Build failed with exit code $LASTEXITCODE"
}

Pop-Location

# Copy build artifacts to platform-independent directory
$LibDir = Join-Path $OutDir "lib"
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null

$BuildDir = Join-Path $SrcDir "build\Windows\Release"
if (Test-Path $BuildDir) {
    Write-Host "Copying build artifacts to $LibDir..."
    Copy-Item -Path "$BuildDir\*" -Destination $LibDir -Recurse -Force
    Write-Host "Build artifacts copied successfully"
} else {
    Write-Error "Build directory not found: $BuildDir"
    exit 1
}

Write-Host "ONNX Runtime build finished under $SrcDir"
Write-Host "Build artifacts available at $LibDir"
