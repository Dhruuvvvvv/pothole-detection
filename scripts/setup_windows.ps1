# RoadGuard AI — Windows Environment Setup Script (PowerShell)
# Run with: powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== RoadGuard AI: Initializing Windows Environment ===" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[1/4] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "[2/4] Upgrading pip and installing backend + ML dependencies..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r ".\backend\requirements.txt"

if (-not (Test-Path ".\.env")) {
    Write-Host "[3/5] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".\.env.example" ".\.env"
}

if (-not (Test-Path ".\models\best.pt")) {
    Write-Host "[4/5] Downloading verified open-source YOLO26s RDD2022 checkpoint (models/best.pt)..." -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" ".\scripts\download_pretrained_rdd2022_weights.py"
}

Write-Host "[5/5] Installing frontend dependencies and verifying TypeScript build..." -ForegroundColor Yellow
Set-Location "$ProjectRoot\frontend"
npm.cmd install
npm.cmd run build
Set-Location $ProjectRoot

Write-Host "`nSetup complete! Start the application using:" -ForegroundColor Green
Write-Host "  Terminal 1: .\scripts\start_backend.cmd"
Write-Host "  Terminal 2: .\scripts\start_frontend.cmd"
