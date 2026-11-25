# RoadGuard AI — Run Automated Pytest Suite
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& "$ProjectRoot\.venv\Scripts\pytest.exe" -v
