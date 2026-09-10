Write-Host "==============================================" -ForegroundColor Red
Write-Host " Cleaning Stance Health Project Artifacts" -ForegroundColor Red
Write-Host "==============================================" -ForegroundColor Red

# 1. Stop running services
if (Test-Path .\stop.ps1) {
    & .\stop.ps1
}

# 2. Stop Docker and clean volumes
try {
    docker compose down -v --remove-orphans 2>$null
} catch {}

# 3. Clean Python caches
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -File -Include "*.pyc", "*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue }

# 4. Clean Frontend build artifacts
if (Test-Path frontend\dist) { Remove-Item -Recurse -Force frontend\dist -ErrorAction SilentlyContinue }
if (Test-Path frontend\.vite) { Remove-Item -Recurse -Force frontend\.vite -ErrorAction SilentlyContinue }

Write-Host "[+] Environment cleaned successfully." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Red
