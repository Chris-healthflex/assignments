Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Starting Stance Health Clinical Pipeline" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Ensure .env exists
if (-not (Test-Path .env)) {
    if (Test-Path .env.example) {
        Write-Host "[*] Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item .env.example .env
    }
}

# 2. Check if Docker is running
$dockerRunning = $false
try {
    $dockerInfo = docker ps 2>$null
    if ($LASTEXITCODE -eq 0) {
        $dockerRunning = $true
    }
} catch {}

if ($dockerRunning) {
    Write-Host "[*] Docker daemon detected. Launching via Docker Compose..." -ForegroundColor Green
    docker compose up -d --build
    Write-Host "`n[+] Backend API: http://localhost:8000" -ForegroundColor Green
    Write-Host "[+] API Docs:    http://localhost:8000/docs" -ForegroundColor Green
    Write-Host "[+] MongoDB:     localhost:27017" -ForegroundColor Green
} else {
    Write-Host "[!] Docker daemon not active. Starting in local mode..." -ForegroundColor Yellow

    # Start FastAPI backend via uvicorn
    Write-Host "[*] Starting FastAPI backend on :8000..." -ForegroundColor Cyan
    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000" -WindowStyle Hidden
    Write-Host "[+] Backend live at http://localhost:8000" -ForegroundColor Green
    Write-Host "[+] API Docs live at http://localhost:8000/docs" -ForegroundColor Green
}

# 3. Start React frontend
if (Test-Path frontend) {
    Write-Host "[*] Starting React frontend dev server on :3000..." -ForegroundColor Cyan
    Set-Location frontend
    if (-not (Test-Path node_modules)) {
        npm install
    }
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev -- --host --port 3000" -WindowStyle Hidden
    Set-Location ..
    Write-Host "[+] Frontend UI live at http://localhost:3000" -ForegroundColor Green
}

Write-Host "`n==============================================" -ForegroundColor Cyan
Write-Host " Pipeline is live! Open: http://localhost:3000" -ForegroundColor Green
Write-Host " Use .\stop.ps1 to terminate." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
