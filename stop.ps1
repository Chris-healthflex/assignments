Write-Host "==============================================" -ForegroundColor Yellow
Write-Host " Stopping Stance Health Clinical Pipeline" -ForegroundColor Yellow
Write-Host "==============================================" -ForegroundColor Yellow

# Stop Docker containers
try {
    docker compose down 2>$null
} catch {}

# Stop local uvicorn and vite processes
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*vite*" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Kill any lingering listeners on 8000 and 3000
$ports = @(8000, 3000)
foreach ($p in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "[+] All services have been stopped." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Yellow
