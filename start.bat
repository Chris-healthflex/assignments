@echo off
setlocal
echo ==============================================
echo  Starting Stance Health Clinical Pipeline
echo ==============================================

if not exist .env (
    if exist .env.example (
        echo [*] Creating .env from .env.example...
        copy .env.example .env >nul
        echo [!] Created .env. Add your LLM API key if desired.
    )
)

where docker >nul 2>&1
if not errorlevel 1 (
    docker ps >nul 2>&1
    if not errorlevel 1 (
        echo [*] Docker daemon running. Launching via Docker Compose...
        docker compose up -d --build
        echo.
        echo [+] Backend: http://localhost:8000
        echo [+] Docs:    http://localhost:8000/docs
        echo [+] Mongo:   localhost:27017
        goto start_frontend
    )
)

echo [!] Docker daemon is not running. Starting in local mode...
echo [*] Starting FastAPI backend via uvicorn on :8000...
start "Stance Health API" cmd /c ".venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo [+] Backend running at http://localhost:8000

:start_frontend
if exist frontend (
    echo [*] Starting React frontend dev server on :3000...
    cd frontend
    if not exist node_modules (
        call npm install
    )
    start "Stance Health UI" cmd /c "npm run dev -- --host --port 3000"
    cd ..
    echo [+] Frontend running at http://localhost:3000
)

echo.
echo ==============================================
echo  Pipeline is live! Use stop.bat to terminate.
echo ==============================================
endlocal
