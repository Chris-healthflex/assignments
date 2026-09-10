@echo off
echo ==============================================
echo  Cleaning Stance Health Project Artifacts
echo ==============================================

call stop.bat >nul 2>&1

where docker >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Cleaning Docker containers and volumes...
    docker compose down -v --remove-orphans >nul 2>&1
)

echo [*] Cleaning Python caches and temporary files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1
del /s /q *.pyc *.pyo >nul 2>&1
if exist .pytest_cache rd /s /q .pytest_cache >nul 2>&1

echo [*] Cleaning frontend build artifacts...
if exist frontend\dist rd /s /q frontend\dist >nul 2>&1
if exist frontend\.vite rd /s /q frontend\.vite >nul 2>&1

echo [+] Environment cleaned.
echo ==============================================
