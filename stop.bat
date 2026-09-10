@echo off
echo ==============================================
echo  Stopping Stance Health Clinical Pipeline
echo ==============================================

where docker >nul 2>&1
if %errorlevel% equ 0 (
    docker compose down >nul 2>&1
)

taskkill /F /IM python.exe /FI "WINDOWTITLE eq Stance Health API*" >nul 2>&1
taskkill /F /IM node.exe /FI "WINDOWTITLE eq Stance Health UI*" >nul 2>&1

echo [+] Services stopped.
echo ==============================================
