@echo off
REM ===========================================================================
REM  One-command setup and launch (Windows).
REM
REM    run.bat          create the venv, install, check the setup, serve
REM    run.bat warm     the same, but also pre-download the Whisper weights
REM    run.bat test     run the test suite
REM    run.bat check    only run the preflight checks, do not serve
REM
REM  The service listens on http://localhost:8000. Once it answers, the review
REM  interface, both API documentation views and two live endpoints open in the
REM  browser by themselves. Set PORT first if 8000 is already taken, or
REM  OPEN_BROWSER=0 to start the server without opening anything:
REM
REM    set PORT=8080 && run.bat
REM    set OPEN_BROWSER=0 && run.bat
REM
REM  Safe to run repeatedly: the venv and the installed packages are reused,
REM  so a second run reaches the server in seconds.
REM
REM  macOS and Linux: use ./run.sh, which does the same thing.
REM ===========================================================================
setlocal EnableDelayedExpansion

REM Work from the repository root regardless of where this was invoked.
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.deps-installed"

REM The port is set here rather than left to uvicorn's default, so the address
REM printed below is the address actually being served. Respects PORT if the
REM caller already set one, which is the way out when 8000 is occupied.
if not defined PORT set "PORT=8000"
set "URL=http://localhost:%PORT%"

echo.
echo  Clinical First Assessment - setup and launch
echo  ============================================
echo.

REM --------------------------------------------------------------------------
REM 1. Find a Python interpreter.
REM    The py launcher ships with python.org installs and is the reliable way
REM    to pick a version; bare "python" on Windows may be the Store stub that
REM    only opens the Microsoft Store.
REM --------------------------------------------------------------------------
if exist "%PY%" goto :have_venv

set "BOOTSTRAP="
py -3.14 --version >nul 2>&1 && set "BOOTSTRAP=py -3.14"
if not defined BOOTSTRAP py -3 --version >nul 2>&1 && set "BOOTSTRAP=py -3"
if not defined BOOTSTRAP python --version >nul 2>&1 && set "BOOTSTRAP=python"

if not defined BOOTSTRAP (
    echo  [FAIL] No Python interpreter found.
    echo.
    echo         Install Python 3.14 from https://python.org/downloads/
    echo         and tick "Add python.exe to PATH" during setup.
    echo.
    exit /b 1
)

echo  [ .. ] Creating virtual environment in %VENV%
%BOOTSTRAP% -m venv "%VENV%"
if errorlevel 1 (
    echo  [FAIL] Could not create the virtual environment.
    exit /b 1
)
echo  [ OK ] Virtual environment created

:have_venv

REM --------------------------------------------------------------------------
REM 2. Install dependencies.
REM    Reinstalling every run means waiting for pip to resolve 30 packages when
REM    nothing has changed. The stamp is a *copy* of requirements.txt, compared
REM    byte for byte rather than by timestamp, because git checkout rewrites
REM    mtimes and would make an unchanged file look newer than the stamp.
REM --------------------------------------------------------------------------
set "NEEDS_INSTALL=1"
if exist "%STAMP%" (
    fc /b "requirements.txt" "%STAMP%" >nul 2>&1 && set "NEEDS_INSTALL="
)

if defined NEEDS_INSTALL (
    echo  [ .. ] Installing dependencies ^(a few minutes on a first run^)
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [FAIL] Dependency installation failed. Scroll up for the reason.
        echo         A common cause is an older Python without prebuilt wheels
        echo         for these pins - 3.14 is what they target.
        exit /b 1
    )
    copy /y "requirements.txt" "%STAMP%" >nul
    echo  [ OK ] Dependencies installed
) else (
    echo  [ OK ] Dependencies already installed
)

REM --------------------------------------------------------------------------
REM 3. Dispatch on the requested mode.
REM --------------------------------------------------------------------------
if /i "%~1"=="test" goto :run_tests
if /i "%~1"=="warm" goto :run_warm
if /i "%~1"=="check" goto :run_check
goto :run_serve


:run_tests
echo.
"%PY%" -m pytest
exit /b %errorlevel%


:run_warm
"%PY%" -m app.doctor --warm
if errorlevel 1 exit /b 1
goto :serve


:run_check
"%PY%" -m app.doctor
exit /b %errorlevel%


:run_serve
"%PY%" -m app.doctor
if errorlevel 1 (
    REM The doctor already printed what is missing and why. Adding a second
    REM error message here would only bury it.
    exit /b 1
)

:serve
echo.
echo  ============================================================
echo   Running at  %URL%/ui/
echo  ============================================================
echo.
echo   Review interface   %URL%/ui/
echo   API explorer       %URL%/docs
echo   API reference      %URL%/redoc
echo   Health check       %URL%/health
echo   Saved assessments  %URL%/assessments
echo.
echo   Press Ctrl+C to stop.
echo.
REM Opens the tabs above once the server answers. Backgrounded with /b so it
REM waits alongside uvicorn instead of before it: uvicorn has not bound the
REM port yet at this point, so opening them here would land on a dead socket.
if not "%OPEN_BROWSER%"=="0" start "" /b "%PY%" -m app.browser
REM Bound to 127.0.0.1 rather than 0.0.0.0 on purpose: this process handles
REM patient audio, and it has no business being reachable from the network
REM until someone decides that deliberately.
"%PY%" -m uvicorn app.main:app --reload --host 127.0.0.1 --port %PORT%
exit /b %errorlevel%
