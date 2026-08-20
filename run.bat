@echo off
REM ===========================================================================
REM  One-command setup and launch (Windows).
REM
REM    run.bat          create the venv, install, check the setup, serve
REM    run.bat warm     the same, but also pre-download the Whisper weights
REM    run.bat test     run the test suite
REM    run.bat check    only run the preflight checks, do not serve
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
echo  Starting the service. Press Ctrl+C to stop.
echo  Review interface: http://localhost:8000/ui/
echo  API explorer:     http://localhost:8000/docs
echo.
"%PY%" -m uvicorn app.main:app --reload
exit /b %errorlevel%
