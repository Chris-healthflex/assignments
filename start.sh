#!/usr/bin/env bash
set -e

echo "=============================================="
echo " Starting Stance Health Clinical Pipeline"
echo "=============================================="

# 1. Ensure .env exists
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "[*] Creating .env from .env.example..."
    cp .env.example .env
    echo "[!] Please remember to add your LLM API key (e.g. GEMINI_API_KEY) in .env"
  fi
fi

# 2. Check if Docker is available and daemon is running
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
  echo "[*] Docker daemon detected. Launching full stack via Docker Compose..."
  docker compose up -d --build

  echo ""
  echo "[+] Services started successfully!"
  echo "    - FastAPI Backend: http://localhost:8000"
  echo "    - OpenAPI Docs:    http://localhost:8000/docs"
  echo "    - MongoDB:         localhost:27017"
  echo ""

  # Start frontend if npm is available
  if [ -d "frontend" ] && command -v npm >/dev/null 2>&1; then
    echo "[*] Starting React frontend dev server..."
    cd frontend
    if [ ! -d "node_modules" ]; then
      npm install
    fi
    npm run dev -- --host &
    echo "    - React UI:        http://localhost:3000"
    cd ..
  fi

else
  echo "[!] Docker is not running or not installed. Starting in local mode..."

  # Check python venv
  PY_BIN=".venv/Scripts/python.exe"
  if [ ! -f "$PY_BIN" ]; then
    PY_BIN=".venv/bin/python"
  fi
  if [ ! -f "$PY_BIN" ]; then
    PY_BIN="python"
  fi

  echo "[*] Starting FastAPI app via uvicorn in background using $PY_BIN..."
  nohup "$PY_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
  API_PID=$!
  echo $API_PID > .api.pid
  echo "[+] Backend running (PID $API_PID) at http://localhost:8000"

  if [ -d "frontend" ] && command -v npm >/dev/null 2>&1; then
    echo "[*] Starting React frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
      npm install
    fi
    nohup npm run dev -- --host --port 3000 > ../vite.log 2>&1 &
    FRONT_PID=$!
    echo $FRONT_PID > ../.frontend.pid
    echo "[+] Frontend running (PID $FRONT_PID) at http://localhost:3000"
    cd ..
  fi
fi

echo "=============================================="
echo " Pipeline is live! Use ./stop.sh to terminate."
echo "=============================================="
