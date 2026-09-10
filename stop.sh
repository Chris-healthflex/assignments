#!/usr/bin/env bash

echo "=============================================="
echo " Stopping Stance Health Clinical Pipeline"
echo "=============================================="

# 1. Stop Docker Compose containers if running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[*] Stopping Docker Compose containers..."
  docker compose down 2>/dev/null || true
fi

# 2. Stop local backend process if PID file exists
if [ -f .api.pid ]; then
  PID=$(cat .api.pid)
  echo "[*] Stopping local API process (PID $PID)..."
  kill $PID 2>/dev/null || taskkill //F //PID $PID 2>/dev/null || true
  rm -f .api.pid
fi

# 3. Stop local frontend process if PID file exists
if [ -f .frontend.pid ]; then
  PID=$(cat .frontend.pid)
  echo "[*] Stopping local Frontend process (PID $PID)..."
  kill $PID 2>/dev/null || taskkill //F //PID $PID 2>/dev/null || true
  rm -f .frontend.pid
fi

# 4. Cleanup any lingering processes on ports 8000 and 3000
echo "[*] Ensuring ports 8000 and 3000 are freed..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp 2>/dev/null || true
  fuser -k 3000/tcp 2>/dev/null || true
fi

echo "[+] All services have been stopped."
echo "=============================================="
