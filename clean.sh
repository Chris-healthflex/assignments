#!/usr/bin/env bash

echo "=============================================="
echo " Cleaning Stance Health Project Artifacts"
echo "=============================================="

# 1. First ensure all services are stopped
if [ -f "./stop.sh" ]; then
  bash ./stop.sh
fi

# 2. Docker compose down with volumes removed
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[*] Removing Docker containers, networks, and persistent volumes..."
  docker compose down -v --remove-orphans 2>/dev/null || true
fi

# 3. Clean Python bytecode and pytest caches
echo "[*] Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true

# 4. Clean Frontend build artifacts (keeps node_modules by default)
echo "[*] Removing frontend build artifacts..."
rm -rf frontend/dist frontend/.vite 2>/dev/null || true

# 5. Remove PID and temporary lock files
rm -f .api.pid .frontend.pid 2>/dev/null || true

echo ""
echo "[+] Environment cleaned successfully!"
echo "=============================================="
