#!/usr/bin/env bash
set -euo pipefail

api_port="${OCR_API_PORT:-8000}"
vllm_port="${OCR_VLLM_PORT:-8001}"
startup_timeout="${OCR_VLLM_STARTUP_TIMEOUT_SECONDS:-600}"

PYTHONPATH=./ python3 -m app.vllm_local_server &
vllm_pid=$!

cleanup() {
  if kill -0 "${vllm_pid}" >/dev/null 2>&1; then
    kill "${vllm_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + startup_timeout))
while (( SECONDS < deadline )); do
  if ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
    echo "vLLM server process exited before becoming ready." >&2
    exit 1
  fi

  if PYTHONPATH=./ python3 - <<PY
import sys
import urllib.request

port = ${vllm_port}
url = f"http://127.0.0.1:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        sys.exit(0 if 200 <= resp.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
  then
    break
  fi

  sleep 1
done

if (( SECONDS >= deadline )); then
  echo "Timed out waiting for local vLLM server readiness." >&2
  exit 1
fi

exec uvicorn main:app --host 0.0.0.0 --port "${api_port}"
