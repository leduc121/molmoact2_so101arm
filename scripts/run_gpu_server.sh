#!/usr/bin/env bash
# Start the authenticated inference server after bootstrap has completed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f .venv/bin/activate || ! -f .env ]]; then
  echo "ERROR: run bash scripts/bootstrap_gpu_server.sh first." >&2
  exit 1
fi

source .venv/bin/activate
set -a
source .env
set +a

if [[ -z "${MOLMOACT_API_TOKEN:-}" || "${MOLMOACT_API_TOKEN}" == "replace-with-a-long-random-secret" ]]; then
  echo "ERROR: set a real MOLMOACT_API_TOKEN in .env." >&2
  exit 1
fi

exec uvicorn server:app --host 127.0.0.1 --port "${MOLMOACT_PORT:-8000}"
