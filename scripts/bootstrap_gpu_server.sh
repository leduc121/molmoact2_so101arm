#!/usr/bin/env bash
# Idempotent setup for an Ubuntu CUDA host. Run: bash scripts/bootstrap_gpu_server.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver/GPU is unavailable. Rent a CUDA Ubuntu instance first." >&2
  exit 1
fi

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v python3.12 >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: Python 3.12 is required for this project's LeRobot dependency.
Choose an Ubuntu image with Python 3.12, or install it, then rerun this script.
EOF
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

# RTX 3090 drivers supporting CUDA 12.2 are compatible with these CUDA 12.1 wheels.
# Installing torch first prevents requirements files from selecting a CPU-only wheel.
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Set MOLMOACT_API_TOKEN before starting the server."
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"PyTorch GPU: {name}; compute capability: {capability[0]}.{capability[1]}")
if capability < (8, 0):
    raise SystemExit("This GPU lacks native BF16 support; select an Ampere-or-newer GPU instead.")
PY

echo
echo "Setup complete. Edit .env, then run:"
echo "  bash scripts/run_gpu_server.sh"
