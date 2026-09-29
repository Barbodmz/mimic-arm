#!/usr/bin/env bash
# Install mimic-arm: LeRobot 0.6.1, the ALOHA simulator, and a headless MuJoCo renderer.
#
# Usage:
#   bash setup.sh            # CPU PyTorch if no NVIDIA GPU is visible, otherwise CUDA
#   bash setup.sh --cpu      # force the CPU build (verified by this repo's smoke test)
#   bash setup.sh --cuda     # force the CUDA 13 wheel that lerobot 0.6.1 resolves on PyPI
#
# Python 3.12 or newer is required (LeRobot 0.6 dropped 3.10 and 3.11).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
MODE="auto"

for arg in "$@"; do
  case "$arg" in
    --cpu) MODE="cpu" ;;
    --cuda) MODE="cuda" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
  echo "LeRobot 0.6.1 needs Python 3.12 or newer. '$PYTHON' is:" >&2
  "$PYTHON" --version >&2 || true
  exit 1
fi

# Headless rendering libraries. libOSMesa is the CPU software renderer.
# libEGL is what MuJoCo uses on a machine with a GPU. ffmpeg is used when
# torchcodec falls back, and PyAV (installed via pip) writes the eval mp4s.
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
    SUDO="sudo"
  else
    SUDO=""
  fi
  echo "Installing system packages for headless MuJoCo (EGL + OSMesa)..."
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libosmesa6 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglew2.2 \
    libglfw3 \
    ffmpeg \
    pkg-config \
    python3-venv \
    python3-dev
else
  echo "apt-get not found. Install libOSMesa and libEGL yourself, then re-run." >&2
  echo "On Debian/Ubuntu the packages are: libosmesa6 libegl1 libgl1 libglib2.0-0 ffmpeg" >&2
fi

if [[ "$MODE" == "auto" ]]; then
  if [[ -e /dev/nvidia0 || -e /proc/driver/nvidia/version ]]; then
    MODE="cuda"
  else
    MODE="cpu"
  fi
fi
echo "PyTorch build: $MODE"

"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Install torch BEFORE the rest so the resolver sees a version lerobot accepts
# (torch>=2.7,<2.12). Installing lerobot on its own pulls the PyPI default,
# which is currently the CUDA 13 wheel, even on a CPU machine.
install_torch() {
  if [[ "$MODE" == "cpu" ]]; then
    python -m pip install "torch==2.11.0+cpu" "torchvision==0.26.0+cpu" \
      --index-url https://download.pytorch.org/whl/cpu
  else
    python -m pip install "torch==2.11.0" "torchvision==0.26.0" \
      --index-url https://download.pytorch.org/whl/cu130
  fi
}

install_torch
python -m pip install -r "$ROOT/requirements.txt"
# requirements.txt can still swap the torch build. Put the chosen wheel back,
# then restore numpy and fsspec. The CPU torch index otherwise upgrades numpy
# past 2.2.x, and lerobot 0.6.1 rejects numpy>=2.3.
install_torch
python -m pip install "numpy==2.2.6" "fsspec==2026.2.0"

python - <<'PY'
import lerobot
import torch
print("lerobot", lerobot.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo
echo "Done. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then, from the repo root:"
echo "  python train.py --help"
echo "  python evaluate.py --help"
