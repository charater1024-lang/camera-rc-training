#!/usr/bin/env bash
# Prepared installer only. Run manually on the future Ubuntu target, never as root.
set -euo pipefail

with_detection=0
case "${1:-}" in
  --with-detection) with_detection=1; shift ;;
  -h|--help)
    printf '%s\n' 'Usage: bash scripts/setup_ubuntu.sh [--with-detection]' \
      'Ubuntu 24.04 LTS x86_64 / Python 3.12 / working NVIDIA driver required.' \
      'Creates .venv-linux; does not install drivers, apt packages, SSH or services.'
    exit 0 ;;
esac
if (( $# != 0 )); then
  printf '%s\n' 'Unknown option. Use --help.' >&2
  exit 2
fi
if (( EUID == 0 )); then
  printf '%s\n' 'Run as your ordinary Ubuntu account, without sudo.' >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This installer is for Ubuntu x86_64, not Windows or Jetson/aarch64.' >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python3.12}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf '%s\n' 'Python 3.12 is required. See docs/HEADLESS_UBUNTU.md.' >&2
  exit 1
fi
"$python_bin" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Python 3.12 on the Ubuntu 24.04 target.')
release = platform.freedesktop_os_release()
if release.get('ID') != 'ubuntu' or release.get('VERSION_ID') != '24.04':
    raise SystemExit('This installer targets Ubuntu 24.04 LTS; other releases are unverified.')
PY
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' 'nvidia-smi is unavailable. Check the existing Ubuntu NVIDIA driver first; this script does not replace it.' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv_dir="$project_root/.venv-linux"
venv_python="$venv_dir/bin/python"
if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
elif [[ ! -x "$venv_python" ]]; then
  printf '%s\n' 'Existing .venv-linux is not a usable Linux virtual environment; preserve it and choose a clean project copy.' >&2
  exit 1
fi
"$venv_python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix:
    raise SystemExit('.venv-linux must be a Python 3.12 virtual environment.')
PY
"$venv_python" -m pip --version
"$venv_python" -m pip install 'torch==2.9.1' 'torchvision==0.24.1' \
  --index-url https://download.pytorch.org/whl/cu126
project_spec="$project_root[runtime]"
check_options=(--require-cuda)
if (( with_detection )); then
  project_spec="$project_root[runtime,detection]"
  check_options+=(--with-detection)
fi
# Explicit versions prevent optional dependencies from upgrading the selected pair.
"$venv_python" -m pip install -e "$project_spec" 'torch==2.9.1' 'torchvision==0.24.1'
"$venv_python" -m pip check
"$venv_python" "$project_root/scripts/check_runtime_environment.py" "${check_options[@]}"
printf '\n%s\n' 'Software checks passed on this machine. Camera, MCU, timing and vehicle behaviour remain unverified.'
printf '%s\n' 'Next follow docs/HEADLESS_UBUNTU.md for calibration and a manual dry-run.'
