#!/usr/bin/env bash
# Install the CUDA Toolkit inside WSL2 Ubuntu (issue #46).
#
# WSL2 gets the GPU driver from Windows: libcuda.so is mapped into
# /usr/lib/wsl/lib by the Windows NVIDIA driver. Never install a Linux
# display driver (cuda-drivers, nvidia-driver-*) here; it would shadow that
# stub. NVIDIA's wsl-ubuntu repository ships the toolkit without a driver.
#
# The toolkit's CUDA version must not exceed what the Windows driver
# supports ("CUDA UMD Version" in nvidia-smi). Override with CUDA_VERSION=13-2.
#
# Run with sudo:  sudo tools/setup-cuda-wsl.sh
set -euo pipefail
cuda_version="${CUDA_VERSION:-13-3}"
repo=https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64

if [[ $EUID -ne 0 ]]; then
  printf '%s\n' 'run as root: sudo tools/setup-cuda-wsl.sh' >&2
  exit 1
fi
if ! grep -qi microsoft /proc/version; then
  printf '%s\n' 'this script is for WSL2 only' >&2
  exit 1
fi
if [[ ! -e /usr/lib/wsl/lib/libcuda.so.1 ]]; then
  printf '%s\n' '/usr/lib/wsl/lib/libcuda.so.1 missing: install or update the NVIDIA driver on Windows first' >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/cuda-keyring.deb" "$repo/cuda-keyring_1.1-1_all.deb"
dpkg -i "$tmp/cuda-keyring.deb"
apt-get update
# cuda-toolkit-X-Y = compiler, runtime, libraries, Nsight Compute/Systems; no driver.
apt-get install -y "cuda-toolkit-$cuda_version"

dotted=${cuda_version/-/.}
printf '\nInstalled CUDA %s at /usr/local/cuda-%s.\n' "$dotted" "$dotted"
printf '%s\n' 'Add to ~/.bashrc:' \
  "  export PATH=/usr/local/cuda-$dotted/bin:\$PATH" \
  "  export LD_LIBRARY_PATH=/usr/local/cuda-$dotted/lib64:\${LD_LIBRARY_PATH:-}"
